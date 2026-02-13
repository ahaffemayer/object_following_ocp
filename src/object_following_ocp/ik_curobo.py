"""
Wrapper class for CuRobo IK Solver
Provides a simple interface for inverse kinematics given a robot name and target pose.
"""

# Standard Library
import time
from typing import Optional, Tuple

import numpy as np

# Third Party
import torch

# CuRobo
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

# Enable performance optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class RobotIKSolver:
    """
    Inverse Kinematics solver wrapper for robotic manipulators.

    This class provides a simple interface to solve IK problems given a robot
    configuration and target end-effector pose.

    Args:
        robot_name: Name of the robot (e.g., 'franka', 'ur5e')
        num_seeds: Number of random seeds for IK solver (default: 20)
        position_threshold: Position error threshold in meters (default: 0.005)
        rotation_threshold: Rotation error threshold in radians (default: 0.05)
        self_collision_check: Enable self-collision checking (default: False)
        use_cuda_graph: Enable CUDA graph optimization (default: True)
        device: Device to run on ('cuda' or 'cpu', default: 'cuda')
    """

    def __init__(
        self,
        robot_name: str,
        num_seeds: int = 20,
        position_threshold: float = 0.005,
        rotation_threshold: float = 0.05,
        self_collision_check: bool = False,
        use_cuda_graph: bool = False,  # Changed default to False to avoid CUDA graph issues
        device: str = "cuda",
    ):
        self.robot_name = robot_name
        self.tensor_args = TensorDeviceType(device=torch.device(device))

        # Load robot configuration
        config_file = load_yaml(
            join_path(get_robot_configs_path(), f"{robot_name}.yml")
        )
        urdf_file = config_file["robot_cfg"]["kinematics"]["urdf_path"]
        base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
        ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]

        # Create robot config
        robot_cfg = RobotConfig.from_basic(
            urdf_file, base_link, ee_link, self.tensor_args
        )

        # Create IK solver config
        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            None,
            rotation_threshold=rotation_threshold,
            position_threshold=position_threshold,
            num_seeds=num_seeds,
            self_collision_check=self_collision_check,
            self_collision_opt=self_collision_check,
            tensor_args=self.tensor_args,
            use_cuda_graph=use_cuda_graph,
        )

        # Initialize IK solver
        self.ik_solver = IKSolver(ik_config)

    def solve(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
        return_all_solutions: bool = False,
    ) -> Tuple[Optional[np.ndarray], dict]:
        """
        Solve inverse kinematics for a target end-effector pose.

        Args:
            position: Target position as [x, y, z] in meters
            quaternion: Target orientation as [w, x, y, z] quaternion
            return_all_solutions: If True, return all valid solutions (default: False)

        Returns:
            Tuple of (joint_configuration, info_dict) where:
                - joint_configuration: 1D numpy array of joint angles (shape: (n_joints,))
                  or None if solution failed
                - info_dict: Dictionary with solve information including:
                    - 'success': bool indicating if solution was found
                    - 'solve_time': time taken to solve in seconds
                    - 'position_error': position error in meters
                    - 'rotation_error': rotation error in radians
                    - 'all_solutions': 2D array of all valid solutions (shape: (n_solutions, n_joints))
                      only present if return_all_solutions=True
        """
        # Convert to numpy arrays if lists
        position = np.asarray(position, dtype=np.float32)
        quaternion = np.asarray(quaternion, dtype=np.float32)

        # Validate input shapes
        if position.shape[-1] != 3:
            raise ValueError(f"Position must have shape (..., 3), got {position.shape}")
        if quaternion.shape[-1] != 4:
            raise ValueError(
                f"Quaternion must have shape (..., 4), got {quaternion.shape}"
            )

        # Convert inputs to tensors
        pos_tensor = torch.tensor(
            position, device=self.tensor_args.device, dtype=torch.float32
        )
        quat_tensor = torch.tensor(
            quaternion, device=self.tensor_args.device, dtype=torch.float32
        )

        # Add batch dimension if needed
        if pos_tensor.ndim == 1:
            pos_tensor = pos_tensor.unsqueeze(0)
        if quat_tensor.ndim == 1:
            quat_tensor = quat_tensor.unsqueeze(0)

        # Create goal pose
        goal = Pose(position=pos_tensor, quaternion=quat_tensor)

        # Solve IK
        st_time = time.time()
        result = self.ik_solver.solve_batch(goal)
        torch.cuda.synchronize()
        solve_time = time.time() - st_time

        # Extract results
        info = {
            "success": result.success[0].item(),
            "solve_time": solve_time,
            "position_error": result.position_error[0].item(),
            "rotation_error": result.rotation_error[0].item(),
        }

        if result.success[0].item():
            # Get solution and ensure it's 1D for single pose
            solution = result.solution[0].cpu().numpy()
            # Flatten to 1D if needed
            if solution.ndim > 1:
                solution = solution.flatten()

            if return_all_solutions:
                # Return all valid solutions (keep 2D for multiple solutions)
                all_solutions = result.solution[result.success].cpu().numpy()
                info["all_solutions"] = all_solutions

            return solution, info
        else:
            return None, info

    def solve_batch(
        self, positions: np.ndarray, quaternions: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        """
        Solve inverse kinematics for multiple target poses in batch.

        Args:
            positions: Target positions as Nx3 array
            quaternions: Target orientations as Nx4 array (w, x, y, z)

        Returns:
            Tuple of (joint_configurations, info_dict) where:
                - joint_configurations: NxDOF array of joint angles
                - info_dict: Dictionary with batch solve information
        """
        # Convert to numpy arrays if needed
        positions = np.asarray(positions, dtype=np.float32)
        quaternions = np.asarray(quaternions, dtype=np.float32)

        # Validate input shapes
        if len(positions) == 0:
            raise ValueError("Cannot solve for empty batch of positions")
        if positions.shape[-1] != 3:
            raise ValueError(f"Positions must have shape (N, 3), got {positions.shape}")
        if quaternions.shape[-1] != 4:
            raise ValueError(
                f"Quaternions must have shape (N, 4), got {quaternions.shape}"
            )
        if len(positions) != len(quaternions):
            raise ValueError(
                f"Number of positions ({len(positions)}) must match number of quaternions ({len(quaternions)})"
            )

        # Convert inputs to tensors
        pos_tensor = torch.tensor(
            positions, device=self.tensor_args.device, dtype=torch.float32
        )
        quat_tensor = torch.tensor(
            quaternions, device=self.tensor_args.device, dtype=torch.float32
        )

        # Create goal pose
        goal = Pose(position=pos_tensor, quaternion=quat_tensor)

        # Solve IK
        st_time = time.time()
        result = self.ik_solver.solve_batch(goal)
        torch.cuda.synchronize()
        solve_time = time.time() - st_time

        # Extract results
        solutions = result.solution.cpu().numpy()
        success = result.success.cpu().numpy()

        info = {
            "success": success,
            "success_rate": success.sum() / len(success),
            "solve_time": solve_time,
            "solve_time_per_pose": solve_time / len(positions),
            "position_errors": result.position_error.cpu().numpy(),
            "rotation_errors": result.rotation_error.cpu().numpy(),
            "mean_position_error": result.position_error.mean().item(),
            "mean_rotation_error": result.rotation_error.mean().item(),
        }

        return solutions, info

    def forward_kinematics(
        self, joint_config: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward kinematics for given joint configuration.

        Args:
            joint_config: Joint angles array

        Returns:
            Tuple of (position, quaternion) of end-effector
        """
        q_tensor = torch.tensor(
            joint_config, device=self.tensor_args.device, dtype=torch.float32
        )

        if q_tensor.ndim == 1:
            q_tensor = q_tensor.unsqueeze(0)

        kin_state = self.ik_solver.fk(q_tensor)

        position = kin_state.ee_position[0].cpu().numpy()
        quaternion = kin_state.ee_quaternion[0].cpu().numpy()

        return position, quaternion


def demo():
    """Demonstration of the RobotIKSolver class."""
    # Initialize solver for Franka robot
    solver = RobotIKSolver(
        robot_name="franka",
        num_seeds=20,
        position_threshold=0.005,
        rotation_threshold=0.05,
        use_cuda_graph=False,  # Disabled by default to avoid CUDA version issues
    )

    # Sample a random configuration and get its FK
    q_sample = solver.ik_solver.sample_configs(1)
    position, quaternion = solver.forward_kinematics(q_sample[0].cpu().numpy())

    print(f"Target position: {position}")
    print(f"Target quaternion: {quaternion}")

    # Solve IK for this pose
    solution, info = solver.solve(position, quaternion)

    if solution is not None:
        print("\nIK Solution found!")
        print(f"Joint configuration: {solution}")
        print(f"Position error: {info['position_error']:.6f} m")
        print(f"Rotation error: {info['rotation_error']:.6f} rad")
        print(f"Solve time: {info['solve_time']:.6f} s")

        # Verify solution with FK
        fk_pos, fk_quat = solver.forward_kinematics(solution)
        print("\nVerification (FK of solution):")
        print(f"Position: {fk_pos}")
        print(f"Quaternion: {fk_quat}")
    else:
        print("IK solution not found!")
        print(f"Position error: {info['position_error']:.6f} m")
        print(f"Rotation error: {info['rotation_error']:.6f} rad")


if __name__ == "__main__":
    demo()

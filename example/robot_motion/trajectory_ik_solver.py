"""
Wrapper for solving IK for entire trajectories.
"""

import numpy as np
import pinocchio as pin
from typing import List, Optional, Tuple

from object_following_ocp.ik_curobo import RobotIKSolver
from object_following_ocp.trajectories import TrajectorySE3, TrajectoryInConfigurationSpace


class TrajectoryIKSolver:
    """
    Solves inverse kinematics for an entire SE3 trajectory.
    
    This wrapper handles:
    - Batch IK solving for trajectory points
    - Success rate tracking
    - Filtering out failed IK solutions
    - Progress reporting
    """
    
    def __init__(
        self,
        robot_name: str = "franka",
        num_seeds: int = 20,
        position_threshold: float = 0.005,
        rotation_threshold: float = 0.05,
        use_cuda_graph: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize trajectory IK solver.
        
        Args:
            robot_name: Name of the robot (e.g., 'franka')
            num_seeds: Number of random seeds for IK solver
            position_threshold: Position error threshold in meters
            rotation_threshold: Rotation error threshold in radians
            use_cuda_graph: Enable CUDA graph optimization
            verbose: Print progress information
        """
        self.solver = RobotIKSolver(
            robot_name=robot_name,
            num_seeds=num_seeds,
            position_threshold=position_threshold,
            rotation_threshold=rotation_threshold,
            use_cuda_graph=use_cuda_graph,
        )
        self.verbose = verbose
        
    def solve_trajectory(
        self,
        trajectory: TrajectorySE3,
        print_every: int = 10,
    ) -> Tuple[TrajectoryInConfigurationSpace, dict]:
        """
        Solve IK for entire trajectory.
        
        Args:
            trajectory: SE3 trajectory to solve IK for
            print_every: Print progress every N points (0 to disable)
            
        Returns:
            Tuple of (joint_trajectory, info_dict) where:
                - joint_trajectory: Valid joint configurations (failed IK excluded)
                - info_dict: Dictionary with solve information:
                    - 'success_count': Number of successful IK solutions
                    - 'success_rate': Percentage of successful solutions
                    - 'total_points': Total number of trajectory points
                    - 'failed_indices': List of indices where IK failed
                    - 'all_solutions': List of all solutions (None for failed)
        """
        if self.verbose:
            print("=" * 60)
            print("Solving IK for trajectory...")
            print("=" * 60)
        
        joint_configurations = []
        failed_indices = []
        success_count = 0
        
        for k, wM_ee in enumerate(trajectory.poses):
            solution, info = self.solver.solve(wM_ee)
            
            if solution is not None:
                joint_configurations.append(solution)
                success_count += 1
                
                if self.verbose and print_every > 0 and (k % print_every == 0 or k == 0):
                    print(
                        f"Point {k}: SUCCESS - "
                        f"pos_err={info['position_error']:.6f}m, "
                        f"rot_err={info['rotation_error']:.6f}rad"
                    )
            else:
                failed_indices.append(k)
                if self.verbose:
                    print(
                        f"Point {k}: FAILED - "
                        f"pos_err={info['position_error']:.6f}m, "
                        f"rot_err={info['rotation_error']:.6f}rad"
                    )
        
        success_rate = 100 * success_count / len(trajectory)
        
        if self.verbose:
            print(
                f"\nIK Success rate: {success_count}/{len(trajectory)} "
                f"({success_rate:.1f}%)"
            )
            print(f"Valid joint configurations: {len(joint_configurations)}")
        
        # Create trajectory from valid configurations
        joint_trajectory = TrajectoryInConfigurationSpace(joint_configurations)
        
        info_dict = {
            'success_count': success_count,
            'success_rate': success_rate,
            'total_points': len(trajectory),
            'failed_indices': failed_indices,
        }
        
        return joint_trajectory, info_dict
    
    def verify_first_solution(
        self,
        solution: np.ndarray,
        target_pose: pin.SE3,
        rmodel: pin.Model,
        rdata: pin.Data,
        tcp_frame_name: str = "panda_hand_tcp",
    ) -> dict:
        """
        Verify IK solution by comparing FK results.
        
        Args:
            solution: Joint configuration to verify
            target_pose: Target SE3 pose
            rmodel: Pinocchio robot model
            rdata: Pinocchio robot data
            tcp_frame_name: Name of TCP frame in robot model
            
        Returns:
            Dictionary with verification results
        """
        print("=" * 60)
        print("Detailed verification for IK solution:")
        print("=" * 60)
        
        # CuRobo FK
        fk_pos_curobo, fk_quat_curobo = self.solver.forward_kinematics(solution)
        print("\nCuRobo FK:")
        print(f"Position: {fk_pos_curobo}")
        print(f"Quaternion (wxyz): {fk_quat_curobo}")
        
        # Convert quaternion to rotation matrix
        R_curobo = pin.Quaternion(
            float(fk_quat_curobo[0]),  # w
            float(fk_quat_curobo[1]),  # x
            float(fk_quat_curobo[2]),  # y
            float(fk_quat_curobo[3]),  # z
        ).toRotationMatrix()
        
        # Pinocchio FK
        pin.framesForwardKinematics(rmodel, rdata, solution)
        tcp_frame_id = rmodel.getFrameId(tcp_frame_name)
        wM_tcp_pinocchio = rdata.oMf[tcp_frame_id]
        
        print(f"\nPinocchio FK ({tcp_frame_name} frame):")
        print(f"Position: {wM_tcp_pinocchio.translation}")
        quat_pinocchio = pin.Quaternion(wM_tcp_pinocchio.rotation)
        print(
            f"Quaternion (wxyz): [{quat_pinocchio.w}, {quat_pinocchio.x}, "
            f"{quat_pinocchio.y}, {quat_pinocchio.z}]"
        )
        
        # Comparison
        pos_diff = fk_pos_curobo - wM_tcp_pinocchio.translation
        print(f"\nPosition difference (CuRobo - Pinocchio): {pos_diff}")
        print(f"Position error norm: {np.linalg.norm(pos_diff):.6f} m")
        
        R_diff = R_curobo.T @ wM_tcp_pinocchio.rotation
        angle_diff = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
        print(f"Rotation angle difference: {np.rad2deg(angle_diff):.6f} degrees")
        
        return {
            'fk_pos_curobo': fk_pos_curobo,
            'fk_quat_curobo': fk_quat_curobo,
            'pinocchio_pose': wM_tcp_pinocchio,
            'position_error': np.linalg.norm(pos_diff),
            'rotation_error_deg': np.rad2deg(angle_diff),
        }

"""
Grid search over camera positions to find valid IK trajectories.
"""

import pathlib
import pickle
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from ik_trajectory_converter import IKTrajectoryConverter
from trajectory_ik_solver import TrajectoryIKSolver

from object_following_ocp.data_loader import DataLoader, RobotConfig
from object_following_ocp.trajectories import (
    TrajectoryInConfigurationSpace,
    TrajectorySE3,
)


@dataclass
class TrajectoryCandidate:
    """Stores a valid IK trajectory with all relevant information."""

    camera_translation: np.ndarray
    object_trajectory: TrajectorySE3
    ee_trajectory: TrajectorySE3
    joint_trajectory: TrajectoryInConfigurationSpace
    ik_success_rate: float
    ik_info: dict

    def to_dict(self) -> dict:
        """Convert to dictionary for saving."""
        return {
            "camera_translation": self.camera_translation.tolist(),
            "object_trajectory_poses": [
                pose.homogeneous.tolist() for pose in self.object_trajectory.poses
            ],
            "ee_trajectory_poses": [
                pose.homogeneous.tolist() for pose in self.ee_trajectory.poses
            ],
            "joint_configurations": [
                q.tolist() for q in self.joint_trajectory.configurations
            ],
            "ik_success_rate": self.ik_success_rate,
            "ik_info": self.ik_info,
        }


class CameraPositionGridSearch:
    """
    Grid search over camera positions to find valid trajectories.

    For each camera position:
    1. Compute EE trajectory
    2. Try IK on first pose
    3. If successful, solve IK for entire trajectory
    4. Store valid trajectories with metadata
    """

    def __init__(
        self,
        dataloader: DataLoader,
        robot_config: RobotConfig,
        object_traj_path: pathlib.Path,
        scale_path: pathlib.Path,
        config_path: pathlib.Path,
        grasp_correction_angle_deg: float = 90.0,
        elevation_angle_deg: float = 25.0,
        verbose: bool = True,
    ):
        """
        Initialize grid search.

        Args:
            dataloader: DataLoader with object trajectory and grasp
            robot_config: Robot configuration
            object_traj_path: Path to object trajectory file
            scale_path: Path to scales file
            config_path: Path to robot config file
            grasp_correction_angle_deg: Grasp correction angle
            elevation_angle_deg: Elevation angle for visualization
            verbose: Print progress information
        """
        self.dataloader = dataloader
        self.robot_config = robot_config
        self.object_traj_path = object_traj_path
        self.scale_path = scale_path
        self.config_path = config_path
        self.grasp_correction_angle_deg = grasp_correction_angle_deg
        self.elevation_angle_deg = elevation_angle_deg
        self.verbose = verbose

        # Initialize IK solver (reused for all positions)
        self.ik_solver = TrajectoryIKSolver(
            robot_name="franka",
            num_seeds=20,
            position_threshold=0.005,
            rotation_threshold=0.05,
            use_cuda_graph=False,
            verbose=False,  # Suppress per-trajectory output during grid search
        )

        # Storage for valid trajectories
        self.valid_trajectories: List[TrajectoryCandidate] = []

    def create_grid(
        self,
        x_range: Tuple[float, float, int],
        y_range: Tuple[float, float, int],
        z_range: Tuple[float, float, int],
    ) -> np.ndarray:
        """
        Create 3D grid of camera positions.

        Args:
            x_range: (min, max, num_points) for X axis
            y_range: (min, max, num_points) for Y axis
            z_range: (min, max, num_points) for Z axis

        Returns:
            Array of shape (total_points, 3) with all camera positions
        """
        x_values = np.linspace(x_range[0], x_range[1], x_range[2])
        y_values = np.linspace(y_range[0], y_range[1], y_range[2])
        z_values = np.linspace(z_range[0], z_range[1], z_range[2])

        # Create meshgrid
        X, Y, Z = np.meshgrid(x_values, y_values, z_values, indexing="ij")

        # Flatten to list of positions
        positions = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        if self.verbose:
            print(f"Created grid with {len(positions)} camera positions")
            print(f"  X: [{x_range[0]:.2f}, {x_range[1]:.2f}] with {x_range[2]} points")
            print(f"  Y: [{y_range[0]:.2f}, {y_range[1]:.2f}] with {y_range[2]} points")
            print(f"  Z: [{z_range[0]:.2f}, {z_range[1]:.2f}] with {z_range[2]} points")

        return positions

    def test_camera_position(
        self,
        camera_translation: np.ndarray,
    ) -> Optional[TrajectoryCandidate]:
        """
        Test a single camera position.

        Args:
            camera_translation: Camera position [x, y, z]

        Returns:
            TrajectoryCandidate if IK succeeds, None otherwise
        """
        # Create converter for this camera position
        ik_converter = IKTrajectoryConverter(
            robot_config=self.robot_config,
            camera_translation=camera_translation,
            grasp_correction_angle_deg=self.grasp_correction_angle_deg,
            elevation_angle_deg=self.elevation_angle_deg,
        )

        # Compute trajectories
        object_traj_world, ee_traj_world = ik_converter.compute_trajectories(
            self.dataloader
        )

        # Test IK on FIRST pose only
        first_pose = ee_traj_world[0]
        solution, info = self.ik_solver.solver.solve(first_pose)

        if solution is None:
            # First pose failed, skip this camera position
            return None

        # First pose succeeded, solve for entire trajectory
        joint_trajectory, ik_info = self.ik_solver.solve_trajectory(
            trajectory=ee_traj_world,
            print_every=0,  # Don't print during grid search
        )

        # Create candidate
        candidate = TrajectoryCandidate(
            camera_translation=camera_translation,
            object_trajectory=object_traj_world,
            ee_trajectory=ee_traj_world,
            joint_trajectory=joint_trajectory,
            ik_success_rate=ik_info["success_rate"],
            ik_info=ik_info,
        )

        return candidate

    def run_grid_search(
        self,
        x_range: Tuple[float, float, int],
        y_range: Tuple[float, float, int],
        z_range: Tuple[float, float, int],
    ) -> List[TrajectoryCandidate]:
        """
        Run grid search over camera positions.

        Args:
            x_range: (min, max, num_points) for X axis
            y_range: (min, max, num_points) for Y axis
            z_range: (min, max, num_points) for Z axis

        Returns:
            List of valid trajectory candidates
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("CAMERA POSITION GRID SEARCH")
            print("=" * 60)

        # Create grid
        camera_positions = self.create_grid(x_range, y_range, z_range)

        # Test each position
        self.valid_trajectories = []

        for idx, camera_pos in enumerate(camera_positions):
            if self.verbose:
                print(
                    f"\nTesting position {idx + 1}/{len(camera_positions)}: {camera_pos}"
                )

            candidate = self.test_camera_position(camera_pos)

            if candidate is not None:
                self.valid_trajectories.append(candidate)
                if self.verbose:
                    print(
                        f"  ✓ SUCCESS - IK success rate: {candidate.ik_success_rate:.1f}%"
                    )
            else:
                if self.verbose:
                    print("  ✗ FAILED - First pose IK failed")

        if self.verbose:
            print("\n" + "=" * 60)
            print("Grid search complete!")
            print(f"Found {len(self.valid_trajectories)} valid trajectories")
            print(
                f"Success rate: {100 * len(self.valid_trajectories) / len(camera_positions):.1f}%"
            )
            print("=" * 60)

        return self.valid_trajectories

    def get_summary(self) -> dict:
        """Get summary statistics of grid search."""
        if not self.valid_trajectories:
            return {
                "num_valid": 0,
                "best_success_rate": 0.0,
                "worst_success_rate": 0.0,
                "mean_success_rate": 0.0,
            }

        success_rates = [t.ik_success_rate for t in self.valid_trajectories]

        return {
            "num_valid": len(self.valid_trajectories),
            "best_success_rate": max(success_rates),
            "worst_success_rate": min(success_rates),
            "mean_success_rate": np.mean(success_rates),
            "camera_positions": [
                t.camera_translation.tolist() for t in self.valid_trajectories
            ],
        }

    def save_results(self, filepath: pathlib.Path):
        """
        Save grid search results to file.

        Args:
            filepath: Path to save results (will use pickle)
        """
        results = {
            "mesh_id": self.dataloader.object_info.mesh_id,
            "object_traj_path": str(self.object_traj_path),
            "scale_path": str(self.scale_path),
            "config_path": str(self.config_path),
            "mesh_path": str(self.dataloader.object_info.mesh_path),
            "texture_path": str(self.dataloader.object_info.texture_path)
            if self.dataloader.object_info.texture_path
            else None,
            "object_scale": self.dataloader.object_info.scale,
            "best_grasp_name": self.dataloader.best_grasp.name
            if self.dataloader.has_grasps
            else None,
            "grasp_correction_angle_deg": self.grasp_correction_angle_deg,
            "elevation_angle_deg": self.elevation_angle_deg,
            "summary": self.get_summary(),
            "trajectories": [t.to_dict() for t in self.valid_trajectories],
        }

        with open(filepath, "wb") as f:
            pickle.dump(results, f)

        if self.verbose:
            print(f"\nSaved {len(self.valid_trajectories)} trajectories to {filepath}")

    @staticmethod
    def load_results(filepath: pathlib.Path) -> dict:
        """Load grid search results from file."""
        with open(filepath, "rb") as f:
            return pickle.load(f)

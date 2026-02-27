"""
Grid search over camera positions to find valid IK trajectories.

Instead of hardcoded absolute camera bounds, the search is centered
on a reference camera translation that places the trajectory's mean
at a desired world-frame target position.
"""

import pathlib
import pickle
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from object_following_ocp.geom.ik_trajectory_converter import IKTrajectoryConverter
from object_following_ocp.solver.trajectory_ik_solver import TrajectoryIKSolver

from object_following_ocp.data.data_loader import DataLoader, RobotConfig
from object_following_ocp.geom.grasp_transforms import (
    GraspTransformChain,
    GraspTransformConfig,
)
from object_following_ocp.geom.trajectories import (
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

    The search is centered on a reference camera translation computed
    automatically so that the trajectory's average lands at a desired
    world-frame target. The grid is then built as ± half-extents around
    that reference.

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
            verbose=False,
        )

        # Storage for valid trajectories
        self.valid_trajectories: List[TrajectoryCandidate] = []

        # Cache the camera-frame trajectory (shared across all grid positions)
        self._object_traj_camera = self.dataloader.to_trajectory_SE3()

    # ------------------------------------------------------------------
    # Reference camera computation
    # ------------------------------------------------------------------

    def compute_reference_camera(
        self,
        target_position: np.ndarray = np.array([0.0, 0.0, 0.0]),
    ) -> np.ndarray:
        """
        Find the camera translation that centres the trajectory's mean
        on ``target_position`` in the world frame.

        The world-frame mean of the object trajectory is:

            avg(cam) = R_align @ cam + avg_at_origin

        where ``avg_at_origin`` is the mean when camera = [0, 0, 0] and
        ``R_align`` is the rotation component of ``worldM_world_aligned``.

        Solving for ``cam``:

            cam = R_align^T @ (target_position - avg_at_origin)

        Args:
            target_position: Desired mean position in world frame.
                             E.g. np.array([0, 0, 0.7]) to place the
                             trajectory at robot-table height.

        Returns:
            Reference camera translation (3,).
        """
        # Build a chain with the camera at the origin
        zero_config = GraspTransformConfig.from_robot_config(
            robot_config=self.robot_config,
            camera_translation=np.zeros(3),
            grasp_correction_angle_deg=self.grasp_correction_angle_deg,
            elevation_angle_deg=self.elevation_angle_deg,
        )
        chain = GraspTransformChain(zero_config)

        traj_world_at_origin = chain.transform_object_trajectory(
            self._object_traj_camera
        )
        avg_at_origin = np.mean(
            [pose.translation for pose in traj_world_at_origin.poses], axis=0
        )

        R_align = chain.worldM_world_aligned.rotation
        ref_cam = R_align.T @ (target_position - avg_at_origin)

        if self.verbose:
            print(f"Reference camera translation: {np.round(ref_cam, 4)}")
            print(f"  (places trajectory mean at {target_position} in world frame)")

        return ref_cam

    # ------------------------------------------------------------------
    # Grid creation
    # ------------------------------------------------------------------

    def create_grid(
        self,
        center: np.ndarray,
        half_extents: Tuple[float, float, float],
        steps: Tuple[int, int, int],
    ) -> np.ndarray:
        """
        Create a 3D grid of camera positions centred on ``center``
        with ± ``half_extents`` in each axis.

        Args:
            center: Reference camera position (3,).
            half_extents: (dx, dy, dz) half-widths of the search box.
            steps: (nx, ny, nz) number of samples per axis.

        Returns:
            Array of shape (total_points, 3) with all camera positions.
        """
        dx, dy, dz = half_extents
        nx, ny, nz = steps

        xs = np.linspace(center[0] - dx, center[0] + dx, nx)
        ys = np.linspace(center[1] - dy, center[1] + dy, ny)
        zs = np.linspace(center[2] - dz, center[2] + dz, nz)

        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        positions = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        if self.verbose:
            print(f"Created grid with {len(positions)} camera positions")
            print(f"  X: [{center[0] - dx:.3f}, {center[0] + dx:.3f}]  {nx} pts")
            print(f"  Y: [{center[1] - dy:.3f}, {center[1] + dy:.3f}]  {ny} pts")
            print(f"  Z: [{center[2] - dz:.3f}, {center[2] + dz:.3f}]  {nz} pts")
            print(f"  Center: {np.round(center, 4)}")

        return positions

    # ------------------------------------------------------------------
    # Legacy grid creation (absolute bounds) — kept for compatibility
    # ------------------------------------------------------------------

    def create_grid_absolute(
        self,
        x_range: Tuple[float, float, int],
        y_range: Tuple[float, float, int],
        z_range: Tuple[float, float, int],
    ) -> np.ndarray:
        """
        Create 3D grid of camera positions with absolute bounds.

        This is the original interface, kept for backward compatibility.

        Args:
            x_range: (min, max, num_points) for X axis
            y_range: (min, max, num_points) for Y axis
            z_range: (min, max, num_points) for Z axis

        Returns:
            Array of shape (total_points, 3) with all camera positions.
        """
        x_values = np.linspace(x_range[0], x_range[1], x_range[2])
        y_values = np.linspace(y_range[0], y_range[1], y_range[2])
        z_values = np.linspace(z_range[0], z_range[1], z_range[2])

        X, Y, Z = np.meshgrid(x_values, y_values, z_values, indexing="ij")
        positions = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        if self.verbose:
            print(f"Created grid with {len(positions)} camera positions")
            print(f"  X: [{x_range[0]:.2f}, {x_range[1]:.2f}] with {x_range[2]} points")
            print(f"  Y: [{y_range[0]:.2f}, {y_range[1]:.2f}] with {y_range[2]} points")
            print(f"  Z: [{z_range[0]:.2f}, {z_range[1]:.2f}] with {z_range[2]} points")

        return positions

    # ------------------------------------------------------------------
    # Single camera position test
    # ------------------------------------------------------------------

    def test_camera_position(
        self,
        camera_translation: np.ndarray,
    ) -> Optional[TrajectoryCandidate]:
        """
        Test a single camera position.

        Args:
            camera_translation: Camera position [x, y, z]

        Returns:
            TrajectoryCandidate if IK succeeds, None otherwise.
        """
        ik_converter = IKTrajectoryConverter(
            robot_config=self.robot_config,
            camera_translation=camera_translation,
            grasp_correction_angle_deg=self.grasp_correction_angle_deg,
            elevation_angle_deg=self.elevation_angle_deg,
        )

        object_traj_world, ee_traj_world = ik_converter.compute_trajectories(
            self.dataloader
        )

        # Test IK on FIRST pose only
        first_pose = ee_traj_world[0]
        solution, info = self.ik_solver.solver.solve(first_pose)

        if solution is None:
            return None

        # First pose succeeded — solve for entire trajectory
        joint_trajectory, ik_info = self.ik_solver.solve_trajectory(
            trajectory=ee_traj_world,
            print_every=0,
        )

        candidate = TrajectoryCandidate(
            camera_translation=camera_translation,
            object_trajectory=object_traj_world,
            ee_trajectory=ee_traj_world,
            joint_trajectory=joint_trajectory,
            ik_success_rate=ik_info["success_rate"],
            ik_info=ik_info,
        )

        return candidate

    # ------------------------------------------------------------------
    # Full grid search (new interface: center + half-extents)
    # ------------------------------------------------------------------

    def run_grid_search(
        self,
        target_position: np.ndarray = np.array([0.0, 0.0, 0.0]),
        half_extents: Tuple[float, float, float] = (0.2, 0.2, 0.2),
        steps: Tuple[int, int, int] = (3, 3, 3),
    ) -> List[TrajectoryCandidate]:
        """
        Run grid search centred on the automatically computed reference
        camera position.

        Args:
            target_position: Desired mean position of the trajectory in
                             world frame (e.g. [0, 0, 0.7] for table
                             height).
            half_extents: (dx, dy, dz) search half-widths in camera
                          frame axes.
            steps: (nx, ny, nz) grid samples per axis.

        Returns:
            List of valid trajectory candidates.
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("CAMERA POSITION GRID SEARCH (auto-centred)")
            print("=" * 60)

        # Compute reference camera
        ref_cam = self.compute_reference_camera(target_position)

        # Build grid around it
        camera_positions = self.create_grid(ref_cam, half_extents, steps)

        # Test each position
        self.valid_trajectories = []

        for idx, camera_pos in enumerate(camera_positions):
            if self.verbose:
                print(
                    f"\nTesting position {idx + 1}/{len(camera_positions)}: "
                    f"{np.round(camera_pos, 4)}"
                )

            candidate = self.test_camera_position(camera_pos)

            if candidate is not None:
                self.valid_trajectories.append(candidate)
                if self.verbose:
                    print(
                        f"  ✓ SUCCESS - IK success rate: "
                        f"{candidate.ik_success_rate:.1f}%"
                    )
            else:
                if self.verbose:
                    print("  ✗ FAILED - First pose IK failed")

        if self.verbose:
            print("\n" + "=" * 60)
            print("Grid search complete!")
            print(f"Found {len(self.valid_trajectories)} valid trajectories")
            print(
                f"Success rate: "
                f"{100 * len(self.valid_trajectories) / len(camera_positions):.1f}%"
            )
            print("=" * 60)

        return self.valid_trajectories

    # ------------------------------------------------------------------
    # Legacy run_grid_search (absolute bounds) — kept for compatibility
    # ------------------------------------------------------------------

    def run_grid_search_absolute(
        self,
        x_range: Tuple[float, float, int],
        y_range: Tuple[float, float, int],
        z_range: Tuple[float, float, int],
    ) -> List[TrajectoryCandidate]:
        """
        Run grid search with absolute camera position bounds.

        Kept for backward compatibility with existing scripts.
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("CAMERA POSITION GRID SEARCH (absolute bounds)")
            print("=" * 60)

        camera_positions = self.create_grid_absolute(x_range, y_range, z_range)

        self.valid_trajectories = []

        for idx, camera_pos in enumerate(camera_positions):
            if self.verbose:
                print(
                    f"\nTesting position {idx + 1}/{len(camera_positions)}: "
                    f"{camera_pos}"
                )

            candidate = self.test_camera_position(camera_pos)

            if candidate is not None:
                self.valid_trajectories.append(candidate)
                if self.verbose:
                    print(
                        f"  ✓ SUCCESS - IK success rate: "
                        f"{candidate.ik_success_rate:.1f}%"
                    )
            else:
                if self.verbose:
                    print("  ✗ FAILED - First pose IK failed")

        if self.verbose:
            print("\n" + "=" * 60)
            print("Grid search complete!")
            print(f"Found {len(self.valid_trajectories)} valid trajectories")
            print(
                f"Success rate: "
                f"{100 * len(self.valid_trajectories) / len(camera_positions):.1f}%"
            )
            print("=" * 60)

        return self.valid_trajectories

    # ------------------------------------------------------------------
    # Summary / Save / Load
    # ------------------------------------------------------------------

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
        """Save grid search results to file."""
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

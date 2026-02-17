"""
Interactive trajectory reviewer with OCP solving.
"""

import pathlib
import pickle
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pinocchio as pin
from camera_grid_search import TrajectoryCandidate
from ocp_trajectory_converter import OCPTrajectoryConverter
from trajectory_visualizer import TrajectoryVisualizer

from object_following_ocp.data_loader import DataLoader, RobotConfig
from object_following_ocp.ocp import OCP


@dataclass
class ReviewedTrajectory:
    """Stores a reviewed and accepted trajectory with OCP solution."""

    camera_translation: np.ndarray
    object_trajectory_poses: List[np.ndarray]  # List of 4x4 homogeneous matrices
    joint_trajectory_ik: List[np.ndarray]  # IK solution
    joint_trajectory_ocp: List[np.ndarray]  # OCP solution states [q, v]
    tcp_trajectory_poses: List[np.ndarray]  # Target TCP poses
    ik_success_rate: float
    user_notes: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for saving."""
        return {
            "camera_translation": self.camera_translation.tolist(),
            "object_trajectory_poses": [
                pose.tolist() for pose in self.object_trajectory_poses
            ],
            "joint_trajectory_ik": [q.tolist() for q in self.joint_trajectory_ik],
            "joint_trajectory_ocp": [xs.tolist() for xs in self.joint_trajectory_ocp],
            "tcp_trajectory_poses": [
                pose.tolist() for pose in self.tcp_trajectory_poses
            ],
            "ik_success_rate": self.ik_success_rate,
            "user_notes": self.user_notes,
        }


class TrajectoryReviewer:
    """
    Interactive reviewer for trajectory candidates.

    For each trajectory:
    1. Solve OCP using IK as warm start
    2. Visualize OCP solution
    3. Ask user to keep or ditch
    4. Save kept trajectories with all metadata
    """

    def __init__(
        self,
        dataloader: DataLoader,
        robot_config: RobotConfig,
        visualizer: TrajectoryVisualizer,
        rmodel: pin.Model,
        cmodel: pin.GeometryModel,
        object_traj_path: pathlib.Path,
        scale_path: pathlib.Path,
        config_path: pathlib.Path,
        grasp_correction_angle_deg: float = 90.0,
        elevation_angle_deg: float = 25.0,
    ):
        """
        Initialize trajectory reviewer.

        Args:
            dataloader: DataLoader with object info
            robot_config: Robot configuration
            visualizer: TrajectoryVisualizer for display
            rmodel: Pinocchio robot model
            cmodel: Pinocchio collision model
            object_traj_path: Path to object trajectory file
            scale_path: Path to scales file
            config_path: Path to robot config file
            grasp_correction_angle_deg: Grasp correction angle
            elevation_angle_deg: Elevation angle
        """
        self.dataloader = dataloader
        self.robot_config = robot_config
        self.visualizer = visualizer
        self.rmodel = rmodel
        self.cmodel = cmodel
        self.object_traj_path = object_traj_path
        self.scale_path = scale_path
        self.config_path = config_path
        self.grasp_correction_angle_deg = grasp_correction_angle_deg
        self.elevation_angle_deg = elevation_angle_deg

        # Storage for kept trajectories
        self.kept_trajectories: List[ReviewedTrajectory] = []

    def solve_ocp_for_candidate(
        self,
        candidate: TrajectoryCandidate,
    ) -> tuple:
        """
        Solve OCP for a trajectory candidate.

        Args:
            candidate: TrajectoryCandidate with IK solution

        Returns:
            Tuple of (ocp, tcp_trajectory)
        """
        # Create OCP trajectory converter for this camera position
        ocp_converter = OCPTrajectoryConverter(
            robot_config=self.robot_config,
            camera_translation=candidate.camera_translation,
            grasp_correction_angle_deg=self.grasp_correction_angle_deg,
            elevation_angle_deg=self.elevation_angle_deg,
        )

        # Compute TCP trajectory
        tcp_traj_world = ocp_converter.compute_tcp_trajectory(self.dataloader)

        # Setup OCP
        T_ocp = len(tcp_traj_world)
        q0 = candidate.joint_trajectory[0]
        x0 = np.concatenate((q0, np.zeros(self.rmodel.nv)))

        weights = {
            "W_xREG": self.robot_config.W_xREG,
            "W_uREG": self.robot_config.W_uREG,
            "W_gripper_pose": self.robot_config.W_gripper_pose,
            "W_gripper_pose_term": self.robot_config.W_gripper_pose_term,
            "W_limit": self.robot_config.W_limit,
        }

        # Create OCP
        OCP_creator = OCP(
            self.rmodel,
            self.cmodel,
            tcp_traj_world,
            x0=x0,
            joint_limits=True,
            joint_limits_constraint=False,
            with_callbacks=False,
            weights=weights,
            safety_threshold=self.robot_config.safety_threshold,
            T=T_ocp,
            dt=self.robot_config.dt,
        )

        ocp = OCP_creator.create_OCP()

        # Create warm start from IK
        X_init = [
            np.concatenate((q, np.zeros(self.rmodel.nv)))
            for q in candidate.joint_trajectory
        ]

        # Ensure correct length
        if len(X_init) > T_ocp:
            X_init = X_init[:T_ocp]
        elif len(X_init) < T_ocp:
            while len(X_init) < T_ocp:
                X_init.append(X_init[-1])

        U_init = ocp.problem.quasiStatic(X_init[:-1])

        # Solve OCP
        print("  Solving OCP...")
        ocp.solve(X_init, U_init)
        return ocp, tcp_traj_world

    def review_trajectory(
        self,
        candidate: TrajectoryCandidate,
        trajectory_idx: int,
        total_trajectories: int,
    ) -> Optional[ReviewedTrajectory]:
        """
        Review a single trajectory candidate.

        Args:
            candidate: TrajectoryCandidate to review
            trajectory_idx: Index of this trajectory
            total_trajectories: Total number of trajectories

        Returns:
            ReviewedTrajectory if kept, None if ditched
        """
        print("\n" + "=" * 70)
        print(f"REVIEWING TRAJECTORY {trajectory_idx + 1}/{total_trajectories}")
        print("=" * 70)
        print(f"Camera position: {candidate.camera_translation}")
        print(f"IK success rate: {candidate.ik_success_rate:.1f}%")

        # Solve OCP
        ocp, tcp_trajectory = self.solve_ocp_for_candidate(candidate)

        # Visualize OCP solution
        print("\nDisplaying OCP solution...")
        print("Press Enter to step through trajectory")
        print("=" * 70)

        self.visualizer.animate_ocp_solution(
            ocp_states=ocp.xs,
            object_trajectory=candidate.object_trajectory,
            rmodel=self.rmodel,
            interactive=True,
        )

        # Ask user decision
        print("\n" + "=" * 70)
        decision = input("Keep this trajectory? (y/n): ").strip().lower()

        if decision == "y":
            notes = input("Optional notes (press Enter to skip): ").strip()

            # Create reviewed trajectory
            reviewed = ReviewedTrajectory(
                camera_translation=candidate.camera_translation,
                object_trajectory_poses=[
                    pose.homogeneous for pose in candidate.object_trajectory.poses
                ],
                joint_trajectory_ik=[
                    q for q in candidate.joint_trajectory.configurations
                ],
                joint_trajectory_ocp=ocp.xs,
                tcp_trajectory_poses=[
                    pose.homogeneous for pose in tcp_trajectory.poses
                ],
                ik_success_rate=candidate.ik_success_rate,
                user_notes=notes,
            )

            print("✓ Trajectory KEPT")
            return reviewed
        else:
            print("✗ Trajectory DITCHED")
            return None

    def review_all_candidates(
        self,
        candidates: List[TrajectoryCandidate],
    ) -> List[ReviewedTrajectory]:
        """
        Review all trajectory candidates.

        Args:
            candidates: List of TrajectoryCandidate to review

        Returns:
            List of kept ReviewedTrajectory
        """
        print("\n" + "=" * 70)
        print(f"REVIEWING {len(candidates)} TRAJECTORY CANDIDATES")
        print("=" * 70)

        self.kept_trajectories = []

        for idx, candidate in enumerate(candidates):
            reviewed = self.review_trajectory(
                candidate=candidate,
                trajectory_idx=idx,
                total_trajectories=len(candidates),
            )

            if reviewed is not None:
                self.kept_trajectories.append(reviewed)

        print("\n" + "=" * 70)
        print("REVIEW COMPLETE")
        print(f"Kept {len(self.kept_trajectories)}/{len(candidates)} trajectories")
        print("=" * 70)

        return self.kept_trajectories

    def save_kept_trajectories(self, filepath: pathlib.Path):
        """
        Save kept trajectories to file.

        Args:
            filepath: Path to save results
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
            "num_kept": len(self.kept_trajectories),
            "trajectories": [t.to_dict() for t in self.kept_trajectories],
        }

        with open(filepath, "wb") as f:
            pickle.dump(results, f)

        print(f"\nSaved {len(self.kept_trajectories)} kept trajectories to {filepath}")

    @staticmethod
    def load_kept_trajectories(filepath: pathlib.Path) -> dict:
        """Load kept trajectories from file."""
        with open(filepath, "rb") as f:
            return pickle.load(f)

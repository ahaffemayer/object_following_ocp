"""
Trajectory reviewer with OCP solving — auto-keeps all valid trajectories.
"""

import pathlib
import pickle
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pinocchio as pin

from object_following_ocp.data.data_loader import DataLoader, RobotConfig
from object_following_ocp.geom.camera_grid_search import TrajectoryCandidate
from object_following_ocp.geom.ocp_trajectory_converter import OCPTrajectoryConverter
from object_following_ocp.solver.ocp import OCP
from object_following_ocp.visualizer.trajectory_visualizer import TrajectoryVisualizer


@dataclass
class ReviewedTrajectory:
    """Stores a reviewed and accepted trajectory with OCP solution."""

    camera_translation: np.ndarray
    object_trajectory_poses: List[np.ndarray]
    joint_trajectory_ik: List[np.ndarray]
    joint_trajectory_ocp: List[np.ndarray]
    tcp_trajectory_poses: List[np.ndarray]
    ik_success_rate: float
    tracking_cost: float = 0.0
    user_notes: str = ""

    def to_dict(self) -> dict:
        return {
            "camera_translation": self.camera_translation.tolist(),
            "object_trajectory_poses": [
                p.tolist() for p in self.object_trajectory_poses
            ],
            "joint_trajectory_ik": [q.tolist() for q in self.joint_trajectory_ik],
            "joint_trajectory_ocp": [xs.tolist() for xs in self.joint_trajectory_ocp],
            "tcp_trajectory_poses": [p.tolist() for p in self.tcp_trajectory_poses],
            "ik_success_rate": self.ik_success_rate,
            "tracking_cost": self.tracking_cost,
            "user_notes": self.user_notes,
        }


class TrajectoryReviewer:
    """
    Reviewer for trajectory candidates.

    For each trajectory:
    1. Solve OCP using IK as warm start
    2. Animate the solution (sleep-based, no interaction)
    3. Auto-keep all valid trajectories
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
        ee_frame_name: str = "panda_hand_tcp",
    ):
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
        self.ee_frame_name = ee_frame_name
        self.kept_trajectories: List[ReviewedTrajectory] = []

    def solve_ocp_for_candidate(self, candidate: TrajectoryCandidate) -> tuple:
        """Solve OCP for a trajectory candidate, return (ocp, tcp_trajectory)."""
        ocp_converter = OCPTrajectoryConverter(
            robot_config=self.robot_config,
            camera_translation=candidate.camera_translation,
            grasp_correction_angle_deg=self.grasp_correction_angle_deg,
            elevation_angle_deg=self.elevation_angle_deg,
        )
        tcp_traj_world = ocp_converter.compute_tcp_trajectory(self.dataloader)

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
            ee_frame_name=self.ee_frame_name,
        )
        ocp = OCP_creator.create_OCP()

        X_init = [
            np.concatenate((q, np.zeros(self.rmodel.nv)))
            for q in candidate.joint_trajectory
        ]
        if len(X_init) > T_ocp:
            X_init = X_init[:T_ocp]
        while len(X_init) < T_ocp:
            X_init.append(X_init[-1])

        U_init = ocp.problem.quasiStatic(X_init[:-1])

        print("  Solving OCP...")
        ocp.solve(X_init, U_init)
        return ocp, tcp_traj_world

    def _compute_tracking_cost(self, ocp, tcp_traj_world) -> float:
        """Sum of squared SE3 log norms between actual OCP TCP poses and target."""
        rdata = self.rmodel.createData()
        ee_frame_id = self.rmodel.getFrameId(self.ee_frame_name)
        cost = 0.0
        for xs, target_pose in zip(ocp.xs, tcp_traj_world.poses):
            q = xs[: self.rmodel.nq]
            pin.framesForwardKinematics(self.rmodel, rdata, q)
            xi = pin.log6(rdata.oMf[ee_frame_id].inverse() * target_pose).vector
            cost += float(np.dot(xi, xi))
        return cost

    def review_trajectory(
        self,
        candidate: TrajectoryCandidate,
        trajectory_idx: int,
        total_trajectories: int,
    ) -> ReviewedTrajectory:
        """Solve OCP and score — does not animate."""
        print("\n" + "=" * 70)
        print(f"TRAJECTORY {trajectory_idx + 1}/{total_trajectories}")
        print("=" * 70)
        print(f"Camera position: {candidate.camera_translation}")
        print(f"IK success rate: {candidate.ik_success_rate:.1f}%")

        ocp, tcp_trajectory = self.solve_ocp_for_candidate(candidate)
        tracking_cost = self._compute_tracking_cost(ocp, tcp_trajectory)
        print(f"  Tracking cost: {tracking_cost:.6f}")

        reviewed = ReviewedTrajectory(
            camera_translation=candidate.camera_translation,
            object_trajectory_poses=[
                p.homogeneous for p in candidate.object_trajectory.poses
            ],
            joint_trajectory_ik=[q for q in candidate.joint_trajectory.configurations],
            joint_trajectory_ocp=ocp.xs,
            tcp_trajectory_poses=[p.homogeneous for p in tcp_trajectory.poses],
            ik_success_rate=candidate.ik_success_rate,
            tracking_cost=tracking_cost,
        )
        return reviewed

    def _animate_reviewed(
        self,
        reviewed: ReviewedTrajectory,
        candidate: TrajectoryCandidate,
    ) -> None:
        print(f"  Displaying trajectory (cost={reviewed.tracking_cost:.6f})...")
        self.visualizer.animate_ocp_solution(
            ocp_states=reviewed.joint_trajectory_ocp,
            object_trajectory=candidate.object_trajectory,
            rmodel=self.rmodel,
        )
        print("  ✓ Done")

    def review_all_candidates(
        self,
        candidates: List[TrajectoryCandidate],
        top_n: Optional[int] = None,
    ) -> List[ReviewedTrajectory]:
        """Solve OCP for all 100%-IK candidates, then animate only the best top_n."""
        eligible = [c for c in candidates if c.ik_success_rate >= 100.0]
        skipped = len(candidates) - len(eligible)

        print("\n" + "=" * 70)
        print(
            f"PROCESSING {len(eligible)}/{len(candidates)} TRAJECTORY CANDIDATES"
            + (f" (skipped {skipped} with IK < 100%)" if skipped else "")
        )
        print("=" * 70)

        # Step 1: solve OCP and score every eligible candidate
        scored: List[tuple] = []
        for idx, candidate in enumerate(eligible):
            reviewed = self.review_trajectory(candidate, idx, len(eligible))
            scored.append((reviewed, candidate))

        # Step 2: rank by tracking cost (lower is better), keep best top_n
        scored.sort(key=lambda x: x[0].tracking_cost)
        best = scored[:top_n] if top_n is not None else scored

        # Step 3: animate selected trajectories
        print("\n" + "=" * 70)
        print(f"ANIMATING {len(best)} BEST TRAJECTORIES (of {len(scored)} solved)")
        print("=" * 70)
        for rank, (reviewed, candidate) in enumerate(best):
            print(f"\n[{rank + 1}/{len(best)}] cost={reviewed.tracking_cost:.6f}  camera={reviewed.camera_translation}")
            self._animate_reviewed(reviewed, candidate)

        self.kept_trajectories = [r for r, _ in best]

        print("\n" + "=" * 70)
        print(f"DONE — kept {len(self.kept_trajectories)}/{len(candidates)} trajectories")
        print("=" * 70)
        return self.kept_trajectories

    def save_kept_trajectories(self, filepath: pathlib.Path):
        """Save all kept trajectories to file."""
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
        print(f"\nSaved {len(self.kept_trajectories)} trajectories to {filepath}")

    @staticmethod
    def load_kept_trajectories(filepath: pathlib.Path) -> dict:
        with open(filepath, "rb") as f:
            return pickle.load(f)

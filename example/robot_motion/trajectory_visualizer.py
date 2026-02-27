"""
Wrapper for visualizing and animating trajectories.
"""

import time
from typing import Dict, List

import numpy as np
import pinocchio as pin
from robomeshcat import Object, Robot, Scene

from object_following_ocp.trajectories import (
    TrajectoryInConfigurationSpace,
    TrajectorySE3,
)

TRAJECTORY_COLORS = [
    [0.95, 0.26, 0.21],  # Red
    [0.13, 0.59, 0.95],  # Blue
    [0.30, 0.69, 0.31],  # Green
    [1.00, 0.76, 0.03],  # Yellow
    [0.61, 0.15, 0.69],  # Purple
    [1.00, 0.60, 0.00],  # Orange
    [0.00, 0.74, 0.83],  # Cyan
    [0.96, 0.26, 0.62],  # Pink
    [0.47, 0.33, 0.28],  # Brown
    [0.38, 0.49, 0.55],  # Blue-grey
]


class TrajectoryVisualizer:
    """
    Handles visualization and animation of trajectories.

    This class provides methods for:
    - Visualizing trajectory waypoints as static spheres
    - Animating the robot through joint / OCP trajectories
    - Replaying all saved trajectories from a TrajectoryLoader
    """

    def __init__(
        self,
        scene: Scene,
        robot: Robot,
        movable_object: Object,
    ):
        self.scene = scene
        self.robot = robot
        self.object = movable_object

        self._displayed_prefixes: List[str] = []
        self._displayed_counts: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Waypoint sphere helpers
    # ------------------------------------------------------------------

    def visualize_waypoints(
        self,
        trajectory: TrajectorySE3,
        sphere_radius: float = 0.01,
        start_color: List[float] = None,
        waypoint_color: List[float] = None,
        name_prefix: str = "target",
    ):
        """Add colored spheres along a trajectory."""
        if start_color is None:
            start_color = [0.0, 1.0, 0.0]
        if waypoint_color is None:
            waypoint_color = [0.5, 0.5, 0.5]

        for k, wM_point in enumerate(trajectory.poses):
            color = start_color if k == 0 else waypoint_color
            sphere_name = f"{name_prefix}_{k}"
            self.scene.add_object(
                Object.create_sphere(
                    radius=sphere_radius, name=sphere_name, color=color
                )
            )
            self.scene[sphere_name].pos[:] = wM_point.translation

        self._displayed_prefixes.append(name_prefix)
        self._displayed_counts[name_prefix] = len(trajectory)
        print(f"  Displayed {len(trajectory)} waypoints (prefix='{name_prefix}')")

    def clear_waypoints(self, name_prefix: str):
        """Remove spheres for a given prefix."""
        if name_prefix not in self._displayed_counts:
            return
        count = self._displayed_counts.pop(name_prefix)
        for k in range(count):
            try:
                self.scene.remove_object(f"{name_prefix}_{k}")
            except Exception:
                pass
        if name_prefix in self._displayed_prefixes:
            self._displayed_prefixes.remove(name_prefix)

    def clear_all_waypoints(self):
        """Remove all spheres added through this visualizer."""
        for prefix in list(self._displayed_prefixes):
            self.clear_waypoints(prefix)

    # ------------------------------------------------------------------
    # Static overview of all trajectories
    # ------------------------------------------------------------------

    def display_all_trajectories(
        self,
        trajectories: List[TrajectorySE3],
        labels: List[str] = None,
        sphere_radius: float = 0.008,
        colors: List[List[float]] = None,
        clear_previous: bool = True,
    ):
        """Paint all trajectories simultaneously as colored spheres in the scene."""
        if clear_previous:
            self.clear_all_waypoints()

        if labels is None:
            labels = [f"traj_{i}" for i in range(len(trajectories))]
        if colors is None:
            colors = [
                TRAJECTORY_COLORS[i % len(TRAJECTORY_COLORS)]
                for i in range(len(trajectories))
            ]

        print(f"\nDisplaying {len(trajectories)} trajectories as static spheres:")
        for i, (traj, label, color) in enumerate(zip(trajectories, labels, colors)):
            start_color = [min(1.0, c * 1.4) for c in color]
            self.visualize_waypoints(
                trajectory=traj,
                sphere_radius=sphere_radius,
                start_color=start_color,
                waypoint_color=color,
                name_prefix=label,
            )
            print(
                f"  [{i + 1}/{len(trajectories)}] {label}  color={[round(c, 2) for c in color]}"
            )

    # ------------------------------------------------------------------
    # Robot animation helpers
    # ------------------------------------------------------------------

    def animate_joint_trajectory(
        self,
        joint_trajectory: TrajectoryInConfigurationSpace,
        object_trajectory: TrajectorySE3,
        rmodel: pin.Model,
        rdata: pin.Data,
        dt: float = 0.01,
    ) -> List[pin.SE3]:
        """
        Animate the robot through a joint trajectory.

        Args:
            joint_trajectory: Sequence of joint configurations
            object_trajectory: Object poses shown in sync
            rmodel / rdata: Pinocchio model / data
            dt: Seconds between frames
        Returns:
            List of TCP SE3 poses collected during animation
        """
        tcp_frame_id = rmodel.getFrameId("panda_hand_tcp")
        tcp_poses = []

        for k, q in enumerate(joint_trajectory):
            time.sleep(dt)
            self.robot[:] = q
            pin.framesForwardKinematics(rmodel, rdata, q)
            tcp_poses.append(rdata.oMf[tcp_frame_id].copy())

            if k < len(object_trajectory):
                self.object.pose = object_trajectory[k].homogeneous

        return tcp_poses

    def animate_ocp_solution(
        self,
        ocp_states: List[np.ndarray],
        object_trajectory: TrajectorySE3,
        rmodel: pin.Model,
        dt: float = 0.01,
    ):
        """
        Animate the robot through an OCP solution (states = [q, v]).

        Args:
            ocp_states: List of [q, v] state vectors
            object_trajectory: Object poses shown in sync
            rmodel: Pinocchio robot model
            dt: Seconds between frames
        """
        for k, xs in enumerate(ocp_states):
            time.sleep(dt)
            self.robot[:] = xs[: rmodel.nq]

            if k < len(object_trajectory):
                self.object.pose = object_trajectory[k].homogeneous

    # ------------------------------------------------------------------
    # Replay all saved trajectories
    # ------------------------------------------------------------------

    def replay_all_saved_trajectories(
        self,
        loader,
        rmodel: pin.Model,
        rdata: pin.Data,
        robot_config,
        dataloader,
        solution_type: str = "ik",
        dt: float = 0.01,
        show_waypoints: bool = True,
    ):
        """
        Replay every trajectory stored in a TrajectoryLoader one by one.

        Args:
            loader: TrajectoryLoader instance
            rmodel / rdata: Pinocchio model / data
            robot_config: RobotConfig
            dataloader: DataLoader
            solution_type: 'ik' or 'ocp' ('ocp' falls back to 'ik' if unavailable)
            dt: Seconds between frames
            show_waypoints: Whether to show target spheres during animation
        """
        from example.robot_motion.utils.ocp_trajectory_converter import (
            OCPTrajectoryConverter,
        )

        num = loader.num_trajectories
        meta = loader.get_metadata()
        print(f"\n{'=' * 60}")
        print(f"Replaying {num} trajectories  [{meta['mesh_id']}]")
        print(f"Solution type : {solution_type}")
        print(f"dt={dt}s per frame")
        print(f"{'=' * 60}")

        for idx in range(num):
            traj_data = loader.get_trajectory(idx)
            cam = np.array(traj_data["camera_translation"])
            object_traj = traj_data["object_trajectory"]
            joint_traj_ik = traj_data["joint_trajectory_ik"]

            print(f"\n--- Trajectory {idx + 1}/{num} ---")
            print(f"Camera : {cam}")
            print(f"IK rate: {traj_data['ik_success_rate']:.1f}%")

            ocp_converter = OCPTrajectoryConverter(
                robot_config=robot_config,
                camera_translation=cam,
                grasp_correction_angle_deg=meta["grasp_correction_angle_deg"],
                elevation_angle_deg=meta["elevation_angle_deg"],
            )
            tcp_traj = ocp_converter.compute_tcp_trajectory(dataloader)

            prefix = f"replay_{idx}"
            if show_waypoints:
                color = TRAJECTORY_COLORS[idx % len(TRAJECTORY_COLORS)]
                start_color = [min(1.0, c * 1.4) for c in color]
                self.visualize_waypoints(
                    trajectory=tcp_traj,
                    sphere_radius=0.008,
                    start_color=start_color,
                    waypoint_color=color,
                    name_prefix=prefix,
                )

            use_ocp = solution_type == "ocp" and "joint_trajectory_ocp" in traj_data
            if solution_type == "ocp" and "joint_trajectory_ocp" not in traj_data:
                print("  ⚠ No OCP solution saved — falling back to IK")

            if use_ocp:
                print("  Replaying OCP solution...")
                self.animate_ocp_solution(
                    ocp_states=traj_data["joint_trajectory_ocp"],
                    object_trajectory=object_traj,
                    rmodel=rmodel,
                    dt=dt,
                )
            else:
                print("  Replaying IK solution...")
                self.animate_joint_trajectory(
                    joint_trajectory=joint_traj_ik,
                    object_trajectory=object_traj,
                    rmodel=rmodel,
                    rdata=rdata,
                    dt=dt,
                )

            if show_waypoints:
                self.clear_waypoints(prefix)

        print(f"\n{'=' * 60}")
        print("Replay complete.")
        print(f"{'=' * 60}")

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    def set_robot_and_object_pose(
        self,
        joint_config: np.ndarray,
        object_pose: pin.SE3,
    ):
        self.robot[:] = joint_config
        self.object.pose = object_pose.homogeneous

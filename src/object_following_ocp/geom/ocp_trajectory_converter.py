"""
Wrapper for computing trajectories for OCP from object trajectories.
"""

import numpy as np

from object_following_ocp.data.data_loader import DataLoader, RobotConfig
from object_following_ocp.geom.grasp_transforms import (
    GraspTransformChain,
    GraspTransformConfig,
)
from object_following_ocp.geom.trajectories import TrajectorySE3


class OCPTrajectoryConverter:
    """
    Converts object trajectories to TCP trajectories for OCP.

    This class handles the transformation from camera-frame object trajectory
    to world-frame TCP trajectory (WITH gripper depth offset).
    """

    def __init__(
        self,
        robot_config: RobotConfig,
        camera_translation: np.ndarray = None,
        grasp_correction_angle_deg: float = 90.0,
        elevation_angle_deg: float = 25.0,
    ):
        """
        Initialize OCP trajectory converter.

        Args:
            robot_config: Robot configuration (contains gripper_depth)
            camera_translation: Camera position in world frame
            grasp_correction_angle_deg: Grasp correction angle
            elevation_angle_deg: Visualization alignment angle
        """
        if camera_translation is None:
            camera_translation = np.array([0, -0.7, -1.0])

        # Create grasp transform configuration
        self.grasp_config = GraspTransformConfig.from_robot_config(
            robot_config=robot_config,
            camera_translation=camera_translation,
            grasp_correction_angle_deg=grasp_correction_angle_deg,
            elevation_angle_deg=elevation_angle_deg,
        )

        # Initialize transformation chain
        self.transform_chain = GraspTransformChain(self.grasp_config)

    def compute_tcp_trajectory(
        self,
        dataloader: DataLoader,
    ) -> TrajectorySE3:
        """
        Compute TCP trajectory for OCP.

        Args:
            dataloader: DataLoader containing object trajectory and grasp

        Returns:
            TCP trajectory in world frame (includes gripper depth offset)
        """
        # Get camera-frame trajectory and grasp
        object_traj_camera = dataloader.to_trajectory_SE3()
        objectM_grasp = dataloader.best_grasp_SE3

        # Get TCP trajectory (WITH gripper depth offset - for OCP)
        tcp_traj_world = self.transform_chain.transform_tcp_trajectory(
            object_traj_camera, objectM_grasp
        )

        return tcp_traj_world

    def get_transform_summary(self) -> str:
        """Get summary of transformations."""
        return self.transform_chain.get_transform_summary()

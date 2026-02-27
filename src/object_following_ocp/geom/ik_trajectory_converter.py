"""
Wrapper for computing trajectories for IK from object trajectories.
"""

import numpy as np
import pinocchio as pin
from typing import Tuple

from object_following_ocp.data.data_loader import DataLoader, RobotConfig
from object_following_ocp.geom.grasp_transforms import (
    GraspTransformChain,
    GraspTransformConfig,
)
from object_following_ocp.geom.trajectories import TrajectorySE3


class IKTrajectoryConverter:
    """
    Converts object trajectories to end-effector trajectories for IK.

    This class handles the transformation from camera-frame object trajectory
    to world-frame end-effector trajectory (without TCP offset).
    """

    def __init__(
        self,
        robot_config: RobotConfig,
        camera_translation: np.ndarray = None,
        grasp_correction_angle_deg: float = 90.0,
        elevation_angle_deg: float = 25.0,
    ):
        """
        Initialize IK trajectory converter.

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

    def compute_trajectories(
        self,
        dataloader: DataLoader,
    ) -> Tuple[TrajectorySE3, TrajectorySE3]:
        """
        Compute object and end-effector trajectories.

        Args:
            dataloader: DataLoader containing object trajectory and grasp

        Returns:
            Tuple of (object_traj_world, ee_traj_world) where:
                - object_traj_world: Object trajectory in world frame
                - ee_traj_world: End-effector trajectory in world frame (for IK)
                                 Does NOT include TCP offset
        """
        # Get camera-frame trajectory and grasp
        object_traj_camera = dataloader.to_trajectory_SE3()
        objectM_grasp = dataloader.best_grasp_SE3

        # Transform to world frame
        object_traj_world = self.transform_chain.transform_object_trajectory(
            object_traj_camera
        )

        # Get EE trajectory (WITHOUT gripper depth offset - for IK)
        ee_traj_world = self.transform_chain.transform_ee_trajectory(
            object_traj_camera, objectM_grasp
        )

        return object_traj_world, ee_traj_world

    def get_transform_summary(self) -> str:
        """Get summary of transformations."""
        return self.transform_chain.get_transform_summary()

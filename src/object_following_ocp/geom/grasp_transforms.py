from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pinocchio as pin

from object_following_ocp.data.data_loader import RobotConfig
from object_following_ocp.geom.trajectories import TrajectorySE3


@dataclass
class GraspTransformConfig:
    """Configuration for grasp transformations."""

    # Camera position in world frame
    camera_translation: np.ndarray = None

    # Gripper depth offset (TCP offset from grasp frame)
    # This can be loaded from RobotConfig.gripper_depth
    gripper_depth: float = 0  # 0.1034

    # Grasp correction angle (frame convention mismatch)
    grasp_correction_angle_deg: float = 90.0
    grasp_correction_axis: np.ndarray = None  # Default: Z-axis

    # Visualization alignment
    elevation_angle_deg: float = 25.0

    def __post_init__(self):
        if self.camera_translation is None:
            self.camera_translation = np.array([0, -1.0, -1.0])
        if self.grasp_correction_axis is None:
            self.grasp_correction_axis = np.array([0, 0, 1])  # Z-axis

    @classmethod
    def from_robot_config(
        cls,
        robot_config: RobotConfig,
        camera_translation: Optional[np.ndarray] = None,
        grasp_correction_angle_deg: float = 90.0,
        elevation_angle_deg: float = 25.0,
    ) -> "GraspTransformConfig":
        """
        Create GraspTransformConfig from RobotConfig.

        Args:
            robot_config: RobotConfig containing gripper_depth
            camera_translation: Camera position in world frame
            grasp_correction_angle_deg: Grasp correction angle
            elevation_angle_deg: Visualization alignment angle

        Returns:
            GraspTransformConfig instance
        """
        return cls(
            camera_translation=camera_translation,
            gripper_depth=robot_config.gripper_depth,
            grasp_correction_angle_deg=grasp_correction_angle_deg,
            elevation_angle_deg=elevation_angle_deg,
        )


class GraspTransformChain:
    """
    Handles the complete transformation chain from camera frame to robot end-effector.

    Transformation chain:
    world <- world_aligned <- camera <- object <- grasp <- grasp_corrected <- ee/tcp

    Notation: frameAM_frameB means "frame B expressed in frame A"

    This class integrates with DataLoader to transform object trajectories and grasps
    from camera frame to robot world frame.
    """

    def __init__(self, config: Optional[GraspTransformConfig] = None):
        """
        Initialize the transformation chain.

        Args:
            config: Configuration for transformations. Uses defaults if None.
        """
        self.config = config if config is not None else GraspTransformConfig()

        # Build fixed transforms
        self._build_fixed_transforms()

    def _build_fixed_transforms(self):
        """Build all fixed transformations that don't depend on object trajectory."""

        # Camera frame in world frame
        self.wM_camera = pin.SE3.Identity()
        self.wM_camera.translation = self.config.camera_translation

        # Grasp correction (frame convention mismatch)
        self.graspM_grasp_corrected = pin.SE3.Identity()
        correction_rotation = pin.exp3(
            self.config.grasp_correction_axis
            * np.deg2rad(self.config.grasp_correction_angle_deg)
        )
        self.graspM_grasp_corrected.rotation = correction_rotation

        # TCP offset from corrected grasp frame (gripper depth)
        self.grasp_correctedM_tcp = pin.SE3.Identity()
        self.grasp_correctedM_tcp.translation = np.array(
            [0, 0, self.config.gripper_depth]
        )

        # Visualization alignment rotation
        R_alignment = pin.exp3(np.array([0, 0, np.deg2rad(90)])) @ pin.exp3(
            np.array([-np.pi / 2 - np.deg2rad(self.config.elevation_angle_deg), 0, 0])
        )
        self.worldM_world_aligned = pin.SE3(R_alignment, np.array([0, 0, 0]))

    def transform_object_trajectory(
        self,
        camera_frame_trajectory: TrajectorySE3,
    ) -> TrajectorySE3:
        """
        Transform object trajectory from camera frame to world frame.

        Args:
            camera_frame_trajectory: Object trajectory in camera frame (from DataLoader)

        Returns:
            Object trajectory in world frame
        """
        return self.worldM_world_aligned * self.wM_camera * camera_frame_trajectory

    def transform_ee_trajectory(
        self,
        camera_frame_trajectory: TrajectorySE3,
        objectM_grasp: pin.SE3,
    ) -> TrajectorySE3:
        """
        Transform object trajectory to end-effector trajectory in world frame.

        This applies the complete chain:
        world <- world_aligned <- camera <- object <- grasp <- grasp_corrected

        Args:
            camera_frame_trajectory: Object trajectory in camera frame (from DataLoader)
            objectM_grasp: Grasp pose in object frame (from DataLoader.best_grasp_SE3)

        Returns:
            End-effector trajectory in world frame
        """
        # First transform object to world frame
        object_traj_world = self.transform_object_trajectory(camera_frame_trajectory)

        # Then apply grasp and correction
        ee_traj_world = object_traj_world * objectM_grasp * self.graspM_grasp_corrected

        return ee_traj_world

    def transform_tcp_trajectory(
        self,
        camera_frame_trajectory: TrajectorySE3,
        objectM_grasp: pin.SE3,
    ) -> TrajectorySE3:
        """
        Transform object trajectory to TCP trajectory in world frame.

        This includes the gripper depth offset.

        Args:
            camera_frame_trajectory: Object trajectory in camera frame (from DataLoader)
            objectM_grasp: Grasp pose in object frame (from DataLoader.best_grasp_SE3)

        Returns:
            TCP trajectory in world frame (includes gripper offset)
        """
        ee_traj = self.transform_ee_trajectory(camera_frame_trajectory, objectM_grasp)
        return ee_traj * self.grasp_correctedM_tcp

    def compute_object_pose(
        self,
        cameraM_object: Union[pin.SE3, TrajectorySE3],
    ) -> Union[pin.SE3, TrajectorySE3]:
        """
        Compute object pose(s) in world frame.

        Args:
            cameraM_object: Object frame in camera frame
                           Can be single SE3 or TrajectorySE3

        Returns:
            wM_object: Object frame in world frame (same type as input)
        """
        return self.worldM_world_aligned * self.wM_camera * cameraM_object

    def compute_ee_pose(
        self,
        cameraM_object: Union[pin.SE3, TrajectorySE3],
        objectM_grasp: pin.SE3,
        include_tcp_offset: bool = False,
    ) -> Union[pin.SE3, TrajectorySE3]:
        """
        Compute end-effector pose(s) in world frame.

        Args:
            cameraM_object: Object frame in camera frame
                           Can be single SE3 or TrajectorySE3
            objectM_grasp: Grasp frame in object frame (from GraspGen)
            include_tcp_offset: If True, includes gripper depth offset

        Returns:
            wM_ee: End-effector frame in world frame (same type as input)
        """
        wM_ee = (
            self.worldM_world_aligned
            * self.wM_camera
            * cameraM_object
            * objectM_grasp
            * self.graspM_grasp_corrected
        )

        if include_tcp_offset:
            wM_ee = wM_ee * self.grasp_correctedM_tcp

        return wM_ee

    def compute_tcp_pose(
        self,
        cameraM_object: Union[pin.SE3, TrajectorySE3],
        objectM_grasp: pin.SE3,
    ) -> Union[pin.SE3, TrajectorySE3]:
        """
        Compute TCP pose(s) in world frame (includes gripper depth offset).

        Args:
            cameraM_object: Object frame in camera frame
                           Can be single SE3 or TrajectorySE3
            objectM_grasp: Grasp frame in object frame (from GraspGen)

        Returns:
            wM_tcp: TCP frame in world frame (same type as input)
        """
        return self.compute_ee_pose(
            cameraM_object, objectM_grasp, include_tcp_offset=True
        )

    def update_camera_transform(self, translation: np.ndarray):
        """
        Update camera position in world frame.

        Args:
            translation: New camera translation [x, y, z]
        """
        self.config.camera_translation = translation
        self.wM_camera.translation = translation

    def compute_object_pose_from_tcp(
        self,
        tcp_pose: Union[pin.SE3, TrajectorySE3],
        objectM_grasp: pin.SE3,
    ) -> Union[pin.SE3, TrajectorySE3]:
        """
        Recover object pose(s) in world frame from TCP pose(s).

        Inverts the chain:
        wM_tcp = wM_object * objectM_grasp * graspM_grasp_corrected * grasp_correctedM_tcp

        So: wM_object = wM_tcp * grasp_correctedM_tcp^{-1} * graspM_grasp_corrected^{-1} * objectM_grasp^{-1}
        """
        suffix_inv = (
            self.grasp_correctedM_tcp.inverse()
            * self.graspM_grasp_corrected.inverse()
            * objectM_grasp.inverse()
        )

        if isinstance(tcp_pose, TrajectorySE3):
            return tcp_pose * suffix_inv
        else:
            return tcp_pose * suffix_inv

    def get_transform_summary(self) -> str:
        """Get a human-readable summary of all transformations."""
        summary = []
        summary.append("=" * 60)
        summary.append("Grasp Transform Chain Summary")
        summary.append("=" * 60)
        summary.append(f"Camera translation: {self.config.camera_translation}")
        summary.append(f"Gripper depth: {self.config.gripper_depth} m")
        summary.append(
            f"Grasp correction: {self.config.grasp_correction_angle_deg}° "
            f"around {self.config.grasp_correction_axis}"
        )
        summary.append(f"Elevation angle: {self.config.elevation_angle_deg}°")
        summary.append("")
        summary.append("Transformation chain:")
        summary.append(
            "  world <- world_aligned <- camera <- object <- grasp <- grasp_corrected <- ee/tcp"
        )
        summary.append("=" * 60)
        return "\n".join(summary)


def quaternion_to_rotation_matrix(
    quat: np.ndarray, normalize: bool = True
) -> np.ndarray:
    """
    Convert quaternion to rotation matrix, handling numpy float types.

    Args:
        quat: Quaternion in wxyz format [w, x, y, z]
        normalize: If True, normalize the quaternion first (default: True)

    Returns:
        3x3 rotation matrix

    Example:
        >>> quat = np.array([0.7071, 0.7071, 0.0, 0.0])
        >>> R = quaternion_to_rotation_matrix(quat)
        >>> print(R.shape)
        (3, 3)
    """
    if normalize:
        quat = quat / np.linalg.norm(quat)

    return pin.Quaternion(
        float(quat[0]),  # w
        float(quat[1]),  # x
        float(quat[2]),  # y
        float(quat[3]),  # z
    ).toRotationMatrix()

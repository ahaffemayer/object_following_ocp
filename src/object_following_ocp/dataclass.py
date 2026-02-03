import json
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import yaml


@dataclass
class RobotConfig:
    """Robot-specific configuration from YAML"""
    W_xREG: float
    W_uREG: float
    W_gripper_pose: float
    W_gripper_pose_term: float
    W_limit: float
    safety_threshold: float
    dt: float
    gripper_depth: float


@dataclass
class ObjectInfo:
    """Object information from JSON"""
    mesh_id: str
    score: float
    scale: float


@dataclass
class PoseData:
    """Single pose from trajectory"""
    im_id: int
    object_id: int
    score: float
    R: np.ndarray  # 3x3 rotation matrix
    t: np.ndarray  # 3x1 translation vector
    bbox_visib: List[int]
    time: float


class TrajectoryParser:
    """Parse object tracking JSON file"""

    def __init__(self, json_path: str):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

    def get_object_info(self, object_id: int) -> ObjectInfo:
        """Extract object information"""
        obj_data = self.data['objects'][str(object_id)]
        return ObjectInfo(
            mesh_id=obj_data['mesh'],
            score=obj_data['score'],
            scale=obj_data['scale']
        )

    def get_trajectory(self) -> List[PoseData]:
        """Extract full trajectory"""
        trajectory = []
        for pose in self.data['poses']:
            trajectory.append(PoseData(
                im_id=pose['im_id'],
                object_id=pose['object_id'],
                score=pose['score'],
                R=np.array(pose['R']),
                t=np.array(pose['t']),
                bbox_visib=pose['bbox_visib'],
                time=pose['time']
            ))
        return trajectory

    def get_poses_for_object(self, object_id: int) -> List[PoseData]:
        """Get trajectory for specific object"""
        trajectory = self.get_trajectory()
        return [pose for pose in trajectory if pose.object_id == object_id]


class ConfigLoader:
    """Load robot configuration from YAML"""

    @staticmethod
    def load(yaml_path: str) -> RobotConfig:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        weights = config['weights']
        return RobotConfig(
            W_xREG=weights['W_xREG'],
            W_uREG=weights['W_uREG'],
            W_gripper_pose=weights['W_gripper_pose'],
            W_gripper_pose_term=weights['W_gripper_pose_term'],
            W_limit=weights['W_limit'],
            safety_threshold=config['safety_threshold'],
            dt=config['dt'],
            gripper_depth=config['gripper_depth']
        )


# Usage example
if __name__ == "__main__":
    # Load robot configuration
    robot_config = ConfigLoader.load(
        "/workspaces/object_following_ocp/example/robot_config.yml")

    # Parse trajectory data
    traj_parser = TrajectoryParser(
        "/workspaces/object_following_ocp/ressources/json/bowl1.props-dinov2-ffa-22.gpt4_scaled.best_object.poses-dinov2-22-graph.smoothed-movavg.json")

    # Get object info
    object_info = traj_parser.get_object_info(object_id=2)
    print(f"Mesh ID: {object_info.mesh_id}")
    print(f"Scale: {object_info.scale}")

    # Get trajectory
    trajectory = traj_parser.get_trajectory()
    print(f"Number of poses: {len(trajectory)}")

    # Access specific pose
    first_pose = trajectory[0]
    print(f"First pose rotation:\n{first_pose.R}")
    print(f"First pose translation: {first_pose.t}")

    # Use robot config
    print(f"Gripper pose weight: {robot_config.W_gripper_pose}")

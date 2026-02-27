import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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
    mesh_path: Path
    texture_path: Optional[Path] = None


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

    def __init__(self, json_path: str | Path, mesh_base_dir: Optional[str | Path] = None):
        """
        Initialize trajectory parser

        Args:
            json_path: Path to JSON file containing trajectory data
            mesh_base_dir: Base directory for meshes. If None, uses ../meshes relative to json_path
        """
        self.json_path = Path(json_path)

        with open(self.json_path, 'r') as f:
            self.data = json.load(f)

        # Set mesh base directory
        if mesh_base_dir is None:
            self.mesh_base_dir = self.json_path.parent.parent / "meshes"
        else:
            self.mesh_base_dir = Path(mesh_base_dir)

    def get_object_info(self, object_id: int, texture_name: str = "material_0.png") -> ObjectInfo:
        """
        Extract object information including mesh paths

        Args:
            object_id: ID of the object
            texture_name: Name of texture file (default: material_0.png)

        Returns:
            ObjectInfo with mesh and texture paths resolved
        """
        obj_data = self.data['objects'][str(object_id)]
        mesh_id = obj_data['mesh']

        # Construct mesh path: ../meshes/{mesh_name}/{mesh_name}.obj
        mesh_dir = self.mesh_base_dir / mesh_id
        mesh_path = mesh_dir / f"{mesh_id}.obj"

        # Construct texture path
        texture_path = mesh_dir / texture_name if texture_name else None

        # Verify paths exist
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

        if texture_path and not texture_path.exists():
            print(f"Warning: Texture file not found: {texture_path}")
            texture_path = None

        return ObjectInfo(
            mesh_id=mesh_id,
            score=obj_data['score'],
            scale=obj_data['scale'],
            mesh_path=mesh_path,
            texture_path=texture_path
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

    def get_available_objects(self) -> Dict[int, str]:
        """Get dictionary of available object IDs and their mesh IDs"""
        return {
            int(obj_id): obj_data['mesh']
            for obj_id, obj_data in self.data['objects'].items()
        }


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

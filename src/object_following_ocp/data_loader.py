import json
import pathlib
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pinocchio as pin
import yaml

from object_following_ocp.trajectories import TrajectorySE3


@dataclass
class ObjectInfo:
    """Object information from JSON"""
    mesh_id: str
    score: float
    scale: float
    mesh_path: pathlib.Path
    texture_path: Optional[pathlib.Path] = None


@dataclass
class PoseData:
    """Single pose from trajectory"""
    im_id: int
    object_id: int
    score: float
    R: np.ndarray  # 3x3 rotation matrix
    t: np.ndarray  # 3x1 translation vector
    bbox_visib: list[int]
    time: float

    def to_SE3(self) -> pin.SE3:
        """Convert to Pinocchio SE3 object"""
        return pin.SE3(self.R, self.t)


@dataclass
class GraspPose:
    """Grasp pose with position and orientation"""
    name: str
    confidence: float
    position: np.ndarray  # 3x1 position vector
    orientation: np.ndarray  # quaternion [w, x, y, z]

    def to_SE3(self) -> pin.SE3:
        """Convert to Pinocchio SE3 object"""
        # Quaternion is [w, x, y, z], Pinocchio expects [x, y, z, w]
        quat_pinocchio = np.array([
            self.orientation[1],  # x
            self.orientation[2],  # y
            self.orientation[3],  # z
            self.orientation[0]   # w
        ])
        return pin.SE3(pin.Quaternion(quat_pinocchio).matrix(), self.position)


@dataclass
class DataLoader:
    """Loader for a single object trajectory with its grasp poses and scale"""
    object_info: ObjectInfo
    object_id: int
    poses: list[PoseData]
    grasp_poses: list[GraspPose]

    def __init__(
        self,
        object_trajectory_path: pathlib.Path,
        grasp_poses_SE3_path: pathlib.Path,
        scales_path: pathlib.Path
    ) -> None:
        """
        Load a single object trajectory with its grasp poses.

        Args:
            object_trajectory_path: Path to object trajectory JSON file
            grasp_poses_SE3_path: Path to grasp poses YAML file
            scales_path: Path to scales JSON file (contains all objects)

        Usage:
            loader = DataLoader(traj_path, grasp_path, scales_path)
            trajectory = loader.to_trajectory_SE3()
            best_grasp = loader.best_grasp
        """
        # --- Load scales first (contains ALL objects) ---
        with open(scales_path, "r") as f:
            scales_data = json.load(f)

        scales: Dict[str, dict] = {}
        for scale_entry in scales_data:
            mesh_name = scale_entry["Name"].strip()
            scales[mesh_name] = {
                "scale": scale_entry["scale"],
                "scale_from_dataset": scale_entry["scale_from_dataset"]
            }

        # --- Load object trajectory ---
        with open(object_trajectory_path, "r") as f:
            object_data = json.load(f)

        # Parse object info (should be only one per file)
        if len(object_data["objects"]) != 1:
            raise ValueError(
                f"Expected 1 object per file, got {len(object_data['objects'])}")

        obj_id_str, obj_info_dict = next(iter(object_data["objects"].items()))
        self.object_id = int(obj_id_str)
        mesh_id = obj_info_dict["mesh"]

        # Use scale from scales file if available (it overrides), otherwise from object file
        if mesh_id in scales:
            scale = scales[mesh_id]["scale"]
        else:
            scale = obj_info_dict["scale"]

        # Mesh path (stored in the parent folder of the trajs, in the directory mesh/id/id.json)
        mesh_path = object_trajectory_path.parent.parent / \
            "meshes" / f"{mesh_id}" / f"{mesh_id}.obj"
        # Texture path (same folder but different name)
        texture_path = mesh_path.parent / "material_0.png"

        self.object_info = ObjectInfo(
            mesh_id=mesh_id,
            score=obj_info_dict["score"],
            scale=scale,
            mesh_path=mesh_path,
            texture_path=texture_path
        )

        # Parse poses
        self.poses: list[PoseData] = []
        for pose_dict in object_data["poses"]:
            pose = PoseData(
                im_id=pose_dict["im_id"],
                object_id=pose_dict["object_id"],
                score=pose_dict["score"],
                R=np.array(pose_dict["R"]),
                t=np.array(pose_dict["t"]),
                bbox_visib=pose_dict["bbox_visib"],
                time=pose_dict["time"]
            )
            self.poses.append(pose)

        # --- Load grasp poses ---
        with open(grasp_poses_SE3_path, "r") as f:
            grasp_data = yaml.safe_load(f)

        self.grasp_poses: list[GraspPose] = []
        for grasp_name, grasp_info in grasp_data["grasps"].items():
            orientation_data = grasp_info["orientation"]
            grasp = GraspPose(
                name=grasp_name,
                confidence=grasp_info["confidence"],
                position=np.array(grasp_info["position"]),
                orientation=np.array([
                    orientation_data["w"],
                    orientation_data["xyz"][0],
                    orientation_data["xyz"][1],
                    orientation_data["xyz"][2]
                ])
            )
            self.grasp_poses.append(grasp)

    def to_trajectory_SE3(self) -> TrajectorySE3:
        """Convert poses to SE3 trajectory. The object trajectory is in the Camera Frame."""
        se3_poses = [pose.to_SE3() for pose in self.poses]
        return TrajectorySE3(se3_poses)

    @property
    def best_grasp(self) -> GraspPose:
        """Get the grasp pose with highest confidence"""
        return max(self.grasp_poses, key=lambda g: g.confidence)

    @property
    def best_grasp_SE3(self) -> pin.SE3:
        """Get the best grasp pose as SE3. The grasp is in the object frame."""
        return self.best_grasp.to_SE3()


if __name__ == "__main__":

    grasp_path = pathlib.Path(
        "/workspaces/object_following_ocp/ressources/filtered_grasps/0d0d1c59b0474d2ea92ce2e172c9f56a_filtered.yml")
    object_traj_path = pathlib.Path(
        "/workspaces/object_following_ocp/ressources/json/bowl1.props-dinov2-ffa-22.gpt4_scaled.best_object.poses-dinov2-22-graph.smoothed-movavg.json")
    scale_path = pathlib.Path(
        "/workspaces/object_following_ocp/ressources/grasps_scales.json")

    dataloader = DataLoader(object_trajectory_path=object_traj_path,
                            grasp_poses_SE3_path=grasp_path, scales_path=scale_path)

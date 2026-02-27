import json
import pathlib
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pinocchio as pin
import yaml

from object_following_ocp.geom.trajectories import TrajectorySE3


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
        quat_pinocchio = np.array(
            [
                self.orientation[1],  # x
                self.orientation[2],  # y
                self.orientation[3],  # z
                self.orientation[0],  # w
            ]
        )
        return pin.SE3(pin.Quaternion(quat_pinocchio).matrix(), self.position)


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
    T: int
    gripper_depth: float


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
        scales_path: pathlib.Path,
        grasp_poses_SE3_path: Optional[pathlib.Path] = None,
        load_grasps: bool = False,
        grasps_directory: Optional[pathlib.Path] = None,
    ) -> None:
        """
        Load a single object trajectory with optional grasp poses.

        Args:
            object_trajectory_path: Path to object trajectory JSON file
            scales_path: Path to scales JSON file (contains all objects)
            grasp_poses_SE3_path: Path to grasp poses YAML file (optional, takes precedence)
            load_grasps: If True and grasp_poses_SE3_path is None, auto-load grasps based on mesh_id
            grasps_directory: Directory containing grasp files (default: trajectory_path/../../filtered_grasps/)

        Usage:
            # Without grasps
            loader = DataLoader(traj_path, scales_path)

            # With explicit grasp path
            loader = DataLoader(traj_path, scales_path, grasp_path)

            # With auto-loaded grasps
            loader = DataLoader(traj_path, scales_path, load_grasps=True)

            # With custom grasps directory
            loader = DataLoader(traj_path, scales_path, load_grasps=True,
                              grasps_directory=Path("custom/grasps/dir"))
        """
        # --- Load scales first (contains ALL objects) ---
        with open(scales_path, "r") as f:
            scales_data = json.load(f)

        scales: Dict[str, dict] = {}
        for scale_entry in scales_data:
            mesh_name = scale_entry["Name"].strip()
            scales[mesh_name] = {
                "scale": scale_entry["scale"],
                "scale_from_dataset": scale_entry["scale_from_dataset"],
            }

        # --- Load object trajectory ---
        with open(object_trajectory_path, "r") as f:
            object_data = json.load(f)

        # Parse object info (should be only one per file)
        if len(object_data["objects"]) != 1:
            raise ValueError(
                f"Expected 1 object per file, got {len(object_data['objects'])}"
            )

        obj_id_str, obj_info_dict = next(iter(object_data["objects"].items()))
        self.object_id = int(obj_id_str)
        mesh_id = obj_info_dict["mesh"]

        # Use scale from scales file if available (it overrides), otherwise from object file
        if mesh_id in scales:
            scale = scales[mesh_id]["scale"]
        else:
            scale = obj_info_dict["scale"]

        # Mesh path (stored in the parent folder of the trajs, in the directory meshes/id/id.obj)
        mesh_path = (
            object_trajectory_path.parent.parent
            / "meshes"
            / f"{mesh_id}"
            / f"{mesh_id}.obj"
        )
        # Texture path (same folder but different name)
        texture_path = mesh_path.parent / "material_0.png"

        self.object_info = ObjectInfo(
            mesh_id=mesh_id,
            score=obj_info_dict["score"],
            scale=scale,
            mesh_path=mesh_path,
            texture_path=texture_path,
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
                time=pose_dict["time"],
            )
            self.poses.append(pose)

        # --- Load grasp poses (optional) ---
        self.grasp_poses: list[GraspPose] = []

        # Determine grasp file path
        grasp_file_path = None

        if grasp_poses_SE3_path is not None:
            # Explicit path provided
            grasp_file_path = grasp_poses_SE3_path
        elif load_grasps:
            # Auto-load based on mesh_id
            if grasps_directory is None:
                # Default: trajectory_path/../../filtered_grasps/
                grasps_directory = (
                    object_trajectory_path.parent.parent / "filtered_grasps"
                )

            grasp_file_path = grasps_directory / f"{mesh_id}_filtered.yml"

            if not grasp_file_path.exists():
                # Try without _filtered suffix
                grasp_file_path = grasps_directory / f"{mesh_id}.yml"

                if not grasp_file_path.exists():
                    print(
                        f"Warning: Grasp file not found for mesh_id '{mesh_id}' in {grasps_directory}"
                    )
                    grasp_file_path = None

        # Load grasps if path was determined
        if grasp_file_path is not None and grasp_file_path.exists():
            with open(grasp_file_path, "r") as f:
                grasp_data = yaml.safe_load(f)

            for grasp_name, grasp_info in grasp_data["grasps"].items():
                orientation_data = grasp_info["orientation"]
                grasp = GraspPose(
                    name=grasp_name,
                    confidence=grasp_info["confidence"],
                    position=np.array(grasp_info["position"]),
                    orientation=np.array(
                        [
                            orientation_data["w"],
                            orientation_data["xyz"][0],
                            orientation_data["xyz"][1],
                            orientation_data["xyz"][2],
                        ]
                    ),
                )
                self.grasp_poses.append(grasp)

    def to_trajectory_SE3(self) -> TrajectorySE3:
        """Convert poses to SE3 trajectory. The object trajectory is in the Camera Frame."""
        se3_poses = [pose.to_SE3() for pose in self.poses]
        return TrajectorySE3(se3_poses)

    @property
    def best_grasp(self) -> Optional[GraspPose]:
        """Get the grasp pose with highest confidence, or None if no grasps loaded"""
        if not self.grasp_poses:
            return None
        return max(self.grasp_poses, key=lambda g: g.confidence)

    @property
    def best_grasp_SE3(self) -> Optional[pin.SE3]:
        """Get the best grasp pose as SE3. The grasp is in the object frame."""
        best = self.best_grasp
        return best.to_SE3() if best is not None else None

    @property
    def has_grasps(self) -> bool:
        """Check if grasp poses were loaded"""
        return len(self.grasp_poses) > 0


class ConfigLoader:
    """Load robot configuration from YAML"""

    @staticmethod
    def load(yaml_path: str | pathlib.Path) -> RobotConfig:
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        weights = config["weights"]
        return RobotConfig(
            W_xREG=weights["W_xREG"],
            W_uREG=weights["W_uREG"],
            W_gripper_pose=weights["W_gripper_pose"],
            W_gripper_pose_term=weights["W_gripper_pose_term"],
            W_limit=weights["W_limit"],
            safety_threshold=config["safety_threshold"],
            T=config["T"],
            dt=config["dt"],
            gripper_depth=config["gripper_depth"],
        )


if __name__ == "__main__":
    grasp_path = pathlib.Path(
        "/workspaces/object_following_ocp/ressources/filtered_grasps/0d0d1c59b0474d2ea92ce2e172c9f56a_filtered.yml"
    )
    object_traj_path = pathlib.Path(
        "/workspaces/object_following_ocp/ressources/json/bowl1.props-dinov2-ffa-22.gpt4_scaled.best_object.poses-dinov2-22-graph.smoothed-movavg.json"
    )
    scale_path = pathlib.Path(
        "/workspaces/object_following_ocp/ressources/grasps_scales.json"
    )

    # Old way (still works)
    print("=== Example 1: Explicit grasp path ===")
    dataloader = DataLoader(
        object_trajectory_path=object_traj_path,
        grasp_poses_SE3_path=grasp_path,
        scales_path=scale_path,
    )
    print(f"Loaded {len(dataloader.grasp_poses)} grasps")

    # New way: auto-load grasps
    print("\n=== Example 2: Auto-load grasps ===")
    dataloader2 = DataLoader(
        object_trajectory_path=object_traj_path,
        scales_path=scale_path,
        load_grasps=True,
    )
    print(f"Loaded {len(dataloader2.grasp_poses)} grasps")

    # Without grasps
    print("\n=== Example 3: No grasps ===")
    dataloader3 = DataLoader(
        object_trajectory_path=object_traj_path, scales_path=scale_path
    )
    print(f"Has grasps: {dataloader3.has_grasps}")

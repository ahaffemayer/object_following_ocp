import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pinocchio as pin

from object_following_ocp.depr.trajectory import Trajectory


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

    def to_se3(self) -> pin.SE3:
        """Convert to Pinocchio SE3"""
        return pin.SE3(self.R, self.t)


class JSONTrajectoryParser:
    """
    Parses object pose trajectories from JSON (BOP format).
    JSON → List[PoseData] → SE3(camera)
    Optional smoothing → SE3(camera)
    Explicit transform → SE3(robot/world)
    """

    def __init__(
        self,
        json_path: str | Path,
        mesh_base_dir: Optional[str | Path] = None,
        smooth_depth: bool = True,
        smooth_k: float = 2.0
    ):
        """
        Initialize JSON trajectory parser

        Args:
            json_path: Path to JSON file containing trajectory data
            mesh_base_dir: Base directory for meshes. If None, uses ../meshes relative to json_path
            smooth_depth: Whether to smooth the depth (z-axis) values
            smooth_k: Number of standard deviations for depth clipping
        """
        self.json_path = Path(json_path)
        self.smooth_depth = smooth_depth
        self.smooth_k = smooth_k

        with open(self.json_path, 'r') as f:
            self.data = json.load(f)

        # Set mesh base directory
        if mesh_base_dir is None:
            self.mesh_base_dir = self.json_path.parent.parent / "meshes"
        else:
            self.mesh_base_dir = Path(mesh_base_dir)

        # Parse and optionally smooth poses
        self._parse_camera_poses()

    # ---------- low-level utilities ----------
    @staticmethod
    def smooth_z(poses: List[pin.SE3], k: float = 2.0) -> List[pin.SE3]:
        """
        Smooth depth values by clipping outliers

        Args:
            poses: List of SE3 poses
            k: Number of standard deviations for clipping

        Returns:
            List of SE3 poses with smoothed depth
        """
        zs = np.array([p.translation[2] for p in poses])
        meanz = zs.mean()
        stdz = zs.std()
        out = []
        for p in poses:
            p = copy.deepcopy(p)
            p.translation[2] = np.clip(
                p.translation[2],
                meanz - k * stdz,
                meanz + k * stdz,
            )
            out.append(p)
        return out

    # ---------- parsing ----------
    def _parse_camera_poses(self):
        """Parse poses from JSON and optionally smooth depth"""
        # Store raw poses by object_id
        self._poses_camera_raw = {}
        self._poses_camera_smoothed = {}

        for pose_data in self.data['poses']:
            obj_id = pose_data['object_id']
            se3_pose = pin.SE3(
                np.array(pose_data['R']),
                np.array(pose_data['t'])
            )

            if obj_id not in self._poses_camera_raw:
                self._poses_camera_raw[obj_id] = []
            self._poses_camera_raw[obj_id].append(se3_pose)

        # Apply smoothing if requested
        for obj_id, poses in self._poses_camera_raw.items():
            if self.smooth_depth:
                self._poses_camera_smoothed[obj_id] = self.smooth_z(
                    poses, self.smooth_k)
            else:
                self._poses_camera_smoothed[obj_id] = poses

    # ---------- public API ----------
    def get_object_info(self, object_id: Optional[int] = None, texture_name: str = "material_0.png") -> ObjectInfo:
        """
        Extract object information including mesh paths

        Args:
            object_id: ID of the object. If None, uses the first available object.
            texture_name: Name of texture file (default: material_0.png)

        Returns:
            ObjectInfo with mesh and texture paths resolved
        """
        # If no object_id specified, use first available
        if object_id is None:
            object_id = int(next(iter(self.data['objects'].keys())))

        # Check if object exists
        if str(object_id) not in self.data['objects']:
            available = list(self.data['objects'].keys())
            raise ValueError(
                f"Object ID {object_id} not found. Available objects: {available}"
            )

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

    def get_trajectory(self, smoothed: bool = True) -> List[PoseData]:
        """
        Extract full trajectory as list of PoseData

        Args:
            smoothed: Whether to use smoothed poses (if smoothing was enabled)

        Returns:
            List of PoseData objects
        """
        trajectory = []
        for pose_dict in self.data['poses']:
            obj_id = pose_dict['object_id']

            # Get the appropriate pose (smoothed or raw)
            if smoothed and obj_id in self._poses_camera_smoothed:
                poses_list = self._poses_camera_smoothed[obj_id]
            else:
                poses_list = self._poses_camera_raw[obj_id]

            # Find the corresponding SE3 pose
            # Map im_id to index in poses list
            im_id = pose_dict['im_id']
            pose_idx = next(i for i, p in enumerate(self.data['poses'])
                            if p['im_id'] == im_id and p['object_id'] == obj_id)

            # Get indices for this object only
            obj_pose_idx = sum(1 for p in self.data['poses'][:pose_idx]
                               if p['object_id'] == obj_id)

            se3_pose = poses_list[obj_pose_idx]

            trajectory.append(PoseData(
                im_id=pose_dict['im_id'],
                object_id=pose_dict['object_id'],
                score=pose_dict['score'],
                R=se3_pose.rotation,
                t=se3_pose.translation,
                bbox_visib=pose_dict['bbox_visib'],
                time=pose_dict['time']
            ))
        return trajectory

    def get_poses_for_object(self, object_id: Optional[int] = None, smoothed: bool = True) -> List[PoseData]:
        """
        Get trajectory for specific object

        Args:
            object_id: Specific object ID. If None, uses first available object.
            smoothed: Whether to use smoothed poses (if smoothing was enabled)
        """
        if object_id is None:
            object_id = int(next(iter(self.data['objects'].keys())))

        trajectory = self.get_trajectory(smoothed=smoothed)
        poses = [pose for pose in trajectory if pose.object_id == object_id]

        if not poses:
            raise ValueError(f"No poses found for object ID {object_id}")

        return poses

    def get_camera_trajectory(self, object_id: Optional[int] = None) -> Trajectory:
        """
        Get object trajectory in CAMERA frame as Trajectory object
        (Uses smoothed poses if smoothing was enabled)

        Args:
            object_id: Specific object ID. If None, uses first available object.

        Returns:
            Trajectory object containing SE3 poses
        """
        if object_id is None:
            object_id = int(next(iter(self.data['objects'].keys())))

        # Get smoothed or raw poses directly
        if object_id in self._poses_camera_smoothed:
            se3_poses = self._poses_camera_smoothed[object_id]
        else:
            se3_poses = self._poses_camera_raw[object_id]

        return Trajectory(se3_poses)

    def to_robot_frame(self, camera_to_robot: pin.SE3, object_id: Optional[int] = None) -> Trajectory:
        """
        Converts trajectory to ROBOT / WORLD frame.

        Args:
            camera_to_robot: SE3(robot <- camera)
            object_id: Specific object ID. If None, uses first available object.

        Returns:
            Trajectory in robot frame
        """
        return self.get_camera_trajectory(object_id).transform(camera_to_robot)

    def get_metadata(self, object_id: Optional[int] = None):
        """Get metadata about the trajectory"""
        if object_id is None:
            object_id = int(next(iter(self.data['objects'].keys())))

        if str(object_id) not in self.data['objects']:
            available = list(self.data['objects'].keys())
            raise ValueError(
                f"Object ID {object_id} not found. Available objects: {available}"
            )

        obj_data = self.data['objects'][str(object_id)]

        # Count poses for this object
        num_poses = len(self._poses_camera_smoothed.get(object_id, []))

        return {
            "obj_id": object_id,
            "mesh_id": obj_data['mesh'],
            "scale": obj_data['scale'],
            "score": obj_data['score'],
            "num_frames": num_poses,
            "smoothed": self.smooth_depth,
            "smooth_k": self.smooth_k if self.smooth_depth else None,
        }

    def get_available_objects(self) -> Dict[int, str]:
        """Get dictionary of available object IDs and their mesh IDs"""
        return {
            int(obj_id): obj_data['mesh']
            for obj_id, obj_data in self.data['objects'].items()
        }

    def __len__(self) -> int:
        """Return number of poses in trajectory"""
        return len(self.data['poses'])

    def __repr__(self) -> str:
        """String representation"""
        n_objects = len(self.data['objects'])
        n_poses = len(self)
        smooth_str = f", smooth_depth={self.smooth_depth}, k={self.smooth_k}" if self.smooth_depth else ""
        return f"JSONTrajectoryParser({n_objects} objects, {n_poses} poses{smooth_str})"


class TrajectoryParser:
    """
    Parses object pose trajectories estimated in the CAMERA frame from CSV.
    CSV → SE3(camera)
    Optional smoothing → SE3(camera)
    Explicit transform → SE3(robot/world)
    """

    def __init__(self, csv_path: Path, smooth_depth=True, smooth_k=2.0):
        self.csv_path = Path(csv_path)
        self.smooth_depth = smooth_depth
        self.smooth_k = smooth_k
        self._load_csv()
        self._parse_camera_poses()

    # ---------- low-level utilities ----------
    @staticmethod
    def pd_row_to_se3(row: pd.Series) -> pin.SE3:
        return pin.SE3(
            np.fromstring(row["R"], dtype=float, sep=" ").reshape(3, 3),
            np.fromstring(row["t"], dtype=float, sep=" "),
        )

    @staticmethod
    def smooth_z(poses, k=2.0):
        zs = np.array([p.translation[2] for p in poses])
        meanz = zs.mean()
        stdz = zs.std()
        out = []
        for p in poses:
            p = copy.deepcopy(p)
            p.translation[2] = np.clip(
                p.translation[2],
                meanz - k * stdz,
                meanz + k * stdz,
            )
            out.append(p)
        return out

    # ---------- parsing ----------
    def _load_csv(self):
        if not self.csv_path.exists():
            raise FileNotFoundError(self.csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.obj_id = self.df.iloc[0]["obj_id"]
        self.scale = float(self.df["scale"].mean())

    def _parse_camera_poses(self):
        poses = [
            self.pd_row_to_se3(row)
            for _, row in self.df.iterrows()
        ]
        if self.smooth_depth:
            poses = self.smooth_z(poses, self.smooth_k)
        self._poses_camera = poses

    # ---------- public API ----------
    def get_camera_trajectory(self) -> Trajectory:
        """Object trajectory in CAMERA frame"""
        return Trajectory(self._poses_camera)

    def to_robot_frame(self, camera_to_robot: pin.SE3) -> Trajectory:
        """
        Converts trajectory to ROBOT / WORLD frame.
        camera_to_robot: SE3(robot <- camera)
        """
        return Trajectory(self._poses_camera).transform(camera_to_robot)

    def get_metadata(self):
        return {
            "obj_id": self.obj_id,
            "scale": self.scale,
            "num_frames": len(self._poses_camera),
        }


if __name__ == "__main__":
    # Example usage for JSON parser with smoothing
    json_path = Path(
        "/workspaces/object_following_ocp/ressources/json/jug.props-dinov2-ffa-22.gpt4_scaled.best_object.poses-dinov2-22-graph.smoothed-movavg.json")

    # With smoothing (default)
    json_parser = JSONTrajectoryParser(
        json_path, smooth_depth=True, smooth_k=2.0)
    print(f"Parser: {json_parser}")

    print("\nAvailable objects:", json_parser.get_available_objects())

    # Get object info
    obj_info = json_parser.get_object_info()
    print(f"\nObject info: {obj_info}")

    # Get metadata
    json_metadata = json_parser.get_metadata()
    print(f"\nJSON Parser Metadata: {json_metadata}")

    # Get trajectory (smoothed)
    json_traj = json_parser.get_camera_trajectory()
    print(f"Trajectory length: {len(json_traj)}")

    # Compare smoothed vs unsmoothed
    print("\nComparing smoothed vs raw:")
    json_parser_raw = JSONTrajectoryParser(json_path, smooth_depth=False)
    traj_raw = json_parser_raw.get_camera_trajectory()
    traj_smooth = json_parser.get_camera_trajectory()

    print(f"Raw depth values: {[p.translation[2] for p in traj_raw[:5]]}")
    print(
        f"Smoothed depth values: {[p.translation[2] for p in traj_smooth[:5]]}")

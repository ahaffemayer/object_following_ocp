from pathlib import Path
import copy
import numpy as np
import pandas as pd
import pinocchio as pin

from trajectory import Trajectory

class TrajectoryParser:
    """
    Parses object pose trajectories estimated in the CAMERA frame.

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
    # Example usage
    csv_path = Path("/home/arthur/Desktop/Projects/PAMI/object_following_ocp/ressources/exp_wildpose/howto100m_poses/howto100m_0ozOhWb3l_E_0-smoothed.csv")
    parser = TrajectoryParser(csv_path, smooth_depth=True, smooth_k=2.0)
    traj_camera = parser.get_camera_trajectory()

    T_camera_robot = pin.SE3(np.eye(3), np.array([0.5, 0.0, 1.0]))  # Example transform
    traj_robot = parser.to_robot_frame(T_camera_robot)

    metadata = parser.get_metadata()
    print(metadata)
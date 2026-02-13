import numpy as np
import pinocchio as pin


class TrajectorySE3:
    def __init__(self, poses: list[pin.SE3]) -> None:
        self.poses = poses
        self.T = len(poses)

    def __getitem__(self, index: int) -> pin.SE3:
        return self.poses[index]

    def __len__(self) -> int:
        return self.T

    def __mul__(self, T: pin.SE3) -> "TrajectorySE3":
        return TrajectorySE3([p * T for p in self.poses])

    def __rmul__(self, T: pin.SE3) -> "TrajectorySE3":
        return TrajectorySE3([T * p for p in self.poses])


class TrajectoryInConfigurationSpace:
    def __init__(self, configurations: list[np.ndarray]) -> None:
        self.configurations = configurations
        self.T = len(configurations)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.configurations[index]

    def __len__(self) -> int:
        return self.T

    def get_EE_poses(self, rmodel: pin.Model) -> list[pin.SE3]:
        poses = []
        rdata = rmodel.createData()
        for q in self.configurations:
            pin.forwardKinematics(rmodel, rdata, q)
            pin.updateFramePlacements(rmodel, rdata)
            ee_frame_id = rmodel.getFrameId("panda_hand_tcp")
            ee_pose = rmodel.frames[ee_frame_id].placement
            poses.append(ee_pose)
        return poses

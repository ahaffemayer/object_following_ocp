import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml

# from object_following_ocp.ocp import OCP
from ocp_test import OCP
from robomeshcat import Object, Robot, Scene

from object_following_ocp.dataclass import ConfigLoader
from object_following_ocp.grasp_generator import GraspGenerator, configure_logging
from object_following_ocp.robot_loader import (
    load_reduced_panda,
    load_talos_arm,
    load_ur5,
    self_collision_pairs,
)
from object_following_ocp.trajectory import (
    Trajectory,
    TrajectoryEvaluator,
    TrajectoryInConfigurationSpace,
)
from object_following_ocp.trajectory_parser import JSONTrajectoryParser


def list_to_se3(poses):
    se3_list = []
    for p in poses:
        R = np.array(p["rotation"])
        t = np.array(p["translation"]) - np.array([0, 0, 0.5])
        se3_list.append(pin.SE3(R, t))
    return se3_list


rot = pin.SE3(pin.utils.rpyToMatrix(0, 90, 140), np.array([0.00, 0.0, 0.05]))

new_robot = "ur"
trajs_path = Path(
    "/workspaces/object_following_ocp/saved_trajectories/saved_trajs_panda_20260205_160605.json"
)
# Load the trajectories

with open(trajs_path, "r") as f:
    trajs = json.load(f)

# Load robot configuration from YAML
robot_config = ConfigLoader.load("/workspaces/object_following_ocp/example/robot_config.yml"
                                 )

# Load trajectory data from JSON
json_path = Path(
    "/workspaces/object_following_ocp/ressources/json/jug.props-dinov2-ffa-22.gpt4_scaled.best_object.poses-dinov2-22-graph.smoothed-movavg.json")

# Parser automatically resolves mesh paths and applies smoothing
parser = JSONTrajectoryParser(
    json_path, smooth_depth=True, smooth_k=2.0)

object_info = parser.get_object_info(texture_name="material_0.png")


rmodel_panda, cmodel_panda, vmodel_panda = load_reduced_panda()

for cp in self_collision_pairs:
    if cmodel_panda.existGeometryName(cp[0]) and cmodel_panda.existGeometryName(cp[1]):
        cmodel_panda.addCollisionPair(
            pin.CollisionPair(
                cmodel_panda.getGeometryId(cp[0]),
                cmodel_panda.getGeometryId(cp[1]),
            )
        )

rdata = rmodel_panda.createData()
cdata = cmodel_panda.createData()
vdata = vmodel_panda.createData()

robot = Robot(
    pinocchio_model=rmodel_panda,
    pinocchio_data=rdata,
    pinocchio_geometry_model=vmodel_panda,
    pinocchio_geometry_data=vdata,
)

# -----------------------------
# Scene
# -----------------------------
scene = Scene()
# scene.add_robot(robot)

# -----------------------------
# Add object to scene (using paths from parser)
# -----------------------------
o = Object.create_mesh(
    path_to_mesh=object_info.mesh_path,
    name="robot/movable_obj",
    texture="/workspaces/object_following_ocp/ressources/meshes/b314794073c44ede838cf61627b5a3b7/material_0.png",
    scale=object_info.scale,
    color=[0.8, 0.8, 0.8],
)
scene.add_object(o)

# Visualize the trajs
# for i, traj in enumerate(trajs):
#     traj_joint = traj['joint_trajectory']
#     target_poses = traj["object_trajectory_se3"]
#     se3_poses = list_to_se3(target_poses)

#     for xs, pose in zip(traj_joint, se3_poses):
#         robot[:] = xs
#         o.pose = pose.homogeneous
#         input()

# Creating the new robot model
if new_robot == "ur":
    rmodel, cmodel, vmodel = load_ur5()
    ee_frame = "tool0"
    # ee_frame = "ee_link"
elif new_robot == "talos_arm":
    rmodel, cmodel, vmodel = load_talos_arm()
    ee_frame = "gripper_left_fingertip_1_link"

for f in rmodel.frames:
    print(f.name)

rdata = rmodel.createData()
cdata = cmodel.createData()
vdata = vmodel.createData()

target_poses = trajs[1]["object_trajectory_se3"]
se3_poses = list_to_se3(target_poses)
# print(se3_poses)
x0 = np.concatenate((np.array([-5.00220099, 4.69746134, -4.66262958,
                    0.56785259,  2.52217891, -3.55780986]), np.zeros(rmodel.nv)))

# Setting up the OCP

weights = {
    "W_xREG": 1e-3,
    "W_uREG": 1e-3,
    "W_gripper_pose": 100.0,
    "W_gripper_pose_term": 100,
    "W_limit": 0,
}

new_ocp_creator = OCP(
    rmodel,
    se3_poses,
    x0,
    weights=weights,
    joint_limits=True,
    joint_limits_constraint=False,
    T=len(se3_poses),
    ee_frame=ee_frame,
    with_callbacks=True
)
new_ocp = new_ocp_creator.create_OCP()
X_init = [x0] * new_ocp_creator._T
U_init = new_ocp.problem.quasiStatic(X_init[:-1])

robot = Robot(
    pinocchio_model=rmodel,
    pinocchio_data=rdata,
    pinocchio_geometry_model=vmodel,
    pinocchio_geometry_data=vdata,
)
scene.add_robot(robot)
print("solving")
new_ocp.solve(X_init, U_init, 100)
# Displaying the new trajectory on the new robot
input()
for x, pose in zip(new_ocp.xs, se3_poses):
    pin.framesForwardKinematics(rmodel, rdata, x[:rmodel.nq])
    ee_pose = rdata.oMf[rmodel.getFrameId(ee_frame)]
    o.pose = (ee_pose*rot).homogeneous
    # o.pose = (pose * rot).homogeneous
    robot[:] = x[:rmodel.nq]
    print(x[:rmodel.nq])
    time.sleep(0.05)
    # input()

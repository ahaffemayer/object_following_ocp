import numpy as np
from pathlib import Path
import pinocchio as pin
from robomeshcat import Scene, Object, Robot

from grasp_generator import GraspGenerator
from ocp import OCP
from parser_config import load_config
from robot_loader import load_reduced_panda, self_collision_pairs
from trajectory_parser import TrajectoryParser
from trajectory import Trajectory, TrajectoryInConfigurationSpace, TrajectoryEvaluator


goal = pin.SE3(np.array([[ 0.93542689,  0.12843582,  0.3293642 , -0.02318742],[ 0.14179023, -0.98975513, -0.01674251, -0.16605873],[ 0.32383957,  0.06236202, -0.94405451,  0.66508423],[ 0. ,  0. ,  0. ,  1. ]]))
print("Goal SE3:\n", goal)
cfg = load_config(Path(__file__).parent / "config.yaml")

# weights = cfg["weights"]
# print(weights)
mesh_dir = Path(cfg["mesh"]["path"])
obj_file = mesh_dir / cfg["mesh"]["obj_file"]
texture_file = mesh_dir / cfg["mesh"]["texture_file"]
scale = cfg["mesh"]["scale"]
color = cfg["mesh"]["color"]

weights = {
    "W_xREG": 0,
    "W_uREG": 0,
    "W_gripper_pose": 10,
    "W_gripper_pose_term": 10000,
    "W_limit": 100,
}

safety_threshold = cfg["safety_threshold"]
dt = cfg["dt"]


rmodel, cmodel, vmodel = load_reduced_panda()
for cp in self_collision_pairs:
    if cmodel.existGeometryName(cp[0]) and cmodel.existGeometryName(cp[1]):
        cmodel.addCollisionPair(
            pin.CollisionPair(
                cmodel.getGeometryId(cp[0]),
                cmodel.getGeometryId(cp[1]),
            )
        )
rdata = rmodel.createData()

cdata = cmodel.createData()
vdata = vmodel.createData()
r = Robot(
    pinocchio_model=rmodel,
    pinocchio_data=rdata,
    pinocchio_geometry_model=vmodel,
    pinocchio_geometry_data=vdata,
)

scene = Scene()
scene.add_robot(r)


o = Object.create_mesh(
    path_to_mesh=obj_file,
    name="robot/movable_obj",
    texture=texture_file,
    scale=scale,
    color=color,
)
scene.add_object(o)
weights = cfg["weights"]

q0_0 = pin.randomConfiguration(rmodel)
print("Initial configuration:", q0_0)

### Generate the different grasps configurations around the object pose


q0_1 = pin.randomConfiguration(rmodel)
print("Initial configuration for OCP:", q0_1)
OCP_creator = OCP(
    rmodel,
    cmodel,
    goal,
    weights=weights,
    x0=np.concatenate((q0_1, np.zeros(rmodel.nv))),
    joint_limits=True,
    joint_limits_constraint=True,
    T = 3
)

ocp = OCP_creator.create_OCP()
X_init = [np.concatenate((q0_1, np.zeros(rmodel.nv)))] * (OCP_creator._T)
U_init = ocp.problem.quasiStatic(X_init[:-1])

scene.add_object(Object.create_sphere(radius=0.05, name=f"goal", color=[1, 1, 1]))
scene[f"goal"].pos[:] = goal.translation

# Solve the OCP
import time

start_time = time.time()
ocp.solve(X_init, U_init)
end_time = time.time()
print(f"OCP solved in {end_time - start_time:.2f} seconds.")
for x in ocp.xs:
    q = x[:rmodel.nq]
    r[:] = q
    input()
    
pin.framesForwardKinematics(rmodel, rdata, q)  
ee_pose = rdata.oMf[rmodel.getFrameId("panda_hand_tcp")]
print("End-effector pose after OCP:\n", ee_pose)
print("Goal pose:\n", goal)
print("Position error:", np.linalg.norm(ee_pose.translation - goal.translation))
print("Orientation error:", np.linalg.norm(ee_pose.rotation - goal.rotation))


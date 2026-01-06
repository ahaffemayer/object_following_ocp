import meshcat
import numpy as np
from pathlib import Path
import pinocchio as pin
import sys
from typing import Any
import numpy as np
import colmpc as col
import crocoddyl
import pinocchio as pin
import mim_solvers
from pinocchio import visualize
from robomeshcat import Scene, Object, Robot

from robot_loader import load_reduced_panda, self_collision_pairs
from ocp import OCP
from parser_config import load_config
from trajectory import se3_sinusoid_trajectory
from trajectory_parser import TrajectoryParser

if __name__ == "__main__":
    
    cfg = load_config(Path(__file__).parent / "config.yaml")

    weights = cfg["weights"]
    mesh_dir = Path(cfg["mesh"]["path"])
    obj_file = mesh_dir / cfg["mesh"]["obj_file"]
    texture_file = mesh_dir / cfg["mesh"]["texture_file"]
    scale = cfg["mesh"]["scale"]
    color = cfg["mesh"]["color"]

    safety_threshold = cfg["safety_threshold"]
    # T = cfg["T"]
    dt = cfg["dt"]

    T_camera = pin.SE3(np.eye(3), np.array([0.5, 0.0, 0.0]))
    
    csv_path = Path("/home/arthur/Desktop/Projects/PAMI/object_following_ocp/ressources/exp_wildpose/howto100m_poses/howto100m_9Mh7jlESPvs_1-smoothed.csv")
    parser = TrajectoryParser(csv_path, smooth_depth=True, smooth_k=2.0)
    traj_camera = parser.get_camera_trajectory()

    T_camera_robot = pin.SE3(np.eye(3), np.array([0.0, 0.0, 0.0]))  # Example transform
    traj_robot = parser.to_robot_frame(T_camera_robot)
        
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
    start_1 = traj_robot[0]
    o.pose = ( start_1
    ).homogeneous
    
    OCP_creator_1 = OCP(
        rmodel,
        cmodel,
        start_1,
        x0=np.concatenate((q0_0, np.zeros(rmodel.nv))),
        joint_limits=True,
        joint_limits_constraint=False,
        with_callbacks=True,
        weights=weights,
        safety_threshold=0.02,
        T=len(traj_robot),
        dt=dt,
    )
    
    ocp_1 = OCP_creator_1.create_OCP()
    X_init = [np.concatenate((q0_0, np.zeros(rmodel.nv)))] * (OCP_creator_1._T)
    U_init = ocp_1.problem.quasiStatic(X_init[:-1])
    ocp_1.solve(X_init, U_init)
    

    q0_1 = ocp_1.xs[-1][:rmodel.nq]
    
    OCP_creator = OCP(
        rmodel,
        cmodel,
        traj_robot,
        x0=np.concatenate((q0_1, np.zeros(rmodel.nv))),
        joint_limits=True,
        joint_limits_constraint=False,
        with_callbacks=True,
        weights=weights,
        safety_threshold=0.02,
        T=len(traj_robot),
        dt=dt,
    )
    ocp = OCP_creator.create_OCP()
    X_init = [np.concatenate((q0_1, np.zeros(rmodel.nv)))] * (OCP_creator._T)
    U_init = ocp.problem.quasiStatic(X_init[:-1])
    
    scene.add_object(Object.create_sphere(radius=0.01, name=f'camera', color=[1, 1, 0]))
    scene["camera"].pos[:] = T_camera.translation
    
    for i, xs in enumerate(ocp_1.xs):
        pin.framesForwardKinematics(rmodel, rdata, xs[:rmodel.nq])
        ee_pose = rdata.oMf[rmodel.getFrameId("panda_hand_tcp")]
        r[:] = xs[:rmodel.nq]
        input("Press Enter to continue to the next state...")
    # Solve the OCP
    import time
    start_time = time.time()
    ocp.solve(X_init, U_init)
    end_time = time.time()
    print(f"OCP solved in {end_time - start_time} seconds")
    for i, (xs, target) in enumerate(zip(ocp.xs, traj_robot)):
        scene.add_object(Object.create_sphere(radius=0.01, name=f'target_{i}', color=[1, 0, 0]))
        scene[f'target_{i}'].pos[:] = target.translation
        pin.framesForwardKinematics(rmodel, rdata, xs[:rmodel.nq])
        ee_pose = rdata.oMf[rmodel.getFrameId("panda_hand_tcp")]
        r[:] = xs[:rmodel.nq]
        o.pose = ( ee_pose
            ).homogeneous
        scene.add_object(Object.create_sphere(radius=0.01, name=f'ee_pose{i}', color=[0, 1, 0]))
        scene[f'ee_pose{i}'].pos[:] = ee_pose.translation
        input("Press Enter to continue to the next state...")
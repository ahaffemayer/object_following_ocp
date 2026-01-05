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


if __name__ == "__main__":
    
    cfg = load_config(Path(__file__).parent / "config.yaml")

    weights = cfg["weights"]
    mesh_dir = Path(cfg["mesh"]["path"])
    obj_file = mesh_dir / cfg["mesh"]["obj_file"]
    texture_file = mesh_dir / cfg["mesh"]["texture_file"]
    scale = cfg["mesh"]["scale"]
    color = cfg["mesh"]["color"]

    safety_threshold = cfg["safety_threshold"]
    T = cfg["T"]
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
    
        
    q0_1 = [-1.93027339 , 1.21002708, -0.27500036, -1.42079896,  0.08216306,  2.60651474,
  0.16107337]

    start_1 = pin.XYZQUATToSE3([-0.40169054, -0.5526958,  -0.11655478,  0.73505497, -0.66944448, -0.10450928,
  0.02482115]
    )
    end_1 = pin.XYZQUATToSE3([0.5, -0.2, 0.5, 0.7071, 0.0, 0.7071, 0.0])
    # start = pin.SE3(np.eye(3), np.array([0.5, 0.0, 0.5]))
    # end = pin.SE3(np.eye(3), np.array([0.5, 0.2, 0.5]))
    TARGET_POSES = se3_sinusoid_trajectory(start_1, end_1, T=50)
    print(TARGET_POSES)
    simple_ocp_creator = OCP(
        rmodel,
        cmodel,
        start_1,
        x0=np.concatenate((q0_0, np.zeros(rmodel.nv))),
        joint_limits=True,
        penalisation=False,
        constraint=False,
        with_callbacks=True,
        weights=weights,
        safety_threshold=0.02,
        T=50,
        dt=0.02,
    )
    
    # simple_ocp = simple_ocp_creator.create_OCP()
    # simple_ocp.solve()
    # xs_init = simple_ocp.xs    

    
    # while True:
    #     q = pin.randomConfiguration(rmodel)
    #     pin.framesForwardKinematics(rmodel, rdata, q)
    #     ee_pos = rdata.oMf[rmodel.getFrameId("panda_hand_tcp")]
    #     print("EE pos:", pin.SE3ToXYZQUAT(ee_pos))
    #     print("q:", q)
    #     viz.display(q)
    #     input()
    
    
    OCP_creator = OCP(
        rmodel,
        cmodel,
        TARGET_POSES,
        x0=np.concatenate((q0_1, np.zeros(rmodel.nv))),
        joint_limits=True,
        penalisation=False,
        constraint=False,
        with_callbacks=True,
        weights=weights,
        safety_threshold=0.02,
        T=50,
        dt=0.02,
    )
    ocp = OCP_creator.create_OCP()
    X_init = [np.concatenate((q0_1, np.zeros(rmodel.nv)))] * (OCP_creator._T)
    U_init = ocp.problem.quasiStatic(X_init[:-1])
    
    for i, target in enumerate(TARGET_POSES):
        scene.add_object(Object.create_sphere(radius=0.01, name=f'target_{i}', color=[1, 0, 0]))
        scene[f'target_{i}'].pos[:] = target.translation
    # Solve the OCP
    import time
    start_time = time.time()
    ocp.solve(X_init, U_init)
    end_time = time.time()
    print(f"OCP solved in {end_time - start_time} seconds")
    # Visualize the solution
    # for i, xs in enumerate(xs_init):
    #     r[:] = xs[:rmodel.nq]
    #     input()
    
    for i, xs in enumerate(ocp.xs):
        pin.framesForwardKinematics(rmodel, rdata, xs[:rmodel.nq])
        ee_pose = rdata.oMf[rmodel.getFrameId("panda_hand_tcp")]
        r[:] = xs[:rmodel.nq]
        o.pose = ( ee_pose
            ).homogeneous
        scene.add_object(Object.create_sphere(radius=0.01, name=f'ee_pose{i}', color=[0, 1, 0]))
        scene[f'ee_pose{i}'].pos[:] = ee_pose.translation
        input("Press Enter to continue to the next state...")
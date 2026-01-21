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


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------
    cfg = load_config(Path(__file__).parent / "config.yaml")

    weights = cfg["weights"]
    mesh_dir = Path(cfg["mesh"]["path"])
    obj_file = mesh_dir / cfg["mesh"]["obj_file"]
    texture_file = mesh_dir / cfg["mesh"]["texture_file"]
    scale = cfg["mesh"]["scale"]
    color = cfg["mesh"]["color"]

    dt = cfg["dt"]
    safety_threshold = cfg["safety_threshold"]

    csv_path = Path(
        "/home/arthur/Desktop/Projects/PAMI/object_following_ocp/ressources/"
        "video_data/d02/jug/jug-obj_2-tracked-4.csv"
    )

    # --------------------------------------------------------
    # Load object trajectory (camera -> robot/world)
    # --------------------------------------------------------
    parser = TrajectoryParser(csv_path, smooth_depth=True, smooth_k=2.0)
    traj_camera = parser.get_camera_trajectory()

    T_camera_world = pin.SE3(
        np.eye(3),
        np.array([-0.5, 0.0, -0.75])
    )

    traj_world_object = parser.to_robot_frame(T_camera_world)

    # --------------------------------------------------------
    # Define TCP offset in object frame (HANDLE)
    # --------------------------------------------------------
    T_object_tcp = pin.XYZQUATToSE3(
        np.array([
            0.0229, -0.260, 0.0230,
            -0.465, -0.523, -0.530, 0.477
        ])
    ).inverse()

    # Frame convention correction (GraspGen -> Panda TCP)
    T_correction = pin.SE3(
        pin.utils.rpyToMatrix(0.0, 0.0, np.pi / 2),
        np.zeros(3)
    )

    T_object_tcp = T_object_tcp * T_correction

    # --------------------------------------------------------
    # Load robot
    # --------------------------------------------------------
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

    robot = Robot(
        pinocchio_model=rmodel,
        pinocchio_data=rdata,
        pinocchio_geometry_model=vmodel,
        pinocchio_geometry_data=vdata,
    )

    # --------------------------------------------------------
    # Visualization scene
    # --------------------------------------------------------
    scene = Scene()
    scene.add_robot(robot)

    obj = Object.create_mesh(
        path_to_mesh=obj_file,
        name="robot/movable_obj",
        texture=texture_file,
        scale=scale,
        color=color,
    )
    scene.add_object(obj)

    # --------------------------------------------------------
    # Initial robot configuration
    # --------------------------------------------------------
    q0 = pin.randomConfiguration(rmodel)
    robot[:] = q0

    # --------------------------------------------------------
    # Initial object pose
    # --------------------------------------------------------
    T_world_object_0 = traj_world_object[0]
    obj.pose = T_world_object_0.homogeneous

    T_world_tcp_0 = T_world_object_0 * T_object_tcp

    scene.add_object(
        Object.create_sphere(
            radius=0.01,
            name="tcp_target_initial",
            color=[1.0, 1.0, 1.0],
        )
    )
    scene["tcp_target_initial"].pos[:] = T_world_tcp_0.translation

    # --------------------------------------------------------
    # Generate grasp configurations
    # --------------------------------------------------------
    grasp_generator = GraspGenerator(
        obj_pose=T_world_tcp_0,
        grasp_configurations_number=3,
    )

    grasp_configurations = grasp_generator.generate_grasps_configurations()
    print(f"Generated {len(grasp_configurations)} grasp configurations")

    for i, qg in enumerate(grasp_configurations):
        print(f"Grasp {i}: {qg}")
        robot[:] = qg
        input("Press Enter for next grasp...")

    q_grasp = grasp_configurations[0]

    # --------------------------------------------------------
    # Build TCP trajectory in world frame
    # --------------------------------------------------------
    traj_world_tcp = [
        T_world_object * T_object_tcp
        for T_world_object in traj_world_object
    ]

    # --------------------------------------------------------
    # Create and solve OCP
    # --------------------------------------------------------
    OCP_creator = OCP(
        rmodel,
        cmodel,
        traj_world_tcp,
        x0=np.concatenate((q_grasp, np.zeros(rmodel.nv))),
        joint_limits=True,
        joint_limits_constraint=False,
        with_callbacks=True,
        weights=weights,
        safety_threshold=safety_threshold,
        T=len(traj_world_tcp),
        dt=dt,
    )

    ocp = OCP_creator.create_OCP()

    X_init = [
        np.concatenate((q_grasp, np.zeros(rmodel.nv)))
        for _ in range(OCP_creator._T)
    ]
    U_init = ocp.problem.quasiStatic(X_init[:-1])

    # --------------------------------------------------------
    # Camera visualization
    # --------------------------------------------------------
    scene.add_object(
        Object.create_sphere(radius=0.01, name="camera", color=[1.0, 1.0, 0.0])
    )
    scene["camera"].pos[:] = T_camera_world.translation

    # --------------------------------------------------------
    # Solve OCP
    # --------------------------------------------------------
    import time

    start = time.time()
    ocp.solve(X_init, U_init)
    print(f"OCP solved in {time.time() - start:.2f} s")

    # --------------------------------------------------------
    # Trajectory evaluation
    # --------------------------------------------------------
    traj_q = TrajectoryInConfigurationSpace(
        [ocp.xs[k][: rmodel.nq] for k in range(OCP_creator._T)]
    )
    traj_target = Trajectory(traj_world_tcp)

    evaluator = TrajectoryEvaluator(traj_target, traj_q, rmodel)
    print("Position error:", evaluator.evaluate_position_error())

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------
    for i, (xs, target) in enumerate(zip(ocp.xs, traj_world_tcp)):
        scene.add_object(
            Object.create_sphere(radius=0.01, name=f"target_{i}", color=[1, 0, 0])
        )
        scene[f"target_{i}"].pos[:] = target.translation

        pin.framesForwardKinematics(rmodel, rdata, xs[: rmodel.nq])
        ee_id = rmodel.getFrameId("panda_hand_tcp")
        ee_pose = rdata.oMf[ee_id]

        robot[:] = xs[: rmodel.nq]
        obj.pose = traj_world_object[i].homogeneous

        scene.add_object(
            Object.create_sphere(radius=0.01, name=f"ee_{i}", color=[0, 1, 0])
        )
        scene[f"ee_{i}"].pos[:] = ee_pose.translation

        input("Press Enter to continue...")

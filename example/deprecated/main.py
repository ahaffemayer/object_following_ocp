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

    csv_path = Path(
        "/home/arthur/Desktop/Projects/PAMI/object_following_ocp/ressources/"
        "video_data/d02/jug/jug-obj_2-tracked-4.csv"
    )

    handle = pin.XYZQUATToSE3(
        np.array([0.0229, -0.260, 0.0230, -0.465, -0.523,-0.530, 0.477])
    ).inverse()
    print("Handle pose:\n", handle)
    parser = TrajectoryParser(csv_path, smooth_depth=True, smooth_k=2.0)
    traj_camera = parser.get_camera_trajectory()
    # print(traj_camera.poses)
    T_camera_robot = pin.SE3(
        np.eye(3), np.array([-0.5, 0.0, -0.75])
    )  # Example transform of the camera in robot frame (world frame)
    traj_robot = parser.to_robot_frame(T_camera_robot)
    print(traj_robot.poses)
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
    o.pose = (start_1).homogeneous
    print("Object pose:\n", start_1)
    start_1_with_handle = start_1 * handle
    print("Object pose with handle:\n", start_1_with_handle)
    scene.add_object(
        Object.create_sphere(radius=0.01, name=f"first_gripper_pose", color=[1, 1, 1])
    )
    scene["first_gripper_pose"].pos[:] = start_1_with_handle.translation
    ### Generate the different grasps configurations around the object pose
    grasp_generator = GraspGenerator(
        obj_pose=start_1_with_handle,
        grasp_configurations_number=3,
    )
    grasp_configurations = grasp_generator.generate_grasps_configurations()
    print(f"Generated {len(grasp_configurations)} grasp configurations.")

    for i, qg in enumerate(grasp_configurations):
        print(f"Grasp configuration {i}: {qg}")
        r[:] = qg
        input("Press Enter to continue to the next grasp configuration...")

    q0_1 = grasp_configurations[0]
    traj_robot_with_handle = []
    for pose in traj_robot:
        traj_robot_with_handle.append(pose * handle)

    ### Create and solve the OCP
    OCP_creator = OCP(
        rmodel,
        cmodel,
        traj_robot_with_handle,
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

    scene.add_object(Object.create_sphere(radius=0.01, name=f"camera", color=[1, 1, 0]))
    scene["camera"].pos[:] = T_camera_robot.translation

    # Solve the OCP
    import time

    start_time = time.time()
    ocp.solve(X_init, U_init)
    end_time = time.time()
    print(f"OCP solved in {end_time - start_time} seconds")

    ### Trajectory evaluation
    traj_in_configuration_space = TrajectoryInConfigurationSpace(
        [ocp.xs[k][: rmodel.nq] for k in range(OCP_creator._T)]
    )
    traj = Trajectory(traj_robot)
    evaluator = TrajectoryEvaluator(traj, traj_in_configuration_space, rmodel)
    position_error = evaluator.evaluate_position_error()
    print("Position error:", position_error)

    ### Visualize the result
    for i, (xs, target) in enumerate(zip(ocp.xs, traj_robot)):
        scene.add_object(
            Object.create_sphere(radius=0.01, name=f"target_{i}", color=[1, 0, 0])
        )
        scene[f"target_{i}"].pos[:] = target.translation
        pin.framesForwardKinematics(rmodel, rdata, xs[: rmodel.nq])
        ee_pose = rdata.oMf[rmodel.getFrameId("panda_hand_tcp")]
        r[:] = xs[: rmodel.nq]
        o.pose = (target).homogeneous
        scene.add_object(
            Object.create_sphere(radius=0.01, name=f"ee_pose{i}", color=[0, 1, 0])
        )
        scene[f"ee_pose{i}"].pos[:] = ee_pose.translation
        input("Press Enter to continue to the next state...")

import numpy as np
from pathlib import Path
import pinocchio as pin
from robomeshcat import Scene, Object, Robot
import time

from grasp_generator import GraspGenerator
from ocp import OCP
from parser_config import load_config
from robot_loader import load_reduced_panda, self_collision_pairs
from trajectory_parser import TrajectoryParser
from trajectory import Trajectory, TrajectoryInConfigurationSpace, TrajectoryEvaluator


vis_candidates = False  # set True to visualize all candidate trajectories
vis_best_case = False  # set True to visualize only the best case found

# -----------------------------
# Utilities
# -----------------------------

CAMERA_YAW_DEG = [0, 45, 90, 135, 180]
CAMERA_YAW_RAD = [np.deg2rad(a) for a in CAMERA_YAW_DEG]


def sample_camera_translations_grid(
    nx=3,
    ny=3,
    nz=3,
    xlim=(-0.5, 0.1),
    ylim=(-0.5, 0.5),
    zlim=(-0.5, 0.5),
):
    xs = np.linspace(xlim[0], xlim[1], nx)
    ys = np.linspace(ylim[0], ylim[1], ny)
    zs = np.linspace(zlim[0], zlim[1], nz)

    grid = np.array(
        [(x, y, z) for x in xs for y in ys for z in zs],
        dtype=np.float64,
    )

    return grid


def se3_from_translation_yaw(translation, yaw):
    R = pin.utils.rpyToMatrix(0.0, 0.0, yaw)
    return pin.SE3(R, translation)


def camera_to_robot_trajectory(parser, camera_translation, yaw):
    T_camera_robot = se3_from_translation_yaw(camera_translation, yaw)
    return parser.to_robot_frame(T_camera_robot)


def is_grasp_valid(q, rmodel, cmodel, rdata, cdata):
    if np.any(q < rmodel.lowerPositionLimit) or np.any(q > rmodel.upperPositionLimit):
        return False

    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)
    has_collision = pin.computeCollisions(cmodel, cdata, False)
    return not has_collision


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":

    tstart = time.time()

    cfg = load_config(Path(__file__).parent / "config.yaml")
    weights = cfg["weights"]
    dt = cfg["dt"]
    safety_threshold = cfg["safety_threshold"]

    mesh_dir = Path(cfg["mesh"]["path"])
    obj_file = mesh_dir / cfg["mesh"]["obj_file"]
    texture_file = mesh_dir / cfg["mesh"]["texture_file"]
    scale = cfg["mesh"]["scale"]
    color = cfg["mesh"]["color"]

    csv_path = Path(
        "/home/arthur/Desktop/Projects/PAMI/object_following_ocp/ressources/"
        "video_data/d02/jug/jug-obj_2-tracked-4.csv"
    )

    # -----------------------------
    # Load trajectory parser
    # -----------------------------

    parser = TrajectoryParser(csv_path, smooth_depth=True, smooth_k=2.0)

    # -----------------------------
    # Load robot and collision model
    # -----------------------------

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

    # -----------------------------
    # Scene
    # -----------------------------

    scene = Scene()
    scene.add_robot(robot)
    o = Object.create_mesh(
        path_to_mesh=obj_file,
        name="robot/movable_obj",
        texture=texture_file,
        scale=scale,
        color=color,
    )
    scene.add_object(o)
    # -----------------------------
    # Stage 1, sample camera poses
    # -----------------------------

    camera_translations = sample_camera_translations_grid(
        nx=3,
        ny=3,
        nz=3,
        xlim=(-0.5, 0.1),
        ylim=(-0.4, 0.4),
        zlim=(-1.0, -0.6),
    )
    all_trajs_robot = []
    camera_configs = []  # (translation, yaw)

    for t in camera_translations:
        for yaw in CAMERA_YAW_RAD:
            traj_robot = camera_to_robot_trajectory(parser, t, yaw)
            all_trajs_robot.append(traj_robot)
            camera_configs.append((t, yaw))

            if vis_candidates:
                idx = len(camera_configs)
                for k, pose in enumerate(traj_robot):
                    scene.add_object(
                        Object.create_sphere(
                            radius=0.005,
                            name=f"traj_{idx}_{k}",
                            color=[0.7, 0.7, 0.7],
                        )
                    )
                    scene[f"traj_{idx}_{k}"].pos[:] = pose.translation

    # -----------------------------
    # Stage 2, grasp feasibility filtering
    # -----------------------------

    valid_cases = []

    for i, traj_robot in enumerate(all_trajs_robot):
        obj_pose = traj_robot[0]

        grasp_generator = GraspGenerator(
            obj_pose=obj_pose,
            grasp_configurations_number=5,
        )
        grasps = grasp_generator.generate_grasps_configurations()

        valid_grasps = []
        for qg in grasps:
            if is_grasp_valid(qg, rmodel, cmodel, rdata, cdata):
                valid_grasps.append(qg)

        if len(valid_grasps) == 0:
            print(f"[SKIP] Camera {i}, no valid grasp")
            continue

        t, yaw = camera_configs[i]

        valid_cases.append(
            {
                "camera_translation": t,
                "camera_yaw": yaw,
                "traj_robot": traj_robot,
                "grasp": valid_grasps[0],
            }
        )

    print(f"Valid cases after grasp filtering: {len(valid_cases)} / {len(all_trajs_robot)}")

    # -----------------------------
    # Stage 3, OCP only on valid cases
    # -----------------------------

    for idx, case in enumerate(valid_cases):
        print(f"[OCP] Solving case {idx}")

        traj_robot = case["traj_robot"]
        q0 = case["grasp"]

        OCP_creator = OCP(
            rmodel,
            cmodel,
            traj_robot,
            x0=np.concatenate((q0, np.zeros(rmodel.nv))),
            joint_limits=True,
            joint_limits_constraint=False,
            with_callbacks=False,
            weights=weights,
            safety_threshold=safety_threshold,
            T=len(traj_robot),
            dt=dt,
        )

        ocp = OCP_creator.create_OCP()
        X_init = [np.concatenate((q0, np.zeros(rmodel.nv)))] * OCP_creator._T
        U_init = ocp.problem.quasiStatic(X_init[:-1])

        try:
            ocp.solve(X_init, U_init)
            case["success"] = True
            case["xs"] = ocp.xs
            print("  success")
        except Exception:
            case["success"] = False
            print("  failed")

    # -----------------------------
    # Stage 4, evaluate results
    # -----------------------------

    best_case = None
    best_error = float("inf")

    for case in valid_cases:
        if not case.get("success", False):
            continue

        traj_robot = case["traj_robot"]
        xs = case["xs"]

        traj_in_configuration_space = TrajectoryInConfigurationSpace(
            configurations=[x[: rmodel.nq] for x in xs]
        )

        evaluator = TrajectoryEvaluator(
            trajectory=traj_robot,
            traj_in_configuration_space=traj_in_configuration_space,
            rmodel=rmodel,
        )

        position_error = evaluator.evaluate_position_error()
        case["position_error"] = position_error

        print(f"Position error: {position_error:.4f} m")

        if position_error < best_error:
            best_error = position_error
            best_case = case

    print(f"Best position error: {best_error:.4f} m")
    print(f"Total time: {time.time() - tstart:.2f} s")

    # -----------------------------
    # Stage 5, visualize best case or all the end-effector trajectories
    # -----------------------------

    if vis_best_case and best_case is not None:
        traj_robot = best_case["traj_robot"]
        xs = best_case["xs"]

        for k, (x, target) in enumerate(zip(xs, traj_robot)):
            robot[:] = x[: rmodel.nq]
            pin.forwardKinematics(rmodel, rdata, x[: rmodel.nq])
            pin.updateFramePlacements(rmodel, rdata)

            ee_frame_id = rmodel.getFrameId("panda_hand_tcp")
            ee_pose = rdata.oMf[ee_frame_id]

            scene.add_object(
                Object.create_sphere(radius=0.01, name=f"ee_{k}", color=[0, 1, 0])
            )
            scene.add_object(
                Object.create_sphere(radius=0.01, name=f"target_{k}", color=[1, 0, 0])
            )

            scene[f"ee_{k}"].pos[:] = ee_pose.translation
            scene[f"target_{k}"].pos[:] = target.translation

            time.sleep(0.05)

    else:
        for i, case in enumerate(valid_cases):
            if not case.get("success", False):
                continue

            traj_robot = case["traj_robot"]
            xs = case["xs"]
            print(f"Visualizing case {i}, step {k}, position error {case['position_error']:.4f} m")

            for k, (x, target) in enumerate(zip(xs, traj_robot)):
                robot[:] = x[: rmodel.nq]
                pin.forwardKinematics(rmodel, rdata, x[: rmodel.nq])
                pin.updateFramePlacements(rmodel, rdata)

                ee_frame_id = rmodel.getFrameId("panda_hand_tcp")
                ee_pose = rdata.oMf[ee_frame_id]

                scene.add_object(
                    Object.create_sphere(radius=0.005, name=f"case_{i}_ee_{k}", color=[0, 1, 0])
                )
                scene.add_object(
                    Object.create_sphere(radius=0.005, name=f"case_{i}_target_{k}", color=[1, 0, 0])
                )

                scene[f"case_{i}_ee_{k}"].pos[:] = ee_pose.translation
                scene[f"case_{i}_target_{k}"].pos[:] = target.translation
                o.pose = (target).homogeneous

                time.sleep(0.05)
            input("Press Enter to continue to the next case...")
            
            for k in range(len(xs)): 
                scene[f'case_{i}_ee_{k}'].hide() 
                scene[f'case_{i}_target_{k}'].hide()
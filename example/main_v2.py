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


# -----------------------------
# Utilities
# -----------------------------

def sample_camera_translations(n=100, low=-0.5, high=0.5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(n, 3))


def camera_to_robot_trajectory(parser, camera_translation):
    T_camera_robot = pin.SE3(np.eye(3), camera_translation)
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
        "exp_wildpose/howto100m_poses/howto100m_9Mh7jlESPvs_1-smoothed.csv"
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

    obj = Object.create_mesh(
        path_to_mesh=obj_file,
        name="robot/movable_obj",
        texture=texture_file,
        scale=scale,
        color=color,
    )
    scene.add_object(obj)

    # -----------------------------
    # Stage 1, sample camera poses and visualize object trajectories
    # -----------------------------

    camera_translations = sample_camera_translations(n=100)
    all_trajs_robot = []

    for i, t in enumerate(camera_translations):
        traj_robot = camera_to_robot_trajectory(parser, t)
        all_trajs_robot.append(traj_robot)

        for k, pose in enumerate(traj_robot):
            scene.add_object(
                Object.create_sphere(
                    radius=0.005,
                    name=f"traj_{i}_{k}",
                    color=[0.7, 0.7, 0.7],
                )
            )
            scene[f"traj_{i}_{k}"].pos[:] = pose.translation

    input("Trajectories visualized. Press Enter to start grasp feasibility checks...")

    # -----------------------------
    # Stage 2, grasp feasibility filtering
    # -----------------------------

    valid_cases = []

    for i, traj_robot in enumerate(all_trajs_robot):
        obj_pose = traj_robot[0]

        grasp_generator = GraspGenerator(
            obj_pose=obj_pose,
            grasp_configurations_number=10,
        )
        grasps = grasp_generator.generate_grasps_configurations()

        valid_grasps = []
        for qg in grasps:
            if is_grasp_valid(qg, rmodel, cmodel, rdata, cdata):
                valid_grasps.append(qg)

        if len(valid_grasps) == 0:
            print(f"[SKIP] Camera {i}, no valid grasp")
            continue

        valid_cases.append(
            {
                "camera_translation": camera_translations[i],
                "traj_robot": traj_robot,
                "grasp": valid_grasps[0],
            }
        )

    print(f"Valid cases after grasp filtering: {len(valid_cases)} / {len(camera_translations)}")

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
        except Exception as e:
            case["success"] = False
            print("  failed")

    print("All done.")

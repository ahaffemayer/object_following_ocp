import time
from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml
from robomeshcat import Object, Robot, Scene

from object_following_ocp.dataclass import ConfigLoader
from object_following_ocp.grasp_generator import GraspGenerator
from object_following_ocp.ocp import OCP
from object_following_ocp.robot_loader import load_reduced_panda, self_collision_pairs
from object_following_ocp.trajectory import (
    Trajectory,
    TrajectoryEvaluator,
    TrajectoryInConfigurationSpace,
)
from object_following_ocp.trajectory_parser import JSONTrajectoryParser

vis_candidates = False  # set True to visualize all candidate trajectories
vis_best_case = False  # set True to visualize only the best case found

# -----------------------------
# Utilities
# -----------------------------

CAMERA_YAW = 0.0  # Fixed yaw since diverse grasps already provide orientational variety


def pose_distance_to_color(T_target, T_ee, d_min=0.05, d_max=0.2):
    """
    Compute RGB color based on distance between two poses.

    Args:
        T_target: 4x4 homogeneous matrix of target pose
        T_ee: 4x4 homogeneous matrix of end-effector pose
        d_min: distance below which color is pure green
        d_max: distance above which color is pure red

    Returns:
        list [r, g, b] with values in [0, 1]
    """
    # Extract translations
    p_target = T_target[:3, 3]
    p_ee = T_ee[:3, 3]

    # Distance
    d = np.linalg.norm(p_target - p_ee)

    # Clamp and normalize
    alpha = np.clip((d - d_min) / (d_max - d_min), 0.0, 1.0)

    # Linear interpolation green -> red
    r = alpha
    g = 1.0 - alpha
    b = 0.0

    return [r, g, b]


def load_grasps_from_yaml(yaml_path, gripper_depth=0.1034):
    """
    Load grasps from YAML and convert from gripper base frame to TCP frame.

    Args:
        yaml_path: Path to the grasp YAML file
        gripper_depth: Distance from gripper base to TCP (from config)
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    grasps = []

    for _, g in data["grasps"].items():
        p = np.array(g["position"], dtype=np.float64)

        qw = g["orientation"]["w"]
        qx, qy, qz = g["orientation"]["xyz"]

        quat = pin.Quaternion(qx, qy, qz, qw)
        quat.normalize()

        # This is the grasp pose at the gripper base (GraspGen convention)
        T_base = pin.SE3(quat, p)

        # Create TCP offset transform (z-axis offset by gripper depth)
        # In GraspGen convention, approach direction is along +z
        T_base_to_tcp = pin.SE3(np.eye(3), np.array([0.0, 0.0, gripper_depth]))

        # Transform to TCP frame
        T_tcp = T_base * T_base_to_tcp

        grasps.append(T_tcp)

    return grasps


def transform_grasp_to_robot_frame(T_object_grasp, T_robot_object):
    """
    Transform a grasp pose from object frame to robot frame.

    Args:
        T_object_grasp: 4x4 SE(3) grasp pose in object frame
        T_robot_object: 4x4 SE(3) object pose in robot frame

    Returns:
        T_robot_grasp: 4x4 SE(3) grasp pose in robot frame
    """
    return T_robot_object @ T_object_grasp


def se3_to_pinocchio(T):
    """Convert 4x4 homogeneous matrix to Pinocchio SE3."""
    return pin.SE3(T[:3, :3], T[:3, 3])


def is_grasp_valid(q, rmodel, cmodel, rdata, cdata):
    """Check if a joint configuration is valid (within limits and collision-free)."""
    if q is None:
        return False

    if np.any(q < rmodel.lowerPositionLimit) or np.any(q > rmodel.upperPositionLimit):
        return False

    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)
    has_collision = pin.computeCollisions(cmodel, cdata, False)
    return not has_collision


def sample_camera_translations_grid(
    nx=3,
    ny=3,
    nz=3,
    xlim=(-0.5, 0.1),
    ylim=(-0.5, 0.5),
    zlim=(-0.0, 1.0),
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


def camera_to_robot_trajectory(parser, camera_translation, yaw, object_id=None):
    T_camera_robot = se3_from_translation_yaw(camera_translation, yaw)
    return parser.to_robot_frame(T_camera_robot, object_id=object_id)


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":

    tstart = time.time()

    # Load robot configuration from YAML
    robot_config = ConfigLoader.load(
        Path(__file__).parent / "robot_config.yml")

    # Load trajectory data from JSON
    json_path = Path(
        "/workspaces/object_following_ocp/ressources/json/jug.props-dinov2-ffa-22.gpt4_scaled.best_object.poses-dinov2-22-graph.smoothed-movavg.json"
    )

    # Path to graspgen output YAML file
    grasp_yaml_path = Path(
        "/workspaces/object_following_ocp/results/jug/jug_grasps_filtered_new.yml"
    )

    # -----------------------------
    # Load trajectory parser
    # -----------------------------

    parser = JSONTrajectoryParser(json_path, smooth_depth=True, smooth_k=2.0)

    # Get object information (auto-selects first available object)
    object_info = parser.get_object_info(texture_name="material_0.png")

    print(f"Object mesh: {object_info.mesh_path}")
    print(f"Object scale: {object_info.scale}")

    # -----------------------------
    # Load grasp poses from graspgen
    # -----------------------------

    print("Loading grasp poses from graspgen...")
    grasps_object_frame = load_grasps_from_yaml(
        grasp_yaml_path, gripper_depth=robot_config.gripper_depth
    )
    print(
        f"Loaded {len(grasps_object_frame)} diverse grasp poses (converted to TCP frame)"
    )

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
        path_to_mesh=object_info.mesh_path,
        name="robot/movable_obj",
        texture=object_info.texture_path,
        scale=object_info.scale,
        color=[0.8, 0.8, 0.8],
    )
    print(f"object_info.texture_path: {object_info.texture_path}")
    scene.add_object(o)

    o_EE = Object.create_mesh(
        path_to_mesh=object_info.mesh_path,
        name="robot/movable_obj_ee",
        scale=object_info.scale,
        texture=object_info.texture_path,
        color=(1.0, 0.0, 0.0),
    )
    scene.add_object(o_EE)

    # -----------------------------
    # Stage 1, sample camera poses
    # -----------------------------

    camera_translations = sample_camera_translations_grid(
        nx=3,
        ny=3,
        nz=3,
        xlim=(-0.3, 0.3),
        ylim=(-0.4, 0.4),
        zlim=(-0.5, 0.0),
    )
    all_trajs_robot = []
    camera_configs = []  # list of translations

    for t in camera_translations:
        traj_robot = camera_to_robot_trajectory(parser, t, CAMERA_YAW)
        all_trajs_robot.append(traj_robot)
        camera_configs.append(t)

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
        # Get object pose at first timestep
        obj_pose = traj_robot[0]  # This is a pin.SE3
        T_robot_object = obj_pose.homogeneous  # Convert to 4x4 matrix

        print(
            f"Processing camera config {i}: checking {len(grasps_object_frame)} grasps"
        )

        # For each diverse grasp from graspgen
        for grasp_idx, T_object_grasp in enumerate(grasps_object_frame):
            # Transform grasp to robot frame
            T_robot_grasp_matrix = transform_grasp_to_robot_frame(
                T_object_grasp, T_robot_object
            )
            T_robot_grasp_se3 = se3_to_pinocchio(T_robot_grasp_matrix)

            # Use GraspGenerator to compute IK for this target grasp pose
            grasp_generator = GraspGenerator(
                obj_pose=T_robot_grasp_se3,
                grasp_configurations_number=1,  # We only need one IK solution per grasp
            )
            grasps = grasp_generator.generate_grasps_configurations()

            # Check validity of generated configurations
            valid_q = None
            for q_candidate in grasps:
                if is_grasp_valid(q_candidate, rmodel, cmodel, rdata, cdata):
                    valid_q = q_candidate
                    break

            if valid_q is None:
                continue

            t = camera_configs[i]

            valid_cases.append(
                {
                    "camera_translation": t,
                    "traj_robot": traj_robot,
                    "grasp": valid_q,
                    "grasp_idx": grasp_idx,
                    "T_robot_grasp": T_robot_grasp_matrix,
                }
            )

    print(
        f"Valid cases after grasp filtering: {len(valid_cases)} (from {len(all_trajs_robot)} camera configs)"
    )

    # -----------------------------
    # Stage 3, OCP only on valid cases
    # -----------------------------

    for idx, case in enumerate(valid_cases):
        print(f"[OCP] Solving case {idx} (grasp {case['grasp_idx']})")

        traj_robot = case["traj_robot"]
        q0 = case["grasp"]

        # Create weights dictionary from robot_config
        weights = {
            "W_xREG": robot_config.W_xREG,
            "W_uREG": robot_config.W_uREG,
            "W_gripper_pose": robot_config.W_gripper_pose,
            "W_gripper_pose_term": robot_config.W_gripper_pose_term,
            "W_limit": robot_config.W_limit,
        }

        OCP_creator = OCP(
            rmodel,
            cmodel,
            traj_robot,
            x0=np.concatenate((q0, np.zeros(rmodel.nv))),
            joint_limits=True,
            joint_limits_constraint=False,
            with_callbacks=False,
            weights=weights,
            safety_threshold=robot_config.safety_threshold,
            T=len(traj_robot),
            dt=robot_config.dt,
        )

        ocp = OCP_creator.create_OCP()
        X_init = [np.concatenate((q0, np.zeros(rmodel.nv)))] * OCP_creator._T
        U_init = ocp.problem.quasiStatic(X_init[:-1])

        try:
            ocp.solve(X_init, U_init)
            case["success"] = True
            case["xs"] = ocp.xs
            print(f"  success (grasp {case['grasp_idx']})")
        except Exception as e:
            case["success"] = False
            print(f"  failed (grasp {case['grasp_idx']}): {e}")

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

        print(
            f"Case with grasp {case['grasp_idx']}: Position error: {position_error:.4f} m"
        )

        if position_error < best_error:
            best_error = position_error
            best_case = case

    if best_case is not None:
        print(f"\nBest case:")
        print(f"  Grasp index: {best_case['grasp_idx']}")
        print(f"  Position error: {best_error:.4f} m")
        print(f"  Camera translation: {best_case['camera_translation']}")
    else:
        print("\nNo successful cases found!")

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
                Object.create_sphere(
                    radius=0.01, name=f"ee_{k}", color=[0, 1, 0])
            )
            scene.add_object(
                Object.create_sphere(
                    radius=0.01, name=f"target_{k}", color=[1, 0, 0])
            )

            scene[f"ee_{k}"].pos[:] = ee_pose.translation
            scene[f"target_{k}"].pos[:] = target.translation
            o.pose = (target).homogeneous

            time.sleep(0.05)

    else:
        for i, case in enumerate(valid_cases):
            if not case.get("success", False):
                continue

            traj_robot = case["traj_robot"]
            xs = case["xs"]
            print(
                f"Visualizing case {i} (grasp {case['grasp_idx']}), position error {case['position_error']:.4f} m"
            )

            for k, (x, target) in enumerate(zip(xs, traj_robot)):
                robot[:] = x[: rmodel.nq]
                pin.forwardKinematics(rmodel, rdata, x[: rmodel.nq])
                pin.updateFramePlacements(rmodel, rdata)

                ee_frame_id = rmodel.getFrameId("panda_hand_tcp")
                ee_pose = rdata.oMf[ee_frame_id]

                scene.add_object(
                    Object.create_sphere(
                        radius=0.005, name=f"case_{i}_ee_{k}", color=[0, 1, 0]
                    )
                )
                scene.add_object(
                    Object.create_sphere(
                        radius=0.005, name=f"case_{i}_target_{k}", color=[1, 0, 0]
                    )
                )

                scene[f"case_{i}_ee_{k}"].pos[:] = ee_pose.translation
                scene[f"case_{i}_target_{k}"].pos[:] = target.translation

                T_offset = pin.SE3(
                    pin.utils.rpyToMatrix(0.0, 0.0, np.pi / 2),
                    np.array([0.0, 0.0, 0.0]),
                )
                color = pose_distance_to_color(
                    (target * T_offset).homogeneous,
                    (ee_pose * T_offset).homogeneous,
                )
                o.pose = (target * T_offset).homogeneous
                o_EE.pose = (ee_pose * T_offset).homogeneous

                time.sleep(0.05)

            input("Press Enter to continue to the next case...")

            for k in range(len(xs)):
                scene[f"case_{i}_ee_{k}"].hide()
                scene[f"case_{i}_target_{k}"].hide()

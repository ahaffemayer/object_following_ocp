import pathlib

import numpy as np
import pinocchio as pin
from robomeshcat import Object, Robot, Scene

from object_following_ocp.data_loader import ConfigLoader, DataLoader
from object_following_ocp.grasp_transforms import (
    GraspTransformChain,
    GraspTransformConfig,
)
from object_following_ocp.ik_curobo import RobotIKSolver
from object_following_ocp.ocp import OCP
from object_following_ocp.robot_loader import load_reduced_panda
from object_following_ocp.trajectories import TrajectoryInConfigurationSpace

if __name__ == "__main__":
    # -----------------------------
    # Load data
    # -----------------------------
    object_traj_path = pathlib.Path(
        "/workspaces/object_following_ocp/ressources/json/bowl1.props-dinov2-ffa-22.gpt4_scaled.best_object.poses-dinov2-22-graph.smoothed-movavg.json"
    )
    scale_path = pathlib.Path(
        "/workspaces/object_following_ocp/ressources/grasps_scales.json"
    )
    config_path = pathlib.Path(
        "/workspaces/object_following_ocp/example/robot_motion/configs/ocp_config.yml"
    )

    # Load data using DataLoader
    dataloader = DataLoader(
        object_trajectory_path=object_traj_path,
        scales_path=scale_path,
        load_grasps=True,  # Auto-load grasps based on mesh_id
    )

    # Load robot configuration
    robot_config = ConfigLoader.load(config_path)

    print(f"Loaded object: {dataloader.object_info.mesh_id}")
    print(f"Trajectory length: {len(dataloader.poses)} poses")
    print(
        f"Best grasp: {dataloader.best_grasp.name if dataloader.has_grasps else 'None'}"
    )
    print(f"Gripper depth from config: {robot_config.gripper_depth} m")

    # Get trajectories from dataloader
    object_traj_camera = dataloader.to_trajectory_SE3()
    objectM_grasp = dataloader.best_grasp_SE3

    # -----------------------------
    # Setup robot and scene
    # -----------------------------
    rmodel, cmodel, vmodel = load_reduced_panda()
    rdata = rmodel.createData()
    cdata = cmodel.createData()
    vdata = vmodel.createData()

    scene = Scene()

    robot = Robot(
        pinocchio_model=rmodel,
        pinocchio_data=rdata,
        pinocchio_geometry_model=vmodel,
        pinocchio_geometry_data=vdata,
    )
    scene.add_robot(robot=robot)

    # Add object to scene
    o = Object.create_mesh(
        path_to_mesh=dataloader.object_info.mesh_path,
        name="robot/movable_obj",
        texture=dataloader.object_info.texture_path,
        scale=dataloader.object_info.scale,
        color=[0.8, 0.8, 0.8],
    )
    scene.add_object(o)

    # -----------------------------
    # Initialize transformation chain
    # -----------------------------
    # Create config from robot config
    grasp_config = GraspTransformConfig.from_robot_config(
        robot_config=robot_config,
        camera_translation=np.array([0, -0.7, -1.0]),
        grasp_correction_angle_deg=90.0,
        elevation_angle_deg=25.0,
    )

    transform_chain = GraspTransformChain(grasp_config)
    print("\n" + transform_chain.get_transform_summary())

    # -----------------------------
    # Compute EE and TCP trajectories in world frame
    # -----------------------------
    print("\nComputing end-effector and TCP trajectories...")

    # Use the high-level API
    object_traj_world = transform_chain.transform_object_trajectory(object_traj_camera)

    # EE trajectory (for IK) - WITHOUT gripper depth offset
    ee_traj_world = transform_chain.transform_ee_trajectory(
        object_traj_camera, objectM_grasp
    )

    # TCP trajectory (for OCP) - WITH gripper depth offset
    tcp_traj_world = transform_chain.transform_tcp_trajectory(
        object_traj_camera, objectM_grasp
    )

    print(f"Computed {len(ee_traj_world)} end-effector poses (for IK)")
    print(f"Computed {len(tcp_traj_world)} TCP poses (for OCP)")
    print(f"TCP trajectory includes {robot_config.gripper_depth}m gripper depth offset")

    # -----------------------------
    # Visualize trajectory
    # -----------------------------
    for k, wM_ee in enumerate(ee_traj_world):
        color = [0.0, 1.0, 0.0] if k == 0 else [0.5, 0.5, 0.5]
        scene.add_object(
            Object.create_sphere(radius=0.01, name=f"target_{k}", color=color)
        )
        scene[f"target_{k}"].pos[:] = wM_ee.translation

    # -----------------------------
    # Solve IK for entire trajectory
    # -----------------------------
    solver = RobotIKSolver(
        robot_name="franka",
        num_seeds=20,
        position_threshold=0.005,
        rotation_threshold=0.05,
        use_cuda_graph=False,
    )

    joint_configurations = []
    ik_success_count = 0

    print("\n" + "=" * 60)
    print("Solving IK for trajectory...")
    print("=" * 60)

    for k, wM_ee in enumerate(ee_traj_world):
        solution, info = solver.solve(wM_ee)

        if solution is not None:
            joint_configurations.append(solution)
            ik_success_count += 1

            if k % 10 == 0 or k == 0:  # Print every 10th point
                print(
                    f"Point {k}: SUCCESS - pos_err={info['position_error']:.6f}m, "
                    f"rot_err={info['rotation_error']:.6f}rad"
                )
        else:
            joint_configurations.append(None)
            print(
                f"Point {k}: FAILED - pos_err={info['position_error']:.6f}m, "
                f"rot_err={info['rotation_error']:.6f}rad"
            )

    print(
        f"\nIK Success rate: {ik_success_count}/{len(ee_traj_world)} "
        f"({100 * ik_success_count / len(ee_traj_world):.1f}%)"
    )

    # Create joint trajectory (filter out None values)
    valid_configurations = [q for q in joint_configurations if q is not None]
    joint_trajectory = TrajectoryInConfigurationSpace(valid_configurations)

    print(f"Joint trajectory has {len(joint_trajectory)} valid configurations")

    # -----------------------------
    # Verify first pose in detail
    # -----------------------------
    if len(joint_trajectory) > 0:
        solution = joint_trajectory[0]
        wM_ee = ee_traj_world[0]

        print("\n" + "=" * 60)
        print("Detailed verification for first pose:")
        print("=" * 60)

        # Verify solution with FK from CuRobo
        fk_pos_curobo, fk_quat_curobo = solver.forward_kinematics(solution)
        print("\nCuRobo FK:")
        print(f"Position: {fk_pos_curobo}")
        print(f"Quaternion (wxyz): {fk_quat_curobo}")

        # Convert quaternion to rotation matrix
        R_curobo = pin.Quaternion(
            float(fk_quat_curobo[0]),  # w
            float(fk_quat_curobo[1]),  # x
            float(fk_quat_curobo[2]),  # y
            float(fk_quat_curobo[3]),  # z
        ).toRotationMatrix()

        # Compute FK with Pinocchio
        pin.framesForwardKinematics(rmodel, rdata, solution)
        tcp_frame_id = rmodel.getFrameId("panda_hand_tcp")
        wM_tcp_pinocchio = rdata.oMf[tcp_frame_id]

        print("\nPinocchio FK (panda_hand_tcp frame):")
        print(f"Position: {wM_tcp_pinocchio.translation}")
        quat_pinocchio = pin.Quaternion(wM_tcp_pinocchio.rotation)
        print(
            f"Quaternion (wxyz): [{quat_pinocchio.w}, {quat_pinocchio.x}, "
            f"{quat_pinocchio.y}, {quat_pinocchio.z}]"
        )

        # Comparison
        pos_diff = fk_pos_curobo - wM_tcp_pinocchio.translation
        print(f"\nPosition difference: {pos_diff}")
        print(f"Position error norm: {np.linalg.norm(pos_diff):.6f} m")

        R_diff = R_curobo.T @ wM_tcp_pinocchio.rotation
        angle_diff = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
        print(f"Rotation angle difference: {np.rad2deg(angle_diff):.6f} degrees")

        # Set robot to first pose
        robot[:] = solution
        o.pose = object_traj_world[0].homogeneous

    # -----------------------------
    # Optionally: Animate through trajectory
    # -----------------------------
    print("\n" + "=" * 60)
    print("Animation controls:")
    print("=" * 60)
    print("Press Enter to step through trajectory")
    print("Type 'q' to quit")
    print("=" * 60)

    poses_tool = []
    valid_idx = 0
    for k in range(len(ee_traj_world)):
        if joint_configurations[k] is not None:
            user_input = input(
                f"\nShowing pose {k}/{len(ee_traj_world) - 1} "
                f"(valid config {valid_idx}/{len(joint_trajectory) - 1}) "
                f"(Enter to continue, 'q' to quit): "
            )
            if user_input.lower() == "q":
                break

            robot[:] = joint_trajectory[valid_idx]
            pin.framesForwardKinematics(
                rmodel, rdata, np.array((joint_trajectory[valid_idx]))
            )
            tool_pose = rdata.oMf[rmodel.getFrameId("panda_hand_tcp")]
            poses_tool.append(tool_pose.copy())
            o.pose = object_traj_world[k].homogeneous
            print(f"Displaying trajectory point {k}")
            valid_idx += 1
        else:
            print(f"Skipping point {k} (IK failed)")

    print("\n" + "=" * 60)
    print("Setting up OCP:")
    print("=" * 60)

    # Use full trajectory length
    T_ocp = len(tcp_traj_world)

    # Initial state: first configuration from IK with zero velocities
    q0 = joint_trajectory[0]
    x0 = np.concatenate((q0, np.zeros(rmodel.nv)))

    # Create weights dictionary from robot_config
    weights = {
        "W_xREG": robot_config.W_xREG,
        "W_uREG": robot_config.W_uREG,
        "W_gripper_pose": robot_config.W_gripper_pose,
        "W_gripper_pose_term": robot_config.W_gripper_pose_term,
        "W_limit": robot_config.W_limit,
    }

    print(f"  T = {T_ocp} (full trajectory length)")
    print(f"  dt = {robot_config.dt}")
    print(f"  Weights: {weights}")
    print(f"  Initial state q0: {q0}")
    print("  Tracking: Analytical TCP trajectory (object + grasp + gripper_depth)")

    # Create OCP with analytical TCP trajectory
    # This is the THEORETICAL trajectory we want the TCP to follow
    # NOT the trajectory achieved by IK
    OCP_creator = OCP(
        rmodel,
        cmodel,
        tcp_traj_world,  # Analytical TCP trajectory with proper offsets!
        x0=x0,
        joint_limits=True,
        joint_limits_constraint=False,
        with_callbacks=False,
        weights=weights,
        safety_threshold=robot_config.safety_threshold,
        T=T_ocp,
        dt=robot_config.dt,
    )

    ocp = OCP_creator.create_OCP()

    # -----------------------------
    # Create warm start from IK trajectory
    # -----------------------------
    print("\nCreating warm start from IK trajectory...")

    X_init = [np.concatenate((q, np.zeros(rmodel.nv))) for q in joint_trajectory]

    # Ensure correct length
    if len(X_init) > T_ocp:
        X_init = X_init[:T_ocp]
    elif len(X_init) < T_ocp:
        while len(X_init) < T_ocp:
            X_init.append(X_init[-1])

    U_init = ocp.problem.quasiStatic(X_init[:-1])

    print(f"Warm start created: {len(X_init)} states, {len(U_init)} controls")

    # -----------------------------
    # Solve OCP
    # -----------------------------
    print("\n" + "=" * 60)
    print("Solving OCP with IK warm start...")
    print("=" * 60)

    ocp.solve(X_init, U_init)

    print("\nOCP solved!")

    # -----------------------------
    # Visualize OCP solution
    # -----------------------------
    print("\n" + "=" * 60)
    print("Visualizing OCP solution:")
    print("=" * 60)
    print("Press Enter to step through OCP trajectory")
    print("Type 'q' to quit")
    print("=" * 60)

    for k, xs in enumerate(ocp.xs):
        user_input = input(
            f"\nShowing OCP state {k}/{len(ocp.xs) - 1} "
            f"(Enter to continue, 'q' to quit): "
        )
        if user_input.lower() == "q":
            break

        robot[:] = xs[: rmodel.nq]

        if k < len(object_traj_world):
            o.pose = object_traj_world[k].homogeneous

        print(f"Displaying OCP trajectory point {k}")

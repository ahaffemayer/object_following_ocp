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
        "/workspaces/object_following_ocp/ressources/json/bowl4.props-dinov2-ffa-22.gpt4_scaled.best_object.poses-dinov2-22-graph.smoothed-movavg.json"
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
    # NOTE: Set gripper_depth=0 because Pinocchio's panda_hand_tcp already includes it
    grasp_config = GraspTransformConfig.from_robot_config(
        robot_config=robot_config,
        camera_translation=np.array([0, -0.7, -1.5]),
        grasp_correction_angle_deg=90.0,
        elevation_angle_deg=25.0,
    )
    # Override gripper_depth since panda_hand_tcp frame already includes the offset
    grasp_config.gripper_depth = 0.0

    transform_chain = GraspTransformChain(grasp_config)
    print("\n" + transform_chain.get_transform_summary())

    # -----------------------------
    # Compute trajectories in world frame
    # -----------------------------
    print("\nComputing trajectories in world frame...")

    # Object trajectory in world frame
    object_traj_world = transform_chain.transform_object_trajectory(object_traj_camera)

    # End-effector trajectory: panda_hand_tcp poses (grasp corrected, no gripper offset)
    # This is the trajectory the OCP will track
    ee_traj_world = transform_chain.transform_ee_trajectory(
        object_traj_camera, objectM_grasp
    )

    print(f"Computed {len(ee_traj_world)} end-effector poses (panda_hand_tcp)")

    # -----------------------------
    # Debug: Verify frame definitions
    # -----------------------------
    print("\n" + "=" * 60)
    print("Frame verification:")
    print("=" * 60)

    tcp_frame_id = rmodel.getFrameId("panda_hand_tcp")
    print(f"panda_hand_tcp frame id: {tcp_frame_id}")

    tcp_placement = rmodel.frames[tcp_frame_id].placement
    print("panda_hand_tcp placement relative to parent:")
    print(f"  Translation: {tcp_placement.translation}")

    # -----------------------------
    # Visualize trajectory targets
    # -----------------------------
    print("\n" + "=" * 60)
    print("Visualizing trajectory targets...")
    print("=" * 60)

    for k, wM_ee in enumerate(ee_traj_world):
        # Color: green for first, gray for rest
        color = [0.0, 1.0, 0.0] if k == 0 else [0.5, 0.5, 0.5]

        # Create sphere at panda_hand_tcp target position
        scene.add_object(
            Object.create_sphere(radius=0.01, name=f"target_tcp_{k}", color=color)
        )
        scene[f"target_tcp_{k}"].pos[:] = wM_ee.translation

        # Also visualize object poses in red for reference
        if k % 5 == 0:  # Only every 5th to avoid clutter
            scene.add_object(
                Object.create_sphere(radius=0.008, name=f"object_{k}", color=[1, 0, 0])
            )
            scene[f"object_{k}"].pos[:] = object_traj_world[k].translation

    print(f"Created {len(ee_traj_world)} TCP target spheres (green/gray)")
    print(f"Created {len(ee_traj_world) // 5 + 1} object reference spheres (red)")

    # -----------------------------
    # Initialize IK solver
    # -----------------------------
    solver = RobotIKSolver(
        robot_name="franka",
        num_seeds=20,
        position_threshold=0.005,
        rotation_threshold=0.05,
        use_cuda_graph=False,
    )

    # -----------------------------
    # Solve IK for entire trajectory
    # -----------------------------
    joint_configurations = []
    ik_success_count = 0

    print("\n" + "=" * 60)
    print("Solving IK for trajectory...")
    print("=" * 60)

    # Use ee_traj_world directly - no offset needed
    # CuRobo and Pinocchio use different TCP frames, but that's okay
    # CuRobo IK will find joint angles that work for the desired pose
    for k, wM_ee in enumerate(ee_traj_world):
        solution, info = solver.solve(wM_ee)

        if solution is not None:
            joint_configurations.append(solution)
            ik_success_count += 1

            if k % 10 == 0 or k == 0:
                print(
                    f"Point {k}: SUCCESS - pos_err={info['position_error']:.6f}m, "
                    f"rot_err={info['rotation_error']:.6f}rad"
                )
        else:
            if len(joint_configurations) > 0:
                joint_configurations.append(joint_configurations[-1])
                print(f"Point {k}: FAILED - using previous configuration")
            else:
                joint_configurations.append(np.zeros(rmodel.nq))
                print(f"Point {k}: FAILED - using default configuration")

    print(
        f"\nIK Success rate: {ik_success_count}/{len(ee_traj_world)} "
        f"({100 * ik_success_count / len(ee_traj_world):.1f}%)"
    )

    joint_trajectory_ik = TrajectoryInConfigurationSpace(joint_configurations)
    print(f"IK trajectory has {len(joint_trajectory_ik)} configurations")

    # -----------------------------
    # Verify IK solution using CuRobo FK
    # -----------------------------
    print("\n" + "=" * 60)
    print("Verifying IK solutions match target poses (using CuRobo FK):")
    print("=" * 60)

    ik_errors_curobo = []
    for k in range(min(5, len(joint_trajectory_ik))):
        q = joint_trajectory_ik[k]

        # Use CuRobo FK to verify (this is the frame IK was solving for)
        fk_pos_curobo, _ = solver.forward_kinematics(q)
        wM_tcp_desired = ee_traj_world[k]

        pos_error = np.linalg.norm(fk_pos_curobo - wM_tcp_desired.translation)
        ik_errors_curobo.append(pos_error)
        print(f"Point {k}: IK position error (CuRobo FK) = {pos_error:.6f} m")

    # Also check with Pinocchio FK to see the frame difference
    print("\n" + "=" * 60)
    print("Frame difference check (Pinocchio FK vs target):")
    print("=" * 60)

    ik_errors_pinocchio = []
    for k in range(min(5, len(joint_trajectory_ik))):
        q = joint_trajectory_ik[k]

        # Use Pinocchio FK
        pin.framesForwardKinematics(rmodel, rdata, q)
        tcp_frame_id = rmodel.getFrameId("panda_hand_tcp")
        wM_tcp_pinocchio = rdata.oMf[tcp_frame_id]
        wM_tcp_desired = ee_traj_world[k]

        pos_error = np.linalg.norm(
            wM_tcp_pinocchio.translation - wM_tcp_desired.translation
        )
        ik_errors_pinocchio.append(pos_error)
        print(f"Point {k}: Frame offset (Pinocchio FK) = {pos_error:.6f} m")

    print(
        "\nNote: The ~10cm difference is expected - it's the TCP frame offset between CuRobo and Pinocchio"
    )

    # -----------------------------
    # Detailed verification for first pose
    # -----------------------------
    if len(joint_trajectory_ik) > 0:
        solution = joint_trajectory_ik[0]

        print("\n" + "=" * 60)
        print("Detailed verification for first pose:")
        print("=" * 60)

        # CuRobo FK
        fk_pos_curobo, fk_quat_curobo = solver.forward_kinematics(solution)
        print("\nCuRobo FK:")
        print(f"Position: {fk_pos_curobo}")

        # Pinocchio FK
        pin.framesForwardKinematics(rmodel, rdata, solution)
        tcp_frame_id = rmodel.getFrameId("panda_hand_tcp")
        wM_tcp_pinocchio = rdata.oMf[tcp_frame_id]

        print("\nPinocchio FK (panda_hand_tcp frame):")
        print(f"Position: {wM_tcp_pinocchio.translation}")

        # Target
        print("\nTarget pose:")
        print(f"Position: {ee_traj_world[0].translation}")

        # Comparison
        pos_diff_curobo = fk_pos_curobo - ee_traj_world[0].translation
        pos_diff_pinocchio = wM_tcp_pinocchio.translation - ee_traj_world[0].translation

        print("\nComparison:")
        print(
            f"CuRobo FK error: {np.linalg.norm(pos_diff_curobo):.6f} m (should be ~0)"
        )
        print(
            f"Pinocchio FK offset: {np.linalg.norm(pos_diff_pinocchio):.6f} m (expected ~0.103m)"
        )

        # Set robot to first pose
        robot[:] = solution
        o.pose = object_traj_world[0].homogeneous

    # -----------------------------
    # Setup and solve OCP
    # -----------------------------
    print("\n" + "=" * 60)
    print("Setting up OCP...")
    print("=" * 60)

    # Use full trajectory length
    T_ocp = len(ee_traj_world)

    # Initial state: first configuration from IK with zero velocities
    q0 = joint_trajectory_ik[0]
    x0 = np.concatenate((q0, np.zeros(rmodel.nv)))

    # Create weights dictionary from robot_config
    weights = {
        "W_xREG": robot_config.W_xREG,
        "W_uREG": robot_config.W_uREG,
        "W_gripper_pose": robot_config.W_gripper_pose,
        "W_gripper_pose_term": robot_config.W_gripper_pose_term,
        "W_limit": robot_config.W_limit,
    }

    print("\nOCP Configuration:")
    print(f"  T = {T_ocp} (full trajectory length)")
    print(f"  dt = {robot_config.dt}")
    print(f"  Weights: {weights}")
    print(f"  Initial state q0: {q0}")

    # Create OCP
    # Important: The OCP uses Pinocchio's panda_hand_tcp frame
    OCP_creator = OCP(
        rmodel,
        cmodel,
        ee_traj_world,  # Pinocchio panda_hand_tcp trajectory
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

    X_init = [np.concatenate((q, np.zeros(rmodel.nv))) for q in joint_trajectory_ik]

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
    # print(f"Solver converged: {ocp.solver.stop < ocp.solver.th_stop}")cl

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

    # -----------------------------
    # Compare IK vs OCP solutions
    # -----------------------------
    print("\n" + "=" * 60)
    print("Comparison: IK vs OCP solutions")
    print("=" * 60)

    joint_diffs = []
    for k in range(min(len(joint_trajectory_ik), len(ocp.xs))):
        q_ik = joint_trajectory_ik[k]
        q_ocp = ocp.xs[k][: rmodel.nq]
        diff = np.linalg.norm(q_ik - q_ocp)
        joint_diffs.append(diff)

    print(f"Mean joint difference: {np.mean(joint_diffs):.6f} rad")
    print(f"Max joint difference: {np.max(joint_diffs):.6f} rad")
    print(f"Min joint difference: {np.min(joint_diffs):.6f} rad")

    # Compute end-effector tracking errors for OCP solution (using Pinocchio FK)
    print("\nOCP End-effector tracking errors (Pinocchio FK):")
    ee_errors_ocp = []
    for k, xs in enumerate(ocp.xs[:-1]):
        q = xs[: rmodel.nq]
        pin.framesForwardKinematics(rmodel, rdata, q)
        tcp_frame_id = rmodel.getFrameId("panda_hand_tcp")
        wM_tcp = rdata.oMf[tcp_frame_id]

        wM_ee_desired = ee_traj_world[k]
        pos_error = np.linalg.norm(wM_tcp.translation - wM_ee_desired.translation)
        ee_errors_ocp.append(pos_error)

    print(f"Mean EE position error: {np.mean(ee_errors_ocp):.6f} m")
    print(f"Max EE position error: {np.max(ee_errors_ocp):.6f} m")

    print("\nComparison IK vs OCP tracking (using appropriate FK for each):")
    print(f"IK mean error (CuRobo FK): {np.mean(ik_errors_curobo):.6f} m")
    print(f"OCP mean error (Pinocchio FK): {np.mean(ee_errors_ocp):.6f} m")

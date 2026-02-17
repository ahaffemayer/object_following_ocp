"""
Refactored object following with OCP - using modular wrappers.
"""

import pathlib

import numpy as np
from example.robot_motion.utils.ik_trajectory_converter import IKTrajectoryConverter
from example.robot_motion.utils.ocp_trajectory_converter import OCPTrajectoryConverter
from robomeshcat import Object, Robot, Scene

# Import our new wrappers
from example.robot_motion.utils.trajectory_ik_solver import TrajectoryIKSolver
from example.robot_motion.utils.trajectory_visualizer import TrajectoryVisualizer

from object_following_ocp.data_loader import ConfigLoader, DataLoader
from object_following_ocp.ocp import OCP
from object_following_ocp.robot_loader import load_reduced_panda

if __name__ == "__main__":
    # ========================================================================
    # SETUP: Load data and robot
    # ========================================================================
    object_traj_path = pathlib.Path(
        "/workspaces/object_following_ocp/ressources/json/bowl1.props-dinov2-ffa-22.gpt4_scaled.best_object.poses-dinov2-22-graph.smoothed-movavg.json"
    )
    scale_path = pathlib.Path(
        "/workspaces/object_following_ocp/ressources/grasps_scales.json"
    )
    config_path = pathlib.Path(
        "/workspaces/object_following_ocp/example/robot_motion/configs/ocp_config.yml"
    )

    # Load data
    dataloader = DataLoader(
        object_trajectory_path=object_traj_path,
        scales_path=scale_path,
        load_grasps=True,
    )
    robot_config = ConfigLoader.load(config_path)

    print(f"Loaded object: {dataloader.object_info.mesh_id}")
    print(f"Trajectory length: {len(dataloader.poses)} poses")
    print(
        f"Best grasp: {dataloader.best_grasp.name if dataloader.has_grasps else 'None'}"
    )
    print(f"Gripper depth from config: {robot_config.gripper_depth} m")

    # Setup robot
    rmodel, cmodel, vmodel = load_reduced_panda()
    rdata = rmodel.createData()
    cdata = cmodel.createData()
    vdata = vmodel.createData()

    # Setup scene
    scene = Scene()
    robot = Robot(
        pinocchio_model=rmodel,
        pinocchio_data=rdata,
        pinocchio_geometry_model=vmodel,
        pinocchio_geometry_data=vdata,
    )
    scene.add_robot(robot=robot)

    # Add object to scene
    movable_object = Object.create_mesh(
        path_to_mesh=dataloader.object_info.mesh_path,
        name="robot/movable_obj",
        texture=dataloader.object_info.texture_path,
        scale=dataloader.object_info.scale,
        color=[0.8, 0.8, 0.8],
    )
    scene.add_object(movable_object)

    # ========================================================================
    # STEP 1: Compute trajectories for IK
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 1: Computing trajectories for IK")
    print("=" * 60)

    ik_converter = IKTrajectoryConverter(
        robot_config=robot_config,
        camera_translation=np.array([0, -0.7, -1.0]),
        grasp_correction_angle_deg=90.0,
        elevation_angle_deg=25.0,
    )

    print(ik_converter.get_transform_summary())

    object_traj_world, ee_traj_world = ik_converter.compute_trajectories(dataloader)

    print(f"\nComputed {len(ee_traj_world)} end-effector poses (for IK)")
    print("  -> These poses do NOT include TCP offset")

    # ========================================================================
    # STEP 2: Solve IK for end-effector trajectory
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Solving IK for end-effector trajectory")
    print("=" * 60)

    ik_solver = TrajectoryIKSolver(
        robot_name="franka",
        num_seeds=20,
        position_threshold=0.005,
        rotation_threshold=0.05,
        use_cuda_graph=False,
        verbose=True,
    )

    joint_trajectory, ik_info = ik_solver.solve_trajectory(
        trajectory=ee_traj_world,
        print_every=10,
    )

    print(f"\nJoint trajectory has {len(joint_trajectory)} valid configurations")
    print(f"IK success rate: {ik_info['success_rate']:.1f}%")

    # Verify first solution
    if len(joint_trajectory) > 0:
        print("\n")
        verification = ik_solver.verify_first_solution(
            solution=joint_trajectory[0],
            target_pose=ee_traj_world[0],
            rmodel=rmodel,
            rdata=rdata,
            tcp_frame_name="panda_hand_tcp",
        )

    # ========================================================================
    # STEP 3: Compute TCP trajectory for OCP
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Computing TCP trajectory for OCP")
    print("=" * 60)

    ocp_converter = OCPTrajectoryConverter(
        robot_config=robot_config,
        camera_translation=np.array([0, -0.7, -1.0]),
        grasp_correction_angle_deg=90.0,
        elevation_angle_deg=25.0,
    )

    tcp_traj_world = ocp_converter.compute_tcp_trajectory(dataloader)

    print(f"Computed {len(tcp_traj_world)} TCP poses (for OCP)")
    print(f"  -> These poses INCLUDE {robot_config.gripper_depth}m TCP offset")
    print("  -> This is the analytical trajectory the OCP will track")

    # ========================================================================
    # STEP 4: Visualize trajectories
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Visualizing trajectories")
    print("=" * 60)

    visualizer = TrajectoryVisualizer(
        scene=scene,
        robot=robot,
        movable_object=movable_object,
    )

    # Visualize EE waypoints
    visualizer.visualize_waypoints(
        trajectory=ee_traj_world,
        sphere_radius=0.01,
        start_color=[0.0, 1.0, 0.0],  # Green for start
        waypoint_color=[0.5, 0.5, 0.5],  # Gray for others
        name_prefix="ee_waypoint",
    )

    # Set initial pose
    if len(joint_trajectory) > 0:
        visualizer.set_robot_and_object_pose(
            joint_config=joint_trajectory[0],
            object_pose=object_traj_world[0],
        )

    # ========================================================================
    # STEP 5: Animate IK trajectory (optional)
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Animate IK trajectory")
    print("=" * 60)

    # Collect TCP poses from IK trajectory
    tcp_poses_from_ik = visualizer.animate_joint_trajectory(
        joint_trajectory=joint_trajectory,
        object_trajectory=object_traj_world,
        rmodel=rmodel,
        rdata=rdata,
        interactive=True,
    )

    # ========================================================================
    # STEP 6: Setup and solve OCP
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 6: Setting up and solving OCP")
    print("=" * 60)

    T_ocp = len(tcp_traj_world)
    q0 = joint_trajectory[0]
    x0 = np.concatenate((q0, np.zeros(rmodel.nv)))

    weights = {
        "W_xREG": robot_config.W_xREG,
        "W_uREG": robot_config.W_uREG,
        "W_gripper_pose": robot_config.W_gripper_pose,
        "W_gripper_pose_term": robot_config.W_gripper_pose_term,
        "W_limit": robot_config.W_limit,
    }

    print("OCP Configuration:")
    print(f"  T = {T_ocp} time steps")
    print(f"  dt = {robot_config.dt} seconds")
    print(f"  Weights: {weights}")
    print("  Tracking: Analytical TCP trajectory (object + grasp + gripper_depth)")

    # Create OCP
    OCP_creator = OCP(
        rmodel,
        cmodel,
        tcp_traj_world,  # Analytical TCP trajectory!
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

    # Create warm start from IK trajectory
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

    # Solve OCP
    print("\nSolving OCP...")
    ocp.solve(X_init, U_init)
    print("OCP solved!")

    # ========================================================================
    # STEP 7: Visualize OCP solution
    # ========================================================================
    print("\n" + "=" * 60)
    print("STEP 7: Visualizing OCP solution")
    print("=" * 60)

    visualizer.animate_ocp_solution(
        ocp_states=ocp.xs,
        object_trajectory=object_traj_world,
        rmodel=rmodel,
        interactive=True,
    )

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)

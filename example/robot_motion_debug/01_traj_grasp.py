import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
from robomeshcat import Object, Robot, Scene

from object_following_ocp.data.data_loader import ConfigLoader, DataLoader
from object_following_ocp.geom.grasp_transforms import (
    GraspTransformChain,
    GraspTransformConfig,
)
from object_following_ocp.solver.ik_curobo import RobotIKSolver
from object_following_ocp.robot.robot_loader import load_reduced_panda
from object_following_ocp.geom.trajectories import TrajectoryInConfigurationSpace

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
        "/workspaces/object_following_ocp/example/robot_motion/configs/ik_config.yml"
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
    # Compute EE trajectory in world frame
    # -----------------------------
    print("\nComputing end-effector trajectory...")

    # Use the high-level API
    object_traj_world = transform_chain.transform_object_trajectory(object_traj_camera)
    ee_traj_world = transform_chain.transform_ee_trajectory(
        object_traj_camera, objectM_grasp
    )

    print(f"Computed {len(ee_traj_world)} end-effector poses")

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

    # -----------------------------
    # Plot trajectory comparison with frame analysis
    # -----------------------------
    # Extract translations from trajectories
    ee_translations = np.array([pose.translation for pose in ee_traj_world])
    tool_translations = np.array([pose.translation for pose in poses_tool])

    # Extract rotations and compute orientation errors
    ee_rotations = [pose.rotation for pose in ee_traj_world[: len(poses_tool)]]
    tool_rotations = [pose.rotation for pose in poses_tool]

    # Compute position errors in world frame
    position_errors_world = (
        tool_translations - ee_translations[: len(tool_translations)]
    )
    position_error_norms = np.linalg.norm(position_errors_world, axis=1)

    # Compute orientation errors (rotation angle between frames)
    orientation_errors = []
    position_errors_gripper = []  # Errors in gripper frame

    for i in range(len(tool_rotations)):
        # Relative rotation: R_error = R_tool^T @ R_target
        R_diff = tool_rotations[i].T @ ee_rotations[i]

        # Compute angle from rotation matrix
        trace = np.trace(R_diff)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        orientation_errors.append(angle.copy())

        # Transform position error to target gripper frame
        # This shows the error along approach/lateral/binormal directions
        pos_error_gripper = ee_rotations[i].T @ position_errors_world[i]
        position_errors_gripper.append(pos_error_gripper.copy())

    orientation_errors = np.array(orientation_errors)
    position_errors_gripper = np.array(position_errors_gripper)

    # Create time indices
    ee_indices = np.arange(len(ee_traj_world))
    tool_indices = np.arange(len(poses_tool))

    # Create figure with 6 subplots (2 columns)
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

    fig.suptitle(
        "Trajectory Comparison: Frame Alignment Analysis",
        fontsize=16,
        fontweight="bold",
    )

    # Left column: World frame positions
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(ee_indices, ee_translations[:, 0], "b-", label="EE Target", linewidth=2)
    ax0.plot(
        tool_indices, tool_translations[:, 0], "r--", label="Tool Actual", linewidth=2
    )
    ax0.set_ylabel("X Position [m]", fontweight="bold")
    ax0.legend(loc="best")
    ax0.grid(True, alpha=0.3)
    ax0.set_title("World Frame Positions", fontweight="bold")

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(ee_indices, ee_translations[:, 1], "b-", label="EE Target", linewidth=2)
    ax1.plot(
        tool_indices, tool_translations[:, 1], "r--", label="Tool Actual", linewidth=2
    )
    ax1.set_ylabel("Y Position [m]", fontweight="bold")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[2, 0])
    ax2.plot(ee_indices, ee_translations[:, 2], "b-", label="EE Target", linewidth=2)
    ax2.plot(
        tool_indices, tool_translations[:, 2], "r--", label="Tool Actual", linewidth=2
    )
    ax2.set_ylabel("Z Position [m]", fontweight="bold")
    ax2.set_xlabel("Trajectory Point Index", fontweight="bold")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    # Right column: Error analysis
    ax3 = fig.add_subplot(gs[0, 1])
    ax3.plot(
        tool_indices,
        position_errors_world[:, 0],
        "r-",
        label="X Error",
        linewidth=1.5,
        alpha=0.7,
    )
    ax3.plot(
        tool_indices,
        position_errors_world[:, 1],
        "g-",
        label="Y Error",
        linewidth=1.5,
        alpha=0.7,
    )
    ax3.plot(
        tool_indices,
        position_errors_world[:, 2],
        "b-",
        label="Z Error",
        linewidth=1.5,
        alpha=0.7,
    )
    ax3.plot(tool_indices, position_error_norms, "k-", label="Total Error", linewidth=2)
    ax3.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax3.set_ylabel("Position Error [m]", fontweight="bold")
    ax3.legend(loc="best", fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_title("Position Errors (World Frame)", fontweight="bold")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(
        tool_indices,
        position_errors_gripper[:, 0],
        "r-",
        label="Lateral (X)",
        linewidth=2,
    )
    ax4.plot(
        tool_indices,
        position_errors_gripper[:, 1],
        "g-",
        label="Binormal (Y)",
        linewidth=2,
    )
    ax4.plot(
        tool_indices,
        position_errors_gripper[:, 2],
        "b-",
        label="Approach (Z)",
        linewidth=2,
    )
    ax4.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax4.axhline(
        y=0.1034,
        color="blue",
        linestyle=":",
        linewidth=2,
        alpha=0.7,
        label="Expected depth (0.1034m)",
    )
    ax4.axhline(y=-0.1034, color="blue", linestyle=":", linewidth=2, alpha=0.7)
    ax4.set_ylabel("Position Error [m]", fontweight="bold")
    ax4.legend(loc="best", fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_title(
        "Position Errors (Gripper Frame) - Key Plot!",
        fontweight="bold",
        color="darkred",
    )

    # Add statistics for gripper frame errors
    stats_text_gripper = (
        f"Approach (Z): {np.mean(position_errors_gripper[:, 2]):.4f} m (mean)\n"
        f"Lateral (X): {np.mean(np.abs(position_errors_gripper[:, 0])):.4f} m (|mean|)\n"
        f"Binormal (Y): {np.mean(np.abs(position_errors_gripper[:, 1])):.4f} m (|mean|)\n"
        f"Std Approach: {np.std(position_errors_gripper[:, 2]):.4f} m"
    )
    ax4.text(
        0.02,
        0.98,
        stats_text_gripper,
        transform=ax4.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(
        tool_indices,
        np.rad2deg(orientation_errors),
        "purple",
        linewidth=2,
        label="Orientation Error",
    )
    ax5.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax5.set_ylabel("Rotation Error [deg]", fontweight="bold")
    ax5.set_xlabel("Trajectory Point Index", fontweight="bold")
    ax5.legend(loc="best")
    ax5.grid(True, alpha=0.3)
    ax5.set_title("Orientation Errors", fontweight="bold")

    # Add orientation statistics
    stats_text_ori = (
        f"Mean: {np.rad2deg(np.mean(orientation_errors)):.3f}°\n"
        f"Max: {np.rad2deg(np.max(orientation_errors)):.3f}°\n"
        f"Std: {np.rad2deg(np.std(orientation_errors)):.3f}°"
    )
    ax5.text(
        0.02,
        0.98,
        stats_text_ori,
        transform=ax5.transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="plum", alpha=0.8),
    )

    # Save plot to file since we're in a dev container
    output_path = pathlib.Path(
        "/workspaces/object_following_ocp/trajectory_comparison.png"
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")

    # Print comprehensive statistics
    print("\n" + "=" * 70)
    print("FRAME ALIGNMENT ANALYSIS")
    print("=" * 70)

    print("\n1. WORLD FRAME POSITION ERRORS:")
    print(f"   Mean total error: {np.mean(position_error_norms):.4f} m")
    print(f"   Max total error: {np.max(position_error_norms):.4f} m")
    print(f"   RMS error: {np.sqrt(np.mean(position_error_norms**2)):.4f} m")
    print(
        f"   Mean X error: {np.mean(position_errors_world[:, 0]):.4f} m (abs: {np.mean(np.abs(position_errors_world[:, 0])):.4f} m)"
    )
    print(
        f"   Mean Y error: {np.mean(position_errors_world[:, 1]):.4f} m (abs: {np.mean(np.abs(position_errors_world[:, 1])):.4f} m)"
    )
    print(
        f"   Mean Z error: {np.mean(position_errors_world[:, 2]):.4f} m (abs: {np.mean(np.abs(position_errors_world[:, 2])):.4f} m)"
    )

    print("\n2. GRIPPER FRAME POSITION ERRORS (This is the key!):")
    print(
        f"   Approach (Z) - mean: {np.mean(position_errors_gripper[:, 2]):.4f} m, std: {np.std(position_errors_gripper[:, 2]):.4f} m"
    )
    print(
        f"   Lateral (X)  - mean: {np.mean(position_errors_gripper[:, 0]):.4f} m, std: {np.std(position_errors_gripper[:, 0]):.4f} m"
    )
    print(
        f"   Binormal (Y) - mean: {np.mean(position_errors_gripper[:, 1]):.4f} m, std: {np.std(position_errors_gripper[:, 1]):.4f} m"
    )
    print("   Expected gripper depth: 0.1034 m")

    is_approach_aligned = np.abs(np.mean(position_errors_gripper[:, 2]) - 0.1034) < 0.01
    is_lateral_aligned = np.mean(np.abs(position_errors_gripper[:, 0])) < 0.005
    is_binormal_aligned = np.mean(np.abs(position_errors_gripper[:, 1])) < 0.005

    print("\n3. ORIENTATION ERRORS:")
    print(
        f"   Mean rotation error: {np.rad2deg(np.mean(orientation_errors)):.3f} degrees"
    )
    print(
        f"   Max rotation error: {np.rad2deg(np.max(orientation_errors)):.3f} degrees"
    )
    print(
        f"   Std rotation error: {np.rad2deg(np.std(orientation_errors)):.3f} degrees"
    )

    print("\n4. DIAGNOSIS:")
    if is_approach_aligned and is_lateral_aligned and is_binormal_aligned:
        print("   ✓ Frames appear CORRECTLY ALIGNED!")
        print("   ✓ Gripper depth offset is along approach axis as expected")
    elif is_approach_aligned:
        print("   ⚠ Approach axis offset matches gripper depth (0.1034m)")
        print("   ⚠ But lateral/binormal errors exist - possible frame twist or offset")
    else:
        print("   ✗ FRAME MISALIGNMENT DETECTED!")
        print(
            f"   ✗ Approach error ({np.mean(position_errors_gripper[:, 2]):.4f}m) != expected depth (0.1034m)"
        )
        print("   → Check frame definitions between GraspNet and Pinocchio")
        print(
            "   → Verify grasp transform chain (camera→world, object→grasp, grasp→EE)"
        )

    if np.rad2deg(np.mean(orientation_errors)) > 1.0:
        print(
            f"   ⚠ Significant orientation error ({np.rad2deg(np.mean(orientation_errors)):.2f}°)"
        )
        print("   → This will cause the approach axis to point in wrong direction")

    print("=" * 70)

    plt.close()

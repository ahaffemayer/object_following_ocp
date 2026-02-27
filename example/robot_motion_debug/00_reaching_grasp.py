import pathlib

import numpy as np
import pinocchio as pin
from robomeshcat import Object, Robot, Scene

from object_following_ocp.data.data_loader import ConfigLoader, DataLoader
from object_following_ocp.solver.ik_curobo import RobotIKSolver
from object_following_ocp.robot.robot_loader import load_reduced_panda

if __name__ == "__main__":
    object_traj_path = pathlib.Path(
        "/workspaces/object_following_ocp/ressources/json/bowl1.props-dinov2-ffa-22.gpt4_scaled.best_object.poses-dinov2-22-graph.smoothed-movavg.json"
    )
    scale_path = pathlib.Path(
        "/workspaces/object_following_ocp/ressources/grasps_scales.json"
    )

    dataloader = DataLoader(
        object_trajectory_path=object_traj_path,
        scales_path=scale_path,
        load_grasps=True,
    )

    IK_config = ConfigLoader.load(
        pathlib.Path(
            "/workspaces/object_following_ocp/example/robot_motion/configs/ik_config.yml"
        )
    )

    object_trajectory_in_camera_frame = dataloader.to_trajectory_SE3()

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

    object_info = dataloader.object_info
    # -----------------------------
    # Add object to scene
    # -----------------------------
    o = Object.create_mesh(
        path_to_mesh=object_info.mesh_path,
        name="robot/movable_obj",
        texture=object_info.texture_path,
        scale=object_info.scale,
        color=[0.8, 0.8, 0.8],
    )
    scene.add_object(o)

    # -----------------------------
    # Define transformation chain
    # -----------------------------
    # Frame naming:
    # - world: robot base frame
    # - camera: camera frame
    # - object: object frame
    # - grasp: grasp frame from GraspGen
    # - tcp: tool center point (gripper tip)
    # - ee: end-effector frame (with corrected orientation)

    # Transform: camera frame in world frame
    wMcamera = pin.SE3.Identity()
    wMcamera.translation = np.array([0, -1.0, -1.0])

    # Transform: object frame in camera frame (from trajectory data)
    cameraM_object = object_trajectory_in_camera_frame[0]

    # Transform: grasp frame in object frame (from GraspGen)
    objectM_grasp = dataloader.best_grasp_SE3

    # Transform: correction for frame convention mismatch (90° rotation around Z)
    graspM_grasp_corrected = pin.SE3.Identity()
    graspM_grasp_corrected.rotation = pin.exp3(np.array([0, 0, np.deg2rad(90)]))

    # Transform: TCP offset from grasp frame (gripper depth)
    grasp_correctedM_tcp = pin.SE3.Identity()
    gripper_depth = 0.1034
    grasp_correctedM_tcp.translation = np.array([0, 0, gripper_depth])

    # Additional rotation for visualization/camera alignment
    elev_angle_deg = 25
    R_alignment = pin.exp3(np.array([0, 0, np.deg2rad(90)])) @ pin.exp3(
        np.array([-np.pi / 2 - np.deg2rad(elev_angle_deg), 0, 0])
    )
    worldM_world_aligned = pin.SE3(R_alignment, np.array([0, 0, 0]))

    # -----------------------------
    # Compute final transformations
    # -----------------------------
    # Object pose in world frame
    wM_object = worldM_world_aligned * wMcamera * cameraM_object

    # End-effector (TCP) pose in world frame
    wM_tcp = (
        worldM_world_aligned
        * wMcamera
        * cameraM_object
        * objectM_grasp
        * graspM_grasp_corrected
        * grasp_correctedM_tcp
    )

    # Note: we don't include gripper offset here for IK target
    wM_ee = (
        worldM_world_aligned
        * wMcamera
        * cameraM_object
        * objectM_grasp
        * graspM_grasp_corrected
    )

    # -----------------------------
    # Visualize in scene
    # -----------------------------
    scene.add_object(Object.create_sphere(radius=0.01, name="target", color=[0, 0, 1]))
    scene["target"].pos[:] = wM_ee.translation
    o.pose = wM_object.homogeneous

    # -----------------------------
    # Solve IK
    # -----------------------------
    solver = RobotIKSolver(
        robot_name="franka",
        num_seeds=20,
        position_threshold=0.005,
        rotation_threshold=0.05,
        use_cuda_graph=False,
    )

    solution, info = solver.solve(wM_ee)

    if solution is not None:
        print("\nIK Solution found!")
        print(f"Joint configuration: {solution}")
        print(f"Position error: {info['position_error']:.6f} m")
        print(f"Rotation error: {info['rotation_error']:.6f} rad")
        print(f"Solve time: {info['solve_time']:.6f} s")

        # Verify solution with FK from CuRobo
        fk_pos_curobo, fk_quat_curobo = solver.forward_kinematics(solution)
        print("\n" + "=" * 60)
        print("CuRobo FK:")
        print("=" * 60)
        print(f"Position: {fk_pos_curobo}")
        print(f"Quaternion (wxyz): {fk_quat_curobo}")

        # Convert quaternion to rotation matrix for comparison
        # CuRobo uses wxyz convention
        R_curobo = pin.Quaternion(
            float(fk_quat_curobo[0]),  # w
            float(fk_quat_curobo[1]),  # x
            float(fk_quat_curobo[2]),  # y
            float(fk_quat_curobo[3]),  # z
        ).toRotationMatrix()

        # Compute FK with Pinocchio for panda_hand_tcp frame
        pin.framesForwardKinematics(rmodel, rdata, solution)

        # Get the TCP frame index
        tcp_frame_id = rmodel.getFrameId("panda_hand_tcp")
        wM_tcp_pinocchio = rdata.oMf[tcp_frame_id]

        print("\n" + "=" * 60)
        print("Pinocchio FK (panda_hand_tcp frame):")
        print("=" * 60)
        print(f"Position: {wM_tcp_pinocchio.translation}")

        # Convert rotation matrix to quaternion for display
        quat_pinocchio = pin.Quaternion(wM_tcp_pinocchio.rotation)
        print(
            f"Quaternion (wxyz): [{quat_pinocchio.w}, {quat_pinocchio.x}, {quat_pinocchio.y}, {quat_pinocchio.z}]"
        )

        print("\n" + "=" * 60)
        print("Comparison:")
        print("=" * 60)

        # Position difference
        pos_diff = fk_pos_curobo - wM_tcp_pinocchio.translation
        print(f"Position difference (CuRobo - Pinocchio): {pos_diff}")
        print(f"Position error norm: {np.linalg.norm(pos_diff):.6f} m")

        # Rotation difference
        R_diff = R_curobo.T @ wM_tcp_pinocchio.rotation
        angle_diff = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
        print(f"Rotation angle difference: {np.rad2deg(angle_diff):.6f} degrees")

        # Show rotation matrices
        print("\nCuRobo rotation matrix:")
        print(R_curobo)
        print("\nPinocchio rotation matrix:")
        print(wM_tcp_pinocchio.rotation)

        # Compute axis-angle representation of the difference
        if angle_diff > 1e-6:
            axis_angle = pin.log3(R_diff)
            axis = axis_angle / (np.linalg.norm(axis_angle) + 1e-10)
            print(f"\nRotation difference axis: {axis}")
            print(f"Rotation difference angle: {np.rad2deg(angle_diff):.6f} degrees")

        robot[:] = solution
    else:
        print("IK solution not found!")
        print(f"Position error: {info['position_error']:.6f} m")
        print(f"Rotation error: {info['rotation_error']:.6f} rad")

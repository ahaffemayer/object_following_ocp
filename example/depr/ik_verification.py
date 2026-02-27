import yaml
import numpy as np
import trimesh
import pinocchio as pin
from pathlib import Path

from object_following_ocp.grasp_generator import GraspGenerator
from object_following_ocp.robot.robot_loader import (
    load_reduced_panda,
    self_collision_pairs,
)
from robomeshcat import Scene, Object, Robot


# ============================================================
# Math utilities
# ============================================================


def quat_to_rot(qw, qx, qy, qz):
    """Convert quaternion to rotation matrix."""
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)

    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def grasp_to_T(grasp, apply_tcp_offset=False, gripper_depth=0.1034):
    """
    Convert grasp dict to 4x4 transform matrix.

    Args:
        grasp: Dict with 'position' and 'orientation' keys
        apply_tcp_offset: If True, apply TCP offset (gripper depth along z-axis)
        gripper_depth: Distance from gripper base to TCP

    Returns:
        4x4 numpy array representing the transform
    """
    t = np.array(grasp["position"], dtype=np.float64)
    qw = grasp["orientation"]["w"]
    qx, qy, qz = grasp["orientation"]["xyz"]

    T = np.eye(4)
    T[:3, :3] = quat_to_rot(qw, qx, qy, qz)
    T[:3, 3] = t

    if apply_tcp_offset:
        # GraspGen convention: approach direction is along +z
        # Apply offset to get TCP position
        T_offset = np.eye(4)
        T_offset[:3, 3] = [0.0, 0.0, gripper_depth]
        T = T @ T_offset

    return T


def matrix_to_pin_se3(T):
    """Convert 4x4 numpy matrix to Pinocchio SE3."""
    return pin.SE3(T[:3, :3], T[:3, 3])


def pin_se3_to_matrix(se3):
    """Convert Pinocchio SE3 to 4x4 numpy matrix."""
    return se3.homogeneous


# ============================================================
# Validation utilities
# ============================================================


def is_grasp_valid(q, rmodel, cmodel, rdata, cdata):
    """Check if a joint configuration is valid (within limits and collision-free)."""
    if q is None:
        return False

    if np.any(q < rmodel.lowerPositionLimit) or np.any(q > rmodel.upperPositionLimit):
        return False

    pin.updateGeometryPlacements(rmodel, rdata, cmodel, cdata, q)
    has_collision = pin.computeCollisions(cmodel, cdata, False)
    return not has_collision


# ============================================================
# Main
# ============================================================


def main():
    # Configuration
    grasp_yaml = "/home/arthur/Desktop/Projects/PAMI/object_following_ocp/results/bowl1/bowl1_grasps_filtered.yml"
    object_mesh_path = "/home/arthur/Desktop/Projects/PAMI/object_following_ocp/ressources/video_data/d02/bowl1/cbb0cdd9bbcc4fdfa2e16db1db4cda61/cbb0cdd9bbcc4fdfa2e16db1db4cda61.obj"
    gripper_depth = 0.1034
    mesh_scale = 0.0681

    # HARDCODED OBJECT POSE IN WORLD FRAME
    # This is where you want the object to be in the robot's workspace
    object_position = np.array([0.3, -0.0, 0.7])  # x, y, z in meters
    object_orientation = pin.utils.rpyToMatrix(0.0, 0.0, np.pi / 4)  # Roll, pitch, yaw

    T_world_object = np.eye(4)
    T_world_object[:3, :3] = object_orientation
    T_world_object[:3, 3] = object_position

    print("=" * 60)
    print("Grasp IK Visualization Example")
    print("=" * 60)
    print(f"\nObject pose in world frame:")
    print(f"  Position: {object_position}")
    print(f"  Orientation (RPY): [0.0, 0.0, {np.pi / 4:.3f}] rad = [0, 0, 45] deg")

    # Number of grasps to process (set to a small number for testing)
    num_grasps_to_process = 5

    # ========================================
    # 1. Load robot model
    # ========================================
    print("\n[1/4] Loading robot model...")
    rmodel, cmodel, vmodel = load_reduced_panda()

    # Add self-collision pairs
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

    # ========================================
    # 2. Setup RoboMeshcat scene
    # ========================================
    print("[2/4] Setting up RoboMeshcat scene...")

    # Create robot for visualization
    robot = Robot(
        pinocchio_model=rmodel,
        pinocchio_data=rdata,
        pinocchio_geometry_model=vmodel,
        pinocchio_geometry_data=vdata,
    )

    scene = Scene()
    scene.add_robot(robot)

    # Add object mesh at specified world pose
    obj = Object.create_mesh(
        path_to_mesh=object_mesh_path,
        name="robot/object_main",
        scale=mesh_scale,
        color=[0.8, 0.8, 0.8],  # Gray
    )
    scene.add_object(obj)
    obj.pose = T_world_object

    # Add object frame visualization
    frame_size = 0.15
    frame_colors = {
        "x": [1.0, 0.0, 0.0],  # Red
        "y": [0.0, 1.0, 0.0],  # Green
        "z": [0.0, 0.0, 1.0],  # Blue
    }

    for axis, color in frame_colors.items():
        axis_obj = Object.create_cylinder(
            radius=0.005,
            length=frame_size,
            name=f"robot/object_frame_{axis}",
            color=color,
        )
        scene.add_object(axis_obj)

        # Create transform for each axis
        T_axis = T_world_object.copy()
        if axis == "x":
            # Rotate around z to point along x
            R_adjust = pin.utils.rpyToMatrix(0, np.pi / 2, 0)
            T_axis[:3, :3] = T_world_object[:3, :3] @ R_adjust
            T_axis[:3, 3] = T_world_object[:3, 3] + T_world_object[:3, :3] @ np.array(
                [frame_size / 2, 0, 0]
            )
        elif axis == "y":
            # Rotate around z to point along y
            R_adjust = pin.utils.rpyToMatrix(-np.pi / 2, 0, 0)
            T_axis[:3, :3] = T_world_object[:3, :3] @ R_adjust
            T_axis[:3, 3] = T_world_object[:3, 3] + T_world_object[:3, :3] @ np.array(
                [0, frame_size / 2, 0]
            )
        elif axis == "z":
            # Already pointing along z
            T_axis[:3, 3] = T_world_object[:3, 3] + T_world_object[:3, :3] @ np.array(
                [0, 0, frame_size / 2]
            )

        axis_obj.pose = T_axis

    # ========================================
    # 3. Load grasps from YAML
    # ========================================
    print("[3/4] Loading grasps from YAML...")
    with open(grasp_yaml, "r") as f:
        data = yaml.safe_load(f)

    grasp_items = list(data["grasps"].items())[:num_grasps_to_process]
    print(f"Processing {len(grasp_items)} grasps")

    # ========================================
    # 4. Process each grasp
    # ========================================
    print("[4/4] Processing grasps and solving IK...")

    successful_grasps = []

    for idx, (grasp_key, grasp) in enumerate(grasp_items):
        print(f"\n  Grasp {idx + 1}/{len(grasp_items)} (key: {grasp_key})")

        # Convert YAML grasp to transform matrices (these are in object frame)
        T_object_base = grasp_to_T(
            grasp, apply_tcp_offset=False, gripper_depth=gripper_depth
        )
        T_object_tcp = grasp_to_T(
            grasp, apply_tcp_offset=True, gripper_depth=gripper_depth
        )

        # Apply 90° Z-rotation to correct frame convention mismatch between GraspGen and robot TCP
        T_correction = np.eye(4)
        T_correction[:3, :3] = pin.utils.rpyToMatrix(
            0.0, 0.0, np.pi / 2
        )  # 90° around Z

        T_object_base_corrected = T_object_base @ T_correction
        T_object_tcp_corrected = T_object_tcp @ T_correction

        # Transform grasps from object frame to world frame
        T_world_base = T_world_object @ T_object_base_corrected
        T_world_tcp = T_world_object @ T_object_tcp_corrected

        print(f"    Object frame - Base: {T_object_base[:3, 3]}")
        print(f"    Object frame - TCP:  {T_object_tcp[:3, 3]}")
        print(f"    World frame - Base:  {T_world_base[:3, 3]}")
        print(f"    World frame - TCP:   {T_world_tcp[:3, 3]}")

        # Visualize the grasp poses in world frame with spheres
        # Red sphere for base
        base_sphere = Object.create_sphere(
            radius=0.006,
            name=f"robot/grasp_{grasp_key}_base",
            color=[1.0, 0.0, 0.0],  # Red
        )
        scene.add_object(base_sphere)
        base_sphere.pos[:] = T_world_base[:3, 3]

        # Green sphere for TCP
        tcp_sphere = Object.create_sphere(
            radius=0.008,
            name=f"robot/grasp_{grasp_key}_tcp",
            color=[0.0, 1.0, 0.0],  # Green
        )
        scene.add_object(tcp_sphere)
        tcp_sphere.pos[:] = T_world_tcp[:3, 3]

        # Convert TCP transform to Pinocchio SE3 (in world frame)
        tcp_se3 = matrix_to_pin_se3(T_world_tcp)

        # Create grasp generator and solve IK
        print(f"    Solving IK...")
        grasp_generator = GraspGenerator(
            obj_pose=tcp_se3,
            grasp_configurations_number=3,  # Try to find 3 IK solutions
            max_attempts=50,
        )

        try:
            ik_solutions = grasp_generator.generate_grasps_configurations()
            print(f"    Found {len(ik_solutions)} IK solutions")

            # Check validity and take first valid solution
            valid_q = None
            for q_candidate in ik_solutions:
                if is_grasp_valid(q_candidate, rmodel, cmodel, rdata, cdata):
                    valid_q = q_candidate
                    break

            if valid_q is not None:
                print(f"    ✓ Valid IK solution found!")
                successful_grasps.append(
                    {
                        "key": grasp_key,
                        "idx": idx,
                        "T_world_base": T_world_base,
                        "T_world_tcp": T_world_tcp,
                        "q": valid_q,
                    }
                )
            else:
                print(f"    ✗ No valid collision-free solution")

        except Exception as e:
            print(f"    ✗ IK failed: {e}")

    # ========================================
    # 5. Visualize successful IK solutions
    # ========================================
    print(f"\n[5/5] Visualizing {len(successful_grasps)} successful grasps...")

    for grasp_info in successful_grasps:
        grasp_key = grasp_info["key"]
        q = grasp_info["q"]
        T_world_tcp = grasp_info["T_world_tcp"]

        # Update robot configuration
        robot[:] = q
        pin.forwardKinematics(rmodel, rdata, q)
        pin.updateFramePlacements(rmodel, rdata)

        # Get actual end-effector pose
        ee_frame_id = rmodel.getFrameId("panda_hand_tcp")
        ee_pose = rdata.oMf[ee_frame_id]
        ee_matrix = pin_se3_to_matrix(ee_pose)

        # Visualize achieved end-effector position with blue sphere
        ee_sphere = Object.create_sphere(
            radius=0.01,
            name=f"robot/grasp_{grasp_key}_ee_achieved",
            color=[0.0, 0.0, 1.0],  # Blue
        )
        scene.add_object(ee_sphere)
        ee_sphere.pos[:] = ee_pose.translation

        # Add a semi-transparent copy of the object at the grasp position
        obj_ghost = Object.create_mesh(
            path_to_mesh=object_mesh_path,
            name=f"robot/object_ghost_{grasp_key}",
            scale=mesh_scale,
            color=[0.3, 0.8, 0.3],  # Light green
        )
        scene.add_object(obj_ghost)
        obj_ghost.pose = T_world_object

        error = np.linalg.norm(ee_pose.translation - T_world_tcp[:3, 3])
        print(f"  Grasp {grasp_key}:")
        print(f"    Target TCP (world):   {T_world_tcp[:3, 3]}")
        print(f"    Achieved EE (world):  {ee_pose.translation}")
        print(f"    Error: {error:.4f} m")

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print(f"SUMMARY:")
    print(f"  Total grasps processed: {len(grasp_items)}")
    print(f"  Successful IK solutions: {len(successful_grasps)}")
    print(f"  Success rate: {len(successful_grasps) / len(grasp_items) * 100:.1f}%")
    print("=" * 60)
    print("\nVisualization legend:")
    print("  ⚪ Gray object = Object at specified world pose")
    print("  🟢 Light green objects = Ghost copies at same pose")
    print("  📍 RGB cylinders = Object frame in world")
    print("  🔴 Red spheres = Gripper base (GraspGen output in world)")
    print("  🟢 Green spheres = TCP target (used for IK in world)")
    print("  🔵 Blue spheres = Achieved end-effector position from IK")
    print(f"\nObject is positioned at: {object_position}")
    print("All grasp poses are transformed from object frame to world frame.")
    print("The robot is shown in the configuration of the last successful grasp.")
    print("=" * 60)


if __name__ == "__main__":
    main()

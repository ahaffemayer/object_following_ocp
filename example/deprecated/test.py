"""
Visualize object trajectories for a full camera position grid, BEFORE running
the grid search. Use this to calibrate half-extents and step counts.

The grid is automatically centred on the camera translation that places
the trajectory's mean at a desired world-frame target position.

Type new grid parameters (half_extents + steps) and instantly see all
trajectories update in Meshcat.

Usage:
    python explore_grid.py <mesh_id>
"""

import pathlib
import sys

import numpy as np
from robomeshcat import Object, Robot, Scene

from object_following_ocp.data.data_loader import ConfigLoader, DataLoader
from object_following_ocp.geom.grasp_transforms import (
    GraspTransformChain,
    GraspTransformConfig,
)
from object_following_ocp.robot.robot_loader import load_reduced_panda

RESOURCES_DIR = pathlib.Path("/workspaces/object_following_ocp/ressources")
CONFIG_PATH = pathlib.Path(
    "/workspaces/object_following_ocp/example/robot_motion/configs/ocp_config.yml"
)

COLORS = [
    [0.95, 0.26, 0.21],  # red
    [0.13, 0.59, 0.95],  # blue
    [0.30, 0.69, 0.31],  # green
    [1.00, 0.76, 0.03],  # yellow
    [0.61, 0.15, 0.69],  # purple
    [1.00, 0.60, 0.00],  # orange
    [0.00, 0.74, 0.83],  # cyan
    [0.96, 0.26, 0.62],  # pink
    [0.47, 0.33, 0.28],  # brown
    [0.38, 0.49, 0.55],  # blue-grey
]


def compute_reference_camera(
    object_traj_camera,
    robot_config,
    target_position,
    grasp_correction_angle_deg=90.0,
    elevation_angle_deg=25.0,
):
    """
    Find the camera translation that places the trajectory's mean
    at target_position in the world frame.

    avg(cam) = R_align @ cam + avg_at_origin
    => cam = R_align^T @ (target_position - avg_at_origin)
    """
    zero_config = GraspTransformConfig.from_robot_config(
        robot_config=robot_config,
        camera_translation=np.zeros(3),
        grasp_correction_angle_deg=grasp_correction_angle_deg,
        elevation_angle_deg=elevation_angle_deg,
    )
    chain = GraspTransformChain(zero_config)

    traj_at_origin = chain.transform_object_trajectory(object_traj_camera)
    avg_at_origin = np.mean([pose.translation for pose in traj_at_origin.poses], axis=0)

    R_align = chain.worldM_world_aligned.rotation
    ref_cam = R_align.T @ (target_position - avg_at_origin)
    return ref_cam


def make_grid_around_center(center, half_extents, steps):
    """Build a grid of camera positions centred on ``center``."""
    dx, dy, dz = half_extents
    nx, ny, nz = steps
    xs = np.linspace(center[0] - dx, center[0] + dx, nx)
    ys = np.linspace(center[1] - dy, center[1] + dy, ny)
    zs = np.linspace(center[2] - dz, center[2] + dz, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])


def compute_display_ref(ref_cam, object_traj_camera, robot_config):
    """Compute the mean world-frame translation for display centring."""
    grasp_config = GraspTransformConfig.from_robot_config(
        robot_config=robot_config,
        camera_translation=ref_cam,
        grasp_correction_angle_deg=90.0,
        elevation_angle_deg=25.0,
    )
    chain = GraspTransformChain(grasp_config)
    traj_world = chain.transform_object_trajectory(object_traj_camera)
    translations = np.array([pose.translation for pose in traj_world.poses])
    return np.mean(translations, axis=0)


def clear_dots(scene, dot_registry):
    """Remove all dots currently in the scene."""
    for name in dot_registry:
        try:
            scene.remove_object(name)
        except Exception:
            pass
    dot_registry.clear()


def draw_grid(
    scene,
    dot_registry,
    camera_positions,
    object_traj_camera,
    robot_config,
    display_ref,
    dot_radius=0.005,
):
    """Draw one coloured dot trail per camera position."""
    clear_dots(scene, dot_registry)

    n_traj = len(camera_positions)
    n_poses = len(object_traj_camera)

    for i, cam in enumerate(camera_positions):
        color = COLORS[i % len(COLORS)]

        grasp_config = GraspTransformConfig.from_robot_config(
            robot_config=robot_config,
            camera_translation=cam,
            grasp_correction_angle_deg=90.0,
            elevation_angle_deg=25.0,
        )
        chain = GraspTransformChain(grasp_config)
        traj_world = chain.transform_object_trajectory(object_traj_camera)

        for k, pose in enumerate(traj_world.poses):
            name = f"g{i}_p{k}"
            dot_color = COLORS[0] if k == 0 else color
            dot = Object.create_sphere(radius=dot_radius, name=name, color=dot_color)
            scene.add_object(dot)
            scene[name].pos[:] = pose.translation - display_ref
            dot_registry.append(name)

    print(
        f"  Showing {n_traj} trajectories × {n_poses} poses = {n_traj * n_poses} dots"
    )


def parse_grid_params(raw):
    """Parse 'dx dy dz nx ny nz' from a string (half-extents + steps)."""
    parts = raw.split()
    if len(parts) != 6:
        raise ValueError("Need exactly 6 values: dx dy dz nx ny nz")
    dx, dy, dz = float(parts[0]), float(parts[1]), float(parts[2])
    nx, ny, nz = int(parts[3]), int(parts[4]), int(parts[5])
    if nx < 1 or ny < 1 or nz < 1:
        raise ValueError("nx, ny, nz must be >= 1")
    return (dx, dy, dz), (nx, ny, nz)


def main():
    if len(sys.argv) < 2:
        print("Usage: python explore_grid.py <mesh_id>")
        sys.exit(1)

    mesh_id = sys.argv[1]
    matches = list((RESOURCES_DIR / "json").glob(f"{mesh_id}*.json"))
    if not matches:
        print(f"No trajectory JSON found for mesh_id '{mesh_id}'")
        sys.exit(1)
    traj_path = matches[0]

    dataloader = DataLoader(
        object_trajectory_path=traj_path,
        scales_path=RESOURCES_DIR / "grasps_scales.json",
        load_grasps=False,
    )
    robot_config = ConfigLoader.load(CONFIG_PATH)
    object_traj_camera = dataloader.to_trajectory_SE3()

    # Target position in world frame (adjust as needed)
    target_position = np.array([0.0, 0.0, 0.0])

    # Compute reference camera that centres the trajectory at target
    ref_cam = compute_reference_camera(
        object_traj_camera, robot_config, target_position
    )
    print(f"\nReference camera (auto-computed): {np.round(ref_cam, 4)}")
    print(f"  (places trajectory mean at {target_position} in world frame)")

    # Display reference for Meshcat centring
    display_ref = compute_display_ref(ref_cam, object_traj_camera, robot_config)
    print(f"Display reference (world frame): {np.round(display_ref, 4)}")

    # Setup scene
    rmodel, cmodel, vmodel = load_reduced_panda()
    rdata = rmodel.createData()
    scene = Scene()
    robot = Robot(
        pinocchio_model=rmodel,
        pinocchio_data=rdata,
        pinocchio_geometry_model=vmodel,
        pinocchio_geometry_data=vmodel.createData(),
    )

    dot_registry = []

    # Default half-extents and steps
    half_extents = (0.5, 0.5, 0.5)
    steps = (3, 3, 2)

    print(f"\nLoaded '{mesh_id}'  —  {len(object_traj_camera)} poses")
    print("Open Meshcat at http://127.0.0.1:7000/static/\n")
    print("Grid is auto-centred. Enter half-extents + steps:")
    print("  dx dy dz  nx ny nz")
    print("Default: 0.2 0.2 0.2  3 3 2\n")

    camera_positions = make_grid_around_center(ref_cam, half_extents, steps)
    draw_grid(
        scene,
        dot_registry,
        camera_positions,
        object_traj_camera,
        robot_config,
        display_ref,
    )
    print(
        f"  → {len(camera_positions)} camera positions, "
        f"centred on ref_cam={np.round(ref_cam, 3)}"
    )

    while True:
        try:
            raw = input("\ngrid > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not raw:
            break

        try:
            half_extents, steps = parse_grid_params(raw)
        except ValueError as e:
            print(f"  {e}")
            continue

        camera_positions = make_grid_around_center(ref_cam, half_extents, steps)
        draw_grid(
            scene,
            dot_registry,
            camera_positions,
            object_traj_camera,
            robot_config,
            display_ref,
        )
        dx, dy, dz = half_extents
        nx, ny, nz = steps
        print(
            f"  ±({dx:.3f}, {dy:.3f}, {dz:.3f})  "
            f"steps=({nx}, {ny}, {nz})  "
            f"→ {len(camera_positions)} positions"
        )

    dx, dy, dz = half_extents
    nx, ny, nz = steps
    print("\nFinal grid parameters:")
    print(f"  TARGET_POSITION = {target_position.tolist()}")
    print(f"  HALF_EXTENTS = ({dx}, {dy}, {dz})")
    print(f"  STEPS = ({nx}, {ny}, {nz})")
    print(f"  (ref_cam = {np.round(ref_cam, 4).tolist()})")
    print("\nCopy TARGET_POSITION, HALF_EXTENTS, STEPS into main_grid_search.py")


if __name__ == "__main__":
    main()

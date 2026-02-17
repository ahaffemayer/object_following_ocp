"""
Visualize object trajectories for a full camera position grid, BEFORE running
the grid search. Use this to calibrate X/Y/Z ranges and step counts.

Type new grid parameters and instantly see all trajectories update in Meshcat.

Usage:
    python explore_grid.py <mesh_id>
"""

import pathlib
import sys

import numpy as np
from robomeshcat import Object, Robot, Scene

from object_following_ocp.data_loader import ConfigLoader, DataLoader
from object_following_ocp.grasp_transforms import (
    GraspTransformChain,
    GraspTransformConfig,
)
from object_following_ocp.robot_loader import load_reduced_panda

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


def make_grid(xmin, xmax, ymin, ymax, zmin, zmax, nx, ny, nz):
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    zs = np.linspace(zmin, zmax, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])


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
    dot_radius=0.005,
):
    """Draw one colored dot trail per camera position."""
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
            dot = Object.create_sphere(radius=dot_radius, name=name, color=color)
            scene.add_object(dot)
            scene[name].pos[:] = pose.translation
            dot_registry.append(name)

    print(
        f"  Showing {n_traj} trajectories × {n_poses} poses = {n_traj * n_poses} dots"
    )


def parse_grid_params(raw):
    """Parse 'xmin xmax nx  ymin ymax ny  zmin zmax nz' from a string."""
    parts = raw.split()
    if len(parts) != 9:
        raise ValueError(
            "Need exactly 9 values: xmin xmax nx  ymin ymax ny  zmin zmax nz"
        )
    xmin, xmax = float(parts[0]), float(parts[1])
    nx = int(parts[2])
    ymin, ymax = float(parts[3]), float(parts[4])
    ny = int(parts[5])
    zmin, zmax = float(parts[6]), float(parts[7])
    nz = int(parts[8])
    if nx < 1 or ny < 1 or nz < 1:
        raise ValueError("nx, ny, nz must be >= 1")
    return xmin, xmax, ymin, ymax, zmin, zmax, nx, ny, nz


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
    scene.add_robot(robot=robot)

    dot_registry = []

    # Defaults
    params = (-0.2, 0.2, -0.9, -0.5, -1.2, -0.8, 3, 3, 2)
    print(f"\nLoaded '{mesh_id}'  —  {len(object_traj_camera)} poses")
    print("Open Meshcat at http://127.0.0.1:7000/static/\n")
    print("Enter grid as:  xmin xmax nx  ymin ymax ny  zmin zmax nz")
    print("Press Enter with no input to quit.\n")
    print("Default: -0.2 0.2 3  -0.9 -0.5 3  -1.2 -0.8 2")

    camera_positions = make_grid(*params)
    draw_grid(scene, dot_registry, camera_positions, object_traj_camera, robot_config)
    print(
        f"Grid: {params[:6]}  steps: {params[6:]}  →  {len(camera_positions)} positions"
    )

    while True:
        try:
            raw = input("\ngrid > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not raw:
            break

        try:
            params = parse_grid_params(raw)
        except ValueError as e:
            print(f"  {e}")
            continue

        camera_positions = make_grid(*params)
        draw_grid(
            scene, dot_registry, camera_positions, object_traj_camera, robot_config
        )
        xmin, xmax, ymin, ymax, zmin, zmax, nx, ny, nz = params
        print(
            f"  X:[{xmin}, {xmax}] nx={nx}  "
            f"Y:[{ymin}, {ymax}] ny={ny}  "
            f"Z:[{zmin}, {zmax}] nz={nz}  "
            f"→ {len(camera_positions)} positions"
        )

    print("\nFinal grid parameters:")
    xmin, xmax, ymin, ymax, zmin, zmax, nx, ny, nz = params
    print(f"  X_RANGE = ({xmin}, {xmax}, {nx})")
    print(f"  Y_RANGE = ({ymin}, {ymax}, {ny})")
    print(f"  Z_RANGE = ({zmin}, {zmax}, {nz})")
    print("\nCopy these into main_grid_search.py")


if __name__ == "__main__":
    main()

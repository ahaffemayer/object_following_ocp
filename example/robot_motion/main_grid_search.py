"""
Main script for camera position grid search and trajectory review.

Workflow:
1. Compute reference camera that centres the trajectory at the robot
2. Grid search ± half-extents around that reference
3. For each position, test IK on first pose
4. If successful, solve IK for entire trajectory
5. Store all valid trajectories
6. For each valid trajectory, solve OCP, animate, and auto-save
"""

import pathlib

import numpy as np
from robomeshcat import Object, Robot, Scene

from object_following_ocp.data.data_loader import ConfigLoader, DataLoader
from object_following_ocp.geom.camera_grid_search import CameraPositionGridSearch
from object_following_ocp.robot.robot_loader import load_reduced_panda
from object_following_ocp.visualizer.trajectory_reviewer import TrajectoryReviewer
from object_following_ocp.visualizer.trajectory_visualizer import TrajectoryVisualizer

if __name__ == "__main__":
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    traj_name = "howto100m_AYkkIu7RArQ_0.props-sam3d.gpt4_scaled.best_object.megapose-ref-3-sym.smoothed-savgol.filtered-iou-0.2"
    object_traj_path = pathlib.Path(
        f"/workspaces/object_following_ocp/ressources/json/{traj_name}.json"
    )
    scale_path = pathlib.Path(
        "/workspaces/object_following_ocp/ressources/grasps_scales.json"
    )
    config_path = pathlib.Path(
        "/workspaces/object_following_ocp/example/robot_motion/configs/ocp_config_panda.yml"
    )

    output_dir = pathlib.Path("/mnt/user-data/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Grid search parameters (relative, not absolute) -----------------
    # Where should the trajectory's average land in world frame?
    # E.g. [0, 0, 0] for origin, or [0, 0, 0.7] for table height.
    TARGET_POSITION = np.array([0.0, 0.0, 0.7])

    # How far to search around the auto-computed reference camera (± per axis)
    HALF_EXTENTS = (0.2, 0.2, 0.2)

    # Number of grid samples per axis
    STEPS = (3, 3, 3)

    # ========================================================================
    # SETUP
    # ========================================================================

    print("=" * 70)
    print("CAMERA POSITION GRID SEARCH & TRAJECTORY REVIEW")
    print("=" * 70)

    dataloader = DataLoader(
        object_trajectory_path=object_traj_path,
        scales_path=scale_path,
        load_grasps=True,
    )
    robot_config = ConfigLoader.load(config_path)

    print(f"\nLoaded object: {dataloader.object_info.mesh_id}")
    print(f"Trajectory length: {len(dataloader.poses)} poses")
    print(
        f"Best grasp: {dataloader.best_grasp.name if dataloader.has_grasps else 'None'}"
    )
    print(f"Gripper depth: {robot_config.gripper_depth} m")

    mesh_id = dataloader.object_info.mesh_id
    grid_search_results_path = output_dir / f"{traj_name}_grid_search_results.pkl"
    kept_trajectories_path = output_dir / f"{traj_name}_kept_trajectories.pkl"

    print(f"\nOutput: {grid_search_results_path.name}, {kept_trajectories_path.name}")

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

    movable_object = Object.create_mesh(
        path_to_mesh=dataloader.object_info.mesh_path,
        name="robot/movable_obj",
        texture=dataloader.object_info.texture_path,
        scale=dataloader.object_info.scale,
        color=[0.8, 0.8, 0.8],
    )
    scene.add_object(movable_object)

    # ========================================================================
    # STEP 1: Grid Search (auto-centred)
    # ========================================================================

    print("\n" + "=" * 70)
    print("STEP 1: Grid Search")
    print("=" * 70)

    grid_search = CameraPositionGridSearch(
        dataloader=dataloader,
        robot_config=robot_config,
        object_traj_path=object_traj_path,
        scale_path=scale_path,
        config_path=config_path,
        grasp_correction_angle_deg=90.0,
        elevation_angle_deg=25.0,
        verbose=True,
    )

    valid_trajectories = grid_search.run_grid_search(
        target_position=TARGET_POSITION,
        half_extents=HALF_EXTENTS,
        steps=STEPS,
    )

    summary = grid_search.get_summary()
    print("\nGrid Search Summary:")
    print(f"  Valid trajectories found: {summary['num_valid']}")
    if summary["num_valid"] > 0:
        print(f"  Best IK success rate:  {summary['best_success_rate']:.1f}%")
        print(f"  Worst IK success rate: {summary['worst_success_rate']:.1f}%")
        print(f"  Mean IK success rate:  {summary['mean_success_rate']:.1f}%")

    grid_search.save_results(grid_search_results_path)

    if len(valid_trajectories) == 0:
        print(
            "\n⚠ No valid trajectories found. Try adjusting half_extents or target_position."
        )
        exit(0)

    # ========================================================================
    # STEP 2: OCP + Visualization (auto-save all)
    # ========================================================================

    print("\n" + "=" * 70)
    print("STEP 2: OCP Solving & Visualization")
    print("=" * 70)

    visualizer = TrajectoryVisualizer(
        scene=scene,
        robot=robot,
        movable_object=movable_object,
    )

    reviewer = TrajectoryReviewer(
        dataloader=dataloader,
        robot_config=robot_config,
        visualizer=visualizer,
        rmodel=rmodel,
        cmodel=cmodel,
        object_traj_path=object_traj_path,
        scale_path=scale_path,
        config_path=config_path,
        grasp_correction_angle_deg=90.0,
        elevation_angle_deg=25.0,
    )

    kept_trajectories = reviewer.review_all_candidates(valid_trajectories)

    if kept_trajectories:
        reviewer.save_kept_trajectories(kept_trajectories_path)
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        for idx, traj in enumerate(kept_trajectories):
            print(
                f"  [{idx + 1}] camera={traj.camera_translation}  "
                f"IK={traj.ik_success_rate:.1f}%"
            )

    # ========================================================================
    # COMPLETE
    # ========================================================================

    print("\n" + "=" * 70)
    print("COMPLETE")
    print(f"  Grid search: {grid_search_results_path}")
    if kept_trajectories:
        print(f"  Trajectories: {kept_trajectories_path}")
    print("=" * 70)

"""
Main script for camera position grid search and trajectory review.

Workflow:
1. Grid search over camera positions
2. For each position, test IK on first pose
3. If successful, solve IK for entire trajectory
4. Store all valid trajectories
5. For each valid trajectory:
   - Solve OCP using IK as warm start
   - Visualize OCP solution
   - Ask user to keep or ditch
6. Save all kept trajectories
"""

import pathlib

from camera_grid_search import CameraPositionGridSearch
from robomeshcat import Object, Robot, Scene
from trajectory_reviewer import TrajectoryReviewer
from trajectory_visualizer import TrajectoryVisualizer

from object_following_ocp.data_loader import ConfigLoader, DataLoader
from object_following_ocp.robot_loader import load_reduced_panda

if __name__ == "__main__":
    # ========================================================================
    # CONFIGURATION
    # ========================================================================

    # Data paths
    object_traj_path = pathlib.Path(
        "/workspaces/object_following_ocp/ressources/json/bowl1.props-dinov2-ffa-22.gpt4_scaled.best_object.poses-dinov2-22-graph.smoothed-movavg.json"
    )
    scale_path = pathlib.Path(
        "/workspaces/object_following_ocp/ressources/grasps_scales.json"
    )
    config_path = pathlib.Path(
        "/workspaces/object_following_ocp/example/robot_motion/configs/ocp_config.yml"
    )

    # Output directory
    output_dir = pathlib.Path("/mnt/user-data/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Grid search parameters
    # Define ranges for camera position: (min, max, num_points)
    X_RANGE = (-0.5, 0.5, 4)  # X: -0.5 to 0.5 with 5 points
    Y_RANGE = (-1.0, -0.4, 3)  # Y: -1.0 to -0.4 with 4 points
    Z_RANGE = (-1.2, -0.8, 2)  # Z: -1.2 to -0.8 with 3 points

    # Total grid points: 5 * 4 * 3 = 60 positions

    # ========================================================================
    # SETUP: Load data and robot
    # ========================================================================

    print("=" * 70)
    print("CAMERA POSITION GRID SEARCH & TRAJECTORY REVIEW")
    print("=" * 70)

    # Load data
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

    # Create output filenames with mesh_id
    mesh_id = dataloader.object_info.mesh_id
    grid_search_results_path = output_dir / f"{mesh_id}_grid_search_results.pkl"
    kept_trajectories_path = output_dir / f"{mesh_id}_kept_trajectories.pkl"

    print("\nOutput files will be:")
    print(f"  Grid search: {grid_search_results_path.name}")
    print(f"  Kept trajectories: {kept_trajectories_path.name}")

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
    # STEP 1: Grid Search over Camera Positions
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

    # Run grid search
    valid_trajectories = grid_search.run_grid_search(
        x_range=X_RANGE,
        y_range=Y_RANGE,
        z_range=Z_RANGE,
    )

    # Print summary
    summary = grid_search.get_summary()
    print("\nGrid Search Summary:")
    print(f"  Valid trajectories found: {summary['num_valid']}")
    if summary["num_valid"] > 0:
        print(f"  Best IK success rate: {summary['best_success_rate']:.1f}%")
        print(f"  Worst IK success rate: {summary['worst_success_rate']:.1f}%")
        print(f"  Mean IK success rate: {summary['mean_success_rate']:.1f}%")

    # Save grid search results
    grid_search.save_results(grid_search_results_path)

    if len(valid_trajectories) == 0:
        print("\n⚠ No valid trajectories found. Try adjusting grid search parameters.")
        exit(0)

    # ========================================================================
    # STEP 2: Review Trajectories with OCP
    # ========================================================================

    print("\n" + "=" * 70)
    print("STEP 2: Review Trajectories")
    print("=" * 70)
    print(f"\nYou will now review {len(valid_trajectories)} trajectories.")
    print("For each trajectory:")
    print("  1. OCP will be solved")
    print("  2. Solution will be displayed")
    print("  3. You decide to keep or ditch it")
    print("\nPress Enter to continue...")
    input()

    # Create visualizer
    visualizer = TrajectoryVisualizer(
        scene=scene,
        robot=robot,
        movable_object=movable_object,
    )

    # Create reviewer
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

    # Review all candidates
    kept_trajectories = reviewer.review_all_candidates(valid_trajectories)

    # Save kept trajectories
    if len(kept_trajectories) > 0:
        reviewer.save_kept_trajectories(kept_trajectories_path)

        print("\n" + "=" * 70)
        print("KEPT TRAJECTORIES SUMMARY")
        print("=" * 70)
        for idx, traj in enumerate(kept_trajectories):
            print(f"\nTrajectory {idx + 1}:")
            print(f"  Camera position: {traj.camera_translation}")
            print(f"  IK success rate: {traj.ik_success_rate:.1f}%")
            if traj.user_notes:
                print(f"  Notes: {traj.user_notes}")
    else:
        print("\n⚠ No trajectories were kept.")

    # ========================================================================
    # COMPLETE
    # ========================================================================

    print("\n" + "=" * 70)
    print("GRID SEARCH AND REVIEW COMPLETE!")
    print("=" * 70)
    print("\nResults saved:")
    print(f"  Grid search: {grid_search_results_path}")
    if len(kept_trajectories) > 0:
        print(f"  Kept trajectories: {kept_trajectories_path}")
    print("\n" + "=" * 70)

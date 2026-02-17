"""
Solve OCP for Kinova using saved Panda TCP trajectories.
Uses inverse dynamics (faster/more stable than forward dynamics).

Usage:
    python kinova_ocp_simple.py <mesh_id>
"""

import pathlib
import pickle
import sys

import crocoddyl
import example_robot_data
import numpy as np
import pinocchio as pin
from robomeshcat import Object, Robot, Scene

from object_following_ocp.data_loader import DataLoader

OUTPUT_DIR = pathlib.Path("/mnt/user-data/outputs")


def main():
    if len(sys.argv) < 2:
        print("Usage: python kinova_ocp_simple.py <mesh_id>")
        sys.exit(1)

    mesh_id = sys.argv[1]
    filepath = OUTPUT_DIR / f"{mesh_id}_kept_trajectories.pkl"

    if not filepath.exists():
        print(f"File not found: {filepath}")
        sys.exit(1)

    # Load Panda trajectories
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    trajectories = data["trajectories"]
    print(f"Loaded {len(trajectories)} Panda trajectories for '{mesh_id}'")

    # Load Kinova
    kinova = example_robot_data.load("kinova")
    rmodel = kinova.model
    state = crocoddyl.StateMultibody(rmodel)
    actuation = crocoddyl.ActuationModelFull(state)

    # Better initial state: use reference configuration
    q0 = rmodel.referenceConfigurations["arm_up"]
    x0 = np.concatenate([q0, np.zeros(rmodel.nv)])

    print(f"Loaded Kinova: nq={rmodel.nq}, nv={rmodel.nv}")
    print("Initial config: arm_up")

    # Setup visualization
    rdata = rmodel.createData()
    scene = Scene()
    robot = Robot(
        pinocchio_model=rmodel,
        pinocchio_data=rdata,
        pinocchio_geometry_model=kinova.visual_model,
        pinocchio_geometry_data=kinova.visual_model.createData(),
    )
    scene.add_robot(robot=robot)

    dataloader = DataLoader(
        object_trajectory_path=pathlib.Path(data["object_traj_path"]),
        scales_path=pathlib.Path(data["scale_path"]),
        load_grasps=False,
    )

    obj = Object.create_mesh(
        path_to_mesh=dataloader.object_info.mesh_path,
        name="robot/movable_obj",
        texture=dataloader.object_info.texture_path,
        scale=dataloader.object_info.scale,
        color=[0.8, 0.8, 0.8],
    )
    scene.add_object(obj)

    ee_frame_id = rmodel.getFrameId("j2s6s200_end_effector")
    nu = state.nv

    # Process each trajectory
    for idx, traj in enumerate(trajectories):
        if "tcp_trajectory_poses" not in traj:
            print(f"\nTrajectory {idx + 1}: no TCP, skipping")
            continue

        # Get TCP poses (limit to first 20 for speed)
        tcp_poses = [pin.SE3(np.array(pose)) for pose in traj["tcp_trajectory_poses"]]
        T = len(tcp_poses)

        print(f"\n{'=' * 60}")
        print(f"Trajectory {idx + 1}/{len(trajectories)}")
        print(f"TCP poses: {T}")
        print(f"{'=' * 60}")

        # Create running models
        running_models = []
        dt = 0.01

        for t in range(T):
            # Cost model for this timestep
            runningCostModel = crocoddyl.CostModelSum(state, nu)

            # Goal tracking - MUCH higher weight to enforce trajectory
            target = tcp_poses[t]
            framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
                state, ee_frame_id, target, nu
            )
            goalCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
            runningCostModel.addCost("goal", goalCost, 1e5)  # Increased from 1e3

            # State regularization - lower weight
            xResidual = crocoddyl.ResidualModelState(state, x0, nu)
            xRegCost = crocoddyl.CostModelResidual(state, xResidual)
            runningCostModel.addCost("xReg", xRegCost, 1e-3)  # Decreased from 1e-1

            # Control regularization
            uResidual = crocoddyl.ResidualModelJointEffort(state, actuation, nu)
            uRegCost = crocoddyl.CostModelResidual(state, uResidual)
            runningCostModel.addCost("uReg", uRegCost, 1e-4)

            # Use inverse dynamics (faster than forward dynamics)
            dam = crocoddyl.DifferentialActionModelFreeInvDynamics(
                state, actuation, runningCostModel
            )
            iam = crocoddyl.IntegratedActionModelEuler(dam, dt)
            running_models.append(iam)

        # Terminal model - even higher weight
        terminalCostModel = crocoddyl.CostModelSum(state, nu)
        terminalTarget = tcp_poses[-1]
        terminalResidual = crocoddyl.ResidualModelFramePlacement(
            state, ee_frame_id, terminalTarget, nu
        )
        terminalGoalCost = crocoddyl.CostModelResidual(state, terminalResidual)
        terminalCostModel.addCost("goal", terminalGoalCost, 1e6)  # Increased from 1e4

        terminalDAM = crocoddyl.DifferentialActionModelFreeInvDynamics(
            state, actuation, terminalCostModel
        )
        terminalModel = crocoddyl.IntegratedActionModelEuler(terminalDAM, 0.0)

        # Create problem and solver
        problem = crocoddyl.ShootingProblem(x0, running_models, terminalModel)
        solver = crocoddyl.SolverIntro(problem)
        solver.setCallbacks([crocoddyl.CallbackVerbose()])

        print("Stage 1: Solving to reach terminal pose...")
        solver.solve()

        print(f"  Converged: {solver.stop < solver.th_stop}")
        print(f"  Iterations: {solver.iter}")

        # Take final state as new initial condition
        x_final = solver.xs[-1]
        print("\nStage 2: Resolving with warm start from final state...")
        print(f"  New x0: q = {x_final[: rmodel.nq]}")

        # Rebuild problem with new x0
        problem2 = crocoddyl.ShootingProblem(x_final, running_models, terminalModel)
        solver2 = crocoddyl.SolverIntro(problem2)
        solver2.setCallbacks([crocoddyl.CallbackVerbose()])

        # Warm start with previous solution
        solver2.solve(solver.xs, solver.us)

        print(f"  Converged: {solver2.stop < solver2.th_stop}")
        print(f"  Iterations: {solver2.iter}")

        # Use second solution for visualization
        final_xs = solver2.xs

        # Visualize
        print("\nVisualizing (Enter to step)...")
        for k, x in enumerate(final_xs):
            robot[:] = x[: rmodel.nq]
            if k < len(traj["object_trajectory_poses"]):
                obj.pose = np.array(traj["object_trajectory_poses"][k])
            input(f"  Frame {k + 1}/{len(final_xs)}: ")

    print("\nDone.")


if __name__ == "__main__":
    main()

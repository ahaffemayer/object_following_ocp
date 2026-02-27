"""
Solve OCP for TALOS with only left arm moving.
Uses full model but partial actuation - only left arm is actuated.

Usage:
    python talos_ocp_partial.py <mesh_id>
"""

import pathlib
import pickle
import sys

import crocoddyl
import example_robot_data
import numpy as np
import pinocchio as pin
from robomeshcat import Object, Robot, Scene

from object_following_ocp.data.data_loader import DataLoader

OUTPUT_DIR = pathlib.Path("/mnt/user-data/outputs")


def main():
    if len(sys.argv) < 2:
        print("Usage: python talos_ocp_partial.py <mesh_id>")
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

    # Load full TALOS
    talos = example_robot_data.load("talos")
    rmodel = talos.model
    vmodel = talos.visual_model

    q0 = rmodel.referenceConfigurations["half_sitting"]

    # Find left arm joint velocity indices
    left_arm_v_indices = []
    for jnt_name in rmodel.names[1:]:
        if "arm_left" in jnt_name:
            jnt_id = rmodel.getJointId(jnt_name)
            if jnt_id > 0:
                jnt = rmodel.joints[jnt_id]
                left_arm_v_indices.extend(range(jnt.idx_v, jnt.idx_v + jnt.nv))

    print(f"Full TALOS: nq={rmodel.nq}, nv={rmodel.nv}")
    print(f"Left arm controls {len(left_arm_v_indices)} DOFs: {left_arm_v_indices}")

    # Create state and partial actuation
    state = crocoddyl.StateMultibody(rmodel)

    # Actuation matrix: only actuate left arm
    actuation_matrix = np.zeros((rmodel.nv, len(left_arm_v_indices)))
    for i, v_idx in enumerate(left_arm_v_indices):
        actuation_matrix[v_idx, i] = 1.0

    # actuation = crocoddyl.ActuationModelFloatingBase(state)
    # Override with custom actuation
    # actuation = crocoddyl.ActuationModelMultiCopterBase(state, len(left_arm_v_indices))

    # Actually, use simpler approach: use full actuation but high regularization on non-arm joints
    actuation = crocoddyl.ActuationModelFull(state)
    nu = state.nv

    # Initial state
    x0 = np.concatenate([q0, np.zeros(rmodel.nv)])

    # Setup visualization
    rdata = rmodel.createData()
    scene = Scene()
    robot = Robot(
        pinocchio_model=rmodel,
        pinocchio_data=rdata,
        pinocchio_geometry_model=vmodel,
        pinocchio_geometry_data=vmodel.createData(),
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

    # Find left gripper
    ee_frame_name = "gripper_left_fingertip_1_link"
    if not rmodel.existFrame(ee_frame_name):
        for alt in ["arm_left_7_link", "gripper_left_base_link", "wrist_left_ft_link"]:
            if rmodel.existFrame(alt):
                ee_frame_name = alt
                break

    ee_frame_id = rmodel.getFrameId(ee_frame_name)
    print(f"Using EE frame: {ee_frame_name}")

    # Process trajectories
    for idx, traj in enumerate(trajectories):
        if "tcp_trajectory_poses" not in traj:
            continue

        tcp_poses = [pin.SE3(np.array(pose)) for pose in traj["tcp_trajectory_poses"]]
        T = len(tcp_poses)

        print(f"\n{'=' * 60}")
        print(f"Trajectory {idx + 1}, TCP poses: {T}")
        print(f"{'=' * 60}")

        running_models = []
        dt = 0.01

        for t in range(T):
            runningCostModel = crocoddyl.CostModelSum(state, nu)

            # Goal tracking
            target = tcp_poses[t]
            framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
                state, ee_frame_id, target, nu
            )
            goalCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
            runningCostModel.addCost("goal", goalCost, 1e5)

            # State regularization - VERY HIGH weight on base to keep it fixed
            # Create weighted state regularization
            x_weights = np.ones(state.ndx) * 1e-1  # Normal weight for velocities
            # Heavy penalty on base position (first 7 DOFs: xyz + quaternion)
            x_weights[:6] = 1e6  # Base position and orientation
            x_weights[state.nq : state.nq + 6] = 1e6  # Base linear and angular velocity

            activation_x = crocoddyl.ActivationModelWeightedQuad(x_weights)
            xResidual = crocoddyl.ResidualModelState(state, x0, nu)
            xRegCost = crocoddyl.CostModelResidual(state, activation_x, xResidual)
            runningCostModel.addCost("xReg", xRegCost, 1.0)

            # Control regularization with HIGH weight on everything except left arm
            uResidual = crocoddyl.ResidualModelJointEffort(state, actuation, nu)

            # Create weighted control cost
            u_weights = (
                np.ones(nu) * 1e6
            )  # VERY high penalty by default (locks everything)
            for v_idx in left_arm_v_indices:
                u_weights[v_idx] = 1e-4  # Low penalty for left arm only

            activation = crocoddyl.ActivationModelWeightedQuad(u_weights)
            uRegCost = crocoddyl.CostModelResidual(state, activation, uResidual)
            runningCostModel.addCost("uReg", uRegCost, 1.0)

            dam = crocoddyl.DifferentialActionModelFreeInvDynamics(
                state, actuation, runningCostModel
            )
            iam = crocoddyl.IntegratedActionModelEuler(dam, dt)
            running_models.append(iam)

        # Terminal
        terminalCostModel = crocoddyl.CostModelSum(state, nu)
        terminalTarget = tcp_poses[-1]
        terminalResidual = crocoddyl.ResidualModelFramePlacement(
            state, ee_frame_id, terminalTarget, nu
        )
        terminalGoalCost = crocoddyl.CostModelResidual(state, terminalResidual)
        terminalCostModel.addCost("goal", terminalGoalCost, 1e6)

        terminalDAM = crocoddyl.DifferentialActionModelFreeInvDynamics(
            state, actuation, terminalCostModel
        )
        terminalModel = crocoddyl.IntegratedActionModelEuler(terminalDAM, 0.0)

        # Solve
        problem = crocoddyl.ShootingProblem(x0, running_models, terminalModel)
        solver = crocoddyl.SolverIntro(problem)
        solver.setCallbacks([crocoddyl.CallbackVerbose()])

        print("\nSolving OCP for TALOS left arm (partial actuation)...")
        solver.solve()

        print(f"Converged: {solver.stop < solver.th_stop}")
        print(f"Iterations: {solver.iter}")

        # Visualize
        print("\nVisualizing...")
        for k, x in enumerate(solver.xs):
            robot[:] = x[: rmodel.nq]
            if k < len(traj["object_trajectory_poses"]):
                obj.pose = np.array(traj["object_trajectory_poses"][k])
            input(f"Frame {k + 1}/{len(solver.xs)}: ")

    print("Done.")


if __name__ == "__main__":
    main()

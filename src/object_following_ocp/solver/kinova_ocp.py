"""
Solve OCP for Bravo7 Gripper using saved Panda TCP trajectories.
Uses inverse dynamics (faster/more stable than forward dynamics).

Usage:
    python bravo7_gripper_ocp.py <mesh_id>
"""

import pathlib
import pickle
import sys
import time

import crocoddyl
import example_robot_data
import numpy as np
import pinocchio as pin
from robomeshcat import Object, Robot, Scene

from object_following_ocp.data.data_loader import DataLoader

OUTPUT_DIR = pathlib.Path("/mnt/user-data/outputs")

# ── robot-specific knobs ────────────────────────────────────────────────────
ROBOT_NAME = "bravo7_gripper"

# EE frame: auto-detected below, but override here if needed
EE_FRAME_HINT = [
    "tool0",
    "ee_link",
    "end_effector",
    "tcp",
    "bravo_tip",
    "gripper",
    "wrist",
    "flange",
    "tool_frame",
]

# Reference config to use as initial pose (first match wins, then neutral)
Q0_CONFIG_HINT = ["home", "arm_up", "ready", "retract", "zeros"]
# ───────────────────────────────────────────────────────────────────────────


def find_ee_frame(rmodel, hints):
    """Return frame id for the first hint that matches (case-insensitive substring)."""
    names_lower = [f.name.lower() for f in rmodel.frames]
    for hint in hints:
        for i, name in enumerate(names_lower):
            if hint.lower() in name:
                print(f"  Auto-detected EE frame: '{rmodel.frames[i].name}' (id={i})")
                return i
    # Fallback: last non-universe frame
    fallback = len(rmodel.frames) - 1
    print(
        f"  WARNING: No EE frame hint matched. Using last frame: "
        f"'{rmodel.frames[fallback].name}' (id={fallback})"
    )
    return fallback


def find_q0(rmodel, hints):
    """Return reference config (first matching hint) or neutral config."""
    refs = rmodel.referenceConfigurations
    print(f"  Available reference configs: {list(refs.keys())}")
    for hint in hints:
        for key in refs.keys():
            if hint.lower() in key.lower():
                print(f"  Using reference config: '{key}'")
                return refs[key].copy()
    print("  WARNING: No reference config hint matched. Using neutral config.")
    return pin.neutral(rmodel)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {pathlib.Path(__file__).name} <mesh_id>")
        sys.exit(1)

    mesh_id = sys.argv[1]
    filepath = OUTPUT_DIR / f"{mesh_id}_kept_trajectories.pkl"

    if not filepath.exists():
        print(f"File not found: {filepath}")
        sys.exit(1)

    with open(filepath, "rb") as f:
        data = pickle.load(f)

    trajectories = data["trajectories"]
    print(f"Loaded {len(trajectories)} trajectories for '{mesh_id}'")

    # ── Load robot ────────────────────────────────────────────────────────
    robot = example_robot_data.load(ROBOT_NAME)
    rmodel = robot.model
    print(f"\nLoaded {ROBOT_NAME}: nq={rmodel.nq}, nv={rmodel.nv}")
    print(rmodel)

    # Print all joints and frames for debugging
    print("\nJoints:")
    for i, j in enumerate(rmodel.joints):
        print(f"  {i}: {rmodel.names[i]}, idx_q={j.idx_q}, nq={j.nq}, nv={j.nv}")
    print("\nFrames:")
    for i, f in enumerate(rmodel.frames):
        print(f"  {i}: {f.name}")

    q0 = find_q0(rmodel, Q0_CONFIG_HINT)
    x0 = np.concatenate([q0, np.zeros(rmodel.nv)])

    ee_frame_id = find_ee_frame(rmodel, EE_FRAME_HINT)

    state = crocoddyl.StateMultibody(rmodel)
    actuation = crocoddyl.ActuationModelFull(state)
    nu = state.nv

    # ── Visualization setup ───────────────────────────────────────────────
    rdata = rmodel.createData()
    scene = Scene()
    viz_robot = Robot(
        pinocchio_model=rmodel,
        pinocchio_data=rdata,
        pinocchio_geometry_model=robot.visual_model,
        pinocchio_geometry_data=robot.visual_model.createData(),
    )
    scene.add_robot(robot=viz_robot)

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

    # ── Process trajectories ──────────────────────────────────────────────
    for idx, traj in enumerate(trajectories):
        if "tcp_trajectory_poses" not in traj:
            print(f"\nTrajectory {idx + 1}: no TCP, skipping")
            continue

        tcp_poses = [pin.SE3(np.array(pose)) for pose in traj["tcp_trajectory_poses"]]
        T = len(tcp_poses)

        print(f"\n{'=' * 60}")
        print(f"Trajectory {idx + 1}/{len(trajectories)}, T={T} steps")
        print(f"{'=' * 60}")

        dt = 0.01
        running_models = []

        for t in range(T):
            costModel = crocoddyl.CostModelSum(state, nu)

            # EE placement tracking
            costModel.addCost(
                "goal",
                crocoddyl.CostModelResidual(
                    state,
                    crocoddyl.ResidualModelFramePlacement(
                        state, ee_frame_id, tcp_poses[t], nu
                    ),
                ),
                1e5,
            )
            # State regularization
            costModel.addCost(
                "xReg",
                crocoddyl.CostModelResidual(
                    state, crocoddyl.ResidualModelState(state, x0, nu)
                ),
                1e-3,
            )
            # Control regularization
            costModel.addCost(
                "uReg",
                crocoddyl.CostModelResidual(
                    state, crocoddyl.ResidualModelJointEffort(state, actuation, nu)
                ),
                1e-4,
            )

            dam = crocoddyl.DifferentialActionModelFreeInvDynamics(
                state, actuation, costModel
            )
            running_models.append(crocoddyl.IntegratedActionModelEuler(dam, dt))

        # Terminal
        termCostModel = crocoddyl.CostModelSum(state, nu)
        termCostModel.addCost(
            "goal",
            crocoddyl.CostModelResidual(
                state,
                crocoddyl.ResidualModelFramePlacement(
                    state, ee_frame_id, tcp_poses[-1], nu
                ),
            ),
            1e6,
        )
        termDAM = crocoddyl.DifferentialActionModelFreeInvDynamics(
            state, actuation, termCostModel
        )
        termModel = crocoddyl.IntegratedActionModelEuler(termDAM, 0.0)

        # Stage 1
        problem = crocoddyl.ShootingProblem(x0, running_models, termModel)
        solver = crocoddyl.SolverIntro(problem)
        solver.setCallbacks([crocoddyl.CallbackVerbose()])
        print("Stage 1: Solving...")
        solver.solve()
        print(f"  Converged: {solver.stop < solver.th_stop}, iters: {solver.iter}")

        # Stage 2: warm start from final state
        x_final = solver.xs[-1]
        print("Stage 2: Warm start from final state...")
        problem2 = crocoddyl.ShootingProblem(x_final, running_models, termModel)
        solver2 = crocoddyl.SolverIntro(problem2)
        solver2.setCallbacks([crocoddyl.CallbackVerbose()])
        solver2.solve(solver.xs, solver.us)
        print(f"  Converged: {solver2.stop < solver2.th_stop}, iters: {solver2.iter}")

        # Visualize
        print("\nPress Enter to visualize...")
        input()
        for k, x in enumerate(solver2.xs):
            viz_robot[:] = x[: rmodel.nq]
            if k < len(traj["object_trajectory_poses"]):
                obj.pose = np.array(traj["object_trajectory_poses"][k])
            time.sleep(0.05)

    print("\nDone.")


if __name__ == "__main__":
    main()

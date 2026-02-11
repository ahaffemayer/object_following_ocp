import sys
from typing import Any

import crocoddyl
import numpy as np
import pinocchio as pin

from object_following_ocp.trajectory import Trajectory


class OCP:
    """Optimal control problem for a robot reaching a target (no collision)"""

    def __init__(
        self,
        rmodel: pin.Model,
        target_poses: pin.SE3 | Trajectory | list[pin.SE3],
        x0: np.ndarray,
        joint_limits: bool = False,
        joint_limits_constraint: bool = False,
        with_callbacks: bool = False,
        weights: dict = {},
        T: int = 50,
        dt: float = 0.02,
        ee_frame: str = "panda_hand_tcp"
    ) -> None:

        self._rmodel = rmodel
        self._x0 = x0
        self._T = T
        self._dt = dt
        self._with_callbacks = with_callbacks
        self._joints_limits = joint_limits
        self._joint_limits_constraint = joint_limits_constraint

        # Weights
        self.weights = weights
        self._WEIGHT_xREG = weights.get("W_xREG", 1e-2)
        self._WEIGHT_uREG = weights.get("W_uREG", 1e-2)
        self._WEIGHT_GRIPPER_POSE = weights.get("W_gripper_pose", 1.0)
        self._WEIGHT_GRIPPER_POSE_TERM = weights.get(
            "W_gripper_pose_term", 10.0)
        self._WEIGHT_LIMIT = weights.get("W_limit", 1.0)

        # Targets
        self._TARGET_POSES = self._process_targets(target_poses)

        # Data
        self._rdata = rmodel.createData()

        # End-effector frame
        self._endeff_frame_id = self._rmodel.getFrameId(ee_frame)
        assert self._endeff_frame_id < len(self._rmodel.frames)

    def _process_targets(self, target_poses):
        if isinstance(target_poses, pin.SE3):
            return [target_poses] * self._T
        if isinstance(target_poses, (list, tuple)):
            if len(target_poses) < self._T:
                raise ValueError(
                    f"Expected at least {self._T} target poses, got {len(target_poses)}")
            return list(target_poses)
        if isinstance(target_poses, Trajectory):
            if len(target_poses) < self._T:
                raise ValueError(
                    f"Expected at least {self._T} target poses, got {len(target_poses)}")
            return list(target_poses.poses)
        raise TypeError("target_poses must be pin.SE3 or list[pin.SE3]")

    def create_OCP(self):

        # State and actuation
        state = crocoddyl.StateMultibody(self._rmodel)
        actuation = crocoddyl.ActuationModelFull(state)

        # Shared residuals
        xResidual = crocoddyl.ResidualModelState(state, self._x0)
        xRegCost = crocoddyl.CostModelResidual(state, xResidual)
        uResidual = crocoddyl.ResidualModelControl(state)
        uRegCost = crocoddyl.CostModelResidual(state, uResidual)

        running_models = []

        for t in range(self._T - 1):
            runningCostModel = crocoddyl.CostModelSum(state)
            runningCostModel.addCost("stateReg", xRegCost, self._WEIGHT_xREG)
            runningCostModel.addCost("ctrlReg", uRegCost, self._WEIGHT_uREG)

            # End-effector translation cost
            target = self._TARGET_POSES[t]
            frameResidual = crocoddyl.ResidualModelFramePlacement(
                state,
                self._endeff_frame_id,
                target,
            )
            goalCost = crocoddyl.CostModelResidual(state, frameResidual)
            runningCostModel.addCost(
                "gripperPose", goalCost, self._WEIGHT_GRIPPER_POSE)

            # Optional joint limits
            if self._joints_limits:
                maxfloat = sys.float_info.max
                xlb = np.concatenate(
                    [self._rmodel.lowerPositionLimit, -maxfloat * np.ones(state.nv)])
                xub = np.concatenate(
                    [self._rmodel.upperPositionLimit, maxfloat * np.ones(state.nv)])
                xLimitResidual = crocoddyl.ResidualModelState(
                    state, self._x0, actuation.nu)

                if self._joint_limits_constraint:
                    limitConstraint = crocoddyl.ConstraintModelResidual(
                        state, xLimitResidual, xlb, xub)
                    runningConstraintManager = crocoddyl.ConstraintModelManager(
                        state, actuation.nu)
                    runningConstraintManager.addConstraint(
                        f"lim_{t}", limitConstraint)
                else:
                    bounds = crocoddyl.ActivationBounds(xlb, xub, 1.0)
                    activation = crocoddyl.ActivationModelQuadraticBarrier(
                        bounds)
                    limitCost = crocoddyl.CostModelResidual(
                        state, activation, xLimitResidual)
                    runningCostModel.addCost(
                        "limitCost", limitCost, self._WEIGHT_LIMIT)

            # Dynamics
            dam = crocoddyl.DifferentialActionModelFreeFwdDynamics(
                state,
                actuation,
                runningCostModel
            )
            iam = crocoddyl.IntegratedActionModelEuler(dam, self._dt)
            iam.differential.armature = np.array(self._rmodel.nv * [0.1])
            running_models.append(iam)

        # Terminal model
        terminalCostModel = crocoddyl.CostModelSum(state)
        terminalCostModel.addCost("stateReg", xRegCost, self._WEIGHT_xREG)
        terminalTarget = self._TARGET_POSES[-1]
        terminalResidual = crocoddyl.ResidualModelFramePlacement(
            state,
            self._endeff_frame_id,
            terminalTarget,
        )
        terminalGoalCost = crocoddyl.CostModelResidual(state, terminalResidual)
        terminalCostModel.addCost(
            "gripperPose", terminalGoalCost, self._WEIGHT_GRIPPER_POSE_TERM)

        terminalDAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, terminalCostModel)
        terminalModel = crocoddyl.IntegratedActionModelEuler(terminalDAM, 0.0)
        terminalModel.differential.armature = np.array(self._rmodel.nv * [0.1])

        # Problem + solver
        problem = crocoddyl.ShootingProblem(
            self._x0, running_models, terminalModel)
        ocp = crocoddyl.SolverFDDP(problem)
        if self._with_callbacks:
            ocp.setCallbacks([crocoddyl.CallbackVerbose()])
        return ocp

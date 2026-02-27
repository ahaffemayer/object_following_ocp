import sys

import colmpc as col
import crocoddyl
import numpy as np
import pinocchio as pin

from object_following_ocp.geom.trajectories import TrajectorySE3


class OCP:
    """Optimal control problem for any robot reaching a target, with collision avoidance"""

    def __init__(
        self,
        rmodel: pin.Model,
        cmodel: pin.GeometryModel,
        target_poses: pin.SE3 | TrajectorySE3 | list[pin.SE3],
        x0: np.ndarray,
        joint_limits: bool = False,
        joint_limits_constraint: bool = False,
        with_callbacks: bool = False,
        weights: dict = {},
        safety_threshold: float = 0.01,
        T: int = 50,
        dt: float = 0.02,
        ee_frame_name: str = "panda_hand_tcp",
    ) -> None:

        # Robot models
        self._rmodel = rmodel
        self._cmodel = cmodel

        # Problem parameters
        self._x0 = x0
        self._T = T
        self._dt = dt
        self._with_callbacks = with_callbacks
        self._SAFETY_THRESHOLD = safety_threshold

        self._joints_limits = joint_limits
        self._joint_limits_constraint = joint_limits_constraint

        # Weights
        self.weights = weights
        self._WEIGHT_xREG = weights["W_xREG"]
        self._WEIGHT_uREG = weights["W_uREG"]
        self._WEIGHT_GRIPPER_POSE = weights["W_gripper_pose"]
        self._WEIGHT_GRIPPER_POSE_TERM = weights["W_gripper_pose_term"]
        self._WEIGHT_LIMIT = weights["W_limit"]

        # Normalize targets to a list
        self._TARGET_POSES = self._process_targets(target_poses)

        # Data
        self._rdata = rmodel.createData()
        self._cdata = cmodel.createData()

        # End-effector frame
        self._endeff_frame = self._rmodel.getFrameId(ee_frame_name)
        assert self._endeff_frame < len(self._rmodel.frames)

    def _process_targets(self, target_poses):
        if isinstance(target_poses, pin.SE3):
            return [target_poses] * self._T

        if isinstance(target_poses, (list, tuple)):
            if len(target_poses) < self._T:
                raise ValueError(
                    f"Expected at least {self._T} target poses, got {len(target_poses)}"
                )
            return list(target_poses)

        if isinstance(target_poses, TrajectorySE3):
            if len(target_poses) < self._T:
                raise ValueError(
                    f"Expected at least {self._T} target poses, got {len(target_poses)}"
                )
            return list(target_poses.poses)

        raise TypeError("target_poses must be pin.SE3 or list[pin.SE3]")

    def create_OCP(self):

        # State and actuation
        self._state = crocoddyl.StateMultibody(self._rmodel)
        self._actuation = crocoddyl.ActuationModelFull(self._state)

        # Shared residuals
        xResidual = crocoddyl.ResidualModelState(self._state, self._x0)
        xRegCost = crocoddyl.CostModelResidual(self._state, xResidual)

        uResidual = crocoddyl.ResidualModelControl(self._state)
        uRegCost = crocoddyl.CostModelResidual(self._state, uResidual)

        running_models = []

        for t in range(self._T - 1):
            # ---------- COSTS ----------
            runningCostModel = crocoddyl.CostModelSum(self._state)

            runningCostModel.addCost("stateReg", xRegCost, self._WEIGHT_xREG)
            runningCostModel.addCost("ctrlReg", uRegCost, self._WEIGHT_uREG)

            target = self._TARGET_POSES[t]
            frameResidual = crocoddyl.ResidualModelFramePlacement(
                self._state,
                self._endeff_frame,
                target,
            )
            goalCost = crocoddyl.CostModelResidual(self._state, frameResidual)

            runningCostModel.addCost("gripperPose", goalCost, self._WEIGHT_GRIPPER_POSE)

            # ---------- CONSTRAINTS ----------
            runningConstraintManager = crocoddyl.ConstraintModelManager(
                self._state, self._actuation.nu
            )

            # Collision constraints
            if len(self._cmodel.collisionPairs) != 0:
                for col_idx in range(len(self._cmodel.collisionPairs)):
                    distResidual = col.ResidualDistanceCollision(
                        self._state,
                        self._state.nv,  # Use state.nv not rmodel.nv
                        self._cmodel,
                        col_idx,
                    )
                    constraint = crocoddyl.ConstraintModelResidual(
                        self._state,
                        distResidual,
                        np.array([self._SAFETY_THRESHOLD]),
                        np.array([np.inf]),
                    )
                    runningConstraintManager.addConstraint(
                        f"col_{t}_{col_idx}", constraint
                    )

            # Joint limits
            if self._joints_limits:
                maxfloat = sys.float_info.max
                xlb = np.concatenate(
                    [
                        self._rmodel.lowerPositionLimit,
                        -maxfloat * np.ones(self._state.nv),  # Use state.nv
                    ]
                )
                xub = np.concatenate(
                    [
                        self._rmodel.upperPositionLimit,
                        maxfloat * np.ones(self._state.nv),  # Use state.nv
                    ]
                )

                xLimitResidual = crocoddyl.ResidualModelState(self._state)

                if self._joint_limits_constraint:
                    limitConstraint = crocoddyl.ConstraintModelResidual(
                        self._state,
                        xLimitResidual,
                        xlb,
                        xub,
                    )
                    runningConstraintManager.addConstraint(f"lim_{t}", limitConstraint)
                else:
                    bounds = crocoddyl.ActivationBounds(xlb, xub)
                    activation = crocoddyl.ActivationModelQuadraticBarrier(bounds)
                    limitCost = crocoddyl.CostModelResidual(
                        self._state, activation, xLimitResidual
                    )
                    runningCostModel.addCost("limitCost", limitCost, self._WEIGHT_LIMIT)

            # ---------- DYNAMICS ----------
            dam = crocoddyl.DifferentialActionModelFreeFwdDynamics(
                self._state,
                self._actuation,
                runningCostModel,
                runningConstraintManager,
            )

            iam = crocoddyl.IntegratedActionModelEuler(dam, self._dt)
            iam.differential.armature = np.full(self._state.nv, 0.1)  # Use state.nv

            running_models.append(iam)

        # ---------- TERMINAL MODEL ----------
        terminalCostModel = crocoddyl.CostModelSum(self._state)

        terminalCostModel.addCost("stateReg", xRegCost, self._WEIGHT_xREG)

        terminalTarget = self._TARGET_POSES[-1]
        terminalResidual = crocoddyl.ResidualModelFramePlacement(
            self._state,
            self._endeff_frame,
            terminalTarget,
        )
        terminalGoalCost = crocoddyl.CostModelResidual(self._state, terminalResidual)

        terminalCostModel.addCost(
            "gripperPose", terminalGoalCost, self._WEIGHT_GRIPPER_POSE_TERM
        )

        terminalConstraintManager = crocoddyl.ConstraintModelManager(
            self._state, self._actuation.nu
        )

        terminalDAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            self._state,
            self._actuation,
            terminalCostModel,
            terminalConstraintManager,
        )

        terminalModel = crocoddyl.IntegratedActionModelEuler(terminalDAM, 0.0)
        terminalModel.differential.armature = np.full(
            self._state.nv, 0.1
        )  # Use state.nv

        # ---------- PROBLEM + SOLVER ----------
        problem = crocoddyl.ShootingProblem(self._x0, running_models, terminalModel)

        # ocp = mim_solvers.SolverCSQP(problem)
        ocp = crocoddyl.SolverFDDP(problem)
        # ocp.use_filter_line_search = False
        # ocp.termination_tolerance = 1e-3
        # ocp.max_qp_iters = 25
        # ocp.eps_abs = 1e-6
        # ocp.eps_rel = 0.0
        # ocp.with_callbacks = self._with_callbacks
        ocp.setCallbacks([crocoddyl.CallbackVerbose()])

        return ocp

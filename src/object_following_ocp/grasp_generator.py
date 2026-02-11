import logging
import time
from typing import Any

import numpy as np
import pinocchio as pin

from object_following_ocp.ocp import OCP
from object_following_ocp.robot_loader import self_collision_pairs

# ------------------------------------------------------------------------------
# Logging configuration, call ONCE from the application
# ------------------------------------------------------------------------------


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="[%(levelname)s] %(name)s: %(message)s",
        force=True,
    )


# ------------------------------------------------------------------------------
# Grasp generator
# ------------------------------------------------------------------------------

class GraspGenerator:
    """Generates grasp configurations for a given object pose."""

    def __init__(
        self,
        rmodel,
        cmodel,
        obj_pose: pin.SE3,
        grasp_configurations_number: int,
        max_attempts: int = 100,
        offset: np.ndarray = np.zeros(3),
        ee_frame: str = "panda_hand_tcp",
    ):
        self.logger = logging.getLogger("GraspGenerator")

        self._obj_pose = obj_pose
        self._grasp_configurations_number = grasp_configurations_number
        self._max_attempts = max_attempts
        self._offset = offset
        self._ee_frame = ee_frame

        self.rmodel = rmodel
        self.cmodel = cmodel

        # Add self-collision pairs
        for name1, name2 in self_collision_pairs:
            if (
                self.cmodel.existGeometryName(name1)
                and self.cmodel.existGeometryName(name2)
            ):
                self.cmodel.addCollisionPair(
                    pin.CollisionPair(
                        self.cmodel.getGeometryId(name1),
                        self.cmodel.getGeometryId(name2),
                    )
                )

        self.rdata = self.rmodel.createData()
        self.cdata = self.cmodel.createData()

        self._target_pose = self._apply_offset(self._obj_pose)

        self.logger.debug("Initialized GraspGenerator")

    # --------------------------------------------------------------------------

    def generate_grasp_configurations(self) -> list[np.ndarray]:
        grasps: list[np.ndarray] = []
        attempts = 0

        self.logger.debug(
            "Starting grasp generation, target=%d, max_attempts=%d",
            self._grasp_configurations_number,
            self._max_attempts,
        )

        t_global = time.perf_counter()

        while attempts < self._max_attempts and len(grasps) < self._grasp_configurations_number:
            attempts += 1
            self.logger.debug("Attempt %d", attempts)

            # ------------------------------------------------------------------
            # Sampling
            # ------------------------------------------------------------------
            try:
                t0 = time.perf_counter()
                q0 = self._sample_collision_free_configuration()
                self.logger.debug(
                    "Sampling took %.3f ms",
                    1e3 * (time.perf_counter() - t0),
                )
            except RuntimeError:
                self.logger.debug("Sampling failed")
                continue

            # ------------------------------------------------------------------
            # OCP creation
            # ------------------------------------------------------------------
            t0 = time.perf_counter()
            ik = self._create_ik_ocp(q0)
            self.logger.debug(
                "OCP creation took %.3f ms",
                1e3 * (time.perf_counter() - t0),
            )

            # ------------------------------------------------------------------
            # Solve
            # ------------------------------------------------------------------
            t0 = time.perf_counter()
            ik.solve()
            solve_time = time.perf_counter() - t0
            self.logger.debug("IK solve took %.3f ms", 1e3 * solve_time)

            q_sol = ik.xs[-1][: self.rmodel.nq]

            # ------------------------------------------------------------------
            # Validation
            # ------------------------------------------------------------------
            t0 = time.perf_counter()
            if self._is_grasp_valid(q_sol):
                grasps.append(q_sol)
                self.logger.debug(
                    "Valid grasp %d / %d",
                    len(grasps),
                    self._grasp_configurations_number,
                )

            self.logger.debug(
                "Validation took %.3f ms",
                1e3 * (time.perf_counter() - t0),
            )

        self.logger.debug(
            "Finished grasp generation in %.3f s, attempts=%d, grasps=%d",
            time.perf_counter() - t_global,
            attempts,
            len(grasps),
        )

        return grasps

    # --------------------------------------------------------------------------

    def _create_ik_ocp(self, q0: np.ndarray):
        x0 = np.concatenate([q0, np.zeros(self.rmodel.nv)])

        ocp = OCP(
            rmodel=self.rmodel,
            cmodel=self.cmodel,
            target_poses=self._target_pose,
            x0=x0,
            joint_limits=True,
            joint_limits_constraint=False,
            with_callbacks=False,
            weights=self._weights(),
            safety_threshold=0.01,
            T=3,
            dt=0.02,
            ee_frame=self._ee_frame,
        )
        return ocp.create_OCP()

    # --------------------------------------------------------------------------

    def _sample_collision_free_configuration(self) -> np.ndarray:
        max_trials = 1000
        t0 = time.perf_counter()

        for i in range(max_trials):
            q = pin.randomConfiguration(self.rmodel)

            pin.framesForwardKinematics(self.rmodel, self.rdata, q)
            pin.updateGeometryPlacements(
                self.rmodel, self.rdata, self.cmodel, self.cdata
            )

            if not pin.computeCollisions(self.cmodel, self.cdata, True):
                self.logger.debug(
                    "Collision-free after %d trials in %.3f ms",
                    i + 1,
                    1e3 * (time.perf_counter() - t0),
                )
                return q

        raise RuntimeError("Collision-free sampling failed")

    # --------------------------------------------------------------------------

    def _is_grasp_valid(self, q: np.ndarray) -> bool:
        pin.framesForwardKinematics(self.rmodel, self.rdata, q)
        pin.updateGeometryPlacements(
            self.rmodel, self.rdata, self.cmodel, self.cdata
        )

        if pin.computeCollisions(self.cmodel, self.cdata, True):
            self.logger.debug("Rejected due to collision")
            return False

        ee_id = self.rmodel.getFrameId(self._ee_frame)
        ee_pose = self.rdata.oMf[ee_id]

        dist = np.linalg.norm(pin.log6(ee_pose.inverse() * self._obj_pose))
        self.logger.debug("EE distance %.4f", dist)

        return dist < 0.05

    # --------------------------------------------------------------------------

    def _apply_offset(self, pose: pin.SE3) -> pin.SE3:
        out = pose.copy()
        out.translation = out.translation + self._offset
        return out

    # --------------------------------------------------------------------------

    @staticmethod
    def _weights() -> dict[str, float]:
        return {
            "W_xREG": 1e-4,
            "W_uREG": 1e-4,
            "W_gripper_pose": 10.0,
            "W_gripper_pose_term": 1e5,
            "W_limit": 0.0,
        }

    # --------------------------------------------------------------------------

    @staticmethod
    def select_best_grasp(
        grasps: list[np.ndarray],
        reference_configuration: np.ndarray,
    ) -> np.ndarray | None:
        if not grasps:
            return None

        dists = [np.linalg.norm(q - reference_configuration) for q in grasps]
        return grasps[int(np.argmin(dists))]

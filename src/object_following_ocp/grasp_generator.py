import meshcat
import numpy as np
from pathlib import Path
import pinocchio as pin
import sys
from typing import Any
import numpy as np
import colmpc as col
import crocoddyl
import pinocchio as pin
import mim_solvers
from pinocchio import visualize
from robomeshcat import Scene, Object, Robot

from object_following_ocp.robot_loader import load_reduced_panda, self_collision_pairs
from object_following_ocp.ocp import OCP
from object_following_ocp.parser_config import load_config


class GraspGenerator:
    """Generates grasp configurations for a given object pose."""

    def __init__(
        self,
        obj_pose: pin.SE3,
        grasp_configurations_number: int,
        max_attempts: int = 100,
        offset: np.ndarray = np.array([0.0, 0.0, 0.0]),
    ):
        self._obj_pose = obj_pose
        self._grasp_configurations_number = grasp_configurations_number
        self._max_attempts = max_attempts
        self._offset = offset
        self._offested_obj_pose = self._add_offest_to_pose(self._obj_pose)

        # Create the robot model
        self.rmodel, self.cmodel, self.vmodel = load_reduced_panda()
        for cp in self_collision_pairs:
            if self.cmodel.existGeometryName(cp[0]) and self.cmodel.existGeometryName(
                cp[1]
            ):
                self.cmodel.addCollisionPair(
                    pin.CollisionPair(
                        self.cmodel.getGeometryId(cp[0]),
                        self.cmodel.getGeometryId(cp[1]),
                    )
                )
        self.rdata = self.rmodel.createData()
        self.cdata = self.cmodel.createData()

    def generate_grasps_configurations(self) -> list[np.ndarray]:
        """Generates a list of grasp configurations around the object pose."""
        grasps = []
        attempts = 0
        while (
            attempts < self._max_attempts
            and len(grasps) < self._grasp_configurations_number
        ):

            # Generate a random collision-free configuration
            try:
                q_random = self._get_random_collision_free_configuration()
            except RuntimeError:
                attempts += 1
                continue

            # Create an IK problem for the grasping task
            ik_ocp = self._create_IK_problem(q_random)

            # Solve the IK problem
            ik_ocp.solve()
            q_sol = ik_ocp.xs[-1][: self.rmodel.nq]
            # if the solution is valid and collision-free, add it to the list
            if self._check_grasp_validity(q_sol):
                grasps.append(q_sol)
            attempts += 1

        return grasps

    def _create_IK_problem(self, q0: np.ndarray) -> OCP:
        """Creates an IK problem for the grasping task."""

        weights = self._get_weights()
        ik_ocp = OCP(
            rmodel=self.rmodel,
            cmodel=self.cmodel,
            target_poses=self._offested_obj_pose,
            x0=np.concatenate((q0, np.zeros(self.rmodel.nv))),
            joint_limits=True,
            joint_limits_constraint=False,
            with_callbacks=False,
            weights=weights,
            safety_threshold=0.01,
            T=3,
            dt=0.02,
        )
        ik = ik_ocp.create_OCP()
        return ik

    def _get_random_collision_free_configuration(self) -> np.ndarray:
        """Generates a random collision-free configuration for the robot."""
        max_trials = 1000
        for _ in range(max_trials):
            q_random = pin.randomConfiguration(self.rmodel)
            pin.framesForwardKinematics(self.rmodel, self.rdata, q_random)
            pin.updateGeometryPlacements(
                self.rmodel, self.rdata, self.cmodel, self.cdata
            )
            collisions = pin.computeCollisions(self.cmodel, self.cdata, True)
            if not collisions:
                return q_random
        raise RuntimeError("Failed to find a collision-free configuration.")

    def _get_weights(self) -> dict:
        """Returns the weights for the IK problem."""
        return {
            "W_xREG": 0.0000,
            "W_uREG": 0.0000,
            "W_gripper_pose": 10.0,
            "W_gripper_pose_term": 100000.0,
            "W_limit": 1000.0,
        }

    def _add_offest_to_pose(self, pose: pin.SE3) -> pin.SE3:
        """Adds an offset to a given pose."""
        new_translation = pose.translation + self._offset
        # return pin.SE3(pose.rotation, new_translation)
        return pose

    def _check_grasp_validity(self, q: np.ndarray) -> bool:
        """Checks if a given grasp configuration is valid (collision-free)."""
        pin.framesForwardKinematics(self.rmodel, self.rdata, q)
        pin.updateGeometryPlacements(self.rmodel, self.rdata, self.cmodel, self.cdata)
        collisions = pin.computeCollisions(self.cmodel, self.cdata, True)
        if not collisions:
            # See if the end-effector is close enough to the object pose
            ee_frame_id = self.rmodel.getFrameId("panda_hand_tcp")
            ee_pose = self.rdata.oMf[ee_frame_id]
            distance = np.linalg.norm(pin.log6(ee_pose.inverse() * self._obj_pose))
            if distance < 0.05:  # 5 cm threshold
                return True
        return False

    def select_best_grasp(
        self, grasps: list[np.ndarray], reference_configuration: np.ndarray
    ) -> np.ndarray:
        """Selects the best grasp configuration based on proximity to a reference configuration."""
        best_grasp = None
        best_distance = float("inf")
        for q in grasps:
            distance = np.linalg.norm(q - reference_configuration)
            if distance < best_distance:
                best_distance = distance
                best_grasp = q
        return best_grasp

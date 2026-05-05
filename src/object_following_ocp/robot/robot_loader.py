import example_robot_data as robex
import numpy as np
import pinocchio as pin
from curobo.util_file import get_assets_path, join_path
from robomeshcat import Object, Scene

robot_links = [
    # "panda_link0_capsule_0",
    "panda_hand_capsule_0",
    "panda_link7_capsule_0",
    "panda_link7_capsule_1",
    "panda_link6_capsule_0",
    "panda_link5_capsule_0",
    "panda_link5_capsule_1",
    "panda_link4_capsule_0",
    "panda_rightfinger_capsule_0",
    "panda_leftfinger_capsule_0",
]


self_collision_pairs = [
    ("panda_link4_capsule_0", "panda_link6_capsule_0"),
    ("panda_link4_capsule_0", "panda_link7_capsule_0"),
    # ("panda_link5_capsule_0", "panda_link7_capsule_0"),
    # ("panda_link6_capsule_0", "panda_link7_capsule_0"),
    ("panda_link0_capsule_0", "panda_link7_capsule_0"),
]


def load_panda():
    panda = robex.load("panda_collision")
    rmodel, cmodel, vmodel = panda.model, panda.collision_model, panda.visual_model
    return rmodel, cmodel, vmodel


def load_reduced_panda():
    rmodel, cmodel, vmodel = load_panda()
    geom_models = [vmodel, cmodel]
    rmodel, geometric_models_reduced = pin.buildReducedModel(
        rmodel,
        list_of_geom_models=geom_models,
        list_of_joints_to_lock=[8, 9],
        reference_configuration=np.array(
            [
                -0.6513877410293797,
                1.3677075286603906,
                -0.17736737718858037,
                -0.3973375018143172,
                -0.11554961778792178,
                1.2408486160482337,
                8.644879755868687e-05,
                0.01,
                0.02,
            ]
        ),
    )

    vmodel, cmodel = geometric_models_reduced[0], geometric_models_reduced[1]
    return rmodel, cmodel, vmodel


def load_kinova():
    asset_root = get_assets_path()

    urdf_path = join_path(asset_root, "robot/kinova/kinova_gen3_7dof.urdf")
    mesh_root = join_path(asset_root, "robot/kinova")
    rmodel, cmodel, vmodel = pin.buildModelsFromUrdf(
        urdf_path,
        package_dirs=[mesh_root],
    )

    return rmodel, cmodel, vmodel


def load_ur5():

    asset_root = get_assets_path()
    urdf_path = join_path(asset_root, "robot/ur_description/ur5e.urdf")
    mesh_root = join_path(asset_root, "robot/ur_description")

    rmodel, cmodel, vmodel = pin.buildModelsFromUrdf(
        urdf_path,
        package_dirs=[mesh_root],
    )

    return rmodel, cmodel, vmodel


def make_gripper_objects(scene: Scene) -> dict:
    """Create gripper box objects in the scene. Returns dict of {name: (Object, placement_in_tool0)}."""
    palm_z = 0.010
    finger_z = 0.050
    finger_spread = 0.010

    gripper_defs = [
        (
            "gripper_palm",
            [0.060, 0.060, 0.030],
            pin.SE3(np.eye(3), np.array([0.0, 0.0, palm_z])),
        ),
        (
            "gripper_finger_left",
            [0.03, 0.01, 0.10],
            pin.SE3(np.eye(3), np.array([0.0, +finger_spread, finger_z])),
        ),
        (
            "gripper_finger_right",
            [0.03, 0.01, 0.10],
            pin.SE3(np.eye(3), np.array([0.0, -finger_spread, finger_z])),
        ),
    ]

    gripper_objects = {}
    for name, extents, placement_in_tool0 in gripper_defs:
        obj = Object.create_cuboid(lengths=extents, name=name, color=[0.6, 0.6, 0.6])
        scene.add_object(obj)
        gripper_objects[name] = (obj, placement_in_tool0)

    return gripper_objects


def update_gripper_pose(gripper_objects: dict, rmodel, rdata, q: np.ndarray):
    """Call this after pin.forwardKinematics to update gripper box poses."""
    pin.framesForwardKinematics(rmodel, rdata, q)
    tool0_id = rmodel.getFrameId("tool0")
    wM_tool0 = rdata.oMf[tool0_id]

    for obj, placement_in_tool0 in gripper_objects.values():
        wM_box = wM_tool0 * placement_in_tool0
        obj.pose = wM_box.homogeneous


def load_ur5_pin():
    ur5 = robex.load("ur5_gripper")
    rmodel, cmodel, vmodel = ur5.model, ur5.collision_model, ur5.visual_model
    return rmodel, cmodel, vmodel

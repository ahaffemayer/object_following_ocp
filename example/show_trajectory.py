import numpy as np
from pathlib import Path
import pinocchio as pin
from robomeshcat import Scene, Object, Robot
import time

from object_following_ocp.grasp_generator import GraspGenerator
from object_following_ocp.ocp import OCP
from object_following_ocp.parser_config import load_config
from object_following_ocp.robot_loader import load_reduced_panda, self_collision_pairs
from object_following_ocp.trajectory_parser import TrajectoryParser
from object_following_ocp.trajectory import Trajectory, TrajectoryInConfigurationSpace, TrajectoryEvaluator


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":

    tstart = time.time()

    cfg = load_config(Path(__file__).parent / "config.yaml")
    weights = cfg["weights"]
    dt = cfg["dt"]
    safety_threshold = cfg["safety_threshold"]

    mesh_dir = Path(cfg["mesh"]["path"])
    obj_file = mesh_dir / cfg["mesh"]["obj_file"]
    texture_file = mesh_dir / cfg["mesh"]["texture_file"]
    scale = cfg["mesh"]["scale"]
    color = cfg["mesh"]["color"]

    csv_path = Path("/home/arthur/Desktop/Projects/PAMI/object_following_ocp/ressources/video_data/d02/campbells2/campbells2-obj_0-tracked-0.csv")

    # -----------------------------
    # Load trajectory parser
    # -----------------------------

    parser = TrajectoryParser(csv_path, smooth_depth=True, smooth_k=2.0)

    # -----------------------------
    # Load robot and collision model
    # -----------------------------

    rmodel, cmodel, vmodel = load_reduced_panda()

    for cp in self_collision_pairs:
        if cmodel.existGeometryName(cp[0]) and cmodel.existGeometryName(cp[1]):
            cmodel.addCollisionPair(
                pin.CollisionPair(
                    cmodel.getGeometryId(cp[0]),
                    cmodel.getGeometryId(cp[1]),
                )
            )

    rdata = rmodel.createData()
    cdata = cmodel.createData()
    vdata = vmodel.createData()

    robot = Robot(
        pinocchio_model=rmodel,
        pinocchio_data=rdata,
        pinocchio_geometry_model=vmodel,
        pinocchio_geometry_data=vdata,
    )

    # -----------------------------
    # Scene
    # -----------------------------

    scene = Scene()
    scene.add_robot(robot)

    # -----------------------------
    # Add object to scene
    # -----------------------------
    
    o = Object.create_mesh(
        path_to_mesh=obj_file,
        name="robot/movable_obj",
        texture=texture_file,
        scale=scale,
        color=color,  
    )
    scene.add_object(o)
    for k, pose in enumerate(parser.get_camera_trajectory()):
        color = [0.0, 1.0, 0.0] if k == 0 else [0.5, 0.5, 0.5]            
        scene.add_object(
            Object.create_sphere(radius=0.01, name=f"target_{k}", color=color)
        )        
        scene[f"target_{k}"].pos[:] = pose.translation
        o.pose = pose.homogeneous
        input("Press Enter to continue to the next target...")
    
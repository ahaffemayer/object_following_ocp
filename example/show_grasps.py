import yaml
import numpy as np
import trimesh
import meshcat
import meshcat.geometry as g
import meshcat.transformations as mtf
import trimesh.transformations as tra


# ============================================================
# Math utilities
# ============================================================

def quat_to_rot(qw, qx, qy, qz):
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)

    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def grasp_to_T(grasp, apply_tcp_offset=False, gripper_depth=0.1034):
    """
    Convert grasp dict to 4x4 transform matrix.
    
    Args:
        grasp: Dict with 'position' and 'orientation' keys
        apply_tcp_offset: If True, apply TCP offset (gripper depth along z-axis)
        gripper_depth: Distance from gripper base to TCP
    """
    t = np.array(grasp["position"], dtype=np.float64)
    qw = grasp["orientation"]["w"]
    qx, qy, qz = grasp["orientation"]["xyz"]

    T = np.eye(4)
    T[:3, :3] = quat_to_rot(qw, qx, qy, qz)
    T[:3, 3] = t
    
    if apply_tcp_offset:
        # GraspGen convention: approach direction is along +z
        # Apply offset to get TCP position
        T_offset = np.eye(4)
        T_offset[:3, 3] = [0.0, 0.0, gripper_depth]
        T = T @ T_offset
    
    return T


# ============================================================
# Visualization helpers
# ============================================================

def create_visualizer(clear=True):
    vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    if clear:
        vis.delete()
    return vis


def visualize_mesh(vis, name, mesh, color=(180, 180, 180), T=None):
    material = g.MeshPhongMaterial(
        color="0x%02x%02x%02x" % color
    )
    vis[name].set_object(
        g.TriangularMeshGeometry(mesh.vertices, mesh.faces),
        material
    )
    if T is not None:
        vis[name].set_transform(T)


def make_frame(vis, name, T, h=0.08, r=0.003):
    colors = {
        "x": 0xFF0000,
        "y": 0x00FF00,
        "z": 0x0000FF,
    }
    axes = {
        "x": ([0, 0, 1], np.pi / 2, [h/2, 0, 0]),
        "y": ([0, 1, 0], np.pi / 2, [0, h/2, 0]),
        "z": ([1, 0, 0], np.pi / 2, [0, 0, h/2]),
    }

    for ax, (axis, angle, trans) in axes.items():
        vis[name][ax].set_object(
            g.Cylinder(height=h, radius=r),
            g.MeshLambertMaterial(color=colors[ax])
        )
        M = mtf.rotation_matrix(angle, axis)
        M[:3, 3] = trans
        vis[name][ax].set_transform(M)

    vis[name].set_transform(T)


def visualize_gripper(vis, name, T, width=0.08, depth=0.06, color=0xFF0000):
    points = []

    def box(x, y, z):
        return np.array([
            [-x, -y, -z],
            [ x, -y, -z],
            [ x,  y, -z],
            [-x,  y, -z],
            [-x, -y,  z],
            [ x, -y,  z],
            [ x,  y,  z],
            [-x,  y,  z],
        ])

    finger = box(0.005, depth/2, 0.02)
    left = finger + np.array([ width/2, 0, 0])
    right = finger + np.array([-width/2, 0, 0])

    for part in [left, right]:
        P = np.hstack([part, np.ones((8, 1))]).T
        points.append(P)

    for i, P in enumerate(points):
        vis[name + f"/finger_{i}"].set_object(
            g.Line(
                g.PointsGeometry(P),
                g.MeshBasicMaterial(color=color)
            )
        )

    vis[name].set_transform(T)


def visualize_gripper_as_sphere(vis, name, T, radius=0.01, color=0xFF0000):
    """
    Replaces the gripper visualization with a sphere at the grasp center.
    """
    if vis is None:
        return

    # Create the sphere geometry
    sphere_geom = g.Sphere(radius)
    # Create the material
    material = g.MeshLambertMaterial(color=color)
    
    # Set the object at the specific name path
    vis[name].set_object(sphere_geom, material)
    
    # Apply the 4x4 transform matrix T
    # This moves the sphere to the grasp position and orientation
    vis[name].set_transform(T.astype(np.float64))

# ============================================================
# Main
# ============================================================

def main1():
    grasp_yaml = "test.yaml"
    object_mesh_path = "/home/arthur/Desktop/Projects/PAMI/object_following_ocp/ressources/video_data/d02/jug/0549d65fc59a45618d6e3d542cbfa5b8/0549d65fc59a45618d6e3d542cbfa5b8.obj"

    vis = create_visualizer(clear=True)

    mesh = trimesh.load(object_mesh_path, force="mesh")
    mesh_scale = 0.1
    mesh.apply_scale(mesh_scale)
    
    # 2. CENTER THE MESH EXACTLY LIKE SCRIPT 1
    # GraspGen centers based on sampled points, not just the vertices
    # but centering vertices is usually close enough:
    center_transform = tra.translation_matrix(-mesh.vertices.mean(axis=0))
    # mesh.apply_transform(center_transform)
    
    visualize_mesh(vis, "object", mesh)

    with open(grasp_yaml, "r") as f:
        data = yaml.safe_load(f)

    gripper_depth = 0.1034  # From config

    for i, (k, grasp) in enumerate(data["grasps"].items()):
        # Get both base and TCP transforms for comparison
        T_base = grasp_to_T(grasp, apply_tcp_offset=False)
        T_tcp = grasp_to_T(grasp, apply_tcp_offset=True, gripper_depth=gripper_depth)
        
        # Visualize base frame (original GraspGen pose)
        make_frame(vis, f"grasps/{k}/frame_base", T_base, h=0.05, r=0.002)
        visualize_gripper(vis, f"grasps/{k}/gripper", T_base)
        
        # Visualize TCP frame (what you actually want to use)
        make_frame(vis, f"grasps/{k}/frame_tcp", T_tcp, h=0.08, r=0.003)
        visualize_gripper_as_sphere(vis, f"grasps/{k}/tcp_sphere", T_tcp, radius=0.008, color=0x00FF00)
        
    print(f"Visualized {len(data['grasps'])} grasps")
    print("Red frames = gripper base (GraspGen output)")
    print("RGB frames + green spheres = TCP positions (what you want)")



def main():
    grasp_yaml = "/home/arthur/Desktop/Projects/PAMI/object_following_ocp/results/campbells2/campbells2_grasps_filtered.yml"
    object_mesh_path = "/home/arthur/Desktop/Projects/PAMI/object_following_ocp/ressources/video_data/d02/campbells2/102ae42bff9940fc8f0b12c8a53d2675/102ae42bff9940fc8f0b12c8a53d2675.obj"
    gripper_depth = 0.0465  # From your config file

    vis = create_visualizer(clear=True)

    # Load mesh with same processing as GraspGen
    mesh = trimesh.load(object_mesh_path, force="mesh")
    mesh.apply_scale(0.0681) 
    
    visualize_mesh(vis, "object", mesh)

    with open(grasp_yaml, "r") as f:
        data = yaml.safe_load(f)

    for i, (k, grasp) in enumerate(data["grasps"].items()):
        # Convert to TCP frame
        T_tcp = grasp_to_T(grasp, apply_tcp_offset=True, gripper_depth=gripper_depth)
        
        # Visualize TCP position
        make_frame(vis, f"grasps/{k}/frame", T_tcp)
        visualize_gripper_as_sphere(vis, f"grasps/{k}/gripper", T_tcp)
    
    print(f"Visualized {len(data['grasps'])} TCP positions")

if __name__ == "__main__":
    main()
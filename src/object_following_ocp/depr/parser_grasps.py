import argparse
import yaml
import numpy as np
import pinocchio as pin


# -------------------------
# SE(3) distance
# -------------------------

def se3_distance(T1, T2, position_weight=1.0, rotation_weight=0.3):
    """
    Distance between two SE3 poses using Pinocchio.
    """
    # Relative transform
    T_rel = T1.inverse() * T2

    # Log map
    xi = pin.log6(T_rel).vector  # [v, omega]

    v = xi[:3]
    omega = xi[3:]

    pos_dist = np.linalg.norm(v)
    rot_dist = np.linalg.norm(omega)

    return position_weight * pos_dist + rotation_weight * rot_dist


def furthest_point_sampling(poses, k, position_weight=1.0, rotation_weight=0.3):
    N = len(poses)
    k = min(k, N)

    selected = [np.random.randint(N)]
    distances = np.full(N, np.inf)

    for _ in range(1, k):
        last = poses[selected[-1]]
        for i in range(N):
            d = se3_distance(
                poses[i],
                last,
                position_weight,
                rotation_weight,
            )
            distances[i] = min(distances[i], d)

        distances[selected] = -1.0
        selected.append(int(np.argmax(distances)))

    return np.array(selected)


# -------------------------
# YAML <-> Pinocchio SE3
# -------------------------

def grasp_to_se3(grasp):
    t = np.array(grasp["position"], dtype=np.float64)

    qw = grasp["orientation"]["w"]
    qx, qy, qz = grasp["orientation"]["xyz"]

    quat = pin.Quaternion(qw, qx, qy, qz)
    quat.normalize()

    R = quat.toRotationMatrix()

    return pin.SE3(R, t)


def se3_to_grasp(T, confidence):
    quat = pin.Quaternion(T.rotation)

    return {
        "confidence": float(confidence),
        "position": T.translation.tolist(),
        "orientation": {
            "w": float(quat.w),
            "xyz": [float(quat.x), float(quat.y), float(quat.z)],
        },
    }


def load_isaac_yaml(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    poses = []
    confidences = []

    for g in data["grasps"].values():
        poses.append(grasp_to_se3(g))
        confidences.append(g["confidence"])

    return poses, np.array(confidences), data


def save_isaac_yaml(path, template, poses, confidences):
    out = {
        "format": template["format"],
        "format_version": template["format_version"],
        "grasps": {},
    }

    for i, (T, c) in enumerate(zip(poses, confidences)):
        out["grasps"][f"grasp_{i}"] = se3_to_grasp(T, c)

    with open(path, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_yaml", type=str, required=True)
    parser.add_argument("--output_yaml", type=str, required=True)
    parser.add_argument("--num_grasps", type=int, required=True)
    parser.add_argument("--top_fraction", type=float, default=0.5)
    parser.add_argument("--pos_weight", type=float, default=1.0)
    parser.add_argument("--rot_weight", type=float, default=0.3)
    args = parser.parse_args()

    poses, conf, template = load_isaac_yaml(args.input_yaml)

    if args.top_fraction < 1.0:
        N = len(conf)
        k = max(1, int(N * args.top_fraction))
        idx = np.argsort(conf)[-k:]
        poses = [poses[i] for i in idx]
        conf = conf[idx]

    fps_idx = furthest_point_sampling(
        poses,
        args.num_grasps,
        position_weight=args.pos_weight,
        rotation_weight=args.rot_weight,
    )

    poses_sel = [poses[i] for i in fps_idx]
    conf_sel = conf[fps_idx]

    save_isaac_yaml(args.output_yaml, template, poses_sel, conf_sel)

    print(f"Saved {len(poses_sel)} diverse grasps to {args.output_yaml}")


if __name__ == "__main__":
    main()

import numpy as np
import pinocchio as pin


class Trajectory:
    def __init__(self, poses: list[pin.SE3]) -> None:
        self.poses = poses
        self.T = len(poses)

    def transform(self, T: pin.SE3) -> "Trajectory":
        return Trajectory([T * p for p in self.poses])
    
    def __getitem__(self, index: int) -> pin.SE3:
        return self.poses[index]

    def __len__(self) -> int:
        return self.T



def se3_sinusoid_trajectory(
    T0: pin.SE3,
    Tf: pin.SE3,
    T: int,
    amplitude: float = 0.05,
):
    """
    SE3 trajectory with a spatial sinusoidal deviation.

    Args:
        T0 initial SE3
        Tf final SE3
        T  number of knots
        amplitude sinusoid amplitude in meters

    Returns:
        list[pin.SE3]
    """

    traj = []

    p0 = T0.translation
    pf = Tf.translation

    R0 = T0.rotation
    Rf = Tf.rotation

    # Main direction
    d = pf - p0
    d_norm = np.linalg.norm(d)
    assert d_norm > 1e-6

    d_hat = d / d_norm

    # Pick an arbitrary orthogonal direction
    ref = np.array([0.0, 0.0, 1.0])
    if abs(d_hat @ ref) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    n = np.cross(d_hat, ref)
    n /= np.linalg.norm(n)

    # Rotation interpolation
    R_rel = R0.T @ Rf
    rotvec_rel = pin.log3(R_rel)

    for k in range(T):
        s = k / (T - 1)

        # Linear progression
        p_lin = p0 + s * d

        # Sinusoidal spatial offset
        offset = amplitude * np.sin(np.pi * s) * n

        p = p_lin + offset

        # Orientation
        R = R0 @ pin.exp3(s * rotvec_rel)

        traj.append(pin.SE3(R, p))

    return Trajectory(traj)
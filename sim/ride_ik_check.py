"""Spawn filter: will the wrist (joint 7) stay clear of its limits on this throw?

Numpy Franka kinematics (published DH table, flange 107 mm, hand −45°, TCP
103.4 mm; matches the sim's logged TCP to < 0.001 mm). ``ride_j7_margin`` replays
the expert's motion as DiffIK would (one damped least-squares step per tick,
hand-body Jacobian, TCP pose error, λ = 0.01): home → chase to the meeting point
at the measured chase speed → ride along the object path with the pads-across
yaw. Returns the smallest distance (rad) from j7 to either limit.

Why: pads-across yaw on some paths asks j7 for more turn than it has; the drive
then saturates and the hand drifts (step-1 probe: apple s1003, avocado s1008,
peach s1044 pinned at 2.897 rad).
"""
from __future__ import annotations

import numpy as np

_DH = (  # (a, d, alpha) per joint, Craig convention
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -np.pi / 2),
    (0.0, 0.316, np.pi / 2),
    (0.0825, 0.0, np.pi / 2),
    (-0.0825, 0.384, -np.pi / 2),
    (0.0, 0.0, np.pi / 2),
    (0.088, 0.0, np.pi / 2),
)
J7_LIMIT = 2.8973
Q_LIMITS = np.array(
    [[-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973], [-3.0718, -0.0698],
     [-2.8973, 2.8973], [-0.0175, 3.7525], [-2.8973, 2.8973]]
)


def _link(a: float, d: float, al: float, th: float) -> np.ndarray:
    ca, sa, ct, st = np.cos(al), np.sin(al), np.cos(th), np.sin(th)
    return np.array(
        [[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1]]
    )


def hand_fk(q: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """panda_hand pose (4×4) and the 7 joint frames (for the Jacobian)."""
    m = np.eye(4)
    frames = []
    for (a, d, al), th in zip(_DH, q):
        m = m @ _link(a, d, al, th)
        frames.append(m.copy())
    return m @ _link(0.0, 0.107, 0.0, 0.0) @ _link(0.0, 0.0, 0.0, -np.pi / 4), frames


def _jacobian(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Geometric Jacobian (6×7) of the hand-body origin, world frame; and hand pose."""
    hand, frames = hand_fk(q)
    p = hand[:3, 3]
    J = np.zeros((6, 7))
    for i, f in enumerate(frames):
        z, o = f[:3, 2], f[:3, 3]
        J[:3, i] = np.cross(z, p - o)
        J[3:, i] = z
    return J, hand


def _rotvec(r: np.ndarray) -> np.ndarray:
    c = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
    ang = np.arccos(c)
    if ang < 1e-9:
        return np.zeros(3)
    v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if np.pi - ang < 1e-6:  # 180°: axis from the symmetric part
        w, vecs = np.linalg.eigh(0.5 * (r + r.T))
        return ang * vecs[:, np.argmax(w)]
    return ang * v / (2.0 * np.sin(ang))


def quat_to_mat(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q_wxyz, dtype=np.float64)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def diffik_step(q: np.ndarray, p_tcp: np.ndarray, r_des: np.ndarray, tcp: float, lam: float = 0.01) -> np.ndarray:
    """One DiffIK update as the scene runs it (absolute pose, DLS), joints clamped."""
    J, hand = _jacobian(q)
    p_now = hand[:3, 3] + hand[:3, 2] * tcp
    err = np.r_[p_tcp - p_now, _rotvec(r_des @ hand[:3, :3].T)]
    dq = J.T @ np.linalg.solve(J @ J.T + lam * lam * np.eye(6), err)
    return np.clip(q + dq, Q_LIMITS[:, 0], Q_LIMITS[:, 1])


def replay_j7(q0: np.ndarray, poses: list[tuple[np.ndarray, np.ndarray]], tcp: float) -> np.ndarray:
    """j7 after each commanded (TCP position, rotation matrix) tick."""
    q = np.asarray(q0, dtype=np.float64)[:7].copy()
    out = []
    for p, r in poses:
        q = diffik_step(q, p, r, tcp)
        out.append(q[6])
    return np.array(out)


def ride_j7_margin(
    *,
    q_home: np.ndarray,
    home_quat_wxyz: np.ndarray,
    pad_quat_wxyz: np.ndarray,
    spawn_xyz: np.ndarray,
    vel_xy: np.ndarray,
    tcp: float,
    dt: float,
    wait_s: float,
    chase_speed: float,
    ride_s: float,
    keep_r_min: float,
) -> dict:
    """Smallest j7 distance to a limit over the chase + ride of one throw.

    Meeting point: first path point the hand can reach at ``chase_speed`` (from
    home TCP, after ``wait_s``). Ride: ``ride_s`` along the path from there,
    stopping where the path enters r < ``keep_r_min`` (the reach-stop zone).
    """
    hand, _ = hand_fk(np.asarray(q_home, dtype=np.float64)[:7])
    p_home = hand[:3, 3] + hand[:3, 2] * tcp
    v = np.array([float(vel_xy[0]), float(vel_xy[1]), 0.0])
    s0 = np.asarray(spawn_xyz, dtype=np.float64)
    r_pad = quat_to_mat(pad_quat_wxyz)
    # meeting time: hand travel ≤ object time (fixed point on a 1-tick grid)
    t_meet = None
    for k in range(1, 400):
        t = k * dt
        x = s0 + v * t
        if float(np.linalg.norm(x - p_home)) / max(chase_speed, 1e-6) + wait_s <= t:
            t_meet = t
            break
    if t_meet is None:
        return {"margin": float("nan"), "t_meet": None, "j7": []}
    poses: list[tuple[np.ndarray, np.ndarray]] = []
    n_wait = int(round(wait_s / dt))
    r_home = quat_to_mat(home_quat_wxyz)
    poses += [(p_home, r_home)] * n_wait
    meet = s0 + v * t_meet
    n_chase = max(1, int(round((t_meet - wait_s) / dt)))
    for k in range(1, n_chase + 1):
        poses.append((p_home + (meet - p_home) * k / n_chase, r_pad))
    for k in range(1, int(round(ride_s / dt)) + 1):
        x = meet + v * k * dt
        if float(np.hypot(x[0], x[1])) < keep_r_min:
            break
        poses.append((x, r_pad))
    j7 = replay_j7(q_home, poses, tcp)
    ride_j7 = j7[n_wait + n_chase - 1 :]
    margin = float(np.min(J7_LIMIT - np.abs(ride_j7)))
    return {"margin": margin, "t_meet": float(t_meet), "j7": ride_j7.tolist()}

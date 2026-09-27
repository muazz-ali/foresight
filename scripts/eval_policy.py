#!/usr/bin/env python3
"""Test a trained policy in Isaac (Gate G1 speed sweep).

Model A = no future numbers. Model B = with future numbers (constant-velocity).
`--zero-conditioning` turns B's future numbers off (sanity check).
`--oracle-conditioning` kept for P5 CLI; same CV pack as Model B (not bounce).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

FORESIGHT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = FORESIGHT_ROOT / "sim" / "phase0_cfg.yaml"

sys.path.insert(0, str(FORESIGHT_ROOT))
from eval.gate_g1 import G1_SPEEDS, HIGH_M_S, MID_M_S  # noqa: E402

G1_TRIALS = 20

# --force-place scripted stages (mirrors expert carry→lower→open→home).
_FP_IDLE = 0
_FP_CARRY = 1
_FP_LOWER = 2
_FP_UNGLUE = 3
_FP_OPEN = 4
_FP_HOME = 5
_FP_DONE = 6
# Soft-start only (right after attach). Place waypoints are full targets — DiffIK smooths.
_FP_SOFT_STEP_M = 0.02


def _step_toward(current: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    """Move at most ``max_step`` meters toward ``target`` (soft-start after attach)."""
    cur = np.asarray(current, dtype=np.float64).reshape(3)
    tgt = np.asarray(target, dtype=np.float64).reshape(3)
    delta = tgt - cur
    dist = float(np.linalg.norm(delta))
    if dist <= max_step or dist < 1e-9:
        return tgt.copy()
    return cur + delta * (max_step / dist)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Foresight G1 closed-loop policy eval")
    p.add_argument("--checkpoint", type=str, default="runs/p1_b/pretrained_model")
    p.add_argument("--model", choices=["A", "B"], default="B")
    p.add_argument("--zero-conditioning", action="store_true", help="B sanity: zero vector")
    p.add_argument(
        "--oracle-conditioning",
        action="store_true",
        help="B_oracle CLI flag; same constant-velocity pack as Model B",
    )
    p.add_argument(
        "--delta",
        type=float,
        default=None,
        help="conditioning look-ahead (s); default = state_machine_params.conditioning_delta_s",
    )
    p.add_argument("--trials-per-speed", type=int, default=G1_TRIALS)
    p.add_argument("--speeds", type=float, nargs="+", default=list(G1_SPEEDS))
    p.add_argument("-c", "--sim_cfg_file", default=str(DEFAULT_CFG))
    p.add_argument("-o", "--output_dir", default=str(FORESIGHT_ROOT / "data" / "p1" / "eval_b_zero"))
    p.add_argument("--seed", type=int, default=9000)
    p.add_argument("--object-usd", default=None)
    p.add_argument("--container-usd", default=None)
    p.add_argument("--debug", action="store_true", help="DEBUG logs + write stitched MP4 per episode")
    p.add_argument(
        "--save-conditioning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-episode .h5 + .json (object motion, cond, cameras if debug). Default on.",
    )
    p.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="How many actions to execute from each policy chunk before replan "
        "(default: value in checkpoint, usually 50). Plan target ~5–8 (~200 ms).",
    )
    p.add_argument(
        "--compare-report",
        type=str,
        default=None,
        help="Optional path to Model A eval report JSON for G1 margin check",
    )
    p.add_argument(
        "--no-attach",
        action="store_true",
        help="Diagnostic: never weld/glue the object; rely on finger close + PhysX only",
    )
    p.add_argument(
        "--force-place",
        action="store_true",
        help="Diagnostic: after lift, servo to drop pose then open+freeze (tests release/drop)",
    )
    return p


def _setup_logging(output_dir: Path, *, debug: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if debug else logging.INFO
    fmt = logging.Formatter("[%(levelname)s] %(asctime)s %(message)s")
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    for h in (
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(output_dir / "eval_policy.log", mode="a"),
    ):
        h.setLevel(level)
        h.setFormatter(fmt)
        root.addHandler(h)


def _as_chw_float(img: np.ndarray) -> torch.Tensor:
    """HWC uint8/float → CHW float32 in [0,1] for SmolVLA."""
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and float(np.nanmax(arr) if arr.size else 1) <= 1.5:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t


def _lerobot_device_tag(device: str) -> str:
    """LeRobot config accepts only cuda|mps|cpu (rejects cuda:N)."""
    d = str(device)
    if d.startswith("cuda"):
        return "cuda"
    if d.startswith("mps"):
        return "mps"
    return "cpu"


def load_policy(
    checkpoint: str,
    model: str,
    device: str,
    *,
    n_action_steps: int | None = None,
):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    # Training may have written device=cuda:N; rewrite so from_pretrained can parse.
    cfg_path = Path(checkpoint) / "config.json"
    if cfg_path.is_file():
        raw = json.loads(cfg_path.read_text())
        tag = _lerobot_device_tag(raw.get("device", device))
        if raw.get("device") != tag:
            raw["device"] = tag
            cfg_path.write_text(json.dumps(raw, indent=4) + "\n")

    policy = SmolVLAPolicy.from_pretrained(checkpoint)
    # chunk_size = how many actions the net predicts per forward.
    # n_action_steps = how many of those we execute before calling the net again.
    if n_action_steps is not None:
        n = int(n_action_steps)
        if n < 1:
            raise ValueError(f"n_action_steps must be >= 1, got {n}")
        chunk = int(policy.config.chunk_size)
        if n > chunk:
            raise ValueError(f"n_action_steps ({n}) > chunk_size ({chunk})")
        policy.config.n_action_steps = n
        logging.info(
            "eval replan: n_action_steps=%d (%.0f ms @ 25 Hz), chunk_size=%d",
            n,
            1000.0 * n / 25.0,
            chunk,
        )
    policy.to(device)
    policy.eval()
    return policy


def build_obs_batch(
    scene,
    *,
    model: str,
    delta: float,
    language: str,
    device: str,
    zero_conditioning: bool,
    oracle_conditioning: bool = False,  # unused: same CV pack as Model B
) -> tuple[dict, np.ndarray | None]:
    """One SmolVLA observation dict + 4-D cond (Model B) or None (A)."""
    from interfaces.state import CONDITIONING_DIM, conditioning_vector
    from policy.convert_h5 import pack_proprio

    imgs = scene.get_images()
    ee_pos, ee_quat = scene.get_ee_pose()
    # Same channel as collect/convert: commanded open∈{0,1}, not raw finger width.
    # (Fingers are snapped to this command on attach; see Phase0Scene._snap_fingers.)
    grip_cmd = float(scene.get_gripper_open())
    state = pack_proprio(ee_pos, ee_quat, grip_cmd)
    cond: np.ndarray | None = None

    if model == "B":
        if zero_conditioning:
            cond = np.zeros(CONDITIONING_DIM, dtype=np.float32)
        else:
            obj = scene.get_object_state()
            cond = conditioning_vector(obj, delta=delta).astype(np.float32)
        state = np.concatenate([state, cond], axis=0).astype(np.float32)

    batch = {
        "observation.state": torch.as_tensor(state, device=device).unsqueeze(0),
        "observation.images.static_cam": _as_chw_float(imgs["static_cam"]).unsqueeze(0).to(device),
        "observation.images.wrist_cam": _as_chw_float(imgs["wrist_cam"]).unsqueeze(0).to(device),
        "task": [language if language.endswith("\n") else language + "\n"],
    }
    return batch, cond


def action_to_ee(action: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Unpack policy action → (pos3, quat_wxyz4, gripper_open01)."""
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    pos = a[0:3]
    quat = a[3:7]
    # Normalize quat.
    n = np.linalg.norm(quat)
    if n < 1e-8:
        quat = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        quat = quat / n
    grip_open = float(a[7] > 0.5) if a.shape[0] > 7 else 1.0
    return pos, quat, grip_open


def run_episode_policy(
    scene,
    policy,
    *,
    seed: int,
    speed: float,
    model: str,
    delta: float,
    device: str,
    zero_conditioning: bool,
    n_steps: int,
    save_images: bool = False,
    oracle_conditioning: bool = False,
    no_attach: bool = False,
    force_place: bool = False,
) -> tuple[dict, dict]:
    """Closed-loop policy rollout. Logs the same arrays as expert collect."""
    from policy.features import default_language

    if zero_conditioning and oracle_conditioning:
        raise ValueError("Pass only one of --zero-conditioning / --oracle-conditioning")

    meta = scene.reset_episode(seed=seed, speed=speed)
    language = default_language(meta.get("category"), meta.get("container_category"))
    meta["language"] = language
    meta["no_attach"] = bool(no_attach)
    meta["force_place"] = bool(force_place)

    log: dict[str, list] = {
        "holding": [],
        "object_pos": [],
        "object_vel": [],
        "ee_pos": [],
        "ee_quat": [],
        "proprio": [],
        "action_pos": [],
        "action_quat": [],
        "gripper": [],
        "oracle_state": [],
        "conditioning": [],
        "timestamp": [],
        "container_pos": [],
        "policy_replan": [],
    }
    if save_images:
        log["static_cam_rgb"] = []
        log["wrist_cam_rgb"] = []
    policy.reset()
    attached = False
    gripping = False  # --no-attach: fingers closed near object (physics hold)
    placed = False  # opened after a hold (release/place done)
    lift_frames = 0
    close_dwell = 0  # frames closed+near before attach (smooth pick)
    fp_stage = _FP_IDLE
    fp_timer = 0
    # LeRobot ACTION queue: empty ⇒ select_action runs a forward; else pops queued action.
    from lerobot.constants import ACTION

    drop_pos = np.asarray(scene.get_drop_target(), dtype=np.float64).reshape(3)
    carry_pose = np.asarray(scene.get_carry_pose(), dtype=np.float64).reshape(-1)
    place_xy = carry_pose[:2].copy()
    retract_z = float(max(carry_pose[2], drop_pos[2] + 0.08))
    carry_xyz = np.array([place_xy[0], place_xy[1], retract_z], dtype=np.float64)
    home_xyz = np.asarray(
        scene.cfg.get("robot", {}).get("init_pose", [0.45, 0.0, 0.40])[:3],
        dtype=np.float64,
    )
    down_quat = np.asarray(
        scene.cfg.get("robot", {}).get("init_pose", [0.45, 0.0, 0.40, 0.0, 1.0, 0.0, 0.0])[3:7],
        dtype=np.float64,
    )
    if down_quat.size != 4:
        down_quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

    for t in range(n_steps):
        will_replan = len(policy._queues[ACTION]) == 0
        batch, cond = build_obs_batch(
            scene,
            model=model,
            delta=delta,
            language=language,
            device=device,
            zero_conditioning=zero_conditioning,
            oracle_conditioning=oracle_conditioning,
        )
        with torch.inference_mode():
            act = policy.select_action(batch)
        act_np = act[0].detach().cpu().numpy()
        pos, quat, grip = action_to_ee(act_np)

        # Grasp latch: close near object → attach (or physics grip if --no-attach).
        obj = scene.get_object_state()
        ee, ee_q = scene.get_ee_pose()
        dist = float(np.linalg.norm(ee - obj.position))
        want_close = grip < 0.5

        holding_now = attached if not no_attach else gripping
        if holding_now and float(obj.position[2]) >= 0.12:
            lift_frames += 1

        # Smooth pick: short closed+near dwell before welding (avoid instant snap).
        attach_dist = 0.05
        attach_dwell_need = 3 if force_place else 1
        if want_close and dist < attach_dist and not attached and not placed:
            close_dwell += 1
        else:
            close_dwell = 0

        # Scripted place: carry → lower → unglue → open → home.
        # Waypoints are full targets (DiffIK smooths). Soft-step only right after attach.
        if force_place and not no_attach:
            if fp_stage == _FP_IDLE and attached:
                pos = _step_toward(ee, pos, _FP_SOFT_STEP_M)
                want_close, grip = True, 0.0
                fp_timer += 1
                if lift_frames >= 10 and fp_timer >= 8:
                    fp_stage = _FP_CARRY
                    fp_timer = 0
                    logging.info("force-place: carry to bowl")
            elif fp_stage == _FP_CARRY:
                pos = carry_xyz.copy()
                quat = down_quat.copy()
                want_close, grip = True, 0.0
                attached = True
                fp_timer += 1
                if (
                    float(np.linalg.norm(ee[:2] - carry_xyz[:2])) < 0.04
                    and abs(float(ee[2] - carry_xyz[2])) < 0.05
                ) or fp_timer > 120:
                    fp_stage = _FP_LOWER
                    fp_timer = 0
                    logging.info("force-place: lower into bowl")
            elif fp_stage == _FP_LOWER:
                pos = drop_pos.copy()
                quat = down_quat.copy()
                want_close, grip = True, 0.0
                attached = True
                fp_timer += 1
                xy_ok = float(np.linalg.norm(ee[:2] - drop_pos[:2])) < 0.04
                z_ok = abs(float(ee[2] - drop_pos[2])) < 0.035
                # Never unglue until XY is over the bowl (timeout alone caused table drops).
                if xy_ok and (z_ok or fp_timer > 80):
                    fp_stage = _FP_UNGLUE
                    fp_timer = 0
                    placed = True
                    attached = False
                    logging.info(
                        "force-place: unglue at drop (fingers closed) xy=%.1fcm z=%.1fcm",
                        100.0 * float(np.linalg.norm(ee[:2] - drop_pos[:2])),
                        100.0 * abs(float(ee[2] - drop_pos[2])),
                    )
                elif fp_timer > 200:
                    logging.warning("force-place: lower hard-timeout; unglue anyway")
                    fp_stage = _FP_UNGLUE
                    fp_timer = 0
                    placed = True
                    attached = False
            elif fp_stage == _FP_UNGLUE:
                pos = drop_pos.copy()
                quat = down_quat.copy()
                want_close, grip = True, 0.0
                attached = False
                placed = True
                fp_timer += 1
                if fp_timer >= 5:
                    fp_stage = _FP_OPEN
                    fp_timer = 0
                    logging.info("force-place: open gripper")
            elif fp_stage == _FP_OPEN:
                pos = drop_pos.copy()
                quat = down_quat.copy()
                want_close, grip = False, 1.0
                attached = False
                placed = True
                fp_timer += 1
                if fp_timer >= 8:
                    fp_stage = _FP_HOME
                    fp_timer = 0
                    logging.info("force-place: retract home")
            elif fp_stage in (_FP_HOME, _FP_DONE):
                if float(ee[2]) < retract_z - 0.03:
                    target = np.array([ee[0], ee[1], retract_z], dtype=np.float64)
                elif float(np.linalg.norm(ee[:2] - home_xyz[:2])) >= 0.04:
                    target = np.array(
                        [home_xyz[0], home_xyz[1], retract_z], dtype=np.float64
                    )
                else:
                    target = home_xyz.copy()
                pos = target
                quat = down_quat.copy()
                want_close, grip = False, 1.0
                attached = False
                placed = True
                fp_timer += 1
                if (
                    float(np.linalg.norm(ee[:2] - home_xyz[:2])) < 0.05
                    and abs(float(ee[2] - home_xyz[2])) < 0.08
                ) or fp_timer > 150:
                    if fp_stage != _FP_DONE:
                        logging.info("force-place: done at home")
                    fp_stage = _FP_DONE

        if no_attach:
            if (not placed) and (not gripping) and want_close and dist < attach_dist:
                if close_dwell >= attach_dwell_need:
                    gripping = True
            if gripping and (not want_close or dist > 0.10):
                if gripping and dist > 0.10 and want_close:
                    logging.info("no-attach slip at t=%d dist=%.3f m", t, dist)
                gripping = False
                if not want_close:
                    placed = True
            attach_flag = False
            freeze = bool(placed and not gripping)
            kin = (not gripping) and (not freeze)
            holding_log = float(gripping)
        else:
            if (
                (not placed)
                and (not attached)
                and want_close
                and dist < attach_dist
                and close_dwell >= attach_dwell_need
                and fp_stage < _FP_UNGLUE
            ):
                attached = True
            if force_place and fp_stage >= _FP_UNGLUE:
                attached = False
            elif (not force_place) and attached and not want_close:
                attached = False
                placed = True
            attach_flag = bool(attached)
            # Freeze once unglued/placed so the object stays in the bowl (no wobble).
            freeze = bool(
                (placed and not attached)
                or (force_place and fp_stage >= _FP_UNGLUE and not attached)
            )
            kin = (not attached) and (not freeze)
            holding_log = float(attached)

        scene.set_ee_target(pos, quat, grip)
        scene.step(
            kinematic_object=kin,
            attach_object=attach_flag,
            freeze_object=freeze,
        )

        st = scene.get_object_state()
        ee, ee_quat = scene.get_ee_pose()
        log["holding"].append(holding_log)
        log["object_pos"].append(st.position.copy())
        log["object_vel"].append(st.velocity.copy())
        log["ee_pos"].append(ee.copy())
        log["ee_quat"].append(ee_quat.copy())
        log["proprio"].append(scene.get_proprio())
        log["action_pos"].append(np.asarray(pos, dtype=np.float64).copy())
        log["action_quat"].append(np.asarray(quat, dtype=np.float64).copy())
        log["gripper"].append(grip)
        log["oracle_state"].append(st.to_flat())
        log["timestamp"].append(st.timestamp)
        ctr = scene.get_container_pose()
        log["container_pos"].append(
            ctr[0].copy() if ctr is not None else np.full(3, np.nan)
        )
        log["policy_replan"].append(1 if will_replan else 0)
        if cond is not None:
            log["conditioning"].append(cond.astype(np.float64).copy())
        if save_images:
            imgs = scene.get_images()
            log["static_cam_rgb"].append(imgs["static_cam"])
            log["wrist_cam_rgb"].append(imgs["wrist_cam"])

    episode = {k: np.asarray(v) for k, v in log.items() if len(v)}
    t_len = int(episode["object_pos"].shape[0])
    assert episode["object_pos"].shape == (t_len, 3), episode["object_pos"].shape
    if "conditioning" in episode:
        from interfaces.state import CONDITIONING_DIM, CONDITIONING_LAYOUT

        assert episode["conditioning"].shape == (t_len, CONDITIONING_DIM), episode[
            "conditioning"
        ].shape
    if "oracle_state" in episode:
        assert episode["oracle_state"].shape == (t_len, 14), episode["oracle_state"].shape
    meta["n_frames"] = t_len
    meta["delta"] = float(delta)
    meta["n_replans"] = int(episode["policy_replan"].sum())
    meta["placed_released"] = bool(placed)
    if model == "B" and episode["conditioning"].size:
        c = episode["conditioning"]
        replan = episode["policy_replan"].astype(bool)
        c_replan = c[replan] if replan.any() else c
        meta["conditioning_debug"] = {
            "layout": list(CONDITIONING_LAYOUT),
            "zero_conditioning": bool(zero_conditioning),
            "oracle_conditioning": bool(oracle_conditioning),
            "first": c[0].tolist(),
            "first_replan": c_replan[0].tolist(),
            "mean": c.mean(axis=0).tolist(),
            "mean_replan": c_replan.mean(axis=0).tolist(),
            "l2_mean": float(np.linalg.norm(c, axis=1).mean()),
            "l2_mean_replan": float(np.linalg.norm(c_replan, axis=1).mean()),
            "all_zero": bool(np.allclose(c, 0.0)),
        }
    return episode, meta


def episode_success_policy(episode: dict, carry_pose: np.ndarray | None = None) -> bool:
    """Policy success (no expert stages): lift + high z or near bowl."""
    from sim.success import episode_diagnostics

    diag = episode_diagnostics(episode)
    lifted = diag["z_max"] >= 0.12
    placed = diag["z_end"] >= 0.10
    if carry_pose is not None and "object_pos" in episode:
        final = episode["object_pos"][-1]
        xy = np.linalg.norm(final[:2] - np.asarray(carry_pose)[:2])
        placed = placed or (lifted and xy < 0.12)
    return bool(lifted and placed)


def _save_conditioning_trace(
    out_dir: Path,
    *,
    model: str,
    speed: float,
    seed: int,
    episode: dict,
    meta: dict,
    zero_conditioning: bool,
    oracle_conditioning: bool,
    success: bool,
    failure: str,
) -> Path | None:
    """Write lightweight JSON + H5 (same arrays as collect)."""
    from sim.collect import episode_for_json, write_episode_h5

    extra = {
        "delta": np.asarray([float(meta["delta"])], dtype=np.float32),
        "speed_m_s": np.asarray([float(speed)], dtype=np.float32),
        "seed": np.asarray([int(seed)], dtype=np.int64),
        "zero_conditioning": np.asarray([int(bool(zero_conditioning))], dtype=np.int8),
        "oracle_conditioning": np.asarray([int(bool(oracle_conditioning))], dtype=np.int8),
        "success": np.asarray([int(bool(success))], dtype=np.int8),
    }
    h5_path = out_dir / f"eval_{model}_v{speed:.2f}_s{seed}_cond.h5"
    write_episode_h5(h5_path, episode, extra=extra)

    n_frames = int(np.asarray(episode["object_pos"]).shape[0]) if "object_pos" in episode else 0
    side = {
        "seed": int(seed),
        "speed_m_s": float(speed),
        "robot": "franka",
        "success": bool(success),
        "failure": failure,
        "model": model,
        "n_frames": n_frames,
        "delta": float(meta["delta"]),
        "zero_conditioning": bool(zero_conditioning),
        "oracle_conditioning": bool(oracle_conditioning),
        "h5": str(h5_path),
        **{k: meta[k] for k in ("max_stage", "z_max", "z_end", "z_min") if k in meta},
        **episode_for_json(episode),
    }
    side_path = out_dir / f"eval_{model}_v{speed:.2f}_s{seed}_cond.json"
    side_path.write_text(json.dumps(side, indent=2, default=str))
    return h5_path


def eval_speed_bin(args, scene, policy, speed: float, metrics_path: Path) -> dict:
    from sim.success import classify_policy_failure, episode_diagnostics

    n = int(args.trials_per_speed)
    seed = int(args.seed) + int(round(speed * 1000))
    successes = 0
    t0 = time.time()
    device = args._device
    from sim.state_machine import inference_window_steps

    n_steps = inference_window_steps(scene.cfg, scene.dt)
    if bool(getattr(args, "force_place", False)):
        # Carry→lower→unglue→open→home needs more than a pick-only window.
        n_steps = max(int(n_steps), 400)
    save_images = bool(args.debug)
    save_cond = bool(getattr(args, "save_conditioning", True))
    out_dir = Path(args.output_dir)

    for i in range(n):
        logging.info("eval model=%s speed=%.2f trial %d/%d seed=%d", args.model, speed, i + 1, n, seed)
        try:
            episode, meta = run_episode_policy(
                scene,
                policy,
                seed=seed,
                speed=speed,
                model=args.model,
                delta=float(args.delta),
                device=device,
                zero_conditioning=bool(args.zero_conditioning),
                oracle_conditioning=bool(args.oracle_conditioning),
                n_steps=n_steps,
                save_images=save_images,
                no_attach=bool(getattr(args, "no_attach", False)),
                force_place=bool(getattr(args, "force_place", False)),
            )
            ok = episode_success_policy(episode, carry_pose=np.asarray(meta.get("carry_pose")))
        except Exception as ex:
            logging.exception("eval episode failed: %s", ex)
            episode = {}
            ok = False
            meta = {}

        if ok:
            successes += 1
        diag = episode_diagnostics(episode if episode else None)
        holding = None
        if episode:
            holding = episode.get("holding")
        tax = classify_policy_failure(
            success=ok,
            holding=holding,
            object_z_max=diag["z_max"],
        )
        cond_path = None
        if save_cond and episode:
            cond_path = _save_conditioning_trace(
                out_dir,
                model=args.model,
                speed=speed,
                seed=seed,
                episode=episode,
                meta=meta,
                zero_conditioning=bool(args.zero_conditioning),
                oracle_conditioning=bool(args.oracle_conditioning),
                success=ok,
                failure=tax,
            )
        rec = {
            "speed_m_s": speed,
            "seed": seed,
            "success": ok,
            "failure": tax,
            "model": args.model,
            "zero_conditioning": bool(args.zero_conditioning),
            "oracle_conditioning": bool(args.oracle_conditioning),
            "no_attach": bool(getattr(args, "no_attach", False)),
            "force_place": bool(getattr(args, "force_place", False)),
            "placed_released": meta.get("placed_released"),
            "n_replans": meta.get("n_replans"),
            **diag,
        }
        if meta.get("conditioning_debug"):
            rec["conditioning_debug"] = meta["conditioning_debug"]
        if cond_path is not None:
            rec["h5"] = str(cond_path)
        with open(metrics_path, "a") as fp:
            fp.write(json.dumps(rec) + "\n")

        if save_images and episode:
            from sim.video import write_debug_mp4

            fps = int(scene.cfg.get("camera", {}).get("fps", 25))
            tag = "ok" if ok else tax
            mp4 = out_dir / f"eval_{args.model}_v{speed:.2f}_s{seed}_{tag}.mp4"
            write_debug_mp4(episode, mp4, fps=fps)

        seed += 1

    elapsed = time.time() - t0
    rate = successes / n if n else 0.0
    return {
        "speed_m_s": speed,
        "attempts": n,
        "successes": successes,
        "success_rate": rate,
        "elapsed_s": elapsed,
    }


def build_g1_report(summaries: list[dict], args, wall: float) -> dict:
    """Partial G1 table for one arm (full pass/fail is eval/gate_g1.py)."""
    by_speed = {s["speed_m_s"]: s for s in summaries}
    rate0 = by_speed.get(0.0, {}).get("success_rate")
    rate_hi = by_speed.get(HIGH_M_S, {}).get("success_rate")
    rate_mid = by_speed.get(MID_M_S, {}).get("success_rate")
    if rate_mid is None:
        rate_mid = by_speed.get(HIGH_M_S, {}).get("success_rate")

    flat_ok = None
    if rate0 is not None and rate_hi is not None:
        flat_ok = (rate0 - rate_hi) < MID_M_S

    compare = None
    margin_ok = None
    if args.compare_report:
        other = json.loads(Path(args.compare_report).read_text())
        other_mid = None
        for s in other.get("summaries", []):
            if abs(s["speed_m_s"] - MID_M_S) < 1e-9 or abs(s["speed_m_s"] - HIGH_M_S) < 1e-9:
                other_mid = s["success_rate"]
                if abs(s["speed_m_s"] - MID_M_S) < 1e-9:
                    break
        if other_mid is not None and rate_mid is not None:
            margin = rate_mid - other_mid
            margin_ok = margin >= HIGH_M_S
            compare = {"other_mid_rate": other_mid, "this_mid_rate": rate_mid, "margin": margin}

    # Zero-out sanity is a separate run; recorded via flag.
    report = {
        "gate": "G1",
        "model": args.model,
        "zero_conditioning": bool(args.zero_conditioning),
        "oracle_conditioning": bool(args.oracle_conditioning),
        "checkpoint": args.checkpoint,
        "criterion": {
            "B_beats_A_mid_ge": HIGH_M_S,
            "B_flat_drop_0_to_20_lt": MID_M_S,
            "mid_speed_m_s": MID_M_S,
        },
        "summaries": summaries,
        "flat_pass": flat_ok,
        "margin_vs_A": compare,
        "margin_pass": margin_ok,
        "wall_time_s": wall,
        "n_action_steps": (
            int(args.n_action_steps)
            if args.n_action_steps is not None
            else None
        ),
    }
    if args.model == "B" and not args.zero_conditioning and not args.oracle_conditioning:
        report["gate_g1_partial"] = {
            "flat_pass": flat_ok,
            "margin_pass": margin_ok,
            "note": "Full G1 needs Model A compare report + zero-out collapse check",
        }
    return report


def main() -> None:
    print("[foresight] G1 eval — starting AppLauncher…", flush=True)
    from isaaclab.app import AppLauncher

    sys.argv = [a for a in sys.argv if a != ""]
    parser = _build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[foresight] Ignoring unknown args: {unknown}", flush=True)

    if bool(args.zero_conditioning) and bool(args.oracle_conditioning):
        raise SystemExit("Choose only one of --zero-conditioning / --oracle-conditioning")

    args.enable_cameras = True
    app_launcher = AppLauncher(args)
    out_dir = Path(args.output_dir)
    _setup_logging(out_dir, debug=bool(args.debug))

    device = getattr(args, "device", None) or ("cuda:0" if torch.cuda.is_available() else "cpu")
    args._device = device

    from interfaces.config import conditioning_delta_s
    from sim.scene import Phase0Scene, load_cfg

    cfg = load_cfg(args.sim_cfg_file)
    if args.delta is None:
        args.delta = conditioning_delta_s(cfg)
    logging.info("eval conditioning Δ=%.3f s (from %s)", args.delta, args.sim_cfg_file)
    if getattr(args, "device", None):
        cfg.setdefault("sim", {})["device"] = args.device

    scene = Phase0Scene(
        cfg,
        enable_cameras=True,
        object_usd=args.object_usd,
        container_usd=args.container_usd,
        asset_seed=int(args.seed),
    )
    policy = load_policy(
        args.checkpoint,
        args.model,
        device,
        n_action_steps=args.n_action_steps,
    )

    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    summaries = []
    wall0 = time.time()
    for speed in args.speeds:
        summaries.append(eval_speed_bin(args, scene, policy, float(speed), metrics_path))
    wall = time.time() - wall0
    report = build_g1_report(summaries, args, wall)
    report_path = out_dir / "gate_g1_report.json"
    with open(report_path, "w") as fp:
        json.dump(report, fp, indent=2)
    logging.info("Wrote %s", report_path)
    print(json.dumps(report, indent=2))
    os._exit(0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Test a trained policy in Isaac.

WHAT THIS FILE DOES
  Load a trained SmolVLA checkpoint. Drop a moving object in the sim at a few
  belt speeds (default 0 / 10 / 15 / 20 cm/s). Let the policy try to pick it up
  and put it in the bowl. Count how often it works at each speed.

WHAT THE POLICY SEES
  Two camera images, the hand state, and a 4-number pack: where the object
  will be (x, y) and how fast it moves (vx, vy), looked ahead by delta seconds
  (constant-velocity guess).

  `--cond-source oracle|filter` picks where those 4 numbers come from.
    oracle = simulator truth (default). This is Gate G1.
    filter = static camera + Kalman filter. This is Gate G2b (filter swap).
  `--zero-conditioning` sets the 4 numbers to all zeros. It is the sanity
    check: if the policy really uses the numbers, success should drop a lot.
    It cannot be mixed with `--cond-source filter`.

WHAT SUCCESS LOOKS LIKE (see foresight_plan.md for the real gate text)
  G1 needs two runs of this script on the same checkpoint:
    (a) normal run: success stays flat from 0 to 20 cm/s (drop < 15 points).
    (b) `--zero-conditioning` run: success at mid speed falls by >= 15 points.
  This script only writes the per-speed table (gate_g1_report.json) and one
  line per episode (metrics.jsonl). 
  The pass/fail verdict is made by scripts/gate_g1.py, which merges the (a) and (b) reports.
  Sample size: trials n >= 80 per speed before you claim G1 passed or failed.

FAILURE LABELS (policy only)
  The code currently stores: never-held | held-no-lift | lifted-no-place |
  success. The plan calls the first two never-glued and glued-no-lift.
  "Held" means welded to the hand, or the diagnostic finger hold with
  --no-attach. Labels use the 0/1 holding log and max object height via
  sim.success.classify_policy_failure.
  We do not use the expert stage labels here.
  Physics-grasp (house / throw) episodes also get a timing label
  (eval/timing_labels.py): never-closed | off-table | aim | early | late |
  short-close | success. Gate G3 scores those with eval/gate_g3.py.

SEEDING
  Every episode seeds python/numpy/torch from the episode seed, and the
  SmolVLA flow-matching noise is drawn from its own seeded generator. 
  Same seed + same flags = same rollout (see --torch-deterministic for GPU kernels).

TIMING OVERLAYS (Phase 3, eval-time only; all default off = old behaviour)
  --reflex on            ReflexClose steers XY + closes in the last ~10 cm.
                         Close time = robot.gripper_close_time_s in the config.
  --action-buffer on     timestamped ActionBuffer: stale chunk slots are dropped.
  --inject-latency-s L   the policy plans from an observation L s old. Buffer
                         off: its chunk still plays from the first action (late
                         by L). Buffer on: slots already in the past are skipped.
"""

from __future__ import annotations

import argparse
import json
import inspect
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

FORESIGHT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = FORESIGHT_ROOT / "sim" / "phase0_cfg.yaml"

sys.path.insert(0, str(FORESIGHT_ROOT))
from eval.gate_g1 import G1_SPEEDS, HIGH_M_S, MID_M_S  # noqa: E402
from interfaces.state import ObjectState  # noqa: E402

# (--trials-per-speed 80) before claiming G1 pass or fail.
G1_TRIALS = 100
# Filter arm: finger gap above this with the gripper commanded closed = holding.
# Fruit holds at 60–67 mm; an empty close or a slip reads under 2 mm.
HELD_MIN_GAP_M = 0.010

# --force-place is a DEBUG helper, not part of the G1 score. After the policy
# picks the object, a simple script (not the policy) carries it to the bowl and
# lets go. It tests "does release/drop work?" on its own.
# These are the script's steps (mirrors expert carry→lower→open→home).
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
    """All CLI flags. Isaac's own launcher flags (--headless etc.) are added in main()."""
    p = argparse.ArgumentParser(description="Foresight G1 closed-loop policy eval")
    p.add_argument("--checkpoint", type=str, default="runs/p3_throw/pretrained_model")
    p.add_argument("--zero-conditioning", action="store_true", help="Sanity: zero conditioning vector")
    p.add_argument(
        "--cond-shift-m",
        type=float,
        nargs=2,
        default=None,
        metavar=("DX", "DY"),
        help="Claim test: add (DX, DY) m to p̂ the model sees. If the model uses the "
        "numbers, the hand should miss by about this much.",
    )
    p.add_argument(
        "--cond-source",
        choices=["oracle", "filter"],
        default="filter",
        help="Where the 4 conditioning numbers come from. oracle = sim truth (default). "
        "filter = static camera + Kalman. Filter runs warm up 2 frames before the first plan.",
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
    p.add_argument("-o", "--output_dir", default=str(FORESIGHT_ROOT / "data" / "p1_throw" / "eval"))
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
        "--no-attach",
        action="store_true",
        help="Diagnostic: never weld/glue the object; rely on finger close + PhysX only",
    )
    p.add_argument(
        "--force-place",
        action="store_true",
        help="Diagnostic: after lift, servo to drop pose then open+freeze (tests release/drop)",
    )
    # Phase 3 timing overlays. Defaults = old eval behaviour.
    p.add_argument("--reflex", choices=["on", "off"], default="off",
                   help="ReflexClose overlay in the last ~10 cm (G3 arm). Default off.")
    p.add_argument("--action-buffer", choices=["on", "off"], default="off",
                   help="Timestamped ActionBuffer: drop chunk slots whose time has passed.")
    p.add_argument("--inject-latency-s", type=float, default=0.0,
                   help="Simulated inference delay (s): plan from an observation this old.")
    p.add_argument("--held-from-hand", choices=["on", "off"], default="on",
                   help="Filter arm: while the fingers hold something, the 4 numbers come "
                   "from the hand (held object rides with it). off = raw filter throughout.")
    # Wrist depth ≈ 1.25 × (hand − object centre height) + 7.8 cm in the demos
    # (σ 0.7 cm); the expert closes at ~8.4 cm. 9.0 cm ≈ hand ≤ 1 cm above centre.
    p.add_argument("--close-gate", choices=["on", "off"], default="off",
                   help="Hold the policy's first close until wrist depth and tracked XY say "
                   "the hand is in place (times out).")
    p.add_argument("--close-gate-depth-m", type=float, default=0.090)
    p.add_argument("--close-gate-xy-m", type=float, default=0.012)
    p.add_argument("--close-gate-max-wait", type=int, default=8, help="frames (25 Hz)")
    p.add_argument("--torch-deterministic", action="store_true",
                   help="torch.use_deterministic_algorithms + cuDNN deterministic (slower).")
    return p


def seed_episode(seed: int) -> None:
    """Seed python, numpy and torch (CPU + every GPU) from the episode seed."""
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def gripper_close_time(cfg: dict) -> tuple[float, str]:
    """One gripper close time for the whole stack: robot.gripper_close_time_s.

    Missing key: fall back to ReflexClose's default and say so in the log."""
    val = (cfg.get("robot") or {}).get("gripper_close_time_s")
    if val is not None:
        return float(val), "robot.gripper_close_time_s"
    from control.reflex import ReflexClose

    default = float(inspect.signature(ReflexClose).parameters["close_time"].default)
    logging.warning("robot.gripper_close_time_s missing in sim config; using ReflexClose default %.3f s",
                    default)
    return default, "control.reflex default"


def _setup_logging(output_dir: Path, *, debug: bool = False) -> None:
    """Send logs to the screen and to <output_dir>/eval_policy.log. Called once in main()."""
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
    # Sim images can be CHW or HWC, 3 or 4 channels, 0-255 or 0-1.
    # Make them all look the same as the training images did.
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


def load_policy(checkpoint: str,device: str, *, n_action_steps: int | None = None):
    """Load the SmolVLA checkpoint onto the GPU in eval mode. Called once in main()."""
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
    # chunk_size = SmolVLA predicts chunk of future actions in one fwd pass (default 50).
    # n_action_steps = how many of those "action chunks"we execute on robot before calling the net again (replan).
    # @25hz (25 actions/s) n_action_steps=50 means 2s of action execution (50*0.04s/action = 2s).
    # smaller n = the net looks at fresh numbers more often (replans sooner).
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
    from policy.transforms import load_frames

    # How this checkpoint reads p̂ and writes actions (abs / rel / delta).
    policy.foresight_frames = load_frames(checkpoint)
    logging.info("policy frames: %s", policy.foresight_frames)
    policy.to(device)
    policy.eval()
    return policy


def _rgb_frame(img: np.ndarray) -> np.ndarray:
    """Static-camera frame as the tracker expects (RGB, no alpha)."""
    frame = np.ascontiguousarray(img)
    if frame.ndim == 3 and frame.shape[-1] == 4:
        frame = np.ascontiguousarray(frame[..., :3])
    return frame


def _hold_slide(scene) -> None:
    """One sim step with the hand held and the object still sliding.

    Used only by warm_filter, so the filter can watch the object move while
    the robot waits."""
    ee, quat = scene.get_ee_pose()
    scene.set_ee_target(ee, quat, float(scene.get_gripper_open()))
    scene.step(kinematic_object=True, attach_object=False, freeze_object=False)


def warm_filter(scene, est) -> int:
    """Show the filter a few frames before the policy plans.

    The filter's first guess has speed 0. Two frames go in before the planning
    frame, so that first chunk is not off by speed × Δ.
    Returns how many frames were fed before that planning frame.

    Called once per episode, only when --cond-source filter (G2b).
    """
    from eval.cond_source import FILTER_WARMUP_BEFORE

    seen_before = 0
    # One extra loop turn: the last frame is the planning frame itself.
    for i in range(FILTER_WARMUP_BEFORE + 1):
        _hold_slide(scene)
        state = scene.get_object_state()
        rgb = _rgb_frame(scene.get_images()["static_cam"])
        if i == 0:
            cam = scene.cfg.get("static_camera", {})
            want_h, want_w = int(cam["height"]), int(cam["width"])
            if rgb.shape[0] != want_h or rgb.shape[1] != want_w:
                raise RuntimeError(
                    f"static frame is {rgb.shape[:2]}, static_camera config is {want_h}x{want_w}"
                )
            logging.info("filter warmup static frame %s", tuple(rgb.shape))
            # Spawn pose stands in for the one-shot detector (estimator.start).
            if not est.start(rgb, state.position):
                logging.warning(
                    "filter did not find the object on the first frame (seed=%s)",
                    getattr(scene, "_meta", {}).get("seed"),
                )
        est.step(rgb, float(state.timestamp))
        if i < FILTER_WARMUP_BEFORE:
            seen_before += 1
    return seen_before


def _filter_state_for_frame(est, rgb: np.ndarray, t: float):
    """Update the filter on this frame once. Same time ⇒ already warmed.

    The last warmup frame was already fed in warm_filter. If we fed it again
    the filter would see the same time twice, so we just reuse its answer."""
    t = float(t)
    if est.kf.started and abs(float(est.kf.t) - t) <= 1e-8:
        return est.kf.state(est.plane_z)
    return est.step(rgb, t)


def build_obs_batch(
    scene,
    *,
    delta: float,
    language: str,
    device: str,
    zero_conditioning: bool,
    cond_source: str = "oracle",
    images: dict | None = None,
    oracle_state=None,
    filter_state=None,
    frames: dict | None = None,
    cond_shift: np.ndarray | None = None,
) -> tuple[dict, dict]:
    """One SmolVLA observation dict + the packs we logged.

    Aux keys: cond, truth, filter, valid.

    Called every sim step from run_episode_policy. "observation.state" is
    the hand state followed by the 4-number pack (cond). That is the same
    layout training used, so the net reads the numbers from the same slots.
    cond is what the net sees (truth, filter, or zeros). truth is the sim
    answer, kept beside it for logs.
    """
    from eval.cond_source import choose_pack
    from policy.convert_h5 import pack_proprio

    imgs = scene.get_images() if images is None else images
    ee_pos, ee_quat = scene.get_ee_pose()
    # Same channel as collect/convert: commanded open∈{0,1}, not raw finger width.
    # (Fingers are snapped to this command on attach; see Phase0Scene._snap_fingers.)
    grip_cmd = float(scene.get_gripper_open())
    state = pack_proprio(ee_pos, ee_quat, grip_cmd)
    obj = scene.get_object_state() if oracle_state is None else oracle_state
    # choose_pack decides the 4 numbers: sim truth (oracle), camera filter,
    # or all zeros (--zero-conditioning). The images and hand state do not change
    # in the zero path; only these 4 numbers are wiped.
    aux = choose_pack(
        source=cond_source,
        oracle_state=obj,
        filter_state=filter_state,
        delta=delta,
        zero=zero_conditioning,
    )
    if cond_shift is not None and not zero_conditioning:
        # Claim test: feed a deliberately wrong p̂ (logged as the cond we fed).
        aux["cond"] = np.asarray(aux["cond"], dtype=np.float64).copy()
        aux["cond"][0:2] += np.asarray(cond_shift, dtype=np.float64).reshape(2)
    from policy.transforms import cond_to_model

    cond_frame = (frames or {}).get("cond_frame", "abs")
    # Zero-out hides the numbers in the model's own frame (all zeros fed).
    model_cond = aux["cond"] if zero_conditioning else cond_to_model(aux["cond"], ee_pos, cond_frame)
    aux["cond_model"] = np.asarray(model_cond, dtype=np.float32)
    state = np.concatenate([state, aux["cond_model"]], axis=0).astype(np.float32)

    # Add a batch dimension of 1. The task text ends with a newline, as in training.
    batch = {
        "observation.state": torch.as_tensor(state, device=device).unsqueeze(0),
        "observation.images.static_cam": _as_chw_float(imgs["static_cam"]).unsqueeze(0).to(device),
        "observation.images.wrist_cam": _as_chw_float(imgs["wrist_cam"]).unsqueeze(0).to(device),
        "task": [language if language.endswith("\n") else language + "\n"],
    }
    return batch, aux


def action_to_ee(action: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Unpack policy action → (pos3, quat_wxyz4, gripper_open01).

    Net output is 8 numbers: hand x,y,z, hand turn (4), gripper. Gripper above
    0.5 means open, otherwise closed. Called once per sim step."""
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
    delta: float,
    device: str,
    zero_conditioning: bool,
    n_steps: int,
    save_images: bool = False,
    cond_source: str = "oracle",
    checkpoint: str = "",
    n_action_steps: int | None = None,
    no_attach: bool = False,
    force_place: bool = False,
    reflex_on: bool = False,
    action_buffer_on: bool = False,
    inject_latency_s: float = 0.0,
    close_time: tuple[float, str] | None = None,
    cond_shift: np.ndarray | None = None,
    held_from_hand: bool = True,
    close_gate: dict | None = None,
) -> tuple[dict, dict]:
    """Closed-loop policy rollout. Logs the same arrays as expert collect.

    Called once per trial by eval_speed_bin. Returns (episode arrays, meta).
    The policy supplies an action every step (often from its queued chunk).
    This function only adds the
    "glue" rules around it (see the attach comments in the loop).
    """
    from policy.features import default_language

    # Zero-out is an oracle-only check. Mixing it with the filter would mean
    # nothing, so stop early.
    if zero_conditioning and cond_source == "filter":
        raise ValueError("Pass only one of --zero-conditioning / --cond-source filter")

    # Same seed + same speed = same start. That makes runs repeatable.
    meta = scene.reset_episode(seed=seed, speed=speed)
    # Same seed = same policy noise too (SmolVLA samples noise every replan).
    seed_episode(seed)
    meta["policy_seed"] = int(seed)
    language = default_language(meta.get("category"), meta.get("container_category"))
    meta["language"] = language
    meta["no_attach"] = bool(no_attach)
    meta["force_place"] = bool(force_place)
    meta["cond_source"] = cond_source
    frames = dict(getattr(policy, "foresight_frames", None) or {"cond_frame": "abs", "action_frame": "abs"})
    meta["frames"] = frames
    meta["cond_shift_m"] = None if cond_shift is None else [float(v) for v in cond_shift]
    meta["checkpoint"] = checkpoint
    meta["n_action_steps"] = None if n_action_steps is None else int(n_action_steps)
    meta["delta"] = float(delta)
    # House / throw default: real PhysX grasp (attach frees the object, no weld).
    physics_grasp = not bool(getattr(scene, "weld_grasp", True))
    meta["grasp_mode"] = "physics" if physics_grasp else "weld"
    # --force-place is a weld-era debug helper; skip it on physics grasps.
    if physics_grasp and force_place:
        logging.warning("physics grasp: ignoring --force-place (no weld to script-place)")
        force_place = False
        meta["force_place"] = False

    # ----- Phase 3 overlays (explicit flags only; config use_reflex is for collect) -----
    from collections import deque

    from control import ActionBuffer, ReflexClose

    dt = float(scene.dt)
    lat_steps = int(round(float(inject_latency_s) / dt))
    if abs(lat_steps * dt - float(inject_latency_s)) > 1e-6:
        logging.warning("inject latency %.3f s rounded to %d steps (%.3f s)",
                        inject_latency_s, lat_steps, lat_steps * dt)
    if close_time is None:
        close_time = gripper_close_time(scene.cfg)
    reflex = None
    if reflex_on:
        if not physics_grasp:
            logging.warning("--reflex on in weld mode: reflex close feeds the weld latch")
        reflex = ReflexClose(close_time=float(close_time[0]), radius=0.10, vmax=0.5)
        reflex.reset()
    action_buf = ActionBuffer(max_age=0.20) if action_buffer_on else None
    # Physics-rate reflex when the scene offers a per-substep hook.
    substep_ok = "substep_cb" in inspect.signature(scene.step).parameters
    meta["overlays"] = {
        "reflex": bool(reflex_on),
        "reflex_rate": (None if reflex is None else ("substep" if substep_ok else "control_step")),
        "action_buffer": bool(action_buffer_on),
        "inject_latency_s": float(inject_latency_s),
        "latency_steps": lat_steps,
        "gripper_close_time_s": float(close_time[0]),
        "gripper_close_time_source": close_time[1],
    }
    meta["reflex"] = bool(reflex_on)
    meta["action_buffer"] = bool(action_buffer_on)
    reflex_close_t = None  # sim time the reflex first forced a close
    reflex_owns_grip = False  # reflex closed; policy has not asked to close yet
    reflex_engaged = False  # the current reflex close was already handled
    reflex_close_t_list: list = []  # earlier reflex closes that slipped
    n_hold_steps = 0  # buffer on: steps with no due action (held the last one)
    ee_prev = None  # (hand xyz, time) last frame: hand velocity while holding
    # Close gate state: first close only (a later re-close is the policy's own).
    gate_done, gate_wait, gate_hold, gate_log = False, 0, None, []
    n_hand_frames = 0  # filter arm: frames whose numbers came from the held hand

    # Only the filter arm (G2b) needs a camera tracker + Kalman filter.
    est = None
    if cond_source == "filter":
        from perception.estimator import StaticCamEstimator

        est = StaticCamEstimator.from_cfg(scene.cfg)
        meta["filter_warmup_frames"] = warm_filter(scene, est)

    # One list per thing we record, one entry per sim step. Same names as the
    # expert collect files so the same tools can read both.
    # "holding" is the 0/1 glue flag. Failure labels are built from it.
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
        "truth_pack": [],
        "filter_pack": [],
        "filter_valid": [],
        "filter_redetect_counts": [],
        "timestamp": [],
        "container_pos": [],
        "policy_replan": [],
    }
    if save_images:
        log["static_cam_rgb"] = []
        log["wrist_cam_rgb"] = []
    # Clear the old action queue from the last trial.
    policy.reset()
    # Our own plan queue (replaces LeRobot's ACTION queue so we can delay and
    # re-time chunks). With no overlays it plays exactly like select_action.
    n_exec = int(policy.config.n_action_steps)
    plan: deque = deque()  # buffer off: actions in play order
    obs_hist: deque = deque(maxlen=lat_steps + 1)  # oldest = obs L seconds ago
    noise_gen = torch.Generator(device=device).manual_seed(int(seed))
    noise_shape = (1, int(policy.config.chunk_size), int(policy.config.max_action_dim))
    last_act = None
    # --- glue state (all reset each episode) ---
    attached = False  # object is welded to the hand ("glued")
    gripping = False  # --no-attach: fingers closed near object (physics hold)
    placed = False  # opened after a hold (release/place done)
    lift_frames = 0  # steps spent holding the object above 12 cm
    close_dwell = 0  # frames closed+near before attach (smooth pick)
    fp_stage = _FP_IDLE  # --force-place step (debug only)
    fp_timer = 0
    # Fixed spots for the debug place script (bowl drop point, carry height, home).
    # Unused unless --force-place.
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

    # ===== MAIN EPISODE LOOP =====
    # One turn = one sim step (25 Hz):
    #   1. read cameras + object   
    #   2. build the 4-number pack
    #   3. ask the policy for an action   
    #   4. apply glue rules
    #   5. step the sim   
    #   6. log everything
    for t in range(n_steps):
        # Empty plan means the net runs this step.
        will_replan = (len(action_buf) == 0) if action_buf is not None else (len(plan) == 0)
        imgs = scene.get_images()
        oracle_now = scene.get_object_state()
        # Filter arm: feed this frame to the camera filter so it can give us
        # its guess. Oracle and zero arms skip this (filt_state stays None).
        filt_state = None
        if est is not None:
            filt_state = _filter_state_for_frame(
                est,
                _rgb_frame(imgs["static_cam"]),
                float(oracle_now.timestamp),
            )
            # Held object rides with the hand: the camera cannot see it inside
            # the fingers, and once lifted it is off the table plane the filter
            # assumes. Fingers stopped apart = something is held (real arm: same
            # width reading). The filter keeps running underneath for a slip.
            ee_now = np.asarray(scene.get_ee_pose()[0], dtype=np.float64)
            t_now = float(oracle_now.timestamp)
            if (
                held_from_hand
                and float(scene.get_gripper_open()) < 0.5
                and scene.get_finger_gap_m() > HELD_MIN_GAP_M
                and ee_prev is not None
                and t_now > ee_prev[1]
            ):
                v = (ee_now - ee_prev[0]) / (t_now - ee_prev[1])
                filt_state = ObjectState(ee_now, v, np.full(6, 1e-6), t_now, True)
                n_hand_frames += 1
            ee_prev = (ee_now, t_now)
        batch, aux = build_obs_batch(
            scene,
            delta=delta,
            language=language,
            device=device,
            zero_conditioning=zero_conditioning,
            cond_source=cond_source,
            images=imgs,
            oracle_state=oracle_now,
            filter_state=filt_state,
            frames=frames,
            cond_shift=cond_shift,
        )
        # Latency: the policy plans from the observation lat_steps ago
        # (clamped to the first frame at episode start). L=0: this frame.
        obs_hist.append((batch, aux))
        batch_used, aux_used = obs_hist[0]
        # cond = the 4 numbers the net sees (zeros in --zero-conditioning).
        cond = aux_used["cond"]
        now_k = t * dt  # step clock for the buffer (the plan's obs is lat_steps older)
        if will_replan:
            # Seeded flow-matching noise (else LeRobot draws torch.normal unseeded).
            noise = torch.randn(noise_shape, generator=noise_gen, device=device)
            with torch.inference_mode():
                chunk = policy.predict_action_chunk(batch_used, noise=noise)
            chunk = chunk[0, :n_exec].detach().cpu().numpy().astype(np.float64)
            # Delta checkpoints predict moves from the hand in the planning obs.
            from policy.transforms import action_from_model

            ee_plan = batch_used["observation.state"][0, 0:3].detach().cpu().numpy()
            chunk = action_from_model(chunk, ee_plan, frames.get("action_frame", "abs"))
            if action_buf is not None:
                # Action i was meant for obs time + i*dt. Slots already past are dropped.
                action_buf.push_chunk(chunk, t0=now_k - lat_steps * dt, dt=dt, now=now_k - 1e-6)
            else:
                plan.extend(chunk)  # play from the first action, however stale
        if action_buf is not None:
            act_np = action_buf.pop_due(now_k + 1e-6)
            if act_np is None:  # everything was stale: hold the last command
                n_hold_steps += 1
                if last_act is None:
                    ee0, q0 = scene.get_ee_pose()
                    last_act = np.concatenate([ee0, q0, [float(scene.get_gripper_open())]])
                act_np = last_act
        else:
            act_np = plan.popleft()
        last_act = np.asarray(act_np, dtype=np.float64).reshape(-1)
        now_t = float(oracle_now.timestamp)
        pos, quat, grip = action_to_ee(last_act)

        # ----- GLUE (grasp latch) -----
        # Weld mode (G1/G2 bounce box): close near object → weld attach.
        # Physics mode (Phase 3 throw): close near object → free to PhysX (no weld).
        # --no-attach: never set attach; finger contact only.
        obj = scene.get_object_state()
        ee, ee_q = scene.get_ee_pose()
        dist = float(np.linalg.norm(ee - obj.position))
        want_close = grip < 0.5

        # ----- CLOSE GATE (Job 3 overlay, eval only) -----
        # The policy picks *when*; the gate checks the hand is really there before
        # the first close: wrist depth (height above the object) and the tracked
        # object XY (filter arm: the filter, never sim truth). While waiting, hold
        # the pre-close target and carry it along with the object. Times out.
        if close_gate is not None and not gate_done and want_close and float(scene.get_gripper_open()) > 0.5:
            src = filt_state if filt_state is not None else obj
            p_now = src.position + src.velocity * max(now_t - float(src.timestamp), 0.0)
            if gate_hold is None:
                gate_hold = (np.asarray(pos, dtype=np.float64).reshape(3).copy(), p_now.copy())
            depth = float(scene.get_wrist_depth_m())
            xy_err = float(np.linalg.norm(ee[:2] - p_now[:2]))
            ready = xy_err <= close_gate["xy_m"] and np.isfinite(depth) and depth <= close_gate["depth_m"]
            if ready or gate_wait >= close_gate["max_wait"]:
                gate_done = True
                gate_log.append({"t": now_t, "wait": gate_wait, "ready": bool(ready),
                                 "depth_m": depth, "xy_err_m": xy_err})
            else:
                gate_wait += 1
                hold_pos, p0 = gate_hold
                pos = hold_pos.copy()
                pos[0:2] += (p_now - p0)[0:2]
                grip, want_close = 1.0, False

        # Reflex (control rate): steer XY at the meeting point, close on time-to-arrival.
        # Uses the filter state when there is one, else sim truth. A close stays latched.
        ref_state = filt_state if filt_state is not None else obj
        reflex_live = reflex is not None and not attached and not placed
        if reflex_live:
            rout = reflex.step(ref_state, ee, max(now_t, float(ref_state.timestamp)))
            if rout.active:
                pos = np.asarray(pos, dtype=np.float64).reshape(3).copy()
                pos[0], pos[1] = float(rout.target[0]), float(rout.target[1])
        # Hand-over: the reflex holds the gripper shut until the policy itself
        # asks to close once; then the policy owns the gripper again (so it can release).
        if reflex is not None and reflex.closed and not reflex_engaged:
            reflex_engaged = True
            reflex_owns_grip = True
            if reflex_close_t is None:
                reflex_close_t = now_t
        if reflex_owns_grip and want_close:
            reflex_owns_grip = False
        if reflex_owns_grip and not placed:
            grip = 0.0
            want_close = True

        # Count steps where we hold it above 12 cm. Only --force-place uses
        # this count, to know the lift is real before it starts carrying.
        holding_now = attached if not no_attach else gripping
        if holding_now and float(obj.position[2]) >= 0.12:
            lift_frames += 1

        # Smooth pick: short closed + near dwell before welding (avoid instant snap).
        # Close within 5 cm for 1 step (3 steps in --force-place) to glue.
        attach_dist = 0.05
        attach_dwell_need = 3 if force_place else 1
        if want_close and dist < attach_dist and not attached and not placed:
            close_dwell += 1
        else:
            close_dwell = 0

        # ----- DEBUG ONLY: --force-place -----
        # Why: separates "can the policy pick?" from "does release work?".
        # Once the policy has glued and lifted, a script takes over the hand
        # and overrides pos/quat/grip. Steps: carry → lower → unglue → open → home.
        # Not used in a normal G1 run.
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

        # ----- Turn the glue decision into 3 flags for the sim step -----
        #   kin    = object slides on its belt path (not yet held, not placed)
        #   attach = weld hand (legacy) OR free-to-PhysX close (physics grasp)
        #   freeze = park object after release (weld mode only; physics ignores it)
        # holding_log is the 0/1 value saved in log["holding"].
        if no_attach:
            # Diagnostic path: never set attach. "gripping" = hand closed within 5 cm.
            # Let go if the hand opens or the object slips >10 cm away.
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
            # Close near the object → attach_flag. Weld mode glues; physics mode
            # frees the object to PhysX with fingers closed (scene decides).
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
            # Physics: slip if the object flees while we still command close.
            if physics_grasp and attached and dist > 0.10 and want_close:
                logging.info("physics slip at t=%d dist=%.3f m", t, dist)
                attached = False
                if reflex is not None:  # lost it: let the reflex try again
                    reflex.reset()
                    reflex_owns_grip = False
                    reflex_engaged = False
                    reflex_close_t_list.append(reflex_close_t)
                    reflex_close_t = None
            attach_flag = bool(attached)
            freeze = bool(
                (not physics_grasp)
                and (
                    (placed and not attached)
                    or (force_place and fp_stage >= _FP_UNGLUE and not attached)
                )
            )
            kin = (not attached) and (not freeze)
            holding_log = float(attached)

        # Send the hand target to the robot and advance the sim one step.
        scene.set_ee_target(pos, quat, grip)
        step_kw = {}
        if reflex_live and substep_ok:
            # Physics-rate reflex: re-aim (and maybe close) before every physics substep.
            # Contract (sim/scene.py): only set_ee_target / get_* / release_object here.
            def _reflex_substep(sc, t_sim, _ref=ref_state, _pos=np.asarray(pos, dtype=np.float64),
                                _quat=quat, _grip=grip):
                nonlocal reflex_close_t
                ee_now, _ = sc.get_ee_pose()
                out = reflex.step(_ref, ee_now, max(float(t_sim), float(_ref.timestamp)))
                if not out.active:
                    return
                p = _pos.reshape(3).copy()
                p[0], p[1] = float(out.target[0]), float(out.target[1])
                if out.close and reflex_close_t is None:
                    reflex_close_t = float(t_sim)
                sc.set_ee_target(p, _quat, 0.0 if out.close else _grip)
                # Physics grasp: closing mid-step must also stop the scripted coast,
                # else the object is teleported through the closing fingers.
                if out.close and physics_grasp and hasattr(sc, "release_object"):
                    sc.release_object()

            step_kw["substep_cb"] = _reflex_substep
        scene.step(
            kinematic_object=kin,
            attach_object=attach_flag,
            freeze_object=freeze,
            **step_kw,
        )

        # ----- Record what happened this step (state AFTER the sim step) -----
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
        # Log the exact 4 numbers the net saw (so a zero run shows all zeros).
        if cond is not None:
            log["conditioning"].append(np.asarray(cond, dtype=np.float64).copy())
        # Filter arm only: also log truth vs filter guess, to score the filter.
        if aux["truth"] is not None and est is not None:
            log["truth_pack"].append(np.asarray(aux["truth"], dtype=np.float64).copy())
            filt = aux["filter"]
            log["filter_pack"].append(
                np.asarray(filt if filt is not None else cond, dtype=np.float64).copy()
            )
            log["filter_valid"].append(float(aux["valid"] if aux["valid"] is not None else 0.0))
            log["filter_redetect_counts"].append(int(est.n_redetects))
        # Images are big. Only keep them in --debug (for the MP4).
        if save_images:
            imgs = scene.get_images()
            log["static_cam_rgb"].append(imgs["static_cam"])
            log["wrist_cam_rgb"].append(imgs["wrist_cam"])

    # ----- After the loop: turn lists into arrays and check shapes -----
    # A wrong shape here means a logging bug. Stop loudly instead of saving bad data.
    episode = {k: np.asarray(v) for k, v in log.items() if len(v)}
    t_len = int(episode["object_pos"].shape[0])
    assert episode["object_pos"].shape == (t_len, 3), episode["object_pos"].shape
    if "conditioning" in episode:
        from interfaces.state import CONDITIONING_DIM, CONDITIONING_LAYOUT

        assert episode["conditioning"].shape == (t_len, CONDITIONING_DIM), episode[
            "conditioning"
        ].shape
        for pack_key in ("truth_pack", "filter_pack"):
            if pack_key in episode:
                assert episode[pack_key].shape == (t_len, CONDITIONING_DIM), episode[
                    pack_key
                ].shape
        if "filter_valid" in episode:
            assert episode["filter_valid"].shape == (t_len,), episode["filter_valid"].shape
        if "filter_redetect_counts" in episode:
            assert episode["filter_redetect_counts"].shape == (t_len,), episode[
                "filter_redetect_counts"
            ].shape
    if "oracle_state" in episode:
        assert episode["oracle_state"].shape == (t_len, 14), episode["oracle_state"].shape
    meta["n_frames"] = t_len
    meta["delta"] = float(delta)
    meta["n_replans"] = int(episode["policy_replan"].sum())
    meta["placed_released"] = bool(placed)
    meta["reflex_close_t"] = reflex_close_t
    meta["reflex_close_t_slipped"] = reflex_close_t_list
    meta["n_hold_steps"] = int(n_hold_steps)
    meta["held_from_hand"] = bool(held_from_hand) if est is not None else None
    meta["n_hand_frames"] = int(n_hand_frames)
    meta["close_gate"] = close_gate
    meta["close_gate_log"] = gate_log
    meta["off_table"] = bool(getattr(scene, "off_table", False))
    meta["table_z"] = float(getattr(scene, "_table_z", 0.0))
    meta["dt"] = dt
    # Summary of the 4 numbers. "all_zero" is the quick proof that a
    # --zero-conditioning run really fed zeros (and a normal run did not).
    # "_replan" versions look only at steps where the net actually ran.
    if episode.get("conditioning", np.array([])).size:
        c = episode["conditioning"]
        replan = episode["policy_replan"].astype(bool)
        c_replan = c[replan] if replan.any() else c
        meta["conditioning_debug"] = {
            "layout": list(CONDITIONING_LAYOUT),
            "zero_conditioning": bool(zero_conditioning),
            "cond_source": cond_source,
            "first": c[0].tolist(),
            "first_replan": c_replan[0].tolist(),
            "mean": c.mean(axis=0).tolist(),
            "mean_replan": c_replan.mean(axis=0).tolist(),
            "l2_mean": float(np.linalg.norm(c, axis=1).mean()),
            "l2_mean_replan": float(np.linalg.norm(c_replan, axis=1).mean()),
            "all_zero": bool(np.allclose(c, 0.0)),
        }
    # Filter arm only: how far was the filter's guess from sim truth (x,y, in cm)?
    # This tells us if a G2b drop comes from a bad filter or from the policy.
    if est is not None and "filter_pack" in episode and "truth_pack" in episode:
        err_m = np.linalg.norm(
            episode["filter_pack"][:, :2] - episode["truth_pack"][:, :2], axis=1
        )
        # G2a scored frames before the first grasp. After that the fruit is in
        # the hand or the bowl, off the table plane the filter assumes.
        held = episode["holding"] > 0.5
        if np.any(held):
            before = np.arange(err_m.shape[0]) < int(np.argmax(held))
        else:
            before = np.ones(err_m.shape[0], dtype=bool)
        replan = episode["policy_replan"].astype(bool)

        def _err_stats(mask: np.ndarray) -> dict | None:
            e = err_m[mask]
            if e.size == 0:
                return None
            return {
                "n": int(e.size),
                "p_median_cm": float(np.median(e) * 100.0),
                "p_rmse_cm": float(np.sqrt(np.mean(np.square(e))) * 100.0),
            }

        meta["filter_redetects"] = int(est.n_redetects)
        meta["filter_valid_rate"] = float(np.mean(episode["filter_valid"][before]))
        meta["filter_vs_truth"] = {
            "before_hold": _err_stats(before),
            "before_hold_replan": _err_stats(before & replan),
        }
        before_stats = meta["filter_vs_truth"]["before_hold"] or {}
        logging.info(
            "filter vs truth (before grasp) p̂ median %.2f cm  rmse %.2f cm  n=%s  valid %.2f  redetects %d",
            float(before_stats.get("p_median_cm", float("nan"))),
            float(before_stats.get("p_rmse_cm", float("nan"))),
            before_stats.get("n"),
            meta["filter_valid_rate"],
            meta["filter_redetects"],
        )
    return episode, meta


def episode_success_policy(episode: dict, carry_pose: np.ndarray | None = None) -> bool:
    """Policy success (no expert stages): lift + high z or near bowl.

    Plain words: the object went above 12 cm AND it ended either still above
    10 cm or within 12 cm (x,y) of the bowl. Called once per trial.
    When carry_pose is given, the final XY distance to it is also checked.
    Whether the object was held is handled by the failure label, not here."""
    from sim.success import episode_diagnostics

    diag = episode_diagnostics(episode)
    lifted = diag["z_max"] >= 0.12
    placed = diag["z_end"] >= 0.10
    if carry_pose is not None and "object_pos" in episode:
        final = episode["object_pos"][-1]
        xy = np.linalg.norm(final[:2] - np.asarray(carry_pose)[:2])
        placed = placed or (lifted and xy < 0.12)
    return bool(lifted and placed)


def episode_timing(episode: dict, meta: dict, success: bool) -> dict:
    """Timing label + closest approach + pre-grasp smoothness (eval/timing_labels.py)."""
    from eval.timing_labels import timing_label

    return timing_label(
        ee_pos=episode["ee_pos"],
        object_pos=episode["object_pos"],
        gripper=episode["gripper"],
        holding=episode.get("holding"),
        success=bool(success),
        dt=float(meta.get("dt", 0.04)),
        table_z=float(meta.get("table_z", 0.0)),
        off_table_flag=bool(meta.get("off_table", False)),
    )


def _save_conditioning_trace(
    out_dir: Path,
    *,
    speed: float,
    seed: int,
    episode: dict,
    meta: dict,
    zero_conditioning: bool,
    success: bool,
    failure: str,
) -> Path | None:
    """Write lightweight JSON + H5 (same arrays as collect).

    One pair of files per trial, so we can replay what the net saw and did.
    Called by eval_speed_bin unless --no-save-conditioning."""
    from sim.collect import episode_for_json, write_episode_h5

    extra = {
        "delta": np.asarray([float(meta["delta"])], dtype=np.float32),
        "speed_m_s": np.asarray([float(speed)], dtype=np.float32),
        "seed": np.asarray([int(seed)], dtype=np.int64),
        "zero_conditioning": np.asarray([int(bool(zero_conditioning))], dtype=np.int8),
        "success": np.asarray([int(bool(success))], dtype=np.int8),
        "cond_source": np.asarray(str(meta.get("cond_source", "oracle")).encode("ascii")),
        "n_action_steps": np.asarray(
            [-1 if meta.get("n_action_steps") is None else int(meta["n_action_steps"])],
            dtype=np.int64,
        ),
        "force_place": np.asarray([int(bool(meta.get("force_place")))], dtype=np.int8),
    }
    h5_path = out_dir / f"eval_model_v{speed:.2f}_s{seed}_cond.h5"
    write_episode_h5(h5_path, episode, extra=extra)

    n_frames = int(np.asarray(episode["object_pos"]).shape[0]) if "object_pos" in episode else 0
    side = {
        "seed": int(seed),
        "speed_m_s": float(speed),
        "robot": "franka",
        "success": bool(success),
        "failure": failure,
        "model": "model",
        "n_frames": n_frames,
        "delta": float(meta["delta"]),
        "zero_conditioning": bool(zero_conditioning),
        "cond_source": meta.get("cond_source", "oracle"),
        "checkpoint": meta.get("checkpoint", ""),
        "n_action_steps": meta.get("n_action_steps"),
        "force_place": bool(meta.get("force_place")),
        "filter_warmup_frames": meta.get("filter_warmup_frames"),
        "filter_redetects": meta.get("filter_redetects"),
        "filter_valid_rate": meta.get("filter_valid_rate"),
        "filter_vs_truth": meta.get("filter_vs_truth"),
        "h5": str(h5_path),
        "grasp_mode": meta.get("grasp_mode"),
        "policy_seed": meta.get("policy_seed"),
        "overlays": meta.get("overlays"),
        "reflex_close_t": meta.get("reflex_close_t"),
        "reflex_close_t_slipped": meta.get("reflex_close_t_slipped"),
        "n_hold_steps": meta.get("n_hold_steps"),
        "off_table": meta.get("off_table"),
        "timing": meta.get("timing"),
        **{k: meta[k] for k in ("max_stage", "z_max", "z_end", "z_min") if k in meta},
        **episode_for_json(episode),
    }
    side_path = out_dir / f"eval_model_v{speed:.2f}_s{seed}_cond.json"
    side_path.write_text(json.dumps(side, indent=2, default=str))
    return h5_path


def eval_speed_bin(args, scene, policy, speed: float, metrics_path: Path) -> dict:
    """Run all trials for ONE belt speed. Called once per speed from main().

    For each trial: run the episode, score it, give it the code's policy label
    (never-held | held-no-lift | lifted-no-place | success), append one
    line to metrics.jsonl. Returns the speed's success rate.
    n = --trials-per-speed. 20 is a scout. Use n >= 80 to claim G1.
    """
    from sim.success import classify_policy_failure, episode_diagnostics

    n = int(args.trials_per_speed)
    # Each speed gets its own seed range, and each trial adds 1.
    # Different speeds never reuse the same start. Reruns repeat exactly.
    seed = int(args.seed) + int(round(speed * 1000))
    successes = 0
    t0 = time.time()
    device = args._device
    from sim.state_machine import inference_window_steps

    # Time limit per trial, in sim steps, from the config.
    n_steps = inference_window_steps(scene.cfg, scene.dt)
    if bool(getattr(args, "force_place", False)):
        # Carry→lower→unglue→open→home needs more than a pick-only window.
        n_steps = max(int(n_steps), 400)
    save_images = bool(args.debug)
    save_cond = bool(getattr(args, "save_conditioning", True))
    out_dir = Path(args.output_dir)

    # ===== TRIAL LOOP =====
    # One turn = one trial at this speed. The same --zero-conditioning /
    # --cond-source setting applies to every trial in the run.
    for i in range(n):
        logging.info("eval speed=%.2f trial %d/%d seed=%d", speed, i + 1, n, seed)
        # A crashed trial counts as a failure (ok=False) and is logged with
        # the full error. It still counts in n, so the rate is not inflated.
        try:
            n_act = args.n_action_steps
            if n_act is None:
                n_act = int(policy.config.n_action_steps)
            episode, meta = run_episode_policy(
                scene,
                policy,
                seed=seed,
                speed=speed,
                delta=float(args.delta),
                device=device,
                zero_conditioning=bool(args.zero_conditioning),
                cond_source=str(args.cond_source),
                checkpoint=str(args.checkpoint),
                n_action_steps=int(n_act),
                n_steps=n_steps,
                save_images=save_images,
                no_attach=bool(getattr(args, "no_attach", False)),
                force_place=bool(getattr(args, "force_place", False)),
                reflex_on=getattr(args, "reflex", "off") == "on",
                action_buffer_on=getattr(args, "action_buffer", "off") == "on",
                inject_latency_s=float(getattr(args, "inject_latency_s", 0.0)),
                close_time=getattr(args, "_close_time", None),
                cond_shift=getattr(args, "cond_shift_m", None),
                held_from_hand=getattr(args, "held_from_hand", "on") == "on",
                close_gate=(
                    {
                        "depth_m": float(args.close_gate_depth_m),
                        "xy_m": float(args.close_gate_xy_m),
                        "max_wait": int(args.close_gate_max_wait),
                    }
                    if getattr(args, "close_gate", "off") == "on"
                    else None
                ),
            )
            ok = episode_success_policy(episode, carry_pose=np.asarray(meta.get("carry_pose")))
        except Exception as ex:
            logging.exception("eval episode failed: %s", ex)
            episode = {}
            ok = False
            meta = {}

        if ok:
            successes += 1
        # diag = max/end/min object height etc., logged so a label can be checked.
        diag = episode_diagnostics(episode if episode else None)
        holding = None
        if episode:
            holding = episode.get("holding")
        # Policy label from the 0/1 holding log + max object height:
        #   never held → never-held; held but z_max < 12 cm → held-no-lift;
        #   lifted but not in bowl → lifted-no-place; else success.
        tax = classify_policy_failure(
            success=ok,
            holding=holding,
            object_z_max=diag["z_max"],
        )
        # Physics grasp (throw): timing label from the logged tracks.
        if episode and meta.get("grasp_mode") == "physics":
            meta["timing"] = episode_timing(episode, meta, ok)
        cond_path = None
        if save_cond and episode:
            cond_path = _save_conditioning_trace(
                out_dir,
                speed=speed,
                seed=seed,
                episode=episode,
                meta=meta,
                zero_conditioning=bool(args.zero_conditioning),
                success=ok,
                failure=tax,
            )
        # One line per trial. Settings are stored with each line so a
        # normal run and a zero-conditioning run can be told apart later.
        rec = {
            "speed_m_s": speed,
            "seed": seed,
            "success": ok,
            "failure": tax,
            "model": "model",
            "zero_conditioning": bool(args.zero_conditioning),
            "cond_source": str(args.cond_source),
            "checkpoint": str(args.checkpoint),
            "n_action_steps": meta.get("n_action_steps"),
            "delta": meta.get("delta", float(args.delta)),
            "no_attach": bool(getattr(args, "no_attach", False)),
            "force_place": bool(getattr(args, "force_place", False)),
            "filter_warmup_frames": meta.get("filter_warmup_frames"),
            "filter_redetects": meta.get("filter_redetects"),
            "filter_valid_rate": meta.get("filter_valid_rate"),
            "filter_vs_truth": meta.get("filter_vs_truth"),
            "placed_released": meta.get("placed_released"),
            "n_replans": meta.get("n_replans"),
            "grasp_mode": meta.get("grasp_mode"),
            "policy_seed": meta.get("policy_seed"),
            "overlays": meta.get("overlays"),
            "reflex_close_t": meta.get("reflex_close_t"),
            "n_hold_steps": meta.get("n_hold_steps"),
            "off_table": meta.get("off_table"),
            "timing": meta.get("timing"),
            # Asset identity (fixed per process; proves multi-worker diversity).
            "frames": meta.get("frames"),
            "cond_shift_m": meta.get("cond_shift_m"),
            "held_from_hand": meta.get("held_from_hand"),
            "n_hand_frames": meta.get("n_hand_frames"),
            "close_gate": meta.get("close_gate"),
            "close_gate_log": meta.get("close_gate_log"),
            "asset_seed": meta.get("asset_seed"),
            "category": meta.get("category"),
            "container_category": meta.get("container_category"),
            "usd": meta.get("usd"),
            "container_usd": meta.get("container_usd"),
            **diag,
        }
        if meta.get("conditioning_debug"):
            rec["conditioning_debug"] = meta["conditioning_debug"]
        if cond_path is not None:
            rec["h5"] = str(cond_path)
        # Append right away so a crash later does not lose finished trials.
        with open(metrics_path, "a") as fp:
            fp.write(json.dumps(rec) + "\n")

        # --debug only: one stitched video per trial. File name has ok or the label.
        if save_images and episode:
            from sim.video import write_debug_mp4

            fps = int(scene.cfg.get("static_camera", {}).get("fps", 25))
            tag = "ok" if ok else tax
            mp4 = out_dir / f"eval_model_v{speed:.2f}_s{seed}_{tag}.mp4"
            write_debug_mp4(episode, mp4, fps=fps)

        seed += 1

    # Summary row for this speed. These rows become the G1 table.
    # Only a rate; no confidence range here. With n=20 it is a scout.
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
    """Partial G1 table for one arm (full pass/fail is eval/gate_g1.py).

    "Arm" = one run of this script (normal, or --zero-conditioning, or filter).
    Called once at the end of main(). With --cond-source filter the report
    is tagged G2b instead of G1.
    """
    by_speed = {s["speed_m_s"]: s for s in summaries}
    rate0 = by_speed.get(0.0, {}).get("success_rate")
    rate_hi = by_speed.get(HIGH_M_S, {}).get("success_rate")
    rate_mid = by_speed.get(MID_M_S, {}).get("success_rate")
    if rate_mid is None:
        rate_mid = by_speed.get(HIGH_M_S, {}).get("success_rate")

    # "Flat" = success at 20 cm/s is less than 15 points below success at 0 cm/s.
    # This is a point guess. It is only trustworthy with n >= 80 per speed.
    flat_ok = None
    if rate0 is not None and rate_hi is not None:
        flat_ok = (rate0 - rate_hi) < MID_M_S

    # Zero-out sanity is a separate run; full G1 is eval/gate_g1.py.
    report = {
        "gate": "G2b" if args.cond_source == "filter" else "G1",
        "model": "model",
        "zero_conditioning": bool(args.zero_conditioning),
        "cond_source": args.cond_source,
        "delta": float(args.delta) if args.delta is not None else None,
        "force_place": bool(args.force_place),
        "checkpoint": args.checkpoint,
        "criterion": {
            "model_flat_drop_0_to_20_lt": MID_M_S,
            "mid_speed_m_s": MID_M_S,
        },
        "summaries": summaries,
        "flat_pass": flat_ok,
        "wall_time_s": wall,
        "n_action_steps": (
            int(args.n_action_steps)
            if args.n_action_steps is not None
            else None
        ),
        "overlays": {
            "reflex": args.reflex,
            "action_buffer": args.action_buffer,
            "inject_latency_s": float(args.inject_latency_s),
            "gripper_close_time_s": (getattr(args, "_close_time", None) or (None,))[0],
        },
        "seed": int(args.seed),
    }
    # Only the normal run claims a "flat" check. The zero-out run is the
    # control, so it carries no partial verdict.
    if not args.zero_conditioning:
        report["gate_g1_partial"] = {
            "flat_pass": flat_ok,
            "note": "Full G1 needs model-zero run + eval/gate_g1.py (zero-out collapse)",
        }
    return report


def main() -> None:
    """Entry point. Start Isaac, load scene + policy, sweep speeds, write report.

    Output files in --output_dir: metrics.jsonl (one line per trial),
    gate_g1_report.json (one row per speed), eval_policy.log, and per-trial
    .h5/.json (+ .mp4 with --debug).
    """
    print("[foresight] G1 eval — starting AppLauncher…", flush=True)
    if "--torch-deterministic" in sys.argv:
        # cuBLAS reads this once, at first CUDA use (Isaac starts CUDA next).
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from isaaclab.app import AppLauncher

    # Isaac must start BEFORE we import sim code that needs omni/isaaclab.
    # That is why the sim imports below sit after AppLauncher(args).
    sys.argv = [a for a in sys.argv if a != ""]
    parser = _build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[foresight] Ignoring unknown args: {unknown}", flush=True)

    if args.cond_source == "filter" and bool(args.zero_conditioning):
        raise SystemExit("Choose only one of --zero-conditioning / --cond-source filter")

    # Cameras must be on or the images are black.
    args.enable_cameras = True
    app_launcher = AppLauncher(args)
    out_dir = Path(args.output_dir)
    _setup_logging(out_dir, debug=bool(args.debug))

    device = getattr(args, "device", None) or ("cuda:0" if torch.cuda.is_available() else "cpu")
    args._device = device
    if args.torch_deterministic:
        # Deterministic GPU kernels (env var for cuBLAS was set before AppLauncher).
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    from interfaces.config import conditioning_delta_s
    from sim.scene import Phase0Scene, load_cfg

    cfg = load_cfg(args.sim_cfg_file)
    # Look-ahead delta must match training. Default comes from the config file.
    if args.delta is None:
        args.delta = conditioning_delta_s(cfg)
    logging.info("eval conditioning Δ=%.3f s (from %s)", args.delta, args.sim_cfg_file)
    args._close_time = gripper_close_time(cfg)
    logging.info("overlays: reflex=%s action_buffer=%s latency=%.3f s gripper_close_time=%.3f s (%s)",
                 args.reflex, args.action_buffer, args.inject_latency_s, *args._close_time)
    if getattr(args, "device", None):
        cfg.setdefault("sim", {})["device"] = args.device

    # Create a new scene for each speed.
    scene = Phase0Scene(cfg, enable_cameras=True, object_usd=args.object_usd,container_usd=args.container_usd,asset_seed=int(args.seed))
    
    # Load the policy for the given checkpoint.
    policy = load_policy(args.checkpoint,device,n_action_steps=args.n_action_steps)

    # Start a fresh metrics file so old trials do not mix into this run.
    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    # ===== SPEED SWEEP =====
    # One scene + one policy, reused for every speed. One summary row per speed.
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
    # Hard exit: Isaac can hang on normal shutdown. Everything is saved by now.
    os._exit(0)


if __name__ == "__main__":
    main()

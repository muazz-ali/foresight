#!/usr/bin/env python3
"""Is the first camera frame after ``reset_episode`` stale (shows the last episode)?

One Isaac process, one scene (asset_seed fixed):
  run A1: reset seed A, frames 0..3      (frame 0 = get_images() right after reset,
                                          frame k = after k hold steps)
  run B : reset seed B, frames 0..3, then more steps so the world differs from A
  run A2: reset seed A again (previous episode = B), frames 0..3
  run A3: reset seed A again (previous episode = A), frames 0..3
  static: after A3, re-render N times with no physics step (pure render noise)

Prints max |diff| and share of pixels > 10/255 for each pair, per camera.
Saves all frames to <out>.npz so two processes can be compared with --compare.

Render overrides (diagnostic only; sim/scene.py is not changed):
  --aa {keep,off,taa,fxaa,dlss,dlaa}   sets /rtx/post/aa/op after the scene is built
  --rendering_mode {performance,balanced,quality}  (AppLauncher flag)

  python tools/diag/check_first_frame.py --headless --enable_cameras --device cuda:0 \
      -o data/p3_smoke_eval/first_frame/default_p1
  python tools/diag/check_first_frame.py --compare a.npz b.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

AA_OP = {"off": 0, "taa": 1, "fxaa": 2, "dlss": 3, "dlaa": 4}
RENDER_KEYS = [
    "/rtx/rendermode",
    "/rtx/post/aa/op",
    "/rtx/post/dlss/execMode",
    "/rtx-transient/dldenoiser/enabled",
    "/rtx-transient/dlssg/enabled",
    "/rtx-transient/post/aa/limitedOps",
    "/rtx/post/taa/samples",
    "/rtx/post/scaling/staticRatio",
    "/rtx/directLighting/sampledLighting/enabled",
    "/rtx/directLighting/sampledLighting/samplesPerPixel",
    "/rtx/raytracing/cached/enabled",
    "/rtx/shadows/enabled",
    "/rtx/ambientOcclusion/enabled",
    "/rtx/indirectDiffuse/enabled",
    "/rtx/reflections/enabled",
    "/rtx/translucency/enabled",
    "/omni/replicator/asyncRendering",
    "/app/renderer/waitIdle",
    "/app/hydraEngine/waitIdle",
    "/app/updateOrder/checkForHydraRenderComplete",
    "/isaaclab/rendering/rendering_mode",
]


def diff_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """max |a-b| (0..255) and share of pixels whose max-channel diff > 10."""
    d = np.abs(a[..., :3].astype(np.int16) - b[..., :3].astype(np.int16)).max(axis=-1)
    return {"max": int(d.max()), "frac_gt10": round(float((d > 10).mean()), 4),
            "mean": round(float(d.mean()), 3)}


def _fmt(s: dict) -> str:
    return f"max {s['max']:3d}  >10: {100 * s['frac_gt10']:5.1f}%  mean {s['mean']:.2f}"


def compare_files(pa: str, pb: str) -> None:
    a, b = np.load(pa), np.load(pb)
    print(f"cross-process: {pa}  vs  {pb}")
    for k in sorted(set(a.files) & set(b.files)):
        print(f"  {k:28s} {_fmt(diff_stats(a[k], b[k]))}")


def main() -> None:
    if "--compare" in sys.argv:
        i = sys.argv.index("--compare")
        compare_files(sys.argv[i + 1], sys.argv[i + 2])
        return

    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser()
    p.add_argument("-c", "--sim_cfg_file", default=str(ROOT / "sim" / "phase0_cfg.yaml"))
    p.add_argument("-o", "--out", required=True, help="output prefix (.npz/.json added)")
    p.add_argument("--seed-a", type=int, default=7200)
    p.add_argument("--seed-b", type=int, default=7203)
    p.add_argument("--speed", type=float, default=0.20)
    p.add_argument("--asset-seed", type=int, default=7000)
    p.add_argument("--b-extra-steps", type=int, default=40)
    p.add_argument("--static-renders", type=int, default=4)
    p.add_argument("--aa", choices=["keep", *AA_OP], default="keep")
    AppLauncher.add_app_launcher_args(p)
    args, _ = p.parse_known_args()
    args.enable_cameras = True
    AppLauncher(args)

    import carb

    from sim.scene import Phase0Scene, load_cfg

    cfg = load_cfg(args.sim_cfg_file)
    if getattr(args, "device", None):
        cfg.setdefault("sim", {})["device"] = args.device
    scene = Phase0Scene(cfg, enable_cameras=True, asset_seed=int(args.asset_seed))
    st = carb.settings.get_settings()
    if args.aa != "keep":
        st.set("/rtx/post/aa/op", AA_OP[args.aa])
        if args.aa not in ("dlss", "dlaa"):
            st.set("/rtx-transient/post/aa/limitedOps", False)
    settings = {k: st.get(k) for k in RENDER_KEYS}
    print("render settings:", json.dumps(settings, indent=1))

    frames: dict[str, np.ndarray] = {}

    def grab(tag: str) -> None:
        for cam, img in scene.get_images().items():
            frames[f"{tag}/{cam}"] = np.asarray(img)[..., :3].copy()

    def hold_step() -> None:
        ee, q = scene.get_ee_pose()
        scene.set_ee_target(ee, q, float(scene.get_gripper_open()))
        scene.step(kinematic_object=True, attach_object=False, freeze_object=False)

    def episode(tag: str, seed: int, extra: int = 0) -> None:
        scene.reset_episode(seed=seed, speed=args.speed)
        grab(f"{tag}_f0")
        for k in range(1, 4):
            hold_step()
            grab(f"{tag}_f{k}")
        for _ in range(extra):
            hold_step()
        grab(f"{tag}_last")

    episode("A1", args.seed_a, extra=args.b_extra_steps)
    episode("B", args.seed_b, extra=args.b_extra_steps)
    episode("A2", args.seed_a, extra=args.b_extra_steps)
    episode("A3", args.seed_a, extra=0)
    # Static re-render: same physics state, render again (A3 has already stepped 3 times).
    for r in range(args.static_renders):
        scene.sim.render()
        for cam in scene.cameras.values():
            cam.update(0.0, force_recompute=True)
        grab(f"static_r{r + 1}")

    cams = sorted({k.split("/")[1] for k in frames})
    pairs = [
        ("A1_f0", "A2_f0", "frame0: A after nothing vs A after B"),
        ("A2_f0", "A3_f0", "frame0: A after B vs A after A"),
        ("A1_f0", "A3_f0", "frame0: A after A-long vs A after A"),
        ("A2_f0", "B_last", "frame0 of A2 vs last frame of B (stale?)"),
        ("A2_f0", "A2_f1", "A2 frame0 vs frame1"),
        ("A3_f0", "A3_f1", "A3 frame0 vs frame1"),
        ("A1_f1", "A2_f1", "frame1: A1 vs A2"),
        ("A2_f1", "A3_f1", "frame1: A2 vs A3"),
        ("A1_f2", "A2_f2", "frame2: A1 vs A2"),
        ("A1_f3", "A2_f3", "frame3: A1 vs A2"),
        ("A2_f3", "A3_f3", "frame3: A2 vs A3"),
        ("A3_f3", "static_r1", "A3 frame3 vs 1 extra render (no step)"),
        ("static_r1", "static_r2", "static render 1 vs 2"),
        ("static_r3", "static_r4", "static render 3 vs 4"),
    ]
    report = {"settings": settings, "aa": args.aa, "pairs": {}}
    for a, b, what in pairs:
        for cam in cams:
            ka, kb = f"{a}/{cam}", f"{b}/{cam}"
            if ka in frames and kb in frames:
                s = diff_stats(frames[ka], frames[kb])
                report["pairs"][f"{a} vs {b} [{cam}]"] = s
                print(f"{what:45s} [{cam:10s}] {_fmt(s)}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out) + ".npz", **{k.replace("/", "__"): v for k, v in frames.items()})
    Path(str(out) + ".json").write_text(json.dumps(report, indent=1, default=str))
    print("wrote", str(out) + ".npz", flush=True)
    import os

    os._exit(0)


if __name__ == "__main__":
    main()

"""B6: expert picks the instructed object 100% of the time.

Plain words: the task sentence names one object. At first CLOSE the hand must
be on that object (GT ``object_pos``), not a distractor. The SM only ever sees
the target ``ObjectState`` — this check proves demos and a small probe match
that contract.

Isaac-free checks:

  1. SM probe — step APPROACH with the target state while a distractor sits
     elsewhere. Command XY must stay nearer the target forecast than the
     distractor (expert cannot “accidentally” aim at the other mesh).

  2. Demo scan — for each episode that closed:
       * language contains the target category noun
       * hand XY within ``radius`` of ``object_pos``
       * if ``distractor_pos`` is logged: nearer target than distractor
     Pass: every closed episode is ``target`` (rate = 100%).

Usage:
  PYTHONPATH=. python eval/b6_instructed_object.py
  PYTHONPATH=. python eval/b6_instructed_object.py --root data/smoke --radius-cm 6
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from interfaces.config import load_yaml  # noqa: E402
from interfaces.state import oracle_from_gt  # noqa: E402
from policy.features import object_key  # noqa: E402
from sim.state_machine import STAGE_APPROACH, STAGE_CLOSE, PickPlaceStateMachine  # noqa: E402

DEFAULT_RADIUS_M = 0.06  # 6 cm — same spirit as eval/e3_report.py
_CARRY = np.array([0.50, 0.30, 0.12], dtype=np.float64)
_DROP = np.array([0.50, 0.30, 0.06], dtype=np.float64)


def language_names_target(language: str, category: str) -> bool:
    """True if the task sentence contains the target object noun."""
    noun = object_key(category)
    if not noun or noun == "object":
        return False
    return noun.lower() in str(language).lower()


def classify_close(
    ee_xy: np.ndarray,
    obj_xy: np.ndarray,
    distractor_xy: np.ndarray | None,
    *,
    radius_m: float,
) -> str:
    """Which thing is the hand over at close: target / distractor / neither."""
    ee = np.asarray(ee_xy, dtype=np.float64).reshape(2)
    obj = np.asarray(obj_xy, dtype=np.float64).reshape(2)
    dt = float(np.linalg.norm(ee - obj))
    if distractor_xy is None:
        return "target" if dt <= float(radius_m) + 1e-12 else "neither"
    dist = np.asarray(distractor_xy, dtype=np.float64).reshape(2)
    dd = float(np.linalg.norm(ee - dist))
    if min(dt, dd) > float(radius_m) + 1e-12:
        return "neither"
    return "target" if dt <= dd else "distractor"


def probe_sm(cfg: dict) -> dict:
    """APPROACH aim must track the target, not a nearby distractor pose.

    Call with loaded phase0 cfg. Returns pass/fail for the report.
    """
    dt = float(cfg["sim"]["dt"])
    sm = PickPlaceStateMachine(cfg, dt=dt)
    sm.set_box_poses(_CARRY, drop_pos=_DROP)
    sm.set_object_half_height(0.03, z_table=0.0)
    sm.stage = STAGE_APPROACH
    sm.timer_s = 0.0
    # Target moving; distractor static 12 cm aside (E3-style spacing).
    obj = np.array([0.55, 0.0, 0.03], dtype=np.float64)
    vel = np.array([0.0, -0.20, 0.0], dtype=np.float64)
    distractor = np.array([0.55, 0.12, 0.03], dtype=np.float64)
    ee = np.array([0.45, 0.0, 0.35], dtype=np.float64)
    cmd = sm.step(oracle_from_gt(obj, vel, 0.0), ee, sm._down_quat)
    aim = np.asarray(cmd.position[:2], dtype=np.float64)
    # Compare to where the SM would forecast the target (same lead it used).
    lead = float(sm._intercept_lead_s(ee, obj, vel, goal_z=float(cmd.position[2])))
    forecast = obj[:2] + vel[:2] * lead
    d_tgt = float(np.linalg.norm(aim - forecast))
    d_dis = float(np.linalg.norm(aim - distractor[:2]))
    ok = bool(d_tgt + 1e-9 < d_dis and d_tgt <= 0.02)
    return {
        "check": "sm_probe",
        "lead_s": lead,
        "aim_xy_m": aim.tolist(),
        "forecast_xy_m": forecast.tolist(),
        "distractor_xy_m": distractor[:2].tolist(),
        "dist_to_forecast_mm": 1000.0 * d_tgt,
        "dist_to_distractor_mm": 1000.0 * d_dis,
        "pass": ok,
    }


def one_demo(path: str, *, radius_m: float) -> dict:
    """One episode: language match + which object the hand closed on."""
    p = Path(path)
    meta: dict = {}
    jp = p.with_suffix(".json")
    if jp.exists():
        meta = json.loads(jp.read_text())
    cat = str(meta.get("category") or meta.get("object_category") or "")
    lang = str(meta.get("language") or "")
    dpos = meta.get("distractor_pos")
    distractor_xy = None
    if dpos is not None:
        dpos_a = np.asarray(dpos, dtype=np.float64).reshape(-1)
        if dpos_a.size >= 2:
            distractor_xy = dpos_a[:2]

    out: dict = {
        "path": path,
        "category": cat or "unknown",
        "language": lang,
        "closed": False,
        "lang_ok": bool(cat) and language_names_target(lang, cat),
        "which": "no-close",
        "dist_target_mm": float("nan"),
        "dist_distractor_mm": float("nan"),
        "has_distractor": distractor_xy is not None,
        "pass": False,
    }
    with h5py.File(path, "r") as h:
        if "stage" not in h or "ee_pos" not in h or "object_pos" not in h:
            return out
        stage = h["stage"][:]
        ee = np.asarray(h["ee_pos"][:], dtype=np.float64)
        obj = np.asarray(h["object_pos"][:], dtype=np.float64)

    hits = np.flatnonzero(np.asarray(stage, dtype=np.int64) == STAGE_CLOSE)
    if hits.size == 0:
        return out
    i = int(hits[0])
    which = classify_close(ee[i, :2], obj[i, :2], distractor_xy, radius_m=radius_m)
    dt = float(np.linalg.norm(ee[i, :2] - obj[i, :2]))
    dd = (
        float(np.linalg.norm(ee[i, :2] - distractor_xy))
        if distractor_xy is not None
        else float("nan")
    )
    ok = bool(out["lang_ok"] and which == "target")
    out.update(
        {
            "closed": True,
            "which": which,
            "dist_target_mm": 1000.0 * dt,
            "dist_distractor_mm": 1000.0 * dd if np.isfinite(dd) else float("nan"),
            "pass": ok,
        }
    )
    return out


def summarize(rows: list[dict], *, label: str, radius_m: float) -> dict:
    """100% of closed episodes must be on the instructed target."""
    closed = [r for r in rows if r.get("closed")]
    n_close = len(closed)
    n_target = sum(1 for r in closed if r.get("which") == "target")
    n_distractor = sum(1 for r in closed if r.get("which") == "distractor")
    n_neither = sum(1 for r in closed if r.get("which") == "neither")
    n_lang_bad = sum(1 for r in closed if not r.get("lang_ok"))
    n_pass = sum(1 for r in closed if r.get("pass"))
    n_with_d = sum(1 for r in closed if r.get("has_distractor"))
    rate = float(n_pass / n_close) if n_close else 0.0
    return {
        "check": "instructed_object",
        "label": label,
        "episodes": len(rows),
        "closed": n_close,
        "radius_mm": 1000.0 * radius_m,
        "n_target": n_target,
        "n_distractor": n_distractor,
        "n_neither": n_neither,
        "n_lang_mismatch": n_lang_bad,
        "n_with_distractor": n_with_d,
        "pass_rate": rate,
        "pass": bool(n_close >= 1 and n_pass == n_close),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        action="append",
        default=None,
        help="Demo root with .h5 (repeatable). Default: A1_velocity + smoke",
    )
    ap.add_argument("--radius-cm", type=float, default=100.0 * DEFAULT_RADIUS_M)
    ap.add_argument(
        "--skip-demos",
        action="store_true",
        help="Only run the SM distractor-aim probe.",
    )
    ap.add_argument(
        "-o",
        "--output",
        default=str(ROOT / "data" / "eval" / "B6_instructed_object" / "b6_report.json"),
    )
    args = ap.parse_args()
    cfg = load_yaml(ROOT / "sim" / "phase0_cfg.yaml")
    radius_m = float(args.radius_cm) * 1e-2

    sm_rep = probe_sm(cfg)
    print(json.dumps(sm_rep, indent=2), flush=True)

    demo_reps: list[dict] = []
    if not args.skip_demos:
        roots = args.root or [
            str(ROOT / "data" / "eval" / "A1_velocity"),
            str(ROOT / "data" / "smoke"),
        ]
        for root in roots:
            root_p = Path(root)
            if not root_p.exists():
                continue
            files = sorted(glob.glob(str(root_p / "**" / "*.h5"), recursive=True))
            rows = [one_demo(f, radius_m=radius_m) for f in files]
            rep = summarize(rows, label=str(root_p), radius_m=radius_m)
            demo_reps.append(rep)
            print(json.dumps(rep, indent=2), flush=True)

    all_pass = bool(sm_rep["pass"]) and (
        True if args.skip_demos else (bool(demo_reps) and all(r["pass"] for r in demo_reps))
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "radius_mm": 1000.0 * radius_m,
        "sm_probe": sm_rep,
        "demos": demo_reps,
        "pass": all_pass,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# B6: expert picks the instructed object 100%",
        "",
        f"Gate: every CLOSE is on the named target (radius {1000.0 * radius_m:.0f} mm); "
        "SM aim nearer target forecast than distractor.",
        "",
        "| check | detail | pass |",
        "|---|---|---|",
        f"| SM probe | d_forecast={sm_rep['dist_to_forecast_mm']:.1f} mm, "
        f"d_distractor={sm_rep['dist_to_distractor_mm']:.1f} mm | "
        f"{'yes' if sm_rep['pass'] else 'no'} |",
    ]
    for r in demo_reps:
        lines.append(
            f"| demos `{Path(r['label']).name}` | "
            f"closed={r['closed']}, target={r['n_target']}, "
            f"distractor={r['n_distractor']}, neither={r['n_neither']}, "
            f"lang_bad={r['n_lang_mismatch']}, "
            f"with_distractor_meta={r['n_with_distractor']}, "
            f"rate={100.0 * r['pass_rate']:.0f}% | "
            f"{'yes' if r['pass'] else 'no'} |"
        )
    lines.append("")
    lines.append(f"**Overall: {'PASS' if all_pass else 'FAIL'}**")
    lines.append("")
    if demo_reps and all(r.get("n_with_distractor", 0) == 0 for r in demo_reps):
        lines.append(
            "Note: no distractor_pos in these demos — hand-on-target + language "
            "match only. Re-run with E3-style distractor collects for the hard form."
        )
        lines.append("")
    md = "\n".join(lines) + "\n"
    (out.parent / "REPORT_auto.md").write_text(md)
    print(md)
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

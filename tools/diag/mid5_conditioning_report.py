#!/usr/bin/env python3
"""Merge mid eval reports + per-episode conditioning debug for B / B_zero."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from interfaces.state import CONDITIONING_LAYOUT  # noqa: E402


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _arm_summary(metrics: list[dict]) -> dict:
    conds = [r["conditioning_debug"] for r in metrics if r.get("conditioning_debug")]
    out: dict = {
        "n_episodes": len(metrics),
        "n_with_conditioning": len(conds),
        "successes": sum(1 for r in metrics if r.get("success")),
        "by_speed": {},
        "episodes": [],
    }
    for r in metrics:
        sp = float(r["speed_m_s"])
        out["by_speed"].setdefault(sp, {"attempts": 0, "successes": 0})
        out["by_speed"][sp]["attempts"] += 1
        out["by_speed"][sp]["successes"] += int(bool(r.get("success")))
        ep = {
            "speed_m_s": sp,
            "seed": r.get("seed"),
            "success": r.get("success"),
            "failure": r.get("failure"),
            "zero_conditioning": r.get("zero_conditioning"),
        }
        cd = r.get("conditioning_debug")
        if cd:
            ep["conditioning"] = {
                "layout": cd.get("layout", list(CONDITIONING_LAYOUT)),
                "first_replan": cd.get("first_replan"),
                "mean_replan": cd.get("mean_replan"),
                "l2_mean_replan": cd.get("l2_mean_replan"),
                "all_zero": cd.get("all_zero"),
            }
        out["episodes"].append(ep)

    if conds:
        first = np.asarray([c["first_replan"] for c in conds], dtype=np.float64)
        mean = np.asarray([c["mean_replan"] for c in conds], dtype=np.float64)
        l2 = np.asarray([c["l2_mean_replan"] for c in conds], dtype=np.float64)
        out["conditioning_pool"] = {
            "layout": list(CONDITIONING_LAYOUT),
            "mean_of_first_replan": first.mean(axis=0).tolist(),
            "std_of_first_replan": first.std(axis=0).tolist(),
            "mean_of_mean_replan": mean.mean(axis=0).tolist(),
            "l2_mean": float(l2.mean()),
            "l2_std": float(l2.std()),
            "fraction_all_zero": float(np.mean([1.0 if c.get("all_zero") else 0.0 for c in conds])),
        }
        out["per_dim"] = [
            {
                "index": i,
                "name": CONDITIONING_LAYOUT[i],
                "mean_first_replan": float(first[:, i].mean()),
                "std_first_replan": float(first[:, i].std()),
                "mean_across_episode": float(mean[:, i].mean()),
            }
            for i in range(12)
        ]
    return out


def _md_table(per_dim: list[dict]) -> str:
    lines = [
        "| idx | name | mean(first_replan) | std | mean(ep mean) |",
        "|---:|---|---:|---:|---:|",
    ]
    for d in per_dim:
        lines.append(
            f"| {d['index']} | `{d['name']}` | {d['mean_first_replan']:.6g} | "
            f"{d['std_first_replan']:.6g} | {d['mean_across_episode']:.6g} |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True, help="OUT_ROOT of mid reeval")
    args = p.parse_args()
    root: Path = args.root

    gate = {}
    gate_path = root / "gate_g1_report.json"
    if gate_path.is_file():
        gate = json.loads(gate_path.read_text())

    arms = {}
    for tag in ("a", "b", "b_zero"):
        metrics = _load_jsonl(root / f"eval_{tag}" / "metrics.jsonl")
        arms[tag] = _arm_summary(metrics)

    report = {
        "root": str(root),
        "gate": gate.get("metrics", gate),
        "note": "n=5 mid is for debug viz + conditioning dump; magnitude claims stay INCONCLUSIVE.",
        "arms": arms,
        "mp4_glob": {
            "a": sorted(str(p) for p in (root / "eval_a").glob("*.mp4")),
            "b": sorted(str(p) for p in (root / "eval_b").glob("*.mp4")),
            "b_zero": sorted(str(p) for p in (root / "eval_b_zero").glob("*.mp4")),
        },
    }
    out_json = root / "mid5_report.json"
    out_json.write_text(json.dumps(report, indent=2))

    md = [
        "# Mid-5 eval report (debug)",
        "",
        f"Root: `{root}`",
        "",
        "## Gate metrics (if merged)",
        "```json",
        json.dumps(report["gate"], indent=2),
        "```",
        "",
        "## Success counts",
    ]
    for tag in ("a", "b", "b_zero"):
        a = arms[tag]
        md.append(
            f"- **{tag}**: {a['successes']}/{a['n_episodes']} "
            f"(conditioning rows: {a['n_with_conditioning']})"
        )
    md.append("")

    for tag in ("b", "b_zero"):
        a = arms[tag]
        md.append(f"## Conditioning 12-vector — `{tag}`")
        md.append("")
        if not a.get("per_dim"):
            md.append("_No conditioning_debug in metrics.jsonl._")
            md.append("")
            continue
        pool = a["conditioning_pool"]
        md.append(
            f"- L2 mean (replan): **{pool['l2_mean']:.6g}** ± {pool['l2_std']:.6g}"
        )
        md.append(f"- fraction all-zero episodes: **{pool['fraction_all_zero']:.3f}**")
        md.append("")
        md.append(_md_table(a["per_dim"]))
        md.append("")
        md.append("### Per-episode first_replan vector")
        md.append("")
        for ep in a["episodes"]:
            c = ep.get("conditioning")
            if not c:
                continue
            vec = ", ".join(f"{x:.4g}" for x in c["first_replan"])
            md.append(
                f"- v={ep['speed_m_s']:.2f} seed={ep['seed']} "
                f"ok={ep['success']} why={ep['failure']} "
                f"L2={c['l2_mean_replan']:.4g} all_zero={c['all_zero']}"
            )
            md.append(f"  - `[{vec}]`")
        md.append("")

    md.append("## MP4 paths")
    for tag in ("a", "b", "b_zero"):
        md.append(f"### {tag}")
        for path in report["mp4_glob"][tag]:
            md.append(f"- `{path}`")
        md.append("")

    out_md = root / "mid5_report.md"
    out_md.write_text("\n".join(md) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()

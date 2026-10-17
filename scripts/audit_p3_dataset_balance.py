#!/usr/bin/env python3
"""WORKSTREAM: eval — audit p3_throw clean-set balance (speed / object / container / gpu).

Read-only on H5+JSON and LeRobot trees. Does not train.

Usage:
  /home/gpuadmin/anaconda3/envs/dynamicVLA_training/bin/python \\
    scripts/audit_p3_dataset_balance.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from policy.convert_h5 import list_success_episodes  # noqa: E402

RAW = ROOT / "data" / "p3_throw"
OUT = ROOT / "data" / "eval" / "p3_throw_balance"
MAIN_N = 4117
TAIL_N = 1451
FULL_N = MAIN_N + TAIL_N  # 5568

SPEED_BINS = ("0 cm/s", "~10 cm/s", "~15 cm/s", "~20 cm/s")
OBJECTS = (
    "apple",
    "avocado",
    "can",
    "egg",
    "kiwi",
    "lemon",
    "lime",
    "onion",
    "orange",
    "peach",
    "tangerine",
)
UNIFORM_SPEED = 0.25
SPEED_IMBALANCE_PTS = 10.0  # share vs 25% differs by >10 pts
OBJ_LO, OBJ_HI = 0.05, 0.15


def speed_bin(speed: float) -> str:
    if speed < 0.05:
        return "0 cm/s"
    if speed < 0.125:
        return "~10 cm/s"
    if speed < 0.175:
        return "~15 cm/s"
    return "~20 cm/s"


def gpu_from_path(h5: Path) -> str:
    # First path segment under data/p3_throw/ is gpu0 or gpu1 (not "gpuadmin" from $HOME).
    try:
        return h5.resolve().relative_to(RAW.resolve()).parts[0]
    except (ValueError, IndexError):
        return "unknown"


def load_rows() -> list[dict]:
    pairs = list_success_episodes(RAW, success_only=True, clean_only=True)
    if len(pairs) != FULL_N:
        print(f"WARN: expected {FULL_N} clean pairs, got {len(pairs)}")
    rows = []
    for i, (h5, js) in enumerate(pairs):
        meta = json.loads(js.read_text())
        rows.append(
            {
                "episode_index": i,
                "split": "main" if i < MAIN_N else "tail",
                "speed_m_s": float(meta.get("speed_m_s", float("nan"))),
                "speed_bin": speed_bin(float(meta.get("speed_m_s", 0.0))),
                "category": str(meta.get("category") or "unknown"),
                "container": str(meta.get("container_category") or "unknown"),
                "gpu": gpu_from_path(h5),
                "h5": str(h5),
            }
        )
    return rows


def counts(rows: list[dict], key: str) -> Counter:
    return Counter(r[key] for r in rows)


def pct_table(ctr: Counter, order: tuple[str, ...] | None, n: int) -> list[tuple[str, int, float]]:
    keys = list(order) if order else sorted(ctr.keys())
    for k in sorted(ctr.keys()):
        if k not in keys:
            keys.append(k)
    return [(k, ctr.get(k, 0), 100.0 * ctr.get(k, 0) / n if n else 0.0) for k in keys]


def flag_speed(rows: list[dict]) -> list[str]:
    n = len(rows)
    if not n:
        return ["empty set"]
    ctr = counts(rows, "speed_bin")
    flags = []
    for b in SPEED_BINS:
        share = ctr.get(b, 0) / n
        pts = abs(100.0 * share - 100.0 * UNIFORM_SPEED)
        if pts > SPEED_IMBALANCE_PTS:
            flags.append(f"speed {b}: {100*share:.1f}% (Δ{pts:.1f} pts vs 25%)")
    return flags


def flag_object(rows: list[dict]) -> list[str]:
    n = len(rows)
    if not n:
        return ["empty set"]
    ctr = counts(rows, "category")
    flags = []
    for obj in OBJECTS:
        share = ctr.get(obj, 0) / n
        if share < OBJ_LO or share > OBJ_HI:
            flags.append(f"object {obj}: {100*share:.1f}% (want 5–15%)")
    for k, c in sorted(ctr.items()):
        if k not in OBJECTS:
            flags.append(f"unexpected object {k}: n={c}")
    return flags


def md_section(name: str, rows: list[dict]) -> str:
    n = len(rows)
    lines = [f"## {name} (n={n})", ""]
    lines.append("### Speed bin")
    lines.append("")
    lines.append("| bin | count | % |")
    lines.append("|---|---:|---:|")
    for k, c, p in pct_table(counts(rows, "speed_bin"), SPEED_BINS, n):
        lines.append(f"| {k} | {c} | {p:.1f} |")
    lines.append("")
    lines.append("### Object")
    lines.append("")
    lines.append("| object | count | % |")
    lines.append("|---|---:|---:|")
    for k, c, p in pct_table(counts(rows, "category"), OBJECTS, n):
        lines.append(f"| {k} | {c} | {p:.1f} |")
    lines.append("")
    lines.append("### Container")
    lines.append("")
    lines.append("| container | count | % |")
    lines.append("|---|---:|---:|")
    for k, c, p in pct_table(counts(rows, "container"), ("box", "bowl"), n):
        lines.append(f"| {k} | {c} | {p:.1f} |")
    lines.append("")
    lines.append("### GPU")
    lines.append("")
    lines.append("| gpu | count | % |")
    lines.append("|---|---:|---:|")
    for k, c, p in pct_table(counts(rows, "gpu"), ("gpu0", "gpu1"), n):
        lines.append(f"| {k} | {c} | {p:.1f} |")
    lines.append("")
    sflags = flag_speed(rows)
    oflags = flag_object(rows)
    lines.append("### Imbalance flags")
    lines.append("")
    if not sflags and not oflags:
        lines.append("- none")
    else:
        for f in sflags + oflags:
            lines.append(f"- {f}")
    lines.append("")
    return "\n".join(lines)


def plot_speed_mix(main: list[dict], full: list[dict], path: Path) -> None:
    labels = list(SPEED_BINS)
    main_pct = [100.0 * counts(main, "speed_bin").get(b, 0) / len(main) for b in labels]
    full_pct = [100.0 * counts(full, "speed_bin").get(b, 0) / len(full) for b in labels]
    x = range(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar([i - w / 2 for i in x], main_pct, w, label=f"main (n={len(main)})", color="#4C78A8")
    ax.bar([i + w / 2 for i in x], full_pct, w, label=f"full (n={len(full)})", color="#F58518")
    ax.axhline(25.0, color="gray", linestyle="--", linewidth=1, label="uniform 25%")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("% of episodes")
    ax.set_title("p3_throw speed mix: main vs full clean set")
    ax.set_ylim(0, max(45, max(main_pct + full_pct) + 5))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def recommend(full: list[dict], main: list[dict]) -> tuple[str, str]:
    """yes/no + one-line reason."""
    main_sf = flag_speed(main)
    full_sf = flag_speed(full)
    main_hi = counts(main, "speed_bin").get("~15 cm/s", 0) + counts(main, "speed_bin").get("~20 cm/s", 0)
    full_hi = counts(full, "speed_bin").get("~15 cm/s", 0) + counts(full, "speed_bin").get("~20 cm/s", 0)
    main_hi_pct = 100.0 * main_hi / len(main)
    full_hi_pct = 100.0 * full_hi / len(full)
    if main_sf and (not full_sf or full_hi_pct > main_hi_pct + 5):
        return (
            "yes",
            f"Main alone is slow-biased ({main_hi_pct:.0f}% at 15–20 cm/s); "
            f"full set lifts that to {full_hi_pct:.0f}% and clears/cuts speed-bin flags.",
        )
    if full_sf:
        return (
            "yes",
            f"Still train on full (more 15–20 cm/s coverage: {full_hi_pct:.0f}% vs main {main_hi_pct:.0f}%), "
            "but watch remaining speed-bin imbalance.",
        )
    return (
        "yes",
        f"Full set is closer to even speed mix and keeps G3 15–20 cm/s coverage "
        f"({full_hi_pct:.0f}% vs main {main_hi_pct:.0f}%).",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    main = [r for r in rows if r["split"] == "main"]
    tail = [r for r in rows if r["split"] == "tail"]
    full = rows

    # Optional: sanity-check tail source map size
    src = ROOT / "data" / "lerobot" / "p3_throw_tail" / "meta" / "source_h5.jsonl"
    if src.exists():
        n_src = sum(1 for line in src.read_text().splitlines() if line.strip())
        print(f"tail source_h5.jsonl lines={n_src} reconstructed_tail={len(tail)}")

    print("NOTE: This is data balance, not model accuracy (Wilson note applies to rates, not mix).")
    for name, subset in (("main", main), ("tail", tail), ("full", full)):
        print(f"\n=== {name} n={len(subset)} ===")
        for k, c, p in pct_table(counts(subset, "speed_bin"), SPEED_BINS, len(subset)):
            print(f"  speed {k:8s} {c:5d}  {p:5.1f}%")
        sf, of = flag_speed(subset), flag_object(subset)
        print("  flags:", "; ".join(sf + of) if (sf or of) else "none")

    png = OUT / "speed_mix_main_vs_full.png"
    plot_speed_mix(main, full, png)

    rec, reason = recommend(full, main)
    report = []
    report.append("# p3_throw clean-set balance audit")
    report.append("")
    report.append(
        "Source: `list_success_episodes(data/p3_throw, success_only=True, clean_only=True)` "
        f"— full n={len(full)}; main = first {MAIN_N}; tail = rest ({len(tail)})."
    )
    report.append("")
    report.append(
        "**Wilson-style note:** these numbers are *data mix* (counts / %), not model "
        "accuracy. Do not treat bin % as a success rate or claim a gate from them."
    )
    report.append("")
    report.append(f"**Recommendation:** train on merged full set only? **{rec}** — {reason}")
    report.append("")
    report.append(f"Plot: `{png.relative_to(ROOT)}`")
    report.append("")
    report.append(md_section("Full clean set", full))
    report.append(md_section("Main only", main))
    report.append(md_section("Tail only", tail))
    report_path = OUT / "REPORT.md"
    report_path.write_text("\n".join(report) + "\n")
    print(f"\nWrote {report_path}")
    print(f"Wrote {png}")
    print(f"RECOMMEND={rec}: {reason}")


if __name__ == "__main__":
    main()

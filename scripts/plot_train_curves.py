#!/usr/bin/env python3
"""Re-plot train/val curves from a run's metrics.jsonl.

Usage:
  python scripts/plot_train_curves.py runs/smoke_metrics
  python scripts/plot_train_curves.py runs/smoke_metrics/metrics.jsonl -o /tmp/curves.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from policy.train_metrics import plot_metrics_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description="Plot train/val loss from metrics.jsonl")
    p.add_argument(
        "path",
        type=str,
        help="run dir (contains metrics.jsonl) or path to metrics.jsonl",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="output PNG (default: <run>/curves.png)",
    )
    args = p.parse_args()
    path = Path(args.path)
    if path.is_dir():
        metrics = path / "metrics.jsonl"
        out = Path(args.output) if args.output else path / "curves.png"
    else:
        metrics = path
        out = Path(args.output) if args.output else path.parent / "curves.png"
    written = plot_metrics_jsonl(metrics, out_png=out)
    if written is None:
        raise SystemExit(f"no train_loss/val_loss rows in {metrics}")
    print(written)


if __name__ == "__main__":
    main()

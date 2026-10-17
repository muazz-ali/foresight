"""Train metrics: jsonl log, curves plot, short end-of-run hints.

Plain helpers for policy/train_smolvla.py. Not a gate — just so you can see
if loss is moving, and if val rises while train falls (overfit hint).
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Soft hints only (not gate claims). Tuned to be conservative.
_UNDERFIT_REL_DROP = 0.05  # train loss barely moved if drop < 5% of first
_OVERFIT_VAL_RISE = 0.10  # val rose >=10% from best while train still fell


@dataclass
class MetricsLogger:
    """Append one JSON object per log step; write CSV + curves at the end."""

    run_dir: Path
    rows: list[dict[str, Any]] = field(default_factory=list)
    _best_val: float | None = None
    _first_train: float | None = None
    _last_train: float | None = None

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        # Fresh file each run so a re-smoke does not mix old lines.
        self.metrics_path.write_text("")

    def log(self, step: int, **fields: Any) -> None:
        row = {"step": int(step), **fields}
        self.rows.append(row)
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(row, default=_json_default) + "\n")

        train = fields.get("train_loss")
        if train is not None:
            t = float(train)
            if self._first_train is None:
                self._first_train = t
            self._last_train = t

        val = fields.get("val_loss")
        if val is not None:
            v = float(val)
            if self._best_val is None or v < self._best_val:
                self._best_val = v

    def write_csv(self) -> Path:
        path = self.run_dir / "metrics.csv"
        if not self.rows:
            path.write_text("step\n")
            return path
        keys: list[str] = []
        seen: set[str] = set()
        for row in self.rows:
            for k in row:
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in self.rows:
                w.writerow(row)
        return path

    def write_curves(self) -> Path | None:
        """Train / val loss vs step → curves.png. Returns path or None if empty."""
        path = self.run_dir / "curves.png"
        train_xy = [
            (r["step"], float(r["train_loss"]))
            for r in self.rows
            if r.get("train_loss") is not None
        ]
        val_xy = [
            (r["step"], float(r["val_loss"]))
            for r in self.rows
            if r.get("val_loss") is not None
        ]
        if not train_xy and not val_xy:
            return None
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib missing; skip curves.png")
            return None

        fig, ax = plt.subplots(figsize=(7, 4))
        if train_xy:
            ax.plot([x for x, _ in train_xy], [y for _, y in train_xy], label="train loss")
        if val_xy:
            ax.plot([x for x, _ in val_xy], [y for _, y in val_xy], label="val loss")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title("train / val loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def summarize(self) -> str:
        """Short plain-English end note with soft under/overfit heuristics."""
        parts = [
            f"final train loss={_fmt(self._last_train)}",
            f"best val loss={_fmt(self._best_val)}",
        ]
        hint = _fit_hint(
            first_train=self._first_train,
            last_train=self._last_train,
            best_val=self._best_val,
            rows=self.rows,
        )
        if hint:
            parts.append(hint)
        text = " | ".join(parts)
        logger.info("end-of-run summary (heuristics, not a gate): %s", text)
        return text

    def finalize(self) -> dict[str, Any]:
        csv_path = self.write_csv()
        curves = self.write_curves()
        summary = self.summarize()
        return {
            "metrics_jsonl": str(self.metrics_path),
            "metrics_csv": str(csv_path),
            "curves_png": str(curves) if curves else None,
            "summary": summary,
        }


def split_episodes(
    n_episodes: int,
    val_fraction: float,
    seed: int,
    *,
    max_episodes: int | None = None,
) -> tuple[list[int], list[int]]:
    """Episode-level train/val split: hold out every N-th episode.

    Spreads val across the file so one speed band (e.g. the slow tail of a
    merge) does not own the whole hold-out. For ``val_fraction=0.05`` that is
    every 20th episode (0, 20, 40, …). The t-rainer still loads all episodes
    and builds train/val as frame Subsets, so a gappy val id list is fine.

    ``seed`` is kept for API stability; the stride split is deterministic.
    val_fraction=0 → all train, no val. max_episodes truncates before split
    (handy for a tiny smoke run).
    """
    del seed  # reserved; stride split does not shuffle
    if n_episodes <= 0:
        raise ValueError(f"n_episodes must be > 0, got {n_episodes}")
    n = int(n_episodes)
    if max_episodes is not None:
        n = min(n, max(1, int(max_episodes)))
    eps = list(range(n))
    frac = float(val_fraction)
    if frac <= 0.0 or n < 2:
        return eps, []

    # 5% → every 20th; 10% → every 10th. Cap so train keeps at least one ep.
    stride = max(2, int(round(1.0 / frac)))
    val = eps[::stride]
    if len(val) >= n:
        val = eps[::stride][: max(1, n - 1)]
    train = [e for e in eps if e not in set(val)]
    if not train:
        # Tiny n edge case: keep episode 0 for train.
        train, val = [eps[0]], eps[1:] or []
    return train, val


def _fit_hint(
    *,
    first_train: float | None,
    last_train: float | None,
    best_val: float | None,
    rows: list[dict[str, Any]],
) -> str:
    hints: list[str] = []
    if first_train is not None and last_train is not None and first_train > 1e-12:
        drop = (first_train - last_train) / first_train
        if drop < _UNDERFIT_REL_DROP:
            hints.append(
                f"underfit hint: train loss barely moved "
                f"({drop:.1%} drop; need >{_UNDERFIT_REL_DROP:.0%} to look alive)"
            )

    val_series = [float(r["val_loss"]) for r in rows if r.get("val_loss") is not None]
    train_series = [
        (int(r["step"]), float(r["train_loss"]))
        for r in rows
        if r.get("train_loss") is not None
    ]
    # Need a few val points so one noisy eval on a toy smoke does not alarm.
    if best_val is not None and len(val_series) >= 3 and len(train_series) >= 2:
        last_val = val_series[-1]
        # Val rising from best while last train is below first → soft overfit hint.
        if last_val > best_val * (1.0 + _OVERFIT_VAL_RISE):
            if train_series[-1][1] < train_series[0][1]:
                rise = (last_val - best_val) / max(best_val, 1e-12)
                hints.append(
                    f"overfit hint: val rose {rise:.1%} off best while train fell "
                    f"(threshold {_OVERFIT_VAL_RISE:.0%}; heuristic only)"
                )
    return "; ".join(hints)


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"not JSON-serializable: {type(obj)}")


def plot_metrics_jsonl(metrics_path: Path, out_png: Path | None = None) -> Path | None:
    """Re-plot curves from an existing metrics.jsonl (used by scripts/)."""
    path = Path(metrics_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    run_dir = path.parent
    ml = MetricsLogger(run_dir=run_dir)
    ml.rows = rows
    # Do not wipe metrics.jsonl — only redraw the image.
    ml.metrics_path = path
    if out_png is not None:
        # Temporary redirect of write path via monkey-patch of run_dir file name.
        png = Path(out_png)
        train_xy = [
            (r["step"], float(r["train_loss"]))
            for r in rows
            if r.get("train_loss") is not None
        ]
        val_xy = [
            (r["step"], float(r["val_loss"]))
            for r in rows
            if r.get("val_loss") is not None
        ]
        if not train_xy and not val_xy:
            return None
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        if train_xy:
            ax.plot([x for x, _ in train_xy], [y for _, y in train_xy], label="train loss")
        if val_xy:
            ax.plot([x for x, _ in val_xy], [y for _, y in val_xy], label="val loss")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title("train / val loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png, dpi=120)
        plt.close(fig)
        return png
    return ml.write_curves()

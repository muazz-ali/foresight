#!/usr/bin/env python3
"""Train SmolVLA on a Foresight LeRobot dataset.

Images + robot pose + text + 4 live future numbers (XY p̂/v̂; fixed Δ from yaml + noise).

Step 2 of 3 (convert -> train -> infer).

What the model reads each step:
  * 2 camera images (front + wrist) and a text task.
  * 12 state numbers = 8 robot numbers + 4 object numbers
    (where the object will be in XY, and its XY speed).
    Layout lives in policy/features.py. Eval / infer must build it the same way.
What it learns: the next chunk of robot actions.

Each batch, policy/transforms.py rebuilds the 4 object numbers from perfect
sim data and adds small noise. This is meant to make training less sensitive
to small tracker errors.

Only the small action part (the "expert") trains. Vision stays as it was
shipped. If results are bad, fix data or the 4 numbers first.

Metrics (runs/<run>/metrics.jsonl + curves.png): train/val loss, lr,
step time, grad norm. Soft under/overfit hints only — not a gate claim.

Conda: dynamicVLA_training
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

FORESIGHT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORESIGHT_ROOT))

logger = logging.getLogger(__name__)


def _collate(batch: list[dict]) -> dict:
    """Stack tensor fields; keep task as list[str]."""
    # Text is not a tensor, so it stays a plain list of strings.
    keys = batch[0].keys()
    out: dict = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if k == "task":
            out[k] = vals
        elif isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out


def build_policy(
    ds_meta,
    device: str,
    pretrained: str | None,
    *,
    chunk_size: int | None = None,
    frames: dict | None = None,
    dataset_root: str | Path | None = None,
):
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    from policy.transforms import expand_state_stats, frame_stats, policy_features_for_model

    # Tell the model its state is 12 numbers wide (8 robot + 4 object),
    # and give it matching scaling stats. Skip this and shapes will not fit.
    features = policy_features_for_model(ds_meta.features)
    stats = expand_state_stats(ds_meta.stats)
    # Rel cond / delta action change what those columns mean: rescale them too.
    if frames and dataset_root is not None:
        stats = frame_stats(
            stats,
            dataset_root,
            cond_frame=frames["cond_frame"],
            action_frame=frames["action_frame"],
            chunk_size=int(chunk_size or 50),
        )

    policy_features = dataset_to_policy_features(features)
    cfg = SmolVLAConfig()
    # Chunk = how many future actions the model predicts at once.
    if chunk_size:
        cfg.chunk_size = int(chunk_size)
        cfg.n_action_steps = int(chunk_size)
    cfg.pretrained_path = pretrained
    # LeRobot validates device as cuda|mps|cpu only (rejects cuda:N). Placement
    # still uses the indexed device via policy.to(device) below.
    cfg.device = (
        "cuda"
        if str(device).startswith("cuda")
        else ("mps" if str(device).startswith("mps") else "cpu")
    )
    cfg.input_features = {
        k: ft for k, ft in policy_features.items() if ft.type is not FeatureType.ACTION
    }
    cfg.output_features = {
        k: ft for k, ft in policy_features.items() if ft.type is FeatureType.ACTION
    }
    # Inputs = images + 12-number state. Output = action chunk.
    # Load the full public SmolVLA checkpoint (vision language model is already inside).
    # Vision is frozen on purpose. Only the action expert learns.
    cfg.freeze_vision_encoder = True
    cfg.train_expert_only = True
    cfg.load_vlm_weights = False

    # Start from the public SmolVLA weights when given; else random start.
    if pretrained:
        policy = SmolVLAPolicy.from_pretrained(
            pretrained_name_or_path=pretrained,
            config=cfg,
            dataset_stats=stats,
        )
    else:
        policy = SmolVLAPolicy(config=cfg, dataset_stats=stats)
    policy.to(device)
    policy.train()
    return policy, cfg


def _to_device(batch: dict, device: str) -> dict:
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


@torch.no_grad()
def _eval_loss(policy, loader: DataLoader | None, device: str, max_batches: int = 8) -> float | None:
    """Mean forward loss on a few held-out batches. None if no val loader."""
    if loader is None:
        return None
    was_training = policy.training
    policy.eval()
    total = 0.0
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = _to_device(batch, device)
        with torch.no_grad():  # val needs no autograd graph (was building one)
            loss, _ = policy.forward(batch)
        total += float(loss.detach())
        n += 1
    if was_training:
        policy.train()
    if n == 0:
        return None
    return total / n


def _make_loader(ds, args: argparse.Namespace, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=_collate,
        drop_last=shuffle,  # drop_last only for train (stable batch size)
    )


def train(args: argparse.Namespace) -> Path:
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    from policy.train_metrics import MetricsLogger, split_episodes
    from policy.transforms import AugConfig, PolicyDataset

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(args.dataset_root)

    # Build a temporary config so delta_timestamps match SmolVLA chunk_size.
    probe_cfg = SmolVLAConfig()
    if args.chunk_size:
        probe_cfg.chunk_size = int(args.chunk_size)
        probe_cfg.n_action_steps = int(args.chunk_size)
    # Peek at episode 0 just to read dataset meta (fps, features).
    meta_ds = LeRobotDataset(repo_id=args.repo_id, root=root, episodes=[0])
    delta_timestamps = resolve_delta_timestamps(probe_cfg, meta_ds.meta)
    n_ep = int(meta_ds.meta.total_episodes)

    train_eps, val_eps = split_episodes(
        n_ep,
        args.val_fraction,
        args.seed,
        max_episodes=args.max_episodes,
    )
    # One contiguous LeRobot load (0..N-1). Gappy episode filters break
    # episode_data_index lookups, so train/val is a frame Subset on top.
    use_eps = sorted(set(train_eps) | set(val_eps))
    logger.info(
        "episode split: train=%d val=%d (every-N-th, val_fraction=%.3f "
        "max_episodes=%s val_eps[:5]=%s ...)",
        len(train_eps),
        len(val_eps),
        args.val_fraction,
        args.max_episodes,
        val_eps[:5],
    )

    base = LeRobotDataset(
        repo_id=args.repo_id,
        root=root,
        episodes=use_eps,
        delta_timestamps=delta_timestamps,
    )

    def _frames_for(eps: list[int]) -> list[int]:
        edi = base.episode_data_index
        out: list[int] = []
        for e in eps:
            out.extend(range(int(edi["from"][e]), int(edi["to"][e])))
        return out

    # Look-ahead Δ + noise sizes for the 4 object numbers. Δ must match what
    # the sim logged and what eval will use. Noise is for robustness only.
    aug = AugConfig(
        delta_min=args.delta_min,
        delta_max=args.delta_max,
        pos_noise_std=args.pos_noise_std,
        vel_noise_std=args.vel_noise_std,
        seed=args.seed,
        cond_frame=args.cond_frame,
        action_frame=args.action_frame,
        ee_noise_std=args.ee_noise_std,
    )
    # Wrapper builds the 12-number state (with fresh noise) on every read.
    ds_train = Subset(
        PolicyDataset(base, aug=aug, apply_aug=True),
        _frames_for(train_eps),
    )
    loader = _make_loader(ds_train, args, shuffle=True)
    val_loader = None
    if val_eps:
        # Val: rebuild the 4 numbers from oracle, but no train noise (cleaner signal).
        aug_val = AugConfig(
            delta_min=args.delta_min,
            delta_max=args.delta_max,
            pos_noise_std=0.0,
            vel_noise_std=0.0,
            seed=args.seed,
            cond_frame=args.cond_frame,
            action_frame=args.action_frame,
            ee_noise_std=0.0,
        )
        ds_val = Subset(
            PolicyDataset(base, aug=aug_val, apply_aug=True),
            _frames_for(val_eps),
        )
        val_loader = _make_loader(ds_val, args, shuffle=False)

    policy, cfg = build_policy(
        base.meta,
        device=device,
        pretrained=args.pretrained,
        chunk_size=args.chunk_size,
        frames={"cond_frame": args.cond_frame, "action_frame": args.action_frame},
        dataset_root=root,
    )
    # Eval reads this next to every checkpoint to rebuild the same inputs.
    frames_meta = {
        "cond_frame": args.cond_frame,
        "action_frame": args.action_frame,
        "ee_noise_std": args.ee_noise_std,
        "pos_noise_std": args.pos_noise_std,
        "vel_noise_std": args.vel_noise_std,
        "delta": args.delta_min,
    }

    def _save(path: Path) -> None:
        from policy.transforms import FRAMES_FILE

        policy.save_pretrained(path)
        (path / FRAMES_FILE).write_text(json.dumps(frames_meta, indent=2) + "\n")

    # ----- Precision -----
    # SmolVLA builds its action expert from the VLM config, which says bfloat16, so
    # the trainable weights arrive in bf16. AdamW on bf16 weights loses small
    # updates (lr * update is below bf16 resolution): before this fix the expert
    # norm weights never moved and 14 % of down_proj entries stayed unchanged.
    # Fix = fp32 master weights for everything that trains + bf16 autocast for
    # the forward pass (the frozen VLM stays bf16 to save memory).
    trainable = [p for p in policy.parameters() if p.requires_grad]
    if args.precision == "fp32-master":
        for prm in trainable:
            prm.data = prm.data.float()
    dtypes: dict = {}
    for prm in trainable:
        dtypes[str(prm.dtype)] = dtypes.get(str(prm.dtype), 0) + prm.numel()
    logger.info("trainable params by dtype: %s (precision=%s)", dtypes, args.precision)
    dev_type = "cuda" if str(device).startswith("cuda") else "cpu"
    amp = (
        (lambda: torch.autocast(device_type=dev_type, dtype=torch.bfloat16))
        if args.precision == "fp32-master"
        else nullcontext
    )

    # ----- Optimizer + schedule: SmolVLA presets (lerobot SmolVLAConfig) -----
    # betas (0.9, 0.95), eps 1e-8, weight decay 1e-10, grad clip 10,
    # 1k linear warm-up then cosine to 2.5e-6. Decay length defaults to the run
    # length (capped at the preset 30k) so a 20k run actually finishes its decay.
    from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

    decay_steps = int(args.scheduler_decay_steps or min(cfg.scheduler_decay_steps, args.steps))
    warmup = int(args.warmup_steps if args.warmup_steps is not None else cfg.scheduler_warmup_steps)
    optim = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        betas=tuple(cfg.optimizer_betas),
        eps=cfg.optimizer_eps,
        weight_decay=cfg.optimizer_weight_decay,
    )
    sched = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=warmup,
        num_decay_steps=decay_steps,
        peak_lr=args.lr,
        decay_lr=cfg.scheduler_decay_lr,
    ).build(optim, args.steps)
    clip_norm = float(args.grad_clip if args.grad_clip is not None else cfg.optimizer_grad_clip_norm)
    train_cfg = {
        "precision": args.precision,
        "trainable_dtypes": dtypes,
        "lr": args.lr,
        "betas": list(cfg.optimizer_betas),
        "eps": cfg.optimizer_eps,
        "weight_decay": cfg.optimizer_weight_decay,
        "grad_clip": clip_norm,
        "warmup_steps": warmup,
        "decay_steps": decay_steps,
        "decay_lr": cfg.scheduler_decay_lr,
        "steps": args.steps,
        "batch_size": args.batch_size,
        **frames_meta,
    }
    logger.info("optimizer/schedule: %s", train_cfg)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_type.txt").write_text("model")  # small tag file, read by tools/diag
    (out_dir / "train_config.json").write_text(json.dumps(train_cfg, indent=2) + "\n")
    metrics = MetricsLogger(out_dir)

    step = 0
    running = 0.0
    n_since_log = 0
    bs = max(1, int(args.batch_size))
    pbar = tqdm(total=args.steps, desc="train-smolvla")
    try:
        while step < args.steps:
            for batch in loader:
                t0 = time.perf_counter()
                batch = _to_device(batch, device)
                with amp():
                    loss, _ = policy.forward(batch)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                # Cap big jumps so one bad batch cannot wreck the weights.
                # Return value = grad norm before the clip (cheap signal).
                grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, clip_norm))
                lr_used = float(optim.param_groups[0]["lr"])  # the lr this step used
                optim.step()
                sched.step()
                step_s = time.perf_counter() - t0

                loss_f = float(loss.detach())
                running += loss_f
                n_since_log += 1
                step += 1
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss_f:.4f}", refresh=False)

                do_log = step % args.log_every == 0 or step >= args.steps
                do_eval = (
                    val_loader is not None
                    and args.eval_every > 0
                    and (step % args.eval_every == 0 or step >= args.steps)
                )
                if do_log or do_eval:
                    mean_train = running / max(1, n_since_log)
                    fields: dict = {
                        "train_loss": mean_train,
                        "train_loss_last": loss_f,
                        "lr": lr_used,
                        "grad_norm": grad_norm,
                        "step_s": step_s,
                        "samples_per_s": bs / max(step_s, 1e-9),
                    }
                    if do_eval:
                        with amp():
                            vloss = _eval_loss(policy, val_loader, device)
                        if vloss is not None:
                            fields["val_loss"] = vloss
                            logger.info(
                                "step=%d train_loss=%.4f val_loss=%.4f lr=%.2e "
                                "grad_norm=%.3f samples/s=%.1f",
                                step,
                                mean_train,
                                vloss,
                                fields["lr"],
                                grad_norm,
                                fields["samples_per_s"],
                            )
                        else:
                            logger.info(
                                "step=%d train_loss=%.4f (val empty) lr=%.2e",
                                step,
                                mean_train,
                                fields["lr"],
                            )
                    else:
                        logger.info(
                            "step=%d train_loss=%.4f lr=%.2e grad_norm=%.3f samples/s=%.1f",
                            step,
                            mean_train,
                            fields["lr"],
                            grad_norm,
                            fields["samples_per_s"],
                        )
                    metrics.log(step, **fields)
                    running = 0.0
                    n_since_log = 0

                if step % args.save_freq == 0 or step >= args.steps:
                    ckpt = out_dir / f"checkpoint_{step:06d}"
                    _save(ckpt)
                    logger.info("saved %s", ckpt)
                if step >= args.steps:
                    break
    except KeyboardInterrupt:
        logger.warning("interrupted at step=%d — writing metrics so far", step)
    finally:
        pbar.close()
        metrics.finalize()

    final = out_dir / "pretrained_model"
    _save(final)
    logger.info("final checkpoint %s", final)
    return final


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", type=str, required=True)
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    # None → fill from smolvla.yaml after parse (CLI wins when set).
    p.add_argument("--pretrained", type=str, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    # --log-freq kept as alias so old scripts still work.
    p.add_argument(
        "--log-every",
        "--log-freq",
        type=int,
        default=None,
        dest="log_every",
        help="write metrics.jsonl / log train loss every N steps",
    )
    p.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="held-out val loss every N steps (0 = never)",
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help="episode-level hold-out: every N-th episode (N≈1/fraction; 0 = no val)",
    )
    p.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="optional cap on episodes before train/val split (smoke runs)",
    )
    p.add_argument("--save-freq", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--policy-cfg",
        type=str,
        default=str(FORESIGHT_ROOT / "policy" / "configs" / "smolvla.yaml"),
        help="YAML for train knobs + Δ / noise (CLI flags override when set)",
    )
    p.add_argument("--delta-min", type=float, default=None)
    p.add_argument("--delta-max", type=float, default=None)
    p.add_argument("--pos-noise-std", type=float, default=None)
    p.add_argument("--vel-noise-std", type=float, default=None)
    p.add_argument(
        "--cond-frame",
        choices=["abs", "rel"],
        default="abs",
        help="rel: the model reads p̂ minus hand XY instead of world p̂",
    )
    p.add_argument(
        "--action-frame",
        choices=["abs", "delta"],
        default="abs",
        help="delta: the model predicts hand XYZ moves from the obs-frame hand",
    )
    p.add_argument(
        "--ee-noise-std",
        type=float,
        default=None,
        help="train-only noise (m) on the hand XYZ the model reads (default: yaml)",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="SmolVLA action chunk length (must match delta_timestamps)",
    )
    p.add_argument(
        "--precision",
        choices=["fp32-master", "legacy-bf16"],
        default="fp32-master",
        help="fp32-master: trainable weights in fp32, forward under bf16 autocast. "
        "legacy-bf16: the old behaviour (bf16 weights, small updates lost) — for A/B only.",
    )
    p.add_argument("--warmup-steps", type=int, default=None, help="default: SmolVLA preset (1000)")
    p.add_argument(
        "--scheduler-decay-steps",
        type=int,
        default=None,
        help="cosine length; default min(SmolVLA preset 30000, --steps)",
    )
    p.add_argument("--grad-clip", type=float, default=None, help="default: SmolVLA preset (10)")
    args = p.parse_args()
    from interfaces.config import policy_aug_defaults, policy_train_defaults

    # Empty CLI flags are filled from smolvla.yaml (steps, batch, lr, val, …).
    train_defs = policy_train_defaults(args.policy_cfg)
    for key in (
        "pretrained",
        "steps",
        "batch_size",
        "lr",
        "log_every",
        "eval_every",
        "val_fraction",
    ):
        if getattr(args, key) is None:
            setattr(args, key, train_defs[key])

    # Any Δ / noise flag left empty is filled from the yaml (+ sim Δ).
    aug = policy_aug_defaults(args.policy_cfg)
    if args.delta_min is None:
        args.delta_min = aug["delta_min"]
    if args.delta_max is None:
        args.delta_max = aug["delta_max"]
    if args.pos_noise_std is None:
        args.pos_noise_std = aug["pos_noise_std"]
    if args.vel_noise_std is None:
        args.vel_noise_std = aug["vel_noise_std"]
    if args.ee_noise_std is None:
        args.ee_noise_std = aug["ee_noise_std"]
    # Δ is not one of the packed numbers yet. A spread of Δ would blur the
    # position the model sees and it could not know which Δ was used.
    if abs(float(args.delta_min) - float(args.delta_max)) > 1e-12:
        raise SystemExit(
            "delta_min must equal delta_max while Δ is not packed "
            f"(got {args.delta_min} / {args.delta_max})"
        )
    logger.info(
        "train from %s: steps=%d batch=%d lr=%.2e log_every=%d eval_every=%d "
        "val_fraction=%.3f pretrained=%s",
        args.policy_cfg,
        args.steps,
        args.batch_size,
        args.lr,
        args.log_every,
        args.eval_every,
        args.val_fraction,
        args.pretrained,
    )
    logger.info(
        "aug from %s: delta=%.3f pos_noise=%.4f vel_noise=%.4f ee_noise=%.4f",
        args.policy_cfg,
        args.delta_min,
        args.pos_noise_std,
        args.vel_noise_std,
        args.ee_noise_std,
    )
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()

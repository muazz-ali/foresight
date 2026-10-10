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

Conda: dynamicVLA_training
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
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
):
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    from policy.transforms import expand_state_stats, policy_features_for_model

    # Tell the model its state is 12 numbers wide (8 robot + 4 object),
    # and give it matching scaling stats. Skip this and shapes will not fit.
    features = policy_features_for_model(ds_meta.features)
    stats = expand_state_stats(ds_meta.stats)

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


def train(args: argparse.Namespace) -> Path:
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

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
    base = LeRobotDataset(
        repo_id=args.repo_id,
        root=root,
        delta_timestamps=delta_timestamps,
    )

    # Look-ahead Δ + noise sizes for the 4 object numbers. Δ must match what
    # the sim logged and what eval will use. Noise is for robustness only.
    aug = AugConfig(
        delta_min=args.delta_min,
        delta_max=args.delta_max,
        pos_noise_std=args.pos_noise_std,
        vel_noise_std=args.vel_noise_std,
        seed=args.seed,
    )
    # Wrapper builds the 12-number state (with fresh noise) on every read.
    ds = PolicyDataset(base, aug=aug, apply_aug=True)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_collate,
        drop_last=True,
    )

    policy, cfg = build_policy(
        base.meta,
        device=device,
        pretrained=args.pretrained,
        chunk_size=args.chunk_size,
    )
    # Only train weights that are not frozen.
    optim = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-10,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_type.txt").write_text("model")  # small tag file, read by tools/diag

    step = 0
    running = 0.0
    pbar = tqdm(total=args.steps, desc="train-smolvla")
    while step < args.steps:
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            loss, _ = policy.forward(batch)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            # Cap big jumps so one bad batch cannot wreck the weights.
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optim.step()
            running += float(loss.detach())
            step += 1
            pbar.update(1)
            if step % args.log_freq == 0:
                avg = running / args.log_freq
                logger.info("step=%d loss=%.4f", step, avg)
                running = 0.0
            if step % args.save_freq == 0 or step >= args.steps:
                ckpt = out_dir / f"checkpoint_{step:06d}"
                policy.save_pretrained(ckpt)
                logger.info("saved %s", ckpt)
            if step >= args.steps:
                break
    pbar.close()

    final = out_dir / "pretrained_model"
    policy.save_pretrained(final)
    logger.info("final checkpoint %s", final)
    return final


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", type=str, required=True)
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--pretrained", type=str, default="lerobot/smolvla_base")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-freq", type=int, default=50)
    p.add_argument("--save-freq", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--policy-cfg",
        type=str,
        default=str(FORESIGHT_ROOT / "policy" / "configs" / "smolvla.yaml"),
        help="YAML for Δ / noise defaults (CLI flags override when set)",
    )
    p.add_argument("--delta-min", type=float, default=None)
    p.add_argument("--delta-max", type=float, default=None)
    p.add_argument("--pos-noise-std", type=float, default=None)
    p.add_argument("--vel-noise-std", type=float, default=None)
    p.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="SmolVLA action chunk length (must match delta_timestamps)",
    )
    args = p.parse_args()
    from interfaces.config import policy_aug_defaults

    # Any Δ / noise flag left empty is filled from the yaml.
    aug = policy_aug_defaults(args.policy_cfg)
    if args.delta_min is None:
        args.delta_min = aug["delta_min"]
    if args.delta_max is None:
        args.delta_max = aug["delta_max"]
    if args.pos_noise_std is None:
        args.pos_noise_std = aug["pos_noise_std"]
    if args.vel_noise_std is None:
        args.vel_noise_std = aug["vel_noise_std"]
    # Δ is not one of the packed numbers yet. A spread of Δ would blur the
    # position the model sees and it could not know which Δ was used.
    if abs(float(args.delta_min) - float(args.delta_max)) > 1e-12:
        raise SystemExit(
            "delta_min must equal delta_max while Δ is not packed "
            f"(got {args.delta_min} / {args.delta_max})"
        )
    logger.info(
        "aug from %s: delta=%.3f pos_noise=%.4f vel_noise=%.4f",
        args.policy_cfg,
        args.delta_min,
        args.pos_noise_std,
        args.vel_noise_std,
    )
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()

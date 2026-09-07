#!/usr/bin/env python3
"""Train SmolVLA Model A or B on a Foresight LeRobot dataset.

Model A = images + robot pose + text.
Model B = A + 4 live future numbers (XY p̂/v̂; random look-ahead + noise).

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
    model: str,
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

    features = policy_features_for_model(ds_meta.features, model)
    stats = expand_state_stats(ds_meta.stats, model)

    policy_features = dataset_to_policy_features(features)
    cfg = SmolVLAConfig()
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
    # DynamicVLA recipe: load full SmolVLA checkpoint (VLM already inside).
    cfg.freeze_vision_encoder = True
    cfg.train_expert_only = True
    cfg.load_vlm_weights = False

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
    meta_ds = LeRobotDataset(repo_id=args.repo_id, root=root, episodes=[0])
    delta_timestamps = resolve_delta_timestamps(probe_cfg, meta_ds.meta)
    base = LeRobotDataset(
        repo_id=args.repo_id,
        root=root,
        delta_timestamps=delta_timestamps,
    )

    aug = AugConfig(
        delta_min=args.delta_min,
        delta_max=args.delta_max,
        pos_noise_std=args.pos_noise_std,
        vel_noise_std=args.vel_noise_std,
        seed=args.seed,
    )
    ds = PolicyDataset(base, model=args.model, aug=aug, apply_aug=True)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=_collate,
        drop_last=True,
    )

    policy, cfg = build_policy(
        args.model,
        base.meta,
        device=device,
        pretrained=args.pretrained,
        chunk_size=args.chunk_size,
    )
    optim = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-10,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_type.txt").write_text(args.model)

    step = 0
    running = 0.0
    pbar = tqdm(total=args.steps, desc=f"train-{args.model}")
    while step < args.steps:
        for batch in loader:
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            loss, _ = policy.forward(batch)
            optim.zero_grad(set_to_none=True)
            loss.backward()
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
    p.add_argument("--model", choices=["A", "B"], required=True)
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
    p.add_argument("--delta-min", type=float, default=0.0)
    p.add_argument("--delta-max", type=float, default=0.40)
    p.add_argument("--pos-noise-std", type=float, default=0.015)
    p.add_argument("--vel-noise-std", type=float, default=0.03)
    p.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="SmolVLA action chunk length (must match delta_timestamps)",
    )
    args = p.parse_args()
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()

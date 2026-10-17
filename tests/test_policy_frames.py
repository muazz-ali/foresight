"""Rel cond / delta action / hand noise: train-side wrapper and eval-side helpers agree."""

import numpy as np
import pytest
import torch

from policy.transforms import (
    AugConfig,
    PolicyDataset,
    action_from_model,
    cond_to_model,
)

EE = np.array([0.48, 0.05, 0.10])
COND = np.array([0.52, 0.02, 0.10, -0.05])  # p̂x, p̂y, v̂x, v̂y (world)


class _Base:
    """One frame: hand at EE, 3-step action chunk, saved clean conditioning."""

    def __len__(self):
        return 1

    def __getitem__(self, i):
        state = torch.tensor([*EE, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        act = torch.tensor([[0.50, 0.03, 0.05, 0, 1, 0, 0, 1]] * 3, dtype=torch.float32)
        return {
            "observation.state": state,
            "action": act,
            "observation.conditioning": torch.tensor(COND, dtype=torch.float32),
        }


def _item(**kw):
    cfg = AugConfig(pos_noise_std=0.0, vel_noise_std=0.0, **kw)
    return PolicyDataset(_Base(), aug=cfg, apply_aug=True)[0]


def test_abs_frames_unchanged():
    it = _item()
    np.testing.assert_allclose(it["observation.state"][8:].numpy(), COND, atol=1e-6)
    np.testing.assert_allclose(it["action"][0, :3].numpy(), [0.50, 0.03, 0.05], atol=1e-6)


def test_rel_cond_is_object_minus_hand_xy():
    it = _item(cond_frame="rel")
    want = cond_to_model(COND, EE, "rel")
    np.testing.assert_allclose(it["observation.state"][8:].numpy(), want, atol=1e-6)
    np.testing.assert_allclose(want[:2], COND[:2] - EE[:2])
    np.testing.assert_allclose(want[2:], COND[2:])  # velocity stays world


def test_delta_action_round_trips_through_eval_helper():
    it = _item(action_frame="delta")
    delta = it["action"].numpy()
    np.testing.assert_allclose(delta[:, :3], [[0.02, -0.02, -0.05]] * 3, atol=1e-6)
    back = action_from_model(delta, EE, "delta")
    np.testing.assert_allclose(back[:, :3], [[0.50, 0.03, 0.05]] * 3, atol=1e-6)
    np.testing.assert_allclose(back[:, 3:], delta[:, 3:])  # rotation / gripper untouched


def test_hand_noise_only_moves_the_read_hand():
    it = _item(cond_frame="rel", action_frame="delta", ee_noise_std=0.03)
    seen = it["observation.state"][:3].numpy()
    assert np.abs(seen - EE).max() > 1e-4  # model reads a shaken hand
    # Rel cond and delta action still use the clean hand.
    np.testing.assert_allclose(it["observation.state"][8:10].numpy(), COND[:2] - EE[:2], atol=1e-6)
    np.testing.assert_allclose(it["action"][0, :3].numpy(), [0.02, -0.02, -0.05], atol=1e-6)


def test_zero_conditioning_stays_zero_in_rel_frame():
    it = _item(cond_frame="rel", zero_conditioning=True)
    assert torch.count_nonzero(it["observation.state"][8:]) == 0


def test_bad_frame_names_raise():
    with pytest.raises(ValueError):
        cond_to_model(COND, EE, "world")
    with pytest.raises(ValueError):
        action_from_model(np.zeros((2, 8)), EE, "relative")

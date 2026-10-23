"""Isaac-free: eval loop plan queue, latency, buffer, seeding, close time.

A fake scene + fake policy stand in for Isaac and SmolVLA. Each fake chunk
action carries (plan number, slot index) in its gripper-free slots so we can
see exactly which action was played at each step.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from interfaces.state import oracle_from_gt

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("eval_policy", ROOT / "scripts" / "eval_policy.py")
ep = importlib.util.module_from_spec(_spec)
sys.modules["eval_policy"] = ep
_spec.loader.exec_module(ep)

DT = 0.04


class FakeScene:
    def __init__(self, substep=False, cfg_extra=None):
        self.cfg = {"state_machine_params": {"inference_window_s": 1.0},
                    "robot": dict(cfg_extra or {})}
        self.dt, self.weld_grasp, self.t = DT, False, 0.0
        self.targets, self.substep_calls, self.releases = [], 0, 0
        self.obj_pos = [0.6, 0.0, 0.03]
        if substep:  # mimic the planned scene.step(..., substep_cb=None)
            self.step = self._step_cb

    def reset_episode(self, *, seed, speed):
        self.t = 0.0
        return {"category": "apple", "container_category": "bowl", "carry_pose": [0.5, 0.3, 0.3]}

    def get_images(self):
        return {"static_cam": np.zeros((4, 4, 3), np.uint8), "wrist_cam": np.zeros((4, 4, 3), np.uint8)}

    def get_object_state(self):
        return oracle_from_gt(self.obj_pos, [0.0, 0.0, 0.0], timestamp=self.t)

    def release_object(self):
        self.releases += 1
        return True

    def get_ee_pose(self):
        return np.array([0.45, 0.0, 0.40]), np.array([0.0, 1.0, 0.0, 0.0])

    def get_gripper_open(self):
        return 1.0

    def get_finger_gap_m(self):
        return 0.08  # fully open

    def set_ee_target(self, pos, quat, grip):
        self.targets.append(np.asarray(pos, dtype=float).copy())

    def step(self, *, kinematic_object, attach_object=False, freeze_object=False):
        self.t += self.dt

    def _step_cb(self, *, kinematic_object, attach_object=False, freeze_object=False, substep_cb=None):
        if substep_cb is not None:
            self.substep_calls += 1
            substep_cb(self, self.t)
        self.t += self.dt

    def get_proprio(self):
        return np.zeros(8)

    def get_container_pose(self):
        return None

    def get_drop_target(self):
        return [0.5, 0.3, 0.1]

    def get_carry_pose(self):
        return [0.5, 0.3, 0.3]


class FakePolicy:
    """Chunk action i of plan k = [k, i, 0.4, quat(1,0,0,0), open]. Records noise."""

    def __init__(self, n_exec=4, chunk=10):
        self.config = SimpleNamespace(n_action_steps=n_exec, chunk_size=chunk, max_action_dim=32)
        self.k, self.noises = 0, []

    def reset(self):
        self.k = 0

    def predict_action_chunk(self, batch, noise=None):
        self.noises.append(noise.clone())
        n = self.config.chunk_size
        a = torch.zeros(1, n, 8)
        a[0, :, 0] = float(self.k)
        a[0, :, 1] = torch.arange(n, dtype=torch.float32)
        a[0, :, 2] = 0.4
        a[0, :, 3] = 1.0
        a[0, :, 7] = 1.0
        self.k += 1
        return a


def run(scene=None, policy=None, n_steps=12, **kw):
    scene = scene or FakeScene()
    policy = policy or FakePolicy()
    episode, meta = ep.run_episode_policy(
        scene, policy, seed=7, speed=0.0, delta=0.25, device="cpu", zero_conditioning=False,
        n_steps=n_steps, close_time=(0.2, "test"), **kw)
    played = np.asarray(episode["action_pos"])[:, :2]  # (plan k, slot i)
    return played, episode, meta, policy


def test_default_plays_chunks_in_order_like_select_action():
    played, ep_, meta, _ = run()
    want = [(k, i) for k in range(3) for i in range(4)]  # replan every n_action_steps
    assert [tuple(map(int, p)) for p in played] == want
    assert ep_["policy_replan"].tolist() == [1, 0, 0, 0] * 3
    assert meta["overlays"] == {
        "reflex": False, "reflex_rate": None, "action_buffer": False, "inject_latency_s": 0.0,
        "latency_steps": 0, "gripper_close_time_s": 0.2, "gripper_close_time_source": "test"}


def test_buffer_on_zero_latency_matches_default():
    a, *_ = run()
    b, *_ = run(action_buffer_on=True)
    np.testing.assert_array_equal(a, b)


def test_latency_buffer_off_plays_stale_chunk_from_start():
    played, *_ = run(inject_latency_s=2 * DT)
    assert [int(p[1]) for p in played[:4]] == [0, 1, 2, 3]


def test_latency_buffer_on_drops_past_slots():
    played, _, meta, _ = run(action_buffer_on=True, inject_latency_s=2 * DT)
    # Plan from an obs 2 steps old: slots 0,1 are already past; play 2,3 then replan.
    assert [tuple(map(int, p)) for p in played[:4]] == [(0, 2), (0, 3), (1, 2), (1, 3)]
    assert meta["overlays"]["latency_steps"] == 2


def test_latency_longer_than_plan_holds_last_command():
    played, _, meta, _ = run(action_buffer_on=True, inject_latency_s=5 * DT)
    assert meta["n_hold_steps"] == 12  # every slot stale: hold, never jump


def test_noise_is_seeded_per_episode():
    _, _, _, p1 = run()
    _, _, _, p2 = run()
    assert len(p1.noises) == 3
    for x, y in zip(p1.noises, p2.noises):
        torch.testing.assert_close(x, y)
    assert not torch.equal(p1.noises[0], p1.noises[1])


def test_reflex_uses_substep_hook_when_scene_has_it():
    scene = FakeScene(substep=True)
    _, _, meta, _ = run(scene=scene, reflex_on=True)
    assert meta["overlays"]["reflex_rate"] == "substep"
    assert scene.substep_calls == 12
    _, _, meta, _ = run(reflex_on=True)
    assert meta["overlays"]["reflex_rate"] == "control_step"


def test_gripper_close_time_from_config_or_default(caplog):
    assert ep.gripper_close_time({"robot": {"gripper_close_time_s": 0.25}}) == (
        0.25, "robot.gripper_close_time_s")
    from control.reflex import ReflexClose

    default = inspect.signature(ReflexClose).parameters["close_time"].default
    val, src = ep.gripper_close_time({"robot": {}})
    assert val == pytest.approx(default) and src == "control.reflex default"
    assert "gripper_close_time_s missing" in caplog.text


def test_parser_defaults_are_off():
    a = ep._build_parser().parse_args([])
    assert (a.reflex, a.action_buffer, a.inject_latency_s, a.torch_deterministic) == (
        "off", "off", 0.0, False)


def test_substep_reflex_close_releases_object_and_holds_grip():
    scene = FakeScene(substep=True)
    scene.obj_pos = [0.455, 0.0, 0.40]  # 5 mm from the hand: touch -> close
    _, ep_, meta, _ = run(scene=scene, reflex_on=True, n_steps=3)
    assert scene.releases >= 1
    assert meta["reflex_close_t"] == pytest.approx(0.0)
    assert ep_["gripper"].tolist() == [0.0, 0.0, 0.0]  # reflex owns the grip

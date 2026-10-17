# Part A — Shortcut-learning verification (code-backed)

**Conveyor / motion axis:** There is **no conveyor belt**. Objects are **thrown** in house mode (`sample_throw`) with planar **XY** velocity aimed toward the bowl. Collect `init_vel` has mean \|vy\| > \|vx\| (bowl-ward, often ±Y-ish). **Gripper fingers open along world Y.** Robot reach / catch band is often along **+X**. Axes are reported separately below.

Machine-readable twin: `foresight_verification/partials/part_a.json`.

---

## A1 — Model & loss

| Item | Value |
|------|--------|
| Class | `SmolVLAPolicy` (`lerobot.policies.smolvla`) |
| LeRobot | **0.3.3** |
| Config `type` | `smolvla` |
| VLM | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` |
| Start weights | `lerobot/smolvla_base` (`policy/configs/smolvla.yaml`) |
| Action head | **Flow matching**: \(x_t = t\cdot\varepsilon + (1-t)a\), target \(u_t=\varepsilon-a\) |
| Train loss | **MSE(\(u_t\), \(v_t\))** then mean (pad-masked) |
| Inference steps | **10** Euler steps, \(dt=-1/10\) |
| Noise seed (eval) | `torch.Generator.manual_seed(episode_seed)` then `torch.randn(..., generator=)` into `predict_action_chunk` |
| LeRobot default | `torch.normal(0,1)` **unseeded** if noise not passed |
| Freeze | vision frozen; `train_expert_only=true` |

**Sources:** `runs/p3_throw/pretrained_model/config.json`; `modeling_smolvla.py` ~840–865, 890–905, 676–684; `scripts/eval_policy.py` ~575–650.

---

## A2 — Chunking

| Item | Value |
|------|--------|
| `chunk_size` | **50** |
| `n_action_steps` (ckpt) | **50** → replan every **2000 ms** @ 25 Hz |
| `n_action_steps` (diverse eval) | **8** → replan every **320 ms** |
| Temporal ensembling | **None** |
| Index to controller | **chunk[0], chunk[1], …** in order (`plan.popleft`) |

**Sources:** checkpoint `config.json`; `scripts/eval_policy.py` ~651–671; `scripts/launch_eval_diverse.sh` `N_ACTION_STEPS=8`.

---

## A3 — Input layout (12-D state)

Order glued in `policy/transforms.py`:

| Idx | Name |
|-----|------|
| 0–2 | `ee_pos_x/y/z` |
| 3–6 | `ee_rot_qw/qx/qy/qz` |
| 7 | `gripper` |
| 8–9 | `p_hat_x/y` |
| 10–11 | `v_hat_x/y` |

Entry: images + language + `observation.state` (12). Aux `conditioning` / `oracle_state` dropped before the net.

**Dims with std < 1e−6** over all 465 293 frames of `data/lerobot/p3_throw` (parquet; no `meta/stats.json`):

- State / cond: **none**
- Action: **aqw, aqx, aqy, aqz** (constant down quat 0,1,0,0)

Std (cm / cm/s): ee_x 3.53, ee_y 16.56, ee_z 11.96; p̂_x 4.38, p̂_y 18.83; v̂_x 6.61, v̂_y 19.79.

---

## A4 — Normalization

- **MEAN_STD** for STATE & ACTION; **IDENTITY** for VISUAL.
- Formula: `(x − mean) / (std + 1e−8)`.
- **No clipping** after MEAN_STD.
- Train noise is applied in `PolicyDataset` **before** `normalize_inputs` inside `forward`.
- Checkpoint has **no** `foresight_policy.json` → **abs / abs** frames.

---

## A5 — Noise augmentation (critical)

| Channel | Train | Val / eval |
|---------|-------|------------|
| p̂ XY | \(\mathcal N(0, 1.5\,\mathrm{cm})\) per axis, independent | 0 |
| v̂ XY | \(\mathcal N(0, 3\,\mathrm{cm/s})\) per axis, independent | 0 |
| Hand XYZ | `ee_noise_std=0` (default) | 0 |
| Action | none | — |
| Images | none in foresight; SmolVLA only resize→512 + [0,1]→[−1,1] | — |

- Drawn **per frame** (`__getitem__`), not once per episode.
- Noise added **after** `conditioning_vector`: **p̂ is not recomputed from noisy v̂** (physically inconsistent on purpose).
- Δ fixed at **0.25 s** (`delta_min=delta_max`); not packed; not noised.

**Sources:** `policy/configs/smolvla.yaml` L19–22; `policy/transforms.py` L140–144; `policy/train_smolvla.py` val aug zeros.

---

## A6 — Label construction (critical)

**Commanded absolute EE target**, not future measured hand pose.

1. Expert returns `ExpertCommand.position/quat`.
2. `sim/collect.py` logs `action_pos` / `action_quat` = that command.
3. `policy/convert_h5.pack_action` writes LeRobot `action` (“command, not measured state”).
4. Proprio uses **post-step** `ee_pos`; action uses the **command** from the same step.

---

## A7 — Expert SM (critical)

Stages **0→9**: start → intercept → *(2 unused)* → close → lift → carry → lower → open → home → done.

**Approach XY:** `aim = forecast(obj, vel, lead_s)` with  
`lead_s = min(t_travel, 0.25 s) + tip_lag 0.16 s`; **Z = object z** (no hover).

**Grasp gate:** \|dx\|≤**1 cm**, \|dy\|≤**1 cm**, \|dz\|≤**1.5 cm** (GT Z).

**Lift:** +**12 cm**. Hand speed cap **90 cm/s**. Conditioning Δ **0.25 s** (policy pack every frame; **not** used for SM aiming). Aim Δ only in stage 1/(2).

Hand-dependent: intercept travel from current EE, grasp gate vs EE−obj, slip if \|obj−ee\| > 10 cm, home retreat from EE anchor.

**Sources:** `sim/state_machine.py`; `sim/phase0_cfg.yaml` `state_machine_params`.

---

## A8 — Controller

- **DiffIK** (`dls`, absolute pose) → joint targets; **FRANKA_PANDA_HIGH_PD_CFG** joint PD.
- Arm stiffness/damping **400 / 80**; fingers **2000 / 100**.
- Control **dt = 40 ms (25 Hz)**; house **physics_substeps = 4** → **100 Hz** physics; render once per control step.

**Sources:** `sim/scene.py` L313–328, L614–617, L1426–1467; IsaacLab `franka.py` HIGH_PD; `phase0_cfg.yaml` `sim`.

---

## A9 — Eval protocol

| Item | Diverse run (`p3_throw_n100_diverse`) |
|------|--------------------------------------|
| Checkpoint | `runs/p3_throw/pretrained_model` |
| p̂ source | **filter** (not oracle) |
| Δ | **0.25 s** |
| Speeds | **0 / 10 / 15 / 20 cm/s** |
| n_action_steps | **8** |
| Reflex | **off** (logged) |
| Trials | 400 (100/speed) |
| Seeds | 10000–28209 |
| Objects | onion 120, avocado 120, egg 80, orange 40, apple 40 |
| Success | `z_max ≥ 12 cm` and (`z_end ≥ 10 cm` or XY to carry < 12 cm) |
| Logs | `data/eval/p3_throw_n100_diverse/`, also `p3_throw_n100/`, `p3_throw_oracle_n20/` |

Note: argparse **default** `--cond-source` is `filter` (help text wrongly says oracle).

---

## A10 — Gripper & object sizes

- Open: **4 cm per finger ≈ 8 cm total**, along **world Y**; TCP offset z **10.7 cm**; max object width gate **7.5 cm**.
- Object **half-height** (collect JSON), cm — apple ~3.6, avocado ~3.8, can ~5.0, egg ~2.8, kiwi ~2.9, lemon ~3.1, lime ~3.4, onion ~3.0, orange ~3.5, peach ~3.1, tangerine ~2.9.
- **UNKNOWN:** full AABB size **along vs across** throw direction (not stored; would need `usd_bbox_size`).

---

## A11 — Dataset & checkpoint

| Item | Value |
|------|--------|
| Root | `data/lerobot/p3_throw` |
| Episodes / frames | **4117 / 465293** @ 25 Hz |
| Checkpoint | `runs/p3_throw/pretrained_model` = `checkpoint_037500`, step **37500** |

Per-object episode counts (from task text): apple 315, avocado 393, can 300, egg 409, kiwi 364, lemon 420, lime 368, onion 407, orange 380, peach 370, tangerine 391.

**Object × speed:** see JSON (`object_x_speed_cm_s`). Method = median \|v̂\| over first 10 frames → round to cm/s.  
**UNKNOWN exact commanded `speed_m_s` crosstab:** no `meta/source_h5.jsonl` on this dataset (only on `p3_throw_tail`). Clean collect pool is 5568 eps; train set is a 4117 subset.

# Foresight

Teach a small robot policy to catch **moving** objects by telling it **where the object will be**, not only what the cameras show right now.

**Idea in one line:** predict the object’s future place with a simple physics tracker → give those numbers to a small policy → close the gripper at the right time.

Full write-up: [`foresight_plan.md`](foresight_plan.md) · How we work: [`AGENTS.md`](AGENTS.md) · Isaac Sim tips: `~/Desktop/muazzam/ppt.md` (outside this repo)

---

## Status

| Checkpoint | What “pass” means | Status |
|---|---|---|
| **G0** | Scripted expert works ≥70% from still → 20 cm/s (bins 0 / 10 / 15 / 20); we can record hundreds of demos per hour | **PASS** on the older 0–40 cm/s sweep (report file not in this checkout) |
| **G1** | Policy **with** future numbers (B) beats policy **without** (A) at 15 cm/s; B stays flat 0 → 20 cm/s; B with numbers zeroed collapses | **PASS** (Sep 13 2026). B 84 / 70 / 71% at 0 / 15 / 20 cm/s; A 9% still, 0% moving; B-zero 2%. n = 100 per B bin. Details: plan §7 |
| G2a | Camera + filter numbers close to sim truth: at 20 cm/s, median ≤ 1 cm and RMSE ≤ 2 cm vs the sim-truth p̂ | **PASS** (Sep 18 2026, offline). 0.34 / 1.14 cm on demos, 0.31 / 1.26 cm on B's eval runs. Report: `data/eval/p2_perception_report/REPORT.md` |
| G2b | Swap sim truth → camera numbers in B, same seeds: success drops < 10 points at 15 and 20 cm/s | next |
| G3 | Fast grasp helper fixes many “too late / early close” fails | not started |
| G4 | Real arm works well at 10–20 cm/s | not started |

“Success” before Phase 3 = the policy's grasp holds under the eval latch, then a script places the object (`--force-place`).

Lesson from G0: do **not** wrap DynamicVLA’s pick state machine. Our expert and scene live in this repo under `sim/`.

---

## Four jobs (about 1000 lines of custom code)

| Job | What it does | Where |
|---|---|---|
| Where / when | Track object + predict a bit ahead | `perception/` |
| What / how | Small policy + 4 future numbers today (~12 later) | `policy/` |
| Exactly when | Fast last-cm grasp close | `control/` (later) |
| Clock | Time-stamped action queue | `control/` (later) |

**Shared object message** (same for sim-truth and camera track; switch with a config flag):

```text
position, velocity, covariance, timestamp, valid
```

The policy’s future numbers are built from that message (`interfaces/state.py`). Today that is **4 numbers**: where the object will be in 0.25 s (x, y) and its speed (x, y). Δ = 0.25 s is fixed in `sim/phase0_cfg.yaml`.

---

## Folder map

```text
interfaces/   shared object state + future-number helper
sim/          Isaac table scene, scripted expert, recording
perception/   static camera → blob tracker → Kalman filter → object state (Phase 2)
policy/       turn demos into LeRobot data; train Model A vs B
scripts/      run collection, convert, train, eval, G2 perception report
eval/         checkpoint score helpers (G0 / G1)
tests/        Isaac-free tests (expert, perception)
data/         local only (git-ignored):
  p1-retrain/        Phase 1 demos (3,718 episodes, with camera frames)
  lerobot/p1-retrain LeRobot copy used for training
  eval/              eval runs + reports (p1_r4k_report, p2_perception_report)
runs/         local only: trained checkpoints (p1_r4k_a / p1_r4k_b = G1 models)
```

---

## Requirements

| Need | Notes |
|---|---|
| Isaac Sim **4.5** + Isaac Lab **2.2.x** | This machine: `~/Desktop/IsaacLab` |
| Sim conda env | e.g. `dynamicVLA_isaac` — **do not** train in this env |
| Train conda env | e.g. `dynamicVLA_training` (LeRobot / SmolVLA) |
| Extra Python (sim) | `h5py`, `imageio`, `opencv-python`, `scipy`, `pyyaml` |
| Meshes (optional) | USD objects under paths in `sim/phase0_cfg.yaml` |

```bash
conda activate dynamicVLA_isaac
python -c "import isaaclab; print('Isaac Lab OK')"
```

---

## Quick start (Gate G0)

```bash
cd /path/to/foresight
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

bash scripts/run_g0.sh smoke      # one still episode + video
bash scripts/run_g0.sh debug-mp4  # debug overlay video
bash scripts/run_g0.sh            # full speed sweep → data/g0/
```

Always pass `--headless --enable_cameras` for real RGB. Headless alone → black images.

No Isaac needed for:

```bash
python scripts/test_state.py
pytest tests/test_perception.py              # perception unit tests
python scripts/g2_perception_report.py       # G2a report from recorded episodes (~25 s)
```

---

## Cameras

| Name | Role |
|---|---|
| `static_cam` | Side / front view of the table |
| `wrist_cam` | On the hand, looking toward the fingers |

---

## Scripted expert (plan §4)

Stages: **go above the moving object** (look ~0.25 s ahead, hover ~10 cm) → **down, grasp, lift** → **place** → **reset**. If the object drops, go back to approach.

Success = lift + hold / place rules in `sim/success.py`.  
Failure labels each episode:  
`never-engaged | too-late | early-close | wrong-object | grasp-slip | off-table | safety-abort | success`

---

## Phase 1 — done (G1 pass)

Models A and B were trained on 3,718 recorded demos (`runs/p1_r4k_a|b`, commands in [`policy/README.md`](policy/README.md)). B (with the 4 numbers) beats A by 70 points at 15 cm/s and drops 13 points from still to 20 cm/s. Zeroing B's numbers drops it to 2%, so B really uses them. So far this shows “knowing where the object *is* helps”. Splitting that from “knowing where it *will be*” needs a model trained with Δ = 0 (plan §11).

---

## Phase 2 — now

Replace sim truth with what the static camera sees: blob tracker → pixel-to-table → Kalman filter → the same 4 numbers. B is **not** retrained.

- **Done — G2a (offline):** `perception/` + `python scripts/g2_perception_report.py`. At 20 cm/s the camera numbers sit 0.3 cm (median) from sim truth, and every fruit is tracked, including the white egg.
- **Next — G2b (Isaac):** add `--cond-source oracle|filter` to `scripts/eval_policy.py`, then rerun B on the G1 seeds and count how many runs flip from success to fail. Steps: plan §11.

---

## What we are **not** building yet

No big learned world model, no optical-flow network, no policy from scratch, no full 6D pose first (center + speed is enough), no 200K demos, no flying objects off the table before Phase 5, no DynamicVLA state machine as our expert.

---

## License / cite

Internal research. Cite Isaac Lab / Isaac Sim (NVIDIA) and LeRobot SmolVLA as appropriate.

# Foresight

Teach a small robot policy to catch **moving** objects by telling it **where the object will be**, not only what the cameras show right now.

**Idea in one line:** predict the object’s future place with a simple physics tracker → give those numbers to a small policy → close the gripper at the right time.

Full write-up: [`foresight_plan.md`](foresight_plan.md) · How we work: [`AGENTS.md`](AGENTS.md) · Isaac Sim tips: [`ppt.md`](ppt.md) · Word list: [`WORDS.md`](WORDS.md)

---

## Status

| Checkpoint | What “pass” means | Status |
|---|---|---|
| **G0** | Scripted expert works ≥70% from still → 20 cm/s (bins 0 / 10 / 15 / 20); we can record hundreds of demos per hour | **PASS** on the older 0–40 sweep — see `data/g0/gate_g0_report.json` |
| G1 | Policy **with** future numbers (B) beats policy **without** (A) at 15 cm/s; B stays flat 0 → 20 cm/s | not started (recollect/eval on the slower sweep) |
| G2 | Camera-based prediction error small; swap sim-truth → camera track without big drop | not started |
| G3 | Fast grasp helper fixes many “too late / early close” fails | not started |
| G4 | Real arm works well at 10–20 cm/s | not started |

Lesson from G0: do **not** wrap DynamicVLA’s pick state machine. Our expert and scene live in this repo under `sim/`.

---

## Four jobs (about 1000 lines of custom code)

| Job | What it does | Where |
|---|---|---|
| Where / when | Track object + predict a bit ahead | `perception/` (later) |
| What / how | Small policy + ~12 future numbers | `policy/` |
| Exactly when | Fast last-cm grasp close | `control/` (later) |
| Clock | Time-stamped action queue | `control/` (later) |

**Shared object message** (same for sim-truth and camera track; switch with a config flag):

```text
position, velocity, covariance, timestamp, valid
```

The policy’s ~12 future numbers are built from that message (`interfaces/state.py`).

---

## Folder map

```text
interfaces/   shared object state + future-number helper
sim/          Isaac table scene, scripted expert, recording
policy/       turn demos into LeRobot data; train Model A vs B
scripts/      run collection, convert, train, eval
eval/         checkpoint score helpers (G0 / G1)
data/g0/      Gate G0 scores
data/p1/      Phase 1 demos (~2K successes)
WORDS.md      plain-language glossary
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

## Phase 1 (next)

We already recorded **2000** successful demos in `data/p1/` and converted them to `data/lerobot/p1/`.

**Next:** train Model A (no future numbers) and Model B (with future numbers), then run the speed test. Commands: [`policy/README.md`](policy/README.md).

---

## What we are **not** building yet

No big learned world model, no optical-flow network, no policy from scratch, no full 6D pose first (center + speed is enough), no 200K demos, no flying objects off the table before Phase 5, no DynamicVLA state machine as our expert.

---

## License / cite

Internal research. Cite Isaac Lab / Isaac Sim (NVIDIA) and LeRobot SmolVLA as appropriate.

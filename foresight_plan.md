# Execution Plan — Explicit-Future Dynamic Manipulation
*Working name: **Foresight**. Hardware later: 1 static RGB camera, 1 wrist RGB-D, a real arm **still to be chosen** (the sim uses a Franka Panda, 7 joints). Simulator: Isaac Sim.*

**Plain words:** **oracle** = the object's true pose from the simulator. **The 4 numbers** = where we say the object will be in XY, and how fast it is moving in XY. **Δ** = how far ahead those numbers look (fixed at 0.25 s). **The model** = the small policy that sees images, the robot, the text, and those 4 numbers.

---

## 0. The idea

**Hypothesis.** A policy that is *explicitly told* where the object will be when its action lands — by a plain physics filter — will beat policies that must figure this out implicitly (DynamicVLA) or through a learned latent world model (AHEAD), while being smaller, cheaper to train, and easier to debug.

**System.** A tracker follows the object in the camera. A Kalman filter turns the track into position, velocity, and a confidence. It can answer "where will it be in Δ seconds?" Those 4 numbers go into the policy beside the robot state. The policy decides how to approach and grasp. A later fast reflex loop handles the last centimeters and closes the gripper on time. Every message has a timestamp so we never act on a stale frame.

**Why a short look-ahead is enough.** Error grows with speed × time, plus a little from surprise acceleration. With delay under about 300 ms and objects at or under 20 cm/s, the filter looks about 0.3 s ahead. On a table, that window is close to constant velocity, and the miss should stay inside a gripper opening.

**What proves the idea.** Two checks on the model only:
1. Grasp success stays nearly flat from still to 20 cm/s.
2. Zero the 4 numbers at eval. Success collapses. That shows the model is using the numbers.

**Where we are (Oct 6 2026).** G1, G2a and G2b passed on the old bounce-box scene. Phase 3 code is mostly in: throw scene without bounces, objects that fall off, real physics grasp at 100 Hz, measured close time, wrist depth, eval overlays, timing labels, G3 scorer, and a house-scene tracker. A full-stack smoke with .mp4 ran on Oct 6 (§5 item 6e). **Not ready to collect yet:** the reflex hurts the expert when switched on, the arm speed limit needs a retune, and several human decisions are open (real arm, camera mount, render mode, object slow-down, cups, too-wide objects).
**Next job is Phase 3 / Gate G3 — throw + timing in one.** One scene change, one expert, one sim demo set, one retrain. Collecting and training are expensive: do **not** split “throw demos” and “timing demos” into two collects. Details: §5.

---

## 1. Four jobs

**Job 1 — Where is it, and where will it be?** Tracker + Kalman filter (~200 lines). Find the object once, follow the blob, lift the pixel onto the table plane, let the filter carry position and velocity. Predict = step the physics forward with no new measurement.

**Job 2 — What should the arm do?** Small VLA (SmolVLA via LeRobot). Images, proprioception, language, plus the 4 numbers. Live pack is `p̂_x, p̂_y, v̂_x, v̂_y` at Δ = 0.25 s (`interfaces/state.py`). The policy handles which object, the approach, and the grasp. It does not have to be the motion predictor.

**Job 3 — Exactly when?** A fast reflex (100–500 Hz) in the last ~10 cm. Steer at the filter's meeting point. Close when time-to-arrival is shorter than the gripper's close time. No learning. Built in Phase 3 with the throw (same gate). It runs **only at eval / deployment**, as an overlay on the model. Demos are recorded with the reflex off, so the model learns its own close and "reflex off" stays a clean baseline. In sim it needs physics faster than the 25 Hz control rate (physics substeps), and one measured gripper close time.

**Job 4 — Never act on old information.** Timestamped action buffer. New chunk overwrites future slots. Slots whose time has passed are dropped. Same Phase 3.

---

## 2. The 4 numbers

Oracle and filter emit the same pack, built by the same helper.

| Field | Dim | Meaning | Now |
|---|---|---|---|
| p̂_x, p̂_y (t+Δ) | 2 | Predicted XY at action time, p̂ = p + v·Δ | packed |
| v̂_x, v̂_y | 2 | XY velocity | packed |
| Δ | 1 | Look-ahead | **fixed 0.25 s, not packed** |
| height, uncertainty, coast flag, intercept | — | extra | not packed. Add only if G3 fails and a retrain on filter-made vectors from the same demos is not enough |

The model's state is 8 robot numbers + these 4 = **12**.

Δ is the same at collect, train, and eval (`conditioning_delta_s` in `sim/phase0_cfg.yaml`). Pack Δ as a 5th number only when real delay starts to vary (Phase 3 if latency is swept, or Phase 4 real arm). Demos store raw sim truth and timestamps, so packing Δ is a **training** choice, never a reason to collect again. Decide it before the last sim retrain (§8 step 9): packing later changes the model's input size.

Train noise on the model: position σ = 1.5 cm, velocity σ = 3 cm/s (`policy/configs/smolvla.yaml`). That noise is an independent wobble. A real filter's error is smooth in time. G2b passed without a filter-vector retrain. Do not open a new collect for pack tweaks.

---

## 3. History — start, and the mistakes we are not repeating

**Start.** Sim scene, our own scripted expert, and logged oracle numbers, then a small policy trained on those numbers. Perception swaps in later, behind the same message: `position, velocity, covariance, timestamp, valid`.

**Borrowed expert (G0).** The first controller was DynamicVLA's pick state machine. Success sat near 40% at every speed. Their latch has holes we cannot patch from outside. The expert now lives in this repo (`sim/`): approach ~0.25 s ahead, hover ~10 cm, descend and grasp, place, reset. Drop returns to approach. Success is our own logged rule.

**Look-ahead sampled but not told to the policy.** The first training run drew Δ from 0 to 0.4 s and did not put Δ in the vector. That smeared the predicted position by up to 8 cm. We froze Δ at 0.25 s everywhere.

**Model A as the comparison (retired).** We once trained a no-numbers policy (Model A) and scored it against the conditioned model. Those are different pipelines, so their score gap does not test the 4 numbers. **That A-vs-B gate is retired.** The proof is zero-out on one checkpoint: with the 4 numbers set to zero, grasp fell to **2/100** while the full model was **84/100** on still objects. Do not train or eval Model A again. Do not put A back in a gate.

**Box and wall bounces.** The first scene slides one fruit around a 24 × 30 cm box. It hits the walls (about 0.4 / 0.7 / 0.9 hits per second at 10 / 15 / 20 cm/s). A bounce is a bad model of the task we want. G1 / G2a / G2b used that scene because the model was trained on it. Phase 3 leaves the bounce box: throw across the table, catch before it falls off. Wall flip must be **off** before the last sim collect.

**Hand hides the fruit.** In 2 of 300 oracle eval runs the gripper sat between the static camera and the fruit. The wrist camera is part of Phase 3 (same collect as the throw). Do not build wrist-only demos in a second pass.

**What "success" means on the bounce-box model (G1 / G2).** The eval latch welds the object when the fingers close near it and stay closed for 3 frames. Then a script carries and places (`--force-place`). The weld radius is 5 cm. Say **grasp success**. Policy labels are `never-glued | glued-no-lift | lifted-no-place | success`, plus `z_max` and glue frames. Do not run the expert failure tags on policy glue flags. Phase 3 replaces the weld with a real physics grasp when that gate starts.

---

## 4. Data we have now

One fruit, one bowl, grey table, no distractors, no lighting randomisation. Speeds in the bins 0 / 10 / 15 / 20 cm/s. Static RGB is 480×360. Pixel to table uses the ray–plane hit at the object's centre height (`object_half_height`), not the table top. In sim the camera has no frame lag. Expert demos and eval files already store both cameras, sim-truth pose, and timestamps. Perception code: `perception/` (~370 lines). Filter details and G2a tables: `data/eval/p2_perception_report/REPORT.md`.

The model was trained on 3,718 demos (`runs/p1_r4k_b`). Checkpoint for every G2 run: `runs/p1_r4k_b/pretrained_model`.

---

## 5. Gates

**G0 — our expert.** Target was ≥70% across 0–20 cm/s and collection at hundreds of episodes per hour. The borrowed state machine missed this. Our expert is what produced the demos above.

**G1 — oracle numbers into the model. PASS (Sep 13 2026).** Same settings G2b must copy: `--n-action-steps 8`, `--force-place`, Δ = 0.25 s. Seeds 5000–5099 (0 cm/s), 5150–5249 (15), 5200–5299 (20). Logs: `data/eval/scout_*_100runs*`, `data/eval/p1_r4k_report/`.

| Check | Result |
|---|---|
| Grasp rate (full numbers) | 84/100 still, 70/100 at 15 cm/s, 71/100 at 20 cm/s |
| Flat from still to 20 cm/s | 84 → 71 = 13 points. Passes a 15-point cap. The 95% range on that drop is about 2–24 points, so the margin is thin |
| Zero the 4 numbers | 2/100 still. The numbers are the channel the model uses |

The 10 cm/s bin was not run. With Δ fixed, the 4 numbers mix "where it is" and "where it will be." We have not trained a current-position-only pack. That split is not the next job.

**G2a — filter vs the numbers the model trained on. PASS (Sep 18 2026).** Offline, no Isaac. At 20 cm/s, filter p̂ vs sim-truth p̂: median / RMSE **0.34 / 1.14 cm** on 609 demos, **0.31 / 1.26 cm** on 100 model eval runs. Bars were median ≤ 1 cm and RMSE ≤ 2 cm. Every fruit is tracked, including the white egg (bright blob). Most of the remaining error is the first frame (filter starts at speed 0) and a few frames after each wall hit. Full table: `data/eval/p2_perception_report/REPORT.md`.

**G2b — closed-loop swap. PASS (Sep 29 2026).** Same model, checkpoint `runs/p1_r4k_b`, Δ = 0.25 s, replan 8, `--force-place`, seeds matching G1. Logs: `data/eval/g2b_n100/`, flips in `flips.json`.

| Speed | Oracle | Filter | Drop |
|---|---|---|---|
| 0 cm/s | 84/100 | 92/100 | −8 points (went up) |
| 15 cm/s | 70/100 | 65/100 | 5 points |
| 20 cm/s | 71/100 | 65/100 | 6 points |

Before the grasp, filter place vs sim-truth place: median about 0.23 / 0.29 / 0.32 cm at 0 / 15 / 20 cm/s.

Seeds that flipped success → fail: 4 at still (all `other`), 19 at 15 cm/s (all `other`, wall hits noted), 17 at 20 cm/s (15 `other`, 2 `hand hid the fruit`). No flip was "filter not warmed up" or "track lost".

The bar that was used:

Pass when all of this is true:
- Scene, checkpoint, Δ, replan (8 steps), `--force-place`, and **seeds** match the G1 oracle runs.
- n = 100 at 0, 15, and 20 cm/s.
- Grasp rate drops by **under 10 points** vs the paired oracle run at **15 cm/s and at 20 cm/s**.
- Each seed that flips from success to fail is listed with one cause.

Causes to log: filter not warmed up, hand hid the fruit, track lost, other. A wall hit may be a note on a flip. It is not a reason to build a bounce model for this gate.

If the drop had been 10 points or more: retrain on filter-made vectors from the *already recorded* demos (no new Isaac collect). That path was not needed — G2b passed.

**Phase 3 / Gate G3 — throw + timing (one phase, one collect).** Merged on purpose. Collecting demos and training the model is expensive. This is the **last sim dataset** and the **last sim retrain** before the real arm. Do not collect once for “throw” and again for “reflex / buffer / wrist.”

Build **all** of this *before* the collect starts. Status as of Oct 6 2026: ✅ done in code, 🔧 in progress, ❓ needs a human decision.

**A. Scene and expert (changes the demos)**
1. ✅ **Throw scene, no wall bounce.** House mode passes no walls to `kinematic_advance`, so a missed throw coasts straight on. Throw spawn is aimed from left / right / front at a catch band.
2. ✅ **Object can leave the table.** On a miss or a physics grasp the object is handed to PhysX (`_free_object`).
3. ✅ **Real physics grasp.** House mode: no weld by default (`use_weld_grasp` false); fingers close by motor, not by teleport.
4. ✅ **Physics faster than control.** House mode: 4 physics substeps per 25 Hz control step (100 Hz physics; `sim.physics_substeps: 4`), cameras render once per control step. 100 Hz makes finger contact physical (an orange held at 6.5 cm instead of fingers sinking 3.5 cm into it at 25 Hz) and costs ~6% time. Bounce-box mode is forced to 1 substep and is bit-identical to before. `scene.step(substep_cb=...)` calls back before each substep; the callback may move the hand, and must call `scene.release_object()` if it closes on the object mid-step.
5. ✅ **One measured gripper close time:** `robot.gripper_close_time_s: 0.08` (fingers stall on ~5 cm fruit in 0.07–0.08 s at 100 Hz; a free close to < 1 mm takes 0.30 s). Collect and eval both read it; the old hard-coded 0.30 s is gone. `close_gripper_s` is a dwell, now labelled as one. Match the real gripper once it is chosen.
6. ✅ **Expert re-checked** at 100 Hz, smoke with .mp4 (`data/p3_sim_agent/`): 19/20 vs 16/20 before (5 seeds × 0 / 10 / 15 / 20 cm/s). Grasps are now mostly first try (mean 112 frames per episode vs 174). Scout only.
6b. ✅ **Throw speed bug fixed.** Before the fix, physics-grasp mode coasted the object 20–60% slower than commanded (PhysX table friction pulled the moved body back), so speed bins were wrong. It now follows its own path: average speed exact (20.0 and 10.0 cm/s checked on 10 episodes). One seed still wobbles ±13% frame to frame around the right average. Any house data recorded before Oct 6 has wrong speed labels: do not use it for training.
6c. ❓ **Arm speed limit.** `ee_speed_max: 0.90` m/s is above the measured peak hand speed (0.55–0.58 m/s); the arm can't follow a 0.4 m/s ramp. Retune the expert's rendezvous solve before the collect.
6d. ❓ **Objects too wide for the gripper.** tomato03 (8.5 cm) and fcan05 (8.2 cm) are wider than the 8 cm opening, and cans ~7.6 cm wide can stay wedged when the fingers open (a release failure). Drop too-wide meshes from the catalog, or fix the expert's open stage.
6e. ✅ **Full-stack smoke, Oct 6** (`data/p3_smoke_full/`: one .mp4 per episode, plus `contact_sheet.png`). Expert with the reflex off (the demo setting), 6 new seeds × 0 / 10 / 15 / 20 cm/s: **17/24** (4/6, 5/6, 4/6, 4/6). Failures: 4 timeouts (incl. the 8.2 cm can fcan05 and a potato), 2 off-table, 1 too-late.
    - Throw speed is right on average (15.0 / 20.0 cm/s), with short spikes only when the hand touches the object.
    - Misses really fall: the object ends 5.7–7.8 cm below the table top.
    - Wrist depth is stored (float16, 0.04% NaN).
    - Raw .h5 with both cameras + depth is ~34 MB per 100 frames, of which depth is only ~2.6 MB.
    - About 166 episodes per hour per process with images, including Isaac start-up.
    - Scout only.
7. ✅ **Held-out objects out of the train list.** `unseen` (apple99, can99, cup99, dbottle99, peach99) is eval-only.
8. ❓ **Do thrown objects slow down?** Today they slide at constant speed, and upright bottles never tip. Real throws slow and roll. Decide before the collect: accept and say so, or add a slow-down.

**B. What gets recorded (anything missing here is missing forever)**
9. ✅ Both cameras, robot state, actions, raw sim truth + timestamps (so Δ can be rebuilt or packed at train time), wrist depth (`wrist_cam_depth`, `wrist_depth_m`).
10. 🔧 Finger width (not only the 0/1 command) and camera poses. Check before the collect.
11. ❓ **Disk** (`data/eval/p3_data_budget/REPORT.md`). About 280 GB free. Today's format costs ~110 MB per episode (raw .h5 + LeRobot PNGs + depth), so only ~2,500 episodes fit. To reach 4,000–5,000: convert the new data to LeRobot **video** (~53 MB/ep end to end), and delete `data/lerobot/p1-retrain/images` (93 GB, an exact duplicate of the PNGs inside the parquet files; training never reads it) after a frame-load test. Store depth at half resolution unless full depth is needed. Measured Oct 6: raw .h5 with depth is ~34 MB per 100 frames (depth gzips to ~2.6 MB, far below the 12–23 MB estimate), so the budget is better than that report's table.

**C. Match the real rig (decide before the collect)**
12. ❓ **Which real arm?** The sim is a Franka Panda (7 joints). If the real arm differs, sim images and motions transfer worse.
13. ❓ **Real static-camera mount and lens.** Sim static cam is at (1.0, 0, 0.6) m, 90° lens, 480×360. Match it, or accept that sim images only help as pretraining.

**D. Eval-time tools (do not change the demos; can keep improving after the freeze)**
14. ✅ `control/` buffer, chunk blend, reflex. Eval flags: `--reflex on|off`, `--action-buffer on|off`, `--inject-latency-s L` (all default off; defaults play exactly as before). In house mode the reflex runs before every 100 Hz physics substep (`substep_cb`) and calls `release_object()` when it closes. Every episode JSON records the flags and the close time used. Reflex is eval-only; collect keeps `use_reflex: false`.
14b. ❗ **The reflex hurts as tuned.** Same seeds, expert with the reflex on: **11/23** vs 17/24 off. At 20 cm/s it's 1/6 vs 4/6 (4 off-table, 1 too-late). The videos show the reflex pulling the arm into stretched, low poses. This doesn't block the collect (demos are reflex-off), but G3 needs a reflex that helps. Retune it (speed limit, radius, steer only when the meeting point is reachable) and smoke again before the G3 eval.
15. ✅ / ❓ **Seed the model per episode.** Done: each episode seeds python / numpy / torch, and every replan passes its own seeded noise to SmolVLA (`policy_seed` logged). Identical inputs now give identical actions (difference 0.0). But **the renderer is not repeatable**: the same reset in two processes gives camera images that differ by up to 18/255 on 17–34% of pixels. So same-seed outcomes still flip (2 of 20, and 1 of 10, at 15 cm/s). G3 stays **rate-based**. See item 15b.
15b. ❓ **Render settings** shape every demo image and are frozen by the collect. Measured Oct 6 (`data/p3_smoke_eval/first_frame/`):
    - The first frame after a reset is **not** stale: it matches frame 1, not the previous episode.
    - Two separate processes give nearly the same images (≤ 0.2% of pixels differ by > 10/255).
    - Resetting the same seed twice **inside one process** differs on 6–7% of pixels with the default anti-aliasing, and on 2–4% with **DLAA**.
    - Recommendation: DLAA for the collect and eval. Decide before collecting.
16. ✅ **Timing labels** (`eval/timing_labels.py`) on every physics-grasp eval episode: success / never-closed / off-table / aim / early / late / short-close, plus `z_max`, closest distances, close time vs closest-approach time, and pre-grasp path length and jerk.
17. ✅ **G3 scorer** `eval/gate_g3.py --baseline <reflex off> --treatment <reflex on> [--zero <dir>]`: success with Wilson intervals per speed, timing-failure rate (late + early + short-close), path length and jerk, success at 20 cm/s, zero-out collapse, and `claim_ready` (n ≥ 80 per bin).
18. 🔧 **Perception in the house scene.** The hist tracker is in (`data/eval/p3_perception_house/REPORT.md`; wooden tables fixed).
    - **Offline, fixed-speed smoke recordings** (`data/eval/p3_perception_house_fixed/`; n = 6 per speed, no cups drawn):
        - p̂ vs sim truth at 20 cm/s: **0.65 cm median** (passes the 1 cm bar) / **2.30 cm RMSE** (bar 2 cm).
        - RMSE at 10 / 15 cm/s: 1.44 / 1.71 cm. Track rate 97–100%.
        - Hardest objects: peach 3.1 cm RMSE, beer 2.6, can 2.3.
    - **Grey cups still fail** (54% tracked in the earlier run). ❓ Drop cups from the G3 set, or add a shape rung / the wrist camera.
    - **Closed loop**, old model in the throw scene (`data/p3_smoke_eval/`): before the object left the scene, filter p̂ was 1.2–5.8 cm (median) from truth, worse than offline. Re-check with the throw-trained model.

**E. Training choices (decide at the train, not the collect)**
19. Old bounce-box data as extra variety: usable if filtered. Index ready: `data/lerobot/p1-retrain-nobounce-index.json` (`scripts/select_box_episodes.py`). Episode level: 1,890 episodes (886 still + 1,004 moving with no wall hit). Frame level: 325,703 frames whose 2 s action chunk has no wall hit, but only 57.5% of loose-object moving frames qualify, so it adds less catch data than its size suggests. The LeRobot features already match the house data. Needed: `policy/train_smolvla.py` must take two dataset roots (old subset + new) instead of one. The old data used the weld and instant fingers, so pre-train on old + throw, then fine-tune on throw only.
20. ❓ Pack Δ or keep it fixed (§2).

Then, once: smoke the full stack on the throw task with .mp4 → **one** demo collection → **one** train → eval Gate G3.

**Gate G3 pass when all of this is true** (scout n=20; claim n≥80 per bin, Wilson intervals; the model is seeded per episode):

- Throw task (not bounce box): catch before off-table; miss → off-table fail is real.
- Filter numbers drive the model (same swap idea as G2b).
- Same model, same seeds, reflex off vs on: the **rate** of late + early-close failures at least halves. (A per-episode "recovered" count is not used: without seeding, about half of failures turn into successes on a plain rerun.)
- Paths get smoother (path length and jerk before the grasp); success at 20 cm/s goes up vs reflex off.
- Zero-out of the 4 numbers still collapses success (the numbers stay the channel).

**Hard rule — last sim collect.** After this dataset is frozen, no new Isaac demo collection for pack fields, bounce tweaks, or “one more timing pass.” Fix with code and the existing logs, or accept the number. The only later collect is **real-arm** data in Phase 4.

**Phase 4 — real arm.** Calibrate the static camera, the table plane, and the wrist camera. Measure latency by waving the gripper in front of the static camera. A few hundred **real** episodes with the filter driving the same expert. **Gate G4:** ≥60–70% on slide / roll / throw-catch tasks at 10–20 cm/s.

**Phase 5 — later.** Harder off-table catches with wrist depth, a small mix of motion models, and wait-when-unsure. Not before G3 passes. No new sim mega-collect just to enter Phase 5.

---

## 6. How to score

Speed bins: **0 / 10 / 15 / 20 cm/s**. Scout is 20 trials. A claim needs **n ≥ 80** per bin and Wilson 95% intervals. Compare **rates**. G1 / G2b ran with unseeded model noise, so their seed-by-seed flips are mostly chance. From Phase 3 on, the model is seeded per episode; same-seed reruns can then be compared episode by episode once that is shown to repeat.

On the bounce-box checkpoint (G1 / G2): success = grasp held by the eval latch, then scripted place.

On Phase 3 / G3: success = real physics grasp + catch before off-table. Log the timing labels (never-closed / aim / late / early / short-close / off-table), `z_max`, whether the object left the table, and the reflex / buffer / latency flags used.

---

## 7. Do not build

No second policy whose only role is "score minus a no-numbers baseline." No latent world model. No optical-flow net. No VLA from scratch. No 6D pose (centre + velocity first). No second sim collect after the Phase 3 dataset freezes. No bounce predictor and no wall-hitting scene in Phase 3. No “throw collect now, timing collect later.” No new pack fields that force another retrain unless G3 fails and the cheap filter-vector retrain on the *same* demos is not enough. No wrist camera / reflex / action clock work that assumes a second demo pass. No reflex inside the demos (it is an eval overlay). No collecting before items 12–13 (real arm, camera mount) are decided or explicitly waived.

---

## 8. Next steps (Phase 3 / G3)

G2b is done (Sep 29 2026). Do not rerun G2b unless the old checkpoint or seeds change.

Order — **code first, collect once, train once:**

1. ✅ Wall bounce off in house / throw mode; a miss leaves the table.
2. ✅ Throw spawn + expert catch at 100 Hz physics, speed bug fixed, smoke 19/20 with .mp4. ❓ Retune `ee_speed_max`; handle too-wide objects (items 6c, 6d).
3. ✅ `control/` buffer, chunk blend, reflex; eval flags; reflex at physics rate via the substep hook.
4. ✅ Real physics grasp in house mode. ✅ Gripper close time measured (0.08 s) and used in collect; 🔧 eval still to switch.
5. ✅ Model seeded per episode (actions repeat exactly). Renderer noise still flips ~10% of same-seed outcomes, so G3 uses rates. ❓ Render settings (item 15b).
6. ✅ Timing labels + `eval/gate_g3.py`.
7. 🔧 Perception on house frames: hist tracker in, close to the bar without cups, cups fail; re-score on fixed-sim frames. Box-scene G2a still passes.
8. ❓ Human decisions: real arm, static camera mount, object slow-down (items 8, 12, 13). Then check recording list and disk budget (items 10–11).
9. ✅ First full-stack smoke done Oct 6 (items 6e, 14b, 15b, 18). ❗ Retune the reflex, then smoke again (expert + eval pipeline) before collecting.
10. **One** sim collection (last sim dataset). Convert → **one** train (decide Δ packing and old-data mix first).
11. Gate G3 eval: scout then n≥80; rates + late/early rate drop + smoothness + zero-out; Wilson intervals.
12. Pass → Phase 4 (real arm). Fail → debug on existing demos / code; do not open a second sim collect unless a human explicitly overrides this rule.

# Execution Plan — Explicit-Future Dynamic Manipulation
*Working name: **Foresight**. Hardware later: 1 static RGB camera, 1 wrist RGB-D, a 6-DOF arm. Simulator: Isaac Sim.*

**Plain words:** **oracle** = the object's true pose from the simulator. **The 4 numbers** = where we say the object will be in XY, and how fast it is moving in XY. **Δ** = how far ahead those numbers look (fixed at 0.25 s). **Model B** = the small policy that sees images, the robot, the text, and those 4 numbers.

---

## 0. The idea

**Hypothesis.** A policy that is *explicitly told* where the object will be when its action lands — by a plain physics filter — will beat policies that must figure this out implicitly (DynamicVLA) or through a learned latent world model (AHEAD), while being smaller, cheaper to train, and easier to debug.

**System.** A tracker follows the object in the camera. A Kalman filter turns the track into position, velocity, and a confidence. It can answer "where will it be in Δ seconds?" Those 4 numbers go into the policy beside the robot state. The policy decides how to approach and grasp. A later fast reflex loop handles the last centimeters and closes the gripper on time. Every message has a timestamp so we never act on a stale frame.

**Why a short look-ahead is enough.** Error grows with speed × time, plus a little from surprise acceleration. With delay under about 300 ms and objects at or under 20 cm/s, the filter looks about 0.3 s ahead. On a table, that window is close to constant velocity, and the miss should stay inside a gripper opening.

**What proves the idea.** Two checks on Model B only:
1. Grasp success stays nearly flat from still to 20 cm/s.
2. Zero the 4 numbers at eval. Success collapses. That shows B is using the numbers.

**Where we are (Sep 27 2026).** G1 passed on oracle numbers. G2a passed (camera + filter vs sim truth, offline). 
**Next job is G2b:** close the loop and feed those filter numbers to the same Model B. Details: §7 and §11.

---

## 1. Four jobs

**Job 1 — Where is it, and where will it be?** Tracker + Kalman filter (~200 lines). Find the object once, follow the blob, lift the pixel onto the table plane, let the filter carry position and velocity. Predict = step the physics forward with no new measurement.

**Job 2 — What should the arm do?** Small VLA (SmolVLA via LeRobot). Images, proprioception, language, plus the 4 numbers. Live pack is `p̂_x, p̂_y, v̂_x, v̂_y` at Δ = 0.25 s (`interfaces/state.py`). The policy handles which object, the approach, and the grasp. It does not have to be the motion predictor.

**Job 3 — Exactly when?** A fast reflex (100–500 Hz) in the last ~10 cm. Steer at the filter's meeting point. Close when time-to-arrival is shorter than the gripper's close time. No learning. This is Phase 3.

**Job 4 — Never act on old information.** Timestamped action buffer. New chunk overwrites future slots. Slots whose time has passed are dropped. Phase 3.

---

## 2. The 4 numbers

Oracle and filter emit the same pack, built by the same helper.

| Field | Dim | Meaning | Now |
|---|---|---|---|
| p̂_x, p̂_y (t+Δ) | 2 | Predicted XY at action time, p̂ = p + v·Δ | packed |
| v̂_x, v̂_y | 2 | XY velocity | packed |
| Δ | 1 | Look-ahead | **fixed 0.25 s, not packed** |
| height, uncertainty, coast flag, intercept | — | extra | add only after a G2b retrain if the swap fails |

Model B's state is 8 robot numbers + these 4 = **12**.

Δ is the same at collect, train, and eval (`conditioning_delta_s` in `sim/phase0_cfg.yaml`). Pack Δ as a 5th number only when real delay starts to vary (Phase 3 latency sweep, Phase 4 real arm).

Train noise on B: position σ = 1.5 cm, velocity σ = 3 cm/s (`policy/configs/smolvla_b.yaml`). That noise is an independent wobble. A real filter's error is smooth in time. If the G2b drop is large, retrain B on vectors the filter made from the saved demos. No new collection for that retrain.

---

## 3. History — start, and the mistakes we are not repeating

**Start.** Sim scene, our own scripted expert, and logged oracle numbers, then a small policy trained on those numbers. Perception swaps in later, behind the same message: `position, velocity, covariance, timestamp, valid`.

**Borrowed expert (G0).** The first controller was DynamicVLA's pick state machine. Success sat near 40% at every speed. Their latch has holes we cannot patch from outside. The expert now lives in this repo (`sim/`): approach ~0.25 s ahead, hover ~10 cm, descend and grasp, place, reset. Drop returns to approach. Success is our own logged rule.

**Look-ahead sampled but not told to the policy.** The first training run drew Δ from 0 to 0.4 s and did not put Δ in the vector. That smeared the predicted position by up to 8 cm. We froze Δ at 0.25 s everywhere.

**Model A as the comparison.** We also trained Model A (images, robot, text, no extra numbers) and scored B against A. A and B are different pipelines, so B minus A does not test the 4 numbers. **That comparison is retired.** The proof is inside B: the same checkpoint with the 4 numbers set to zero fell to **2/100** while full B was **84/100** on still objects. Do not train or eval Model A again. Do not put A back in a gate.

**Box and wall bounces.** The first scene slides one fruit around a 24 × 30 cm box. It hits the walls (about 0.4 / 0.7 / 0.9 hits per second at 10 / 15 / 20 cm/s). A bounce is a bad model of the task we want. We keep this scene only for the initial tests already in flight (G1, G2a, G2b), because that is the motion B was trained on. After G2b, the scene changes (§7, after G2b).

**Hand hides the fruit.** In 2 of 300 oracle eval runs the gripper sat between the static camera and the fruit. The fix is a wrist camera. That is Phase 3. Count it if it flips a G2b seed. Do not build the wrist camera to pass G2b.

**What "success" means until Phase 3.** The eval latch welds the object when the fingers close near it and stay closed for 3 frames. Then a script carries and places (`--force-place`). The weld radius is 5 cm. Say **grasp success**. Policy labels are `never-glued | glued-no-lift | lifted-no-place | success`, plus `z_max` and glue frames. Do not run the expert failure tags on policy glue flags.

---

## 4. Data we have now

One fruit, one bowl, grey table, no distractors, no lighting randomisation. Speeds in the bins 0 / 10 / 15 / 20 cm/s. Static RGB is 480×360. Pixel to table uses the ray–plane hit at the object's centre height (`object_half_height`), not the table top. In sim the camera has no frame lag. Expert demos and eval files already store both cameras, sim-truth pose, and timestamps. Perception code: `perception/` (~370 lines). Filter details and G2a tables: `data/eval/p2_perception_report/REPORT.md`.

B was trained on 3,718 demos (`runs/p1_r4k_b`). Checkpoint for every G2 run: `runs/p1_r4k_b/pretrained_model`.

---

## 5. Gates

**G0 — our expert.** Target was ≥70% across 0–20 cm/s and collection at hundreds of episodes per hour. The borrowed state machine missed this. Our expert is what produced the demos above.

**G1 — oracle numbers into B. PASS (Sep 13 2026).** Same settings G2b must copy: `--n-action-steps 8`, `--force-place`, Δ = 0.25 s. Seeds 5000–5099 (0 cm/s), 5150–5249 (15), 5200–5299 (20). Logs: `data/eval/scout_*_100runs*`, `data/eval/p1_r4k_report/`.

| Check | Result |
|---|---|
| B grasp rate | 84/100 still, 70/100 at 15 cm/s, 71/100 at 20 cm/s |
| Flat from still to 20 cm/s | 84 → 71 = 13 points. Passes a 15-point cap. The 95% range on that drop is about 2–24 points, so the margin is thin |
| Zero the 4 numbers | 2/100 still. The numbers are the channel B uses |

The 10 cm/s bin was not run. With Δ fixed, the 4 numbers mix "where it is" and "where it will be." We have not trained a current-position-only pack. That split is not the next job.

**G2a — filter vs the numbers B trained on. PASS (Sep 18 2026).** Offline, no Isaac. At 20 cm/s, filter p̂ vs sim-truth p̂: median / RMSE **0.34 / 1.14 cm** on 609 demos, **0.31 / 1.26 cm** on 100 B eval runs. Bars were median ≤ 1 cm and RMSE ≤ 2 cm. Every fruit is tracked, including the white egg (bright blob). Most of the remaining error is the first frame (filter starts at speed 0) and a few frames after each wall hit. Full table: `data/eval/p2_perception_report/REPORT.md`.

**G2b — the next job. Closed-loop swap.**

Goal: the same Model B, with the 4 numbers coming from the camera and the filter instead of the simulator.

Pass when all of this is true:
- Scene, checkpoint, Δ, replan (8 steps), `--force-place`, and **seeds** match the G1 oracle runs.
- n = 100 at 0, 15, and 20 cm/s.
- Grasp rate drops by **under 10 points** vs the paired oracle run at **15 cm/s and at 20 cm/s**.
- Each seed that flips from success to fail is listed with one cause.

Causes to log: filter not warmed up, hand hid the fruit, track lost, other. A wall hit may be a note on a flip. It is not a reason to build a bounce model for this gate.

If the drop is 10 points or more: retrain B on filter-made vectors from the recorded demos, then rerun the same seeds. If that retrain still misses, only then consider packing uncertainty or a coast flag (another retrain).

**After G2b — the throw.** Leave the box. Throw the object from a **random direction**. The policy has to **catch it before it falls off the table**. Bouncing off a wall is not the task. This needs new demos and a new train of B. It starts only after the G2b number is in.

**Phase 3 — timing, and the wrist camera.** Add the timestamped executor, chunk blend, and the reflex. The wrist camera is the eye for when the hand hides the object from the static camera, and for the last centimeters of the grasp. **Gate G3:** recover at least half of the late / early-close failures; paths get smoother; success at 20 cm/s goes up. Replace the weld latch with a real physics grasp when this phase starts.

**Phase 4 — real arm.** Calibrate the static camera, the table plane, and the wrist camera. Measure latency by waving the gripper in front of the static camera. Collect a few hundred real episodes with the filter driving the same expert. **Gate G4:** ≥60–70% on slide / roll tasks at 10–20 cm/s.

**Phase 5 — later.** Off-table catches with wrist depth, a small mix of motion models, and behavior that waits when the filter is unsure. Not before the throw task and Phase 3.

---

## 6. How to score

Speed bins: **0 / 10 / 15 / 20 cm/s**. Scout is 20 trials. A claim needs **n ≥ 80** per bin and Wilson 95% intervals. For oracle vs filter, rerun the **same seeds** and count flips.

Until Phase 3, success = grasp held by the eval latch, then scripted place. Label it that way.

---

## 7. Do not build

No second policy whose only role is "B minus that policy." No latent world model. No optical-flow net. No VLA from scratch. No 6D pose (centre + velocity first). No huge episode counts until the throw task asks for new demos. No bounce predictor and no wall-hitting scene after G2b. No wrist camera, reflex, or action clock inside G2b. No new pack fields before the G2b number, and then only if the filter-vector retrain still fails.

---

## 8. Next steps (G2b only)

1. Add `scripts/eval_policy.py --cond-source oracle|filter`. Default stays oracle.
2. Build the filter with `StaticCamEstimator.from_cfg(scene.cfg)`. Feed it the same static frame B sees, at that frame's sim time.
3. Warm the filter on **at least 2 frames** before B's first call. Otherwise the first 8-step chunk is planned from a guess that is off by speed × 0.25 s.
4. Log filter pack and sim-truth pack side by side, plus valid and redetects. Write checkpoint, replan steps, force-place, cond source, and Δ into every episode JSON.
5. Smoke test: 5 episodes per speed. Compare logged filter vs truth.
6. Full G2b: n = 100, G1 seeds, paired flips with a cause.
7. Pass (drop under 10 points at 15 and at 20) → stop and start the random-direction throw (§5). Fail → filter-vector retrain, then the same seeds again.

Wrist camera stays in Phase 3 even if the hand hides the fruit on a flipped seed.

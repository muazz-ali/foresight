# Execution Plan — Explicit-Future Dynamic Manipulation
*Working name: **Foresight** (rename freely). Hardware: 1 static RGB camera, 1 wrist-mounted RGB-D, any 6-DOF arm. Simulator: Isaac Sim.*

**Plain words:** **oracle** = perfect object pose from the simulator; 
**conditioning vector** = the 4 future numbers we give the policy (Phases 1–2), growing toward ~12 when the filter is live.

---

## 0. The whole idea on one page

**Hypothesis.** A policy that is *explicitly told* where the object will be when its action lands — by a plain physics filter — will beat policies that must figure this out implicitly (DynamicVLA) or through a learned latent world model (AHEAD), while being smaller, cheaper to train, and easier to debug.

**System in one paragraph.** A tracker follows the object in the camera images. A Kalman filter turns the track into position + velocity + "how sure am I," and can be asked "where will it be X ms from now?" X is set to the *measured* delay of our own pipeline, so the answer describes the world at the exact moment our action takes effect. That answer — about 12 numbers — is appended to the robot-state input of a small VLA policy. The policy handles semantics and strategy (which object, how to approach); it never has to be fast or clairvoyant. A separate 100–500 Hz reflex loop handles the last ~10 cm and closes the gripper at the mathematically right moment. Timestamps everywhere make sure nobody ever acts on old information.

**Why it's feasible (one number).** Prediction error grows roughly as (velocity error × horizon) + ½(acceleration surprise × horizon²). With a ≤300 ms end-to-end delay and objects at ≤20 cm/s, the filter only ever needs to look ~0.3 s ahead — over which tabletop motion is almost perfectly constant-velocity. Expected prediction error: well under 1 cm. That is smaller than a gripper opening. Both papers solve a harder problem than the physics requires.

**Success criterion for the project.** On a shared benchmark (mini-DOM style tasks), our system's success rate stays nearly flat as object speed rises from 0 → 20 cm/s (eval bins **0 / 10 / 15 / 20 cm/s**), while the same policy *without* the future-state input degrades steeply — in sim first, then on the real arm.

**Where we are (Sep 18 2026): G1 passed; G2a (camera numbers vs sim truth, offline) passed; G2b (closed-loop swap) is next.** Retrain on 3,718 demos (`runs/p1_r4k_a|b`), 8-step replan, scripted place after the grasp. Model B: **84% still / 70% at 15 cm/s / 71% at 20 cm/s** (n=100 each). Model A: 9% still (n=100), 0% at 15 and 20 (n=20). B with its 4 numbers zeroed: **2%** still (n=100). So the numbers are the channel B uses, and B barely drops with speed. Two honest limits: (1) the 13-point drop from still to 20 cm/s passes the <15 bar, but its 95% range is about 2–24 points; (2) we have not yet split "B knows where the object **is**" from "B knows where it **will be**" (no Δ=0 model). Say "explicit object state helps" until that ablation runs. Details and settings: §7.

---

## 1. Methodology in plain words: four jobs, four tools

The core design principle: give each job to the tool that is naturally best at it, and let no tool do a job it's bad at.

**Job 1 — "Where is it, and where will it be?" → tracker + Kalman filter.**
A detector finds the named object once at the start ("the red cup" → a box in the image). A lightweight tracker follows that box every frame. The pixel position is converted to meters (Section 3). The Kalman filter is just a running best-guess of position and velocity: each camera frame nudges the guess; between frames, textbook physics (position += velocity × dt) carries it forward; and it honestly tracks its own uncertainty, which shrinks with good measurements and grows while coasting. "Predict the future" = run the physics step forward without measurements. Whole thing: ~200 lines with an off-the-shelf library. This replaces AHEAD's 4.9M-parameter latent world model and RAFT optical flow.

**Job 2 — "What should I do about it?" → a small policy that is told the future.**
A SmolVLA-class policy receives images, proprioception, the language instruction — and our extra future vector: *the object's predicted XY at the moment the action will land and its XY velocity.* Live pack today = **4 numbers** with Δ fixed at 0.25 s (Δ becomes a 5th number once latency really varies; full ~12 when filter uncertainty / coast are live). Nothing else about the network changes. The policy's remaining job is what neural nets are genuinely good at: which object, what grasp, what approach, what to do with it afterward. It no longer needs to infer motion from pixel differences (DynamicVLA's implicit route) or be big enough to imagine futures (AHEAD's route).

**Job 3 — "Exactly when?" → a reflex loop.**
A dumb, fast (100–500 Hz) loop that activates only in the final ~10 cm: it steers the gripper straight at the filter's predicted meeting point and closes the fingers when *time-until-object-arrives* drops below *time-the-gripper-needs-to-close*. No learning. This is the piece neither paper has — both sidestep it with lenient success criteria, nets, and paddles — and it is what absorbs the last centimeter of prediction error on hard tasks.

**Job 4 — "Never act on old information." → clock discipline.**
Every image is timestamped at capture. We measure, once, the true delay from capture to motor motion. The policy's output actions carry the wall-clock times they are meant for; the executor plays them like a metronome. When a new action chunk arrives, it simply overwrites the future entries of the old one; entries whose time has already passed are dropped unplayed. That single buffer-with-overwrite is DynamicVLA's Continuous Inference + LAAS reduced to ~150 lines — and it transfers to any policy.

**Division of labor summary:** the filter answers *where/when* (physics is superb at this over 0.1–0.5 s), the policy answers *what/how* (semantics), the reflex answers *now* (precision timing), and the clock keeps them honest.

---

## 2. Hardware mapping (your rig, exactly)

**Static RGB (assumed third-person, front-facing ~1 m from workspace, like both papers).** This is the eye that never loses the scene — the primary tracking sensor. It has no depth, but Phase 1 doesn't need any: with a calibrated camera and a known table height, a pixel defines a 3D ray, and the object's 3D position is simply where that ray hits the table plane. One matrix multiply. (This is why Phase 1 constrains motion to *on the table* — rolling, sliding — which covers most of both papers' tasks.)

**Wrist RGB-D.** Two roles. (a) Close-range refinement: as the hand approaches, it measures the object *relative to the hand* — which is exactly the quantity grasping needs — so base-calibration errors largely cancel in the final centimeters. (b) It removes the on-table assumption later: real depth enables ballistic/off-table tasks in Phase 5. A short-range depth unit (D405-class) is ideal on a wrist. The Kalman filter fuses both cameras naturally — they're just two measurement sources arriving asynchronously.

**6-DOF arm.** Sufficient — note AgileX PiPER (used in DynamicVLA's real experiments) is itself 6-DOF, which helps comparability. Two checks matter far more than DOF count: (1) the arm must accept *streamed* joint/Cartesian targets at ≥100 Hz (needed by the reflex layer — verify this before committing to a specific arm); (2) measure the gripper's close time **t_close** with a stopwatch/log — typically 0.2–0.7 s, and it becomes a literal constant in the grasp trigger.

**One PC, one clock.** Run everything on a single machine; timestamp frames at the driver level. Measure end-to-end latency with a neat trick: wave the robot's own gripper in front of the static camera and cross-correlate when the *camera* sees it move vs. when the *encoders* say it moved. That difference is your true perception latency; add policy + actuation time for the full pipeline delay Δ.

---

## 3. Components, defaults, and code budget

| Module | Default choice | Why | Custom code |
|---|---|---|---|
| Detector (once/episode) | Open-vocab detector (OWLv2 / YOLO-World); in sim, ground-truth masks | Language → box, then hand off | ~50 lines glue |
| Tracker (every frame) | Escalation ladder: HSV blob → OpenCV CSRT → SAM2-tiny only if needed | Start with the dumbest thing that works | ~100 lines |
| 3D lift | Static cam: ray–table-plane intersection. Wrist cam: median depth in mask → transform via forward kinematics | Depth-free 3D on the table; hand-relative 3D up close | ~80 lines |
| State filter | Constant-velocity Kalman filter (6 states; optional 9 with acceleration), e.g. FilterPy | Exact enough over 0.3 s; gives covariance for free | ~200 lines |
| Policy | SmolVLA fine-tune via LeRobot; conditioning vector appended to the state input | Open, small (~450M), consumer-GPU trainable, async stack exists; natural baseline vs. DynamicVLA lineage | ~50 lines change |
| Executor | Timestamped action buffer with overwrite + drop-stale | CI + LAAS in one data structure | ~150 lines |
| Reflex | Cartesian PD servo toward predicted intercept + time-to-contact grasp trigger | Precision timing without learning | ~150 lines |

**Total bespoke code: roughly 1,000 lines** beyond off-the-shelf parts. That is the "implementation as simple as the hypothesis" claim, made concrete.

**The conditioning vector.** Plan target remains ~12 numbers. **Live pack today is 4** (`interfaces/state.py`, `CONDITIONING_LAYOUT`). Oracle and filter must emit this exact pack:

| Field | Dim | Meaning | Now (G1 model, G2 swap) |
|---|---|---|---|
| p̂_x, p̂_y (t+Δ) | 2 | Predicted object XY at action-landing time, p̂ = p + v·Δ | packed |
| v̂_x, v̂_y | 2 | XY velocity | packed |
| Δ | 1 | Look-ahead used | **fixed 0.25 s, not packed** (`conditioning_delta_s`) |
| p̂_z, v̂_z | 2 | Height / vertical speed | dropped (on-table, near-constant) |
| σ_p | 1–3 | Position uncertainty | dropped; add only after the G2 swap result (retrain) |
| valid / coast-time | 1 | Track live vs coasting | dropped; add only after the G2 swap result (retrain) |
| (optional) intercept XY | 2–3 | Hand–object meeting point | later |

How Δ was settled: the first G1 try sampled Δ ∈ [0, 0.4] s in training without packing it, which smeared p̂ by up to 8 cm. We fixed Δ = 0.25 s at collect, train and eval instead, which closes the same hole. With Δ fixed, [p̂, v̂] carries the same information as [p, v]. So G1 shows "explicit state helps". Showing "future state helps" needs a Δ=0 model (§8 baseline 3). Pack Δ as a 5th number when latency really varies (Phase 3 latency sweep, Phase 4 real arm). Model B state = 8 robot + 4 numbers = **12**.

**Default rates.** Sim cameras and control run at **25 Hz** (`sim.dt = 0.04`); real cameras 30 Hz. Filter: update at camera rate, predict on demand. Policy: one chunk every 150–300 ms **or** a 50-step chunk at 25 Hz (2 s) for still-object sanity; G1 speed claim uses the short chunk **with a hold latch**. Each chunk = 8–50 actions at 25–30 Hz. Reflex: as fast as the arm accepts (100–500 Hz). **Latency budget target ≤ 300 ms**: capture 33 + tracker 5–10 + filter ~0 + policy 100–200 (amortized by chunking) + transport/actuation 50–100. At 20 cm/s object speed that is a ≤ 6 cm lead — comfortably inside constant-velocity accuracy. Short chunks without a latch unglue; long chunks without overwrite go stale — report both, do not pick one and call it G1.

---

## 4. Data: a mini-DOM in Isaac Sim

Replicate DynamicVLA's collection recipe at 1/50th scale — it is the part of that paper most worth copying, and Isaac Sim is exactly the tool they used.

**Scene.** Table + arm + 5–15 household objects (one language-specified target, rest distractors). Object speeds sampled **0–0.20 m/s** (bins 0 / 10 / 15 / 20 cm/s; some static); friction, lighting, textures randomized. Two cameras placed to mirror your real rig (static front RGB; wrist RGB-D).

**Scripted expert.** A four-stage state machine reading *ground-truth* object state: (1) approach a point ~0.25 s ahead of the object, hovering ~10 cm above, continuously updated; (2) descend, stabilize, grasp, lift; (3) move to target, place; (4) reset. Drop detected → back to stage 1.

**Log per timestep:** both camera streams, proprioception, expert actions, ground-truth object state, and — crucially — *the conditioning vector exactly as the policy will later see it.*

**Two training augmentations that carry most of the robustness (both are one-liners):**
1. **Latency randomization (deferred).** Sample Δ ∈ [0, 400 ms] per sequence, feed the state at t+Δ, **and pack Δ in the vector**. The policy becomes delay-agnostic; at deployment you just set Δ = measured latency. Not used yet: in sim the camera has zero frame lag (checked, §7 Phase 2), so a fixed Δ = 0.25 s is honest. Turn this on with the 5th number in Phase 3/4. Never sample Δ without packing it.
2. **Filter-realistic noise.** Add noise to the conditioning vector matching what the real filter produces. Today B trains with independent Gaussian noise, pos σ = 1.5 cm and vel σ = 3 cm/s (`policy/configs/smolvla_b.yaml`). A real filter's error is smooth over time and spikes at wall bounces, which is not the same thing. If the G2 swap costs too much, the cheap fix is to run the tracker + filter over the recorded training demos (they store the static RGB) and retrain B on those vectors. No recollection needed.

**Budget.** 2,000–5,000 episodes for one task family ("pick the moving X, place in the static Y"), collected headless at hundreds/hour — a day or two of compute, not two weeks on 32 A100s. If the hypothesis is right, the explicit conditioning should also make the policy *more sample-efficient*, which we measure directly (Section 8).

**Real-world data later (Phase 4):** reuse DynamicVLA's best real-world trick — drive the *same* scripted expert with your *real* filter instead of teleoperation. ~200–500 real episodes, no human reaction time needed.

---

## 5. Training recipe

Fine-tune SmolVLA (LeRobot) on the sim demos. Train two models that are identical except for one thing:
- **Model A (baseline):** images + proprio + language.
- **Model B (ours):** the same, plus the **4-number** live vector (XY p̂, XY v̂ at fixed Δ = 0.25 s). Full ~12 when the filter is live.

Single GPU, hours-to-days. One mandatory sanity check: at eval, zero out B's vector — success should collapse **to about Model A**. If it doesn't, the network is ignoring the input (fix: increase vector weight/normalization, or drop out images occasionally during training so the vector carries signal). Do **not** score this as “collapse failed” when B is already 10% and B-zero is 0% (a 10-point drop can miss a 15-point bar while zeros are a total collapse). Pass if B-zero ≈ A, or if the drop is ≥15 points **and** B itself is at least ~20% so the bar is meaningful.

---

## 6. Execution stack (runtime)

1. **Buffer.** Policy outputs (timestamp, action) pairs. Executor pops whichever entry matches "now" at each control tick. New chunk arrives → overwrite all future entries; already-passed entries are silently dropped. Blend across the overwrite boundary with a short linear crossfade to kill the chunk-switch jerk that inflates DynamicVLA's path lengths.
2. **Reflex takeover.** When the gripper is within ~10 cm of the predicted intercept, the reflex PD loop takes over fine positioning directly from the filter (bypassing policy latency entirely) and fires the gripper when *time-to-contact < t_close + margin*.
3. **Safety rules.** Workspace bounding box (abort outside). Coast rule: if the filter has been coasting (no measurements) for >0.5 s, hold position, re-run the detector, and refuse blind grasps.

---

## 7. Phases and go/no-go gates

**Phase 0 — Environment + oracle (setup).** Build the Isaac Sim scene, the message interface (`position, velocity, covariance, timestamp, valid` — *use this exact same interface for oracle and estimator*, so every later swap is a config flag), and the scripted expert. **Gate G0:** expert ≥70% success across the **0–20 cm/s** range (0 / 10 / 15 / 20); auto-collection verified at hundreds of episodes/hour.

**Phase 1 — Oracle-conditioned policy (the hypothesis test).** Collect 2–5K episodes; train A and B; evaluate on a speed sweep. **Gate G1:** B beats A by ≥20 points at **mid speed (15 cm/s)**, and B loses <15 points from **0 → 20 cm/s**. Zero-out: B-zero ≈ A (see §5). **n ≥ 80 trials per speed bin** to claim pass/fail; 20 is scout only (Wilson intervals overlap a 20-point bar).

The earlier holes are closed: Δ is fixed (not sampled-unpacked); the eval hold latch welds on close + within 5 cm with a 3-frame dwell; policy labels are `never-held | held-no-lift | success`; eval replans every 8 steps (320 ms).

**G1 result — PASS (Sep 13 2026).** Settings (freeze these, G2 reuses them): checkpoints `runs/p1_r4k_a|b/pretrained_model`, `--n-action-steps 8`, `--force-place`, Δ = 0.25 s, seeds 5000–5099 (0 cm/s), 5150–5249 (15), 5200–5299 (20). Data: `data/eval/scout_*_100runs*`, `data/eval/p1_r4k_report/`.

| Check | Bar | Result | Verdict |
|---|---|---|---|
| B vs A at 15 cm/s | B ≥ A + 20 | B 70/100 (Wilson 60–78%) vs A 0/20 (0–16%) | pass. A is only n=20, but its upper bound is 44 points under B's lower bound, so more A runs cannot flip it |
| B flat 0 → 20 cm/s | drop < 15 | 84 → 71 = 13 points (15 → 20 is flat: 70 vs 71) | pass, but thin: the drop's 95% range is ~2–24 points |
| Zero-out | B-zero ≈ A | B-zero still 2/100 (1–7%) vs A still 9/100 (5–16%) | pass: full collapse |

What G1 does **not** show yet: (1) the 10 cm/s bin was not run for either model; (2) "future" vs "now" is not split. B's hand ends up 2.2 cm from where the object *is* and 5.0 cm from p̂ (`agents/a24_REPORT.md`), so B leans on the vector as position. (3) "Success" is grasp success. Once the latch welds, a script carries and places (`--force-place`), and the 5 cm weld radius is lenient. A and B share all of this, so the comparison is fair, but say "grasp success" in any write-up. None of these block Phase 2 (see §11).

**Phase 2 — Real perception, still in sim.** Replace sim truth with tracker + filter on the rendered static camera, feed the **same 4 numbers** to the **unchanged** G1 Model B, and measure what the swap costs. One change only: no retrain, no new inputs, no wrist fusion.

*Settings checked on recorded data (Sep 2026):*

| Item | Value | Why it matters |
|---|---|---|
| Static cam | 480×360, fx = fy = 240, cx = 240, cy = 180 (90° wide); at (1.0, 0, 0.60), XYZ-Euler [0, 60, 90]°, OpenGL convention; looks 30° down | Projecting sim truth with this model lands on the fruit (checked visually). ~3.5–3.8 px per cm; fruit ~22 px wide |
| Frame lag | **0 steps.** Blob centre is 1.2 px from truth at step t, 3.8 px from truth at t−1 | In sim, perception adds no delay. Δ = 0.25 s stays the whole look-ahead |
| Pixel → table | Intersect the ray with the plane **z = object centre height** (episode `object_half_height`, 2.5–4.5 cm by fruit), not `table_height` = 0 | Using z = 0 gives a 2–3.5 cm bias. Correct height gives 0.1–0.6 cm median. (Real-arm equivalent: a known-size table or wrist depth) |
| Object motion | Constant speed inside a 24 × 30 cm box with wall bounces; 0.40 / 0.67 / 0.93 wall hits per second at 10 / 15 / 20 cm/s | A straight-line guess is wrong across a bounce. **Even sim truth** (p + 0.25 v) misses the real 0.25 s future by RMSE 0.9 / 1.7 / 2.6 cm (22% of frames > 2 cm at 20 cm/s) |
| Scene | One fruit + one container, grey table/robot/bowl, one dome light, no randomization, no distractors | A colour blob is enough for most fruits. Distractors and randomization (§4) are **not** in the A/B training data; do not add them in Phase 2 |
| Fruits | 12 kinds. **Egg is white** (~9% of episodes): a colour blob loses it on 100% of frames | Solve it or report egg as its own row. Never drop it silently |
| Choosing the blob | "Largest coloured blob" grabs the wrong thing for apple / egg / lemon / peach. "Nearest blob to the filter's guess, within 25 px" works | Detector-once = project the sim spawn pose (plan §3: sim may use truth to *find* the object once); tracking after that uses pixels only |
| Libraries | No `filterpy`, no OpenCV CSRT (needs `opencv-contrib`), no ultralytics / SAM2, in either conda env | Rung 1 (HSV blob + numpy Kalman) needs no new installs. Try-import before adding any |
| Eval harness | `scripts/eval_policy.py` builds B's vector from `scene.get_object_state()` (truth). `--oracle-conditioning` does nothing. Episode JSONs do not record checkpoint / replan steps / force-place (only the gate report and log do) | Needs a `--cond-source oracle|filter` switch and those fields in every episode JSON |

*First number (scratch test, 40 recorded demos per speed, frames before the grasp):* HSV blob + 4-state constant-velocity Kalman filter (reset speed when a measurement jumps, i.e. a bounce) gives filter p̂ vs sim-truth p̂ **median 0.27 cm, RMSE 0.50 cm at 10 cm/s; median 0.28 cm, RMSE 1.39 cm at 20 cm/s**. Position error median 0.23 cm, speed error median 0.4–0.5 cm/s. That sits inside B's training noise (1.5 cm / 3 cm/s). The error that remains comes from bounces and the egg.

**Gate G2 (revised).** The old bar "error @250 ms ≤ 1–2 cm vs ground truth" cannot be passed at 15–20 cm/s: sim truth itself fails it because of bounces. Measure the filter against what B was trained on:
- **G2a — prediction (offline, no Isaac):** filter p̂ vs sim-truth p̂ at Δ = 0.25 s, free-object frames, all fruits pooled: **median ≤ 1 cm and RMSE ≤ 2 cm at 20 cm/s.** Also report, at 100 / 200 / 300 ms horizons: error vs the true future, split into bounce-free windows and windows with a bounce, with sim truth's own error beside it. Also report track rate per fruit.
- **G2b — the swap (Isaac):** B-filter vs B-oracle success, same settings and **same seeds** (so each seed is a paired before/after), n = 100 at 0 / 15 / 20 cm/s: **drop < 10 points at 15 and 20 cm/s.** List the seeds that flipped from success to fail, with their cause (bounce in window / fruit lost / coasting).

**Phase 3 — Clock discipline + reflex.** Add the timestamped executor, chunk blending, and reflex layer; inject artificial latency to stress-test. **Gate G3:** among failures previously tagged "too late / early close," ≥half are recovered; path length and jerk visibly drop; success at **20 cm/s** improves.

**Phase 4 — Real robot.** Calibration checklist: static-camera extrinsics to robot base (ArUco board); table plane by touching three points with the TCP (simplest and most accurate); wrist hand-eye (standard chessboard); latency measurement (gripper-wave trick, Section 2). Port tracker + filter (they should transfer almost unchanged — that is the point of explicit state). Collect ~300 real episodes with the filter-driven expert; fine-tune. **Gate G4:** ≥60–70% on sliding/rolling tasks at 10–20 cm/s.

**Phase 5 — Stretch (the "complex tasks" story).** (a) Upgrade the filter to IMM (a small committee of motion models — rolling / decelerating / ballistic / stopped — that votes); (b) uncertainty-driven behavior: when σ_p is large, *stage* at the intercept corridor with the gripper pre-opened and delay commitment instead of chasing the mean; (c) ballistic catches using wrist depth; (d) occlusion demo — the filter coasts through occluders with growing covariance *by construction*, which is the transparent version of AHEAD's flashiest result.

---

## 8. Evaluation protocol

**Primary curve:** success rate vs. object speed (**0 / 10 / 15 / 20 cm/s**). **Scout:** 20 trials per point. **Claim / gate:** ≥80 per point, with Wilson 95% intervals in the report — the flatness of this curve *is* the claim. 20 trials cannot resolve a 20-point G1 margin. For before/after comparisons of one model (oracle → filter, latency on/off), rerun the **same seeds** and count flips. That is much tighter than two independent n=100 samples, where the gap itself carries about ±12 points of noise. Until Phase 3, "success" = the policy's grasp held under the eval latch, then a scripted place (`--force-place`); label it that way.

**Secondary:** filter prediction RMSE vs. horizon; per-component latency table (report on your deployable GPU, not a data-center card); path length + smoothness; **sample-efficiency curve** (A vs. B trained on 250/500/1K/2K/4K demos — if B wins big here, that's a headline result on its own); **latency-injection sweep** (artificially inflate Δ, plot degradation — our method should be uniquely flat because the policy was trained delay-agnostic).

**Failure taxonomy, logged every episode from day one.** Expert logs (real SM stages): never-engaged / too-late / early-close / wrong-object / grasp-slip / off-table / safety-abort. **Policy eval must not reuse that classifier on `{0,4}` glue flags** — 4 can never become lift / slip / early-close, so almost every grasp-that-did-not-place becomes “too-late.” For policy, log and report: `never-glued | glued-no-lift | lifted-no-place | success`, plus glue frames, glue-run count, `z_max`, `z_end`, closest XY/3D. Success % itself uses height / bowl, not those names — do not confuse the story with the pass/fail number.

**Baselines:** (1) A — vanilla SmolVLA; (2) A + our executor only (isolates scheduling from conditioning); (3) B with Δ=0 (current-state conditioning — isolates *explicit* state from *future* state); (4) B full; (5) oracle-state ceiling. Optional if time: an AHEAD-style latent predictor bolted to the same backbone, to isolate explicit vs. latent prediction at matched latency.

---

## 9. Risks → simple mitigations

**Tracker loses the object** → filter coasts with growing covariance; reflex refuses blind grasps past the coast limit; detector re-fires. The failure is graceful and visible, not silent.
**Calibration error dominates at grasp scale** → the wrist RGB-D measures the object hand-relative in the final approach, canceling base-frame error where it matters most.
**Arm control API too slow for the reflex** → verify streaming rate before hardware purchase; if capped ~50 Hz, widen the grasp-trigger margin and reduce top object speed — the architecture still works, just with a smaller envelope.
**Table-plane assumption breaks (object bounces/lifts)** → covariance spikes when measurements disagree with the plane model; treat as "uncertain" and stage rather than chase; full fix is Phase 5's depth fusion.
**Policy ignores the vector** → the zero-out sanity check catches it in Phase 1, where it's cheap. Score it against Model A, not a 15-point bar when B is already near floor.
**Eval glue unsticks on open** → closed for G1 (weld on close + near with a 3-frame dwell, scripted place after). The cost: success means grasp, and the weld is lenient. Phase 3 should replace it with a real physics grasp (`--no-attach`).
**Δ sampled but not packed** → closed by fixing Δ = 0.25 s everywhere. Never widen the Δ range without packing it as a 5th number.
**Wall bounces break straight-line prediction** → even sim truth is 2.6 cm RMSE @250 ms at 20 cm/s. Grade the filter against sim-truth p̂ (what B learned), and report bounce windows separately. A bounce-aware predictor (`sim/motion.py:kinematic_forecast`) or IMM would change B's input, so it needs a retrain. That is Phase 5, not a G2 fix.
**White egg invisible to a colour blob** → solved in G2a. The tracker picks a "bright" look on the first frame (white pixels, then an opening that erases thin grid lines) and tracks the egg on 99.9% of frames.
**Filter error is smooth in time, training noise was not** → if the G2 swap drop is ≥10 points, retrain B on filter-made vectors from the recorded demos (§4 aug 2) before adding σ / valid / coast.
**"Need a latent world model" (AHEAD / DynaWM / PhysMani)** → those papers also act on a **predicted future** and keep the VLA from inventing physics. Our cheap version is the Kalman + 4-number pack. Do not vendor their nets.
**"It's not end-to-end" reviewer critique** → the ablation ladder (Section 8) reframes the paper as *finding the minimal sufficient structure*, which is a stronger claim than another architecture.

---

## 10. What we deliberately do not build (and why)

No latent world model (a Kalman filter answers the same question in 200 lines, in 3D, with honest uncertainty). No AHEAD / DynaWM / PhysMani modules — steal their *habits* (freeze the VLA, condition on an explicit future, evaluate with enough trials), not their architectures. No optical flow network (the tracker + filter already yield velocity). No VLA trained from scratch (fine-tuning a 450M open model is days, not GPU-months). No 6D object pose (centroid + velocity suffices for grasping convex household objects; add orientation only when a task demands it). No 200K episodes (explicit conditioning should slash the data the policy needs — and we measure that claim rather than assume it). No off-table motion in Phase 1 (the plane constraint is what makes a single RGB camera a 3D sensor). No unfreezing the vision encoder to “make cameras work” while B-zero already proves the vector is the channel.

---

## 11. Next, concretely (Phase 2, after G1 pass)

One change at a time. The G2 swap keeps Model B, the 4-number pack, Δ = 0.25 s, and every eval setting from G1. Steps 1–2 need **no Isaac**: every recorded demo and eval `.h5` already stores static + wrist RGB, sim-truth object pose / speed, and timestamps.

1. ✅ **Build perception offline** (done Sep 18 2026). `perception/camera.py`, `track.py`, `kalman.py`, `estimator.py` (~370 lines total). Filter knobs were tuned on demo speeds 12/14/16/18 cm/s only, kept out of the report. `tests/test_perception.py` has 12 tests and needs no Isaac. Added from the data: **re-detect after 0.5 s blind** (search the play area for the same look and size), because B sometimes parks its hand between the camera and the fruit.
2. ✅ **Score G2a offline — PASS** (`data/eval/p2_perception_report/REPORT.md`). At 20 cm/s, filter p̂ vs sim-truth p̂: median / RMSE **0.34 / 1.14 cm** on 609 demos and **0.31 / 1.26 cm** on 100 B eval runs. Every fruit is tracked (egg via the bright look). Error comes from: the first frame (the filter starts at speed 0, so 5 cm off at 20 cm/s), a few frames after each wall hit, and 2 of 300 eval runs where the hand hid the fruit (46 frames, ~14 cm off).
3. **Wire the switch.** `scripts/eval_policy.py --cond-source oracle|filter` (default oracle, so G1 reruns are unchanged). Use `StaticCamEstimator.from_cfg(scene.cfg)`. Feed it the same pre-step static frame B receives, at that frame's sim time, and **warm it on ≥ 2 frames before B's first call** (else the first 8-step chunk is planned from a p̂ that is v × 0.25 s off). Delete the dead `--oracle-conditioning` flag. Log the filter pack and the sim-truth pack side by side, plus valid / redetects per frame. Write checkpoint, replan steps, force-place, cond source and Δ into every episode JSON. Smoke test: 5 episodes per speed, compare logged filter vs truth vectors.
4. **Score G2b in Isaac.** B-filter at 0 / 15 / 20 cm/s, n = 100, seeds and settings copied from G1. Report paired flips with a cause per flip (warm-up / wall hit / hand hid the fruit / other). Hand-hides-fruit may be more common in closed loop than the 2/300 seen offline; if it drives the drop, the fix is the wrist camera (Phase 3), not a bigger tracker. Pass (drop < 10 at 15 and 20) → Phase 3. Fail → retrain B on filter-made vectors from the recorded demos (§4 aug 2), then rerun step 4. Only after that consider packing σ / valid / coast (needs a retrain).

**Run beside steps 1–2 (Isaac GPU is free while perception is offline; none of these block G2):**
- B-oracle and A at **10 cm/s**, n = 100: fills the missing bin on the G1 curve.
- **Model B trained with Δ = 0** (§8 baseline 3; ~3 h train + 3 × 100 eval). Needed before any "future state" claim, since G1 so far shows "object state". Evaluating the Δ = 0.25 model with Δ = 0 inputs is not a substitute, because those inputs are outside its training range.

**Do not build in Phase 2:** wrist-camera fusion (the static cam sees the fruit until the grasp), bounce-aware or IMM prediction (changes B's input), distractors or randomization, the timestamped executor or reflex (Phase 3), or new pack fields before the swap result.

Job 3 (reflex) is still the last-centimeter closer for **moving** objects, and the place to tighten the lenient 5 cm weld into a real physics grasp.
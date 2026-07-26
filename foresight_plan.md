# Execution Plan — Explicit-Future Dynamic Manipulation
*Working name: **Foresight** (rename freely). Hardware: 1 static RGB camera, 1 wrist-mounted RGB-D, any 6-DOF arm. Simulator: Isaac Sim.*

---

## 0. The whole idea on one page

**Hypothesis.** A policy that is *explicitly told* where the object will be when its action lands — by a plain physics filter — will beat policies that must figure this out implicitly (DynamicVLA) or through a learned latent world model (AHEAD), while being smaller, cheaper to train, and easier to debug.

**System in one paragraph.** A tracker follows the object in the camera images. A Kalman filter turns the track into position + velocity + "how sure am I," and can be asked "where will it be X ms from now?" X is set to the *measured* delay of our own pipeline, so the answer describes the world at the exact moment our action takes effect. That answer — about 12 numbers — is appended to the robot-state input of a small VLA policy. The policy handles semantics and strategy (which object, how to approach); it never has to be fast or clairvoyant. A separate 100–500 Hz reflex loop handles the last ~10 cm and closes the gripper at the mathematically right moment. Timestamps everywhere make sure nobody ever acts on old information.

**Why it's feasible (one number).** Prediction error grows roughly as (velocity error × horizon) + ½(acceleration surprise × horizon²). With a ≤300 ms end-to-end delay and objects at ≤40 cm/s, the filter only ever needs to look ~0.3 s ahead — over which tabletop motion is almost perfectly constant-velocity. Expected prediction error: ~1 cm. That is smaller than a gripper opening. Both papers solve a harder problem than the physics requires.

**Success criterion for the project.** On a shared benchmark (mini-DOM style tasks), our system's success rate stays nearly flat as object speed rises from 0 → 30–40 cm/s, while the same policy *without* the future-state input degrades steeply — in sim first, then on the real arm.

---

## 1. Methodology in plain words: four jobs, four tools

The core design principle: give each job to the tool that is naturally best at it, and let no tool do a job it's bad at.

**Job 1 — "Where is it, and where will it be?" → tracker + Kalman filter.**
A detector finds the named object once at the start ("the red cup" → a box in the image). A lightweight tracker follows that box every frame. The pixel position is converted to meters (Section 3). The Kalman filter is just a running best-guess of position and velocity: each camera frame nudges the guess; between frames, textbook physics (position += velocity × dt) carries it forward; and it honestly tracks its own uncertainty, which shrinks with good measurements and grows while coasting. "Predict the future" = run the physics step forward without measurements. Whole thing: ~200 lines with an off-the-shelf library. This replaces AHEAD's 4.9M-parameter latent world model and RAFT optical flow.

**Job 2 — "What should I do about it?" → a small policy that is told the future.**
A SmolVLA-class policy receives images, proprioception, the language instruction — and our extra ~12-number vector: *the object's predicted position at the moment the action will land, its velocity, the lookahead time used, and the uncertainty.* Nothing else about the network changes. The policy's remaining job is what neural nets are genuinely good at: which object, what grasp, what approach, what to do with it afterward. It no longer needs to infer motion from pixel differences (DynamicVLA's implicit route) or be big enough to imagine futures (AHEAD's route).

**Job 3 — "Exactly when?" → a reflex loop.**
A dumb, fast (100–500 Hz) loop that activates only in the final ~10 cm: it steers the gripper straight at the filter's predicted meeting point and closes the fingers when *time-until-object-arrives* drops below *time-the-gripper-needs-to-close*. No learning. This is the piece neither paper has — both sidestep it with lenient success criteria, nets, and paddles — and it is what absorbs the last centimeter of prediction error on hard tasks.

**Job 4 — "Never act on old information." → clock discipline.**
Every image is timestamped at capture. We measure, once, the true delay from capture to motor motion. The policy's output actions carry the wall-clock times they are meant for; the executor plays them like a metronome. When a new action chunk arrives, it simply overwrites the future entries of the old one; entries whose time has already passed are dropped unplayed. That single buffer-with-overwrite is DynamicVLA's Continuous Inference + LAAS reduced to ~150 lines — and it transfers to any policy.

**Division of labor summary:** the filter answers *where/when* (physics is superb at this over 0.1–0.5 s), the policy answers *what/how* (semantics), the reflex answers *now* (precision timing), and the clock keeps them honest.

---

## 2. Hardware mapping (your rig, exactly)

**Static RGB (assumed third-person, front-facing ~1 m from workspace, like both papers).** This is the eye that never loses the scene — the primary tracking sensor. It has no depth, but Phase 1 doesn't need any: with a calibrated camera and a known table height, a pixel defines a 3D ray, and the object's 3D position is simply where that ray hits the table plane. One matrix multiply. (This is why Phase 1 constrains motion to *on the table* — rolling, sliding, conveyor — which covers most of both papers' tasks.)

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

**The conditioning vector (~12 numbers, normalized to workspace scale):**

| Field | Dim | Meaning |
|---|---|---|
| p̂(t+Δ) | 3 | Predicted object position at action-landing time |
| v̂ | 3 | Filtered velocity |
| Δ | 1 | The lookahead actually used (also an input → delay-agnostic policy) |
| σ_p | 1–3 | Position uncertainty (std) |
| valid / coast-time | 1 | Is the track live, or coasting (and for how long) |
| (optional) intercept point on table | 3 | Where trajectories of hand and object can meet |

**Default rates.** Cameras 30 Hz. Filter: update at 30 Hz, predict continuously on demand. Policy: one chunk every 150–300 ms, each chunk = 20–50 actions at 25–30 Hz. Reflex: as fast as the arm accepts (100–500 Hz). **Latency budget target ≤ 300 ms**: capture 33 + tracker 5–10 + filter ~0 + policy 100–200 (amortized by chunking) + transport/actuation 50–100. At 30 cm/s object speed that is a ≤ 9 cm lead — comfortably inside constant-velocity accuracy.

---

## 4. Data: a mini-DOM in Isaac Sim

Replicate DynamicVLA's collection recipe at 1/50th scale — it is the part of that paper most worth copying, and Isaac Sim is exactly the tool they used.

**Scene.** Table + arm + 5–15 household objects (one language-specified target, rest distractors). Object speeds sampled 0–0.5 m/s (some static); friction, lighting, textures randomized. Two cameras placed to mirror your real rig (static front RGB; wrist RGB-D).

**Scripted expert.** A four-stage state machine reading *ground-truth* object state: (1) approach a point ~0.25 s ahead of the object, hovering ~10 cm above, continuously updated; (2) descend, stabilize, grasp, lift; (3) move to target, place; (4) reset. Drop detected → back to stage 1.

**Log per timestep:** both camera streams, proprioception, expert actions, ground-truth object state, and — crucially — *the conditioning vector exactly as the policy will later see it.*

**Two training augmentations that carry most of the robustness (both are one-liners):**
1. **Latency randomization.** Sample Δ ∈ [0, 400 ms] per sequence, feed the state at t+Δ, and feed Δ itself as an input. The policy becomes delay-agnostic; at deployment you just set Δ = measured latency.
2. **Filter-realistic noise.** Add Gaussian noise to the conditioning vector matching the covariance you expect from the real filter. Sim→real of the state channel becomes a non-event, because the policy never saw a perfect oracle anyway.

**Budget.** 2,000–5,000 episodes for one task family ("pick the moving X, place in the static Y"), collected headless at hundreds/hour — a day or two of compute, not two weeks on 32 A100s. If the hypothesis is right, the explicit conditioning should also make the policy *more sample-efficient*, which we measure directly (Section 8).

**Real-world data later (Phase 4):** reuse DynamicVLA's best real-world trick — drive the *same* scripted expert with your *real* filter instead of teleoperation. ~200–500 real episodes, no human reaction time needed.

---

## 5. Training recipe

Fine-tune SmolVLA (LeRobot) on the sim demos. Train two models that are identical except for one thing:
- **Model A (baseline):** images + proprio + language.
- **Model B (ours):** the same, plus the 12-number vector.

Single GPU, hours-to-days. One mandatory sanity check: at eval, zero out B's vector — success should collapse. If it doesn't, the network is ignoring the input (fix: increase vector weight/normalization, or drop out images occasionally during training so the vector carries signal).

---

## 6. Execution stack (runtime)

1. **Buffer.** Policy outputs (timestamp, action) pairs. Executor pops whichever entry matches "now" at each control tick. New chunk arrives → overwrite all future entries; already-passed entries are silently dropped. Blend across the overwrite boundary with a short linear crossfade to kill the chunk-switch jerk that inflates DynamicVLA's path lengths.
2. **Reflex takeover.** When the gripper is within ~10 cm of the predicted intercept, the reflex PD loop takes over fine positioning directly from the filter (bypassing policy latency entirely) and fires the gripper when *time-to-contact < t_close + margin*.
3. **Safety rules.** Workspace bounding box (abort outside). Coast rule: if the filter has been coasting (no measurements) for >0.5 s, hold position, re-run the detector, and refuse blind grasps.

---

## 7. Phases and go/no-go gates

**Phase 0 — Environment + oracle (setup).** Build the Isaac Sim scene, the message interface (`position, velocity, covariance, timestamp, valid` — *use this exact same interface for oracle and estimator*, so every later swap is a config flag), and the scripted expert. **Gate G0:** expert ≥70% success across the 0–40 cm/s range; auto-collection verified at hundreds of episodes/hour.

**Phase 1 — Oracle-conditioned policy (the hypothesis test).** Collect 2–5K episodes; train A and B; evaluate on a speed sweep. **Gate G1:** B beats A by ≥20 points at mid speed, and B loses <15 points from 0 → 30 cm/s. *If G1 fails, the core hypothesis is wrong or the conditioning is ignored — stop and diagnose before building anything else.* This is the single most important experiment in the project, and it requires zero perception code.

**Phase 2 — Real perception, still in sim.** Run detector + tracker + filter on Isaac's rendered camera streams. Measure prediction RMSE at 100/200/300 ms horizons vs. ground truth. Swap oracle → filter in B's input. **Gate G2:** prediction error @250 ms ≤ ~1–2 cm on-table; success drop from the swap <10 points. (The noise augmentation from Section 4 is what makes this gate cheap to pass.)

**Phase 3 — Clock discipline + reflex.** Add the timestamped executor, chunk blending, and reflex layer; inject artificial latency to stress-test. **Gate G3:** among failures previously tagged "too late / early close," ≥half are recovered; path length and jerk visibly drop; success at 40 cm/s improves.

**Phase 4 — Real robot.** Calibration checklist: static-camera extrinsics to robot base (ArUco board); table plane by touching three points with the TCP (simplest and most accurate); wrist hand-eye (standard chessboard); latency measurement (gripper-wave trick, Section 2). Port tracker + filter (they should transfer almost unchanged — that is the point of explicit state). Collect ~300 real episodes with the filter-driven expert; fine-tune. **Gate G4:** ≥60–70% on conveyor/rolling tasks at 10–20 cm/s.

**Phase 5 — Stretch (the "complex tasks" story).** (a) Upgrade the filter to IMM (a small committee of motion models — rolling / decelerating / ballistic / stopped — that votes); (b) uncertainty-driven behavior: when σ_p is large, *stage* at the intercept corridor with the gripper pre-opened and delay commitment instead of chasing the mean; (c) ballistic catches using wrist depth; (d) occlusion demo — the filter coasts through occluders with growing covariance *by construction*, which is the transparent version of AHEAD's flashiest result.

---

## 8. Evaluation protocol

**Primary curve:** success rate vs. object speed (0/10/20/30/40 cm/s), ≥20 trials per point — the flatness of this curve *is* the claim.

**Secondary:** filter prediction RMSE vs. horizon; per-component latency table (report on your deployable GPU, not a data-center card); path length + smoothness; **sample-efficiency curve** (A vs. B trained on 250/500/1K/2K/4K demos — if B wins big here, that's a headline result on its own); **latency-injection sweep** (artificially inflate Δ, plot degradation — our method should be uniquely flat because the policy was trained delay-agnostic).

**Failure taxonomy, logged every episode from day one:** never-engaged / too-late / early-close / wrong-object / grasp-slip / off-table / safety-abort. This makes every gate diagnosable instead of a mystery number.

**Baselines:** (1) A — vanilla SmolVLA; (2) A + our executor only (isolates scheduling from conditioning); (3) B with Δ=0 (current-state conditioning — isolates *explicit* state from *future* state); (4) B full; (5) oracle-state ceiling. Optional if time: an AHEAD-style latent predictor bolted to the same backbone, to isolate explicit vs. latent prediction at matched latency.

---

## 9. Risks → simple mitigations

**Tracker loses the object** → filter coasts with growing covariance; reflex refuses blind grasps past the coast limit; detector re-fires. The failure is graceful and visible, not silent.
**Calibration error dominates at grasp scale** → the wrist RGB-D measures the object hand-relative in the final approach, canceling base-frame error where it matters most.
**Arm control API too slow for the reflex** → verify streaming rate before hardware purchase; if capped ~50 Hz, widen the grasp-trigger margin and reduce top object speed — the architecture still works, just with a smaller envelope.
**Table-plane assumption breaks (object bounces/lifts)** → covariance spikes when measurements disagree with the plane model; treat as "uncertain" and stage rather than chase; full fix is Phase 5's depth fusion.
**Policy ignores the vector** → the zero-out sanity check catches it in Phase 1, where it's cheap.
**"It's not end-to-end" reviewer critique** → the ablation ladder (Section 8) reframes the paper as *finding the minimal sufficient structure*, which is a stronger claim than another architecture.

---

## 10. What we deliberately do not build (and why)

No latent world model (a Kalman filter answers the same question in 200 lines, in 3D, with honest uncertainty). No optical flow network (the tracker + filter already yield velocity). No VLA trained from scratch (fine-tuning a 450M open model is days, not GPU-months). No 6D object pose (centroid + velocity suffices for grasping convex household objects; add orientation only when a task demands it). No 200K episodes (explicit conditioning should slash the data the policy needs — and we measure that claim rather than assume it). No off-table motion in Phase 1 (the plane constraint is what makes a single RGB camera a 3D sensor).

---

## 11. First two weeks, concretely

Week 1: choose the arm asset in Isaac Sim (UR5e/PiPER-class with good support); build the table scene with one moving object (kinematic conveyor motion is fine to start — physics-driven rolling later); define and freeze the state-message interface; write the 4-stage scripted expert against ground truth; confirm one full successful episode end-to-end with logging.

Week 2: scale collection headless (target: 300+ episodes/day); write the Kalman filter + unit-test it on synthetic noisy tracks (known ground truth → verify prediction error vs. horizon matches theory); stand up LeRobot with SmolVLA and confirm a training run on 100 episodes overfits; implement the Δ-randomization and noise augmentation in the dataloader. At the end of week 2 you are staring down Gate G1 — the experiment that decides everything else.
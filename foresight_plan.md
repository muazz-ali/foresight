# Execution Plan — Explicit-Future Dynamic Manipulation
*Working name: **Foresight** (rename freely). Hardware: 1 static RGB camera, 1 wrist-mounted RGB-D, any 6-DOF arm. Simulator: Isaac Sim.*

**Plain-language word list:** [`WORDS.md`](WORDS.md)  
(e.g. **oracle** = perfect object pose from the simulator; **conditioning vector** = the 5 future numbers we give the policy in Phase 1, growing toward ~12 when the filter is live.)

---

## 0. The whole idea on one page

**Hypothesis.** A policy that is *explicitly told* where the object will be when its action lands — by a plain physics filter — will beat policies that must figure this out implicitly (DynamicVLA) or through a learned latent world model (AHEAD), while being smaller, cheaper to train, and easier to debug.

**System in one paragraph.** A tracker follows the object in the camera images. A Kalman filter turns the track into position + velocity + "how sure am I," and can be asked "where will it be X ms from now?" X is set to the *measured* delay of our own pipeline, so the answer describes the world at the exact moment our action takes effect. That answer — about 12 numbers — is appended to the robot-state input of a small VLA policy. The policy handles semantics and strategy (which object, how to approach); it never has to be fast or clairvoyant. A separate 100–500 Hz reflex loop handles the last ~10 cm and closes the gripper at the mathematically right moment. Timestamps everywhere make sure nobody ever acts on old information.

**Why it's feasible (one number).** Prediction error grows roughly as (velocity error × horizon) + ½(acceleration surprise × horizon²). With a ≤300 ms end-to-end delay and objects at ≤20 cm/s, the filter only ever needs to look ~0.3 s ahead — over which tabletop motion is almost perfectly constant-velocity. Expected prediction error: well under 1 cm. That is smaller than a gripper opening. Both papers solve a harder problem than the physics requires.

**Success criterion for the project.** On a shared benchmark (mini-DOM style tasks), our system's success rate stays nearly flat as object speed rises from 0 → 20 cm/s (eval bins **0 / 10 / 15 / 20 cm/s**), while the same policy *without* the future-state input degrades steeply — in sim first, then on the real arm.

**Where Phase 1 actually is (Aug 2026).** Gate G1 on 5k demos did **not** cleanly test that criterion. Independent holes (full write-up: [`foresight_revised_plan2.md`](foresight_revised_plan2.md)): Δ sampled in train but **not packed**; eval glue unsticks on any open; policy logs `{0,4}` so the expert labeler calls glued-no-lift “too-late”; n=20 cannot resolve a 20-point bar; `n_action_steps=8` vs checkpoint 50. Same Model B at 50-step: still **18/20**, moving objects still ~5–10%. Do **not** call the hypothesis dead, and do **not** add a latent world model, until Δ is an input, glue matches the expert hold, and n≥80.

---

## 1. Methodology in plain words: four jobs, four tools

The core design principle: give each job to the tool that is naturally best at it, and let no tool do a job it's bad at.

**Job 1 — "Where is it, and where will it be?" → tracker + Kalman filter.**
A detector finds the named object once at the start ("the red cup" → a box in the image). A lightweight tracker follows that box every frame. The pixel position is converted to meters (Section 3). The Kalman filter is just a running best-guess of position and velocity: each camera frame nudges the guess; between frames, textbook physics (position += velocity × dt) carries it forward; and it honestly tracks its own uncertainty, which shrinks with good measurements and grows while coasting. "Predict the future" = run the physics step forward without measurements. Whole thing: ~200 lines with an off-the-shelf library. This replaces AHEAD's 4.9M-parameter latent world model and RAFT optical flow.

**Job 2 — "What should I do about it?" → a small policy that is told the future.**
A SmolVLA-class policy receives images, proprioception, the language instruction — and our extra future vector: *the object's predicted XY at the moment the action will land, its XY velocity, and the lookahead time Δ used.* Phase-1 live pack = **5 numbers** (full ~12 when filter uncertainty / coast are live). Nothing else about the network changes. The policy's remaining job is what neural nets are genuinely good at: which object, what grasp, what approach, what to do with it afterward. It no longer needs to infer motion from pixel differences (DynamicVLA's implicit route) or be big enough to imagine futures (AHEAD's route).

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

**The conditioning vector.** Plan target remains ~12 numbers. **Phase-1 live pack is 5** (oracle demos; G2 adds σ / valid / coast when the filter is live):

| Field | Dim | Meaning | Phase 1 |
|---|---|---|---|
| p̂_x, p̂_y (t+Δ) | 2 | Predicted object XY at action-landing time | packed |
| v̂_x, v̂_y | 2 | XY velocity | packed |
| **Δ** | **1** | Lookahead actually used (delay-agnostic policy) | **packed — do not drop** |
| p̂_z, v̂_z | 2 | Height / vertical speed | dropped (on-table, near-constant) |
| σ_p | 1–3 | Position uncertainty | dropped until filter (G2) |
| valid / coast-time | 1 | Track live vs coasting | dropped until filter (G2) |
| (optional) intercept XY | 2–3 | Hand–object meeting point | later |

G1 lesson: we shipped a **4-number** pack (XY p̂ + XY v̂) and sampled Δ ∈ [0, 0.4] s **without telling the net which Δ built p̂**. Eval used a fixed 0.25 s. That smear grows with speed (~5 cm at 20 cm/s for Δ=0.25; up to 8 cm across the train range). Restore Δ as an input **before** another G1 claim. Model B state = 8 robot + 5 numbers = **13**. Oracle and filter must emit the same pack.

**Default rates.** Cameras 30 Hz. Filter: update at 30 Hz, predict continuously on demand. Policy: one chunk every 150–300 ms **or** a 50-step chunk at 25 Hz (2 s) for still-object sanity; G1 speed claim uses the short chunk **with a hold latch**. Each chunk = 8–50 actions at 25–30 Hz. Reflex: as fast as the arm accepts (100–500 Hz). **Latency budget target ≤ 300 ms**: capture 33 + tracker 5–10 + filter ~0 + policy 100–200 (amortized by chunking) + transport/actuation 50–100. At 20 cm/s object speed that is a ≤ 6 cm lead — comfortably inside constant-velocity accuracy. Short chunks without a latch unglue; long chunks without overwrite go stale — report both, do not pick one and call it G1.

---

## 4. Data: a mini-DOM in Isaac Sim

Replicate DynamicVLA's collection recipe at 1/50th scale — it is the part of that paper most worth copying, and Isaac Sim is exactly the tool they used.

**Scene.** Table + arm + 5–15 household objects (one language-specified target, rest distractors). Object speeds sampled **0–0.20 m/s** (bins 0 / 10 / 15 / 20 cm/s; some static); friction, lighting, textures randomized. Two cameras placed to mirror your real rig (static front RGB; wrist RGB-D).

**Scripted expert.** A four-stage state machine reading *ground-truth* object state: (1) approach a point ~0.25 s ahead of the object, hovering ~10 cm above, continuously updated; (2) descend, stabilize, grasp, lift; (3) move to target, place; (4) reset. Drop detected → back to stage 1.

**Log per timestep:** both camera streams, proprioception, expert actions, ground-truth object state, and — crucially — *the conditioning vector exactly as the policy will later see it.*

**Two training augmentations that carry most of the robustness (both are one-liners):**
1. **Latency randomization.** Sample Δ ∈ [0, 400 ms] per sequence, feed the state at t+Δ, **and pack Δ in the vector**. The policy becomes delay-agnostic; at deployment you just set Δ = measured latency. (If you sample Δ and omit it from the input, the net cannot tell 0.1 s from 0.4 s — that is a G1 hole, not a negative on the hypothesis.)
2. **Filter-realistic noise.** Add Gaussian noise to the conditioning vector matching the covariance you expect from the real filter. Sim→real of the state channel becomes a non-event, because the policy never saw a perfect oracle anyway.

**Budget.** 2,000–5,000 episodes for one task family ("pick the moving X, place in the static Y"), collected headless at hundreds/hour — a day or two of compute, not two weeks on 32 A100s. If the hypothesis is right, the explicit conditioning should also make the policy *more sample-efficient*, which we measure directly (Section 8).

**Real-world data later (Phase 4):** reuse DynamicVLA's best real-world trick — drive the *same* scripted expert with your *real* filter instead of teleoperation. ~200–500 real episodes, no human reaction time needed.

---

## 5. Training recipe

Fine-tune SmolVLA (LeRobot) on the sim demos. Train two models that are identical except for one thing:
- **Model A (baseline):** images + proprio + language.
- **Model B (ours):** the same, plus the **5-number** live vector (XY p̂, XY v̂, Δ). Full ~12 when the filter is live.

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

*If G1 fails after the holes below are closed, the core hypothesis is wrong or the conditioning is ignored — stop and diagnose before building anything else.* This is the single most important experiment in the project, and it requires zero perception code.

**G1 is not a fair hypothesis test while any of these are true:** (1) Δ is sampled but not packed; (2) eval glue unsticks on any open (expert only unsticks at place) — 8-step still B was 20/20 glue and 15/20 never lifted; 50-step still was 18/20 because fewer replans; (3) policy `sm_state` is only `{0,4}` so the expert labeler cannot emit lift / slip / early-close; (4) `n_action_steps=8` without a hold latch (checkpoint default is 50; plan §3 already said 20–50). Close those, then re-run G1. Do not add a latent world model to “fix G1.”

**Two eval curves, same checkpoint.** (A) `n_action_steps=50` — can the net pick at all when the chunk can contain close→lift. (B) `n_action_steps=8` (~320 ms) **with a hold latch** — does the future vector help at speed. G1 pass/fail uses (B) after glued-no-lift is rare. Report (A) as sanity, not as the speed claim.

**Phase 2 — Real perception, still in sim.** Run detector + tracker + filter on Isaac's rendered camera streams. Measure prediction RMSE at 100/200/300 ms horizons vs. ground truth. Swap oracle → filter in B's input. **Gate G2:** prediction error @250 ms ≤ ~1–2 cm on-table; success drop from the swap <10 points. (The noise augmentation from Section 4 is what makes this gate cheap to pass.)

**Phase 3 — Clock discipline + reflex.** Add the timestamped executor, chunk blending, and reflex layer; inject artificial latency to stress-test. **Gate G3:** among failures previously tagged "too late / early close," ≥half are recovered; path length and jerk visibly drop; success at **20 cm/s** improves.

**Phase 4 — Real robot.** Calibration checklist: static-camera extrinsics to robot base (ArUco board); table plane by touching three points with the TCP (simplest and most accurate); wrist hand-eye (standard chessboard); latency measurement (gripper-wave trick, Section 2). Port tracker + filter (they should transfer almost unchanged — that is the point of explicit state). Collect ~300 real episodes with the filter-driven expert; fine-tune. **Gate G4:** ≥60–70% on sliding/rolling tasks at 10–20 cm/s.

**Phase 5 — Stretch (the "complex tasks" story).** (a) Upgrade the filter to IMM (a small committee of motion models — rolling / decelerating / ballistic / stopped — that votes); (b) uncertainty-driven behavior: when σ_p is large, *stage* at the intercept corridor with the gripper pre-opened and delay commitment instead of chasing the mean; (c) ballistic catches using wrist depth; (d) occlusion demo — the filter coasts through occluders with growing covariance *by construction*, which is the transparent version of AHEAD's flashiest result.

---

## 8. Evaluation protocol

**Primary curve:** success rate vs. object speed (**0 / 10 / 15 / 20 cm/s**). **Scout:** 20 trials per point. **Claim / gate:** ≥80 per point, with Wilson 95% intervals in the report — the flatness of this curve *is* the claim. 20 trials cannot resolve a 20-point G1 margin.

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
**Eval glue unsticks on open** → policy close/open chatter drops the object; expert only unsticks at place. Latch: stick on close+near; unstick only if open **and** (far or placed). Do not wait for the reflex layer to hide this.
**Δ omitted from the vector** → restore the 5th number before another speed-curve story.
**20-trial G1** → scout only; n≥80 to claim the gate.
**"Need a latent world model" (AHEAD / DynaWM / PhysMani)** → those papers also act on a **predicted future** and keep the VLA from inventing physics. Our cheap version is the Kalman + 5-number pack. Do not vendor their nets in Phase 1. See `foresight_revised_plan2.md`.
**"It's not end-to-end" reviewer critique** → the ablation ladder (Section 8) reframes the paper as *finding the minimal sufficient structure*, which is a stronger claim than another architecture.

---

## 10. What we deliberately do not build (and why)

No latent world model (a Kalman filter answers the same question in 200 lines, in 3D, with honest uncertainty). No AHEAD / DynaWM / PhysMani modules — steal their *habits* (freeze the VLA, condition on an explicit future, evaluate with enough trials), not their architectures. No optical flow network (the tracker + filter already yield velocity). No VLA trained from scratch (fine-tuning a 450M open model is days, not GPU-months). No 6D object pose (centroid + velocity suffices for grasping convex household objects; add orientation only when a task demands it). No 200K episodes (explicit conditioning should slash the data the policy needs — and we measure that claim rather than assume it). No off-table motion in Phase 1 (the plane constraint is what makes a single RGB camera a 3D sensor). No unfreezing the vision encoder to “make cameras work” while B-zero already proves the vector is the channel.

---

## 11. Next, concretely (after the 5k G1 scout)

Do **one hole at a time**. Do not combine latch + Δ + n_obs_steps + a world model into one re-run of 20×4.

1. **Latch (eval only, no train).** Stick on close+near; unstick only if open and (far or placed). Re-score still at `n_action_steps=8`. Expect glued-no-lift to collapse (50-step still already did this by committing the chunk).
2. **Pack Δ (5-D vector).** Retrain Model B on the same 5k. Keep A. Scout 20, then 80 if still looks alive.
3. **Labels + G1 script.** Policy taxonomy from glue+height; zero-out vs A; n≥80 to claim the gate.
4. **G1** on 8-step + latch + packed Δ. Side sanity: 50-step still. Only then perception (G2) or the overwrite buffer (Job 4) if moving objects still miss while still objects work.

Job 3 (reflex) is still the last-centimeter closer for **moving** objects. It is not the fix for “glued, never lifted” on a still apple.
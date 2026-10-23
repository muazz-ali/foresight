# Foresight — literature map for 8 research issues

Date: 2026-10-21. **97 papers**, all on arXiv. Every ID was checked against the arXiv API: the title and first author match the link.
Findings come from each paper's abstract (or a search summary of it). The papers were **not** read in full, so confirm any number in the paper before citing it.

Each section gives: what the field knows → the papers → our own numbers → the gap we could fill.
Our numbers come from `data/eval/PLAN_V2_RESULTS.md`, `foresight_docs/RESEARCH.md` and the probe and eval reports.

---

## 1. Shortcut / copycat learning

**Field:** Behaviour cloning picks whichever input predicts the next action most cheaply. A low-dimensional signal (proprioception, or the previous action) beats pixels or object state when it correlates with the label.

**Our numbers:** In `p3_throw`, the probe gain on the hand was ~1.0 and on p̂ only 0.02–0.08. The policy aimed at "hand + 2.5 cm" because the expert's hand trails its target by ~2 cm. V2 added 3 cm of hand noise (`policy/transforms.py:221-224`, `ee_noise_std: 0.03`). Gains became p̂ 0.96/0.85 and hand 0.01/0.06.

| # | Paper | Year / venue | Key finding | Why it matters to us |
|---|---|---|---|---|
| 1 | [Causal Confusion in Imitation Learning](https://arxiv.org/abs/1905.11979) — de Haan, Jayaraman, Levine | NeurIPS 2019 | Giving BC more information can make it worse. Fixed with targeted interventions. | Names our failure mode |
| 2 | [Fighting Copycat Agents in Behavioral Cloning from Observation Histories](https://arxiv.org/abs/2010.14876) — Wen et al. | NeurIPS 2020 | Policies trained on observation histories learn to repeat the previous action. Adversarially removing that information helps. | Our hand pose acts as a stand-in for the previous action |
| 3 | [Resolving Copycat Problems in Visual Imitation Learning via Residual Action Prediction](https://arxiv.org/abs/2207.09705) — Chuang et al. | ECCV 2022 | An architecture that removes the shortcut path (copycat-free memory) | Architectural fix instead of noise tuning |
| 4 | [Keyframe-Focused Visual Imitation Learning](https://arxiv.org/abs/2106.06452) — Wen et al. | ICML 2021 | Up-weights frames where the expert's action changes | Our approach→descend→close switch points |
| 5 | [Exploring the Limitations of Behavior Cloning for Autonomous Driving](https://arxiv.org/abs/1904.08980) — Codevilla et al. | ICCV 2019 | Shows the "inertia problem", which comes from causal confusion. Large seed-to-seed variance. | Report seed variance. Inertia ≈ copying the hand |
| 6 | [DART: Noise Injection for Robust Imitation Learning](https://arxiv.org/abs/1703.09327) — Laskey et al. | CoRL 2017 | Adds noise to the expert during collection. 62% better than BC on grasping in clutter. | We add noise at train time; DART adds it at collect time (recovery data) |
| 7 | [Object-Aware Regularization for Addressing Causal Confusion in Imitation Learning](https://arxiv.org/abs/2110.14118) — Park et al. | NeurIPS 2021 | A regularizer that keeps the policy from relying on a single object feature | Principled alternative to hand noise |
| 8 | [Shortcut Learning in Generalist Robot Policies: The Role of Dataset Diversity and Fragmentation](https://arxiv.org/abs/2508.06426) — Xing et al. | CoRL 2025 | Low diversity inside each sub-dataset plus gaps between sub-datasets cause shortcuts. Targeted augmentation reduces them in π0. | Our data has one object per scene: low diversity |
| 9 | [Shortcut Learning in Deep Neural Networks](https://arxiv.org/abs/2004.07780) — Geirhos et al. | Nat. Mach. Intell. 2020 | General framework and definition of shortcut learning | Vocabulary for the paper |
| 10 | [A Reduction of Imitation Learning … to No-Regret Online Learning (DAgger)](https://arxiv.org/abs/1011.0686) — Ross, Gordon, Bagnell | AISTATS 2011 | Querying the expert on the policy's own rollouts fixes compounding error | **Cheap for us**: our scripted expert can label policy rollouts. We have never used it this way. |
| 11 | [When would Vision-Proprioception Policies Fail in Robotic Manipulation?](https://arxiv.org/abs/2602.12032) — Lu et al. | 2026 | Proprioception lowers the loss faster, so it dominates training and suppresses vision during motion transitions. Fixed by gradient adjustment. | The **mechanism** behind our copycat |
| 12 | [Do You Need Proprioceptive States in Visuomotor Policies?](https://arxiv.org/abs/2509.18644) — Zhao et al. | 2025 | Removing the state input improves spatial generalization: height 0→85%, horizontal 6→64%. | Alternative to hand noise: drop hand XYZ, use relative actions |
| 13 | [Register-Routed Delayed Fusion: Rewiring Shortcut-Prone Observation Fusion in Visuomotor Imitation](https://arxiv.org/abs/2610.02813) — Long et al. | Oct 2026 | Delays fusion of compact (proprio) tokens with visual tokens to stop early shortcuts | Fusion design for the state projection |

**Gap:**
- **No standard metric** for which input a policy actually uses. Our probe ("shift one input, measure how far the aim follows") could be a contribution.
- The literature studies proprioception vs **vision**. Proprioception vs an **explicit object-state channel** is not studied.
- **Clock / phase shortcut:** no paper found that tests directly whether a policy times its actions from episode progress instead of the world. That is open, and cheap for us to test.
- **DAgger with our scripted expert** is an untried, cheap fix.

---

## 2. Language ignored

**Field (consensus 2025–26):** VLAs largely ignore the instruction whenever the scene alone determines the task. They follow vision priors and fall to ~0 when the instruction asks for something different.

**Our numbers:** In E3, V2 went to the numbers' object 40/40 times even when told to pick the other one. In E0, language content was below the 0.8 cm noise floor. There is one template, "Pick the {obj} and place it in the {ctr}" (`policy/features.py:110-113`), and one object per scene.

| # | Paper | Year | Key finding | Why it matters to us |
|---|---|---|---|---|
| 14 | [LIBERO-Plus: In-depth Robustness Analysis of VLA Models](https://arxiv.org/abs/2510.13626) — Fei et al. | 2025 | Models are largely insensitive to language and further tests show they ignore it. Swapping the goal object drops success to near zero. | Same failure as ours, at scale |
| 15 | [LIBERO-PRO: Towards Robust and Fair Evaluation of VLA Models Beyond Memorization](https://arxiv.org/abs/2510.03827) — Zhou et al. | 2025 | >90% on standard LIBERO collapses to **0.0%** under perturbation. Policies keep grasping with corrupted instructions or meaningless tokens. | Memorization, not grounding |
| 16 | [When Vision Overrides Language: Counterfactual Failures in VLAs (LIBERO-CF)](https://arxiv.org/abs/2602.17659) — Fang et al. | 2026 | Vision shortcuts from dataset bias. CAG is a training-free dual-branch guidance fix. | Benchmark design for our E3 |
| 17 | [Restoring Linguistic Grounding in VLA Models via Train-Free Attention Recalibration (ICBench)](https://arxiv.org/abs/2603.06001) — Zhang et al. | 2026 | "Linguistic blindness": policies keep acting when the instruction contradicts the scene | Diagnostic protocol |
| 18 | [Robust Skills, Brittle Grounding: Diagnosing Restricted Generalization in VLA Policies via Multi-Object Picking](https://arxiv.org/abs/2602.24143) — Emukpere et al. | 2026 | In **SmolVLA** and π0.5, the primitive stays reliable while instruction-following fails. Skill is decoupled from grounding, and more data doesn't close the gap. | **Same base model as ours** |
| 19 | [CAST: Counterfactual Labels Improve Instruction Following in VLAs](https://arxiv.org/abs/2508.13446) — Glossop et al. | 2025 | Policies learn to ignore language. Counterfactual relabeling fixes it. | Data-side fix |
| 20 | [CounterAlign: Counterfactual Supervision for VLA Models](https://arxiv.org/abs/2608.21740) — Kondoh et al. | 2026 | Pairs expert actions with mismatched instructions to create negative (offline RL) supervision with no new data | Cheap: relabel existing demos |
| 21 | [ConflictVLA-Bench: Behavioral Responses of VLAs to Premise Conflicts](https://arxiv.org/abs/2609.31792) — Hou et al. | 2026 | Across 8 VLAs, invalid premises cut goal completion by 17.3–56.2 points ("failed persistence") | Evaluation design |
| 22 | [VLA Grounder: Language-Conditioning Space Optimization for Black-Box VLA Models](https://arxiv.org/abs/2607.04517) — Shodiev et al. | 2026 | RL rewrites the instruction for a frozen VLA. Results depend heavily on phrasing. | Grounding outside the policy |
| 23 | [TAG: Target-Agnostic Guidance for Object-Centric Inference in VLAs](https://arxiv.org/abs/2603.24584) — Zhou et al. | 2026 | Errors are instance-level grounding failures (lands on the wrong instance). Contrasts predictions against an object-erased observation. | Distractor robustness |
| 24 | [Clutter-Robust VLA Models through Object-Centric and Geometry Grounding (OBEYED-VLA)](https://arxiv.org/abs/2512.22519) — Vo et al. | 2025 | A separate perception-grounding module produces object-centric observations. Stops over-grasping when the target is absent. | Same idea as our "Design A" (detector picks the track) |
| 25 | [Knowledge Insulating Vision-Language-Action Models](https://arxiv.org/abs/2505.23705) — Driess et al. | NeurIPS 2025 | Gradients from the action expert damage VLM knowledge and training speed. Insulating the backbone fixes it. | Context for our frozen VLM |
| 26 | [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818) — Brohan et al. | 2023 | Semantic generalization comes from web co-training | Baseline claim for "VLA" |

**Gap:** Every paper studies vision-only VLAs. Nobody studies language **when an explicit object-state channel exists**: there the state completely dominates (our E3). Two candidate fixes:
- Language picks which tracked object feeds the numbers (Design A, close to #22 and #24).
- Two-object contrastive pairs (#19, #20). These are cheap for us, because the scripted expert can record both targets in the same scene.

---

## 3. Visual input (pixels) ignored

**Field:** Visual sensitivity fades in deeper layers. Low-dimensional inputs crowd out vision. The evidence on frozen vs fine-tuned encoders is mixed.

**Our numbers:** The wrist image is never used (a grey wrist image still succeeds). The static image matters only for the place step (container). Aiming comes from the 4 numbers.

| # | Paper | Year | Key finding | Why it matters to us |
|---|---|---|---|---|
| 27 | [Look Before Acting: Enhancing Vision Foundation Representations for VLAs](https://arxiv.org/abs/2603.15618) — Luo et al. | 2026 | Across VLAs, sensitivity to visual tokens **decreases in deeper layers** during action generation | Explains the weak use of vision |
| 28 | [Probing a VLA Model for Symbolic States](https://arxiv.org/abs/2502.04558) — Lu et al. | 2025 | OpenVLA's layers encode object and action states (probe accuracy >0.90) | Probing method we could apply |
| 29 | [AVA-VLA: Improving VLAs with Active Visual Attention](https://arxiv.org/abs/2511.18960) — Xiao et al. | 2025 | Single-frame VLAs miss history. A recurrent belief re-weights visual tokens. | Single-frame input is also ours |
| 30 | [VLM4VLA: Revisiting Vision-Language-Models in VLA Models](https://arxiv.org/abs/2601.03309) — Zhang et al. | 2026 | A VLM's general skill poorly predicts VLA performance. Freezing the vision encoder hurts on CALVIN and SIMPLER. | We freeze SigLIP |
| 31 | [Enhancing Generalization in VLAs by Preserving Pretrained Representations](https://arxiv.org/abs/2509.11417) — Grover et al. | 2025 | Two encoders: one frozen, one trainable. Plain fine-tuning damages the pretrained features. | Middle path between frozen and unfrozen |
| 32 | [MAPS: Module-Wise Proximity Scheduling for VLA Generalization](https://arxiv.org/abs/2511.19878) — Huang et al. | 2025 | Keep the vision encoder close to its pretrained weights and let the action layers adapt | Same |
| 33 | [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246) — Kim et al. | 2024 | Fine-tuning the vision encoder was needed for good VLA performance | Counter-evidence to freezing |

Also relevant: #11, #12, #13 (proprioception suppresses vision) and #66 (a diffusion policy outputs training actions even for cat and dog images).

**Gap:** For us, ignoring pixels for aiming may be the *right* division of labour. But then nothing rescues the policy when the tracker fails near the grasp (2–5 cm error). No paper studies **when a policy should shift trust from an explicit state channel to pixels.** Our state message already carries covariance (`policy/features.py` ORACLE_NAMES), but it is never packed into the policy input. Uncertainty-gated fusion is untested.

---

## 4. Perception (tracking, velocity, occlusion)

**Field:** Dynamic-manipulation VLAs learn motion from pixels (optical flow, latent forecasting). Classical systems use filters plus pose trackers. Event cameras reach 13–15 m/s in catching.

**Our numbers:** Tracker input gets 75% vs 92.5% with sim truth at 0.20 m/s, a 17-point cost. Tracker error is ~0.4 cm far from the hand and 2–5 cm near the grasp. Offline, a wrist RGB-D fit gives a 1–3 mm median. The `--held-from-hand` workaround (`scripts/eval_policy.py:755-785`) covers the held phase.

| # | Paper | Year | Key finding | Why it matters to us |
|---|---|---|---|---|
| 34 | [FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects](https://arxiv.org/abs/2312.08344) — Wen et al. | CVPR 2024 | 6D tracking of novel objects from a CAD model or a few reference images, no fine-tuning | Off-the-shelf tracker candidate |
| 35 | [DynamicPose: Real-time 6D Pose Tracking for Fast-Moving Cameras and Objects](https://arxiv.org/abs/2508.11950) — Liang et al. | 2025 | Kalman filter guided by visual-inertial odometry, for fast motion | Wrist camera moving fast |
| 36 | [Correlation-Aware Dual-View Pose and Velocity Estimation for Dynamic Robotic Manipulation](https://arxiv.org/abs/2510.05536) — Zarei et al. | 2025 | Fuses eye-in-hand and eye-to-hand cameras with adaptive EKFs on a Lie group | **Directly our gripper-occlusion case** |
| 37 | [KARL: Kalman-Filter Assisted RL for Dynamic Object Tracking and Grasping](https://arxiv.org/abs/2506.15945) — Boyalakuntla et al. | 2025 | Kalman filter plus RL, tested with tracking loss on a real UR5e | Filter + learned policy, like our stack |
| 38 | [EV-Catcher: High-Speed Object Catching Using Low-latency Event-based Neural Networks](https://arxiv.org/abs/2304.07200) — Wang et al. | 2023 | 81% catches up to 13 m/s on a Jetson NX | Sensor path for throws |
| 39 | [Event-based Agile Object Catching with a Quadrupedal Robot](https://arxiv.org/abs/2303.17479) — Forrai et al. | ICRA 2023 | 83% at up to 15 m/s, 100 Hz | Same |
| 40 | [Fast Trajectory End-Point Prediction with Event Cameras](https://arxiv.org/abs/2302.13796) — Monforte et al. | 2023 | 80% at 13 m/s by predicting the end point | Predict-then-act |
| 41 | [Intercepting the Future: Latent-Space Predictive World Model for Dynamic VLA Manipulation (AHEAD)](https://arxiv.org/abs/2606.02486) — Najam Syed et al. | 2026 | Wraps a frozen OpenVLA with a 4.9M-parameter latent world model driven by optical-flow velocity and acceleration. 79–97% vs 31–58% baseline over 20 dynamic sim scenarios. | **Rival**: learns motion from pixels |
| 42 | [Towards Generalizable Robotic Manipulation in Dynamic Environments (DOMINO / PUMA)](https://arxiv.org/abs/2603.15620) — Fang et al. | 2026 | Single-frame VLAs fail on moving targets. 35 tasks and 110K trajectories. Historical optical flow gives +6.3% absolute. | **Rival** benchmark |
| 43 | [PhysMani: Physics-principled 3D World Model for Dynamic Object Manipulation](https://arxiv.org/abs/2607.01938) — Yun et al. | 2026 | 3D Gaussian world model with a divergence-free velocity field, plus a future-aware policy | **Rival**: physics prior |
| 44 | [Object-Centric Conditioning for Visuomotor Flow Matching (SlotFlow)](https://arxiv.org/abs/2609.24155) — Li et al. | 2026 | "What" (semantic) and "where" (image-plane) object cues. Robust to distractors and spatial shifts. | Object-centric input, close to our numbers |
| 45 | [Object-Centric Residual RL for Zero-Shot Sim-to-Real VLA Enhancement](https://arxiv.org/abs/2606.18953) — Kim et al. | 2026 | Object poses as a compact observation for sim-to-real residual RL on a frozen VLA | Explicit pose transfers sim→real better than images |
| 46 | [Confidence Calibration in Vision-Language-Action Models](https://arxiv.org/abs/2507.17383) — Zollo et al. | 2025 | VLA confidence vs actual success. Prompt ensembles and Platt scaling. | Our P_s calibration work |
| 47 | [Physics-Aware Object-Centric SE(3) Forecasting for Real-Time Proactive Dynamic Grasping](https://arxiv.org/abs/2610.17843) | Oct 2026 | Argues the object's rigid-body state (SE(3) pose) is all the robot needs to predict. Forecasts from short pose histories, with a reachability-aware interception selector. | **Closest to our thesis** ("PAF-Grasp" in old notes) |

**Gap:**
- Almost all dynamic-VLA papers (#41–43) learn motion from pixels. Few compare **explicit filtered state vs pixel forecasting on the same data budget**. Our E5 (69% vs 3%) and the CRLB argument fit there.
- Gripper occlusion in the last 10 cm: dual-view fusion (#36) and wrist RGB-D are the documented paths. Neither has been shown inside a VLA loop.

---

## 5. Latency (robot and model pipeline)

**Field (very active, 2025–26):** Asynchronous chunking, prediction from stale inputs, faster flow sampling and streaming KV-caches. Most methods fix *robot-state* staleness and jerky chunk boundaries. Few handle *object* motion during the delay.

**Our numbers:**
- One chunk takes 68–164 ms to infer. The 8-step chunk is 0.32 s of blind execution.
- E4: adding 100 ms of delay drops success from 75% to 42%. The close layer brings it back to 70%.
- Raising the fixed Δ does nothing, because Δ is fixed at 0.25 s (`sim/phase0_cfg.yaml:143`).

| # | Paper | Year | Key finding | Why it matters to us |
|---|---|---|---|---|
| 48 | [Real-Time Execution of Action Chunking Flow Policies (RTC)](https://arxiv.org/abs/2506.07339) — Black, Galliker, Levine | NeurIPS 2025 | Freezes the actions that will run during the delay and inpaints the rest. No retraining. | Drop-in for SmolVLA flow |
| 49 | [Training-Time Action Conditioning for Efficient Real-Time Chunking](https://arxiv.org/abs/2512.05964) — Black et al. | 2025 | Simulates the delay during training and conditions on the action prefix | **= our "Δ as trained input" plan** |
| 50 | [VLASH: Real-Time VLAs via Future-State-Aware Asynchronous Inference](https://arxiv.org/abs/2512.01031) — Tang et al. | 2025 | Rolls the robot state forward with the previous chunk before inferring | Robot side only |
| 51 | [FutureRTC: Anticipatory-Conditioned Action Chunking](https://arxiv.org/abs/2607.24008) — Jiang et al. | 2026 | Predicts execution-time *observations* as well as states, because VLASH ignores the stale image | Scene motion during delay |
| 52 | [DEFLECT: Temporal Counterfactual Preference Learning for Delay-Robust Asynchronous VLAs](https://arxiv.org/abs/2605.19294) — Zhu et al. | 2026 | Prefers the chunk computed from the execution-time observation over the stale one. Offline, no inference cost. | Post-training option |
| 53 | [Action ControlNet: A Lightweight Delay-Aware Adapter](https://arxiv.org/abs/2606.25985) — Guo et al. | 2026 | Uses the executed motion suffix as a residual condition for a frozen action head | Small adapter, no retrain |
| 54 | [FASTER: Rethinking Real-Time Flow VLAs](https://arxiv.org/abs/2603.19199) — Lu et al. | 2026 | Reaction time is uniformly distributed, set by time-to-first-action **and execution horizon**. Horizon-aware denoising schedule. | Our 8-step horizon is part of our delay |
| 55 | [Reflex: Real-Time VLA Control through Streaming Inference](https://arxiv.org/abs/2607.14695) — Guo, Liu | ICML 2026 | KV-cache streaming. 2.58× speedup, 50 Hz, up to 54% lower reaction latency. | Inference speed-up path |
| 56 | [RAVEL: Asynchronous Rolling Inference for Flow-Based VLAs](https://arxiv.org/abs/2609.34170) — Chen et al. | 2026 | Near-term actions run after one denoising step from a rolling buffer | Same |
| 57 | [FlashVLA: Streaming Action Decoding for Fast and Asynchronous VLA Inference](https://arxiv.org/abs/2608.27384) — Li et al. | 2026 | Streaming buffer of chunks at different noise levels | Same |
| 58 | [HiRT: Enhancing Robotic Control with Hierarchical Robot Transformers](https://arxiv.org/abs/2410.05273) — Zhang et al. | 2024 | Slow VLM plus fast policy. Dynamic tasks 48→75%. | Slow planner + fast servo split |
| 59 | [Asynchronous Fast-Slow VLA Policies for Whole-Body Manipulation](https://arxiv.org/abs/2512.20188) — Zou et al. | 2025 | Truly asynchronous fast/slow paths. Argues HiRT-style designs still run in lock-step. | Same |
| 60 | [Bidirectional Decoding: Improving Action Chunking via Guided Test-Time Sampling](https://arxiv.org/abs/2408.17355) — Liu et al. | 2024 | Chunking cuts reactivity; resampling at test time restores it | Chunk length vs reactivity |
| 61 | [SmolVLA: A VLA Model for Affordable and Efficient Robotics](https://arxiv.org/abs/2506.01844) — Shukor et al. | 2025 | Our base model. Asynchronous inference stack, consumer-GPU deployment. | Base |
| 62 | [DynamicVLA: A VLA Model for Dynamic Object Manipulation](https://arxiv.org/abs/2601.22153) — Xie et al. | 2026 | 0.4B model with a convolutional encoder. Continuous inference, and drops latency-invalid action prefixes. 200K sim episodes. | **Main rival**: ~47% on 0–1 m/s and 226 ms/chunk per our notes |
| 63 | [Reflex: Fast and Predictive VLAs for Reaction-Critical Manipulation (ReflexBench)](https://arxiv.org/abs/2608.14379) — Chen et al. | 2026 | Benchmark with configurable latency. Latent future prediction and multi-frame fusion. | **Rival**; benchmark we could report on |
| 64 | [F2F-AP: Flow-to-Future Asynchronous Policy for Real-time Dynamic Manipulation](https://arxiv.org/abs/2604.02408) — Wei et al. | 2026 | Predicted object flow builds a future observation. Horizon H = ⌈total latency / dt⌉. | Pixel-side version of our Δ |

**Gap:** Delay handling for **object** motion exists only as pixel forecasting (#51, #64, #41). No paper conditions on an **explicit object-state forecast with a trained Δ**, and few report success over a **speed × latency** grid. Our E4 grid, plus training-time Δ (the idea of #49 applied to object state), fills that.

---

## 6. Interpolation vs extrapolation

**Field:** Imitation policies mostly interpolate or memorize. Extrapolation needs structure: anchors plus residuals, counterfactual augmentation, or object-frame actions.

**Our numbers:** 0.05 m/s (inside the training range) gets 95–100%. 0.25 m/s (outside it) collapses to 38%, or 55% with the close rule.

| # | Paper | Year | Key finding | Why it matters to us |
|---|---|---|---|---|
| 65 | [Static In, Dynamic Out: Counterfactual Action Augmentation for Moving Object Manipulation (SIDO)](https://arxiv.org/abs/2607.27890) — Shin et al. | 2026 | Trained on **static** demos only. Moves the object to a counterfactual future position and morphs the action chunk to keep the hand–object pose. A pose predictor supplies the future position at test time. | **Closest method to ours.** Recipe for making faster-speed data |
| 66 | [Demystifying Diffusion Policies: Action Memorization and Simple Lookup Table Alternatives](https://arxiv.org/abs/2505.05787) — He et al. | ICLR 2026 | A diffusion policy behaves like a lookup table: it outputs training actions even for out-of-distribution images (cats, dogs) | Why extrapolation fails |
| 67 | [Real-time Motion Generation and Data Augmentation for Grasping Moving Objects with Dynamic Speed and Position Changes](https://arxiv.org/abs/2309.12547) — Yamamoto et al. | 2023 | Learns the position–velocity relationship and generalizes to **untaught speeds** on a conveyor | Direct counter-example to our 0.25 collapse |
| 68 | [Bilinear Flow Policy: Distributional Extrapolation for Goal-Conditioned Visuomotor Imitation](https://arxiv.org/abs/2610.05765) — Jung et al. | Oct 2026 | Treats an unseen case as a retrieved training anchor plus a residual. 2.63× out-of-distribution success. | Speed as "anchor + residual" |
| 69 | [Out-of-Distribution Recovery with Object-Centric Keypoint Inverse Policy](https://arxiv.org/abs/2411.03294) — Gao et al. | 2024 | Steers the robot back into the training distribution. +77.7% out of distribution. | Recovery layer |
| 70 | [Inductive Generalization for Robotic Manipulation](https://arxiv.org/abs/2606.20999) — Macaluso et al. | 2026 | Test along progressively harder out-of-distribution axes, not by interpolation | Protocol for our speed axis |
| 71 | [SAIL: Faster-than-Demonstration Execution of Imitation Learning Policies](https://arxiv.org/abs/2506.11948) — Arachchige et al. | CoRL 2025 | Runs policies up to 4× (sim) and 3.2× (real) faster than the demos, with latency-aware scheduling | Running faster than the teacher |
| 72 | [Dynamic Grasping with a Learned Meta-Controller](https://arxiv.org/abs/2302.08463) — Jia et al. | 2023 | RL tunes the look-ahead time and planning budget during the episode. Up to +28%. | Adaptive Δ |
| 73 | [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137) — Chi et al. | RSS 2023 | Generative action chunks | Foundation |
| 74 | [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT)](https://arxiv.org/abs/2304.13705) — Zhao et al. | RSS 2023 | Action chunking and temporal ensembling | Foundation of chunking |
| 75 | [π0: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164) — Black et al. | 2024 | Flow-matching action expert on a VLM (SmolVLA follows this design) | Foundation |

**Gap:** No paper shows **speed extrapolation for a learned grasp policy** with numbers. The best-supported candidate fix for our 0.25 m/s collapse is SIDO-style counterfactual augmentation in the object frame plus object-relative actions (#65, #67).

---

## 7. Orientation and grasp failures

**Field:** Dynamic grasping needs motion-aware grasp choice, such as aligning with the object's narrow axis. VLA failure studies find grasping is the dominant failure, often from sub-centimetre misalignment. Catching systems survive by being compliant.

**Our numbers:**
- 33% of the policy's first closes never lift.
- At 0.20 m/s the main failure is "held but no lift": the close is ~1.3 cm too high and the hand doesn't match the object's speed.
- 3/40 failures in the confirmation run were the object slipping out during carry.
- Pads along the motion help the expert at 0.30 m/s (55→78%).
- Slack is ~5 mm per side.

| # | Paper | Year | Key finding | Why it matters to us |
|---|---|---|---|---|
| 76 | [Dynamic Grasping with Reachability and Motion Awareness](https://arxiv.org/abs/2103.10562) — Akinola et al. | IROS 2021 | Grasp quality learned as a function of object motion. Aligns the gripper with the narrow axis while the object moves. Motion prediction is key. | Same as our pads-along result |
| 77 | [GraspARL: Dynamic Grasping via Adversarial Reinforcement Learning](https://arxiv.org/abs/2203.02119) — Wu et al. | 2022 | UKF prediction error grows with the object/robot speed ratio. An adversarial mover improves generalization. | Error grows with speed |
| 78 | [GAP-RL: Grasps As Points for RL Towards Dynamic Object Grasping](https://arxiv.org/abs/2410.03509) — Xie et al. | 2024 | Grasp-as-point representation that handles varying translation and rotation speeds | Orientation under motion |
| 79 | [EARL: Eye-on-Hand RL for Dynamic Grasping with Active Pose Estimation](https://arxiv.org/abs/2310.06751) — Huang et al. | 2023 | The wrist camera actively keeps the object in view | Wrist-camera loop |
| 80 | [Multi-fingered Dynamic Grasping for Unknown Objects](https://arxiv.org/abs/2310.17923) — Burkhardt et al. | 2023 | Estimating rotational velocity **did not** improve success (singularities, collisions) | Limits of orientation tracking |
| 81 | [DynaMOMA: Instantaneous Prediction of Grasp Poses for Mobile Manipulation of Dynamic Objects](https://arxiv.org/abs/2606.25295) — Yu et al. | 2026 | Diffusion predicts time-consistent short-horizon grasp trajectories | Grasp-pose prediction |
| 82 | [Adaptive Grasping of Moving Objects in Dense Clutter](https://arxiv.org/abs/2502.05916) — Chen et al. | 2025 | Maps static grasp poses onto predicted future positions, from a single in-hand RGB-D camera | Same factorization as ours |
| 83 | [Self-Supervised Learning of Grasping Arbitrary Objects On-the-Move](https://arxiv.org/abs/2411.09904) — Kiyokawa et al. | 2024 | Self-supervised dynamic grasping | Data without a scripted expert |
| 84 | [Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes](https://arxiv.org/abs/2103.14127) — Sundermeyer et al. | ICRA 2021 | Learned 6-DoF grasps, >90% real success on unseen objects | Grasp-pose module for odd shapes |
| 85 | [Catch It! Learning to Catch in Flight with Mobile Dexterous Hands](https://arxiv.org/abs/2409.10319) — Zhang et al. | 2024 | ~80% in sim, deployed on a real robot | Catching ceiling |
| 86 | [Modular Neural Network Policies for In-Flight Object Catching](https://arxiv.org/abs/2312.13987) — Hu et al. | 2023 | Five modules (supervised plus RL) generalize to household objects | Modular vs end-to-end |
| 87 | [CushionCatch: A Compliant Catching Mechanism for Mobile Manipulators](https://arxiv.org/abs/2409.14754) — Chen et al. | 2024 | Compliant, cushioned catching | Impact / slip (our carry slips) |
| 88 | [Failing Forward: Adaptive Failure-Informed Learning for VLAs](https://arxiv.org/abs/2605.08434) — Zheng et al. | 2026 | Grasp failure is **267/424 (63%)** of failures. Trains on failed rollouts as negative guidance. | Same failure share; method to use our failed rollouts |
| 89 | [Experiences from Benchmarking VLA Models for Robotic Manipulation](https://arxiv.org/abs/2511.11298) — Zhang et al. | 2025 | Policies reach roughly the right place but fail on a small misalignment at the grasp. Sub-centimetre pose offsets. | Our 5 mm slack problem |
| 90 | [Benchmarking VLA Models on SO-101: Failure and Recovery Analysis](https://arxiv.org/abs/2606.08881) — Yu et al. | 2026 | π0.5, **SmolVLA**, Wall-X and ACT on real low-cost arms. Grasp instability persists. | Real-hardware failure taxonomy for our model |
| 91 | [Grasp-Then-Plan with Failure Attribution](https://arxiv.org/abs/2606.03385) — Xu et al. | 2026 | Learned attribution separates grasp failures from planning failures | Failure labelling (our eval labels) |

**Gap:** VLAs output position plus open/close, with no reasoning about **pad direction vs object velocity**. No VLA paper was found that conditions on orientation for moving objects. Our pads-along expert result and the capture-region P_s are candidate contributions. Carry slip under dynamic grasping is barely studied.

---

## 8. Generalization (objects, scenes)

**Field:** Generalization scales with *diversity* (number of environments and objects), not with repeated demos. Benchmarks expose memorization.

**Our numbers:** Peach (unseen) 100/68% vs tangerine 98/70%. Can −15 points. Tall bottle fails. Apple and cup are too wide for the gripper.

| # | Paper | Year | Key finding | Why it matters to us |
|---|---|---|---|---|
| 92 | [Decomposing the Generalization Gap in Imitation Learning for Visual Robotic Manipulation](https://arxiv.org/abs/2307.03659) — Xie et al. | ICRA 2024 | Ranks 11 factors of variation by difficulty, consistently in sim and real | Which factors to vary first |
| 93 | [Data Scaling Laws in Imitation Learning for Robotic Manipulation](https://arxiv.org/abs/2410.18647) — Lin et al. | ICLR 2025 | Power law in the number of environments and objects. Extra demos per object add little after a threshold. | Size data by diversity |
| 94 | [What Matters in Building VLA Models for Generalist Robots](https://arxiv.org/abs/2412.14058) — Li et al. | 2024 | Systematic study of backbone, structure and data choices | Design choices |
| 95 | [From Intention to Execution: Probing the Generalization Boundaries of VLA Models](https://arxiv.org/abs/2506.09930) — Fang et al. | 2025 | VLA generalization is narrower than claimed. Current evaluations are not enough. | Evaluation framing |
| 96 | [ObjectVLA: End-to-End Open-World Object Manipulation Without Demonstration](https://arxiv.org/abs/2502.19250) — Zhu et al. | 2025 | New objects from vision-language pairs, without robot demos | Object transfer |
| 97 | [Open X-Embodiment: Robotic Learning Datasets and RT-X Models](https://arxiv.org/abs/2310.08864) | 2023 | Cross-embodiment pretraining data | SmolVLA pretraining context |

Also relevant: #8 (fragmentation causes shortcuts), #14 and #15 (collapse under perturbation), #33 (OpenVLA).

**Gap:** Our explicit state carries only **position and velocity, with no size or shape**, so transfer depends on shape (can, bottle). Adding object extent (from the wrist ellipsoid fit) to the state is untested anywhere.

---

## Closest rivals (all 2026) — what they don't do

| Paper | What they do | What they don't do (our angle) |
|---|---|---|
| SIDO #65 | Future pose → goal-conditioned policy, trained from static demos | No VLA, no latency study, no test that the policy uses the state |
| SE(3) forecasting #47 | Object-centric rigid-body forecast + interception selector | Not a learned visuomotor policy |
| DynamicVLA #62 | Latency-aware 0.4B VLA, 200K episodes | Learns motion implicitly; ~47% on 0–1 m/s (our notes) |
| AHEAD #41 | Latent world model on frozen OpenVLA | Pixel forecasting with an extra model; reported in sim only |
| ReflexVLA #63, DOMINO #42, F2F-AP #64, PhysMani #43 | Benchmarks and pixel or flow forecasting | No explicit filtered state; no same-budget no-state baseline |

**What looks open across all 8 issues:**
1. A standard "which input does the policy use" probe for VLAs (our swap/shift tests).
2. Explicit state vs pixel forecasting at the same data budget (our E5).
3. Delay handling for **object** motion with a trained Δ (#49's idea on object state).
4. Speed extrapolation for learned grasping (SIDO-style augmentation).
5. Language selecting *which* tracked object feeds the policy.
6. Uncertainty-gated trust between the state channel and pixels near the grasp.

## Not found (searched, no direct paper)
- A paper that directly tests a **clock / phase shortcut** (the policy timing itself from episode progress).
- An orientation-aware VLA for moving objects.
- A soft or caging gripper capture-region analysis for moving targets. Only Maeda-group caging work turned up, and none of it is on arXiv with moving targets.

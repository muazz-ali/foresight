# Robots Without a Screen
## How to spawn, physics-simulate, and control a robot in Isaac Sim — fully headless

> Slide-by-slide script for a PowerPoint tutorial.
> `## Slide N` = one slide. **Title** = heading · bullets = slide body · `Notes:` = speaker notes ·
> code blocks = code panels. Real APIs from the **DynamicVLA** project, cross-checked against
> NVIDIA's official Isaac Sim / Isaac Lab docs. Target: after this talk, anyone can set up
> their environment and *play with a robot without ever opening the GUI*.

---

## Slide 1 — Title / The Hook

**"You have a robot and a server with no monitor. Can you still make it move, grasp, and record what it sees?"**

- Yes — and it's how serious robotics is actually done at scale
- Today: spawn a robot → turn on physics → drive it → capture data — **zero GUI**
- Live proof from a real project (DynamicVLA) + NVIDIA's official docs

Notes: Open with the problem everyone hits: cloud GPUs, SSH boxes, CI, and multi-GPU data farms have no display. The GUI is a *debugging convenience*, not a requirement. By the end you'll run a robot from a terminal and get files back. Keep the promise concrete: "spawn, physics, drive, record — no window."

---

## Slide 2 — Why headless is the default, not the exception

**The GUI is the exception. Headless is production.**

- Servers/cloud have no display; a window would just waste a GPU
- Automation & CI can't click buttons
- Batch data generation: run many sims in parallel, unattended, for hours
- Reproducible: everything is a script + flags, not mouse clicks

Notes: Reframe the audience's instinct. People think "simulator = window." In robotics research and data generation, 99% of compute-hours are headless. The window is only for the first day of eyeballing a scene. Everything valuable — training data, evaluations, sweeps — happens without one.

---

## Slide 3 — What Isaac Sim actually *is* (mental model)

**Isaac Sim = an Omniverse "Kit" app = engine + physics + renderer**

- **Kit runtime** — the application core (loads extensions from a `.kit` "experience" file)
- **PhysX** — the physics engine (gravity, contacts, joints, solvers)
- **RTX renderer** — turns the 3D scene into images
- **USD** — the scene/asset file format (robots, tables, objects)

> Headless simply means: **start the same engine, don't attach a window.**
> Physics still runs. Rendering can still run — just off-screen.

Notes: This is the key insight that kills the confusion later. Headless doesn't disable physics or even rendering. It only removes the on-screen viewport. PhysX doesn't care if there's a window; the RTX renderer can write into an off-screen buffer you read as a numpy array. Say it plainly: "no window ≠ no rendering."

---

## Slide 4 — Two knobs you must understand

**`headless` and "cameras" are independent switches**

| Want... | GUI window? | Rendering on? | How |
|---|---|---|---|
| Debug locally | Yes | Yes | *(default, no flags)* |
| Physics only, fast | No | No | `--headless` |
| Physics **+ images**, no window | No | Yes | `--headless --enable_cameras` |
| Watch a headless run remotely | No (streamed) | Yes | `--livestream 2` |

Notes: The single most common beginner bug: `--headless` alone and then "why are my camera images black/empty?" Because rendering is OFF. You must add `--enable_cameras` to get off-screen rendering. And if you *do* want to watch it live from another machine, `--livestream` streams the viewport over WebRTC while still running headless on the host.

---

## Slide 5 — THE golden rule (write it on your hand)

**Launch the app BEFORE importing any `omni.*` / `isaaclab.*` module**

```python
# 1) launch first
from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})

# 2) ONLY NOW import Omniverse / Isaac Lab
import omni.usd
import isaaclab.sim as sim_utils
```

- Isaac's modules only exist *after* the Kit runtime boots
- Break this and you get: `Modules were loaded before SimulationApp was started`

Notes: This trips up literally everyone once. The Isaac/Omniverse Python modules are provided by the running Kit app — they aren't importable until the app is up. So the launch call must be the first real line, and every heavy import comes after (in this repo you'll see `# noqa: E402` on those imports, or imports deferred inside functions). Official NVIDIA docs state the same: "various dependency modules of Isaac Sim are only available after the simulation app is running."

---

## Slide 6 — Two doors into the simulator

**Door A: `SimulationApp` (raw)  ·  Door B: `AppLauncher` (Isaac Lab, recommended)**

```python
# Door A — raw Isaac Sim. Headless is hard-coded.
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "enable_cameras": True})
```

```python
# Door B — Isaac Lab. Headless comes from CLI flags / env vars.
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)     # adds --headless, --enable_cameras, --livestream...
args = parser.parse_args()
app_launcher = AppLauncher(args)              # <-- sim starts here
sim_app = app_launcher.app
```

Notes: Door A (raw) is great for tiny utilities — open a USD, bake collisions — where you just want a headless app with no CLI. Door B is what you use for anything real: `AppLauncher` wraps `SimulationApp` and, crucially, gives you `--headless`/`--enable_cameras`/`--livestream` **for free** via `add_app_launcher_args`. The DynamicVLA pipeline uses Door B everywhere.

---

## Slide 7 — Flags, env vars, and the auto-selected "experience"

**You set the intent; Isaac Lab picks the right `.kit` file**

- CLI flags: `--headless`, `--enable_cameras`, `--livestream {0,1,2}`, `--device cuda:0`
- Equivalent env vars: `HEADLESS=1`, `ENABLE_CAMERAS=1`, `LIVESTREAM=2`
- `LIVESTREAM={1,2}` **forces** headless (you watch via stream, not a local window)
- Isaac Lab maps your combo → an experience `.kit` file automatically:

| headless | cameras | experience file loaded |
|---|---|---|
| ✔ | ✔ | `isaaclab.python.headless.rendering.kit` |
| ✔ | ✘ | `isaaclab.python.headless.kit` |
| ✘ | ✔ | `isaaclab.python.rendering.kit` |

Notes: You almost never name a kit file yourself — the flag combination selects it. Confirm from a real run log: with `--headless --enable_cameras` you'll see `Loading experience file: .../isaaclab.python.headless.rendering.kit`. If you accidentally see plain `headless.kit`, you forgot `--enable_cameras`. Env vars are handy for containers/CI where you don't control argv.

---

## Slide 8 — Set up your environment (one-time)

**Get the stack installed and pick the right Python**

- Isaac Sim **4.5** + Isaac Lab **2.2.1**, Python 3.10 (this project's versions)
- Use a dedicated conda env for simulation; never mix with your training env
- Extra deps used here: `pip install shapely pyzmq h5py`

```bash
# the simulation interpreter used for every command in this deck
conda activate dynamicVLA_isaac
python -c "import isaaclab; print('Isaac Lab OK')"
```

Notes: 90% of "it won't import isaaclab" problems are the wrong conda env or a broken editable-install symlink. Keep one env for sim, one for training. Verify the import works headless before writing any scene code. NVIDIA's install guide is the canonical reference; this project pins 4.5 / 2.2.1.

---

## Slide 9 — Step 1: create a world and turn on physics

**`SimulationContext(dt)` = the physics clock**

```python
import isaaclab.sim as sim_utils

sim_cfg = sim_utils.SimulationCfg(dt=0.04, device="cuda:0")   # 0.04 s = 25 Hz
sim = sim_utils.SimulationContext(sim_cfg)

# a floor and a light so things don't fall through / render black
sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=600.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=600.0))
```

- `dt` is the physics timestep — smaller = more accurate, slower
- Physics is fully active headless; no window needed to simulate gravity/contacts

Notes: `SimulationContext` is the beating heart — it owns the PhysX timeline. `dt=0.04` means the world advances 40 ms per step (25 Hz), matching this project's config. Ground plane + dome light are the minimum: without a floor objects fall forever; without a light your future camera images are black. All of this is display-free.

---

## Slide 10 — Physics knobs that matter (PhysX)

**Tune the solver so grasps don't explode or slip**

```python
# from EnvCfg.__post_init__ in this project
self.sim.dt = 0.04                                   # 25 Hz
self.sim.physx.bounce_threshold_velocity = 0.01
self.sim.physx.friction_correlation_distance = 0.00625
self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 4 * 1024 * 1024
```

- Solver iteration counts, contact/rest offsets, friction → grasp stability
- GPU pair capacities → avoid overflow with many contacts / many envs

Notes: You don't need to memorize these, but know they exist: contact-rich tasks (grasping) are sensitive to solver settings. Too-low iterations or wrong offsets and the fingers "explode" the articulation or the object squirts out. These are set once in config and apply identically headless. Point: headless gives you *more* reason to get physics right, because no one's watching to catch a glitch.

---

## Slide 11 — Step 2: spawn a robot (the `ArticulationCfg`)

**A robot = a USD asset + physics props + initial pose + actuators**

```python
from isaaclab.assets.articulation import ArticulationCfg, Articulation
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils

robot_cfg = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path="RB5_850E/rb5_850e.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=12)),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={"elbow": 2.0944, "wrist1": -0.6981, "wrist2": 1.5708}),   # table-ready pose
    actuators={"arm": ImplicitActuatorCfg(joint_names_expr=["base","shoulder","elbow","wrist1","wrist2","wrist3"],
        effort_limit_sim=150.0, velocity_limit_sim=3.14, stiffness=10000.0, damping=1000.0)},
)
robot = Articulation(robot_cfg)     # spawns it into the stage
```

Notes: Walk through the four parts. **spawn** = which USD + its rigid/collision/articulation physics. **init_state** = the joint angles it starts at (here a "table-ready" grasp pose, not the near-singular all-zeros). **actuators** = the motors: `stiffness`/`damping` are the PD gains, `effort_limit`/`velocity_limit` cap force and speed. This one config object is literally "set up robot + enable its physics." No GUI involved — it's declarative.

---

## Slide 12 — Add a gripper, add cameras (still just config)

**Grippers are extra joints; cameras are sensors you attach**

```python
# gripper = finger joints with their own actuator + open/close targets
"gripper": ImplicitActuatorCfg(joint_names_expr=["joint7","joint8"],
    effort_limit_sim=10.0, velocity_limit_sim=0.2, stiffness=2e3, damping=6e2, friction=50)
# open -> {joint7:+0.035, joint8:-0.035} ;  close -> {0.0, 0.0}
```

```python
# a camera that renders off-screen (works headless with --enable_cameras)
from isaaclab.sensors import Camera, CameraCfg
cam = Camera(CameraCfg(prim_path="/World/Robot/.../GripperCamera",
    height=360, width=480, data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=6.0, clipping_range=(0.001, 5.0)),
    offset=CameraCfg.OffsetCfg(pos=(0,0.16,0), rot=(...), convention="opengl")))
```

Notes: A parallel gripper is just two prismatic finger joints with their own actuator gains; "open/close" is a pair of target positions. Cameras are sensors you place with a pose and intrinsics — mounted on a link (wrist/gripper cam) or fixed in the world. With `--enable_cameras` these render into GPU buffers off-screen. This is how you "give the robot eyes" with no monitor attached.

---

## Slide 13 — Step 3: the heartbeat (reset, then step)

**`sim.reset()` once → then loop `sim.step()` + `robot.update()`**

```python
sim.reset()                       # MUST call before stepping (initializes physics buffers)

while sim_app.is_running():
    robot.set_joint_position_target(target)   # what you WANT
    robot.write_data_to_sim()                 # push commands into PhysX
    sim.step()                                # advance physics (+ render if cameras on)
    robot.update(sim.get_physics_dt())        # pull fresh state back out
    cam.update(sim.get_physics_dt())          # refresh camera buffers
```

Notes: This is the loop that runs whether or not there's a window. `sim.reset()` must happen before the first step or buffers are uninitialized (a classic silent failure). Each iteration: command → write → step → update. `write_data_to_sim` sends targets down; `robot.update` reads joint positions/velocities/torques back up. `sim.step()` is where PhysX integrates and, if cameras are enabled, the renderer produces a frame — all off-screen.

---

## Slide 14 — Make it move: joint space (the simple way)

**Command target joint angles; PD actuators drive there**

```python
target = robot.data.default_joint_pos.clone()
target[0, elbow_idx] = 0.9 * upper_limit      # drive elbow near its limit
robot.set_joint_position_target(target)
for _ in range(settle_steps):
    robot.write_data_to_sim(); sim.step(); robot.update(dt)
reached = robot.data.joint_pos[0, elbow_idx]   # verify it got there
```

- Great for: joint-limit sweeps, homing, gripper open/close
- You directly say "joint X → angle θ"; the actuator's stiffness/damping do the rest

Notes: The most direct control. You set a target angle per joint; the implicit PD actuator applies force to reach it. This is exactly how the project's `robot_test.py` sweeps every joint to 90% of its soft limit and logs reached-vs-target — a pure headless validation that the robot is configured correctly. No IK, no math — just "go to this pose."

---

## Slide 15 — Make it move smart: task space (differential IK)

**Say where the *hand* should go; solve for joint angles**

```python
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
ik = DifferentialIKController(DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls"), num_envs=1, device=dev)

ik.set_command(torch.cat([des_pos_b, des_quat_b], dim=-1))   # desired TCP pose (root frame)
jacobian = robot.root_physx_view.get_jacobians()[:, jac_idx, :, arm_idx]
new_q    = ik.compute(tcp_pos_b, tcp_quat_b, jacobian, joint_pos)  # -> joint targets
robot.set_joint_position_target(new_q); robot.write_data_to_sim(); sim.step()
```

- `dls` = damped least-squares (robust near singularities)
- Jacobian comes from PhysX; poses converted world→root frame

Notes: This is how you trace a square with the end-effector or reach a grasp pose. You give a Cartesian target; the differential IK controller uses the PhysX-computed Jacobian to output joint targets each step. The official Isaac Lab tutorial (`run_diff_ik.py`) and this project's `robot_test.py` use the identical pattern — proof that what you learn here is the standard API, not a project quirk. All headless.

---

## Slide 16 — "Seeing" without a screen

**Render off-screen → pull pixels to numpy → save**

```python
from isaaclab.utils import convert_dict_to_backend
frame = convert_dict_to_backend(dict(cam.data.output), backend="numpy")
rgb = frame["rgb"]        # (H, W, 3) uint8 — a real image, no window ever opened
```

- Save state + images to **HDF5**, debug video to **MP4** (libx264) — all display-free
- Want to *watch* live? `--livestream 2` streams the viewport over WebRTC to your laptop

Notes: This is the payoff of `--enable_cameras`. `cam.data.output` holds the rendered tensors; convert to numpy and you have images identical to what a window would show. Persist them with `h5py` (arrays) and `imageio` (mp4). And when you genuinely need to *look* — say to debug a weird grasp on a remote server — livestreaming gives you the viewport over the network without ever running a local GUI.

---

## Slide 17 — Verify a headless run (you can't just "look")

**Trust logs, numbers, and saved files — not your eyes**

- Log the contract: `joint target=X reached=Y (limit [lo,hi])`, `IK corner err=0.004 m`
- Save `.h5` (joint_pos/vel, applied_torque, ee_pose) + `.mp4` for a later eyeball
- Seed every run explicitly for reproducibility: `Running simulation with seed: 200`
- Sanity-check shapes: action = `[ee_pos(3), ee_rot, gripper(1)]`, state pos+quat = 7D

Notes: Headless discipline: since nobody's watching, your logs and asserts are your eyes. The project's harness prints reached-vs-limit for every joint and IK tracking error per waypoint — if those numbers are wrong, the setup is wrong, window or not. Always seed (numpy/torch/sim reset) so a run is reproducible, and assert tensor shapes at boundaries so a silent mis-wire can't corrupt your dataset.

---

## Slide 18 — Scale it: many envs, many GPUs

**Headless is what makes throughput possible**

- One process, **many parallel envs**: `--num_envs 2048` on a single GPU (vectorized)
- One GPU per process: launch **one process per GPU**, each pinned + seeded disjointly

```bash
CUDA_VISIBLE_DEVICES=0 python simulate.py --seed 40  -n 250 --headless --enable_cameras &
CUDA_VISIBLE_DEVICES=1 python simulate.py --seed 290 -n 250 --headless --enable_cameras &
```

- Correctness: index per env (`[env_i]`, never `[0]`); disjoint seeds → no duplicate data

Notes: This is *why* headless matters beyond convenience. No window means you can pack thousands of parallel environments per GPU and run one process per card, unattended. Two rules keep it correct: give each worker a distinct-but-reproducible seed (same seed = duplicated data), and index tensors per-env instead of collapsing to `[0]` (which silently throws away all but one env). This project's `run_multi_gpu.sh` automates exactly this.

---

## Slide 19 — Reading the logs: what's normal, what's not

**Display warnings are EXPECTED headless — don't panic**

- ✅ Harmless on a display-less box:
  - `GLFW initialization failed`, `failed to open the default display`
  - `carb.windowing-glfw.plugin ... failed` → there's no window, by design
- ✅ Healthy signs: the experience-file line, the Manager tables, `Saving episode ... N frames`
- ❌ Real problems: `module isaaclab not found`, `loaded before SimulationApp`, black images

Notes: New users see a wall of GLFW/display warnings and assume it's broken. It isn't — those plugins can't start because there's intentionally no window. The signals that actually matter: the experience `.kit` line (confirms the right mode), the environment/manager summary tables, and "Saving episode" lines proving data lands on disk. Learn to skim past the noise.

---

## Slide 20 — Troubleshooting cheat sheet

| Symptom | Cause | Fix |
|---|---|---|
| Images black / empty | rendering off | add `--enable_cameras` |
| `loaded before SimulationApp` | imported too early | launch app first; defer imports |
| `module isaaclab not found` | wrong conda env | `conda activate dynamicVLA_isaac` |
| Robot falls through floor | no ground / no `sim.reset()` | add `GroundPlaneCfg`; `reset()` before stepping |
| Grasp explodes / object slips | solver/friction gains | tune iterations, friction, velocity caps |
| Duplicate data across GPUs | shared seed | disjoint `--seed` per worker |
| Want to watch remotely | headless has no window | `--livestream 2` (WebRTC) |

Notes: Keep this as the "when it breaks" slide. Almost every headless issue is one of these. The pattern to internalize: black images → cameras; import errors → launch order or env; physics weirdness → reset/ground/solver.

---

## Slide 21 — Put it together: run a robot right now

**From this repo — the fastest way to "play with a robot," no GUI**

```bash
conda activate dynamicVLA_isaac
cd DynamicVLA

# joint sweep + gripper cycle + IK square, headless, saving video:
python -m simulations.robot_test --enable_cameras --debug

# or just the IK trace, saving to a folder:
python -m simulations.robot_test --tests tcp -o ../datasets
```

- Outputs: `rb5_robot_test.h5` (+ per-camera `.mp4` with `--debug`)
- Watch it move via logs; review the `.mp4` afterward

Notes: `robot_test.py` is the perfect sandbox: it spawns the RB5 arm, enables physics, sweeps joints, cycles the gripper, and traces a square with IK — everything from slides 9–17 in ~250 lines, fully headless. Tell the audience: clone the pattern from this file and you have a template for your own robot. This is the "go home and try it" slide.

---

## Slide 22 — The bigger picture: headless as a data factory

**Same machinery powers a full VLA data + eval pipeline**

```text
launch (--headless --enable_cameras)
   └─► simulate.py       → raw episodes (.h5 + .mp4)      [Isaac, headless]
        └─► translate     → validated/filtered episodes    [Isaac, headless]
             └─► LeRobot   → training dataset               [PyTorch, no Isaac]
                  └─► train / evaluate policy
```

- Everything left of "train" is headless Isaac Sim
- The exact same launch + step + camera pattern, just scripted for scale

Notes: Zoom out so the audience sees where this leads. In DynamicVLA the headless robot loop is the front of a factory that produces training data for a Vision-Language-Action model. Data generation, replay/validation, and policy evaluation are all headless. Once you own the fundamentals (launch, physics, control, capture), the "pipeline" is just those fundamentals scripted and scaled.

---

## Slide 23 — Key takeaways

**Remember these six**

1. Headless = same engine (PhysX + RTX), no window — physics & rendering still work
2. **Launch the app before importing `omni`/`isaaclab`**
3. `--headless` ≠ rendering; add `--enable_cameras` for images; `--livestream` to watch remotely
4. Robot = `ArticulationCfg` (USD + physics + init pose + actuators); world = `SimulationContext(dt)`
5. Loop = `reset()` → `set_target` → `write_data_to_sim` → `step` → `update`; IK for task-space
6. Verify with logs/`.h5`/`.mp4` + seeds; scale with num_envs + one process per GPU

Notes: If they leave with slides 5 (launch-before-import) and 13 (the reset/step loop), they can run a robot headless. Everything else refines it. End by repeating the promise from slide 1: spawn, physics, drive, record — no screen.

---

## Slide 24 — References & where to look

**In this repo**
- Standalone robot harness (best starting template): `simulations/robot_test.py`
- Robot/physics config (`ArticulationCfg`, actuators): `simulations/robots/rb5_850e.py`
- World/physics settings (`dt`, PhysX): `simulations/configs/env_cfg.py`
- Camera config: `simulations/configs/scene_cfg.py`, `sim_cfg.yaml`
- Data-gen + multi-GPU: `simulations/simulate.py`, `simulations/run_multi_gpu.sh`

**Official NVIDIA docs**
- Standalone Python / `SimulationApp` headless: docs.isaacsim.omniverse.nvidia.com → *Python Environment*
- `AppLauncher` deep-dive & flags: isaac-sim.github.io/IsaacLab → *Deep-dive into AppLauncher*
- Differential IK tutorial: isaac-sim.github.io/IsaacLab → *Using a task-space controller* (`run_diff_ik.py`)
- Empty scene / `SimulationContext`: isaac-sim.github.io/IsaacLab → *Creating an empty scene*

Notes: Send people to `robot_test.py` first for a working headless template, then `rb5_850e.py` to see how a robot is described. The NVIDIA links confirm every API shown is the standard, supported way — not a project hack. Encourage them to open `--help` on any AppLauncher script to see all the flags live.

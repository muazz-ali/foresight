# Sim — house scene + scripted state machine

How the state machine works: [`STATE_MACHINE.md`](STATE_MACHINE.md).  
## Config (split)

| Path | What it does |
|---|---|
| `phase0_cfg.yaml` | Throw / claim entry — `include:` of the four files below |
| `mode2_cfg.yaml` | Static cubes + language — same includes, Mode-2 overrides only |
| `cfg/collect.yaml` | Cameras, robot, scene, lights, paths, sim clock |
| `cfg/state_machine.yaml` | State-machine stages + `claim_height` (centre aim, `z_slack`) |
| `cfg/training.yaml` | Demo delay numbers for the **policy** pack (`policy_training_delay`) |
| `cfg/evaluation.yaml` | Eval delay / old-frame knobs |

Loader: `interfaces.config.load_yaml` (resolves `include:`; yaml is SI m/s).

## Files
| Path | What it does |
|---|---|
| `scene.py` | Franka + house USD + throw object + container + cameras |
| `motion.py` | On-table coast (straight line; miss leaves the table) |
| `physics_grasp.py` | Helpers for the real PhysX grasp |
| `state_machine.py` | Intercept-servo pick stages (0–8) |
| `STATE_MACHINE.md` | Short walkthrough |
| `state_machine_logs.py` | Plain log lines (banners, stage lines, results) |
| `success.py` | Did we succeed? + failure label |
| `collect.py` | One episode loop; saves robot pose, images, pack |
| `video.py` | Debug MP4 (static \| wrist + text) |

Shared object message: `interfaces/state.py`.

## Meshes
| Root | Use |
|---|---|
| `/home/gpuadmin/Desktop/muazzam/objects` | Grasp objects + place containers |
| `/home/gpuadmin/Desktop/muazzam/scenes` | House USDs (random table) |

Object + container meshes are chosen **once per process** (`--seed` / `--object-usd` / `--container-usd`). For more variety, run several workers with different seeds or category names.

## Run
```bash
conda activate dynamicVLA_isaac
bash scripts/run_g0.sh smoke
bash scripts/run_g0.sh            # full Gate G0 → data/g0/
```

Isaac launch tips: [`../ppt.md`](../ppt.md).

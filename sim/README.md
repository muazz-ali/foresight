# Sim — table scene + scripted expert

How the expert works (plain words + code links): [`STATE_MACHINE.md`](STATE_MACHINE.md).


## Files
| Path | What it does |
|---|---|
| `phase0_cfg.yaml` | Scene, cameras, expert timings, mesh folders |
| `scene.py` | Franka + table + object + container + cameras |
| `motion.py` | On-table sliding motion + bounce-aware forecast |
| `state_machine.py` | Intercept-servo pick stages (0–9) |
| `STATE_MACHINE.md` | Simple full walkthrough with links into the code |
| `state_machine_logs.py` | Plain log lines (banners, stage lines, results) |
| `success.py` | Did we succeed? + failure label |
| `collect.py` | One episode loop; saves robot pose, images, future numbers |
| `video.py` | Debug MP4 (static \| wrist + text) |

Shared object message: `interfaces/state.py`.

## Meshes
| Root | Use |
|---|---|
| `/home/gpuadmin/Desktop/muazzam/objects` | Grasp objects + place containers |
| `/home/gpuadmin/Desktop/muazzam/scenes` | Unused for now |

Object + container meshes are chosen **once per process** (`--seed` / `--object-usd` / `--container-usd`). For more variety, run several workers with different seeds or category names.

## Run
```bash
conda activate dynamicVLA_isaac
bash scripts/run_g0.sh smoke
bash scripts/run_g0.sh            # full Gate G0 → data/g0/
```

Isaac launch tips: [`../ppt.md`](../ppt.md).

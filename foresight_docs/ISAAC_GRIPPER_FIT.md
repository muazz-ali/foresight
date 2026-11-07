# Isaac Props vs our gripper (one page)

Gate from `sim/cfg/collect.yaml`: **collision / bbox XY width ≤ 70 mm**  
(Franka open ≈ 80 mm). Width = max(L, W). Measured with `sim/usd_bbox.py` (+ instance proxies for DexCube).

## Use now (≤ 70 mm)

| Asset (Nucleus `Isaac/Props/…`) | Size L×W×H (mm) | XY width | Role |
|---|---:|---:|---|
| `Blocks/blue_block.usd` | 47×47×47 | 47 | color language |
| `Blocks/red_block.usd` | 47×47×47 | 47 | color language |
| `Blocks/green_block.usd` | 47×47×47 | 47 | color language |
| `Blocks/yellow_block.usd` | 47×47×47 | 47 | color language |
| `Blocks/basic_block.usd` | 47×47×47 | 47 | generic |
| `Blocks/block.usd` | 50×50×50 | 50 | generic |
| `Blocks/block_instanceable.usd` | 50×50×50 | 50 | generic |
| `Blocks/MultiColorCube/…` | 50×50×50 | 50 | multi-color faces |
| `Blocks/DexCube/dex_cube_instanceable.usd` | **60×60×60** | 60 | standard grasp cube |
| `Blocks/nvidia_cube.usd` | 65×65×65 | 65 | branded (tight margin) |

**Best starter set for “grasp the red/green cube”:** the four colored blocks (47 mm, ~23 mm margin).

## Borderline (70–80 mm) — skip for our gate

| Asset | Size (mm) | XY width |
|---|---:|---:|
| `Rubiks_Cube/rubiks_cube.usd` | 72×72×72 | 72 |
| `YCB/…/061_foam_brick.usd` | 78×51×53 | 78 |

## Too wide for Franka (do not use as grasp target)

Almost all **YCB** and **Mugs** fail the 70 mm gate when upright:

| Asset | XY width (mm) | Asset | XY width (mm) |
|---|---:|---|---:|
| tuna_fish_can | 86 | gelatin_box | 89 |
| tomato_soup_can | 102 | master_chef_can | 140 |
| pudding_box | 114 | sugar_box | 176 |
| mug (YCB / SM_Mug_*) | 117–137 | cracker_box | 213 |
| banana / scissors | ~197 | wood_block | 206 |
| mustard_bottle | 191 | bleach / pitcher | 251 / 242 |
| bowl / drill / clamps | 161–203 | beaker_500ml | 179 |
| KLT bin | container only | — | — |

## Shapes / spheres (no size catalog)

`Shapes/{sphere,cube,cone,cylinder,torus}.usd` are **1 m unit meshes**. Usable only if you scale them, or better: spawn with Isaac Lab:

| Spawner | Fit rule |
|---|---|
| `SphereCfg(radius=r)` | **r ≤ 35 mm** (diameter ≤ 70) |
| `CuboidCfg(size=(a,b,c))` | **max(a,b) ≤ 70 mm** |

Tint with `PreviewSurfaceCfg(diffuse_color=…)`. Any color/size — not a fixed Nucleus pack.

## Practical shortlist for Foresight

1. **Colored blocks ×4** (blue/red/green/yellow) — language + distractor.
2. **DexCube** — if you want the Isaac Lab “standard” 6 cm cube.
3. **Tinted `CuboidCfg` / `SphereCfg`** — if you need more colors or smaller balls.
4. **Do not expect YCB** to fill the cube/sphere slot under our 70 mm rule.

_Counts measured: USE=10 (incl. instanceables), borderline=2, too wide≈35._

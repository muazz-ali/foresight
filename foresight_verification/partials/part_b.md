# Part B — dataset statistics (p3_throw demos)

Seed=0. Units: cm / cm/s. Noise: `augment_conditioning` pos_std=0.015 m, vel_std=0.03 m/s.

## Data

- H5 scanned: 6529
- success+clean used: 5568
- episodes with frames: 5568
- LeRobot episodes (ref): 4117

## Conditioning check

- p := saved conditioning XY (noise-free). Equals o + v·Δ exactly. object_pos[t+6] does NOT match (throw / non-CV motion). Looked: H5 'conditioning', 'object_pos', 'object_vel', attrs conditioning_layout.
- max |cond − (o+vΔ)| m = 0.0
- max |cond − object_pos[t+6]| m = 0.2970

## Summary table — pooled k=1..8

| axis | d (cm) | σ_u (cm) | σ_n (cm) | w_h | e* (cm) | n |
|---|---:|---:|---:|---:|---:|---:|
| x | 0.7934 | 1.3183 | 1.5653 | 0.5850 | 1.1186 | 44544 |
| y | 0.8214 | 1.3147 | 1.5859 | 0.5927 | 1.1952 | 44544 |

## B8 arithmetic (pooled k=1..8)

- **x:** λ_h=1/σ_u²=1/1.3183²=0.575401; λ_n=1/σ_n²=1/1.56529²=0.40814; w_h=λ_h/(λ_h+λ_n)=0.58503; w_n=λ_n/(λ_h+λ_n)=0.41497; e*=d·λ_h/λ_n=0.793422·0.575401/0.40814=1.11858
  - with σ_u_cond: λ_h=1/σ_u²=1/0.53194²=3.53407; λ_n=1/σ_n²=1/1.56529²=0.40814; w_h=λ_h/(λ_h+λ_n)=0.896469; w_n=λ_n/(λ_h+λ_n)=0.103531; e*=d·λ_h/λ_n=0.793422·3.53407/0.40814=6.87021
- **y:** λ_h=1/σ_u²=1/1.31473²=0.578534; λ_n=1/σ_n²=1/1.58587²=0.397617; w_h=λ_h/(λ_h+λ_n)=0.592669; w_n=λ_n/(λ_h+λ_n)=0.407331; e*=d·λ_h/λ_n=0.821446·0.578534/0.397617=1.19521
  - with σ_u_cond: λ_h=1/σ_u²=1/0.410211²=5.94273; λ_n=1/σ_n²=1/1.58587²=0.397617; w_h=λ_h/(λ_h+λ_n)=0.937288; w_n=λ_n/(λ_h+λ_n)=0.0627121; e*=d·λ_h/λ_n=0.821446·5.94273/0.397617=12.2772

## Action chunk note

H5 has action_pos (T,3) single-frame only (looked: dataset 'action_pos'). Chunk indices 10/25/49 use action_pos[i+idx] clipped to [0,T-1].

## Figures

- `foresight_verification/figures/part_b_hist_pooled_k1_8_x.png`
- `foresight_verification/figures/part_b_hist_pooled_k1_8_y.png`
- `foresight_verification/figures/part_b_hist_speed_0.00_k1_8_x.png`
- `foresight_verification/figures/part_b_hist_speed_0.00_k1_8_y.png`
- `foresight_verification/figures/part_b_hist_speed_0.20_k1_8_x.png`
- `foresight_verification/figures/part_b_hist_speed_0.20_k1_8_y.png`

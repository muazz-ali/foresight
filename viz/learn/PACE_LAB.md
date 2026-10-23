# Pace Lab (experiment)

Editable teaching layer for Foresight numbers. Not Archify core.

Open `pace-lab.html` via the viz server (`python3 -m http.server 8765` in `viz/`).

## What we stole

| Source | Steal |
|---|---|
| [ViviDoc](https://github.com/MisterBrookT/vividoc) SRTC | State · Render · Transition · Constraint loop |
| [Curvenote](https://github.com/curvenote/components) / [Tangle](https://github.com/worrydream/Tangle) | Numbers in the sentence; drag left/right to scrub |
| [compound-effect](https://github.com/alexadam/compound-effect) | Amber = yours, green = computed |
| [TensorFlow Playground](https://playground.tensorflow.org/) / Polo Club | Feature mute / B-zero “turn it off and watch” |
| Nicky Case sandbox patterns | Insert / delete / reset at the reader’s pace |

## What we built on top

- **Insert / delete / reset** slots (not only scrub fixed knobs)
- **Catalog + constraints** (unique keys, max 8, clamped ranges)
- **Undo + shareable URL hash** (`#p_hat_x=0.51&delta=0.25…`)
- **Toy chase picture** tied to the pack (teaching only, not a gate)
- **Drag-scrub** on amber story numbers (Tangle `dx/5 * step`)

## Scout notes (keep for GitHub ship)

- Archify HTML is explore-only (~750KB); do **not** bake labs into it — wrap with `?embed=1` + `#focus=` later ([viz audit](0e0316fc-6a9b-416f-8560-0326b32bfff0)).
- Gap nobody fills well: **locked diagram + reader-owned values** ([diagram playgrounds](a6dadb39-f976-4d37-8a05-2beb5ddb23e4)).
- B-zero UI pattern ≈ What-If Tool / Playground feature toggles; pair later with shift/swap probes, not zero-only ([ablation scan](c53b9029-1511-4d5c-b08c-c3c946fa9f5a)).
- Share links: flat hash params are enough; skip LZ-string until payload grows ([URL scan](d8739693-257c-408f-8285-8c351e79221f)).
- Runtime **add/remove** of parameters is rare on GitHub; Curvenote `defineVariable`/`removeVariable` is the closest API — Pace Lab already does insert/delete in plain JS ([topic scan](e2427969-8c83-4adf-a093-e901ab0ee481)).
- Δ teaching: seconds scrubber + ghost aim (rteammco Kalman demo); our toy chase is that mechanic ([Kalman scan](5a2669f2-ab64-4561-9c57-76927b39e9e5)).
- Formal plan for this page: [`pace-lab.spec.json`](pace-lab.spec.json) (ViviDoc DocSpec / SRTC fields).

## Open later (standalone repo)

1. Extract `srtc-runtime.js` as a tiny MIT package
2. Host shell: Pace Lab side panel + Archify iframe (`?embed=1`, hash focus)
3. Templates: attention (`from-scratch`), expert SM, Kalman Δ
4. Optional: step-synced presentation scrubber on workflows (gg-charts / KernelPath style)

External skeleton twin (sliders + SVG + Reset, MIT, one file): [Mohl/loop-studio](https://github.com/Mohl/loop-studio) — see [single-file kits](4f823f0b-36ca-4650-b606-feb006fbc037). Pace Lab already follows that shape.

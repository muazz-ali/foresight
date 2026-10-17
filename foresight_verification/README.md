# Foresight verification (shortcut / copycat model)

Evidence-only check of the mathematical shortcut model
`target ≈ hand + d` vs reading `p̂`.

## Hard rules
- No training / fine-tuning
- No Isaac sim run >10 min without human approval
- Do not edit production `policy/`, `sim/`, `scripts/eval_policy.py`
- New code lives under `foresight_verification/scripts/` only

## Agent split
| Part | Owner | Output |
|---|---|---|
| A settings | explore agent | `partials/part_a.{json,md}` |
| B dataset stats | eval agent | `partials/part_b.*` + figures |
| C controller + eval logs | eval agent | `partials/part_c.*` |
| D sensitivity probe | eval agent | `partials/part_d.*` |
| E DART design | explore agent | `partials/part_e.*` |
| F merge | parent | `report.json`, `report.md` |

## Merge
```bash
python foresight_verification/scripts/merge_report.py
```

## Checkpoint under test
`runs/p3_throw/pretrained_model` (SmolVLA on `data/p3_throw` clean demos).

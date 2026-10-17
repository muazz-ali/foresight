# Part E2 — DART collect patch proposal (DO NOT APPLY)

Goal: add **zero-mean** noise to the **executed** EE command while the **logged training label** stays the clean expert target.

Production code must stay untouched until a human approves.

---

## Where (cite)

| Role | Location |
|---|---|
| Clean expert target | `sim/state_machine.py:709–717` — `return ExpertCommand(position=target, ...)` |
| Episode loop | `sim/collect.py:294` — `cmd = expert.step(...)` |
| Execute | `sim/collect.py:300` — `scene.set_ee_target(cmd.position, cmd.quat_wxyz, cmd.gripper_open)` |
| Label | `sim/collect.py:326–328` — `log["action_pos"].append(cmd.position.copy())` (same vector today) |

**Do not** put noise inside `PickPlaceStateMachine.step`. That return is the label source for every collect path (`run_phase0`, `run_phase1_collect` → `run_episode`).

Optional later: same split in `scripts/eval_policy.py` if you want DART-style noise at eval (out of scope for collect).

---

## Proposed patch text (`sim/collect.py`)

Insert after the reflex overlay (~L298), before `set_ee_target`. Flag-gated; default off.

```python
        # --- DART (proposal only; default off) ---
        # Label = clean expert target. Execute = label + zero-mean noise.
        label_pos = np.asarray(cmd.position, dtype=np.float64).reshape(3).copy()
        exec_pos = label_pos.copy()
        dart_sigma_m = float(meta.get("dart_sigma_m", 0.0))  # or from cfg / run_episode kwarg
        if dart_sigma_m > 0.0:
            # White: each frame independent. Prefer OU if α known (see part_e E3).
            rng = meta.get("dart_rng")  # np.random.Generator, seeded per episode
            if rng is None:
                raise RuntimeError("dart_sigma_m>0 requires meta['dart_rng']")
            noise = rng.normal(0.0, dart_sigma_m, size=3)
            # Optional: only XY (conveyor / finger plane); leave Z clean:
            # noise[2] = 0.0
            exec_pos = label_pos + noise

        scene.set_ee_target(exec_pos, cmd.quat_wxyz, cmd.gripper_open)
        scene.step(
            kinematic_object=cmd.kinematic_object,
            attach_object=cmd.attach_object,
            freeze_object=cmd.freeze_object,
        )
        # ... existing post-step reads ...
        log["action_pos"].append(label_pos)          # CLEAN label
        # log["action_pos_executed"] = exec_pos      # optional debug channel
        log["action_quat"].append(cmd.quat_wxyz.copy())
        log["gripper"].append(cmd.gripper_open)
```

Wiring sketch for `run_episode(...)`:

```python
def run_episode(..., dart_sigma_m: float = 0.0, dart_seed: int | None = None):
    ...
    meta["dart_sigma_m"] = float(dart_sigma_m)
    meta["dart_rng"] = (
        np.random.default_rng(int(dart_seed)) if dart_sigma_m > 0.0 else None
    )
```

---

## Design notes

1. **Stages:** noise only while chasing / closing (stages 1–3) is safer; lift–place can stay clean. Decide FOR_HUMAN.
2. **Clip:** hard-cap `‖noise‖` or per-axis to something ≤ design τ (catalog min τ ≈ 2.5 mm → 0.0025 m is very tight; many objects allow more).
3. **σ size:** from E3, `σ_c = σ_δ √((2−α)/α)` once part_c α exists. Until then, treat white σ as a scout knob or use OU.
4. **Reflex:** if `overlay_reflex` runs, apply DART **after** reflex on the position that will be executed, still log pre-DART (or pre-reflex+DART-clean) consistently — pick one and document in meta.
5. **H5 schema:** adding `action_pos_executed` is additive and safe for LeRobot convert; do not rename `action_pos` (that is the label).

---

## Smoke checklist (after approve)

1. `dart_sigma_m=0` → bit-identical to today’s collect (same action_pos).
2. `dart_sigma_m>0` → `action_pos` mean offset ≈ 0 vs expert; executed EE wanders.
3. One episode shapes unchanged; optional executed channel shape `(T,3)`.
4. No change to `sim/state_machine.py` return path.

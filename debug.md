# Debug brief — parent orchestrator (finalize)

Source of truth for this diagnostic run. Parent follows the **Parent runbook** exactly. Children receive **only** the Shared preamble + their own `P#` block. Never paste another child’s prompt or results into a child.

---

## Parent runbook (do not improvise)

```text
WORKSTREAM: explore (orchestration only)

1. Read this entire file once. Do not edit production code.
2. Run P0 alone (you may do P0 yourself, or one child with ONLY preamble + P0).
3. Emit P0’s GO / NO-GO verdict + evidence paths. STOP. Wait for human confirm.
4. After human confirm: launch P1, P2, P3, P4 as FOUR separate child agents.
   Each child gets ONLY:
     - Shared preamble
     - That child’s prompt block
   Children must NOT see: other P# prompts, other results, or this Parent runbook’s
   branch logic beyond what’s already inside their own block.
5. When all four return: write the GATE merge report (template below).
6. Do NOT launch P5, P6, P7, or P8 in this run. Stop after the GATE report.
7. Return: GATE merge report + paths to evidence. Nothing else.
```

**Isolation rule.** If a child cites another P#’s evidence, that child is out of brief — discard that cross-cite and keep only what that child measured itself.

**P0 visibility.** Every later header (GATE, and any future P5+) must restate `P0: GO|NO-GO` so magnitude claims stay labeled.

---

## Ordering principle

Cheapest checks that can end or re-scope the investigation go first. Expensive ablations (P5+) wait until the GATE picks a branch. Process/plan prompts (P7–P8) wait until real defects exist.

| # | Prompt | Cost | Role | Covers |
|---|---|---|---|---|
| **P0** | Measurement validity | ~10 min | **Blocking alone** → human confirm | power / n |
| **P1** | Eval wiring trace | ~30 min | Parallel after GO/confirm | Q2 |
| **P2** | Training data audit | ~30 min | Parallel | Q1 (data) |
| **P3** | Conditioning liveness | ~45 min | Parallel | Q1 (signal) |
| **P4** | Post-place triage | ~15 min | Parallel; no SM deep dive | Q3 |
| **GATE** | Parent merge + branch | — | After P1–P4; **stop** | — |
| P5 | Four-arm ablation | GPU hours | **Deferred** — only if GATE selects it later | Q1+Q2 |
| P6 | Expert SM audit | ~1 hr | **Deferred** — only if P4 = EXPERT-SM or BOTH | Q4 |
| P7 | Gate calibration | ~45 min | **Deferred** | Q5, Q6 |
| P8 | Process checklist | ~30 min | **Deferred** | Q7 |

P1–P4 are disjoint and read-only → parallel after human confirms P0.

---

## Shared preamble (prepend to every child)

```text
FORESIGHT DIAGNOSTIC SERIES — shared context

Contract: foresight_plan.md, AGENTS.md, .cursor/rules/15-own-the-stack.mdc
Read-only on production trees. Scratch scripts ONLY under tools/diag/.
Do not edit sim/, policy/, eval/, scripts/, or interfaces/ during diagnosis.
Do not import DynamicVLA for expert or success latch.
Do not invent omni.* / isaaclab.* APIs — verify against ppt.md / IsaacLab.

Every claim needs one of: path/file.py:LINE, a command + its output, or a printed
number. Anything else must be labeled UNKNOWN: or UNVERIFIED:.

Glossary (use these exact terms):
  "mid"     = success rate at 0.20 m/s (fallback 0.30 m/s if 0.20 missing);
              see eval/gate_g1.py MID=0.20, MID_ALT=0.30
  "drop"    = B_rate(0.0) - B_rate(0.30)  [flat_drop_0_to_30]; pass if drop < 0.15
  "margin"  = B_mid - A_mid; pass if margin >= 0.20
  "zero-out"= B_mid - B_zero_mid; pass if >= 0.15 (conditioning must hurt when zeroed)
  arms      = A (no conditioning) | B (predicted) | B_zero (zeroed) | B_oracle (GT future; P5 only)
  reeval    = data/p1/reeval_n8/gate_g1_report.json  (verify n from summaries.*.attempts)

Max 3 iterations. If still blocked, stop and return the single specific blocker.
End every report with a non-empty section: WHAT I COULD NOT DETERMINE
Do not read or cite other P# prompts or other agents' results.
```

---

## P0 — Measurement validity (blocking, runs alone)

```text
WORKSTREAM: eval
GOAL: Decide whether data/p1/reeval_n8/gate_g1_report.json can distinguish
"conditioning helps" from "conditioning does nothing." Nothing else.

STEPS
1. From gate_g1_report.json and eval/gate_g1.py, state exactly what n is per
   cell: episodes per arm per speed bin. Cite the field you read (e.g.
   summaries_B[].attempts).
2. Compute Wilson 95% CIs for: A mid, B mid, B_zero mid, and the 0→30 drop
   (treat drop as difference of two binomial rates; state the method).
3. Compute episodes-per-cell required to resolve a 20-point margin at 80% power,
   alpha 0.05, around p≈0.3 (two-proportion comparison A vs B at mid).
4. Report wall-clock cost of one episode from reeval logs (elapsed_s / attempts
   if present), so we know what that n costs.

REQUIRED OUTPUT
- Table: arm | n | point estimate | 95% CI
- Required n, and required n × per-episode cost in hours
- Verdict: GO or NO-GO
    GO    = current table can resolve a 20-pt margin (CIs / power support it)
    NO-GO = it cannot

DECISION RULE
If NO-GO: every downstream verdict about A-vs-B *magnitude* is
INCONCLUSIVE-BY-CONSTRUCTION and must be labeled so. P1–P4 still run after
human confirm (they are about mechanism, not magnitude), but any later P5 must
be re-scoped to the computed n before it is worth running.
Do not propose any code fix in this prompt.
```

---

## P1 — Eval wiring trace (parallel; isolated)

```text
WORKSTREAM: eval
GOAL: Determine whether Model B's conditioning tensor actually reaches the
policy at inference, and whether it matches what training saw.

STEPS
1. Trace scripts/eval_policy.py: how observation.conditioning is constructed,
   passed, and consumed. Cite every hop with file:LINE.
2. Find --zero-conditioning. Confirm from the reeval_n8 invocation (command
   line in scripts/run_g1_reeval.sh, config, or logs under
   data/p1/reeval_n8/) which arms had it set. Quote the evidence.
3. Instrument ONE inference step via a scratch script in tools/diag/ only
   (do not edit scripts/): print conditioning tensor shape, dtype, L2 norm,
   and whether it == zeros. Do this for B and for B_zero if runnable without
   full Isaac; if Isaac is required and unavailable, mark UNKNOWN and fall
   back to static trace + log evidence.
4. Compare train-time feature dim vs eval-time feature dim (proprio ± 12).
   Prefer checkpoint metadata / saved stats over a config file guess.
5. Confirm the checkpoint loaded for "B" is actually the conditioned model —
   check checkpoint meta / train args, not filename alone.

REQUIRED OUTPUT
Verdict, exactly one of:
  NOT-WIRED       conditioning absent or dropped at eval
  ZEROED          --zero-conditioning set on an arm that should not have it
  DIM-MISMATCH    train dim != eval dim
  WRONG-CKPT      B is actually A
  WIRING-CLEAN    tensor arrives, nonzero (when not B_zero), correct dim, correct ckpt
  UNKNOWN         blocked; state the single blocker
plus the shape/norm printout for both arms when available.

DECISION RULE
If anything other than WIRING-CLEAN / UNKNOWN: this is a candidate root cause.
Propose a <=30-line fix sketch (do not apply it). Recommend re-measure before
any ablation.
If WIRING-CLEAN: "ignored" is behavioral, not plumbing. Say so explicitly.
```

---

## P2 — Training data audit (parallel; isolated)

```text
WORKSTREAM: policy
GOAL: Explain the 0→30 cm/s drop of ~0.85 from the DATA side before blaming
the policy. This is the largest reported effect in reeval_n8.

STEPS
1. Over data/lerobot/p1/ (and collection metadata / sim/collect.py logs if
   present), histogram episode count by object speed. Report n per speed bin
   matching G1 speeds 0 / 0.1 / 0.2 / 0.3 / 0.4 m/s.
2. Compute EXPERT success rate per speed bin if logs exist (from sim/success.py
   latch fields, not from watching video). If unavailable → UNKNOWN with path
   searched.
3. Report any bin where: n < 10% of the dataset, OR expert success < 0.7.
4. Check whether eval speed bins are inside the training speed support, or are
   extrapolation.

REQUIRED OUTPUT
Table: speed bin | n episodes | % of dataset | expert success rate | in-support?
Verdict: DATA-EXPLAINS-DROP | DATA-ADEQUATE | UNKNOWN (with what is missing)

DECISION RULE
If the expert itself is weak at speed, or high-speed episodes are rare, the
policy's speed collapse is a data problem and conditioning debugging is
secondary. Say that plainly and rank it as a root cause candidate.
```

---

## P3 — Conditioning signal liveness (parallel; isolated)

```text
WORKSTREAM: policy
GOAL: Establish whether the ~12 numbers carry live information, independent of
whether the model uses them.

STEPS
1. From interfaces/state.py conditioning_vector, print the exact field layout:
   index → name → coordinate FRAME → units. All 12. Cite lines; no guessing.
2. Over the training set (or a large sample), per dimension: mean, std, min,
   max, %zeros, %NaN/Inf, unique-value count. Flag std ≈ 0 or unique count < 10.
3. For ONE episode at v≈0 and ONE at v≥0.20 m/s, table ||v||, p̂, and Δ across
   time. Do the fields move with the object?
4. Compute error of predicted future state vs logged ground-truth future
   (RMSE and correlation) when both exist. Predictor quality ≠ policy use.
5. Confirm policy/transforms.py appends conditioning to the policy state, and
   that normalization stats for these dims exist and are non-degenerate
   (zero-std dims wipe signal or NaN after norm).

REQUIRED OUTPUT
12-row table with per-dim verdict: INFORMATIVE | CONSTANT | DEGENERATE |
MISLABELED-FRAME | NAN.
Plus predictor RMSE vs oracle when computable, and top-line:
  SIGNAL-ALIVE | SIGNAL-DEAD | SIGNAL-NOISY (alive but high RMSE) | UNKNOWN

DECISION RULE
SIGNAL-DEAD → fix the producer before any ablation.
SIGNAL-NOISY → later B_oracle arm (P5) becomes the critical comparison.
```

---

## P4 — Post-place triage (parallel; 15 min; do NOT audit the SM)

```text
WORKSTREAM: sim-scene
GOAL: Decide in ~15 minutes whether the "goes back to the bowl" behavior
matters for G1. Do not open a deep expert state-machine audit in this prompt.

STEPS
1. Read sim/success.py: when does the success latch fire relative to place?
   If it fires at place, post-place motion cannot change the G1 metric.
2. Read sim/collect.py early-stop (stage >= 7): does the episode terminate at
   place during EXPERT recording?
3. From data/p1/eval_b_zero/ (and sibling eval dirs if needed), for 2 episodes:
   note place time vs re-reach time if readable from video filenames/logs;
   whether the episode was already scored success/fail.
4. State whether the behavior appears in EXPERT recordings, POLICY rollouts,
   or both. Policy rollout has no expert SM — if only there, expert SM is not
   implicated.

REQUIRED OUTPUT
Verdict, exactly one of:
  COSMETIC          post-place, after latch; does not affect any G1 number
  POLICY-ONLY       policy habit from demos; expert SM not implicated
  EXPERT-SM         appears in expert recordings → escalate to P6 (later)
  BOTH
  UNKNOWN

DECISION RULE
COSMETIC or POLICY-ONLY → Q4 closed for this run; P6 stays deferred.
Only EXPERT-SM or BOTH unlocks P6 in a later parent run.
```

---

## GATE — Parent merge (after P1–P4; no new work)

Parent fills this. Do not launch P5–P8.

```text
WORKSTREAM: explore (orchestration only)

## P0 header
P0 measurement: GO / NO-GO
n per cell: __
note: if NO-GO, all A-vs-B magnitude claims below are INCONCLUSIVE-BY-CONSTRUCTION

## Verdicts
P1 wiring:       ____
P2 data:         ____
P3 signal:       ____
P4 post-place:   ____

## Ranked root causes (max 5)
1. ...  evidence: <file:LINE | number | path>

## Branch decision — pick ONE (for a *later* run; do not execute now)
[ ] P1 found a wiring bug     → patch (<=30 lines) + re-measure; defer P5
[ ] P2 found a data gap       → fix data/expert first; conditioning secondary
[ ] P3 found dead signal      → fix producer; defer P5
[ ] All clean, signal alive   → next run may launch P5 (four-arm ablation)
[ ] P0 was NO-GO              → next decisive step is P5 at corrected n only
[ ] P4 escalated EXPERT-SM    → next run may launch P6 (not now)

## Do not do yet
- P5, P6, P7, P8 (this run)
- ...

## Open questions (max 2)
- ...

## Evidence paths
- ...
```

---

## Deferred prompts (do not launch in this run)

Kept for a later parent pass after human reviews the GATE report.

### P5 — Four-arm ablation

```text
WORKSTREAM: eval
GOAL: One experiment that answers Q1 and Q2 together. Run only if a later GATE
selected this branch and human approved.

SETUP
Four arms, SAME seeds, SAME episodes, n = value P0 computed:
  A         no conditioning
  B         predicted future state
  B_zero    conditioning zeroed
  B_oracle  ground-truth future from sim at inference

READ THE RESULT
  A < B_zero ≈ B ≈ B_oracle   → policy ignores the conditioning slot
  B_zero < B < B_oracle         → conditioning works; predictor noisy
  B_zero < B ≈ B_oracle         → conditioning at ceiling; bottleneck elsewhere
  no arm separates              → n still too small, or conditioning irrelevant here

REQUIRED OUTPUT
Full foresight_plan.md §8 metric table, all four arms, with n and Wilson CIs.
Do NOT claim G1 pass without the complete §8 table.
```

### P6 — Expert SM audit (only if P4 = EXPERT-SM or BOTH)

```text
WORKSTREAM: sim-scene
GOAL: Is the Foresight expert SM wrong, incomplete, or fine?
Run ONLY if P4 escalated. Otherwise record "closed by P4."

STEPS
1. Extract actual state graph from sim/; Mermaid diagram.
2. Table vs foresight_plan.md §4 (approach → descend/grasp/lift → place → reset;
   drop → approach).
3. Log sm_state vs time for one SUCCESS and one FAIL; count illegal jumps.
4. After place: is pick target invalidated? Is hold flag cleared on release?
   Timeouts / failure exits?

REQUIRED OUTPUT
Diagrams + defect table (id, BLOCKER/MAJOR/MINOR, file:LINE, symptom).
Recommendation: NO-CHANGE | SMALL-FIX (<=30 lines) | LEAVE-UNTIL-POLICY-PROVEN.
Do not mix policy-rollout evidence into expert defects.
```

### P7 — Gate calibration and literature sanity

```text
WORKSTREAM: explore
GOAL: Q5 and Q6 with numbers, informed by prior P results. Read-only.

PART A — gates (Q5)
Quote G1 from foresight_plan.md §8 and eval/gate_g1.py.
Using P0 required-n and cost: hours to legitimately claim G1.
Split: learn without relaxing / ONE debug-only relaxation / FROZEN for claim.

PART B — literature (Q6)
Job split only. Only papers opened from disk; else UNVERIFIED.
5-row: where/when | what/how | exactly when | clock | success criterion
Per row: KEEP | REVIEW-ONE-PIECE | DO-NOT-ADOPT.
```

### P8 — Process checklist (from real defects only)

```text
WORKSTREAM: explore
GOAL: Q7. <=10 bullets. Each item names the defect it would have caught.
No generic advice. Separate "speculative, not adopted" list for the rest.
```

---

## Parent kickoff (paste this to start the run)

```text
WORKSTREAM: explore (orchestration only)

Read /home/gpuadmin/Desktop/foresight/debug.md.
Run P0 first, alone. Report its GO/NO-GO verdict and stop.
After I confirm, launch P1, P2, P3, P4 as four separate child agents, each
receiving ONLY the shared preamble plus its own prompt block. Children must not
see each other's prompts or results.
When all four return, write the GATE merge report. Do not launch P5–P8.
Return the merge report and paths to evidence, nothing else.
```

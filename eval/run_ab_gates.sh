#!/usr/bin/env bash
# Score A1–A6 / B1–B6 on one recording root; each gate writes its default REPORT_auto.md.
# Usage: bash eval/run_ab_gates.sh data/eval/AB_recordings_vN
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=. PYTHONNOUSERSITE=1
PY=${PY:-$HOME/anaconda3/envs/dynamicVLA_isaac/bin/python}
R=${1:?usage: run_ab_gates.sh RECORDING_ROOT}
run(){ echo "######## $1"; shift; timeout 1800 $PY "$@" 2>&1 | grep -v "Warning\|warn(" | sed -n '/^# /,$p'; echo "exit=${PIPESTATUS[0]}"; }
run A1 eval/a1_velocity_ratio.py --root $R --workers 12
run A2 eval/a2_delta_inject.py
run A3 eval/a3_forecast_error.py --root $R --workers 12
run A4 eval/a4_z_grasp_depth.py --root $R --workers 12
run A5 eval/a5_parallax.py --root $R --workers 12
run A6 eval/a6_stale_motions.py
run B1 eval/b1_oracle_grid.py --root $R
run B2 eval/b2_descend_z_grasp.py --root $R
run B3 eval/b3_close_speed_match.py --root $R
run B3s eval/b3_speed_match.py --root $R
run B4 eval/b4_dk_invariance.py --root $R
run B5 eval/b5_hand_trail_bias.py --root $R
run B5c eval/b5_close_d_bias.py --root $R
run B6 eval/b6_instructed_object.py --root $R

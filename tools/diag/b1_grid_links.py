"""Run tools/diag/link_chain_analyze.analyze_episode on every episode of a B1 grid
folder (cells/<cell>/episodes.jsonl) and add joint margins. Writes one JSON list.

Usage: PYTHONPATH=. python tools/diag/b1_grid_links.py data/eval/<run> data/eval/<run>/link_rows.json
"""
import json, glob, sys
from pathlib import Path
from multiprocessing import Pool
import numpy as np, h5py
sys.path.insert(0, "/home/gpuadmin/Desktop/foresight")
from tools.diag.link_chain_analyze import analyze_episode

LO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
HI = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

def one(row):
    out = {"row": row}
    try:
        out["link"] = analyze_episode(Path(row["h5"]))
    except Exception as e:  # keep going; report the error
        out["link_error"] = repr(e)
    try:
        with h5py.File(row["h5"], "r") as f:
            st = f["stage"][:]; q = f["proprio"][:, :7]; ee = f["ee_pos"][:]
            ob = f["object_pos"][:]
            sel = (st >= 1) & (st <= 4)
            m = np.degrees(np.minimum(q - LO, HI - q))
            out["j_margin_min_deg"] = m[sel].min(0).round(2).tolist() if sel.any() else None
            out["hand_r_min"] = float(np.linalg.norm(ee[sel, :2], axis=1).min()) if sel.any() else None
            closes = np.where(st == 2)[0]
            out["close_frame"] = int(closes[0]) if len(closes) else None
            out["obj_r_at_close"] = float(np.linalg.norm(ob[closes[0], :2])) if len(closes) else None
            out["n_frames"] = int(len(st))
            out["stages"] = sorted(set(st.tolist()))
    except Exception as e:
        out["h5_error"] = repr(e)
    return out

if __name__ == "__main__":
    D, OUT = sys.argv[1], sys.argv[2]
    rows = [json.loads(l) for p in sorted(glob.glob(f"{D}/cells/*/episodes.jsonl")) for l in open(p)]
    with Pool(32) as pool:
        res = pool.map(one, rows, chunksize=8)
    json.dump(res, open(OUT, "w"))
    print(len(res), "episodes;", sum("link_error" in r for r in res), "link errors;", sum("h5_error" in r for r in res), "h5 errors →", OUT)

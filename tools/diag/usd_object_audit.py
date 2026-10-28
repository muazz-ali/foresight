#!/usr/bin/env python3
"""Static USD audit: visual vs collision bbox, origin, mass, friction.

No Isaac. Per mesh under ``object_dir/<category>/``:

  - visual AABB (meshes without CollisionAPI)
  - collision AABB (prims with CollisionAPI)
  - origin (root translate) vs each bbox centre
  - scale ops on root
  - mass / density (MassAPI)
  - friction / restitution (MaterialAPI if present)
  - collision approximation (MeshCollisionAPI) or collider type

Flags any visual↔collision size or centre gap > 1 mm, and origin↔centre > 1 mm.

Usage:
  PYTHONPATH=. python tools/diag/usd_object_audit.py \\
    --categories apple,avocado,egg,kiwi,lemon,lime,onion,orange,peach,tangerine \\
    -o data/eval/usd_object_audit.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FORESIGHT = Path(__file__).resolve().parents[2]
DEFAULT_OBJ = Path("/home/gpuadmin/Desktop/muazzam/objects")
SCOUT = (
    "apple",
    "avocado",
    "egg",
    "kiwi",
    "lemon",
    "lime",
    "onion",
    "orange",
    "peach",
    "tangerine",
)
FLAG_MM = 1.0


def _union(mins: list, maxs: list):
    if not mins:
        return None, None, None
    mn = np.min(np.asarray(mins, dtype=np.float64), axis=0)
    mx = np.max(np.asarray(maxs, dtype=np.float64), axis=0)
    return mn, mx, mx - mn


def audit_usd(path: Path) -> dict:
    """One USD → visual/collision bbox + physics attrs."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        return {"path": str(path), "error": "open failed"}
    root = stage.GetDefaultPrim()
    if not root:
        return {"path": str(path), "error": "no default prim"}

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    vis_mins, vis_maxs, col_mins, col_maxs = [], [], [], []
    approx = None
    collider_types: list[str] = []
    mass = density = None
    static_f = dynamic_f = restitution = None

    geom_types = (
        UsdGeom.Mesh,
        UsdGeom.Cube,
        UsdGeom.Sphere,
        UsdGeom.Capsule,
        UsdGeom.Cylinder,
        UsdGeom.Cone,
    )

    for prim in stage.Traverse():
        if not any(prim.IsA(t) for t in geom_types):
            continue
        rng = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        mn = np.array(rng.GetMin(), dtype=np.float64)
        mx = np.array(rng.GetMax(), dtype=np.float64)
        if not (np.all(np.isfinite(mn)) and np.all(np.isfinite(mx))):
            continue
        if np.any(mx <= mn):
            continue
        has_col = prim.HasAPI(UsdPhysics.CollisionAPI)
        if has_col:
            col_mins.append(mn)
            col_maxs.append(mx)
            collider_types.append(prim.GetTypeName())
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mc = UsdPhysics.MeshCollisionAPI(prim)
                approx = mc.GetApproximationAttr().Get()
        else:
            vis_mins.append(mn)
            vis_maxs.append(mx)

        if prim.HasAPI(UsdPhysics.MassAPI):
            mapi = UsdPhysics.MassAPI(prim)
            mass = mapi.GetMassAttr().Get()
            density = mapi.GetDensityAttr().Get()

    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.MaterialAPI):
            mat = UsdPhysics.MaterialAPI(prim)
            static_f = mat.GetStaticFrictionAttr().Get()
            dynamic_f = mat.GetDynamicFrictionAttr().Get()
            restitution = mat.GetRestitutionAttr().Get()
            break

    # Root translate / scale
    origin = np.zeros(3, dtype=np.float64)
    scale = np.ones(3, dtype=np.float64)
    xf = UsdGeom.Xformable(root)
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            v = op.Get()
            if v is not None:
                origin = np.array([v[0], v[1], v[2]], dtype=np.float64)
        elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
            v = op.Get()
            if v is not None:
                scale = np.array([v[0], v[1], v[2]], dtype=np.float64)

    # Also root world bbox (what scene.usd_bbox_size uses today)
    root_rng = cache.ComputeWorldBound(root).ComputeAlignedBox()
    root_mn = np.array(root_rng.GetMin(), dtype=np.float64)
    root_mx = np.array(root_rng.GetMax(), dtype=np.float64)
    root_size = root_mx - root_mn
    root_centre = 0.5 * (root_mn + root_mx)

    vm, vx, vs = _union(vis_mins, vis_maxs)
    cm, cx, cs = _union(col_mins, col_maxs)
    v_c = None if vs is None else 0.5 * (vm + vx)
    c_c = None if cs is None else 0.5 * (cm + cx)

    flags: list[str] = []
    size_diff = centre_diff = None
    if vs is not None and cs is not None:
        size_diff = np.abs(vs - cs) * 1000.0
        centre_diff = np.abs(v_c - c_c) * 1000.0
        if float(np.max(size_diff)) > FLAG_MM:
            flags.append(f"vis↔col size Δ={size_diff.round(2)} mm")
        if float(np.max(centre_diff)) > FLAG_MM:
            flags.append(f"vis↔col centre Δ={centre_diff.round(2)} mm")
    if c_c is not None and float(np.max(np.abs(c_c - origin))) * 1000.0 > FLAG_MM:
        flags.append(
            f"origin↔col centre Δ={(1000*np.abs(c_c-origin)).round(2)} mm"
        )
    if v_c is not None and float(np.max(np.abs(v_c - origin))) * 1000.0 > FLAG_MM:
        flags.append(
            f"origin↔vis centre Δ={(1000*np.abs(v_c-origin)).round(2)} mm"
        )

    def _mm3(a):
        if a is None:
            return None
        return (np.asarray(a, dtype=np.float64) * 1000.0).round(2).tolist()

    return {
        "path": str(path),
        "category": path.parent.name,
        "stem": path.stem,
        "visual_size_mm": _mm3(vs),
        "collision_size_mm": _mm3(cs),
        "root_bbox_size_mm": _mm3(root_size if np.all(np.isfinite(root_size)) else None),
        "visual_centre_mm": _mm3(v_c),
        "collision_centre_mm": _mm3(c_c),
        "root_centre_mm": _mm3(root_centre if np.all(np.isfinite(root_centre)) else None),
        "origin_mm": _mm3(origin),
        "scale": np.asarray(scale, dtype=np.float64).round(4).tolist(),
        "mass_kg": None if mass is None else float(mass),
        "density": None if density is None else float(density),
        "static_friction": None if static_f is None else float(static_f),
        "dynamic_friction": None if dynamic_f is None else float(dynamic_f),
        "restitution": None if restitution is None else float(restitution),
        "collision_approx": None if approx is None else str(approx),
        "collider_types": sorted(set(collider_types)),
        "vis_col_size_diff_mm": None if size_diff is None else size_diff.round(2).tolist(),
        "vis_col_centre_diff_mm": None
        if centre_diff is None
        else centre_diff.round(2).tolist(),
        "flags": flags,
        "collision_width_mm": None
        if cs is None
        else float(max(cs[0], cs[1]) * 1000.0),
        "visual_width_mm": None
        if vs is None
        else float(max(vs[0], vs[1]) * 1000.0),
        "root_width_mm": None
        if not np.all(np.isfinite(root_size))
        else float(max(root_size[0], root_size[1]) * 1000.0),
    }


def build_report(rows: list[dict], *, max_width_mm: float = 70.0) -> str:
    lines = [
        "# USD object audit (static)",
        "",
        f"**Flag threshold:** > {FLAG_MM:.0f} mm visual↔collision or origin↔centre.  ",
        f"**Width gate proposal:** collision XY max ≤ {max_width_mm:.0f} mm (80 mm stroke).",
        "",
        "## Per mesh",
        "",
        "| cat | stem | vis WH mm | col WH mm | root WH | "
        "Δsize max mm | origin↔col mm | mass | fric | approx/type | flags |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    flagged = []
    for r in rows:
        if "error" in r:
            lines.append(f"| {r.get('category','?')} | {r.get('stem','?')} | ERR | | | | | | | | {r['error']} |")
            continue
        vs = r["visual_size_mm"]
        cs = r["collision_size_mm"]
        vwh = "—" if vs is None else f"{vs[0]:.1f}×{vs[1]:.1f}×{vs[2]:.1f}"
        cwh = "—" if cs is None else f"{cs[0]:.1f}×{cs[1]:.1f}×{cs[2]:.1f}"
        dsz = r["vis_col_size_diff_mm"]
        dsz_m = "—" if dsz is None else f"{max(dsz):.1f}"
        oc = r["collision_centre_mm"]
        orig = r["origin_mm"]
        oc_d = "—"
        if oc is not None and orig is not None:
            oc_d = f"{max(abs(a - b) for a, b in zip(oc, orig)):.1f}"
        fr = r["static_friction"]
        fr_s = "—" if fr is None else f"{fr:.2f}"
        approx = r["collision_approx"] or ",".join(r["collider_types"]) or "—"
        fl = "; ".join(r["flags"]) if r["flags"] else "—"
        if r["flags"]:
            flagged.append(r)
        root_w = "—" if r["root_width_mm"] is None else f"{r['root_width_mm']:.1f}"
        mass_s = "—" if r["mass_kg"] is None else f"{r['mass_kg']:.3f}"
        lines.append(
            f"| {r['category']} | {r['stem']} | {vwh} | {cwh} | {root_w} | "
            f"{dsz_m} | {oc_d} | {mass_s} | {fr_s} | {approx} | {fl} |"
        )

    lines += ["", "## Flagged (>1 mm)", ""]
    if not flagged:
        lines.append("None.")
    else:
        for r in flagged:
            lines.append(f"- **{r['category']}/{r['stem']}**: {'; '.join(r['flags'])}")

    lines += [
        "",
        "## Width gate (collision XY)",
        "",
        f"| cat | n | pass ≤{max_width_mm:.0f} mm | fail | notes |",
        "|---|---:|---:|---:|---|",
    ]
    by_cat: dict[str, list] = {}
    for r in rows:
        if "error" in r:
            continue
        by_cat.setdefault(r["category"], []).append(r)
    for cat in sorted(by_cat):
        rs = by_cat[cat]
        ok = [r for r in rs if r["collision_width_mm"] is not None and r["collision_width_mm"] <= max_width_mm + 1e-9]
        bad = [r for r in rs if r not in ok]
        note = ""
        if bad and ok:
            note = f"exclude {len(bad)} wide; keep {len(ok)}"
        elif not ok:
            note = "ALL fail — category empty under 70 mm collision gate"
        lines.append(f"| {cat} | {len(rs)} | {len(ok)} | {len(bad)} | {note} |")

    lines += [
        "",
        "## Fix pointers",
        "",
        "1. Size / w_pre / z_grasp half-height → **collision** bbox (not root/visual).",
        "2. If origin↔col centre > 1 mm → recenter USD (or offset spawn by −centre).",
        f"3. Gate width on collision XY ≤ {max_width_mm:.0f} mm (stroke 80 mm).",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--object-dir", type=Path, default=DEFAULT_OBJ)
    ap.add_argument("--categories", type=str, default=",".join(SCOUT))
    ap.add_argument("--max-width-mm", type=float, default=70.0)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    cats = [c.strip() for c in args.categories.split(",") if c.strip()]
    rows = []
    for cat in cats:
        d = args.object_dir / cat
        if not d.is_dir():
            rows.append({"category": cat, "stem": "?", "error": f"missing dir {d}"})
            continue
        for usd in sorted(d.glob("*.usd")):
            rows.append(audit_usd(usd))
    text = build_report(rows, max_width_mm=float(args.max_width_mm))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    jpath = args.json or args.out.with_suffix(".json")
    jpath.write_text(json.dumps(rows, indent=2))
    n_flag = sum(1 for r in rows if r.get("flags"))
    print(f"wrote {args.out} ({len(rows)} meshes, {n_flag} flagged)")
    print(f"wrote {jpath}")


if __name__ == "__main__":
    main()

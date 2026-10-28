"""In-repo Franka Phase-0 scene (Isaac Lab). Import only AFTER AppLauncher.

Owns: Franka Panda + house USD table + one rigid object + static container
+ static/wrist cameras + DiffIK.

Loads a random house USD (paths.scene_dir), moves it so a free long table
edge sits at the world origin (robot stays at 0,0,0), randomizes lights, and
aims throws across a catch band. Misses coast off the table.

Grasp (``scene.use_weld_grasp``): default = real PhysX (fingers close, no weld;
object is scripted until the expert closes, then free). Optional weld override.

Meshes (object + container USDs + house) are fixed for the process lifetime
(plan 2A). Diversify across workers / restarts via asset_seed.

Runs once at start (build the world), then every control step (move things, step physics).
Timing: control step ``sim.dt`` (claim default 30 Hz: expert, policy, cameras, logs).
``sim.physics_substeps`` PhysX steps run inside each control step.
Each camera update stamps ``last_capture_t0`` (sim time).
Why here: Foresight owns the scene and the object's motion.
Must not: pick or judge grasps. The expert and success rules live elsewhere.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply, subtract_frame_transforms
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from interfaces.state import ObjectState, oracle_from_gt
from sim.motion import kinematic_advance, physics_substeps
from sim.physics_grasp import bind_grip_material, gripper_depth_m, xy_outside_box

logger = logging.getLogger(__name__)

ARM_JOINT_EXPR = "panda_joint"
FINGER_JOINTS = ("panda_finger_joint1", "panda_finger_joint2")
EE_BODY = "panda_hand"


def _require(cfg: dict, *keys: str) -> Any:
    """Read a nested yaml value. On a miss, log the full key path and raise.

    Why: no silent fallbacks. A wrong config should stop the run, not hide.
    """
    node: Any = cfg
    path: list[str] = []
    for k in keys:
        path.append(str(k))
        try:
            node = node[k]
        except (KeyError, TypeError):
            logger.error("missing required yaml key: %s", ".".join(path))
            raise KeyError(".".join(path)) from None
    return node


def load_cfg(path: str | Path) -> dict:
    """Read the yaml config into a dict (mm/ms in file → metres/seconds here).

    No defaults are filled in. A missing key fails loudly later (see ``_require``),
    so a typo cannot quietly change an experiment.
    """
    from interfaces.cfg_units import normalize_cfg_units

    with open(path) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    return normalize_cfg_units(cfg)


def _euler_deg_to_wxyz(rotation_deg: list[float]) -> list[float]:
    """Turn camera angles in degrees (easy to write in yaml) into the quaternion Isaac wants."""
    from scipy.spatial.transform import Rotation as R

    q = R.from_euler("XYZ", rotation_deg, degrees=True).as_quat()  # xyzw -> wxyz
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])] # wxyz


def list_category_usds(
    object_dir: Path,
    categories: list[str],
    *,
    exclude: set[str] | None = None,
) -> list[Path]:
    """List the .usd mesh files in each category folder (e.g. can/, bowl/).

    ``exclude`` skips bad meshes by stem, category folder, or stem prefix
    (e.g. ``tomato`` drops ``tomato03.usd``). Raises if nothing is left,
    so we never start a run with an empty catalog.
    """
    from sim.asset_filters import stem_or_category_banned

    ban = {s.lower().removesuffix(".usd") for s in (exclude or set())}
    usds: list[Path] = []
    for cat in categories:
        cat_dir = object_dir / cat
        if not cat_dir.is_dir():
            continue
        for p in sorted(cat_dir.glob("*.usd")):
            if stem_or_category_banned(p, ban):
                continue
            usds.append(p)
    if not usds:
        raise FileNotFoundError(
            f"No USD objects under {object_dir} for categories={categories} "
            f"(exclude={sorted(ban)})"
        )
    return usds


def _filter_graspable_usds(
    catalog: list[Path],
    *,
    max_width_m: float,
) -> list[Path]:
    """Drop meshes wider than the open gripper (plan §5 item 6d)."""
    from sim.asset_filters import object_fits_gripper
    from sim.state_machine_logs import fmt_size_cm

    kept: list[Path] = []
    for p in catalog:
        try:
            size = usd_bbox_size(p)
        except RuntimeError as ex:
            logger.warning("Skip %s (bbox failed): %s", p.name, ex)
            continue
        if object_fits_gripper(size, max_width_m):
            kept.append(p)
        else:
            logger.info(
                "Skip object %s — width %s > gripper max %.0f mm",
                p.stem,
                fmt_size_cm(size),
                1000.0 * max_width_m,
            )
    return kept


def _filter_holdable_containers(
    catalog: list[Path],
    object_size: np.ndarray,
    *,
    min_clearance_m: float,
    min_xy_margin_m: float,
    open_categories: set[str] | None = None,
    finger_clearance_m: float = 0.0,
) -> list[Path]:
    """Drop flat / tiny containers that cannot keep this object after place."""
    from sim.asset_filters import container_can_hold
    from sim.state_machine_logs import fmt_size_cm

    open_cats = {c.lower() for c in (open_categories or set())}
    kept: list[Path] = []
    for p in catalog:
        try:
            size = usd_bbox_size(p)
        except RuntimeError as ex:
            logger.warning("Skip container %s (bbox failed): %s", p.name, ex)
            continue
        open_top = p.parent.name.lower() in open_cats
        if container_can_hold(
            object_size,
            size,
            min_clearance_m=min_clearance_m,
            min_xy_margin_m=min_xy_margin_m,
            open_top=open_top,
            finger_clearance_m=finger_clearance_m,
        ):
            kept.append(p)
        else:
            logger.info(
                "Skip container %s — size %s cannot hold object %s",
                p.stem,
                fmt_size_cm(size),
                fmt_size_cm(object_size),
            )
    return kept


def usd_bbox_size(usd_path: Path) -> np.ndarray:
    """Collision AABB size (m) for w_pre / sit height / width gate.

    Prefer CollisionAPI prims (pads touch the collider, not the render mesh).
    Falls back to visual if the USD has no collider. Raises on any problem —
    a wrong size would bury/float the object or mis-size the pre-shape.
    """
    try:
        from sim.usd_bbox import usd_collision_size

        size = usd_collision_size(usd_path)
        if not np.all(np.isfinite(size)) or np.any(size <= 0):
            raise RuntimeError(f"USD bbox failed: invalid size {size} for {usd_path}")
        return size
    except RuntimeError:
        raise
    except Exception as ex:
        logger.error("USD bbox failed for %s: %s", usd_path, ex)
        raise RuntimeError(f"USD bbox failed for {usd_path}") from ex


def half_height_from_size(size: np.ndarray) -> float:
    """How high the object's center sits above whatever it rests on (half its height).

    Tiny floor of 1 mm so a flat mesh is never at zero height.
    """
    hz = float(size[2]) * 0.5 if size.shape[0] >= 3 else float(np.max(size)) * 0.5
    return float(max(hz, 1e-3))


def pick_usd(
    catalog: list[Path],
    rng: np.random.Generator,
    override: str | Path | None,
    *,
    object_dir: Path | None = None,
) -> Path:
    """Choose which mesh to use. Random from the catalog, or what the user asked for.

    The seeded ``rng`` keeps the random choice repeatable.
    Override may be:
      - absolute/relative path to a ``.usd`` file
      - category folder name under ``object_dir`` (e.g. ``can``, ``bowl``)
      - USD stem / filename (e.g. ``can00`` or ``can00.usd``) matched in catalog
    """
    if override is None or str(override).strip() == "":
        idx = int(rng.integers(0, len(catalog)))
        return catalog[idx]

    raw = str(override).strip()
    path = Path(raw)
    if path.is_file():
        return path

    # Category shorthand: --object-usd can → random USD under objects/can/
    # Only use the filtered catalog. Never fall back to the raw folder — that
    # re-admits excluded / too-small meshes and then crashes on the fit check.
    if object_dir is not None:
        cat_dir = object_dir / raw
        if cat_dir.is_dir():
            pool = [p for p in catalog if p.parent.resolve() == cat_dir.resolve()]
            if not pool:
                raise FileNotFoundError(
                    f"No usable .usd left for category {raw!r} after exclude/size "
                    f"filters (dir exists: {cat_dir}). Pick another category or "
                    f"relax scene fit knobs."
                )
            return pool[int(rng.integers(0, len(pool)))]

    stem = path.stem if raw.endswith(".usd") else raw
    matches = [p for p in catalog if p.stem == stem or p.name == raw]
    if matches:
        return matches[int(rng.integers(0, len(matches)))]

    raise FileNotFoundError(
        f"USD override not found: {raw!r}. "
        "Pass a .usd path, a category name (e.g. can / bowl), or a stem (e.g. can00)."
    )


class Phase0Scene:
    """The whole Phase-0 world: Franka arm, table, one moving object, one box, cameras.

    The object is moved by our script (not by physics) so its speed is exact.
    One env only. The expert calls this; this class does not decide what to do.
    """

    def __init__(
        self,
        cfg: dict,
        *,
        enable_cameras: bool = True,
        object_usd: Path | str | None = None,
        container_usd: Path | str | None = None,
        asset_seed: int = 40,
        distractor_usd: Path | str | None = None,
    ):
        """Build everything once: arm, table, object, box, cameras, IK.

        Yaml: ``sim.dt/device``, ``scene.*``, ``robot.*``, ``static_camera`` /
        ``wrist_camera``, ``paths.*``.
        Meshes are fixed after this. To get a different one, restart with a new
        ``asset_seed`` or pass a USD override.
        """
        self.cfg = cfg
        self.enable_cameras = bool(enable_cameras)
        device = str(_require(cfg, "sim", "device"))
        dt = float(_require(cfg, "sim", "dt"))
        self.dt = dt
        self.device = device
        self._asset_seed = int(asset_seed)

        sc = _require(cfg, "scene")

        # Control rate stays dt (expert, policy, cameras, logs). Physics runs
        # ``n_substeps`` smaller steps inside each control step.
        # render_interval = n: one render per control step
        # (Isaac Lab DirectRLEnv uses the same decimation pattern).
        self.n_substeps = physics_substeps(cfg)
        self.physics_dt = dt / self.n_substeps

        # Isaac calls must come after AppLauncher has started (see module top).
        if self.n_substeps == 1:
            sim_cfg = sim_utils.SimulationCfg(dt=dt, device=device)
        else:
            sim_cfg = sim_utils.SimulationCfg(
                dt=self.physics_dt, render_interval=self.n_substeps, device=device
            )
        self.sim = sim_utils.SimulationContext(sim_cfg)
        logger.info(
            "Physics %.0f Hz (%d substeps per %.0f ms control step)",
            1.0 / self.physics_dt,
            self.n_substeps,
            1000.0 * dt,
        )
        self._house_meta: dict[str, Any] = {}
        self._play_layout = None
        self._table_box: tuple[np.ndarray, np.ndarray] | None = None  # table XY footprint
        # Weld = optional scripted glue. Default = real PhysX grasp.
        self._weld = bool(sc.get("use_weld_grasp", False))

        # One seeded rng for house + mesh choice (same asset_seed → same world).
        asset_rng = np.random.default_rng(self._asset_seed)

        # Invisible floor under the table (slightly low so it does not fight z=0).
        sim_utils.GroundPlaneCfg().func(
            "/World/GroundPlane",
            sim_utils.GroundPlaneCfg(),
            translation=(0.0, 0.0, -0.1),
        )

        self._setup_house_world(cfg, asset_rng)

        robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.copy()
        robot_cfg.prim_path = "/World/Robot"
        # Base sits at the table-edge anchor (house is moved so that anchor = origin).
        robot_cfg.init_state.pos = (0.0, 0.0, 0.0)
        # Optional PD override (s1003: hand hunted on a smooth aim → more damping).
        robot_yaml = cfg.get("robot") if isinstance(cfg.get("robot"), dict) else {}
        stiff = robot_yaml.get("arm_joint_stiffness")
        damp = robot_yaml.get("arm_joint_damping")
        if stiff is not None or damp is not None:
            for name in ("panda_shoulder", "panda_forearm"):
                act = robot_cfg.actuators[name]
                if stiff is not None:
                    act.stiffness = float(stiff)
                if damp is not None:
                    act.damping = float(damp)
        self.robot = Articulation(robot_cfg)

        self._object_dir = Path(_require(cfg, "paths", "object_dir"))
        self._categories = list(_require(cfg, "scene", "object_categories"))
        self._container_categories = list(_require(cfg, "scene", "container_categories"))
        self._use_primitive = bool(_require(cfg, "scene", "use_primitive"))
        self._object_radius = float(_require(cfg, "scene", "object_radius"))
        self._n_containers = int(_require(cfg, "scene", "n_containers"))
        self._place_hover = float(_require(cfg, "scene", "place_hover_above_container"))
        self._spawn_xy_pad = float(_require(cfg, "scene", "spawn_xy_pad"))
        self._container_jitter_xy = float(_require(cfg, "scene", "container_jitter_xy"))
        ctr_clip = _require(cfg, "scene", "container_xy_clip")
        self._container_xy_clip = (
            (float(ctr_clip[0][0]), float(ctr_clip[0][1])),
            (float(ctr_clip[1][0]), float(ctr_clip[1][1])),
        )
        self._attach_z_offset_max = float(_require(cfg, "scene", "attach_z_offset_max"))
        from sim.asset_filters import merge_exclude_names

        sc = cfg.get("scene", {}) if isinstance(cfg.get("scene"), dict) else {}
        # Name bans: legacy usd_exclude + per-role lists (tomato, plate, …).
        self._usd_exclude_objects = merge_exclude_names(
            sc.get("usd_exclude"), sc.get("exclude_usd_objects")
        )
        self._usd_exclude_containers = merge_exclude_names(
            sc.get("usd_exclude"), sc.get("exclude_usd_containers")
        )
        # Franka open width ≈ 2 * gripper_open (each finger). Leave a small margin.
        grip_open = float(_require(cfg, "robot", "gripper_open"))
        self._max_object_width_m = float(
            sc.get("max_object_width_m", max(0.0, 2.0 * grip_open - 0.005))
        )
        self._min_container_clearance_m = float(sc.get("min_container_clearance_m", 0.01))
        self._min_container_xy_margin_m = float(sc.get("min_container_xy_margin_m", 0.01))
        # Closed fingers must fit across the container (they open along world Y).
        self._container_finger_clearance_m = float(
            sc.get("min_container_finger_clearance_m", 0.0)
        )
        self._open_container_categories = {
            str(c).lower() for c in (sc.get("open_container_categories") or [])
        }
        self._robot_base_keepout_m = float(sc.get("robot_base_keepout_m", 0.12))
        self._ee_table_margin_m = float(sc.get("ee_table_margin_m", 0.02))
        self._object_usds: list[Path] = []
        self._active_usd: Path | None = None
        self._object_size: np.ndarray | None = None
        self._object_half_height = self._object_radius
        # Collision AABB centre in USD root frame (m). Spawn root = aim − this
        # so the collider centre sits on the expert aim point.
        self._object_centre_offset = np.zeros(3, dtype=np.float64)
        # Per-episode sit height (tipped cans use diameter/2, not upright height/2).
        self._episode_object_half_height = float(self._object_half_height)
        self.abort_reason: str | None = None  # set mid-episode by safety checks

        if not self._use_primitive:
            if self._usd_exclude_objects:
                logger.info(
                    "USD object exclude: %s", sorted(self._usd_exclude_objects)
                )
            raw_objects = list_category_usds(
                self._object_dir,
                self._categories,
                exclude=self._usd_exclude_objects,
            )
            # Size gate (plan 6d): drop meshes wider than the open gripper.
            self._object_usds = _filter_graspable_usds(
                raw_objects, max_width_m=self._max_object_width_m
            )
            if not self._object_usds:
                raise FileNotFoundError(
                    f"No graspable objects left after width gate "
                    f"({1000.0 * self._max_object_width_m:.0f} mm) from {len(raw_objects)} meshes"
                )
            self._active_usd = pick_usd(
                self._object_usds,
                asset_rng,
                object_usd,
                object_dir=self._object_dir,
            )
            from sim.usd_bbox import usd_visual_collision_boxes

            boxes = usd_visual_collision_boxes(self._active_usd)
            self._object_size = np.asarray(boxes["size"], dtype=np.float64)
            # Spawn so the collision centre sits on the aim point (not USD origin).
            self._object_centre_offset = np.asarray(boxes["centre"], dtype=np.float64)
            # Override path may bypass the catalog filter — re-check.
            from sim.asset_filters import object_fits_gripper
            from sim.state_machine_logs import fmt_size_cm

            if not object_fits_gripper(self._object_size, self._max_object_width_m):
                raise RuntimeError(
                    f"Object {self._active_usd.stem} size {fmt_size_cm(self._object_size)} "
                    f"wider than gripper max {1000.0 * self._max_object_width_m:.0f} mm"
                )
            self._object_half_height = half_height_from_size(self._object_size)

            logger.info(
                "Object mesh: %s   collision size %s (%s)   sits %.1f cm on the table",
                self._active_usd.stem if self._active_usd is not None else "unknown",
                fmt_size_cm(self._object_size),
                boxes["used"],
                100.0 * float(self._object_half_height),
            )

        # Table top Z. Layout maps the chosen table edge to z=0.
        if self._play_layout is None:
            raise RuntimeError("house layout missing after _setup_house_world")
        self._table_z = float(self._play_layout.table_z)

        # Start resting on the table top. Real start pose is set in reset_episode.
        z0 = self._table_z + self._object_half_height
        self.object = self._make_rigid(
            prim_path="/World/Object",
            usd_path=self._active_usd,
            pos=np.array([0.50, 0.0, z0]),
            lin_vel=np.zeros(3),
            mass=float(_require(cfg, "scene", "object_mass")),
            # Weld mode: kinematic body, motion is ours (pose write). Physics grasp:
            # dynamic body; we still write the pose (velocity 0, so PhysX does not add
            # a second v·dt) until the expert closes, then PhysX owns it.
            kinematic=self._weld,
            primitive_sphere=self._use_primitive,
        )

        if not self._weld:
            bind_grip_material("/World/Object", float(_require(cfg, "scene", "object_friction")))

        # E3 option: a second, static object on the table (a distractor for language tests).
        # Kinematic and never grasp-tracked; logged in meta only. Off unless a USD is given.
        self.distractor: RigidObject | None = None
        self._distractor_usd: Path | None = Path(distractor_usd) if distractor_usd else None
        self._distractor_half_height = 0.0
        if self._distractor_usd is not None:
            dsize = usd_bbox_size(self._distractor_usd)
            self._distractor_half_height = half_height_from_size(dsize)
            self.distractor = self._make_rigid(
                prim_path="/World/Distractor",
                usd_path=self._distractor_usd,
                pos=np.array([0.50, 0.30, self._table_z + self._distractor_half_height]),
                lin_vel=np.zeros(3),
                mass=float(_require(cfg, "scene", "object_mass")),
                kinematic=True,
                primitive_sphere=False,
            )
        self._distractor_pos: np.ndarray | None = None

        self.container: RigidObject | None = None
        self._container_usd: Path | None = None
        self._container_size: np.ndarray | None = None
        self._container_half_height = 0.0
        ctr_xy = _require(cfg, "scene", "container_xy")
        self._container_pos = np.array(
            [float(ctr_xy[0]), float(ctr_xy[1]), 0.0],
            dtype=np.float64,
        )

        # The box the object is placed into. Also kinematic: it stays where we put it.
        if (not self._use_primitive) and self._n_containers >= 1 and self._container_categories:
            if self._usd_exclude_containers:
                logger.info(
                    "USD container exclude: %s", sorted(self._usd_exclude_containers)
                )
            raw_ctrs = list_category_usds(
                self._object_dir,
                self._container_categories,
                exclude=self._usd_exclude_containers,
            )
            assert self._object_size is not None
            ctr_usds = _filter_holdable_containers(
                raw_ctrs,
                self._object_size,
                min_clearance_m=self._min_container_clearance_m,
                min_xy_margin_m=self._min_container_xy_margin_m,
                open_categories=self._open_container_categories,
                finger_clearance_m=self._container_finger_clearance_m,
            )
            if not ctr_usds:
                raise FileNotFoundError(
                    f"No holdable containers left for object size "
                    f"{self._object_size.tolist()} from {len(raw_ctrs)} meshes"
                )
            self._container_usd = pick_usd(
                ctr_usds,
                asset_rng,
                container_usd,
                object_dir=self._object_dir,
            )
            self._container_size = usd_bbox_size(self._container_usd)
            from sim.asset_filters import container_can_hold
            from sim.state_machine_logs import fmt_size_cm

            open_top = (
                self._container_usd.parent.name.lower()
                in self._open_container_categories
            )
            if not container_can_hold(
                self._object_size,
                self._container_size,
                min_clearance_m=self._min_container_clearance_m,
                min_xy_margin_m=self._min_container_xy_margin_m,
                open_top=open_top,
                finger_clearance_m=self._container_finger_clearance_m,
            ):
                raise RuntimeError(
                    f"Container {self._container_usd.stem} size "
                    f"{fmt_size_cm(self._container_size)} cannot hold object "
                    f"{fmt_size_cm(self._object_size)}"
                )
            self._container_half_height = half_height_from_size(self._container_size)
            cz = self._table_z + self._container_half_height
            self._container_pos = np.array(
                [float(ctr_xy[0]), float(ctr_xy[1]), cz], dtype=np.float64
            )
            self.container = self._make_rigid(
                prim_path="/World/Container",
                usd_path=self._container_usd,
                pos=self._container_pos.copy(),
                lin_vel=np.zeros(3),
                mass=float(_require(cfg, "scene", "container_mass")),
                kinematic=True,
                primitive_sphere=False,
            )
            from sim.state_machine_logs import fmt_size_cm

            logger.info(
                "Box mesh: %s   size %s   sits %.1f cm on the table",
                self._container_usd.stem if self._container_usd is not None else "unknown",
                fmt_size_cm(self._container_size),
                100.0 * float(self._container_half_height),
            )

        # The expert's last stage is "place". No box means no valid task, so stop now.
        if self.container is None:
            raise RuntimeError(
                "Place needs a container. Set scene.n_containers >= 1, "
                "scene.container_categories, and use_primitive: false."
            )

        self.cameras: dict[str, Camera] = {}
        if self.enable_cameras:
            self._spawn_cameras()

        self.sim.set_camera_view(eye=[1.5, 1.2, 1.0], target=[0.45, 0.0, 0.2])
        # reset() once, after every asset exists. Body/joint indexes are only valid after this.
        self.sim.reset()

        self._ee_idx = self.robot.data.body_names.index(EE_BODY)
        names = list(self.robot.data.joint_names)
        self._arm_idx = [names.index(n) for n in names if n.startswith(ARM_JOINT_EXPR)]
        self._finger_idx = [names.index(n) for n in FINGER_JOINTS]
        # The arm's Jacobian skips the fixed base body, so its row is one lower.
        self._jac_idx = self._ee_idx - 1

        # TCP = the point between the fingertips. We aim IK at it, not at the hand base.
        tcp = _require(cfg, "robot", "tcp_offset")
        self._tcp_offset = torch.tensor(
            tcp, device=self.robot.device, dtype=torch.float32
        ).unsqueeze(0)
        self._tcp_offset_quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.robot.device, dtype=torch.float32
        ).unsqueeze(0)

        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"
        )
        self.ik = DifferentialIKController(ik_cfg, num_envs=1, device=self.robot.device)
        # Velocity feedforward gain K (1/s). 0 = absolute pose only.
        # See set_ee_target: v_cmd = v + K·(p* − p), IK aims at p + v_cmd/K.
        robot_cfg = cfg.get("robot") or {}
        self._diffik_ff_k = float(robot_cfg.get("diffik_ff_gain", 0.0))
        # Joint velocity feedforward (replaces the v/K position lead when on).
        # The drive is τ = k(q* − q) + d(q̇* − q̇). With q̇* = 0 the damping brakes
        # every moving joint → the hand trails by d/k = 200 ms. Here q̇* = J⁺·v̂
        # and q* ramps by q̇*·dt_phys every substep (to the substep END: PhysX
        # implicit drives solve for the end of the substep), so a held per-tick
        # setpoint no longer costs half a control step either. Probe
        # (scratch probe_lag_variants, Oct 28): 30 cm/s trail −7.9 → +0.2 mm.
        self._joint_vel_ff = bool(robot_cfg.get("joint_velocity_ff", False))
        self._ramp_q0: torch.Tensor | None = None  # arm q* at the start of the tick
        self._ramp_qd: torch.Tensor | None = None  # arm q̇* (refreshed every substep)
        self._ramp_v = np.zeros(3)  # TCP velocity (world) the ramp follows this tick
        # Spawn filter (reset_episode): re-draw throws whose ride needs j7 within
        # this many rad of a limit. 0 = off.
        scene_cfg = cfg.get("scene", {}) if isinstance(cfg.get("scene"), dict) else {}
        self._spawn_j7_margin = float(scene_cfg.get("spawn_j7_margin_rad", 0.0))
        self._spawn_max_redraws = int(scene_cfg.get("spawn_max_redraws", 20))
        self._spawn_redraws, self._last_ride_j7_margin = 0, None

        # Three object velocities on purpose:
        #   _obj_velocity  — live speed that drives the coast
        #   _obj_speed_cmd — same speed latched, so a grab/release does not lose it
        #   _obj_vel_obs   — measured after the step; what logs and the policy see
        self._obj_velocity = np.zeros(3, dtype=np.float64)
        self._obj_speed_cmd = np.zeros(3, dtype=np.float64)
        # Observed velocity after the last physics step (FD of root pose, or
        # root_lin_vel_w). Commanded motion speed stays in ``_obj_velocity``.
        self._obj_vel_obs = np.zeros(3, dtype=np.float64)
        self._prev_obj_pos: np.ndarray | None = None
        self._coast_pos: np.ndarray | None = None  # physics grasp: scripted path position
        # Scripted path orientation (spawn quat). Pinned with _coast_pos: reading the
        # PhysX quat back let table contact tip eggs/lemons while z stayed at
        # table + half-height (top 2–9 mm low at 10–30 cm/s, A4).
        self._coast_quat: np.ndarray | None = None
        self._obj_kinematic = True
        self._phys_free = False  # physics grasp: True once PhysX owns the object
        self.off_table = False  # object left the table footprint this episode
        self.abort_reason = None  # mid-episode safety stop (ee off table / through box)
        self._clear_play_band_of_container(cfg)
        bounds = _require(cfg, "scene", "object_xy_bounds")
        self._xy_bounds = (
            (float(bounds[0][0]), float(bounds[0][1])),
            (float(bounds[1][0]), float(bounds[1][1])),
        )
        self._sim_time = 0.0
        # Sim time of the last camera render (Step 1 frame stamp t0).
        self._last_capture_t0 = 0.0
        self._gripper_open = float(_require(cfg, "robot", "gripper_open"))
        self._gripper_close = float(_require(cfg, "robot", "gripper_close"))
        # Last commanded open∈[0,1]. Attach grasp snaps fingers to this (no squeeze).
        self._last_gripper_open = 1.0
        self._meta: dict[str, Any] = {}
        self._attached = False
        self._attach_offset = np.zeros(3, dtype=np.float64)
        # Attach = scripted "weld" of object to hand. Freeze = pin the object in
        # place after it is dropped in the box so nothing can knock it around.
        self._freeze_pose: tuple[np.ndarray, np.ndarray] | None = None
        self._was_frozen = False

    def _setup_house_world(self, cfg: dict, asset_rng: np.random.Generator) -> None:
        """Load a random house, put the robot at a free table edge, set play bounds.

        The house is moved so that edge is world origin (robot stays at 0,0,0).
        Table top becomes z=0. Lights are randomized. No DynamicVLA import.
        """
        from sim.house_layout import (
            compute_play_layout,
            look_at_quat_wxyz,
            pick_house,
            sample_light,
        )

        scene_dir = Path(_require(cfg, "paths", "scene_dir"))
        static = _require(cfg, "static_camera")
        # Collision filter only needs camera XYZ relative to the robot mount.
        cam_specs = [{"name": "static_cam", "position": list(static["position"])}]

        max_reach = float(_require(cfg, "robot", "max_reach_dist"))
        # Retry houses until play+bowl both fit (narrow tables get skipped).
        house = None
        layout = None
        last_err: Exception | None = None
        for _try in range(40):
            try:
                cand = pick_house(scene_dir, asset_rng, cam_specs, max_tries=1)
                # Prefer yaml container_xy (meters after cfg_units); else house defaults.
                bowl_kw: dict = {}
                ctr = cfg["scene"].get("container_xy")
                if ctr is not None and len(ctr) >= 2:
                    bowl_kw["container_x"] = float(ctr[0])
                    bowl_kw["container_y"] = float(ctr[1])
                layout = compute_play_layout(
                    cand.table_bbox_min_w,
                    cand.table_bbox_max_w,
                    # Full arm reach: old min(..., 0.75) clipped far-edge spawns
                    # to mid-table on short coffee tables.
                    max_reach_dist=float(max_reach),
                    **bowl_kw,
                )
                house = cand
                break
            except Exception as ex:
                last_err = ex
                continue
        if house is None or layout is None:
            raise RuntimeError(
                f"No house with room for play+bowl under {scene_dir} "
                f"(last error: {last_err})"
            )

        usd_cfg = sim_utils.UsdFileCfg(usd_path=str(house.usd_path))
        usd_cfg.func(
            "/World/House",
            usd_cfg,
            translation=tuple(float(x) for x in house.spawn_pos),
            orientation=tuple(float(x) for x in house.spawn_quat_wxyz),
        )
        self._play_layout = layout
        self._table_box = (
            np.asarray(house.table_bbox_min_w[:2], dtype=np.float64),
            np.asarray(house.table_bbox_max_w[:2], dtype=np.float64),
        )

        # Push layout into cfg so the expert (built after us) sees the same walls.
        cfg["scene"]["table_height"] = float(layout.table_z)
        cfg["scene"]["object_xy_bounds"] = [
            [float(layout.object_xy_bounds[0][0]), float(layout.object_xy_bounds[0][1])],
            [float(layout.object_xy_bounds[1][0]), float(layout.object_xy_bounds[1][1])],
        ]
        cfg["scene"]["container_xy"] = [
            float(layout.container_xy[0]),
            float(layout.container_xy[1]),
        ]
        cfg["scene"]["container_xy_clip"] = [
            [float(layout.container_xy_clip[0][0]), float(layout.container_xy_clip[0][1])],
            [float(layout.container_xy_clip[1][0]), float(layout.container_xy_clip[1][1])],
        ]
        self._container_xy_clip = layout.container_xy_clip

        light = sample_light(
            asset_rng,
            static["position"],
            lighting=cfg.get("lighting"),
        )
        # Soft fill + one hard distant light (DynamicVLA-style diversity).
        dome = sim_utils.DomeLightCfg(
            intensity=350.0,
            enable_color_temperature=True,
            color_temperature=6500.0,
        )
        dome.func("/World/Light", dome)
        # Aim distant light at the table (origin). Right-handed look-at quat.
        light_quat = look_at_quat_wxyz(light["position"], (0.0, 0.0, 0.0))
        dist = sim_utils.DistantLightCfg(
            enable_color_temperature=True,
            color_temperature=float(light["temperature"]),
            intensity=float(light["intensity"]),
        )
        dist.func(
            "/World/DistantLight",
            dist,
            translation=tuple(float(x) for x in light["position"]),
            orientation=tuple(light_quat),
        )
        logger.info(
            "Distant light %s of static_cam at (%.1f, %.1f, %.1f) m  %d K  intensity %.0f",
            light.get("style", "?"),
            light["position"][0],
            light["position"][1],
            light["position"][2],
            int(light["temperature"]),
            float(light["intensity"]),
        )

        self._house_meta = {
            "house_usd": str(house.usd_path),
            "table_name": house.table.name,
            "table_height_src": float(house.table.height),
            "anchor_side": house.anchor.side,
            "object_xy_bounds": cfg["scene"]["object_xy_bounds"],
            "container_xy": cfg["scene"]["container_xy"],
            "light": light,
        }
        logger.info(
            "House %s  table %s  play x=%s y=%s  bowl at %s",
            house.usd_path.name,
            house.table.name,
            cfg["scene"]["object_xy_bounds"][0],
            cfg["scene"]["object_xy_bounds"][1],
            cfg["scene"]["container_xy"],
        )

    def _clear_play_band_of_container(self, cfg: dict) -> None:
        """Shrink the throw/catch band on the container's side to the real box size.

        The house layout runs before the meshes are picked, so it reserves only a
        4 cm container half-width. Real bowls / boxes are 6–12 cm, so on narrow
        tables the box overlapped the catch band and throws aborted as
        ``object-through-container`` before the hand could catch them. Edits
        ``cfg['scene']['object_xy_bounds']`` in place (house meta shares it).
        """
        if self._play_layout is None or self._container_size is None:
            return
        b = cfg["scene"]["object_xy_bounds"]
        y0, y1 = float(b[1][0]), float(b[1][1])
        cy = float(self._container_pos[1])
        obj_half = (
            0.5 * float(max(self._object_size[0], self._object_size[1]))
            if self._object_size is not None
            else 0.0
        )
        # Box half-depth + per-episode nudge + object half-width + 2 cm air gap.
        margin = 0.5 * float(self._container_size[1]) + self._container_jitter_xy + obj_half + 0.02
        new_y0, new_y1 = (y0, min(y1, cy - margin)) if cy >= 0.0 else (max(y0, cy + margin), y1)
        if new_y1 - new_y0 < 0.12:
            raise RuntimeError(
                f"play band too thin after clearing container {self._container_usd}: "
                f"y=({new_y0:.3f},{new_y1:.3f}) (need ≥0.12 m); pick another seed"
            )
        if (new_y0, new_y1) != (y0, y1):
            logger.info(
                "Play band y (%.3f, %.3f) → (%.3f, %.3f) to clear the container",
                y0, y1, new_y0, new_y1,
            )
            b[1][0], b[1][1] = float(new_y0), float(new_y1)

    def _spawn_cameras(self) -> None:
        """Add the cameras that mirror the real rig: fixed table cam + one on the hand.

        Policy sees these images. Needs ``--enable_cameras`` or frames are black.
        Log / feature keys stay ``static_cam`` and ``wrist_cam``.
        """
        static = _require(self.cfg, "static_camera")
        quat = _euler_deg_to_wxyz(static["rotation_deg"])
        self.cameras["static_cam"] = Camera(
            CameraCfg(
                prim_path="/World/static_cam",
                update_period=0.0,
                height=int(static["height"]),
                width=int(static["width"]),
                data_types=list(static["data_types"]),
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=float(static["focal_length"]),
                    focus_distance=float(static["focus_distance"]),
                    horizontal_aperture=float(static["horizontal_aperture"]),
                    clipping_range=(
                        float(_require(self.cfg, "static_camera", "clip", "near")),
                        float(_require(self.cfg, "static_camera", "clip", "far")),
                    ),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=tuple(static["position"]),
                    rot=tuple(quat),
                    convention="opengl",
                ),
            )
        )
        logger.info("Table camera at %s m", tuple(static["position"]))

        # Wrist cam is parented to the hand prim, so it moves with the arm.
        wrist = _require(self.cfg, "wrist_camera")
        wrist_pos = tuple(float(x) for x in wrist["pos"])
        wrist_quat_wxyz = tuple(float(x) for x in wrist["quat_wxyz"])
        wrist_clip = wrist["clipping_range"]
        self.cameras["wrist_cam"] = Camera(
            CameraCfg(
                prim_path="/World/Robot/panda_hand/WristCamera",
                update_period=0.0,
                height=int(wrist["height"]),
                width=int(wrist["width"]),
                # Wrist = RGB-D (Isaac Lab "depth" = distance_to_image_plane, metres).
                data_types=list(wrist["data_types"]),
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=float(wrist["focal_length"]),
                    focus_distance=float(wrist["focus_distance"]),
                    horizontal_aperture=float(wrist["horizontal_aperture"]),
                    clipping_range=(float(wrist_clip[0]), float(wrist_clip[1])),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=wrist_pos,
                    rot=wrist_quat_wxyz,
                    convention="opengl",
                ),
            )
        )
        logger.info("Wrist camera on the hand (%s).", ",".join(wrist["data_types"]))

    def _make_rigid(
        self,
        *,
        prim_path: str,
        usd_path: Path | None,
        pos: np.ndarray,
        lin_vel: np.ndarray,
        mass: float,
        kinematic: bool,
        primitive_sphere: bool,
    ) -> RigidObject:
        """Create the object or the box (a mesh, or a plain sphere for quick tests).

        Both are kinematic in practice (we place them by script).
        """
        # Extra solver iterations: steadier contact between small parts.
        rigid = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
            kinematic_enabled=bool(kinematic),
        )
        collision = sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.01,
            rest_offset=0.0,
        )
        if primitive_sphere:
            spawner = sim_utils.SphereCfg(
                radius=self._object_radius,
                rigid_props=rigid,
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                collision_props=collision,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.15, 0.1)),
            )
        else:
            if usd_path is None:
                raise RuntimeError("USD path required when scene.use_primitive is false")
            spawner = sim_utils.UsdFileCfg(
                usd_path=str(usd_path),
                rigid_props=rigid,
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                collision_props=collision,
            )
        cfg = RigidObjectCfg(
            prim_path=prim_path,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(float(pos[0]), float(pos[1]), float(pos[2])),
                lin_vel=(float(lin_vel[0]), float(lin_vel[1]), float(lin_vel[2])),
            ),
            spawn=spawner,
        )
        return RigidObject(cfg)

    def get_container_pose(self) -> tuple[np.ndarray, float] | None:
        """Where the box is this episode: (center xyz, half height). None if no box."""
        if self.container is None:
            return None
        return self._container_pos.copy(), float(self._container_half_height)

    def get_carry_pose(self) -> np.ndarray:
        """Hand spot for carrying: right above the box, a safe gap over its rim.

        The expert moves here before it lowers the object in.
        """
        if self.container is None:
            raise RuntimeError("Place needs a container; no fallback pose.")
        top_z = float(self._container_pos[2] + self._container_half_height)
        return np.array(
            [
                float(self._container_pos[0]),
                float(self._container_pos[1]),
                top_z + self._place_hover,
            ],
            dtype=np.float64,
        )

    def get_drop_target(self) -> np.ndarray:
        """EE pose inside the box: container XY, Z from floor + object size.

        Stays below the rim and below carry height so lower-into-box goes down, not up.
        """
        if self.container is None:
            raise RuntimeError("Place needs a container; no fallback pose.")
        floor_z = float(self._container_pos[2] - self._container_half_height)
        rim_z = float(self._container_pos[2] + self._container_half_height)
        sit_z = floor_z + float(self._object_half_height)
        # Start from "object sits on the box floor" plus a small gap, then clamp
        # below: under the rim (hand fits in) and under carry height (move goes down).
        gap = float(_require(self.cfg, "state_machine_params", "drop_z_gap"))
        # Prefer drop_rim_clear_m from yaml; fall back to grasp_xyz_ok Z if omitted.
        sm = _require(self.cfg, "state_machine_params")
        grasp_ok = sm["grasp_xyz_ok"]
        place_z = float(grasp_ok[2])
        rim_clear = float(sm.get("drop_rim_clear_m", place_z))
        hover_clear = place_z
        drop_z = sit_z + gap
        if rim_z > sit_z:
            drop_z = min(drop_z, rim_z - rim_clear)
        drop_z = max(drop_z, sit_z)
        carry_z = float(self.get_carry_pose()[2])
        drop_z = min(drop_z, carry_z - hover_clear)
        return np.array(
            [
                float(self._container_pos[0]),
                float(self._container_pos[1]),
                drop_z,
            ],
            dtype=np.float64,
        )

    def reset_episode(self, *, seed: int, speed: float) -> dict[str, Any]:
        """Start a fresh episode: arm to home, object at a random spot, box nudged.

        ``seed`` makes the layout repeatable. ``speed`` (m/s) is the object's constant
        speed for this episode. Returns a meta dict to save with the log.
        """
        rng = np.random.default_rng(int(seed))

        speed = float(speed)
        pad = self._spawn_xy_pad
        sc = self.cfg.get("scene", {}) if isinstance(self.cfg.get("scene"), dict) else {}
        throw_r = float(sc.get("throw_randomness", 1.0))

        # Throw: always +X → −X (toward robot). randomness only spreads the Y angle.
        from sim.house_layout import (
            ROLL_TIP_CATEGORIES,
            sample_throw,
            sit_half_height_m,
            tip_on_side_quat_wxyz,
        )

        # Spawn filter: re-draw the throw while the ride would push the wrist
        # (joint 7) within ride_j7_margin of a limit (sim/ride_ik_check.py).
        spawn_redraws, ride_j7_margin = 0, None
        while True:
            pos0, lin_vel = sample_throw(
                rng,
                self._xy_bounds,
                speed=speed,
                spawn_pad=pad,
                throw_randomness=throw_r,
            )

            # Tip cans/bottles/cups on their side so they can roll (not upright slide).
            # Empty tip_roll_categories → never tip (physics grasp cannot reach the
            # low center of a side-lying can; see phase0_cfg comment).
            tip_cats = sc.get("tip_roll_categories")
            tip_set = (
                {str(c).lower() for c in tip_cats}
                if tip_cats is not None
                else set(ROLL_TIP_CATEGORIES)
            )
            cat = (
                "primitive"
                if self._use_primitive
                else (self._active_usd.parent.name if self._active_usd is not None else "")
            )
            tip = (not self._use_primitive) and cat.lower() in tip_set and speed > 1e-6
            upright = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            size = (
                self._object_size
                if self._object_size is not None
                else np.array([0.06, 0.06, 0.06], dtype=np.float64)
            )
            if tip:
                quat = tip_on_side_quat_wxyz(lin_vel[:2], rng).astype(np.float64)
                sit_h = sit_half_height_m(size, tipped=True)
            else:
                quat = upright.copy()
                sit_h = float(self._object_half_height)
            # Episode sit height (coast keeps this Z). Upright mesh uses mesh half-height.
            self._episode_object_half_height = float(sit_h)
            # Root pose so collision centre is at (xy, table+sit_h).
            off = np.asarray(self._object_centre_offset, dtype=np.float64).reshape(3)
            pos = np.array(
                [
                    float(pos0[0]) - float(off[0]),
                    float(pos0[1]) - float(off[1]),
                    float(self._table_z) + float(sit_h) - float(off[2]),
                ],
                dtype=np.float64,
            )
            # Far-rim spawn is the play +X edge; mesh half-width can hang past the
            # table (egg seed-50194: center 0.78 → PhysX walked it to 0.82 off-table
            # during settle). Pull the center in by half-XY + 5 mm.
            half_xy = 0.5 * float(max(abs(size[0]), abs(size[1])))
            rim_pad = float(half_xy) + 0.005
            if self._table_box is not None:
                pos[0] = min(float(pos[0]), float(self._table_box[1][0]) - rim_pad)
                pos[0] = max(float(pos[0]), float(self._table_box[0][0]) + rim_pad)
                pos[1] = float(
                    np.clip(
                        pos[1],
                        float(self._table_box[0][1]) + rim_pad,
                        float(self._table_box[1][1]) - rim_pad,
                    )
                )
            (bx0, bx1), (by0, by1) = self._xy_bounds
            pos[0] = float(np.clip(pos[0], bx0, bx1))
            pos[1] = float(np.clip(pos[1], by0, by1))
            # Cap spawn inside arm reach. Long tables put the far rim at ~1.5 m
            # (smoke i14); DiffIK never meets the object before the catch-box exit.
            robot_cfg = self.cfg.get("robot", {}) if isinstance(self.cfg.get("robot"), dict) else {}
            max_reach = float(robot_cfg.get("max_reach_dist", 0.85))
            reach_spawn_cap = max(float(bx0) + 0.05, float(max_reach) - 0.05)
            if float(pos[0]) > reach_spawn_cap:
                pos[0] = reach_spawn_cap
            ride_j7_margin = self._ride_j7_margin(pos, lin_vel, speed)
            if (
                ride_j7_margin is None
                or ride_j7_margin >= self._spawn_j7_margin
                or spawn_redraws >= self._spawn_max_redraws
            ):
                break
            spawn_redraws += 1
            logger.info(
                "    spawn re-draw %d: j7 margin %.2f rad < %.2f (seed %d)",
                spawn_redraws, ride_j7_margin, self._spawn_j7_margin, int(seed),
            )
        self._spawn_redraws, self._last_ride_j7_margin = spawn_redraws, ride_j7_margin

        # Small random box shift per episode, clipped to where the arm can reach.
        # House layout already parked the bowl off the throw band.
        if self.container is not None:
            cx0 = float(
                self._play_layout.container_xy[0]
                if self._play_layout is not None
                else self._container_pos[0]
            )
            cy0 = float(
                self._play_layout.container_xy[1]
                if self._play_layout is not None
                else self._container_pos[1]
            )
            j = self._container_jitter_xy
            (cxmin, cxmax), (cymin, cymax) = self._container_xy_clip
            cx = float(np.clip(cx0 + rng.uniform(-j, j), cxmin, cxmax))
            cy = float(np.clip(cy0 + rng.uniform(-j, j), cymin, cymax))
            cz = self._table_z + self._container_half_height
            self._container_pos = np.array([cx, cy, cz], dtype=np.float64)

        self.robot.reset()
        self.object.reset()
        if self.container is not None:
            self.container.reset()
        for cam in self.cameras.values():
            cam.reset()

        home = self.robot.data.default_joint_pos.clone()
        self.robot.write_joint_state_to_sim(home, self.robot.data.default_joint_vel)
        self.robot.set_joint_position_target(home)
        self.robot.set_joint_velocity_target(torch.zeros_like(home))
        self._ramp_q0, self._ramp_qd = None, None

        root = torch.zeros((1, 13), device=self.robot.device, dtype=torch.float32)
        root[0, 0:3] = torch.tensor(pos, device=self.robot.device)
        root[0, 3:7] = torch.tensor(quat, device=self.robot.device)
        # Dynamic body must start at rest in PhysX (we coast it by pose); kinematic keeps lin_vel.
        if self._weld:
            root[0, 7:10] = torch.tensor(lin_vel, device=self.robot.device)
        self.object.write_root_state_to_sim(root)
        self.object.write_data_to_sim()

        if self.container is not None:
            croot = torch.zeros((1, 13), device=self.robot.device, dtype=torch.float32)
            croot[0, 0:3] = torch.tensor(self._container_pos, device=self.robot.device)
            croot[0, 3:7] = torch.tensor(upright, device=self.robot.device)
            self.container.write_root_state_to_sim(croot)
            self.container.write_data_to_sim()

        if self.distractor is not None:
            self._distractor_pos = self._place_distractor(pos, rng)
            droot = torch.zeros((1, 13), device=self.robot.device, dtype=torch.float32)
            droot[0, 0:3] = torch.tensor(self._distractor_pos, device=self.robot.device)
            droot[0, 3:7] = torch.tensor(upright, device=self.robot.device)
            self.distractor.write_root_state_to_sim(droot)
            self.distractor.write_data_to_sim()

        self.robot.write_data_to_sim()

        # Pin kinematic hold BEFORE settle. Without `_coast_pos`, each substep
        # read back a PhysX-nudged rim pose and wrote it again — the object
        # walked off the far edge before the throw armed (smoke egg @ 0.15).
        self._obj_kinematic = True
        self._phys_free = False
        self._obj_velocity = np.zeros(3, dtype=np.float64)
        self._coast_pos = None if self._weld else pos.copy()
        self._coast_quat = None if self._weld else np.asarray(quat, dtype=np.float64).copy()
        self.off_table = False
        self.abort_reason = None

        # A few idle control steps so the arm settles; object stays pinned.
        for _ in range(8):
            for k in range(self.n_substeps):
                self._physics_substep(last=(k == self.n_substeps - 1))

        # Arm the throw from the commanded spawn (ignore any settle jitter).
        self._obj_velocity = lin_vel.copy()
        self._obj_speed_cmd = lin_vel.copy()
        self._obj_vel_obs = lin_vel.copy()
        if not self._weld:
            self._coast_pos = pos.copy()
            self._write_object_pose(pos, quat, np.zeros(3))
        self._prev_obj_pos = pos.copy()
        self._obj_kinematic = True
        self._phys_free = False
        self.off_table = False
        self.abort_reason = None
        self._attached = False
        self._attach_offset = np.zeros(3, dtype=np.float64)
        self._freeze_pose: tuple[np.ndarray, np.ndarray] | None = None
        self._was_frozen = False
        self._sim_time = 0.0
        self._last_capture_t0 = 0.0
        self._last_gripper_open = 1.0
        self.ik.reset()

        tag = "sphere" if self._use_primitive else (
            self._active_usd.stem if self._active_usd else "object"
        )
        self._meta = {
            "usd": str(self._active_usd) if self._active_usd else "primitive:sphere",
            "category": "primitive" if self._use_primitive else (
                self._active_usd.parent.name if self._active_usd else "object"
            ),
            # Sit height this episode (tipped cans: diameter/2). Expert uses this for z_grasp.
            "object_half_height": float(self._episode_object_half_height),
            "object_half_height_upright": float(self._object_half_height),
            "object_tipped": bool(tip),
            "throw_randomness": float(throw_r),
            "table_z": float(self._table_z),
            "container_usd": str(self._container_usd) if self._container_usd else None,
            "container_category": (
                self._container_usd.parent.name if self._container_usd else None
            ),
            "distractor_usd": str(self._distractor_usd) if self._distractor_usd else None,
            "distractor_category": self._distractor_usd.parent.name if self._distractor_usd else None,
            "distractor_pos": None if self._distractor_pos is None else self._distractor_pos.tolist(),
            "container_half_height": float(self._container_half_height),
            "container_pos": self._container_pos.tolist(),
            "carry_pose": self.get_carry_pose().tolist(),
            "drop_target": self.get_drop_target().tolist(),
            "asset_seed": int(self._asset_seed),
            "speed_m_s": speed,
            "seed": int(seed),
            "tags": [tag],
            "init_pos": pos.tolist(),
            "init_vel": lin_vel.tolist(),
            "spawn_redraws": int(self._spawn_redraws),
            "ride_j7_margin_rad": self._last_ride_j7_margin,
            "init_quat_wxyz": quat.tolist(),
            "grasp_mode": "weld" if self._weld else "physics",
            **self._house_meta,
        }
        return dict(self._meta)

    def _place_distractor(self, target_pos: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Static distractor 12-16 cm from the target's start, inside the play bounds and
        clear of the container. Tries side-by-side positions first (same distance to the arm)."""
        (xmin, xmax), (ymin, ymax) = self._xy_bounds
        pad = self._spawn_xy_pad
        z = self._table_z + self._distractor_half_height
        r = float(rng.uniform(0.12, 0.16))
        sign = 1.0 if rng.random() < 0.5 else -1.0
        cands = [(0.0, sign * r), (0.0, -sign * r), (r, 0.0), (-r, 0.0)]
        for dx, dy in cands:
            x, y = float(target_pos[0] + dx), float(target_pos[1] + dy)
            if not (xmin + pad <= x <= xmax - pad and ymin + pad <= y <= ymax - pad):
                continue
            if self.container is not None and np.linalg.norm(np.array([x, y]) - self._container_pos[:2]) < 0.15:
                continue
            return np.array([x, y, z], dtype=np.float64)
        dx, dy = cands[0]
        return np.array([float(target_pos[0] + dx), float(target_pos[1] + dy), z], dtype=np.float64)

    def get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Where the fingertip point (TCP) is in the world: (position, quaternion wxyz)."""
        body_pos = self.robot.data.body_pos_w[:, self._ee_idx]
        body_quat = self.robot.data.body_quat_w[:, self._ee_idx]
        tcp_pos, tcp_quat = combine_frame_transforms(
            body_pos, body_quat, self._tcp_offset, self._tcp_offset_quat
        )
        return (
            tcp_pos[0].detach().cpu().numpy().astype(np.float64),
            tcp_quat[0].detach().cpu().numpy().astype(np.float64),
        )

    def get_hand_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """``panda_hand`` body pose in world: (position, quaternion wxyz).

        Use with ``sim.pinch_point.pinch_world_from_hand`` for the geometric
        pad midpoint (not the configured TCP).
        """
        pos = self.robot.data.body_pos_w[0, self._ee_idx].detach().cpu().numpy()
        quat = self.robot.data.body_quat_w[0, self._ee_idx].detach().cpu().numpy()
        return (
            np.asarray(pos, dtype=np.float64),
            np.asarray(quat, dtype=np.float64),
        )

    def get_pinch_point(self) -> np.ndarray:
        """World-frame geometric pinch (pad midpoint) from hand pose + URDF Z.

        ``panda_leftfinger`` / ``panda_rightfinger`` are finger-*joint* origins
        (Z=58.4 mm), not the pads — do not midpoint those. Prefer true
        fingertip bodies when present; else ``pinch_world_from_hand`` (Z=103.4 mm).
        """
        from sim.pinch_point import pinch_from_finger_bodies, pinch_world_from_hand

        names = list(self.robot.data.body_names)
        # Pad tips only (not finger-link origins — those sit 45 mm short of the pads).
        pair = ("panda_leftfingertip", "panda_rightfingertip")
        if pair[0] in names and pair[1] in names:
            i0 = names.index(pair[0])
            i1 = names.index(pair[1])
            p0 = self.robot.data.body_pos_w[0, i0].detach().cpu().numpy()
            p1 = self.robot.data.body_pos_w[0, i1].detach().cpu().numpy()
            return pinch_from_finger_bodies(p0, p1)
        hand_pos, hand_quat = self.get_hand_pose()
        return pinch_world_from_hand(hand_pos, hand_quat)

    def get_tcp_minus_pinch_hand_m(self) -> np.ndarray:
        """Configured TCP − geometric pinch, expressed in the hand frame (m)."""
        from sim.pinch_point import tcp_minus_pinch_in_hand_m

        tcp = self._tcp_offset[0].detach().cpu().numpy().astype(np.float64)
        return tcp_minus_pinch_in_hand_m(tcp)

    def _object_centre_world(self, root_pos: np.ndarray) -> np.ndarray:
        """Collision-centre world pose from USD root (constant local offset)."""
        return np.asarray(root_pos, dtype=np.float64).reshape(3) + np.asarray(
            self._object_centre_offset, dtype=np.float64
        ).reshape(3)

    def get_object_state(self) -> ObjectState:
        """True object **collision centre** + velocity, as ObjectState.

        Expert / w_pre / z_grasp aim at the collider centre, not the USD origin
        (audit: some meshes have origin≠centre by >1 mm).
        After place the object is pinned, so we report zero speed.
        """
        if self._was_frozen and self._freeze_pose is not None and not self._attached:
            pos = self._object_centre_world(self._freeze_pose[0])
            vel = np.zeros(3, dtype=np.float64)
        else:
            root = self.object.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
            pos = self._object_centre_world(root)
            # Post-step pose FD. With kinematic_enabled, this should match the
            # scripted coast command (constant velocity). Offset is constant →
            # centre velocity = root velocity.
            vel = self._obj_vel_obs.copy()
        return oracle_from_gt(pos, vel, timestamp=self._sim_time, valid=True)

    def _refresh_object_vel_obs(self) -> None:
        """Measure object speed from how far it moved this step.

        Why: logs and the policy should see the speed that really happened.

        While attached, force zeros so EE teleports do not spike logged velocity.
        Still refresh ``_prev_obj_pos`` so release does not FD-spike either.
        """
        if self._was_frozen and self._freeze_pose is not None and not self._attached:
            self._obj_vel_obs = np.zeros(3, dtype=np.float64)
            self._prev_obj_pos = self._freeze_pose[0].copy()
            return
        pos = self.object.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
        if self._attached:
            self._obj_vel_obs = np.zeros(3, dtype=np.float64)
            self._prev_obj_pos = pos.copy()
            return
        if self._prev_obj_pos is not None and self.dt > 0.0:
            self._obj_vel_obs = (pos - self._prev_obj_pos) / float(self.dt)
        else:
            # First sample: fall back to PhysX linvel after integration.
            self._obj_vel_obs = (
                self.object.data.root_lin_vel_w[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        self._prev_obj_pos = pos.copy()

    def get_object_state_forecast(self, delta: float) -> ObjectState:
        """Where the object will be in ``delta`` seconds if it keeps its speed.

        Same math as ``conditioning_vector`` / ``ObjectState.predict``.

        Coast stays in ``kinematic_advance`` (world motion), not in this predictor.
        When the object is welded to the EE or free in PhysX, return now.
        """
        now = self.get_object_state()
        if not self._obj_kinematic or self._attached or self._phys_free:
            return now
        return now.predict(float(delta))

    def get_proprio(self) -> np.ndarray:
        """All robot joint angles (the arm's own "body sense" for the policy)."""
        return self.robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float64)

    def get_gripper_open(self) -> float:
        """Commanded gripper open∈[0,1] (matches collect ``gripper`` / LeRobot state[7])."""
        return float(self._last_gripper_open)

    def get_finger_gap_m(self) -> float:
        """Measured gap between the two fingers (m): both finger joints summed.

        Closed on fruit it stays at the fruit width; closed on nothing it is ~0.
        """
        q = self.robot.data.joint_pos[0, self._finger_idx]
        return float(q.sum().item())

    def _finger_width(self, gripper_open: float) -> float:
        """Map open 0..1 to a finger joint position (0 = closed, 1 = fully open)."""
        g = float(np.clip(gripper_open, 0.0, 1.0))
        return self._gripper_close + g * (self._gripper_open - self._gripper_close)

    def _snap_fingers(self, gripper_open: float) -> None:
        """Teleport finger joints to the command.

        Grasp is attach/weld, not contact squeeze — PD alone stalls ~0.03 m open
        against the object, so eval's finger-width read never saw ``closed``.
        """
        finger = self._finger_width(gripper_open)
        n = len(self._finger_idx)
        pos = torch.full(
            (1, n), finger, device=self.robot.device, dtype=torch.float32
        )
        vel = torch.zeros((1, n), device=self.robot.device, dtype=torch.float32)
        self.robot.write_joint_position_to_sim(pos, joint_ids=self._finger_idx)
        self.robot.write_joint_velocity_to_sim(vel, joint_ids=self._finger_idx)
        target = self.robot.data.joint_pos_target.clone()
        for fi in self._finger_idx:
            target[0, fi] = finger
        self.robot.set_joint_position_target(target)

    @property
    def last_capture_t0(self) -> float:
        """Sim time when the latest camera frames were captured (Step 1 stamp)."""
        return float(self._last_capture_t0)

    def get_images(self) -> dict[str, np.ndarray]:
        """Latest RGB frame per camera, keyed by camera name.

        Capture time is ``last_capture_t0`` (set at the end of ``step`` when
        cameras update). Use ``get_image_packet`` when you need both together.
        """
        out: dict[str, np.ndarray] = {}
        for name, cam in self.cameras.items():
            if "rgb" in cam.data.output:
                out[name] = cam.data.output["rgb"][0].detach().cpu().numpy()
        return out

    def get_image_packet(self) -> tuple[dict[str, np.ndarray], float]:
        """RGB frames plus their capture sim-time t0 (same stamp for all cams)."""
        return self.get_images(), self.last_capture_t0

    @property
    def weld_grasp(self) -> bool:
        """True = legacy weld glue; False = real PhysX grasp."""
        return self._weld

    def get_wrist_depth(self) -> np.ndarray | None:
        """Wrist depth (H, W) float32 metres, or None if the camera is off."""
        cam = self.cameras.get("wrist_cam")
        if cam is None or "depth" not in cam.data.output:
            return None
        return cam.data.output["depth"][0, ..., 0].detach().cpu().numpy().astype(np.float32)

    def get_wrist_depth_m(self) -> float:
        """Nearest thing between the fingertips in the wrist view (m). NaN = no reading."""
        return gripper_depth_m(self.get_wrist_depth())

    def get_wrist_cam_pose(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Wrist camera world pose: (position xyz, quat wxyz), or None if off.

        Use this for depth→height (cam sits on the hand, not at the TCP).
        """
        cam = self.cameras.get("wrist_cam")
        if cam is None:
            return None
        pos = cam.data.pos_w[0].detach().cpu().numpy().astype(np.float64).copy()
        quat = cam.data.quat_w_world[0].detach().cpu().numpy().astype(np.float64).copy()
        return pos, quat

    def get_static_depth(self) -> np.ndarray | None:
        """Static camera depth (H, W) float32 metres, or None if depth is off."""
        cam = self.cameras.get("static_cam")
        if cam is None or "depth" not in cam.data.output:
            return None
        return cam.data.output["depth"][0, ..., 0].detach().cpu().numpy().astype(np.float32)

    def static_cam_z(self) -> float:
        """World Z of the static camera (for rough depth→height)."""
        cam = self.cameras.get("static_cam")
        if cam is None:
            return float(self._table_z) + 0.6
        return float(cam.data.pos_w[0, 2].detach().cpu().item())

    def _write_object_pose(self, pos: np.ndarray, quat: np.ndarray, vel: np.ndarray) -> None:
        """Teleport the object to a pose and tell PhysX its speed (we move it, not physics)."""
        pose = torch.zeros((1, 7), device=self.robot.device, dtype=torch.float32)
        pose[0, 0:3] = torch.tensor(pos, device=self.robot.device)
        pose[0, 3:7] = torch.tensor(quat, device=self.robot.device)
        self.object.write_root_pose_to_sim(pose)
        v = torch.zeros((1, 6), device=self.robot.device, dtype=torch.float32)
        v[0, 0:3] = torch.tensor(vel, device=self.robot.device)
        self.object.write_root_velocity_to_sim(v)

    def _pin_quat(self) -> np.ndarray | None:
        """Spawn quat while the fingers are open; None (= PhysX quat) once they close.

        Closing pads on a body whose pose *and* tilt are forced every substep
        press into it; freed after finger travel it jumped ~6 mm (upright eggs).
        Letting contacts turn it during the close keeps that slide ~1 mm.
        """
        if self._coast_quat is None or float(self._last_gripper_open) < 0.5:
            return None
        return self._coast_quat

    def _write_kinematic_object(self) -> None:
        """Slide the object one PHYSICS step along its path (the on-table motion for G0).

        Called once per physics substep, so it moves ``speed * physics_dt`` each
        time and exactly ``speed * dt`` per control step.
        """
        if not self._obj_kinematic:
            return
        pos = self.object.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
        quat = self.object.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
        if not self._weld and self._coast_pos is not None:
            if self._pin_quat() is not None:
                quat = self._pin_quat()  # slide flat: no PhysX tipping (see __init__)
            # Dynamic body: advance OUR copy of the path, not the read-back pose.
            # PhysX table friction pulls a teleported resting body back ~20% of
            # each step (tools/diag/check_coast_speed.py), which used to add up
            # to a throw ~20% slower than commanded.
            pos = self._coast_pos
        # Coast straight; a miss leaves the table.
        new_pos, new_vel = kinematic_advance(
            pos,
            self._obj_velocity,
            self.physics_dt,
        )
        # Keep it flat on the table: motion is XY only. Root Z accounts for
        # collision-centre offset so the collider bottom stays on the table.
        off_z = float(self._object_centre_offset[2])
        new_pos[2] = (
            self._table_z + float(self._episode_object_half_height) - off_z
        )
        self._obj_velocity = new_vel
        if not self._weld:
            self._coast_pos = new_pos.copy()
        # Dynamic body: PhysX velocity 0 so it does not add a second v·dt.
        self._write_object_pose(new_pos, quat, new_vel if self._weld else np.zeros(3))

    def _ee_lin_vel_w(self) -> np.ndarray:
        """Hand linear velocity in world (m/s). Used to match the object on grasp free."""
        return (
            self.robot.data.body_lin_vel_w[0, self._ee_idx]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

    def get_ee_lin_vel(self) -> np.ndarray:
        """World-frame EE-link linear velocity (m/s).

        Close-gate ``v_hand`` source (not finite-diff of EE pose). Same vector
        grasp-free uses to match object speed on contact.
        """
        return self._ee_lin_vel_w()

    def _free_object(self, *, match_ee: bool = False) -> None:
        """Hand the object to PhysX (physics grasp / fell off the table).

        ``match_ee=True`` (grasp close): give the object the hand's velocity so
        relative speed ≈ 0 when the fingers shut. 
        Coast speed alone made ≥50 cm/s closes slip, then retries chased off the table (ee-off-table).
        """
        if match_ee:
            vel = self._ee_lin_vel_w()
            self._obj_velocity = vel.copy()
            self._obj_vel_obs = vel.copy()
        else:
            vel = np.asarray(self._obj_velocity, dtype=np.float64).reshape(3)
        v = torch.zeros((1, 6), device=self.robot.device, dtype=torch.float32)
        v[0, 0:3] = torch.tensor(vel, device=self.robot.device, dtype=torch.float32)
        self.object.write_root_velocity_to_sim(v)
        self._phys_free = True
        self._obj_kinematic = False

    def release_object(self) -> bool:
        """Physics grasp only: hand the object to PhysX NOW (stop the scripted coast).

        For a ``substep_cb`` that closes the gripper mid control step: without it the
        object keeps being teleported along its path until the next ``step`` call
        gets ``attach_object=True``. One-way, like the expert's close.
        Weld mode: does nothing, returns False (weld needs ``attach_object``).
        Match hand speed so a mid-step reflex close does not free at coast speed.
        """
        if self._weld:
            return False
        if not self._phys_free:
            self._free_object(match_ee=True)
        return True

    def _pads_on_object(self, margin_m: float = 0.001) -> bool:
        """Closing fingers have reached the object's width along the finger axis.

        From here PhysX must own the object: squeezing a body we still teleport
        every substep shoved the arm ~3 cm up/sideways and the pads shut on air
        (B1: every first grasp with a pre-shaped 180 ms close). Before contact the
        scripted coast stays, so a moving object does not drag on the table.
        """
        if self._object_size is None or float(self._last_gripper_open) >= 0.5:
            return False
        from scipy.spatial.transform import Rotation

        from sim.obs_delay import finger_axis_width_m

        _, q_ee = self.get_ee_pose()
        R = Rotation.from_quat([q_ee[1], q_ee[2], q_ee[3], q_ee[0]]).as_matrix()
        fy = R[:2, 1]  # finger-open axis = hand Y, projected on the table
        yaw_hand = float(np.arctan2(-fy[0], fy[1]))
        q_o = self._coast_quat if self._coast_quat is not None else np.array([1.0, 0.0, 0.0, 0.0])
        yaw_obj = float(Rotation.from_quat([q_o[1], q_o[2], q_o[3], q_o[0]]).as_euler("xyz")[2])
        w = finger_axis_width_m(self._object_size, grasp_yaw_rad=yaw_hand - yaw_obj)
        return self.get_finger_gap_m() <= w + float(margin_m)

    def _step_physics_object(self, release: bool) -> None:
        """Physics-grasp object update, before ``sim.step``.

        Scripted coast until the expert closes (``release``) or the
        object leaves the table footprint (then it falls: a real miss).
        """
        if self._phys_free:
            return
        if release or self._pads_on_object():
            # Grasp free: match hand velocity so fingers can close without a 50 cm/s shear.
            self._free_object(match_ee=True)
            return
        self._write_kinematic_object()
        pos = self.object.data.root_pos_w[0].detach().cpu().numpy()  # pose we just wrote
        if self._table_box is not None and xy_outside_box(pos[:2], *self._table_box):
            self.off_table = True
            self._free_object(match_ee=False)

    def check_safety_abort(self, *, attached: bool) -> str | None:
        """Return a short reason to stop the episode early, else None.

        Catches two bad cases that used to keep running forever:
        1. Hand leaves the table (chasing a fallen object under the floor).
        2. Scripted coast drives the object through the box or the robot base
           (kinematic pose writes ignore PhysX collisions).
        """
        if self.abort_reason:
            return self.abort_reason

        ee, _ = self.get_ee_pose()
        # Hand past the table footprint, or buried under the table top.
        if self._table_box is not None:
            margin = float(self._ee_table_margin_m)
            lo = np.asarray(self._table_box[0], dtype=np.float64) - margin
            hi = np.asarray(self._table_box[1], dtype=np.float64) + margin
            if xy_outside_box(ee[:2], lo, hi) or float(ee[2]) < float(self._table_z) - 0.05:
                self.abort_reason = "ee-off-table"
                return self.abort_reason

        # Coasting object only: once PhysX owns it, collisions are real.
        if attached or self._phys_free or self._attached:
            return None

        obj = self.object.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
        # Through the robot pedestal (near world XY origin).
        if float(np.linalg.norm(obj[:2])) < float(self._robot_base_keepout_m):
            self.abort_reason = "object-through-base"
            return self.abort_reason

        # Through the place box while still on the scripted path.
        # Skip when the hand is already on the object (about to close): a 50 cm/s
        # near-hit used to die as object-through-container with gap ~2 cm.
        if self.container is not None and self._container_size is not None:
            hand_gap = float(np.linalg.norm(ee[:3] - obj[:3]))
            if hand_gap > 0.05:
                half = 0.5 * np.asarray(self._container_size[:2], dtype=np.float64)
                # Shrink slightly so sitting *next to* the box is fine; only the interior counts.
                half = np.maximum(half - 0.01, 0.01)
                cxy = self._container_pos[:2]
                if np.all(np.abs(obj[:2] - cxy) <= half):
                    self.abort_reason = "object-through-container"
                    return self.abort_reason
        return None

    def _write_attached_object(self) -> None:
        """While held, glue the object to the hand at a fixed offset."""
        ee_pos, ee_quat = self.get_ee_pose()
        pos = ee_pos + self._attach_offset
        self._obj_velocity = np.zeros(3, dtype=np.float64)
        self._write_object_pose(pos, ee_quat, self._obj_velocity)

    def set_attach(self, attach: bool) -> None:
        """Grab (weld object to hand) or let go.

        Grasp here is scripted, not a physics squeeze. Why: it is steady and
        repeatable, so the expert's success does not depend on friction luck.
        """
        if attach and not self._attached:
            ee_pos, _ = self.get_ee_pose()
            obj = self.object.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
            # Keep the object where it was relative to the hand; cap how far below
            # (``attach_z_offset_max``) so it cannot jump on grab.
            z_off = float(
                np.clip(float(obj[2] - ee_pos[2]), self._attach_z_offset_max, 0.0)
            )
            self._attach_offset = np.array([0.0, 0.0, z_off], dtype=np.float64)
            self._attached = True
            self._obj_kinematic = False
            self._freeze_pose = None
            self._was_frozen = False
            # Held ⇒ fingers closed (command + measured), so eval latch/obs agree.
            self._last_gripper_open = 0.0
            self._snap_fingers(0.0)
        elif not attach and self._attached:
            # Snapshot weld pose before clearing — place freeze must pin this.
            ee_pos, ee_quat = self.get_ee_pose()
            pin = (ee_pos + self._attach_offset).astype(np.float64)
            self._freeze_pose = (pin.copy(), np.asarray(ee_quat, dtype=np.float64).copy())
            self._attached = False
            self._attach_offset = np.zeros(3, dtype=np.float64)

    def set_ee_target(
        self,
        pos_w: np.ndarray,
        quat_wxyz: np.ndarray,
        gripper_open: float,
        vel_w: np.ndarray | None = None,
    ) -> None:
        """Tell the arm where to put the fingertip point, and how open the gripper is.

        Runs IK (turns a hand target into joint angles). Sets targets only;
        the move happens on the next ``step``.

        Optional ``vel_w`` (m/s, world): velocity feedforward when
        ``robot.diffik_ff_gain`` K > 0. Plain form::

            v_cmd = v_target + K · (p_target − p_hand)
            p_ik  = p_hand + v_cmd / K   # = p_target + v_target / K

        ``vel_w`` must be v̂_obj (forecast/coast object velocity), never FD of
        consecutive aims. Zeros for a fixed pose. ``vel_w=None`` or K=0 →
        absolute-pose DiffIK.
        """
        device = self.robot.device
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w

        body_pos = self.robot.data.body_pos_w[:, self._ee_idx]
        body_quat = self.robot.data.body_quat_w[:, self._ee_idx]
        tcp_pos_w, tcp_quat_w = combine_frame_transforms(
            body_pos, body_quat, self._tcp_offset, self._tcp_offset_quat
        )

        p_target = np.asarray(pos_w, dtype=np.float64).reshape(3)
        p_ik = p_target
        k_ff = float(self._diffik_ff_k)
        if self._joint_vel_ff:
            pass  # lead comes from q̇* + the substep ramp (below), not from v/K
        elif k_ff > 0.0 and vel_w is not None:
            # Full v̂_obj lead every tick: p_ik = p* + v̂_obj/K.
            # A near-field blend that zeroed the lead when |aim−hand| was small
            # made the hand trail and hunt on s1003 (aim stayed smooth).
            v_hat = np.asarray(vel_w, dtype=np.float64).reshape(3)
            tau = 1.0 / k_ff
            p_ik = p_target + v_hat * tau
        des_pos_w = torch.tensor(p_ik, device=device, dtype=torch.float32).view(1, 3)
        des_quat_w = torch.tensor(quat_wxyz, device=device, dtype=torch.float32).view(1, 4)
        des_pos_b, des_quat_b = subtract_frame_transforms(
            root_pos, root_quat, des_pos_w, des_quat_w
        )
        self.ik.set_command(torch.cat([des_pos_b, des_quat_b], dim=-1))

        tcp_pos_b, tcp_quat_b = subtract_frame_transforms(
            root_pos, root_quat, tcp_pos_w, tcp_quat_w
        )
        jacobian = self.robot.root_physx_view.get_jacobians()[
            :, self._jac_idx, :, self._arm_idx
        ]
        joint_pos = self.robot.data.joint_pos[:, self._arm_idx]
        new_joint_pos = self.ik.compute(tcp_pos_b, tcp_quat_b, jacobian, joint_pos)

        target = self.robot.data.joint_pos.clone()
        target[:, self._arm_idx] = new_joint_pos
        g = float(np.clip(gripper_open, 0.0, 1.0))
        finger = self._finger_width(g)
        for fi in self._finger_idx:
            target[0, fi] = finger
        self.robot.set_joint_position_target(target)
        if self._joint_vel_ff:
            v = np.zeros(3) if vel_w is None else np.asarray(vel_w, dtype=np.float64).reshape(3)
            qd = self._tcp_joint_velocity(jacobian, v)
            vt = torch.zeros_like(self.robot.data.joint_vel_target)
            vt[:, self._arm_idx] = qd
            self.robot.set_joint_velocity_target(vt)
            self._ramp_q0, self._ramp_qd, self._ramp_v = new_joint_pos.clone(), qd, v
        self._last_gripper_open = g
        # Weld: teleport fingers. Physics grasp: PD target only, fingers stall on the object.
        if self._weld:
            self._snap_fingers(g)

    def _ride_j7_margin(self, root_pos: np.ndarray, lin_vel: np.ndarray, speed: float) -> float | None:
        """Smallest j7 distance (rad) to a limit over the planned chase + ride.

        None when the filter is off or the object does not move (no pads-across
        yaw then). Path = collision centre (root + centre offset) at grasp height.
        """
        if self._spawn_j7_margin <= 0.0 or float(speed) <= 0.02:
            return None
        from sim.ride_ik_check import ride_j7_margin
        from sim.state_machine import _yaw_about_z

        sm = self.cfg.get("state_machine_params", {})
        off = np.asarray(self._object_centre_offset, dtype=np.float64).reshape(3)
        centre = np.asarray(root_pos, dtype=np.float64) + off
        down = np.asarray(self.cfg["robot"]["init_pose"][3:], dtype=np.float64)
        heading = float(np.arctan2(lin_vel[1], lin_vel[0]))
        q_home = self.robot.data.default_joint_pos[0, self._arm_idx].detach().cpu().numpy()
        r = ride_j7_margin(
            q_home=q_home,
            home_quat_wxyz=down,
            pad_quat_wxyz=_yaw_about_z(down, heading),
            spawn_xyz=centre,
            vel_xy=np.asarray(lin_vel, dtype=np.float64)[:2],
            tcp=float(self._tcp_offset[0, 2].item()),
            dt=float(self.dt),
            wait_s=float(sm.get("settle_s", 0.05)),
            chase_speed=float(sm.get("ee_speed_max", 0.365)),
            ride_s=1.0,
            keep_r_min=float(sm.get("reach_zone_radius", 0.2)),
        )
        m = r["margin"]
        return None if not np.isfinite(m) else float(m)

    def _tcp_joint_velocity(self, jacobian: torch.Tensor, v_w: np.ndarray) -> torch.Tensor:
        """Arm q̇* that moves the TCP (pinch) at ``v_w`` with no rotation.

        ``jacobian`` is the hand-body Jacobian (1, 6, 7). TCP linear rows =
        J_lin − [r]× J_ang, r = hand→TCP in world. DLS pseudo-inverse, same
        damping as the IK (λ = 0.01). Root frame = world (base at origin).
        """
        if not np.any(v_w):
            return torch.zeros((1, len(self._arm_idx)), device=self.robot.device)
        bq = self.robot.data.body_quat_w[:, self._ee_idx]
        r = quat_apply(bq, self._tcp_offset)[0]
        rx = torch.zeros((3, 3), device=self.robot.device, dtype=torch.float32)
        rx[0, 1], rx[0, 2], rx[1, 2] = -r[2], r[1], -r[0]
        rx[1, 0], rx[2, 0], rx[2, 1] = r[2], -r[1], r[0]
        J = jacobian[0].clone()
        J[:3, :] = J[:3, :] - rx @ J[3:, :]
        x = torch.zeros(6, device=self.robot.device, dtype=torch.float32)
        x[:3] = torch.tensor(v_w, device=self.robot.device, dtype=torch.float32)
        A = J @ J.T + (0.01**2) * torch.eye(6, device=self.robot.device)
        return (J.T @ torch.linalg.solve(A, x)).view(1, -1)

    def _step_weld_object(
        self, kinematic_object: bool, attach_object: bool, freeze_object: bool
    ) -> None:
        """Legacy weld path: frozen (placed) > attached (welded) > sliding."""
        # Releasing+freezing same frame: snapshot weld pose before clearing attach.
        if (not attach_object) and self._attached and freeze_object:
            ee_pos, ee_quat = self.get_ee_pose()
            pin = (ee_pos + self._attach_offset).astype(np.float64)
            self._freeze_pose = (pin.copy(), np.asarray(ee_quat, dtype=np.float64).copy())
        self.set_attach(bool(attach_object))
        if freeze_object:
            self._obj_velocity[:] = 0.0
            if self._freeze_pose is None:
                pos = self.object.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
                quat = self.object.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
                self._freeze_pose = (pos.copy(), quat.copy())
            pos, quat = self._freeze_pose
            self._write_object_pose(pos, quat, self._obj_velocity)
            self._obj_kinematic = True
            self._was_frozen = True
        elif self._attached:
            # Picked up again: clear the old pin so it follows the hand.
            self._freeze_pose = None
            self._was_frozen = False
            self._write_attached_object()
        else:
            # After place, keep pinning — never resume table-slide.
            if self._was_frozen and self._freeze_pose is not None:
                self._obj_velocity[:] = 0.0
                pos, quat = self._freeze_pose
                self._write_object_pose(pos, quat, self._obj_velocity)
                self._obj_kinematic = True
            else:
                # Free on the table: restore the commanded speed (a grab zeroes it).
                self._obj_velocity = self._obj_speed_cmd.copy()
                self._obj_kinematic = bool(kinematic_object)
                if self._obj_kinematic:
                    self._write_kinematic_object()
                    self._obj_speed_cmd = self._obj_velocity.copy()

    def _physics_substep(self, *, last: bool) -> None:
        """One PhysX step + refresh the asset buffers. Renders only on the ``last`` substep.

        n_substeps == 1 keeps ``sim.step()`` (physics AND render in one call).
        """
        if self.n_substeps == 1:
            self.sim.step()
        else:
            self.sim.step(render=False)
            if last:
                self.sim.render()  # cameras see the end of the control step
        self.robot.update(self.physics_dt)
        self.object.update(self.physics_dt)
        if self.container is not None:
            self.container.update(self.physics_dt)

    def step(
        self,
        *,
        kinematic_object: bool,
        attach_object: bool = False,
        freeze_object: bool = False,
        substep_cb=None,
    ) -> None:
        """Advance the world one CONTROL step (``dt``). The expert calls this once per step.

        Physics runs ``n_substeps`` times inside it (``physics_dt = dt / n``).
        Cameras render once, at the end. Logs, timestamps and images stay at
        the control rate.

        Weld mode, object has three modes, checked in this order:
        frozen (placed, pinned) > attached (held in hand) > sliding on the table.
        Physics mode: scripted coast until the expert closes, then PhysX.
        Order per substep: [substep_cb] → write object / robot / container →
        PhysX step → refresh buffers → re-fix anything physics nudged.
        After the last substep: update cameras, measure object speed.

        ``substep_cb(scene, t_sim)`` (default None = old behaviour), for a fast
        reflex. Called BEFORE each physics substep, k = 0..n-1:
          - ``t_sim`` = sim time of the state it sees = start of substep k
            (control-step start + k * physics_dt). ``get_object_state().timestamp``
            equals ``t_sim`` inside the callback.
          - It may call ``set_ee_target(pos, quat, gripper_open)``: IK and finger
            targets are recomputed from the live joint state and applied to this
            substep and every later one (until the next call).
          - It may read ``get_ee_pose()`` / ``get_object_state()``: position is
            fresh (after substep k-1). Object velocity is the last control-step
            measurement (refreshed once per ``step``), exact for a scripted coast.
          - Physics grasp: to close on the object mid-step, also call
            ``release_object()`` (stops the scripted coast from this substep on).
          - It must NOT call ``step`` and cannot change kinematic / attach /
            freeze for this control step (those are this call's arguments).
          - k = 0 sees the same state the caller saw before ``step``.
        Calls per control step = ``n_substeps``.
        """
        n = self.n_substeps
        for k in range(n):
            if substep_cb is not None:
                substep_cb(self, self._sim_time)
            if self._ramp_q0 is not None:
                if k > 0:
                    # Fresh q̇* = J_tcp⁺·v from this substep's state. A straight hand
                    # line needs changing joint rates; one q̇* held for the whole
                    # tick left a v²-scaling cross-track offset (step-1 probe:
                    # ≤ 1.3 mm → ≤ 0.5 mm at 30 cm/s). Ramp continues from where
                    # the previous substep's target ended.
                    jac = self.robot.root_physx_view.get_jacobians()[
                        :, self._jac_idx, :, self._arm_idx
                    ]
                    qd = self._tcp_joint_velocity(jac, self._ramp_v)
                    self._ramp_q0 = self._ramp_q0 + (self._ramp_qd - qd) * k * self.physics_dt
                    self._ramp_qd = qd
                    vt = torch.zeros_like(self.robot.data.joint_vel_target)
                    vt[:, self._arm_idx] = qd
                    self.robot.set_joint_velocity_target(vt)
                # Arm q* for the END of this substep: q0 + q̇*·(k+1)·dt_phys.
                tgt = self.robot.data.joint_pos_target.clone()
                tgt[:, self._arm_idx] = self._ramp_q0 + self._ramp_qd * (k + 1) * self.physics_dt
                self.robot.set_joint_position_target(tgt)
            if self._weld:
                self._step_weld_object(kinematic_object, attach_object, freeze_object)
            else:
                # Physics grasp: no weld, no pin. Closing (attach / not-kinematic) frees the object.
                self._step_physics_object(bool(attach_object) or not bool(kinematic_object))
            self.robot.write_data_to_sim()
            self.object.write_data_to_sim()
            if self.container is not None:
                # Hold container kinematic pose (seed jitter applied at reset).
                pose = torch.zeros((1, 7), device=self.robot.device, dtype=torch.float32)
                pose[0, 0:3] = torch.tensor(self._container_pos, device=self.robot.device)
                pose[0, 3:7] = torch.tensor(
                    [1.0, 0.0, 0.0, 0.0], device=self.robot.device
                )
                self.container.write_root_pose_to_sim(pose)
                self.container.write_data_to_sim()
            # n > 1: render once below, AFTER the re-asserts, so frames show the true pose.
            self._physics_substep(last=False)
            # Attach grasp: fingers are scripted. Re-snap after PhysX so object contact
            # cannot leave them jammed ~0.03 m open (breaks eval closed threshold).
            if self._weld:
                if self._attached:
                    self._last_gripper_open = 0.0
                self._snap_fingers(self._last_gripper_open)
            # Re-assert freeze after PhysX so contacts cannot kick the placed object.
            if (freeze_object or self._was_frozen) and self._freeze_pose is not None and not self._attached:
                pos, quat = self._freeze_pose
                self._obj_velocity[:] = 0.0
                self._write_object_pose(pos, quat, self._obj_velocity)
                self.object.write_data_to_sim()
                self.object.update(self.physics_dt)  # buffers must match pin (else 1-frame flicker)
            # Physics-grasp coast: PhysX table friction drags a teleported resting
            # body back a little every step (uneven read-back, e.g. 4 / 20 cm/s on
            # alternate frames at 25 Hz). Put it back on the scripted path so logs,
            # images and the next substep all see exactly speed * time.
            if not self._weld and self._obj_kinematic and not self._phys_free and self._coast_pos is not None:
                q = self._pin_quat()
                if q is None:
                    q = self.object.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
                self._write_object_pose(self._coast_pos, q, np.zeros(3))
                self.object.write_data_to_sim()
                self.object.update(self.physics_dt)
            # n == 1: add the full dt at once (legacy, identical float sums).
            self._sim_time += self.dt if n == 1 else self.physics_dt
        if self._ramp_q0 is not None:
            # Ramp used up: hold its end point with no velocity target until the
            # next set_ee_target (a step without a new command must not re-ramp).
            self.robot.set_joint_velocity_target(torch.zeros_like(self.robot.data.joint_vel_target))
            self._ramp_q0, self._ramp_qd = None, None
        if n > 1:
            self.sim.render()  # cameras see the end of the control step
            # Snap to the control grid so float sums never drift off k * dt.
            self._sim_time = round(self._sim_time / self.dt) * self.dt
        for cam in self.cameras.values():
            cam.update(self.dt)
        # Stamp frames with the sim time of this render (Step 1 t0).
        self._last_capture_t0 = float(self._sim_time)
        self._refresh_object_vel_obs()

"""In-repo Franka Phase-0 scene (Isaac Lab). Import only AFTER AppLauncher.

Owns: Franka Panda + work surface + one rigid object + static container
+ static/wrist cameras + DiffIK.

``scene.use_house: true`` loads a random house USD (paths.scene_dir), moves it
so a free long table edge sits at the world origin (robot stays at 0,0,0),
randomizes lights, and aims throws across a catch band. 
Floor mode keeps the old kinematic slab.

Grasp (``scene.use_weld_grasp``): house default = real PhysX (fingers close,
no weld; the object is scripted until the expert closes, then free). Floor
default = legacy weld. House throws never bounce; a miss coasts off the table.

Meshes (object + container USDs, and the house when enabled) are fixed for the
process lifetime (plan 2A). Diversify across workers / restarts via asset_seed.

Runs once at start (build the world), then every control step (move things, step physics).
Timing: control step ``sim.dt`` (25 Hz: expert, policy, cameras, logs). House mode runs
``sim.physics_substeps`` PhysX steps inside each control step (floor mode: always 1).
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
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms
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
    """Box size (m) around a mesh. We need it to set the object on the table at the right height.

    Raises on any problem. A wrong size would bury or float the object.
    """
    try:
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(usd_path))
        if stage is None:
            raise RuntimeError(f"USD bbox failed: cannot open {usd_path}")
        prim = stage.GetDefaultPrim()
        if not prim:
            raise RuntimeError(f"USD bbox failed: no default prim in {usd_path}")
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        bbox = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        size = np.array(bbox.GetSize(), dtype=np.float64)
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
        self._use_house = bool(sc.get("use_house", False))

        # Control rate stays dt (expert, policy, cameras, logs). Physics runs
        # ``n_substeps`` smaller steps inside each control step (house only;
        # floor is always 1). render_interval = n: one render per control step
        # (Isaac Lab DirectRLEnv uses the same decimation pattern).
        self.n_substeps = physics_substeps(cfg)
        self.physics_dt = dt / self.n_substeps

        # Isaac calls must come after AppLauncher has started (see module top).
        if self.n_substeps == 1:
            sim_cfg = sim_utils.SimulationCfg(dt=dt, device=device)  # legacy, unchanged
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
        self._table_box: tuple[np.ndarray, np.ndarray] | None = None  # house table XY footprint
        # Weld = old scripted glue. Default: weld on the floor slab, real physics in a house.
        self._weld = bool(sc.get("use_weld_grasp", not self._use_house))
        # Bounce only in the legacy floor box. House throws coast straight.
        self._bounce = not self._use_house

        # One seeded rng for house + mesh choice (same asset_seed → same world).
        asset_rng = np.random.default_rng(self._asset_seed)

        # Invisible floor under the work surface. House mode drops it slightly so
        # it does not fight the table top at z=0.
        ground_z = -0.1 if self._use_house else 0.0
        sim_utils.GroundPlaneCfg().func(
            "/World/GroundPlane",
            sim_utils.GroundPlaneCfg(),
            translation=(0.0, 0.0, ground_z),
        )

        if self._use_house:
            self._setup_house_world(cfg, asset_rng)
        else:
            light_cfg = sim_utils.DomeLightCfg(intensity=800.0, color=(0.95, 0.95, 0.95))
            light_cfg.func("/World/Light", light_cfg)
            try:
                table_size = tuple(float(x) for x in sc["table_size"])
                table_pos = tuple(float(x) for x in sc["table_pos"])
            except KeyError as ex:
                logger.error("missing required yaml key: scene.%s", ex.args[0])
                raise KeyError(f"scene.{ex.args[0]}") from None
            # Table is kinematic: fixed solid slab, never pushed.
            table_cfg = sim_utils.CuboidCfg(
                size=table_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.45, 0.35, 0.25)
                ),
            )
            table_cfg.func("/World/Table", table_cfg, translation=table_pos)

        robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.copy()
        robot_cfg.prim_path = "/World/Robot"
        # Base sits at the table-edge anchor (house is moved so that anchor = origin).
        robot_cfg.init_state.pos = (0.0, 0.0, 0.0)
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
            self._object_size = usd_bbox_size(self._active_usd)
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
                "Object mesh: %s   size %s   sits %.1f cm on the table",
                self._active_usd.stem if self._active_usd is not None else "unknown",
                fmt_size_cm(self._object_size),
                100.0 * float(self._object_half_height),
            )

        # Table top Z. House mode maps the chosen table edge to z=0.
        self._table_z = (
            float(self._play_layout.table_z)
            if self._play_layout is not None
            else float(_require(cfg, "scene", "table_height"))
        )

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

        # Three object velocities on purpose:
        #   _obj_velocity  — live speed that drives the slide (wall bounces flip it)
        #   _obj_speed_cmd — same speed latched, so a grab/release does not lose it
        #   _obj_vel_obs   — measured after the step; what logs and the policy see
        self._obj_velocity = np.zeros(3, dtype=np.float64)
        self._obj_speed_cmd = np.zeros(3, dtype=np.float64)
        # Observed velocity after the last physics step (FD of root pose, or
        # root_lin_vel_w). Commanded motion speed stays in ``_obj_velocity``.
        self._obj_vel_obs = np.zeros(3, dtype=np.float64)
        self._prev_obj_pos: np.ndarray | None = None
        self._coast_pos: np.ndarray | None = None  # physics grasp: scripted path position
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
                layout = compute_play_layout(
                    cand.table_bbox_min_w,
                    cand.table_bbox_max_w,
                    max_reach_dist=min(max_reach, 0.75),
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
        (xmin, xmax), (ymin, ymax) = self._xy_bounds
        pad = self._spawn_xy_pad
        z = self._table_z + self._object_half_height

        if self._use_house:
            # Throw toward the bowl (straight / diagonal / side). Catch is still
            # inside the play band; a miss heads at the bowl and fail-fasts.
            from sim.house_layout import sample_throw

            bowl_xy = None
            if self.container is not None:
                bowl_xy = (
                    self._play_layout.container_xy
                    if self._play_layout is not None
                    else self._container_pos[:2]
                )
            pos0, lin_vel = sample_throw(
                rng,
                self._xy_bounds,
                speed=speed,
                spawn_pad=pad,
                container_xy=bowl_xy,
            )
            pos = np.array([pos0[0], pos0[1], z], dtype=np.float64)
        else:
            # Floor mode: random start + random heading (legacy G0).
            if speed <= 1e-6:
                lin_vel = np.zeros(3, dtype=np.float64)
            else:
                angle = float(rng.uniform(0.0, 2.0 * np.pi))
                lin_vel = np.array(
                    [speed * np.cos(angle), speed * np.sin(angle), 0.0],
                    dtype=np.float64,
                )
            x = float(rng.uniform(xmin + pad, xmax - pad))
            y = float(rng.uniform(ymin + pad, ymax - pad))
            pos = np.array([x, y, z], dtype=np.float64)

        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

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
            croot[0, 3:7] = torch.tensor(quat, device=self.robot.device)
            self.container.write_root_state_to_sim(croot)
            self.container.write_data_to_sim()

        if self.distractor is not None:
            self._distractor_pos = self._place_distractor(pos, rng)
            droot = torch.zeros((1, 13), device=self.robot.device, dtype=torch.float32)
            droot[0, 0:3] = torch.tensor(self._distractor_pos, device=self.robot.device)
            droot[0, 3:7] = torch.tensor(quat, device=self.robot.device)
            self.distractor.write_root_state_to_sim(droot)
            self.distractor.write_data_to_sim()

        self.robot.write_data_to_sim()

        # A few idle control steps so everything settles before the episode starts.
        for _ in range(8):
            for k in range(self.n_substeps):
                self._physics_substep(last=(k == self.n_substeps - 1))

        # Clear all per-episode state so nothing leaks from the last run.
        self._obj_velocity = lin_vel.copy()
        self._obj_speed_cmd = lin_vel.copy()
        self._obj_vel_obs = lin_vel.copy()
        self._prev_obj_pos = (
            self.object.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64).copy()
        )
        # Scripted throw path (physics grasp): starts where the object settled.
        self._coast_pos = None if self._weld else self._prev_obj_pos.copy()
        self._obj_kinematic = True
        self._phys_free = False
        self.off_table = False
        self.abort_reason = None
        self._attached = False
        self._attach_offset = np.zeros(3, dtype=np.float64)
        self._freeze_pose: tuple[np.ndarray, np.ndarray] | None = None
        self._was_frozen = False
        self._sim_time = 0.0
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
            "object_half_height": float(self._object_half_height),
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
            "use_house": bool(self._use_house),
            "grasp_mode": "weld" if self._weld else "physics",
            "bounce": bool(self._bounce),
            **(self._house_meta if self._use_house else {}),
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

    def get_object_state(self) -> ObjectState:
        """True object position + velocity right now, in the shared ObjectState message.

        This is the "oracle" (ground truth). Same message the filter will fill later.
        After place the object is pinned, so we report zero speed.
        """
        if self._was_frozen and self._freeze_pose is not None and not self._attached:
            pos = self._freeze_pose[0].copy()
            vel = np.zeros(3, dtype=np.float64)
        else:
            pos = self.object.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
            # Post-step pose FD. With kinematic_enabled, this should match the
            # scripted motion command (constant velocity + wall bounce).
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

        Table bounce stays in ``kinematic_advance`` (world motion), not in this predictor.
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

    def get_images(self) -> dict[str, np.ndarray]:
        """Latest RGB frame per camera, keyed by camera name."""
        out: dict[str, np.ndarray] = {}
        for name, cam in self.cameras.items():
            if "rgb" in cam.data.output:
                out[name] = cam.data.output["rgb"][0].detach().cpu().numpy()
        return out

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

    def _write_object_pose(self, pos: np.ndarray, quat: np.ndarray, vel: np.ndarray) -> None:
        """Teleport the object to a pose and tell PhysX its speed (we move it, not physics)."""
        pose = torch.zeros((1, 7), device=self.robot.device, dtype=torch.float32)
        pose[0, 0:3] = torch.tensor(pos, device=self.robot.device)
        pose[0, 3:7] = torch.tensor(quat, device=self.robot.device)
        self.object.write_root_pose_to_sim(pose)
        v = torch.zeros((1, 6), device=self.robot.device, dtype=torch.float32)
        v[0, 0:3] = torch.tensor(vel, device=self.robot.device)
        self.object.write_root_velocity_to_sim(v)

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
            # Dynamic body: advance OUR copy of the path, not the read-back pose.
            # PhysX table friction pulls a teleported resting body back ~20% of
            # each step (tools/diag/check_coast_speed.py), which used to add up
            # to a throw ~20% slower than commanded.
            pos = self._coast_pos
        # Bounce only in floor mode. House / throw: xy_bounds=None → straight line.
        new_pos, new_vel = kinematic_advance(
            pos,
            self._obj_velocity,
            self.physics_dt,
            xy_bounds=self._xy_bounds if self._bounce else None,
        )
        # Keep it flat on the table: motion is XY only.
        new_pos[2] = self._table_z + self._object_half_height
        self._obj_velocity = new_vel
        if not self._weld:
            self._coast_pos = new_pos.copy()
        # Dynamic body: PhysX velocity 0 so it does not add a second v·dt.
        self._write_object_pose(new_pos, quat, new_vel if self._weld else np.zeros(3))

    def _free_object(self) -> None:
        """Hand the object to PhysX (physics grasp / fell off the table). Keeps its speed once."""
        v = torch.zeros((1, 6), device=self.robot.device, dtype=torch.float32)
        v[0, 0:3] = torch.tensor(self._obj_velocity, device=self.robot.device)
        self.object.write_root_velocity_to_sim(v)
        self._phys_free = True
        self._obj_kinematic = False

    def release_object(self) -> bool:
        """Physics grasp only: hand the object to PhysX NOW (stop the scripted coast).

        For a ``substep_cb`` that closes the gripper mid control step: without it the
        object keeps being teleported along its path until the next ``step`` call
        gets ``attach_object=True``. One-way, like the expert's close.
        Weld mode: does nothing, returns False (weld needs ``attach_object``).
        """
        if self._weld:
            return False
        if not self._phys_free:
            self._free_object()
        return True

    def _step_physics_object(self, release: bool) -> None:
        """Physics-grasp object update, before ``sim.step``.

        Scripted coast (no bounce) until the expert closes (``release``) or the
        object leaves the table footprint (then it falls: a real miss).
        """
        if self._phys_free:
            return
        if release:
            self._free_object()
            return
        self._write_kinematic_object()
        pos = self.object.data.root_pos_w[0].detach().cpu().numpy()  # pose we just wrote
        if self._table_box is not None and xy_outside_box(pos[:2], *self._table_box):
            self.off_table = True
            self._free_object()

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
        if self.container is not None and self._container_size is not None:
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
    ) -> None:
        """Tell the arm where to put the fingertip point, and how open the gripper is.

        Runs IK (turns a hand target into joint angles). Sets targets only;
        the move happens on the next ``step``.
        """
        device = self.robot.device
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w
        des_pos_w = torch.tensor(pos_w, device=device, dtype=torch.float32).view(1, 3)
        des_quat_w = torch.tensor(quat_wxyz, device=device, dtype=torch.float32).view(1, 4)
        des_pos_b, des_quat_b = subtract_frame_transforms(
            root_pos, root_quat, des_pos_w, des_quat_w
        )
        self.ik.set_command(torch.cat([des_pos_b, des_quat_b], dim=-1))

        body_pos = self.robot.data.body_pos_w[:, self._ee_idx]
        body_quat = self.robot.data.body_quat_w[:, self._ee_idx]
        tcp_pos_w, tcp_quat_w = combine_frame_transforms(
            body_pos, body_quat, self._tcp_offset, self._tcp_offset_quat
        )
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
        self._last_gripper_open = g
        # Weld: teleport fingers. Physics grasp: PD target only, fingers stall on the object.
        if self._weld:
            self._snap_fingers(g)

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
            # After place, keep pinning — never resume table-slide / bounce bounds.
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

        n_substeps == 1 keeps the legacy call (``sim.step()`` steps physics AND
        renders), so floor mode is bit-for-bit what it was.
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

        Physics runs ``n_substeps`` times inside it (``physics_dt = dt / n``; house
        mode only, floor is 1). Cameras render once, at the end. Logs, timestamps
        and images stay at the control rate.

        Weld mode, object has three modes, checked in this order:
        frozen (placed, pinned) > attached (held in hand) > sliding on the table.
        Physics mode: scripted coast (no bounce) until the expert closes, then PhysX.
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
        Calls per control step = ``n_substeps`` (1 in floor mode).
        """
        n = self.n_substeps
        for k in range(n):
            if substep_cb is not None:
                substep_cb(self, self._sim_time)
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
                self._write_object_pose(self._coast_pos, self.object.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64), np.zeros(3))
                self.object.write_data_to_sim()
                self.object.update(self.physics_dt)
            # n == 1: add the full dt at once (legacy, identical float sums).
            self._sim_time += self.dt if n == 1 else self.physics_dt
        if n > 1:
            self.sim.render()  # cameras see the end of the control step
            # Snap to the control grid so float sums never drift off k * dt.
            self._sim_time = round(self._sim_time / self.dt) * self.dt
        for cam in self.cameras.values():
            cam.update(self.dt)
        self._refresh_object_vel_obs()

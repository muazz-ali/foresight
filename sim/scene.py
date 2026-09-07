"""In-repo Franka Phase-0 scene (Isaac Lab). Import only AFTER AppLauncher.

Owns: Franka Panda + table + one rigid object + static container
+ static/wrist cameras + DiffIK.

Meshes (object + container USDs) are fixed for the process lifetime (plan 2A).
Diversify across workers / restarts via asset_seed or CLI USD overrides.
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
from sim.motion import kinematic_advance

logger = logging.getLogger(__name__)

ARM_JOINT_EXPR = "panda_joint"
FINGER_JOINTS = ("panda_finger_joint1", "panda_finger_joint2")
EE_BODY = "panda_hand"


def _require(cfg: dict, *keys: str) -> Any:
    """Nested yaml lookup. Logs the missing path and raises KeyError."""
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
    """Load ``phase0_cfg.yaml``. Callers must supply every required key."""
    with open(path) as fp:
        return yaml.load(fp, Loader=yaml.FullLoader)


def _euler_deg_to_wxyz(rotation_deg: list[float]) -> list[float]:
    """XYZ Euler degrees → wxyz quaternion (``scene.cameras[].rotation_deg``)."""
    from scipy.spatial.transform import Rotation as R

    q = R.from_euler("XYZ", rotation_deg, degrees=True).as_quat()  # xyzw -> wxyz
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])] # wxyz


def list_category_usds(object_dir: Path, categories: list[str]) -> list[Path]:
    """List ``*.usd`` under ``paths.object_dir`` / each category name."""
    usds: list[Path] = []
    for cat in categories:
        cat_dir = object_dir / cat
        if not cat_dir.is_dir():
            continue
        usds.extend(sorted(cat_dir.glob("*.usd")))
    if not usds:
        raise FileNotFoundError(
            f"No USD objects under {object_dir} for categories={categories}"
        )
    return usds


def usd_bbox_size(usd_path: Path) -> np.ndarray:
    """Axis-aligned USD size (m). Raises if the stage/bbox cannot be read."""
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
    """Table placement z-offset from a USD bbox (prefer Z extent)."""
    hz = float(size[2]) * 0.5 if size.shape[0] >= 3 else float(np.max(size)) * 0.5
    return float(max(hz, 1e-3))


def pick_usd(
    catalog: list[Path],
    rng: np.random.Generator,
    override: str | Path | None,
    *,
    object_dir: Path | None = None,
) -> Path:
    """Pick a USD from catalog, or resolve an override.

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
    if object_dir is not None:
        cat_dir = object_dir / raw
        if cat_dir.is_dir():
            cat_usds = sorted(cat_dir.glob("*.usd"))
            if not cat_usds:
                raise FileNotFoundError(f"No .usd files in category dir: {cat_dir}")
            return cat_usds[int(rng.integers(0, len(cat_usds)))]

    stem = path.stem if raw.endswith(".usd") else raw
    matches = [p for p in catalog if p.stem == stem or p.name == raw]
    if matches:
        return matches[int(rng.integers(0, len(matches)))]

    raise FileNotFoundError(
        f"USD override not found: {raw!r}. "
        "Pass a .usd path, a category name (e.g. can / bowl), or a stem (e.g. can00)."
    )


class Phase0Scene:
    """Single-env Franka pick scene with scripted object motion hooks."""

    def __init__(
        self,
        cfg: dict,
        *,
        enable_cameras: bool = True,
        object_usd: Path | str | None = None,
        container_usd: Path | str | None = None,
        asset_seed: int = 40,
    ):
        """Spawn Franka, table, USD object+container, cameras.

        Yaml: ``sim.dt/device``, ``scene.*``, ``robot.*``, ``camera.*``, ``paths.*``.
        """
        self.cfg = cfg
        self.enable_cameras = bool(enable_cameras)
        device = str(_require(cfg, "sim", "device"))
        dt = float(_require(cfg, "sim", "dt"))
        self.dt = dt
        self.device = device
        self._asset_seed = int(asset_seed)

        sim_cfg = sim_utils.SimulationCfg(dt=dt, device=device)
        self.sim = sim_utils.SimulationContext(sim_cfg)

        sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())
        light_cfg = sim_utils.DomeLightCfg(intensity=800.0, color=(0.95, 0.95, 0.95))
        light_cfg.func("/World/Light", light_cfg)

        sc = _require(cfg, "scene")
        try:
            table_size = tuple(float(x) for x in sc["table_size"])
            table_pos = tuple(float(x) for x in sc["table_pos"])
        except KeyError as ex:
            logger.error("missing required yaml key: scene.%s", ex.args[0])
            raise KeyError(f"scene.{ex.args[0]}") from None
        table_cfg = sim_utils.CuboidCfg(
            size=table_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.35, 0.25)),
        )
        table_cfg.func("/World/Table", table_cfg, translation=table_pos)

        robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.copy()
        robot_cfg.prim_path = "/World/Robot"
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

        asset_rng = np.random.default_rng(self._asset_seed)
        self._object_usds: list[Path] = []
        self._active_usd: Path | None = None
        self._object_size: np.ndarray | None = None
        self._object_half_height = self._object_radius

        if not self._use_primitive:
            self._object_usds = list_category_usds(self._object_dir, self._categories)
            self._active_usd = pick_usd(
                self._object_usds,
                asset_rng,
                object_usd,
                object_dir=self._object_dir,
            )
            self._object_size = usd_bbox_size(self._active_usd)
            self._object_half_height = half_height_from_size(self._object_size)
            from sim.state_machine_logs import fmt_size_cm

            logger.info(
                "Object mesh: %s   size %s   sits %.1f cm on the table",
                self._active_usd.stem if self._active_usd is not None else "unknown",
                fmt_size_cm(self._object_size),
                100.0 * float(self._object_half_height),
            )

        z0 = float(_require(cfg, "scene", "table_height")) + self._object_half_height
        self.object = self._make_rigid(
            prim_path="/World/Object",
            usd_path=self._active_usd,
            pos=np.array([0.50, 0.0, z0]),
            lin_vel=np.zeros(3),
            mass=float(_require(cfg, "scene", "object_mass")),
            # Object motion is ours (pose write). Dynamic PhysX was adding a second
            # v·dt so "20 cm/s" bins ran ~33 cm/s. Ablation still *samples*
            # 0/10/15/20; each episode stays constant-velocity at that command.
            kinematic=True,
            primitive_sphere=self._use_primitive,
        )

        self.container: RigidObject | None = None
        self._container_usd: Path | None = None
        self._container_size: np.ndarray | None = None
        self._container_half_height = 0.0
        ctr_xy = _require(cfg, "scene", "container_xy")
        self._container_pos = np.array(
            [float(ctr_xy[0]), float(ctr_xy[1]), 0.0],
            dtype=np.float64,
        )

        if (not self._use_primitive) and self._n_containers >= 1 and self._container_categories:
            ctr_usds = list_category_usds(self._object_dir, self._container_categories)
            self._container_usd = pick_usd(
                ctr_usds,
                asset_rng,
                container_usd,
                object_dir=self._object_dir,
            )
            self._container_size = usd_bbox_size(self._container_usd)
            self._container_half_height = half_height_from_size(self._container_size)
            cz = float(_require(cfg, "scene", "table_height")) + self._container_half_height
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

        if self.container is None:
            raise RuntimeError(
                "Place needs a container. Set scene.n_containers >= 1, "
                "scene.container_categories, and use_primitive: false."
            )

        self.cameras: dict[str, Camera] = {}
        if self.enable_cameras:
            self._spawn_cameras()

        self.sim.set_camera_view(eye=[1.5, 1.2, 1.0], target=[0.45, 0.0, 0.2])
        self.sim.reset()

        self._ee_idx = self.robot.data.body_names.index(EE_BODY)
        names = list(self.robot.data.joint_names)
        self._arm_idx = [names.index(n) for n in names if n.startswith(ARM_JOINT_EXPR)]
        self._finger_idx = [names.index(n) for n in FINGER_JOINTS]
        self._jac_idx = self._ee_idx - 1

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

        self._obj_velocity = np.zeros(3, dtype=np.float64)
        self._obj_speed_cmd = np.zeros(3, dtype=np.float64)
        # Observed velocity after the last physics step (FD of root pose, or
        # root_lin_vel_w). Commanded motion speed stays in ``_obj_velocity``.
        self._obj_vel_obs = np.zeros(3, dtype=np.float64)
        self._prev_obj_pos: np.ndarray | None = None
        self._obj_kinematic = True
        self._table_z = float(_require(cfg, "scene", "table_height"))
        bounds = _require(cfg, "scene", "object_xy_bounds")
        self._xy_bounds = (
            (float(bounds[0][0]), float(bounds[0][1])),
            (float(bounds[1][0]), float(bounds[1][1])),
        )
        self._sim_time = 0.0
        self._gripper_open = float(_require(cfg, "robot", "gripper_open"))
        self._gripper_close = float(_require(cfg, "robot", "gripper_close"))
        self._meta: dict[str, Any] = {}
        self._attached = False
        self._attach_offset = np.zeros(3, dtype=np.float64)

    def _spawn_cameras(self) -> None:
        """Spawn static cams from ``scene.cameras`` plus wrist from ``camera.wrist_*``."""
        cam_cfg = _require(self.cfg, "camera")
        for cam in _require(self.cfg, "scene", "cameras"):
            quat = _euler_deg_to_wxyz(cam["rotation_deg"])
            name = cam["name"]
            self.cameras[name] = Camera(
                CameraCfg(
                    prim_path=f"/World/{name}",
                    update_period=0.0,
                    height=int(cam_cfg["height"]),
                    width=int(cam_cfg["width"]),
                    data_types=list(cam_cfg["data_types"]),
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=float(cam_cfg["focal_length"]),
                        focus_distance=float(cam_cfg["focus_distance"]),
                        horizontal_aperture=float(cam_cfg["horizontal_aperture"]),
                        clipping_range=(
                            float(_require(self.cfg, "camera", "clip", "near")),
                            float(_require(self.cfg, "camera", "clip", "far")),
                        ),
                    ),
                    offset=CameraCfg.OffsetCfg(
                        pos=tuple(cam["position"]),
                        rot=tuple(quat),
                        convention="opengl",
                    ),
                )
            )
            nice = "Table camera" if name == "static_cam" else f"Camera {name}"
            logger.info("%s at %s m", nice, tuple(cam["position"]))

        wrist_pos = tuple(float(x) for x in _require(self.cfg, "camera", "wrist_pos"))
        wrist_quat_wxyz = tuple(
            float(x) for x in _require(self.cfg, "camera", "wrist_quat_wxyz")
        )
        wrist_clip = _require(self.cfg, "camera", "wrist_clipping_range")
        self.cameras["wrist_cam"] = Camera(
            CameraCfg(
                prim_path="/World/Robot/panda_hand/WristCamera",
                update_period=0.0,
                height=int(cam_cfg["height"]),
                width=int(cam_cfg["width"]),
                data_types=list(cam_cfg["data_types"]),
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=float(cam_cfg["focal_length"]),
                    focus_distance=float(cam_cfg["focus_distance"]),
                    horizontal_aperture=float(cam_cfg["horizontal_aperture"]),
                    clipping_range=(float(wrist_clip[0]), float(wrist_clip[1])),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=wrist_pos,
                    rot=wrist_quat_wxyz,
                    convention="opengl",
                ),
            )
        )
        logger.info("Wrist camera on the hand.")

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
        """Spawn a rigid body (USD, or sphere if ``scene.use_primitive``)."""
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
        """Return (center_xyz, half_height) or None if no container."""
        if self.container is None:
            return None
        return self._container_pos.copy(), float(self._container_half_height)

    def get_carry_pose(self) -> np.ndarray:
        """Hand pose over the container: XY + rim + ``scene.place_hover_above_container``."""
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
        gap = float(_require(self.cfg, "state_machine_params", "drop_z_gap"))
        # Prefer drop_rim_clear_m; fall back to grasp_z_tol until yaml adds the key.
        sm = _require(self.cfg, "state_machine_params")
        rim_clear = float(sm.get("drop_rim_clear_m", sm["grasp_z_tol"]))
        hover_clear = float(_require(self.cfg, "state_machine_params", "approach_z_tol"))
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
        """Reset robot + object pose/vel. Uses ``spawn_xy_pad``, container jitter/clip."""
        rng = np.random.default_rng(int(seed))

        speed = float(speed)
        if speed <= 1e-6:
            lin_vel = np.zeros(3, dtype=np.float64)
        else:
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            lin_vel = np.array(
                [speed * np.cos(angle), speed * np.sin(angle), 0.0], dtype=np.float64
            )

        (xmin, xmax), (ymin, ymax) = self._xy_bounds
        pad = self._spawn_xy_pad
        x = float(rng.uniform(xmin + pad, xmax - pad))
        y = float(rng.uniform(ymin + pad, ymax - pad))
        z = self._table_z + self._object_half_height
        pos = np.array([x, y, z], dtype=np.float64)
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        if self.container is not None:
            cx0, cy0 = float(self._container_pos[0]), float(self._container_pos[1])
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
        root[0, 7:10] = torch.tensor(lin_vel, device=self.robot.device)
        self.object.write_root_state_to_sim(root)
        self.object.write_data_to_sim()

        if self.container is not None:
            croot = torch.zeros((1, 13), device=self.robot.device, dtype=torch.float32)
            croot[0, 0:3] = torch.tensor(self._container_pos, device=self.robot.device)
            croot[0, 3:7] = torch.tensor(quat, device=self.robot.device)
            self.container.write_root_state_to_sim(croot)
            self.container.write_data_to_sim()

        self.robot.write_data_to_sim()

        for _ in range(8):
            self.sim.step()
            self.robot.update(self.dt)
            self.object.update(self.dt)
            if self.container is not None:
                self.container.update(self.dt)

        self._obj_velocity = lin_vel.copy()
        self._obj_speed_cmd = lin_vel.copy()
        self._obj_vel_obs = lin_vel.copy()
        self._prev_obj_pos = (
            self.object.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64).copy()
        )
        self._obj_kinematic = True
        self._attached = False
        self._attach_offset = np.zeros(3, dtype=np.float64)
        self._sim_time = 0.0
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
        }
        return dict(self._meta)

    def get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
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
        pos = self.object.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
        # Post-step pose FD. With kinematic_enabled, this should match the
        # scripted motion command (constant velocity + wall bounce).
        vel = self._obj_vel_obs.copy()
        return oracle_from_gt(pos, vel, timestamp=self._sim_time, valid=True)

    def _refresh_object_vel_obs(self) -> None:
        """Set ``_obj_vel_obs`` from pose FD after solver integration.

        While attached, force zeros so EE teleports do not spike logged velocity.
        Still refresh ``_prev_obj_pos`` so release does not FD-spike either.
        """
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
        """Constant-velocity future (same as ``conditioning_vector`` / ``ObjectState.predict``).

        Table bounce stays in ``kinematic_advance`` (world motion), not in this predictor.
        When the object is welded to the EE, robot motion is unknown → return now.
        """
        now = self.get_object_state()
        if not self._obj_kinematic or self._attached:
            return now
        return now.predict(float(delta))

    def get_proprio(self) -> np.ndarray:
        return self.robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float64)

    def get_images(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for name, cam in self.cameras.items():
            if "rgb" in cam.data.output:
                out[name] = cam.data.output["rgb"][0].detach().cpu().numpy()
        return out

    def _write_object_pose(self, pos: np.ndarray, quat: np.ndarray, vel: np.ndarray) -> None:
        pose = torch.zeros((1, 7), device=self.robot.device, dtype=torch.float32)
        pose[0, 0:3] = torch.tensor(pos, device=self.robot.device)
        pose[0, 3:7] = torch.tensor(quat, device=self.robot.device)
        self.object.write_root_pose_to_sim(pose)
        v = torch.zeros((1, 6), device=self.robot.device, dtype=torch.float32)
        v[0, 0:3] = torch.tensor(vel, device=self.robot.device)
        self.object.write_root_velocity_to_sim(v)

    def _write_kinematic_object(self) -> None:
        if not self._obj_kinematic:
            return
        pos = self.object.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
        quat = self.object.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
        new_pos, new_vel = kinematic_advance(
            pos, self._obj_velocity, self.dt, xy_bounds=self._xy_bounds
        )
        new_pos[2] = self._table_z + self._object_half_height
        self._obj_velocity = new_vel
        self._write_object_pose(new_pos, quat, new_vel)

    def _write_attached_object(self) -> None:
        ee_pos, ee_quat = self.get_ee_pose()
        pos = ee_pos + self._attach_offset
        self._obj_velocity = np.zeros(3, dtype=np.float64)
        self._write_object_pose(pos, ee_quat, self._obj_velocity)

    def set_attach(self, attach: bool) -> None:
        """Weld object to EE (scripted grasp) or release. Uses ``scene.attach_z_offset_max``."""
        if attach and not self._attached:
            ee_pos, _ = self.get_ee_pose()
            obj = self.object.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
            z_off = float(
                np.clip(float(obj[2] - ee_pos[2]), self._attach_z_offset_max, 0.0)
            )
            self._attach_offset = np.array([0.0, 0.0, z_off], dtype=np.float64)
            self._attached = True
            self._obj_kinematic = False
        elif not attach and self._attached:
            self._attached = False
            self._attach_offset = np.zeros(3, dtype=np.float64)

    def set_ee_target(
        self,
        pos_w: np.ndarray,
        quat_wxyz: np.ndarray,
        gripper_open: float,
    ) -> None:
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
        finger = self._gripper_close + g * (self._gripper_open - self._gripper_close)
        for fi in self._finger_idx:
            target[0, fi] = finger
        self.robot.set_joint_position_target(target)

    def step(
        self,
        *,
        kinematic_object: bool,
        attach_object: bool = False,
        freeze_object: bool = False,
    ) -> None:
        self.set_attach(bool(attach_object))
        if freeze_object:
            # Pin + zero vel even while attached (close must not slide).
            self._obj_velocity[:] = 0.0
            pos = self.object.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
            quat = self.object.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
            self._write_object_pose(pos, quat, self._obj_velocity)
            self._obj_kinematic = True
        elif self._attached:
            self._write_attached_object()
        else:
            # Resume commanded motion velocity after a freeze window.
            self._obj_velocity = self._obj_speed_cmd.copy()
            self._obj_kinematic = bool(kinematic_object)
            if self._obj_kinematic:
                self._write_kinematic_object()
                self._obj_speed_cmd = self._obj_velocity.copy()
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
        self.sim.step()
        self.robot.update(self.dt)
        self.object.update(self.dt)
        if self.container is not None:
            self.container.update(self.dt)
        for cam in self.cameras.values():
            cam.update(self.dt)
        self._refresh_object_vel_obs()
        self._sim_time += self.dt

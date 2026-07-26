"""In-repo Franka Phase-0 scene (Isaac Lab). Import only AFTER AppLauncher.

Owns: Franka Panda + table + one rigid object + static/wrist cameras + DiffIK.
Does not import DynamicVLA.
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


def load_cfg(path: str | Path) -> dict:
    with open(path) as fp:
        return yaml.load(fp, Loader=yaml.FullLoader)


def _euler_deg_to_wxyz(rotation_deg: list[float]) -> list[float]:
    from scipy.spatial.transform import Rotation as R

    q = R.from_euler("XYZ", rotation_deg, degrees=True).as_quat()  # xyzw
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def _list_object_usds(object_dir: Path, categories: list[str]) -> list[Path]:
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


class Phase0Scene:
    """Single-env Franka pick scene with scripted object motion hooks."""

    def __init__(self, cfg: dict, *, enable_cameras: bool = True, object_usd: Path | None = None):
        self.cfg = cfg
        self.enable_cameras = bool(enable_cameras)
        device = cfg.get("sim", {}).get("device", "cuda:0")
        dt = float(cfg.get("sim", {}).get("dt", 0.04))
        self.dt = dt
        self.device = device

        sim_cfg = sim_utils.SimulationCfg(dt=dt, device=device)
        self.sim = sim_utils.SimulationContext(sim_cfg)

        sim_utils.GroundPlaneCfg().func("/World/GroundPlane", sim_utils.GroundPlaneCfg())
        light_cfg = sim_utils.DomeLightCfg(intensity=800.0, color=(0.95, 0.95, 0.95))
        light_cfg.func("/World/Light", light_cfg)

        sc = cfg["scene"]
        table_size = tuple(float(x) for x in sc["table_size"])
        table_pos = tuple(float(x) for x in sc["table_pos"])
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

        self._object_dir = Path(cfg["paths"]["object_dir"])
        self._categories = list(sc["object_categories"])
        self._use_primitive = bool(sc.get("use_primitive", True))
        self._object_radius = float(sc.get("object_radius", 0.03))
        self._object_usds: list[Path] = []
        self._active_usd: Path | None = None
        if not self._use_primitive:
            self._object_usds = _list_object_usds(self._object_dir, self._categories)
            self._active_usd = Path(object_usd) if object_usd else self._object_usds[0]
        z0 = float(sc["table_height"]) + self._object_radius
        self.object = self._make_object(
            self._active_usd,
            pos=np.array([0.50, 0.0, z0]),
            lin_vel=np.zeros(3),
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

        tcp = cfg["robot"]["tcp_offset"]
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
        self._obj_kinematic = True
        self._table_z = float(sc["table_height"])
        self._xy_bounds = ((0.30, 0.70), (-0.30, 0.30))
        self._sim_time = 0.0
        self._gripper_open = float(cfg["robot"]["gripper_open"])
        self._gripper_close = float(cfg["robot"]["gripper_close"])
        self._meta: dict[str, Any] = {}
        self._attached = False
        self._attach_offset = np.zeros(3, dtype=np.float64)

    def _spawn_cameras(self) -> None:
        cam_cfg = self.cfg["camera"]
        for cam in self.cfg["scene"]["cameras"]:
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
                            float(cam_cfg["clip"]["near"]),
                            float(cam_cfg["clip"]["far"]),
                        ),
                    ),
                    offset=CameraCfg.OffsetCfg(
                        pos=tuple(cam["position"]),
                        rot=tuple(quat),
                        convention="opengl",
                    ),
                )
            )
            logger.info("Camera %s at %s", name, cam["position"])

        self.cameras["wrist_cam"] = Camera(
            CameraCfg(
                prim_path="/World/Robot/panda_hand/WristCamera",
                update_period=0.0,
                height=int(cam_cfg["height"]),
                width=int(cam_cfg["width"]),
                data_types=list(cam_cfg["data_types"]),
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=6.0,
                    focus_distance=float(cam_cfg["focus_distance"]),
                    horizontal_aperture=float(cam_cfg["horizontal_aperture"]),
                    clipping_range=(0.01, 2.0),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(0.05, 0.0, 0.0),
                    rot=tuple(_euler_deg_to_wxyz([0.0, -90.0, 0.0])),
                    convention="opengl",
                ),
            )
        )

    def _make_object(
        self, usd_path: Path | None, pos: np.ndarray, lin_vel: np.ndarray
    ) -> RigidObject:
        sc = self.cfg["scene"]
        mass = float(sc["object_mass"])
        rigid = sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )
        collision = sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.01,
            rest_offset=0.0,
        )
        if self._use_primitive or usd_path is None:
            # Reliable collision geometry for G0; USD household assets optional later.
            spawner = sim_utils.SphereCfg(
                radius=self._object_radius,
                rigid_props=rigid,
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                collision_props=collision,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.15, 0.1)),
            )
        else:
            spawner = sim_utils.UsdFileCfg(
                usd_path=str(usd_path),
                rigid_props=rigid,
                mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                collision_props=collision,
            )
        cfg = RigidObjectCfg(
            prim_path="/World/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(float(pos[0]), float(pos[1]), float(pos[2])),
                lin_vel=(float(lin_vel[0]), float(lin_vel[1]), float(lin_vel[2])),
            ),
            spawn=spawner,
        )
        return RigidObject(cfg)

    def reset_episode(self, *, seed: int, speed: float) -> dict[str, Any]:
        """Reset robot joints + object pose/velocity (same geometry for process lifetime)."""
        rng = np.random.default_rng(int(seed))

        speed = float(speed)
        if speed <= 1e-6:
            lin_vel = np.zeros(3, dtype=np.float64)
        else:
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            lin_vel = np.array(
                [speed * np.cos(angle), speed * np.sin(angle), 0.0], dtype=np.float64
            )

        x = float(rng.uniform(0.40, 0.55))
        y = float(rng.uniform(-0.12, 0.12))
        z = self._table_z + self._object_radius
        pos = np.array([x, y, z], dtype=np.float64)
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        self.robot.reset()
        self.object.reset()
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
        self.robot.write_data_to_sim()

        for _ in range(8):
            self.sim.step()
            self.robot.update(self.dt)
            self.object.update(self.dt)

        self._obj_velocity = lin_vel.copy()
        self._obj_speed_cmd = lin_vel.copy()
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
        if self._obj_kinematic:
            vel = self._obj_velocity.copy()
        else:
            vel = (
                self.object.data.root_lin_vel_w[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        return oracle_from_gt(pos, vel, timestamp=self._sim_time, valid=True)

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
        new_pos[2] = self._table_z + self._object_radius
        self._obj_velocity = new_vel
        self._write_object_pose(new_pos, quat, new_vel)

    def _write_attached_object(self) -> None:
        ee_pos, ee_quat = self.get_ee_pose()
        pos = ee_pos + self._attach_offset
        self._obj_velocity = np.zeros(3, dtype=np.float64)
        self._write_object_pose(pos, ee_quat, self._obj_velocity)

    def set_attach(self, attach: bool) -> None:
        """Weld object to EE (scripted grasp) or release."""
        if attach and not self._attached:
            ee_pos, _ = self.get_ee_pose()
            obj = self.object.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
            self._attach_offset = obj - ee_pos
            # Keep a small downward offset so object sits in the fingers.
            self._attach_offset[2] = min(self._attach_offset[2], -0.005)
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
        if self._attached:
            self._write_attached_object()
        else:
            if freeze_object:
                self._obj_velocity[:] = 0.0
            else:
                # Resume commanded conveyor velocity after a freeze window.
                self._obj_velocity = self._obj_speed_cmd.copy()
            self._obj_kinematic = bool(kinematic_object) or bool(freeze_object)
            if self._obj_kinematic:
                self._write_kinematic_object()
                # Keep bounce updates in the speed command for next resume.
                if not freeze_object:
                    self._obj_speed_cmd = self._obj_velocity.copy()
        self.robot.write_data_to_sim()
        self.object.write_data_to_sim()
        self.sim.step()
        self.robot.update(self.dt)
        self.object.update(self.dt)
        for cam in self.cameras.values():
            cam.update(self.dt)
        self._sim_time += self.dt

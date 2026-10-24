# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import math
import torch
import torch.nn.functional as F

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor
    from isaaclab.sensors import Camera, Imu, JointWrenchSensor, Pva, RayCaster, RayCasterCamera


def upright_orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    g_xy_norm = torch.linalg.norm(asset.data.projected_gravity_b.torch[:, :2], dim=1)
    return torch.exp(-2.0 * g_xy_norm**2) + 0.1 * torch.exp(-g_xy_norm)


def velocity_slack_bonus(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_x_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w.torch), asset.data.root_lin_vel_w.torch[:, :3])[:, 0]
    cmd_vx = env.command_manager.get_command(command_name)[:, 0] + 1e-8

    ratio = vel_x_yaw / cmd_vx
    in_band = (ratio >= 0.3) & (ratio <= 1.5)
    return in_band.float()


def joint_limit(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    out_of_limits = -(
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits.torch[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0) * (joint_vel < 0.0)
    out_of_limits += (
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits.torch[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0) * (joint_vel > 0.0)
    return torch.sum(out_of_limits, dim=1)


def illegal_footstep(
    env: ManagerBasedEnv, left_sensor_cfg: SceneEntityCfg, right_sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg, offset: float = 0.035,
) -> torch.Tensor:
    left_raycaster_sensor: RayCaster = env.scene.sensors[left_sensor_cfg.name]
    right_raycaster_sensor: RayCaster = env.scene.sensors[right_sensor_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    left_distance = left_raycaster_sensor.data.pos_w.torch[:, 2].unsqueeze(1) - left_raycaster_sensor.data.ray_hits_w.torch[..., 2] - offset
    right_distance = right_raycaster_sensor.data.pos_w.torch[:, 2].unsqueeze(1) - right_raycaster_sensor.data.ray_hits_w.torch[..., 2] - offset
    left_pct = torch.sum((left_distance > 0.1).float(), dim=1) / left_raycaster_sensor.num_rays
    right_pct = torch.sum((right_distance > 0.1).float(), dim=1) / right_raycaster_sensor.num_rays
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_normal_forces_w_history.torch
    is_contact = (
        torch.max(torch.linalg.norm(net_contact_forces[:, :, contact_sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1.0
    )
    return left_pct * is_contact[:, 0] + right_pct * is_contact[:, 1]


def heading_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    heading_error = env.command_manager.get_term(command_name).heading_target - asset.data.heading_w.torch
    return torch.abs(heading_error)


def opposite_direction(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w.torch), asset.data.root_lin_vel_w.torch[:, :3])[:, :2]
    cmd_xy_dir = F.normalize(env.command_manager.get_command(command_name)[:, :2], dim=-1)
    return torch.clamp(-(vel_yaw * cmd_xy_dir).sum(dim=-1), min=0.0)


class foot_acceleration(ManagerTermBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        tau = cfg.params.get("tau", 0.1)
        delta_t = env.step_dt
        self.alpha = math.exp(-delta_t / tau)
        self.e = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        self.e[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(self, env, asset_cfg: SceneEntityCfg, a_thresh: float = 30.0, tau: float = 0.1):
        asset: RigidObject = env.scene[asset_cfg.name]
        acc = torch.linalg.norm(asset.data.body_lin_acc_w.torch[:, asset_cfg.body_ids, :], dim=-1)
        excess = torch.clamp(acc - a_thresh, min=0.0).sum(dim=-1)
        active = env.episode_length_buf > 8
        self.e = self.alpha * self.e + excess * active.float()
        return self.e

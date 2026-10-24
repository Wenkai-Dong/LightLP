# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject, Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def subterrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    if env.scene.cfg.terrain.terrain_type == "plane":
        # we have infinite terrain because it is a plane
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]
        subterrain_origins = env.scene.env_origins[:, :2]
        pos_rel = asset.data.root_pos_w.torch[:, :2] - subterrain_origins

        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(pos_rel[:, 0]) > 0.5 * grid_width + distance_buffer
        y_out_of_bounds = torch.abs(pos_rel[:, 1]) > 0.5 * grid_length + distance_buffer
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")


def excessive_acceleration(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    acc = torch.linalg.norm(asset.data.body_lin_acc_w.torch[:, asset_cfg.body_ids, :], dim=-1).squeeze(-1)
    warm = env.episode_length_buf * env.step_dt > 1.0
    return (acc > 40.0) & warm & ~env.immunity_flag


def fall_over(
    env: ManagerBasedRLEnv, limit_angle: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's orientation is too far from the desired orientation limits.

    This is computed by checking the angle between the projected gravity vector and the z-axis.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    bad_orientation = torch.acos(-asset.data.projected_gravity_b.torch[:, 2]).abs() > limit_angle
    prob = torch.rand(env.num_envs, device=env.device) < 0.01
    return bad_orientation & prob


def illegal_contact_immunity(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_normal_forces_w_history.torch
    # check if any contact force exceeds the threshold
    return torch.any(
        torch.max(torch.linalg.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    ) & ~env.immunity_flag
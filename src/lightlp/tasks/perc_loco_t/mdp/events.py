# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from typing import TYPE_CHECKING, Literal

import torch

from isaaclab.envs.mdp.events import randomize_actuator_gains, _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv


def resample_immunity_flag(env: ManagerBasedRLEnv, env_ids, ratio: float = 0.1) -> None:
    env.immunity_flag[:] = torch.rand(env.num_envs, device=env.device) < ratio

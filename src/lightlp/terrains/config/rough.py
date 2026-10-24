# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

import lightlp.terrains as game_terrain_gen

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(10.0, 10.0),
    border_width=10.0,
    num_rows=10,
    num_cols=32,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "random_rough": game_terrain_gen.HfRandomUniformDifficultyTerrainCfg(
            proportion=0.1, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "flat": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.0, 0.0), noise_step=0.02, border_width=0.25
        ),
        "rail": terrain_gen.MeshRailsTerrainCfg(
            proportion=0.2,
            rail_thickness_range=(0.1, 0.2),
            rail_height_range=(0.05, 0.45),
            platform_width=1.5,
        ),
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.6,
            stone_height_max=0.1,
            stone_width_range=(0.3, 0.6),
            stone_distance_range=(0.05, 0.25),
            holes_depth=-0.6,
            platform_width=1.5,
            border_width=0.25,
        ),
        "stones": terrain_gen.MeshRepeatedBoxesTerrainCfg(
            proportion=0.4,
            abs_height_noise=(-0.5, 0.5),
            object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=1280, height=0.05, size=(0.13, 0.26), max_yx_angle=0.0, degrees=True
            ),
            object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=2000, height=0.2, size=(0.13, 0.26), max_yx_angle=0.0, degrees=True
            ),
            platform_width=1.5,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.4,
            grid_width=0.45,
            grid_height_range=(0.05, 0.2),
            platform_width=2.0,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "beams": terrain_gen.MeshStarTerrainCfg(
            proportion=0.6,
            num_bars=5,
            bar_width_range=(0.1, 0.6),
            bar_height_range=(0.6, 0.6),
        ),
    },
)
"""Rough terrains configuration."""

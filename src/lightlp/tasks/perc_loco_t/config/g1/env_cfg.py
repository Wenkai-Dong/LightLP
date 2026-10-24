# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

from isaaclab_newton.physics import (
    KaminoPADMMSolverCfg,
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
)
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass, replace
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.utils import PresetCfg

from lightlp.assets.robots import UNITREE_G1_29DOF_MIMIC_CFG
from lightlp.terrains.config.rough import ROUGH_TERRAINS_CFG

from ... import mdp

##
# Physics backend presets
##


@configclass
class RoughPhysicsCfg(PresetCfg):
    """Shared backend presets for locomotion velocity environments."""

    isaacsim_physx = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    ovphysx = OvPhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    physx = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=1000,
            nconmax=300,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        num_substeps=2,
        debug_mode=False,
        default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0),
    )
    newton_kamino = NewtonCfg(solver_cfg=KaminoPADMMSolverCfg(max_contacts_per_world=64))
    default = newton_mjwarp


##
# Scene definition
##


@configclass
class PercLocoTSceneCfg(InteractiveSceneCfg):
    """Define robot assets, ground, lighting, and other scene objects."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",   # "plane", "generator"
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = replace(UNITREE_G1_29DOF_MIMIC_CFG, prim_path="{ENV_REGEX_NS}/Robot")
    # sensors: the concrete implementation is selected automatically from the active physics backend;
    # backend-specific fields such as ``global_world_only`` are ignored by the backends that do not use them
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.4, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
    )
    left_foot_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.045, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=[0.19, 0.05]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
    )
    right_foot_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.045, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=[0.19, 0.05]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/[^/]*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.5), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Define action terms for your robot."""

    action: ActionTermCfg = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Define observation groups and terms."""

    @configclass
    class ActorCfg(ObsGroup):
        """Policy observation terms."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        contact_flag = ObsTerm(
            func=mdp.contact_flag,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"), "threshold": 1.0},
    )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5


    @configclass
    class CriticCfg(ActorCfg):
        immunity_flag = ObsTerm(func=mdp.immunity_flag)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = None


    @configclass
    class HeightScanCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True


    @configclass
    class FootHeightScanCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        left_foot_height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("left_foot_height_scanner")},
            clip=(-1.0, 1.0),
        )
        right_foot_height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("right_foot_height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    actor: ActorCfg = ActorCfg()
    critic: CriticCfg = CriticCfg()
    height_scan: HeightScanCfg = HeightScanCfg()
    foot_height_scan: FootHeightScanCfg = FootHeightScanCfg()


@configclass
class EventCfg:
    """Define reset and other event terms."""

    # resample_immunity_flag
    resample_immunity_flag = EventTerm(
        func=mdp.resample_immunity_flag, mode="interval", interval_range_s=(4.0, 4.0), is_global_time=True
    )
    # Ground contact (startup)
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.3),
            "dynamic_friction_range": (0.2, 1.3),
            "restitution_range": (0.0, 0.8),
            "num_buckets": 64,
        },
    )
    # Inertial properties (startup)
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            # multiplicative +-25% log-uniform: scale-invariant across robot sizes with geometric mean 1.0
            "mass_distribution_params": (1 / 1.25, 1.25),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    torso_payload = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            # multiplicative +-25% log-uniform: scale-invariant across robot sizes with geometric mean 1.0
            "mass_distribution_params": (1.0, 1.2),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25), "z": (-0.25, 0.25)},
        },
    )
    # Actuation (slow resample)
    randomize_joint_params = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        min_step_count_between_reset=50000,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "operation": "scale",
            "distribution": "uniform",
            "armature_distribution_params": (0.75, 1.25),
            "friction_distribution_params": (0.7, 1.3),
        },
    )
    randomize_joint_viscous_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        min_step_count_between_reset=50000,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "operation": "add",
            "distribution": "uniform",
            "friction_distribution_params": (0.0, 0.05)
        },
    )
    joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        min_step_count_between_reset=50000,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "operation": "scale",
            "distribution": "uniform",
            "stiffness_distribution_params": [0.875, 1.075],
            "damping_distribution_params": [0.875, 1.075],
        },
    )
    # Disturbance (interval)
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.7, 4.2),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )
    # Initialization (reset)
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (-1.0, 1.0),
        },
    )


@configclass
class RewardsCfg:
    """Define task reward terms and weights."""

    # Tracking rewards
    linear_velocity_tracking = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    angular_velocity_tracking = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    upright_orientation = RewTerm(func=mdp.upright_orientation, weight=1.0)
    velocity_slack_bonus = RewTerm(func=mdp.velocity_slack_bonus, weight=1.5, params={"command_name": "base_velocity"})
    # Penalties
    undesired_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]), "threshold": 1.0},
    )
    joint_limit = RewTerm(func=mdp.joint_limit, weight=-10.0)
    illegal_footstep = RewTerm(
        func=mdp.illegal_footstep,
        weight=-1.0,
        params={
            "left_sensor_cfg": SceneEntityCfg("left_foot_height_scanner"),
            "right_sensor_cfg": SceneEntityCfg("right_foot_height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    heading_error = RewTerm(func=mdp.heading_error, weight=-1.0, params={"command_name": "base_velocity"},)
    opposite_direction = RewTerm(func=mdp.opposite_direction, weight=-1.0, params={"command_name": "base_velocity"})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    foot_acceleration = RewTerm(
        func=mdp.foot_acceleration,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"), "tau": 0.06}
    )


@configclass
class TerminationsCfg:
    """Define termination and timeout terms."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    out_of_bounds = DoneTerm(func=mdp.subterrain_out_of_bounds, time_out=True, params={"distance_buffer": 2.0},)
    Joint_velocity_guard = DoneTerm(
        func=mdp.joint_vel_out_of_manual_limit, time_out=True,
        params={"max_velocity": 30.0, "asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    torso_contact = DoneTerm(
        func=mdp.illegal_contact_immunity,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0},
    )
    excessive_acceleration = DoneTerm(func=mdp.excessive_acceleration)
    fall_over = DoneTerm(func=mdp.fall_over, params={"limit_angle": 1.1})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.sub_terrain_levels_vel)


@configclass
class PercLocoTEnvCfg(ManagerBasedRLEnvCfg):
    """Configure the manager-based task."""

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(physics=RoughPhysicsCfg())
    # Scene settings
    scene: PercLocoTSceneCfg = PercLocoTSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.left_foot_height_scanner is not None:
            self.scene.left_foot_height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.right_foot_height_scanner is not None:
            self.scene.right_foot_height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

    def play_mode(self):
        """Play-mode overrides shared by the velocity-tracking environments."""
        super().play_mode()
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        # remove random pushing events
        self.events.base_external_force_torque = None
        self.events.push_robot = None


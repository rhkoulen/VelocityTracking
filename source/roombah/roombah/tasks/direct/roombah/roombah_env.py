# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .roombah_env_cfg import RoombahEnvCfg


class RoombahEnv(DirectRLEnv):
    cfg: RoombahEnvCfg

    def __init__(self, cfg: RoombahEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Here's my terrible plan, keep a list of waypoints to which the robot can drive in a loop.
        # Each waypoint has a target speed, which is how fast the robot is supposed to be going on that segment.
        self.waypoints = torch.zeros((self.cfg.scene.num_envs, self.cfg.num_waypoints, 2)).cuda()
        self.target_speeds = torch.zeros((self.cfg.scene.num_envs, self.cfg.num_waypoints)).cuda()
        self.target_idx = torch.zeros(self.cfg.scene.num_envs, dtype=torch.long).cuda()
        self._sample_waypoints(torch.arange(self.cfg.scene.num_envs).cuda())

    def _sample_waypoints(self, env_ids):
        n = len(env_ids)
        dev = self.waypoints.device
        angles = torch.rand((n, self.cfg.num_waypoints), device=dev) * 2 * math.pi
        radii = self.cfg.waypoint_radius * torch.sqrt(torch.rand((n, self.cfg.num_waypoints), device=dev))
        offsets = torch.stack([radii * torch.cos(angles), radii * torch.sin(angles)], dim=-1)
        origins = self.scene.env_origins[env_ids, :2].unsqueeze(1)
        self.waypoints[env_ids] = origins + offsets

        low, high = self.cfg.speed_range
        self.target_speeds[env_ids] = low + (high - low) * torch.rand((n, self.cfg.num_waypoints), device=dev)
        self.target_idx[env_ids] = 0

    def _lookahead_local_and_speeds(self):
        dev = self.target_idx.device
        env_ids = torch.arange(self.cfg.scene.num_envs, device=dev).unsqueeze(-1) # (N, 1)
        # For each environment, take the current index and add [+0, +1, +2, ...] to get waypoints for the next objs
        step = torch.arange(self.cfg.lookahead, device=dev).unsqueeze(0) # (1, L)
        lookahead_idx = (self.target_idx.unsqueeze(-1) + step) % self.cfg.num_waypoints # (N, L)
        # Get the appropriate waypoints and speeds.
        wp_world = self.waypoints[env_ids, lookahead_idx] # (N, L, 2)
        speeds = self.target_speeds[env_ids, lookahead_idx] # (N, L)
        # Convert to local coords
        to_wp_world = wp_world - self.robot.data.root_pos_w[:, :2].unsqueeze(1)
        to_wp_world_3d = torch.cat([to_wp_world, torch.zeros_like(to_wp_world[..., :1])], dim=-1)
        quat = self.robot.data.root_quat_w.unsqueeze(1).expand(-1, self.cfg.lookahead, -1)
        to_wp_local = math_utils.quat_apply_inverse(
            quat.reshape(-1, 4), to_wp_world_3d.reshape(-1, 3)
        ).reshape(self.cfg.scene.num_envs, self.cfg.lookahead, 3)[..., :2]

        return to_wp_local, speeds

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        self.robot.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)

    def _get_observations(self) -> dict:
        self.velocity = self.robot.data.root_com_lin_vel_b
        to_wp_local, speeds = self._lookahead_local_and_speeds()
        waypoint_obs = torch.cat([to_wp_local, speeds.unsqueeze(-1)], dim=-1) # (N, L, 3)
        objective_info = waypoint_obs.reshape(self.cfg.scene.num_envs, -1) # (N, L*3)
        observations = {
            "policy": torch.cat([self.velocity, objective_info], dim=-1)
        }
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # this could be entirely nothing, i'm a bit out of steam doing trigonometry
        velocity_xy = self.robot.data.root_com_lin_vel_b[:, :2]
        to_wp_local, speeds = self._lookahead_local_and_speeds()
        to_target_local = to_wp_local[:, 0]
        target_speed = speeds[:, 0]

        dist = torch.linalg.norm(to_target_local, dim=-1)
        direction = to_target_local / (dist.unsqueeze(-1) + 1e-8)
        target_velocity = target_speed.unsqueeze(-1) * direction

        velocity_error = torch.sum((velocity_xy - target_velocity) ** 2, dim=-1, keepdim=True)
        tracking_reward = torch.exp(-velocity_error / self.cfg.velocity_tracking_std**2)

        arrived = dist < self.cfg.arrival_threshold
        arrival_bonus = arrived.float().unsqueeze(-1) * 10.0 # TODO: what should the sparse reward be?
        self.target_idx = torch.where(arrived, (self.target_idx + 1) % self.cfg.num_waypoints, self.target_idx)

        return tracking_reward + arrival_bonus

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return False, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_state_to_sim(default_root_state, env_ids)
        self._sample_waypoints(env_ids)
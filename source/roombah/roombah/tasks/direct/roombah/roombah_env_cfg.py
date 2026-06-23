# from roombah.robots.jetbot import JETBOT_CONFIG
from roombah.robots.turtlebot import TURTLEBOT4_CFG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

@configclass
class RoombahEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    # - spaces definition
    action_space = 2
    observation_space = 12   # velocity (3) + 3 lookahead waypoints x (local xy + target speed)
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = TURTLEBOT4_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=100, env_spacing=4.0, replicate_physics=True)

    # custom parameters/scales
    dof_names = ["left_wheel_joint", "right_wheel_joint"]

    # params for my experiment
    num_waypoints = 6
    lookahead = 3
    waypoint_radius = 3.0
    arrival_threshold = 0.3
    speed_range = (0.1, 0.5)
    velocity_tracking_std = 0.25
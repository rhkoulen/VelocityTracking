"""Turtlebot robot configuration."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Turtlebot4 (Create3) configuration
TURTLEBOT4_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/iRobot/Create3/create_3.usd",
        activate_contact_sensors=True,
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
            stiffness=0.0,
            damping=1e5,
            velocity_limit=1000.0,
        ),
    },
)

# Turtlebot3 (Burger) configuration
TURTLEBOT3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="source/collaborative_manipulation/collaborative_manipulation/turtle3.usd",
    ),
    actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=10000000.0, stiffness=0.0)},
)
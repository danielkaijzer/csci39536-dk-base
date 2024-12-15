#!/usr/bin/env python3

import rospy
from stable_baselines3 import PPO
from turtlebot_env import TurtleBotEnv

rospy.init_node('turtlebot_rl_testing')

# Initialize environment and load the trained model
env = TurtleBotEnv()
model = PPO.load("ppo_turtlebot")

# Run the trained model
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

#!/usr/bin/env python3

import gym
import rospy
from stable_baselines3 import PPO # Proximal Policy Optimization
from stable_baselines3.common.callbacks import BaseCallback
from turtlebot_env import TurtleBotEnv

# Initialize the ROS node
rospy.init_node('turtlebot_rl_training', anonymous=True)

# Custom callback to display rewards per episode
class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.episode_rewards = 0

    def _on_step(self) -> bool:
        # Add reward at each step
        self.episode_rewards += self.locals['rewards'][0]  # current reward
        if self.locals['dones'][0]:  # end of episode
            print(f"Episode Reward: {self.episode_rewards}")
            self.episode_rewards = 0  # Reset for the next episode
        return True

# Initialize environment
env = TurtleBotEnv()

# Set up the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model with the reward callback
model.learn(total_timesteps=10000, callback=RewardCallback())

# Save the trained model
model.save("ppo_turtlebot")


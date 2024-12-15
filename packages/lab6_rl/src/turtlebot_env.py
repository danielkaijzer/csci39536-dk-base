#!/usr/bin/env python3

import gym
from gym import spaces
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState

class TurtleBotEnv(gym.Env):
    def __init__(self):
        super(TurtleBotEnv, self).__init__()
        
        # Define action and observation space
        self.action_space = spaces.Discrete(4)  # Forward, Backward, Left, Right
        self.observation_space = spaces.Box(low=0, high=10, shape=(360,), dtype=np.float32)
        
        # Initialize lidar data to avoid AttributeError
        self.lidar_data = np.array([10.0] * 360)  # Initial value representing max range
        
        # ROS publishers and subscribers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/scan', LaserScan, self._scan_callback)
        
        # Start a ROS node for this environment
        # COMMENT THIS OUT BEFORE YOU START PART 2
        rospy.init_node('turtlebot_env', anonymous=True)
        
        # Allow time for initialization and first sensor data update
        rospy.sleep(1)  

    def reset_robot_position(self):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            # Define the robot's starting pose
            start_pose = ModelState()
            start_pose.model_name = 'turtlebot3'  # Model name as in Gazebo
            start_pose.pose.position.x = -3.0
            start_pose.pose.position.y = 1.0
            start_pose.pose.position.z = 0.0
            start_pose.pose.orientation.x = 0.0
            start_pose.pose.orientation.y = 0.0
            start_pose.pose.orientation.z = 0.0
            start_pose.pose.orientation.w = 1.0
            set_state(start_pose)
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
        
    def reset(self):
        # Reset the environment and return an initial observation
        self._stop_robot()  # Stop any movement
        self.reset_robot_position()
        rospy.sleep(1)  # Allow time for sensors to update
        return self._get_state()

    def step(self, action):
        # Execute action and receive feedback
        self._take_action(action)
        rospy.sleep(0.1)  # Small delay to allow action to take effect
        state = self._get_state()
        reward, done = self._compute_reward(state)
        return state, reward, done, {}

    def _take_action(self, action):
        # Map action to robot movement
        vel_cmd = Twist()
        if action == 0:  # Forward
            vel_cmd.linear.x = 0.5
        elif action == 1: # Backward
            vel_cmd.linear.x = -0.5
        elif action == 2:  # Left
            vel_cmd.linear.x = 0.01
            vel_cmd.angular.z = 0.5
        elif action == 3:  # Right
            vel_cmd.linear.x = 0.01
            vel_cmd.angular.z = -0.5
        self.cmd_vel_pub.publish(vel_cmd)

    def _get_state(self):
        # Return the current lidar observation
        return np.clip(self.lidar_data, 0, 10)

    def _compute_reward(self, state):
        # Define a reward function
        done = False
        if min(state) < 0.2:  # Collision detected if any distance is below threshold
            reward = -10
            done = True
        else:
            reward = 1
        return reward, done

    def _scan_callback(self, data):
        # Replace any NaN or inf values in lidar data with a max distance value, e.g., 10.0 meters
        self.lidar_data = np.array([10.0 if np.isnan(x) or np.isinf(x) else x for x in data.ranges])

    def _stop_robot(self):
        # Send zero velocity to stop the robot
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)

if __name__ == "__main__":
    env = TurtleBotEnv()
    env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        print(state, reward, done)
        if done:
            env.reset()


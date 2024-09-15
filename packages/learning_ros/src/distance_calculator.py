#!/usr/bin/env python3

import rospy
from turtlesim.msg import Pose
from turtlesim_helper.msg import UnitsLabelled
import math

class DistanceCalculator:
    def __init__(self):
    
        # Initialize the ROS node
        rospy.init_node('distance_calculator')

        # Subscribe to output of turtlesim/sim
        self.pose_subscriber = rospy.Subscriber('/turtlesim/turtle1/pose', Pose, self.pose_callback)

        # Publish message to topic called "distance output"
        self.distance_publisher = rospy.Publisher('/turtle_distance', UnitsLabelled, queue_size=10)

        # Variables to store the last pose
        self.last_x = None
        self.last_y = None

    def pose_callback(self, data):
        # Get the current pose
        current_x = data.x
        current_y = data.y

        # for each message that your node receives on this topic
        # Calculate the distance traveled by turtle
        if self.last_x is not None and self.last_y is not None:
            distance = math.sqrt((current_x - self.last_x)**2 + (current_y - self.last_y)**2)
        else:
            distance = 0.0

        # Update the last position
        self.last_x = current_x
        self.last_y = current_y

        # Create and publish the UnitsLabelled message
        distance_msg = UnitsLabelled()
        distance_msg.value = distance
        distance_msg.units = 'meters'
        self.distance_publisher.publish(distance_msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        distance_calculator = DistanceCalculator()
        distance_calculator.run()
    except rospy.ROSInterruptException:
        pass

#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist

rospy.init_node('turtle_square')

# Publish message to topic "/turtlesim/turtle1/cmd_vel"
pub = rospy.Publisher('/turtlesim/turtle1/cmd_vel', Twist, queue_size=10)

rate = rospy.Rate(1) # 1hz

def move_square():
    twist = Twist()

    # Loop to move turtle in square
    for _ in range(4):
        # Move forward 2 units
        twist.linear.x = 2.0
        twist.angular.z = 0.0
        pub.publish(twist)
        rospy.sleep(2) # turtle moves 2 secs before next command

        # Stop and turn
        twist.linear.x = 0.0
        twist.angular.z = 1.571 # approx 90 degrees
        pub.publish(twist)
        rospy.sleep(2)

if __name__ == '__main__':
    try:
        # slight delay before publishing commands
        rospy.sleep(1)
        while not rospy.is_shutdown():
            move_square()
            rate.sleep()
    except rospy.ROSInterruptException:
        pass

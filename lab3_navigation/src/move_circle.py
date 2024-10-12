#!/usr/bin/env python3
# A simple ROS publisher node in Python to move robot in circle

import rospy 
from geometry_msgs.msg import Twist 

class moveCircle(): 

    def __init__(self): 
        self.node_name = "circle_mover" 
        topic_name = "/cmd_vel" 

        self.pub = rospy.Publisher(topic_name, Twist, queue_size=10) # queue size of 10 usually works!
        rospy.init_node(self.node_name, anonymous=True) 
        self.rate = rospy.Rate(10) 

        self.ctrl_c = False 
        rospy.on_shutdown(self.shutdownhook) 
        
        # create Twist() message instance and assignt it to object called "vel_cmd"
        self.vel_cmd = Twist()
        
        # Max linear velocity
        self.vel_cmd.linear.x = 0.26 # m/s
        
        # since r = 0.5 m = linear_vel/angular_vel => (0.26m/s)/(0.5m) = angular velocity = 0.52 rad/s
        self.vel_cmd.angular.z = 0.52 # rad/s

        rospy.loginfo(f"The '{self.node_name}' node is active...") 

    def shutdownhook(self): 
        print(f"Stopping the '{self.node_name}' node at: {rospy.get_time()}")
        self.ctrl_c = True
        
        self.vel_cmd.linear.x = 0.0
        self.vel_cmd.angular.z = 0.0
        self.pub.publish(self.vel_cmd) # stop message

    def main_loop(self):
        while not self.ctrl_c: 
            self.pub.publish(self.vel_cmd)
            self.rate.sleep()

if __name__ == '__main__': 
    publisher_instance = moveCircle() 
    try:
        publisher_instance.main_loop() 
    except rospy.ROSInterruptException:
        pass

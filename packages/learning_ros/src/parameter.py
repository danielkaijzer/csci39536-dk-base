#!/usr/bin/env python3
import rospy

class ParameterNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('parameter_node')
        
        self.units = ['Meters', 'Feet', 'Smoots']
        
        # Set rate to 1/15 Hz (once every 15 seconds)
        self.rate = rospy.Rate(1/15)
        
    def change_param(self):
        for unit in self.units:
            rospy.set_param('converter', unit)
            rospy.loginfo(f"Setting converter param to: {unit}")
            rospy.sleep(5)  # Sleep for 5 seconds before changing to next unit
        
    def run(self):
        # Set initial parameter to Smoots
        rospy.set_param('converter', 'Smoots')
        rospy.loginfo("Initially setting converter param to: Smoots")
    
        rospy.sleep(5) # delay to fix issue where it starts at Meters
    
        while not rospy.is_shutdown():
            self.change_param()
            
if __name__ == '__main__':
    try:
        parameter = ParameterNode()
        parameter.run()
    except rospy.ROSInterruptException:
        pass

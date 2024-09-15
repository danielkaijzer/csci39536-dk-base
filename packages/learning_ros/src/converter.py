#!/usr/bin/env python3

import rospy
from turtlesim_helper.msg import UnitsLabelled

# this node converts the output of distance_calculator.py (by listening to the topic "turtle_distance")

# conversions
METER_TO_FEET = 3.28084 # 1 meter = 3.28084 feet
METER_TO_SMOOT = 1/1.7018 # 1 meter = 1/1.7018 smoots

class DistanceConverter:
    def __init__(self):
    
        # Initialize the ROS node
        rospy.init_node('distance_converter')
    
        # listen to topic "turtle_distance"
        rospy.Subscriber('/turtle_distance', UnitsLabelled, self.converter_callback)
    
        # publish message to topi called "converted_distance"
        self.pub = rospy.Publisher('/converted_distance', UnitsLabelled, queue_size=10)
    
    def converter_callback(self, msg):    
        # check parameter (decides which unit to convert to)
        unit = rospy.get_param('converter','Smoots').lower() # Default is Smoots
        
        new_distance_msg = UnitsLabelled()
        new_distance_msg.value = msg.value
        new_distance_msg.units = unit.capitalize()

        # convert!
        # based on parameter you will change the value portion of the UnitsLabelled msg
        if unit == 'meters':
            pass
        elif unit == 'feet':
            new_distance_msg.value *= METER_TO_FEET
        elif unit == 'smoots':
            new_distance_msg.value *= METER_TO_SMOOT
        else: # unknown unit
            rospy.logwarn(f"Invalid conversion unit: {unit}")
            new_distance_msg.units = 'meters'
    
        # publish conversion to new topic e.g., converted distance
        self.pub.publish(new_distance_msg)
        
    def run(self):
        rospy.spin()
        
if __name__ == '__main__':
    try:
        distance_converter = DistanceConverter()
        distance_converter.run()
    except rospy.ROSInterruptException:
        pass


#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

class RobotFollower:
    def __init__(self):
        rospy.init_node('robot_follower', anonymous=True)
        self.rate = rospy.Rate(10)
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Subscribe to the camera feed
        self.image_sub = rospy.Subscriber('/tb2/camera/rgb/image_raw', Image, self.camera_callback)
        
        # Publisher for robot movement
        self.cmd_vel_pub = rospy.Publisher('/tb2/cmd_vel', Twist, queue_size=1)
        
        # Initialize movement message
        self.move_cmd = Twist()
        
        # Target color (red) in HSV
        # Using two ranges for red color as it wraps around in HSV
        self.lower_red1 = np.array([0, 100, 100])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([160, 100, 100])
        self.upper_red2 = np.array([180, 255, 255])
        
        self.target_found = False
        self.min_area = 500  # Minimum area to consider as valid detection
        
        # PID control parameters - increased for more responsive movement
        self.p_gain = 0.005  # Increased from 0.002
        self.d_gain = 0.002  # Increased from 0.001
        self.last_error = 0
        
        # Create and position window
        cv2.namedWindow("Robot Follower", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Robot Follower", 640, 480)
        
        rospy.loginfo("Robot Follower initialized")

    def camera_callback(self, data):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return
            
        # Resize image for display
        display_image = cv2.resize(cv_image.copy(), (640, 480))
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Create masks for red color (combining both ranges)
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Apply some noise reduction
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Reset movement command
        self.move_cmd = Twist()
        
        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area > self.min_area:
                self.target_found = True
                
                # Get the centroid of the largest contour
                M = cv2.moments(largest_contour)
                if M['m00'] > 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    
                    # Draw circle at centroid and bounding box
                    cv2.circle(display_image, (cx, cy), 5, (0, 255, 0), -1)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    cv2.rectangle(display_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Calculate error from center of image
                    error_x = cx - cv_image.shape[1]/2
                    
                    # PID control for smoother movement
                    error_diff = error_x - self.last_error
                    self.last_error = error_x
                    
                    # Set angular velocity using PID
                    self.move_cmd.angular.z = -(self.p_gain * error_x + self.d_gain * error_diff)
                    
                    # Set forward velocity based on area (distance)
                    target_area = 5000  # Decreased from 8000 to maintain closer distance
                    area_error = target_area - area
                    self.move_cmd.linear.x = min(max(area_error * 0.0002, 0.0), 0.5)  # Increased multiplier and max speed
                    
                    # Debug information
                    rospy.loginfo(f"Linear velocity: {self.move_cmd.linear.x:.2f}, Angular velocity: {self.move_cmd.angular.z:.2f}")
                    rospy.loginfo(f"Area: {area:.0f}, Error X: {error_x:.0f}")
                    
                    # Display distance estimation
                    cv2.putText(display_image, f"Area: {int(area)}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(display_image, f"Vel: {self.move_cmd.linear.x:.2f}", (10, 90),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                self.target_found = False
                rospy.loginfo("Target too small")
        else:
            # If target not found, spin to search
            self.target_found = False
            self.move_cmd.angular.z = 0.5
            self.last_error = 0
            rospy.loginfo("No target found, searching...")
        
        # Add status text to display
        status = "Target Found" if self.target_found else "Searching"
        cv2.putText(display_image, status, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Publish movement command
        self.cmd_vel_pub.publish(self.move_cmd)
        
        # Display the image
        cv2.imshow("Robot Follower", display_image)
        cv2.waitKey(1)

    def run(self):
        try:
            rospy.loginfo("Robot Follower Node Started")
            while not rospy.is_shutdown():
                self.rate.sleep()
        except rospy.ROSInterruptException:
            pass
        finally:
            # Stop the robot before shutting down
            self.move_cmd = Twist()
            self.cmd_vel_pub.publish(self.move_cmd)
            cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        follower = RobotFollower()
        follower.run()
    except rospy.ROSException as e:
        rospy.logerr(str(e))

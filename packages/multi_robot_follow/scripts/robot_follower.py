#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
import math

class RobotFollower:
    def __init__(self):
        rospy.init_node('robot_follower', anonymous=True)
        self.rate = rospy.Rate(10)
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Subscribe to sensors
        self.image_sub = rospy.Subscriber('/tb2/camera/rgb/image_raw', Image, self.camera_callback)
        self.lidar_sub = rospy.Subscriber('/tb2/scan', LaserScan, self.lidar_callback)
        self.odom_sub = rospy.Subscriber('/tb2/odom', Odometry, self.odom_callback)
        
        # Publisher for robot movement
        self.cmd_vel_pub = rospy.Publisher('/tb2/cmd_vel', Twist, queue_size=1)
        
        # Initialize movement message
        self.move_cmd = Twist()
        
        # Target color (red) in HSV
        self.lower_red1 = np.array([0, 100, 100])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([160, 100, 100])
        self.upper_red2 = np.array([180, 255, 255])
        
        # Vision tracking parameters
        self.target_found = False
        self.min_area = 500
        self.last_target_position = None  # Store last known target position
        self.frames_without_target = 0
        self.max_frames_without_target = 30  # Number of frames before considering target lost
        
        # Robot state
        self.current_yaw = 0.0
        self.obstacle_detected = False
        self.last_target_angle = None
        
        # PID control parameters
        self.p_gain = 0.005
        self.d_gain = 0.002
        self.last_error = 0
        
        # Obstacle avoidance parameters
        self.min_obstacle_distance = 0.4  # meters
        self.obstacle_scan_angle = 90  # degrees total scan angle
        self.lidar_data = None
        
        # Create window for visualization
        cv2.namedWindow("Robot Follower", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Robot Follower", 640, 480)
        
        rospy.loginfo("Robot Follower initialized")
    
    def odom_callback(self, msg):
        """Update robot's current orientation from odometry data"""
        orientation = msg.pose.pose.orientation
        _, _, self.current_yaw = euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w])
    
    def lidar_callback(self, msg):
        """Process LiDAR data for obstacle detection"""
        self.lidar_data = msg.ranges
        
        # Calculate the indices for the front sector of the robot
        angle_increment = msg.angle_increment
        front_angles = int(self.obstacle_scan_angle * (math.pi/180) / angle_increment)
        mid_idx = len(msg.ranges) // 2
        start_idx = mid_idx - front_angles // 2
        end_idx = mid_idx + front_angles // 2
        
        # Check for obstacles in the front sector
        front_distances = msg.ranges[start_idx:end_idx]
        valid_distances = [d for d in front_distances if not math.isinf(d) and not math.isnan(d)]
        
        if valid_distances and min(valid_distances) < self.min_obstacle_distance:
            self.obstacle_detected = True
            # Find the direction with more space
            left_distances = msg.ranges[start_idx:mid_idx]
            right_distances = msg.ranges[mid_idx:end_idx]
            left_avg = np.mean([d for d in left_distances if not math.isinf(d) and not math.isnan(d)])
            right_avg = np.mean([d for d in right_distances if not math.isinf(d) and not math.isnan(d)])
            self.avoidance_direction = 1 if left_avg > right_avg else -1
        else:
            self.obstacle_detected = False

    def find_safe_direction(self):
        """Find the safest direction to move based on LiDAR data"""
        if not self.lidar_data:
            return 0
            
        # Create sectors for directional analysis
        sectors = np.array_split(self.lidar_data, 8)  # Split into 8 sectors
        sector_averages = []
        
        for sector in sectors:
            valid_readings = [r for r in sector if not math.isinf(r) and not math.isnan(r)]
            if valid_readings:
                sector_averages.append(np.mean(valid_readings))
            else:
                sector_averages.append(0)
        
        # Return the index of the sector with the most space
        return (np.argmax(sector_averages) - 4) / 4.0  # Normalize to [-1, 1]

    def camera_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return
            
        # Process image for target detection
        display_image = cv2.resize(cv_image.copy(), (640, 480))
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Create masks for red color
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Apply noise reduction
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Reset movement command
        self.move_cmd = Twist()
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area > self.min_area:
                self.target_found = True
                self.frames_without_target = 0
                
                # Calculate centroid
                M = cv2.moments(largest_contour)
                if M['m00'] > 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    self.last_target_position = (cx, cy)
                    
                    # Calculate target angle relative to image center
                    image_center = cv_image.shape[1]/2
                    self.last_target_angle = math.atan2(cx - image_center, cv_image.shape[1]/2)
                    
                    # Draw visualization
                    cv2.circle(display_image, (cx, cy), 5, (0, 255, 0), -1)
                    # cv2.rectangle(display_image, *cv2.boundingRect(largest_contour)[:4], (0, 255, 0), 2)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Calculate control inputs
                    if not self.obstacle_detected:
                        # Normal following behavior
                        error_x = cx - image_center
                        error_diff = error_x - self.last_error
                        self.last_error = error_x
                        
                        # Set velocities
                        self.move_cmd.angular.z = -(self.p_gain * error_x + self.d_gain * error_diff)
                        target_area = 5000
                        area_error = target_area - area
                        self.move_cmd.linear.x = min(max(area_error * 0.0002, 0.0), 0.5)
                    else:
                        # Obstacle avoidance mode
                        safe_direction = self.find_safe_direction()
                        self.move_cmd.angular.z = safe_direction
                        self.move_cmd.linear.x = 0.05  # Slow movement while avoiding
                        
                        # Back if too close to obstacle
                        if any(d < 0.3 for d in self.lidar_data if not math.isinf(d) and not math.isnan(d)):
                            self.move_cmd.linear.x = -0.05
                            rospy.loginfo("Too close to obstacle - backing up")
            else:
                self.handle_target_loss(display_image)
        else:
            self.handle_target_loss(display_image)
        
        # Display status and metrics
        self.display_status(display_image)
        
        # Publish movement command
        self.cmd_vel_pub.publish(self.move_cmd)
        
        # Update display
        cv2.imshow("Robot Follower", display_image)
        cv2.waitKey(1)

    def handle_target_loss(self, display_image):
        """Handle behavior when target is temporarily lost"""
        self.frames_without_target += 1
        if self.frames_without_target > self.max_frames_without_target:
            self.target_found = False
            if not self.obstacle_detected:
                # Search pattern - rotate towards last known position
                if self.last_target_angle is not None:
                    self.move_cmd.angular.z = 0.2 * np.sign(self.last_target_angle)
                else:
                    self.move_cmd.angular.z = 0.4  # Default search rotation
            self.last_error = 0
            cv2.putText(display_image, "Searching for target...", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    def display_status(self, image):
        """Display robot status and metrics on the image"""
        status = "Target Found" if self.target_found else "Searching"
        cv2.putText(image, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Linear Vel: {self.move_cmd.linear.x:.2f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Angular Vel: {self.move_cmd.angular.z:.2f}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def run(self):
        try:
            rospy.loginfo("Robot Follower Node Started")
            while not rospy.is_shutdown():
                self.rate.sleep()
        except rospy.ROSInterruptException:
            pass
        finally:
            # Clean shutdown
            self.move_cmd = Twist()
            self.cmd_vel_pub.publish(self.move_cmd)
            cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        follower = RobotFollower()
        follower.run()
    except rospy.ROSException as e:
        rospy.logerr(str(e))

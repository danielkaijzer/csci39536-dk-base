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
        self.min_area = 150
        self.last_target_position = None
        self.last_target_world_pos = None  # Store target's estimated world position
        self.frames_without_target = 0
        self.max_frames_without_target = 30
        
        # Robot state
        self.current_yaw = 0.0
        self.obstacle_detected = False
        self.last_target_angle = None
        self.search_start_time = None
        self.search_pattern_counter = 0
        
        # PID control parameters
        self.p_gain = 0.005
        self.d_gain = 0.002
        self.last_error = 0
        
        # Improved obstacle avoidance parameters
        self.min_obstacle_distance = 0.4  # meters
        self.obstacle_scan_angle = 180  # increased from 90 to 180 degrees
        self.lidar_data = None
        self.sector_data = {'front': None, 'left': None, 'right': None}
        
        # Create window for visualization
        cv2.namedWindow("Robot Follower", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Robot Follower", 640, 480)
        
        rospy.loginfo("Robot Follower initialized")
    
    def odom_callback(self, msg):
        """Update robot's current orientation from odometry data"""
        orientation = msg.pose.pose.orientation
        _, _, self.current_yaw = euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w])
    
    def check_sectors(self, front_sector, left_sector, right_sector):
        """Analyze different sectors for obstacles"""
        def process_sector(sector):
            valid = [d for d in sector if not math.isinf(d) and not math.isnan(d)]
            return min(valid) if valid else float('inf')
        
        self.sector_data = {
            'front': process_sector(front_sector),
            'left': process_sector(left_sector),
            'right': process_sector(right_sector)
        }
        
        # Check if any sector has an obstacle
        self.obstacle_detected = any(dist < self.min_obstacle_distance 
                                   for dist in self.sector_data.values() 
                                   if dist != float('inf'))
    
    def lidar_callback(self, msg):
        """Enhanced LiDAR processing with multiple sectors"""
        self.lidar_data = msg.ranges
        
        # Calculate indices for wider front sector (180 degrees)
        angle_increment = msg.angle_increment
        front_angles = int(self.obstacle_scan_angle * (math.pi/180) / angle_increment)
        mid_idx = len(msg.ranges) // 2
        start_idx = mid_idx - front_angles // 2
        end_idx = mid_idx + front_angles // 2
        
        # Define sectors
        front_sector = msg.ranges[start_idx:end_idx]
        left_sector = msg.ranges[:start_idx]
        right_sector = msg.ranges[end_idx:]
        
        # Process all sectors
        self.check_sectors(front_sector, left_sector, right_sector)

    def find_safe_direction(self):
        """Enhanced safe direction finding with target consideration"""
        if not self.lidar_data:
            return 0
        
        # Split into 8 sectors for better granularity
        sectors = np.array_split(self.lidar_data, 8)
        sector_scores = []
        
        # Calculate target weights if target is known
        target_weight = [1.0] * 8  # Default equal weights
        if self.last_target_angle is not None:
            target_sector = int((self.last_target_angle + math.pi) / (2 * math.pi / 8))
            target_weight = [math.exp(-0.5 * abs(i - target_sector)) for i in range(8)]
        
        for i, sector in enumerate(sectors):
            valid_readings = [r for r in sector if not math.isinf(r) and not math.isnan(r)]
            if valid_readings:
                # Combine distance and target direction
                mean_distance = np.mean(valid_readings)
                safety_score = mean_distance * target_weight[i]
                sector_scores.append(safety_score)
            else:
                sector_scores.append(0)
        
        # Convert sector index to normalized direction
        best_sector = np.argmax(sector_scores)
        return (best_sector - 4) / 4.0

    def generate_search_pattern(self):
        """Generate an expanding spiral search pattern"""
        if self.search_start_time is None:
            self.search_start_time = rospy.Time.now()
        
        search_duration = (rospy.Time.now() - self.search_start_time).to_sec()
        self.search_pattern_counter += 0.1
        
        # Create spiral pattern
        radius = min(0.1 * self.search_pattern_counter, 2.0)
        angular_velocity = 0.3 * (1 - math.exp(-radius))
        linear_velocity = 0.05 * math.sin(radius)
        
        # Reset search if it's taking too long
        if search_duration > 30.0:  # Reset after 30 seconds
            self.search_start_time = None
            self.search_pattern_counter = 0
        
        return linear_velocity, angular_velocity

    def handle_target_loss(self, display_image):
        """Improved target loss handling with spiral search"""
        
        self.frames_without_target += 1
        
        if self.frames_without_target > self.max_frames_without_target:
            self.target_found = False
            
            if self.obstacle_detected:
                # Even with obstacle, keep moving but carefully
                safe_direction = self.find_safe_direction()
                self.move_cmd.angular.z = safe_direction
                # Very slow forward movement while searching with obstacle
                if self.sector_data['front'] > self.min_obstacle_distance:
                    self.move_cmd.linear.x = 0.05
            else:
                # Normal search behavior
                linear_vel, angular_vel = self.generate_search_pattern()
                self.move_cmd.linear.x = linear_vel
                self.move_cmd.angular.z = angular_vel
                
            self.last_error = 0
            cv2.putText(display_image, "Searching for target...", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    def calculate_control(self, error_x, area):
        """Calculate control signals with smooth transitions"""
        
    def calculate_control(self, error_x, area):
        """Calculate control signals with better obstacle handling"""
        if self.obstacle_detected:
            safe_direction = self.find_safe_direction()
            base_angular = safe_direction
        
            # Check if we can move forward
            closest_front = self.sector_data['front']
            if closest_front < 0.3:
                base_linear = -0.05  # Back up if too close
                rospy.loginfo("Too close to obstacle - backing up")
            elif closest_front < self.min_obstacle_distance:
                base_linear = 0  # Stop if close
            else:
                base_linear = 0.05  # Slow forward movement if path is clear
        else:
            # Normal following behavior
            error_diff = error_x - self.last_error
            self.last_error = error_x
            base_angular = -(self.p_gain * error_x + self.d_gain * error_diff)
        
            # Distance-based speed control
            target_area = 5000
            area_error = abs(target_area - area)
            base_linear = min(max(area_error * 0.0002, 0.0), 0.5)
    
        return base_linear, base_angular

    def camera_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return
        
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
                self.search_start_time = None
                self.search_pattern_counter = 0
                
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
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Calculate and apply control
                    error_x = cx - image_center
                    linear_vel, angular_vel = self.calculate_control(error_x, area)
                    self.move_cmd.linear.x = linear_vel
                    self.move_cmd.angular.z = angular_vel
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

    def display_status(self, image):
        """Enhanced status display"""
        status = "Target Found" if self.target_found else "Searching"
        cv2.putText(image, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Linear Vel: {self.move_cmd.linear.x:.2f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Angular Vel: {self.move_cmd.angular.z:.2f}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if self.obstacle_detected:
            cv2.putText(image, "Obstacle Detected!", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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

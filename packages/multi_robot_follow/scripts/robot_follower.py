#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
import math

class RobotFollower:
	def __init__(self):
		
		# Initialize the ROS node
		rospy.init_node('robot_follower', anonymous=True)
		self.rate = rospy.Rate(10) # 10 Hz update rate

		# Initialize OpenCV bridge
		self.bridge = CvBridge()

		# Subscribe to sensors
		self.image_sub = rospy.Subscriber('/tb2/camera/rgb/image_raw', Image, self.camera_callback)
		self.lidar_sub = rospy.Subscriber('/tb2/scan', LaserScan, self.lidar_callback)

		# Publisher for robot movement commands
		self.cmd_vel_pub = rospy.Publisher('/tb2/cmd_vel', Twist, queue_size=1)

		# Initialize movement command message
		self.move_cmd = Twist()

		# Target color (red) in HSV
		self.lower_red1 = np.array([0, 100, 100])
		self.upper_red1 = np.array([10, 255, 255])
		self.lower_red2 = np.array([160, 100, 100])
		self.upper_red2 = np.array([180, 255, 255])

		# Vision tracking parameters
		self.target_found = False
		self.min_area = 300 # Can be smaller
		self.last_target_position = None  # Store last known target position
		self.frames_without_target = 0
		self.max_frames_without_target = 30  # Number of frames before considering target lost

		# Robot state
		self.obstacle_detected = False

		# PID control parameters
		self.p_gain = 0.005
		self.d_gain = 0.002
		self.last_error = 0 # Used later for determining search direction

		# Obstacle avoidance parameters
		self.min_obstacle_distance = 0.5  # Minimum acceptable distance to an obstacle in meters
		self.obstacle_scan_angle = 120  # Total angle (in degrees) to scan for obstacles
		self.lidar_data = None

		# Create window for visualization
		cv2.namedWindow("Robot Follower", cv2.WINDOW_NORMAL)
		cv2.resizeWindow("Robot Follower", 640, 480)

		rospy.loginfo("Robot Follower initialized")
    
    
	def lidar_callback(self, msg):
		"""
		Process LiDAR data for obstacle detection
		"""
		self.lidar_data = msg.ranges

		# Calculate the indices for the front sector of the robot
		angle_increment = msg.angle_increment
		front_angles = int(self.obstacle_scan_angle * (math.pi/180) / angle_increment)
		mid_idx = len(msg.ranges) // 2
		start_idx = mid_idx - front_angles // 2
		end_idx = mid_idx + front_angles // 2

		# Check for obstacles in the front sector
		front_distances = msg.ranges[start_idx:end_idx]
		obstacle_distances = [d for d in front_distances if not math.isinf(d) and not math.isnan(d)]
		
		if not obstacle_distances:
			self.obstacle_detected = False
			return

		# print("obs distances ", obstacle_distances)
		# print("minimum: ", min(obstacle_distances))
		if obstacle_distances and min(obstacle_distances) < self.min_obstacle_distance:
			self.obstacle_detected = True
			print("Obstacle detected ", self.obstacle_detected)
			
			# Find the direction with more space
			left_distances = msg.ranges[start_idx:mid_idx]
			right_distances = msg.ranges[mid_idx:end_idx]
			left_avg = np.mean([d for d in left_distances if not math.isinf(d) and not math.isnan(d)])
			right_avg = np.mean([d for d in right_distances if not math.isinf(d) and not math.isnan(d)])
			self.avoidance_direction = 1 if left_avg > right_avg else -1
		else:
			self.obstacle_detected = False
			

	def find_safe_direction(self):
		"""
		Find the safest direction to move based on the LiDAR data.

		This function splits the LiDAR sensor data into 8 equal sectors and calculates the average
		distance for each sector. It then determines the sector with the largest average distance,
		which is considered the safest direction to move in.

		The function returns a normalized value between -1 and 1, where -1 represents turning left,
		0 represents moving straight, and 1 represents turning right. This value can be used to
		adjust the robot's angular velocity to steer it towards the safest available direction.

		Returns:
		float: A normalized value between -1 and 1 representing the safest direction to move.
		"""
		if not self.lidar_data:
			# If no LiDAR data is available, return 0 to move straight
			return 0

		# Split the LiDAR data into 8 equal sectors
		sectors = np.array_split(self.lidar_data, 8)

		# Calculate the average distance for each sector, excluding any infinite or NaN values
		sector_averages = []
		for sector in sectors:
			valid_readings = [r for r in sector if not math.isinf(r) and not math.isnan(r)]
			if valid_readings:
				sector_averages.append(np.mean(valid_readings))
			else:
				# If a sector has no valid readings, assume it has a large obstacle and set the average to 0
				sector_averages.append(0)

		# Determine the index of the sector with the largest average distance
		# This is considered the safest direction to move in
		safest_sector_idx = np.argmax(sector_averages)

		# Normalize the index to a value between -1 and 1
		# -1 represents turning left, 0 represents moving straight, and 1 represents turning right
		return (safest_sector_idx - 4) / 4.0


	def camera_callback(self, data):
		"""
		Callback function for processing camera images and controlling the robot's movement.
		
		This function is called whenever a new camera image is received. It processes the image to
        detect the target object, calculates the control inputs based on the target's position, and
        publishes the movement commands to the robot.

        If the target is detected, the function calculates the error between the target's position and
        the center of the image, and uses a PID controller to set the robot's angular velocity. It also
        adjusts the linear velocity based on the size of the detected target.

        If an obstacle is detected, the function switches to obstacle avoidance mode, using the
        `find_safe_direction` method to determine the safest direction to move.

        If the target is lost, the function calls the `search_behavior` method to search for the target.
		
		Args:
            data (sensor_msgs.msg.Image): The incoming camera image.
		"""


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

			if area > self.min_area: # If the target is in frame
				self.target_found = True
				self.frames_without_target = 0

				# Calculate centroid
				M = cv2.moments(largest_contour)
				if M['m00'] > 0:
					cx = int(M['m10']/M['m00'])
					cy = int(M['m01']/M['m00'])
					self.last_target_position = (cx, cy)

					# Calculate control inputs
					if not self.obstacle_detected:
						# Normal following behavior
						image_center = cv_image.shape[1]/2
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

						# More aggressive obstacle avoidance
						min_distance = min([d for d in self.lidar_data if not math.isinf(d) and not math.isnan(d)])
						if min_distance < 0.3:
							self.move_cmd.linear.x = -0.15  # Back up more aggressively
							rospy.loginfo("Too close to obstacle - backing up")
						else:
							self.move_cmd.linear.x = 0.05  # Slow movement while avoiding
			else:
				self.search_behavior(display_image)
		else:
			self.search_behavior(display_image)
        
		# Display status and metrics
		self.display_status(display_image)

		# Publish movement command
		self.cmd_vel_pub.publish(self.move_cmd)

		# Update display
		cv2.imshow("Robot Follower", display_image)
		cv2.waitKey(1)


	def search_behavior(self, display_image):
		"""
		Implement the robot's search behavior when the target is lost or hasn't been seen.
		
		If an obstacle is detected, the robot first backs up to create some distance from the obstacle,
        and then uses the last known target direction to search for the target. If no obstacle is
        detected, the robot performs a pure rotational search, turning left or right based on the
        last known target direction.
		
		Args:
            display_image (numpy.ndarray): The current camera image to be displayed.
		"""
	
		self.frames_without_target += 1
		
		# if target is considered "lost"
		if self.frames_without_target > self.max_frames_without_target:
		
			self.target_found = False
			
			# If target lost and obstacle, back up!
			if self.obstacle_detected:
				# First back up from obstacle
				
				self.move_cmd.linear.x = -0.15  # Back up
				self.move_cmd.angular.z = 0.0
				
				# If we've backed up enough (using frames as a simple timer)
				if self.frames_without_target > self.max_frames_without_target + 15:  # Extra 15 frames for backing up
					self.move_cmd.linear.x = 0.0  # Stop backing up
					
					# Use the last known target direction for search
					if self.last_error is not None:
						search_direction = 0.3 if self.last_error > 0 else -0.3
						self.move_cmd.angular.z = search_direction
					else:
						self.move_cmd.angular.z = -0.4
			else:
				# Pure rotation search when no obstacles
				if self.last_error is not None:
					search_direction = 0.3 if self.last_error > 0 else -0.3
					self.move_cmd.angular.z = search_direction
				else:
					self.move_cmd.angular.z = -0.4
					
				self.move_cmd.linear.x = 0.0  # MAYBE ADD SUBTLE MOVEMENT
				
				
			cv2.putText(display_image, "Searching for target...", (10, 150),
				   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
		
		

	def display_status(self, image):
		"""Display robot status and metrics on the image"""
		status = "Target Found" if self.target_found else "Searching"
		cv2.putText(image, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
		cv2.putText(image, f"Linear Vel: {self.move_cmd.linear.x:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
		cv2.putText(image, f"Angular Vel: {self.move_cmd.angular.z:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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

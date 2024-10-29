#!/usr/bin/env python3

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

from tb3 import Tb3Move

class LineFollower(object): 
    def __init__(self):
        node_name = "line_follower"
        rospy.init_node(node_name, anonymous=True)
        self.rate = rospy.Rate(5)

        self.cvbridge_interface = CvBridge()
        self.img_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.camera_cb)
        self.robot_controller = Tb3Move()

        self.ctrl_c = False
        rospy.on_shutdown(self.shutdown_ops)
        
        self.last_y_error = 0.0 # track last y_error to turn appropriately when line gets lost

    def shutdown_ops(self):
        self.robot_controller.stop()
        cv2.destroyAllWindows()
        self.ctrl_c = True

    def camera_cb(self, img_data):
        try:
            cv_img = self.cvbridge_interface.imgmsg_to_cv2(img_data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)

        height, width, _ = cv_img.shape 
        crop_width = 50
        crop_height = 250
        crop_x = int((width / 2) - (crop_width / 2))
        crop_y = height - crop_height
        cropped_img = cv_img[crop_y:height, 10:crop_x+crop_width]

        hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

        lower_line = (125, 150, 100)
        upper_line = (165, 255, 255)
        
        lower_finish = (175,200,100)
        upper_finish = (255,255,255)
        
        mask = cv2.inRange(hsv_img, lower_line, upper_line) 
        finish_mask = cv2.inRange(hsv_img, lower_finish, upper_finish)
        
        finish_moments = cv2.moments(finish_mask)
        if finish_moments['m00'] > 500:  # Threshold for finish line detection
            print("Finish line detected.")
            rospy.sleep(5.0) # adding a delay so I can actually end on the finish line.
            self.robot_controller.stop()
            return

        m = cv2.moments(mask)
        
        
        combined_mask = cv2.bitwise_or(mask, finish_mask)
        res = cv2.bitwise_and(cropped_img, cropped_img, mask = combined_mask)
        
        if m['m00'] > 0:
            # Line is in view, calculate centroid and move forward
            cy = m['m10'] / m['m00']
            cz = m['m01'] / m['m00']

            # Visualize the centroid
            cv2.circle(res, (int(cy), crop_height // 2), 10, (255, 0, 0), 2)
            cv2.imshow("filtered image", res)
            cv2.waitKey(1)

            y_error = cy - (crop_width / 2)
            self.last_y_error = y_error
            

            #if abs(y_error) <= 100:
                #ang_vel = 0.0  # No turning if error small
                #fwd_vel = 0.3
            #else:
            
            fwd_vel = 0.5
            if abs(y_error) > 100:
                fwd_vel = 0.1
                
            kp = 1.0 / 100
            ang_vel = kp * y_error
            max_ang_vel = 0.05
            min_ang_vel = -0.05
            ang_vel = max(min(ang_vel, max_ang_vel), min_ang_vel)

            print(f"Line detected. Y-error = {y_error:.3f} pixels, ang_vel = {ang_vel:.3f} rad/s")
            self.robot_controller.set_move_cmd(fwd_vel, ang_vel)

        else:
            # Line not in view - stop forward motion and turn to search
            print("Line lost. Stop and turn to search")
            
            if self.last_y_error > 0: # turn right
                self.robot_controller.set_move_cmd(0.0, -0.4)
            else:
                self.robot_controller.set_move_cmd(0.0, 0.4) # turn left

        self.robot_controller.publish()

    def main(self):
        while not self.ctrl_c:
            self.rate.sleep()

if __name__ == '__main__':
    lf_instance = LineFollower()
    try:
        lf_instance.main()
    except rospy.ROSInterruptException:
        pass


#!/usr/bin/env python3

## Ros run command "ros2 run my_ros2_node node"

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import JointState
import math
from PyKDL import Rotation
from surgical_robotics_challenge.units_conversion import *

SimToSI = 0.05

class RobotData:
    def __init__(self):
        self.measured_js = JointState()
        self.measured_cp = PoseStamped()

class SurChalCrtkTestNode(Node):
    def __init__(self):
        super().__init__('sur_chal_crtk_test')
        self.robData = RobotData()

        # Setup topic names (adjust as needed)
        arm_name   = "PSM1"
        measured_js_topic =  arm_name + "/measured_js"
        measured_cp_topic = arm_name + "/measured_cp"
        self.servo_jp_topic = arm_name + "/servo_jp"
        self.servo_cp_topic = arm_name + "/servo_cp"

        # Create subscriptions
        self.create_subscription(JointState, measured_js_topic, self.measured_js_cb, 10)
        self.create_subscription(PoseStamped, measured_cp_topic, self.measured_cp_cb, 10)

        # Create publishers
        self.servo_jp_pub = self.create_publisher(JointState, self.servo_jp_topic, 10)
        self.servo_cp_pub = self.create_publisher(PoseStamped, self.servo_cp_topic, 10)

        # Initialize messages to publish
        self.servo_jp_msg = JointState()
        self.servo_jp_msg.position = [0., 0., 0.1 * SimToSI, 0., 0., 0.]

        self.servo_cp_msg = PoseStamped()
        self.servo_cp_msg.pose.position.x = 0.0 * SimToSI
        self.servo_cp_msg.pose.position.y = 0.0 * SimToSI
        self.servo_cp_msg.pose.position.z = -0.1 * SimToSI
        R_7_0 = Rotation.RPY(-1.57079, 0.0, 1.57079)
        quat = R_7_0.GetQuaternion()
        self.servo_cp_msg.pose.orientation.x = quat[0]
        self.servo_cp_msg.pose.orientation.y = quat[1]
        self.servo_cp_msg.pose.orientation.z = quat[2]
        self.servo_cp_msg.pose.orientation.w = quat[3]

        # Ask for selection mode
        valid_key = False
        while not valid_key:
            print("NOTE!!! For this example to work, please RUN the launch_crtk_interface.py script beforehand.")
            try:
                key = int(input("Press: \n"
                                "1 - (For reading joint and Cartesian state), \n"
                                "2 - (For joint control demo), \n"
                                "3 - (For Cartesian control demo) \n"))
            except ValueError:
                key = None
            if key in [1, 2, 3]:
                valid_key = True
                self.mode = key
            else:
                print("Invalid Entry")

        # Create a timer at 50 Hz
        timer_period = 0.02  
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def measured_js_cb(self, msg):
        self.robData.measured_js = msg

    def measured_cp_cb(self, msg):
        self.robData.measured_cp = msg

    def timer_callback(self):
        # Get current time in seconds
        current_time = self.get_clock().now().nanoseconds / 1e9
        if self.mode == 1:
            print("measured_js: ", self.robData.measured_js)
            print("------------------------------------")
            print("measured_cp: ", self.robData.measured_cp.pose)
        elif self.mode == 2:
            # Update joint positions in a sinusoidal pattern
            self.servo_jp_msg.position[0] = 0.2 * math.sin(current_time)
            self.servo_jp_msg.position[1] = 0.2 * math.cos(current_time)
            self.servo_jp_pub.publish(self.servo_jp_msg)
        elif self.mode == 3:
            # Update Cartesian target in a sinusoidal pattern
            self.servo_cp_msg.pose.position.x = 0.02 * SimToSI * math.sin(current_time)
            self.servo_cp_msg.pose.position.y = 0.02 * SimToSI * math.cos(current_time)
            self.servo_cp_pub.publish(self.servo_cp_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SurChalCrtkTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
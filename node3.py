#!/usr/bin/env python3
"""
Simplified ROS 2 blood suction controller with 4 algorithms:
 1. Global zigzag: sweep full FOV
 2. Adaptive zigzag: sweep bounding box around blood blob
 3. Centroid follower: move to blob centre
 4. Inward spiral: circle around centre, decreasing radius

All control is blocking step-by-step.
Now uses PyKDL for transforms (transform "R_7_0").
"""

import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from sensor_msgs.msg import Image, TimeReference
from geometry_msgs.msg import PoseStamped, PointStamped

import PyKDL as kdl

import threading
import cv2    

import csv
    
import os
from datetime import datetime

from util.util.AntColonyOptimisation import AntColony, pre_process_data
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion


try:
    from cv_bridge import CvBridge
    _HAS_CV = True
except Exception:
    _HAS_CV = False


def rpy_to_quat(r, p, y):
    rot = kdl.Rotation.RPY(r, p, y)
    x, y, z, w = rot.GetQuaternion()
    return (x, y, z, w)


class SimpleController(Node):
    def __init__(self):
        super().__init__('simple_blood_controller')
        self.arm = 'PSM1'
        self.fov_x = 4.4
        self.fov_y = 4.4
        self.z_depth = -1.1
        self.d_zig = 0.4
        self.max_step = 0.2
        self.tol_xy = 0.1
        self.control_rate_hz = 50

        self.spir_rate_deg = 12.0
        self.spiral_dr = 0.0002

        # self.algo = self._menu_select()

        qos = QoSProfile(depth=10)
        self.pose_sub = self.create_subscription(PoseStamped, f"/{self.arm}/measured_cp", self._pose_cb, qos)
        self.mask_sub = self.create_subscription(Image, '/camera/blood_mask', self._mask_cb, qos)
        self.center_sub = self.create_subscription(PointStamped, '/camera/blood_blob_center', self._center_cb, qos)
        self.servo_pub = self.create_publisher(PoseStamped, f"/{self.arm}/servo_cp", 1)
        self.total_particles_sub = self.create_subscription(TimeReference, 'numparticles', self._total_particles_cb, qos)
        self.particles_removed_sub = self.create_subscription(TimeReference, 'numparticles/removed', self._particles_removed_cb, qos)

        self.bridge = CvBridge() if _HAS_CV else None

        self.latest_pose = None
        self.latest_center = None
        self.latest_mask = None

        self.total_particles = 0
        self.particles_removed = 0
        self.particles_remaining = 0

        qx, qy, qz, qw = rpy_to_quat(0.0, math.pi/2, -math.pi/2)
        self.orientation = (qx, qy, qz, qw)

        self._spir_phase = 0.0
        self._spir_radius = 0.5*min(self.fov_x, self.fov_y)

        self._spiral_contour_idx = 0
        self._spiral_last_contour_len = None
        self._csv_path = None
        self.counter = 0


        # self._publish_pose(0, 0, -0.50)

        rclpy.spin_once(self, timeout_sec=0.1)


        #threading.Thread(target=self.worker, daemon=True).start()

    def _get_algo_name(self, algo_num):
        return {1: 'global_zigzag', 2: 'adaptive_zigzag', 3: 'centroid', 4: 'spiral'}.get(algo_num, 'unknown')

    def _init_csv(self):
        algo_name = self._get_algo_name(self.algo)
        dt_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f"{dt_str}_{algo_name}.csv"
        self._csv_path = os.path.join(os.getcwd(), fname)
        with open(self._csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'total_particles', 'particles_remaining'])

    def _log_csv(self):
        if not hasattr(self, '_csv_path'):
            self._init_csv()
        ts = datetime.now().isoformat()
        with open(self._csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ts, self.total_particles, self.particles_remaining])

    def _close_csv(self):
        if hasattr(self, '_csv_path'):
            self._csv_path = None

    def _total_particles_cb(self, msg):
        self.total_particles = msg.time_ref.sec

    def _particles_removed_cb(self,msg):
        self.particles_removed = msg.time_ref.sec
        self.particles_remaining = self.total_particles - self.particles_removed

    def _pose_cb(self, msg):
        self.latest_pose = msg.pose
        # print("Latest pose is", self.latest_pose.position)
        
    def _mask_cb(self, msg):
        try:
            if self.bridge is not None:
                arr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            else:
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
            self.latest_mask = np.flip(arr.astype(bool), axis=0)
        except Exception as e:
            self.get_logger().warn(f"mask decode failed: {e}")

        # print(self.latest_mask.shape)
        # plt.imshow(self.latest_mask)
        # plt.show()
        self._generate_ant_colony_path()

    def _center_cb(self, msg):
        self.latest_center = (msg.point.x, msg.point.y)

    def px_to_xy(self, u, v):
        W, H = 100.0, 100.0
        x = (((u+0.5)/W) - 0.5) * self.fov_x
        y = (0.5 - ((v+0.5)/H)) * self.fov_y
        return x, y

    def _publish_pose(self, x, y, z):
        if self.latest_pose is None:
            return
        out = PoseStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.pose.position.x = x
        out.pose.position.y = y
        out.pose.position.z = z
        # Use the fixed orientation
        out.pose.orientation.x = self.orientation[0]
        out.pose.orientation.y = self.orientation[1]
        out.pose.orientation.z = self.orientation[2]
        out.pose.orientation.w = self.orientation[3]
        # print(f"Publishing pose: {out.pose.position.x}, {out.pose.position.y}, {out.pose.position.z}")
        self.servo_pub.publish(out)

    def move_blocking(self, goal):
        rate = 1.0/self.control_rate_hz
        print(f"Moving to {goal} with rate {rate:.2f} Hz")
        print("rclpy.ok() is", rclpy.ok())
        # goal can be (x, y) or (x, y, z)
        while rclpy.ok() and self.latest_pose is not None:
            cur = np.array([
                self.latest_pose.position.x,
                self.latest_pose.position.y
            ])
            current_z = self.latest_pose.position.z
            
            d = goal - cur
            dist = np.linalg.norm(d)
            # print(f"Current position: {cur}, Goal: {goal}, Distance: {dist:.4f}")
            # z=-1.4
            step = d * min(1.0, self.max_step/(dist+1e-9))
            target = cur + step
            x = float(target[0])
            y = float(target[1])
            r = math.sqrt(x**2 + y**2)
            r_max = math.sqrt((self.fov_x/2)**2 + (self.fov_y/2)**2)
            origin_height = -1.4
            outer_height = -1.2
            z = origin_height + (r / r_max) * (outer_height - origin_height)
            if current_z < z:
                z_interim = z
            else:
                z_interim = current_z-0.025
            
            if dist < self.tol_xy and current_z < z:
                return
            
            


            z_target = z_interim + current_z  # Limit z step to 0.1 m
            
            y_target, x_target = target
            x_target = -x_target  # Invert y-coordinate for ROS compatibility

            self._publish_pose(x_target, y_target, z_interim)
            if self.counter % 10 == 0:
                if not self._csv_path:
                    self._init_csv()
                self._log_csv()
                self.counter =8
            else:
                self.counter -= 1
            time.sleep(rate)
        if self.latest_pose is None:
            self.get_logger().warn("Latest pose is None, cannot move.")

    def run_zigzag(self, x_min, x_max, y_min, y_max):
        y = y_min
        row = 0
        while y <= y_max:
            if self.particles_remaining == 0:
                print("All particles removed. Returning to menu.")
                return
            if row % 2 == 0:
                pts = [(x_min, y), (x_max, y)]
            else:
                pts = [(x_max, y), (x_min, y)]
            for p in pts:
                
                if self.particles_remaining == 0:
                    print("All particles removed. Returning to menu.")
                    return
                print("Moving to point:", p if p is not None else "None")
                self.move_blocking(p)
            y += self.d_zig
            row += 1
    def run_global_zigzag(self):
        self.run_zigzag(-self.fov_x/2, self.fov_x/2, -self.fov_y/2, self.fov_y/2)

    def run_adaptive_zigzag(self):
        if self.particles_remaining == 0:
            print("All particles removed. Returning to menu.")
            return
        if self.latest_mask is None or not self.latest_mask.any():
            self.run_global_zigzag()
            return
        idx = np.argwhere(self.latest_mask)
        vmin, umin = idx.min(axis=0)
        vmax, umax = idx.max(axis=0)
        x_min, y_max = self.px_to_xy(umin, vmin)
        x_max, y_min = self.px_to_xy(umax, vmax)
        self.run_zigzag(x_min, x_max, y_min, y_max)

    def run_centroid(self):
        while rclpy.ok():
            if self.particles_remaining == 0:
                print("All particles removed. Returning to menu.")
                return
            if self.latest_center is not None:
                cx, cy = self.px_to_xy(*self.latest_center)
                print(f"Moving to centroid at ({cx}, {cy})")
                self.move_blocking((cx, cy))
            time.sleep(0.05)

    def run_spiral(self):
        angle = getattr(self, '_spiral_angle', 0.0)
        step_deg = 2.5 ##self.spir_rate_deg
        offset_pixels = 5  # Distance to stay inside the blood field (in pixels)
        while rclpy.ok():
            if self.particles_remaining == 0:
                print("All particles removed. Returning to menu.")
                return
            if self.latest_center is None or self.latest_mask is None or not self.latest_mask.any():
                time.sleep(0.05)
                print("No center or mask, waiting...")
                continue

            mask = self.latest_mask.astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                time.sleep(0.05)
                print("No contours found, waiting...")
                continue
            contour = max(contours, key=cv2.contourArea).squeeze()
            if contour.ndim != 2 or contour.shape[0] < 3:
                angle = 0.0
                self._spiral_angle = angle
                print("Contour too small, resetting angle.")
                continue

            # find the largest contour 
            # contours, _ = cv2.findContours(mask_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

            # draw the largest contour to fill in the holes in the mask
            final_result = np.ones(mask.shape[:2]) # create a blank canvas to draw the final result
            final_result = cv2.drawContours(final_result, contours, -1, color=(0, 255, 0), thickness=1)
            # add centroid pixel
            if self.latest_center is not None:
                cx, cy = map(int, self.latest_center)
                final_result = cv2.circle(final_result, (cx, cy), radius=1, color=(0,0,0), thickness=1)

            # # # show results
            # cv2.imshow('mask', mask)
            # # # cv2.imshow('img', img)
            # cv2.imshow('final_result', final_result)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()



            # Get centroid in pixel coordinates
            cx_px, cy_px = self.latest_center
            centroid = np.array([cx_px, cy_px])

            # Cast ray from centroid at current angle
            theta = math.radians(angle)
            ray_dir = np.array([math.cos(theta), math.sin(theta)])
            # make the start and exnd coords of a line to draw. start is centroid, end is the edge of the mask along the ray_dir
            line_length = max(mask.shape) * 2
            line_start = centroid 
            line_end = centroid + ray_dir * line_length
            # cap at image bounds
            line_end[0] = np.clip(line_end[0], 0, mask.shape[1]-1)
            line_end[1] = np.clip(line_end[1], 0, mask.shape[0]-1)
            # draw the ray on the final_result
            ray_mask = np.ones(mask.shape[:2])
            ray_mask = cv2.line(ray_mask, tuple(map(int, line_start)), tuple(map(int, line_end)), color=(0,128,0), thickness=1)
            print(f"Centroid pixels: ({cx_px}, {cy_px})")
            # print (f"Ray from {line_start} to {line_end}")
            
            # both images are white background an black contoure.  I want to find where both images are black
            combo = (1 - final_result) * (1 - ray_mask)
            active_pixels = np.argwhere(combo > 0.5)
            # print(f"Active pixels along ray: {active_pixels}")

            #target furthest from centroid
            if len(active_pixels) == 0:
                print("No active pixels found along ray, skipping step.")
                angle = (angle + step_deg) % 360
                self._spiral_angle = angle
                time.sleep(0.05)
                continue
            centroid_comp = np.array([centroid[1], centroid[0]])  # swap to (v, u) for comparison
            target_px = active_pixels[np.argmax(np.linalg.norm(active_pixels - centroid_comp, axis=1))]
            target_px = (target_px[1], target_px[0])  # swap back to (u, v)

            offset = ray_dir * offset_pixels
            targetpx = np.array(target_px) - offset

            print(f"Target pixel along ray: {target_px}")          

            u, v = target_px

            cv2.imshow('combo', combo*255)

            x, y = self.px_to_xy(u, v)

            print(f"Spiral moving to ({x:.3f}, {y:.3f}) at angle {angle:.1f} deg")
            self.move_blocking((x, y))

            angle = (angle + step_deg) % 360
            self._spiral_angle = angle
            # time.sleep(0.05)

    def worker(self):
        while rclpy.ok():
            self.get_logger().info(f"Algorithm {self.algo} selected.")
            if self.particles_remaining == 0:
                self._init_csv()
            if self.algo == 1:
                self.run_global_zigzag()
            elif self.algo == 2:
                self.run_adaptive_zigzag()
            elif self.algo == 3:
                self.run_centroid()
            elif self.algo == 4:
                self.run_spiral()
            # If finished, go back to menu
            if self.particles_remaining == 0:
                self._close_csv()
                print("Returning to menu for new algorithm selection.")
                self.algo = self._menu_select()

    def _menu_select(self):
        print("\nSelect algorithm:\n 1 - Global zigzag\n 2 - Adaptive zigzag\n 3 - Centroid follower\n 4 - Inward spiral\n")
        sel = 0
        while sel not in [1,2,3,4]:
            try:
                sel = int(input('Enter 1-4: ').strip())
            except Exception:
                sel = 0
        return sel

    def _generate_coordinates(self):
        """
        The function overlays the coordinates onto the mask. Each element in the mask will have a corresponding coordinate
        """
        self.img_height = 100
        self.img_width = 100
        coord = np.zeros((self.img_height, self.img_width, 2))
        x_coordinates = np.linspace(-self.fov_x / 2, self.fov_x / 2, self.img_width)
        y_coordinates = np.linspace(-self.fov_y / 2, self.fov_y / 2, self.img_height)
        for index, y in enumerate(y_coordinates):
            coord[index] = np.array([x_coordinates, np.ones(self.img_width) * y]).T
        return coord

    def _generate_ant_colony_path(self):
        coord = self._generate_coordinates()
        mask = erode(self.latest_mask)
        nodes, distances, _ = pre_process_data(mask, coord) # Generates a distance matrix and extracts the blood pixels
        model = AntColony(nodes, distances)

        first_node = 0
        shortest_path, distance = model.ant_colony(first_node)

        node_list = shortest_path[:, 1]
        node_list = np.concatenate(([first_node], node_list))
        path = nodes[node_list]

        def plot():
            target_arr = np.zeros((path.shape[0], 2))
            print(f"Path:\n{path}\n\n")
            for index, node in enumerate(path):
                temp = np.all(coord == node, axis=2)
                x, y = np.where(temp)
                target_arr[index] = np.array([x, y]).T
            print(f"Approximate Distance Travelled: {distance}m")
            plt.imshow(mask, cmap='gray')
            plt.plot(target_arr[:, 1], target_arr[:, 0])
            plt.show()
        # plot()

        return path
    
    def _run_ant_colony(self):
        path = self._generate_ant_colony_path()
        # ------- Algorithm to follow path written here ------- #
        # TODO

def erode(mask):
    eroded_mask = binary_erosion(mask, structure=np.ones((5, 5))).astype(np.uint8)
    return eroded_mask

def main(argv=None):
    rclpy.init(args=argv)
    node = SimpleController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

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

from sensor_msgs.msg import Image, TimeReference, PointCloud2
from geometry_msgs.msg import PoseStamped, PointStamped

import PyKDL as kdl

import threading
import cv2    

import csv
    
import os
from datetime import datetime

from util.util.AntColonyOptimisation import AntColony, pre_process_data
from util.util.ArtificialPotentialField import ArtificialPotentialField
from util.util.CoveragePathPlanner import PathPlanner
from util.util.ModelPredictiveControl import ModelPredicitveController
from util.util.ContourFinder import isolate_contour, isolate_all_contours, find_max_contour
from util.util.Abstraction import redefine_values, avg_pool
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion


try:
    from cv_bridge import CvBridge
    _HAS_CV = True
except Exception:
    _HAS_CV = False

MODEL = 'Model1'

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

        self.algo = self._menu_select()
        self.prev_pos = None
        self.distance_travelled = 0
        self.completed_50 = False
        self.completed_90 = False
        self.planner = None

        qos = QoSProfile(depth=10)
        self.pose_sub = self.create_subscription(PoseStamped, f"/{self.arm}/measured_cp", self._pose_cb, qos)
        self.mask_sub = self.create_subscription(Image, '/camera/blood_mask', self._mask_cb, qos)
        self.center_sub = self.create_subscription(PointStamped, '/camera/blood_blob_center', self._center_cb, qos)
        self.servo_pub = self.create_publisher(PoseStamped, f"/{self.arm}/servo_cp", 1)
        self.total_particles_sub = self.create_subscription(TimeReference, 'numparticles', self._total_particles_cb, qos)
        # self.depth = self.create_subscription(PointCloud2, '/depth', self.callback, qos)
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


        threading.Thread(target=self.worker, daemon=True).start()

    # def callback(self, msg):
    #     print("Point Cloud Received")
    #     print(msg)

    def _get_algo_name(self, algo_num):
        return {1: 'global_zigzag', 2: 'adaptive_zigzag', 3: 'centroid', 4: 'spiral', 5: 'aco',
                6: 'nearest_neighbour', 7: 'apf', 8: 'sweep', 9: 'spiral_2', 10: 'mpc', 11: 'mpc_path', 12: 'mpc_no_apf'}.get(algo_num, 'unknown')

    def _init_csv(self):
        algo_name = self._get_algo_name(self.algo)
        dt_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f"{dt_str}_{algo_name}.csv"
        self._csv_path = os.path.join(os.getcwd(), 'recordings', fname)
        with open(self._csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'total_particles', 'particles_remaining', 'distance_travelled'])

    def _log_csv(self):
        if not hasattr(self, '_csv_path'):
            self._init_csv()
        ts = datetime.now().isoformat()
        with open(self._csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ts, self.total_particles, self.particles_remaining, self.distance_travelled])

    def _close_csv(self):
        if hasattr(self, '_csv_path'):
            self._csv_path = None

    def _total_particles_cb(self, msg):
        self.total_particles = msg.time_ref.sec

    def _particles_removed_cb(self,msg):
        self.particles_removed = msg.time_ref.sec
        self.particles_remaining = self.total_particles - self.particles_removed

    def _find_distance_moved(self, msg):
        """
        Close enough approximation for the distance travelled. Assumes that the arm travelled in a straight line from the previous point.
        """
        if self.prev_pos is not None:
            pos_x = msg.pose.position.x
            pos_y = msg.pose.position.y
            pos_z = msg.pose.position.z
            prev_x = self.prev_pos.x
            prev_y = self.prev_pos.y
            prev_z = self.prev_pos.z
            prev = np.array([prev_x, prev_y, prev_z])
            pos = np.array([pos_x, pos_y, pos_z])
            self.distance_travelled += np.linalg.norm(prev - pos)
        self.prev_pos = msg.pose.position

    def _pose_cb(self, msg):
        self.latest_pose = msg.pose
        self._find_distance_moved(msg)
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
        #print(f"Moving to {goal} with rate {rate:.2f} Hz")
        #print("rclpy.ok() is", rclpy.ok())
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
        try:
            while rclpy.ok():
                self.get_logger().info(f"Algorithm {self.algo} selected.")
                if self.particles_remaining == 0:
                    self._my_init_csv()
                    self._init_csv()
                if self.algo == 1:
                    self.run_global_zigzag()
                elif self.algo == 2:
                    self.run_adaptive_zigzag()
                elif self.algo == 3:
                    self.run_centroid()
                elif self.algo == 4:
                    self.run_spiral()
                elif self.algo == 5:
                    self._run_ant_colony()
                elif self.algo == 6:
                    self._run_nearest_neighbour()
                elif self.algo == 7:
                    self._run_potential_fields()
                elif self.algo == 8:
                    self._run_sweep()
                elif self.algo == 9:
                    self._run_spiral_2()
                elif self.algo == 10:
                    self._run_mpc()
                elif self.algo == 11:
                    self._run_mpc_with_path()
                elif self.algo == 12:
                    self._run_mpc_no_apf()
                # If finished, go back to menu
                self.check_90_50_point()
                if self.particles_remaining == 0 or (not np.any(self.latest_mask) and self.particles_remaining != 8000):
                    self._close_csv()
                    self._my_close_csv()
                    self.get_logger().info(f"Total distance travelled: {self.distance_travelled}m.")
                    stop = time.time()
                    t = stop - self.start
                    self.get_logger().info(f"Total time taken: {t}s.")
                    self.record_results(t)

                    print("Returning to menu for new algorithm selection.")
                    self.algo = self._menu_select()
                    self.initialise()
        except KeyboardInterrupt:
            self.record_results(np.inf)

    def check_90_50_point(self):
        self._my_log_csv()
        # Record the 90% and 50% point
        if self.particles_remaining < 800 and not self.completed_90:
            self.completed_90 = True
            print("90% Complete")
            stop = time.time()
            self.t_90 = stop - self.start
            self.distance_90 = self.distance_travelled
        elif self.particles_remaining < 4000 and not self.completed_50:
            self.completed_50 = True
            print("50% Complete")
            stop = time.time()
            self.t_50 = stop - self.start
            self.distance_50 = self.distance_travelled

    def initialise(self):
        self.distance_travelled = 0
        self.planner = None
        self.completed_50 = False
        self.completed_90 = False

    def record_results(self, t):
        options = {1: 'basic', 2: 'fourpits', 3: 'random', 4: 'multiregions', 5: 'centre_tissue'}
        correct_input = False
        while not correct_input:
            choice = int(input("Enter the model that was used: "))
            if choice <= 5 and choice >= 1:
                correct_input = True
        model = options[choice]
        with open(f'results_{model}.txt', 'a') as f:
            # f.write(f"{self._get_algo_name(self.algo)}: Distance = {self.distance_travelled:.3f}m, Time Taken = {t:.5f}s, 90% Point -> Distance = {self.distance_90:.3f}m, Time Taken = {self.t_90:.5f}s, 50% Point -> Distance = {self.distance_50:.3f}m, Time Taken = {self.t_50:.5f}s\n")
            f.write(f"{self._get_algo_name(self.algo)},{self.distance_travelled:.3f},{t:.5f},{self.distance_90:.3f},{self.t_90:.5f},{self.distance_50:.3f},{self.t_50:.5f}\n")

    def _menu_select(self):
        print("\nSelect algorithm:\n 1 - Global zigzag\n 2 - Adaptive zigzag\n 3 - Centroid follower\n 4 - Inward spiral\n" \
        " 5 - ACO\n 6 - Nearest Neighbour\n 7 - APF\n 8 - Sweep\n 9 - Spiral 2\n 10 - MPC\n 11 - MPC-w-path\n 12 - MPC-no-apf")
        sel = 0
        while sel not in [1,2,3,4,5,6,7,8,9,10,11,12]:
            try:
                sel = int(input('Enter 1-12: ').strip())
            except Exception:
                sel = 0
        self.start = time.time()
        return sel

    def _generate_coordinates(self):
        """
        The function overlays the coordinates onto the mask. Each element in the mask will have a corresponding coordinate.
        This function is needed as some of the path planners operate on the coordinates and not the image pixels.
        """
        self.img_height = 100
        self.img_width = 100
        coord = np.zeros((self.img_height, self.img_width, 2))
        x_coordinates = np.linspace(-self.fov_x / 2, self.fov_x / 2, self.img_width)
        y_coordinates = np.linspace(-self.fov_y / 2, self.fov_y / 2, self.img_height)
        for index, y in enumerate(y_coordinates):
            coord[index] = np.array([x_coordinates, np.ones(self.img_width) * y]).T
        return coord
    
    def _alt_gen_coordinates(self):
        """
        This is plan B if self._gen_coordinates does not work. This might be even better as it is based on self.px_to_xy
        """
        W, H = 100, 100
        u = np.linspace(0, W - 1, W)
        v = np.linspace(0, H - 1, H)
        x = (((u+0.5)/W) - 0.5) * self.fov_x
        y = (0.5 - ((v+0.5)/H)) * self.fov_y
        coord = np.zeros((W, H, 2))
        for index, y_val in enumerate(y):
            coord[index] = np.array([x, np.ones(W) * y_val]).T
        return coord
    
    def _filter_targets(self, mask, method='closest'):
        """
        When a blood pool splits into several blood pools, some algorithms will struggle to find an efficient path. This function isolates onde blood pool from the rest.
        Parameters:
        -----------
        mask: numpy array of the binary mask from the blood detector
        method: ['closest', 'largest']. The parameter used to determine which blood pool to isolate.
        """
        if method.lower() == 'largest':
            # Largest First Method
            contour = find_max_contour(mask.astype(np.uint8)) # Find the largest contour
            if type(contour) != int:
                _, isolated_contour = isolate_contour(mask.astype(np.uint8), contour) # Make a mask that isolates the largest contour
                return isolated_contour
            else:
                return mask # There are no contours
        else:
            # Closest First Method
            while self.latest_pose is None:
                time.sleep(0.1) # Wait to avoid any errors
            pos = np.array([self.latest_pose.position.x, self.latest_pose.position.y]) # Get the position of the arm
            coordinates = self._alt_gen_coordinates() # Generate coordinates for the frame
            _, all_masks = isolate_all_contours(mask.astype(np.uint8)) # Get a mask for each contour separated from the others
            shortest_distance = np.inf # Initialise the shortest distance
            closest_contour = mask
            for m in all_masks:
                coord = coordinates[m > 0] # Extract the coordinates to find the distance to the closest point
                distance = np.linalg.norm(coord - pos, axis=1) # Finds the distance to all the points
                distance = np.min(distance) # Takes only the shortest distance to the contour
                if distance < shortest_distance:
                    shortest_distance = distance
                    closest_contour = m # Assign the current contour as the closest
            return closest_contour

    def _aco_helper_function(self):
        """
        The line of code below is used for the both the Nearest Neighbour and ACO, so a function is made so it can be reused.
        The function generates coordinates and makes the distance matrix. It initialises the model and finds the closest node to the arm to make it the start node.
        """
        while self.latest_mask is None:
            time.sleep(0.5)
        coord = self._alt_gen_coordinates() # Generating coordinates
        mask = self.latest_mask
        # mask = erode(self.latest_mask)
        nodes, distances, _ = pre_process_data(mask, coord) # Generates a distance matrix and extracts the blood pixels
        # The above function will also reduce the number of blood pixels by applying a filter to reduce the number of nodes
        model = AntColony(nodes, distances) # Initialisation

        # first_node = 0
        while self.latest_pose is None:
            print("Waiting for pose...")
            time.sleep(1)
        x_pos = self.latest_pose.position.x
        y_pos = self.latest_pose.position.y
        first_node = np.argmin(np.linalg.norm(nodes - np.array([x_pos, y_pos]), axis=1)) # Find the closest node to start from there
        return model, first_node, nodes, coord, mask

    def _generate_ant_colony_path(self):
        """
        The function generates a path that is based on the ACO algorithm. The function uses the latest mask and generates the 2D coordinates for
        the mask and plans the path based on the coordinates
        """
        model, first_node, nodes, coord, mask = self._aco_helper_function()
        if len(nodes) == 0:
            return
        shortest_path, distance = model.ant_colony(first_node) # Run algorithm to find the shortest path
        # The distance above is the theoretical distance to the target

        # The node_list is just an index list each row contains 2 indexes: the current node index and the next node index.
        node_list = shortest_path[:, 1] # Only the list of next nodes are needed, so ignore the first column.
        node_list = np.concatenate(([first_node], node_list)) # Add the first node to the path list
        # The below line extracts the path from the index list
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
    
    def _generate_nearest_neighbour_path(self):
        """
        The function generates a path that is based on the nearest neighbour algorithm. The function uses the latest mask and generates the 2D coordinates for
        the mask and plans the path based on the coordinates
        """
        model, first_node, nodes, coord, mask = self._aco_helper_function()
        if len(nodes) == 0:
            return
        shortest_path, distance = model.greedy_nearest_neigbour(first_node) # Run algorithm to find the shortest path
        # The distance above is the theoretical distance to the target

        node_list = np.concatenate(([first_node], shortest_path)) # Add the first node to the path list
        # I might just move the above line so that this is done in the algorithm itself
        # The node_list is just an index list specifying the index of the nodes to go to. The below line extracts the path from the index list
        path = nodes[node_list]

        def plot():
            """
            For Visualisation Only
            """
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
        # plot() # Comment out when running

        return path
    
    def _initialise_artificial_potential_field(self, coordinates):
        while self.latest_mask is None:
            time.sleep(0.1)
        mask = self.latest_mask.copy()
        mask = binary_erosion(mask, np.ones((3, 3)))
        mask = self._filter_targets(mask, method='closest')
        target_coords = coordinates[mask > 0]
        # obstacle_coors = np.array([[np.inf, np.inf], [np.inf, np.inf]]) # Dummy variable as there are no obstacles (since no point cloud)
        model = ArtificialPotentialField(
            targets=target_coords,
            obstacles=None,
            att_gain=0.5,
            rep_gain=0.5, # Useless since no obstacles
            max_distance=0.1, # Useless since no obstacles
            mini_batches=False
        )
        return model

    def _artificial_potential_field(self, model, coordinates, random=False):
        # mask = self.latest_mask
        # target_coords = coordinates[mask > 0]
        # print("Length:", len(target_coords), np.sum(mask))
        while self.latest_pose is None:
            time.sleep(0.1)

        if not random:
            x_pos = self.latest_pose.position.x
            y_pos = self.latest_pose.position.y
            pos = np.array([x_pos, y_pos])
            force = model.resultant_force(pos, targets=targets)
            return force + pos
        
        random_row, random_col = np.random.randint(0, 100), np.random.randint(0, 100)
        row_top_limit = min(100, random_row + 24)
        col_top_limit = min(100, random_col + 24)
        row_bottom_limit = min(0, random_row - 24)
        col_bottom_limit = min(0, random_col - 24)
        mask = np.zeros((100, 100)) # Creating an empty mask

        latest_mask = self._filter_targets(self.latest_mask, method='closest')

        empty = False
        counter = 0
        while not empty:
            # Assign a random region of the mask to the empty mask to focus on
            mask[row_bottom_limit:row_top_limit, col_bottom_limit:col_top_limit] = latest_mask[row_bottom_limit:row_top_limit, col_bottom_limit:col_top_limit]
            targets = coordinates[mask > 0]

            x_pos = self.latest_pose.position.x
            y_pos = self.latest_pose.position.y
            pos = np.array([x_pos, y_pos])
            force = model.resultant_force(pos, targets=targets)

            empty = not np.any(mask)
            force += pos
            print(f"Force: {force}")
            self.move_blocking((force[0], force[1]))
            counter += 1
            if counter > 80:
                break
        return pos + force
    
    def check_blood_on_path(self, path, coords):
        for target in path:
            search = (coords == target)
            search = np.all(search, axis=2)
            row, col = np.where(search)
            if np.all(self.latest_mask[row, col]):
                return True
        return False

    def _follow_path(self, path, min_distance_to_target=0.02, start_closest=False):
        """
        The function accepts a path in the two-dimensional space and follows it.
        Parameters:
        -----------
        path: nx2 array of n points to go to. Each point consists of an x and y coordinate
        min_distance_to_target: The minimum distance that the arm has to be to the target for it to move to the next point
        """
        # ------- Algorithm to follow path written here ------- #
        coords = self._alt_gen_coordinates()
        path_length = len(path)
        while self.latest_pose is None:
            time.sleep(0.5)

        if start_closest:
            x_pos = self.latest_pose.position.x
            y_pos = self.latest_pose.position.y
            distance = np.linalg.norm(path - np.array([x_pos, y_pos]), axis=1)
            path_index = np.argmin(distance)
            # print(path_index)
            if path_index != 0:
                path = np.concatenate((path[path_index:], path[path_index - 1::-1]))
        path_index = 0

        target_pos = path[path_index] # First point to go to
        counter = 0
        while path_index < path_length:
            self.check_90_50_point()
            # Check the path every timesteps
            if not self.check_blood_on_path(path[path_index:], coords):
                print('No more blood remaining along path.')
                break
            x_pos = self.latest_pose.position.x
            y_pos = self.latest_pose.position.y
            distance = np.linalg.norm(target_pos - np.array([x_pos, y_pos])) # Get the distance to the point it has to go to
            # print(f" Index: \n {path_index}\n\n Position: \n {x_pos, y_pos} \n\n Distance: \n {distance}\n\n Target:\n {target_pos}")
            if distance < min_distance_to_target or counter == 20:
                # If the arm is close enough, then move to the next point
                path_index += 1
                print('Moving to next index')
                if path_index == path_length:
                    break
                counter = 0
            target_pos = path[path_index]
            counter += 1
            print('Moving...')
            self.move_blocking((target_pos[0], target_pos[1]))
            time.sleep(0.2)
    
    def _run_ant_colony(self):
        """
        Runs the ACO algorithm.
        See https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms for more information about the algorithm
        """
        path = self._generate_ant_colony_path()
        self._follow_path(path)

    def _run_nearest_neighbour(self):
        """
        Runs the Greedy Nearest Neighbour algorithm.
        See https://en.wikipedia.org/wiki/Travelling_salesman_problem#Constructive_heuristics for more information about the algorithm
        """
        path = self._generate_nearest_neighbour_path()
        self._follow_path(path)

    def _run_potential_fields(self):
        """
        Runs the Artificial Potential Field Algorithm.
        See https://en.wikipedia.org/wiki/Motion_planning#Artificial_potential_fields for more information
        """
        coordinates = self._alt_gen_coordinates()
        model = self._initialise_artificial_potential_field(coordinates)
        force = self._artificial_potential_field(model, coordinates, random=True)
        # print(f"Force: {force}")
        # self.move_blocking((force[0], force[1]))

    def _run_sweep(self):
        while self.latest_mask is None:
            time.sleep(0.5)
        coordinates = self._alt_gen_coordinates()
        mask = self.latest_mask
        mask = self._filter_targets(mask, method='closest')
        prev_mask = mask
        if np.sum(mask) != 0:
            mask = erode(mask, 10) # Erode the mask as there is no need to move all the way to the end as it will be suctioned in
            if np.sum(mask) <= 1:
                mask = prev_mask # If the region is very small, eroding the mask may leave nothing. In that case, revert to the uneroded mask
            model = PathPlanner(mask=mask, algorithm="SWEEP", depths=coordinates)
            path = model.sweep_planner()
            if path is None or len(path) == 0:
                return
            # print(path)
            self._follow_path(path, start_closest=False)

    def _run_spiral_2(self):
        while self.latest_mask is None:
            time.sleep(0.5)
        coordinates = self._alt_gen_coordinates()
        x = coordinates[:, :, 0]
        y = coordinates[:, :, 1]
        z = np.zeros((y.shape[0], y.shape[1]))
        coordinates = np.array([x, y, z]).T # The z-coordinates are needed for the algorithm.
        mask = self.latest_mask
        mask = self._filter_targets(mask, method='closest')
        if np.sum(mask) == 0:
            # No blood pixels so nothing to plan
            return
        prev_mask = mask
        mask = erode(mask, 10)
        if np.sum(mask) == 0:
            mask = prev_mask # If the region is very small, eroding the mask may leave nothing. In that case, revert to the uneroded mask
        model = PathPlanner(mask=mask, algorithm="SPIRAL", depths=coordinates, res=15)
        path = model.spiral_path()
        if len(path) != 0:
            path = path[:, 0:2] # Drop the z-axis
            # self.move_blocking(path[0])
            self._follow_path(path)

    def _run_mpc(self):
        while self.latest_mask is None:
            time.sleep(0.5)
        mask = self.latest_mask.copy()
        mask = self._filter_targets(mask, method='closest')
        coordinates = self._alt_gen_coordinates()

        APF = ArtificialPotentialField(
            targets=None,
            obstacles=None,
            att_gain=0.04,
            rep_gain=0.2,
            max_distance=0.10
        )

        controller = ModelPredicitveController (
            6,
            400,
            np.identity(2) * 0.04,
            -0.4,
            0.4,
            APF=APF,
            n=2,
            rem_w=10000,
            dist_w=0
        )

        x_pos = self.latest_pose.position.x
        y_pos = self.latest_pose.position.y
        pos = np.array([x_pos, y_pos])

        target_coords = coordinates[mask > 0]
        if len(target_coords) > 150:
            new_mask = avg_pool(mask, 10)
            new_coordinates = redefine_values(coordinates, 10, 2)
            target_coords = new_coordinates[new_mask > 0]
        # target_coords, _, _ = pre_process_data(mask, coordinates)
        u, _ = controller.control(pos, target_coords, None)
        pos += u
        self.move_blocking(pos)

    def _run_mpc_with_path(self):
        while self.latest_mask is None:
            time.sleep(0.5)
        mask = self.latest_mask
        mask = self._filter_targets(mask, method='closest')
        coordinates = self._alt_gen_coordinates()

        APF = ArtificialPotentialField(
            targets=None,
            obstacles=None,
            att_gain=0.05,
            rep_gain=0.35,
            max_distance=0.10
        )
        if self.planner is None:
            self.planner = input("Enter the path planner: ").strip().upper()
            print(self.planner)
            if self.planner == 'SPIRAL':
                print('Spiral Chosen')
                model = PathPlanner(mask=mask, algorithm="SPIRAL", depths=coordinates, res=20)
                self.path = model.spiral_path()
            elif self.planner == 'SWEEP':
                print('Sweep Chosen')
                model = PathPlanner(mask=mask, algorithm="SWEEP", depths=coordinates)
                self.path = model.sweep_planner()
            elif self.planner == 'ACO':
                print('ACO Chosen')
                self.path = self._generate_ant_colony_path()
            else:
                print('Nearest Neighbour Chosen')
                self.path = self._generate_nearest_neighbour_path()

        if np.sum(mask) < 200 and self.path is not None:
            self.path = None
            print('Dropping path')
            APF = ArtificialPotentialField(
                targets=None,
                obstacles=None,
                att_gain=0.2,
                rep_gain=0.35,
                max_distance=0.10
            )

        controller = ModelPredicitveController (
            8,
            600,
            np.identity(2) * 0.015,
            -0.5,
            0.5,
            APF=APF,
            n=2,
            path=self.path,
            rem_w=0,
            dist_w=0,
            path_w=4000
        )

        x_pos = self.latest_pose.position.x
        y_pos = self.latest_pose.position.y
        pos = np.array([x_pos, y_pos])

        target_coords = coordinates[mask > 0]
        if len(target_coords) > 400:
            target_coords = target_coords[::2]
        u, _ = controller.control(pos, target_coords, None)
        pos += u
        self.move_blocking(pos)

    def _run_mpc_no_apf(self):
        while self.latest_mask is None:
            time.sleep(0.5)
        mask = self.latest_mask
        mask = self._filter_targets(mask, method='closest')
        coordinates = self._alt_gen_coordinates()

        controller = ModelPredicitveController (
            8,
            600,
            np.identity(2) * 0.03,
            -0.45,
            0.45,
            n=2
        )

        while self.latest_pose is None:
            time.sleep(0.5)
        x_pos = self.latest_pose.position.x
        y_pos = self.latest_pose.position.y
        pos = np.array([x_pos, y_pos])

        target_coords = coordinates[mask > 0]
        if len(target_coords) > 400:
            target_coords = target_coords[::2]
        u, _ = controller.control(pos, target_coords, None)
        pos += u
        self.move_blocking(pos)

    def _my_init_csv(self):
        algo_name = self._get_algo_name(self.algo)
        dt_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f"{dt_str}_{algo_name}.csv"
        self._my_csv_path = os.path.join(os.getcwd(), MODEL, fname)
        with open(self._my_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'total_particles', 'particles_remaining', 'distance_travelled'])

    def _my_log_csv(self):
        if not hasattr(self, '_my_csv_path'):
            self._my_init_csv()
        if not self._my_csv_path:
            self._my_init_csv()
        ts = time.time() - self.start
        with open(self._my_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ts, self.particles_remaining, self.distance_travelled])

    def _my_close_csv(self):
        if hasattr(self, '_my_csv_path'):
            self._my_csv_path = None

def erode(mask, amount):
    eroded_mask = binary_erosion(mask, structure=np.ones((amount, amount))).astype(np.uint8)
    return eroded_mask

def main(argv=None):
    rclpy.init(args=argv)
    node = SimpleController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.record_results(np.inf)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

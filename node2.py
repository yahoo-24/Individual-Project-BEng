#!/usr/bin/env python3
"""
ROS 2 blood suction controller — **blocking moves** version
(cartesian control via <arm>/servo_cp)

What changed
- Replaced timer-based incremental control with *blocking move* primitives.
- Each algorithm generates a sequence of waypoints in the image plane (XY, metres).
  The controller issues a target pose and **waits** until the robot reaches it
  within a configurable tolerance before proceeding (or re-planning).
- Speed is governed by simple limits (max_step per command and a dwell/settle loop),
  so motion is slow and predictable.
- Default end-effector orientation is set once at startup and then held fixed,
  matching your previous "send default message then edit only Cartesian position" pattern.

I/O
- Subscribes:
    /camera/blood_mask (sensor_msgs/Image, 100x100 binary)
    /camera/blood_blob_center (geometry_msgs/PointStamped)
    <arm>/measured_cp (geometry_msgs/PoseStamped)
- Publishes:
    <arm>/servo_cp (geometry_msgs/PoseStamped)

Algorithms (choose with -p algo:=1..5 or interactive)
  1. Global zigzag        — lawnmower across full FOV (d_zig spacing)
  2. Adaptive zigzag      — lawnmower inside current blood bounding box (d_zig)
  3. Centroid follower    — repeatedly move to current blob centre (kp not used here)
  4. Inward spiral        — waypoint spiral around centre; radius shrinks (spir_rate_deg, spiral_dr)
  5. Selective raster     — scan only rows that currently contain blood (d_zig)

Key parameters (override with --ros-args -p ...)
  arm_name:               default "PSM1"
  fov_x, fov_y:           metres spanned by the 100×100 image (default 0.04, 0.04)
  d_zig:                  path spacing for zigzags (m)
  spir_rate_deg:          degrees advanced per spiral waypoint
  spiral_dr:              radius decrease per spiral waypoint (m)
  tol_xy:                 position tolerance to consider a waypoint reached (m, default 0.0008)
  max_step:               max XY step per issued command (m, default 0.001)
  settle_time:            seconds to wait after entering tolerance before advancing (default 0.05)
  replan_period_s:        how often to re-scan mask & rebuild path for zigzags (default 1.0 s)
  control_rate_hz:        loop rate for blocking move polling (default 100)
  # Orientation control
  freeze_orientation:     if true, fix tool orientation to configured default
  default_roll_deg/pitch_deg/yaw_deg or default_qx/qy/qz/qw

Usage example
  ros2 run my_ros2_node blood_controller --ros-args \
    -p arm_name:=PSM1 -p algo:=2 -p fov_x:=0.05 -p fov_y:=0.05 -p d_zig:=0.004 \
    -p tol_xy:=0.0008 -p max_step:=0.001 -p control_rate_hz:=100 \
    -p default_yaw_deg:=90.0 -p freeze_orientation:=true
"""

from __future__ import annotations
import math
import threading
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, PointStamped, Quaternion

try:
    from cv_bridge import CvBridge
    _HAS_CV = True
except Exception:
    _HAS_CV = False

# ---------------------- Small helpers ----------------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# Quaternion helpers

def rpy_deg_to_quat(roll_deg: float, pitch_deg: float, yaw_deg: float) -> Quaternion:
    r = math.radians(roll_deg); p = math.radians(pitch_deg); y = math.radians(yaw_deg)
    cr, sr = math.cos(r/2.0), math.sin(r/2.0)
    cp, sp = math.cos(p/2.0), math.sin(p/2.0)
    cy, sy = math.cos(y/2.0), math.sin(y/2.0)
    q = Quaternion()
    q.w = cr*cp*cy + sr*sp*sy
    q.x = sr*cp*cy - cr*sp*sy
    q.y = cr*sp*cy + sr*cp*sy
    q.z = cr*cp*sy - sr*sp*cy
    return q

# ---------------------- Main Node ----------------------

class BloodSuctionController(Node):
    def __init__(self):
        super().__init__('blood_suction_controller_blocking')


        # Direct attribute configuration
        self.arm_name = 'PSM1'
        self.algo = 0
        self.fov_x = 1.1
        self.fov_y = 1.1
        self.z_depth = -0.9
        self.d_zig = 0.1
        self.spir_rate_deg = 12.0
        self.spiral_dr = 0.0005
        self.tol_xy = 0.0008
        self.max_step = 0.1
        self.settle_time = 0.05
        self.replan_period_s = 1.0
        self.control_rate_hz = 5.0
        # Orientation
        self.freeze_orientation = True
        self.default_roll_deg = 0.0
        self.default_pitch_deg = 90.0
        self.default_yaw_deg = -90.0
        self.default_qx = None
        self.default_qy = None
        self.default_qz = None
        self.default_qw = None

        self.measured_cp_topic = f"{self.arm_name}/measured_cp"
        self.servo_cp_topic = f"{self.arm_name}/servo_cp"

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.history = HistoryPolicy.KEEP_LAST

        # I/O
        self.pose_sub = self.create_subscription(PoseStamped, self.measured_cp_topic, self._measured_cb, qos)
        self.mask_sub = self.create_subscription(Image, '/camera/blood_mask', self._mask_cb, qos)
        self.center_sub = self.create_subscription(PointStamped, '/camera/blood_blob_center', self._center_cb, qos)
        self.servo_pub = self.create_publisher(PoseStamped, self.servo_cp_topic, 1)

        # State
        self.bridge = CvBridge() if _HAS_CV else None
        self.latest_pose: Optional[PoseStamped] = None
        self.latest_mask: Optional[np.ndarray] = None
        self.latest_center_px: Optional[np.ndarray] = None
        self.default_orientation: Optional[Quaternion] = None
        self.freeze_orientation: bool = bool(self.freeze_orientation)
        self.did_snap_orientation = False

        # Select algorithm
        if self.algo not in [1,2,3,4,5]:
            self.algo = self._menu_select()
        self.get_logger().info(f"Algorithm {self.algo} selected (blocking moves).")

        # Compute default orientation now
        self._compute_default_orientation()

        # Worker thread runs the blocking algorithm loop
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    # ---------- Callbacks ----------
    def _measured_cb(self, msg: PoseStamped):
        self.latest_pose = msg

    def _mask_cb(self, msg: Image):
        try:
            if self.bridge is not None:
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                arr = np.asarray(cv_img)
            else:
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width))
            self.latest_mask = arr.astype(bool)
        except Exception as e:
            self.get_logger().warn(f"Failed to decode mask: {e}")

    def _center_cb(self, msg: PointStamped):
        self.latest_center_px = np.array([msg.point.x, msg.point.y], dtype=float)

    # ---------- Coordinate utilities ----------
    def px_to_xy(self, u: float, v: float) -> Tuple[float,float]:
        # H = 100.0; W = 100.0
        # x = (u - (W-1)/2.0) / W * self.fov_x
        # y = ((H-1)/2.0 - v) / H * self.fov_y
        # return float(x), float(y)
        H = 100.0; W = 100.0
        # Pixel-centre mapping: ((u+0.5)/W - 0.5) ∈ (-0.5, 0.5]
        x = (((u + 0.5) / W) - 0.5) * self.fov_x
        y = (0.5 - ((v + 0.5) / H)) * self.fov_y
        return float(x), float(y)

    def bbox_from_mask(self, mask: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
        idx = np.argwhere(mask)
        if idx.size == 0:
            return None
        vmin, umin = idx.min(axis=0)
        vmax, umax = idx.max(axis=0)
        return int(umin), int(vmin), int(umax), int(vmax)

    # ---------- Orientation handling ----------
    def _compute_default_orientation(self):
        if all(v is not None for v in (self.default_qx, self.default_qy, self.default_qz, self.default_qw)):
            norm = math.sqrt(self.default_qx**2 + self.default_qy**2 + self.default_qz**2 + self.default_qw**2)
            if norm > 0:
                qx, qy, qz, qw = self.default_qx/norm, self.default_qy/norm, self.default_qz/norm, self.default_qw/norm
            else:
                qx, qy, qz, qw = self.default_qx, self.default_qy, self.default_qz, self.default_qw
            self.default_orientation = Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw))
            self.get_logger().info("Using quaternion from attributes as default orientation.")
        else:
            r = float(self.default_roll_deg)
            p = float(self.default_pitch_deg)
            y = float(self.default_yaw_deg)
            self.default_orientation = rpy_deg_to_quat(r, p, y)
            self.get_logger().info(f"Using default RPY(deg)=({r:.1f},{p:.1f},{y:.1f}).")

    def _apply_default_orientation_once(self):
        if self.did_snap_orientation or not self.freeze_orientation or self.latest_pose is None or self.default_orientation is None:
            return
        p = self.latest_pose
        out = PoseStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = p.header.frame_id
        out.pose.position = p.pose.position
        out.pose.orientation = self.default_orientation
        self.servo_pub.publish(out)
        self.did_snap_orientation = True
        self.get_logger().info("Applied initial default orientation (XYZ held).")

    # ---------- Blocking move primitives ----------
    def _publish_pose_xy(self, x: float, y: float):
        p = self.latest_pose
        if p is None:
            return
        out = PoseStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = p.header.frame_id
        out.pose.position.x = x
        out.pose.position.y = y
        out.pose.position.z = float(self.z_depth)  # Use z_depth attribute
        out.pose.orientation = self.default_orientation if (self.freeze_orientation and self.default_orientation) else p.pose.orientation
        self.servo_pub.publish(out)

    def _move_to_xy_blocking(self, goal_xy: Tuple[float,float], timeout_s: Optional[float] = None) -> bool:
        """Issue bounded steps toward goal and block until within tol_xy.
        If timeout_s is None, compute an **adaptive** timeout from path length and
        an implied minimum progress (based on max_step and control_rate_hz).
        """
        rate = 1.0 / max(1.0, self.control_rate_hz)
        t0 = time.time()
        ok_once = False
        # compute adaptive timeout if needed (distance / achievable_speed * safety)
        # achievable_speed ≈ max_step * control_rate_hz
        if self.latest_pose is not None and timeout_s is None:
            cur_xy0 = np.array([self.latest_pose.pose.position.x, self.latest_pose.pose.position.y], dtype=float)
            d0 = float(np.linalg.norm(np.array(goal_xy, dtype=float) - cur_xy0))
            v_min = max(1e-6, self.max_step * self.control_rate_hz * 0.7)
            timeout_s = 0.5 + d0 / v_min  # 0.5 s base + distance term
        while rclpy.ok():
            if self.latest_pose is None:
                time.sleep(rate); continue
            cur_xy = np.array([self.latest_pose.pose.position.x, self.latest_pose.pose.position.y], dtype=float)
            goal = np.array(goal_xy, dtype=float)
            d = goal - cur_xy
            dist = float(np.linalg.norm(d))
            if dist <= self.tol_xy:
                if not ok_once:
                    ok_once = True
                    time.sleep(self.settle_time)
                else:
                    return True
            else:
                step = d * min(1.0, self.max_step / (dist + 1e-9))
                self._publish_pose_xy(float(cur_xy[0] + step[0]), float(cur_xy[1] + step[1]))
            if timeout_s is not None and timeout_s > 0.0 and (time.time() - t0) > timeout_s:
                self.get_logger().warn("move_to_xy_blocking timeout; proceeding.")
                return False
            time.sleep(rate)
        return False
    # ---------- Path planners (waypoint generators) ----------
    def _plan_global_zigzag(self) -> List[Tuple[float,float]]:
        pts: List[Tuple[float,float]] = []
        x_min, x_max = -self.fov_x/2.0, self.fov_x/2.0
        y_min, y_max = -self.fov_y/2.0, self.fov_y/2.0
        y = y_min; row = 0
        while y <= y_max + 1e-9:
            if row % 2 == 0:
                pts.extend([(x_min, y), (x_max, y)])
            else:
                pts.extend([(x_max, y), (x_min, y)])
            row += 1; y += self.d_zig
        return pts

    def _plan_adaptive_zigzag(self) -> List[Tuple[float,float]]:
        if self.latest_mask is None or not self.latest_mask.any():
            return self._plan_global_zigzag()
        umin, vmin, umax, vmax = self.bbox_from_mask(self.latest_mask)  # type: ignore
        x_min, y_max = self.px_to_xy(umin, vmin)
        x_max, y_min = self.px_to_xy(umax, vmax)
        pts: List[Tuple[float,float]] = []
        y = y_min; row = 0
        while y <= y_max + 1e-9:
            if row % 2 == 0:
                pts.extend([(x_min, y), (x_max, y)])
            else:
                pts.extend([(x_max, y), (x_min, y)])
            row += 1; y += self.d_zig
        return pts

    def _plan_selective_raster(self) -> List[Tuple[float,float]]:
        if self.latest_mask is None or not self.latest_mask.any():
            return self._plan_global_zigzag()
        H, W = self.latest_mask.shape
        rows = np.where(self.latest_mask.any(axis=1))[0]
        pts: List[Tuple[float,float]] = []
        for k, v in enumerate(rows):
            y = self.px_to_xy(0.0, float(v))[1]
            x_min = self.px_to_xy(0.0, float(v))[0]
            x_max = self.px_to_xy(W-1.0, float(v))[0]
            if k % 2 == 0:
                pts.extend([(x_min, y), (x_max, y)])
            else:
                pts.extend([(x_max, y), (x_min, y)])
        return pts

    # ---------- Algorithm runner ----------
    def _run(self):
        # Wait for first pose
        self.get_logger().info("Waiting for first measured pose...")
        while rclpy.ok() and self.latest_pose is None:
            time.sleep(0.01)
        if not rclpy.ok():
            return
        # Snap orientation once if requested
        self._apply_default_orientation_once()

        last_replan = 0.0
        path: List[Tuple[float,float]] = []
        idx = 0

        while rclpy.ok():
            now = time.time()

            # (Re)plan paths periodically for the raster/zigzag variants
            if self.algo in (1,2,5) and (now - last_replan > self.replan_period_s or not path):
                if self.algo == 1:
                    path = self._plan_global_zigzag()
                elif self.algo == 2:
                    path = self._plan_adaptive_zigzag()
                else:
                    path = self._plan_selective_raster()
                idx = 0
                last_replan = now

            if self.algo in (1,2,5):
                if not path:
                    time.sleep(0.02); continue
                goal = path[idx]
                # Use adaptive timeout based on distance so long legs reach the image bounds
                self._move_to_xy_blocking(goal, timeout_s=None)
                idx = (idx + 1) % len(path)
                continue

            if self.algo == 3:  # Centroid follower
                if self.latest_center_px is not None:
                    cx, cy = self.px_to_xy(self.latest_center_px[0], self.latest_center_px[1])
                    self._move_to_xy_blocking((cx, cy), timeout_s=5.0)
                else:
                    # fallback to mask centroid if available
                    if self.latest_mask is not None and self.latest_mask.any():
                        ys, xs = np.nonzero(self.latest_mask)
                        u = float(np.mean(xs)); v = float(np.mean(ys))
                        cx, cy = self.px_to_xy(u, v)
                        self._move_to_xy_blocking((cx, cy), timeout_s=5.0)
                    else:
                        time.sleep(0.05)
                continue

            if self.algo == 4:  # Inward spiral
                if self.latest_center_px is None:
                    time.sleep(0.05); continue
                cx, cy = self.px_to_xy(self.latest_center_px[0], self.latest_center_px[1])
                # set initial radius to a fraction of FOV and shrink
                r0 = 0.5 * min(self.fov_x, self.fov_y) * 0.6
                # phase accumulator in radians stored locally
                if not hasattr(self, '_spir_phase'):
                    self._spir_phase = 0.0
                    self._spir_radius = r0
                self._spir_phase += math.radians(self.spir_rate_deg)
                self._spir_radius = max(0.0, self._spir_radius - self.spiral_dr)
                x = cx + self._spir_radius * math.cos(self._spir_phase)
                y = cy + self._spir_radius * math.sin(self._spir_phase)
                self._move_to_xy_blocking((x, y), timeout_s=5.0)
                continue

        self.get_logger().info("Exiting algorithm thread.")

    # ---------- Menu ----------
    def _menu_select(self) -> int:
        print("\nSelect suction algorithm:\n"
              " 1 - Global zigzag (lawnmower)\n"
              " 2 - Adaptive zigzag (bbox)\n"
              " 3 - Centroid follower\n"
              " 4 - Inward spiral (around centre)\n"
              " 5 - Selective raster sweep (rows with blood)\n")
        sel = 0
        while sel not in [1,2,3,4,5]:
            try:
                sel = int(input('Enter 1-5: ').strip())
            except Exception:
                sel = 0
        return sel

# ---------------------- Entrypoint ----------------------

def main(argv=None):
    rclpy.init(args=argv)
    node = BloodSuctionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

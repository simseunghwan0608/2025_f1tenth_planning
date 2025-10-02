#!/usr/bin/env python3
import numpy as np
import math
import os
from typing import Union
import scipy.spatial

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose, PoseArray, Point
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker, MarkerArray
from collections import deque


"""
Constant Definition
"""
WIDTH = 0.2032  # (m)
WHEEL_LENGTH = 0.0381  # (m)
MAX_STEER = 0.36  # (rad)


def safe_changeIdx(length, inp, plus):
    return (inp + plus + length) % (length)


class LaneFollow(Node):
    """
    Class for lane follow (Pure Pursuit steering)
    """

    def __init__(self):
        super().__init__("lane_follow_node")

        # ROS Params
        self.declare_parameter("visualize")

        self.declare_parameter("lane_occupied_dist")
        self.declare_parameter("obs_activate_dist")

        self.declare_parameter("real_test")
        self.declare_parameter("map_name")
        self.declare_parameter("num_lanes")
        self.declare_parameter("lane_files")
        self.declare_parameter("traj_file")

        self.declare_parameter("lookahead_distance")
        self.declare_parameter("lookahead_attenuation")
        self.declare_parameter("lookahead_idx")
        self.declare_parameter("lookbehind_idx")

        # Keep PID params declared for compatibility but they are unused in Pure Pursuit
        self.declare_parameter("kp_steer")
        self.declare_parameter("ki_steer")
        self.declare_parameter("kd_steer")
        self.declare_parameter("max_steer")
        self.declare_parameter("alpha_steer")

        self.declare_parameter("kp_pos")
        self.declare_parameter("ki_pos")
        self.declare_parameter("kd_pos")

        self.declare_parameter("follow_speed")
        self.declare_parameter("lane_dist_thresh")

        # interp
        self.declare_parameter('minL')
        self.declare_parameter('maxL')
        self.declare_parameter('minP')
        self.declare_parameter('maxP')
        self.declare_parameter('interpScale')
        self.declare_parameter('Pscale')
        self.declare_parameter('Lscale')
        self.declare_parameter('D')
        self.declare_parameter('vel_scale')

        self.declare_parameter('minL_corner')
        self.declare_parameter('maxL_corner')
        self.declare_parameter('minP_corner')
        self.declare_parameter('maxP_corner')
        self.declare_parameter('Pscale_corner')
        self.declare_parameter('Lscale_corner')

        self.declare_parameter('avoid_v_diff')
        self.declare_parameter('avoid_L_scale')
        self.declare_parameter('pred_v_buffer')
        self.declare_parameter('avoid_buffer')
        self.declare_parameter('avoid_span')

        # NEW: wheelbase for Pure Pursuit (default ~0.27 m for F1TENTH)
        self.declare_parameter('wheelbase', 0.27)
        self.wheelbase = self.get_parameter('wheelbase').get_parameter_value().double_value

        # Control state (PID fields kept but unused)
        self.prev_steer_error = 0.0
        self.steer_integral = 0.0
        self.prev_steer = 0.0
        self.prev_ditem = 0.0

        # Global Map Params
        self.real_test = self.get_parameter("real_test").get_parameter_value().bool_value
        self.map_name = self.get_parameter("map_name").get_parameter_value().string_value
        print(self.map_name)

        # Optimal Trajectory
        self.traj_file = self.get_parameter("traj_file").get_parameter_value().string_value
        traj_csv_loc = os.path.join("src", "lane_follow", "csv", self.map_name, self.traj_file + ".csv")
        print(f'read optimal raceline from {traj_csv_loc}')
        traj_data = np.loadtxt(traj_csv_loc, delimiter=';', skiprows=0)
        self.num_traj_pts = len(traj_data)
        self.traj_x = traj_data[:, 1]
        self.traj_y = traj_data[:, 2]
        self.traj_pos = np.vstack((self.traj_x, self.traj_y)).T
        self.traj_yaw = traj_data[:, 3]
        self.traj_v = traj_data[:, 5]
        print(f'length of traj_v{len(self.traj_v)}')

        # Lanes Waypoints
        self.num_lanes = self.get_parameter("num_lanes").get_parameter_value().integer_value
        self.lane_files = self.get_parameter("lane_files").get_parameter_value().string_array_value

        self.num_lane_pts = []
        self.lane_x = []
        self.lane_y = []
        self.lane_v = []
        self.lane_pos = []

        assert len(self.lane_files) == self.num_lanes
        for i in range(self.num_lanes):
            lane_csv_loc = os.path.join("src", "lane_follow", "csv", self.map_name, self.lane_files[i] + ".csv")
            lane_data = np.loadtxt(lane_csv_loc, delimiter=",")
            self.num_lane_pts.append(len(lane_data))
            self.lane_x.append(lane_data[:, 0])
            self.lane_y.append(lane_data[:, 1])
            self.lane_v.append(lane_data[:, 2])
            self.lane_pos.append(np.vstack((self.lane_x[-1], self.lane_y[-1]), ).T)
        print(f'length of last lane{len(self.lane_pos[-1])}')
        print(f'max v{np.max(self.lane_v[-1])}')

        overtaking_idx_csv_loc = os.path.join("src", "lane_follow", "csv", self.map_name, 'overtaking_wp_idx.npy')
        data = np.load(overtaking_idx_csv_loc, mmap_mode='r')
        # self.overtake_wpIdx = set(list(data))
        self.overtake_wpIdx = set(range(0, 70))
        print(self.overtake_wpIdx)

        slow_idx_csv_loc = os.path.join("src", "lane_follow", "csv", self.map_name, 'slowdown_wp_idx.npy')
        data2 = np.load(slow_idx_csv_loc, mmap_mode='r')
        self.slow_wpIdx = set(list(data2))
        print(self.slow_wpIdx)

        self.corner_wpIdx = set(list(range(0, 2)))
        self.fast_wpIdx = set(list(range(28, 65))) # 내가 추가

        # Car Status Variables
        self.lane_idx = 0
        self.curr_idx = None
        self.goal_idx = None
        self.curr_vel = 0.0
        self.target_point = None

        # Obstacle Variables
        self.obstacles = None
        self.opponent = np.array([np.inf, np.inf])
        self.lane_free = [True] * self.num_lanes
        self.declare_parameter('avoid_dist')
        self.opponent_v = 0.0
        self.opponent_last = np.array([0.0, 0.0])
        self.opponent_timestamp = 0.0
        self.pred_v_buffer = self.get_parameter('pred_v_buffer').get_parameter_value().integer_value
        self.pred_v_counter = 0
        self.avoid_buffer = self.get_parameter('avoid_buffer').get_parameter_value().integer_value
        self.avoid_counter = 0
        self.detect_oppo = False
        self.avoid_L_scale = self.get_parameter('avoid_L_scale').get_parameter_value().double_value

        # >>> PATCH: optimal lane index + logical index helper
        self.last_lane = -1  # keep using -1 as "optimal"
        self.opt_idx = self.num_lanes - 1  # last file is lane_optimal

        def _real_lane_idx(idx: int) -> int:
            return self.opt_idx if idx == -1 else idx
        self._real_lane_idx = _real_lane_idx
        # <<< PATCH

        # Topics & Subs, Pubs
        pose_topic = "/pf/viz/inferred_pose" if self.real_test else "/ego_racecar/odom"
        odom_topic = "/odom" if self.real_test else "/ego_racecar/odom"
        obstacle_topic = "/opp_predict/bbox"
        opponent_topic = "/opp_predict/state"
        drive_topic = "/drive"
        waypoint_topic = "/waypoint"

        if self.real_test:
            self.pose_sub_ = self.create_subscription(PoseStamped, pose_topic, self.pose_callback, 1)
        else:
            self.pose_sub_ = self.create_subscription(Odometry, pose_topic, self.pose_callback, 1)
        self.odom_sub_ = self.create_subscription(Odometry, odom_topic, self.odom_callback, 1)
        self.obstacle_sub_ = self.create_subscription(PoseArray, obstacle_topic, self.obstacle_callback, 1)
        self.opponent_sub_ = self.create_subscription(PoseStamped, opponent_topic, self.opponent_callback, 1)
        self.drive_pub_ = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
        self.waypoint_pub_ = self.create_publisher(Marker, waypoint_topic, 10)

        # >>> NEW: Pure Pursuit visualization publisher
        self.pp_vis_pub_ = self.create_publisher(MarkerArray, "/pure_pursuit/vis", 10)
        # <<<

        print('node_init_files')

    # >>> PATCH: safest-lane picker for N lanes
    def _pick_safest_lane(self, curr_target_point: np.ndarray, opponent_xy: np.ndarray):
        """
        Choose among free lanes:
         - maximize min distance to opponent on that lane
         - tie-break: minimize distance to current target_point
        Returns: lane index (real index, 0..num_lanes-1) or None
        """
        candidates = [i for i in range(self.num_lanes) if self.lane_free[i]]
        if not candidates:
            return None

        best = None
        best_score = -1e18
        for i in candidates:
            lane_pts = self.lane_pos[i][:, :2]
            # distance to opponent
            if opponent_xy is not None and not np.any(np.isinf(opponent_xy)):
                dmin_opp = np.min(np.linalg.norm(lane_pts - opponent_xy[None, :], axis=1))
            else:
                dmin_opp = 0.0
            # closeness to current target point
            d_to_target = np.min(np.linalg.norm(lane_pts - curr_target_point[None, :], axis=1))
            score = dmin_opp - 0.5 * d_to_target
            if score > best_score:
                best_score = score
                best = i
        return best
    # <<< PATCH

    def odom_callback(self, odom_msg: Odometry):
        self.curr_vel = odom_msg.twist.twist.linear.x

    def obstacle_callback(self, obstacle_msg: PoseArray):
        obstacle_list = []
        for obstacle in obstacle_msg.poses:
            x = obstacle.position.x
            y = obstacle.position.y
            obstacle_list.append([x, y])
        self.obstacles = np.array(obstacle_list) if obstacle_list else None

        if self.obstacles is None:
            self.lane_free = np.array([True] * self.num_lanes)
            return

        lane_occupied_dist = self.get_parameter("lane_occupied_dist").get_parameter_value().double_value
        for i in range(self.num_lanes):
            d = scipy.spatial.distance.cdist(self.lane_pos[i], self.obstacles)
            self.lane_free[i] = (np.min(d) > lane_occupied_dist)
        # print(f'lane_free_situation {self.lane_free}')

    def opponent_callback(self, opponent_msg: PoseStamped):
        opponent_x = opponent_msg.pose.position.x
        opponent_y = opponent_msg.pose.position.y
        self.opponent = np.array([opponent_x, opponent_y])

        ## velocity
        if not np.any(np.isinf(self.opponent)):
            if self.detect_oppo:
                oppoent_dist_diff = np.linalg.norm(self.opponent - self.opponent_last)
                if self.pred_v_counter == 7:
                    self.pred_v_counter = 0
                    cur_time = opponent_msg.header.stamp.nanosec/1e9 + opponent_msg.header.stamp.sec
                    time_interval = cur_time - self.opponent_timestamp
                    self.opponent_timestamp = cur_time
                    opponent_v = oppoent_dist_diff / max(time_interval, 0.005)
                    self.opponent_last = self.opponent.copy()
                    self.opponent_v = opponent_v
                    print(f'cur oppoent v {self.opponent_v}')
                else:
                    self.pred_v_counter += 1
            else:
                self.detect_oppo = True
                self.opponent_last = self.opponent.copy()
        else:
            self.detect_oppo = False

    def find_wp_target(self, L, traj_distances, curr_pos, curr_idx, lane_idx=None):
        interpScale = self.get_parameter('interpScale').get_parameter_value().integer_value
        segment_end = curr_idx
        while traj_distances[segment_end] <= L:
            segment_end = (segment_end + 1) % self.num_traj_pts
        segment_begin = safe_changeIdx(self.num_traj_pts, segment_end, -1)
        x_array = np.linspace(self.traj_x[segment_begin], self.traj_x[segment_end], interpScale)
        y_array = np.linspace(self.traj_y[segment_begin], self.traj_y[segment_end], interpScale)
        v_array = np.linspace(self.traj_v[segment_begin], self.traj_v[segment_end], interpScale)
        xy_interp = np.vstack([x_array, y_array]).T
        dist_interp = np.linalg.norm(xy_interp-curr_pos, axis=1) - L
        i_interp = np.argmin(np.abs(dist_interp))
        target_global = np.array([x_array[i_interp], y_array[i_interp]])
        target_v = v_array[i_interp]
        L = np.linalg.norm(curr_pos - target_global)
        target_point = np.array([x_array[i_interp], y_array[i_interp]])
        return target_point, target_v

    def find_interp_point(self, L, begin, target):
        interpScale = self.get_parameter('interpScale').get_parameter_value().integer_value
        x_array = np.linspace(begin[0], target[0], interpScale)
        y_array = np.linspace(begin[1], target[1], interpScale)
        xy_interp = np.vstack([x_array, y_array]).T
        dist_interp = np.linalg.norm(xy_interp-target, axis=1) - L
        i_interp = np.argmin(np.abs(dist_interp))
        interp_point = np.array([x_array[i_interp], y_array[i_interp]])
        return interp_point

    def avoid_static(self):
        pass

    def avoid_dynamic(self):
        pass

    def pose_callback(self, pose_msg: Union[PoseStamped, Odometry]):
        """
        The pose callback when subscribed to particle filter's inferred pose
        """

        cur_speed = self.curr_vel
        print(f'cur_speed{cur_speed}')

        #### Read pose data ####
        if self.real_test:
            curr_x = pose_msg.pose.position.x
            curr_y = pose_msg.pose.position.y
            curr_pos = np.array([curr_x, curr_y])
            curr_quat = pose_msg.pose.orientation
        else:
            curr_x = pose_msg.pose.pose.position.x
            curr_y = pose_msg.pose.pose.position.y
            curr_pos = np.array([curr_x, curr_y])
            curr_quat = pose_msg.pose.pose.orientation

        curr_yaw = math.atan2(2 * (curr_quat.w * curr_quat.z + curr_quat.x * curr_quat.y),
                              1 - 2 * (curr_quat.y ** 2 + curr_quat.z ** 2))

        #### use optimal traj to get curr_pose_idx ####
        curr_pos_idx = np.argmin(np.linalg.norm(self.lane_pos[-1][:, :2] - curr_pos, axis=1))
        print(f'curr pos idx{curr_pos_idx}')
        if curr_pos_idx in self.fast_wpIdx:
            fast_factor = 1.5
        else:
            fast_factor = 1.0
        if curr_pos_idx in self.slow_wpIdx:
            slow_factor = 0.7375
        else:
            slow_factor = 1.0

        #### switch back to optimal raceline (generalized) ####
        if self.lane_free[self.opt_idx]:
            self.avoid_counter = min(self.avoid_buffer, self.avoid_counter + 1)
        else:
            self.avoid_counter = 0
        if self.avoid_counter == self.avoid_buffer and self.lane_free[self.opt_idx]:
            self.last_lane = -1
        if curr_pos_idx not in self.overtake_wpIdx:
            self.last_lane = -1
        #### end switch-back ####

        #### interp for finding target (generalized) ####
        lane_idx_real = self._real_lane_idx(self.last_lane)
        curr_lane_nearest_idx = np.argmin(
            np.linalg.norm(self.lane_pos[lane_idx_real][:, :2] - curr_pos, axis=1)
        )
        traj_distances = np.linalg.norm(
            self.lane_pos[lane_idx_real][:, :2] - self.lane_pos[lane_idx_real][curr_lane_nearest_idx, :2], axis=1
        )
        segment_end = np.argmin(traj_distances)
        num_lane_pts = len(self.lane_pos[lane_idx_real])
        if curr_pos_idx in self.corner_wpIdx:
            L = self.get_L_w_speed(cur_speed, corner=True)
        else:
            L = self.get_L_w_speed(cur_speed)
        interpScale = self.get_parameter('interpScale').get_parameter_value().integer_value
        while traj_distances[segment_end] <= L:
            segment_end = (segment_end + 1) % num_lane_pts
        segment_begin = (segment_end - 1 + num_lane_pts) % num_lane_pts

        x_array = np.linspace(self.lane_x[lane_idx_real][segment_begin], self.lane_x[lane_idx_real][segment_end], interpScale)
        y_array = np.linspace(self.lane_y[lane_idx_real][segment_begin], self.lane_y[lane_idx_real][segment_end], interpScale)
        v_array = np.linspace(self.lane_v[lane_idx_real][segment_begin], self.lane_v[lane_idx_real][segment_end], interpScale)
        xy_interp = np.vstack([x_array, y_array]).T
        dist_interp = np.linalg.norm(xy_interp-curr_pos, axis=1) - L
        i_interp = np.argmin(np.abs(dist_interp))
        target_global = np.array([x_array[i_interp], y_array[i_interp]])
        if self.last_lane == -1:
            target_v = v_array[i_interp]
        else:
            target_v = self.lane_v[self.opt_idx][curr_pos_idx] * 0.6  # conservative when not on optimal
            print(f'target_V {target_v}')
        self.target_point = np.array([x_array[i_interp], y_array[i_interp]])
        #### end interp ####

        # avoidance params
        avoid_dist = self.get_parameter('avoid_dist').get_parameter_value().double_value
        avoid_dist = avoid_dist * max(cur_speed, 3.0)
        avoid_v_diff = self.get_parameter('avoid_v_diff').get_parameter_value().double_value
        avoid_span = self.get_parameter('avoid_span').get_parameter_value().double_value

        # obstacle logic (generalized)
        if not np.any(np.isinf(self.opponent)):
            cur_obs_dist = np.linalg.norm(self.opponent - curr_pos)
            v_diff = self.curr_vel - self.opponent_v
            print(f'opponent_v: {self.opponent_v}')
            print(f'cur_obs_distance: {cur_obs_dist}')
            print(f'last_lane {self.last_lane}')
            print(f'lane free {self.lane_free}')

            if not np.any(self.lane_free):
                target_v = max(self.opponent_v * 0.8, 0.0)
            else:
                if self.detect_oppo and cur_obs_dist <= avoid_dist and not self.lane_free[self._real_lane_idx(self.last_lane)]:
                    print('obs_detected')
                    if curr_pos_idx in self.overtake_wpIdx and any(self.lane_free[i] for i in range(self.num_lanes)):
                        print('overtake (generalized)')
                        safest = self._pick_safest_lane(self.target_point, self.opponent)
                        if safest is not None and safest != self._real_lane_idx(self.last_lane):
                            # reflect choice into last_lane, keeping -1 for optimal
                            self.last_lane = -1 if safest == self.opt_idx else safest
                            dist_to_lane = np.linalg.norm(self.lane_pos[safest][:, :2] - self.target_point, axis=1)
                            min_idx = np.argmin(dist_to_lane)
                            lane_target = self.lane_pos[safest][min_idx]
                            self.target_point = lane_target
                    else:
                        # follow or slow depending on opponent speed
                        if self.opponent_v < 1.0:
                            target_v = max(self.opponent_v * 0.8, 1.0)
                        else:
                            target_v = max(self.opponent_v * 0.8, 0.0)
        # (else: target_v already set from lane speed)

        # body frame transform
        R = np.array([[np.cos(curr_yaw), np.sin(curr_yaw)],
                      [-np.sin(curr_yaw), np.cos(curr_yaw)]])
        target_x, target_y = R @ np.array([self.target_point[0] - curr_x,
                                           self.target_point[1] - curr_y])

        # speed
        vel_scale = self.get_parameter('vel_scale').get_parameter_value().double_value
        speed = target_v * vel_scale * slow_factor * fast_factor

        # PURE PURSUIT STEERING
        # Ld: euclidean distance to target point in body frame (lookahead distance actually used)
        Ld = max(1e-3, float(np.hypot(target_x, target_y)))
        alpha = math.atan2(target_y, target_x)  # angle to the target in body frame
        steer = math.atan2(2.0 * self.wheelbase * math.sin(alpha), Ld)

        # clip to vehicle limits
        max_control = self.get_parameter("max_steer").get_parameter_value().double_value
        steer = float(np.clip(steer, -max_control, max_control))

        # publish drive
        message = AckermannDriveStamped()
        message.drive.speed = speed * 0.7
        message.drive.steering_angle = steer
        self.drive_pub_.publish(message)

        # visualize (legacy target + pure-pursuit overlays)
        visualize = self.get_parameter("visualize").get_parameter_value().bool_value
        if visualize:
            self.visualize_target()
            self.visualize_pure_pursuit(curr_pos, curr_yaw, self.target_point, Ld, steer, speed)

        return None

    def get_lookahead_dist(self, curr_idx):
        L = self.get_parameter("lookahead_distance").get_parameter_value().double_value
        lookahead_idx = self.get_parameter("lookahead_idx").get_parameter_value().integer_value
        lookbehind_idx = self.get_parameter("lookbehind_idx").get_parameter_value().integer_value
        slope = self.get_parameter("lookahead_attenuation").get_parameter_value().double_value

        yaw_before = self.traj_yaw[(curr_idx - lookbehind_idx) % self.num_traj_pts]
        yaw_after = self.traj_yaw[(curr_idx + lookahead_idx) % self.num_traj_pts]
        yaw_diff = abs(yaw_after - yaw_before)
        if yaw_diff > np.pi:
            yaw_diff = yaw_diff - 2 * np.pi
        if yaw_diff < -np.pi:
            yaw_diff = yaw_diff + 2 * np.pi
        yaw_diff = abs(yaw_diff)
        if yaw_diff > np.pi / 2:
            yaw_diff = np.pi / 2
        L = max(0.5, L * (np.pi / 2 - yaw_diff * slope) / (np.pi / 2))
        return L

    # PID helpers retained for compatibility (unused in pure pursuit path)
    def get_steer(self, error):
        kp = self.get_parameter("kp_steer").get_parameter_value().double_value
        ki = self.get_parameter("ki_steer").get_parameter_value().double_value
        kd = self.get_parameter("kd_steer").get_parameter_value().double_value
        max_control = self.get_parameter("max_steer").get_parameter_value().double_value
        alpha = self.get_parameter("alpha_steer").get_parameter_value().double_value

        d_error = error - self.prev_steer_error
        self.prev_steer_error = error
        self.steer_integral += error
        steer = kp * error + ki * self.steer_integral + kd * d_error
        new_steer = np.clip(steer, -max_control, max_control)
        new_steer = alpha * new_steer + (1 - alpha) * self.prev_steer
        self.prev_steer = new_steer
        return new_steer

    def visualize_target(self):
        marker = Marker()
        marker.header.frame_id = "/map"
        marker.id = 0
        marker.ns = "target_waypoint"
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(self.target_point[0])
        marker.pose.position.y = float(self.target_point[1])

        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        this_scale = 0.2
        marker.scale.x = this_scale
        marker.scale.y = this_scale
        marker.scale.z = this_scale
        marker.pose.orientation.w = 1.0
        marker.lifetime.nanosec = int(1e8)
        self.waypoint_pub_.publish(marker)

    # >>> NEW: Pure Pursuit visualization bundle
    def visualize_pure_pursuit(self, curr_pos, curr_yaw, target_point, Ld, steer, speed):
        """
        Publishes to /pure_pursuit/vis:
          - id=10: target point (SPHERE, red)
          - id=11: lookahead circle centered at current position (LINE_STRIP)
          - id=12: current heading arrow (ARROW)
          - id=13: line current->target (LINE_LIST)
          - id=14: text with Ld / steer / speed (TEXT_VIEW_FACING)
        """
        ma = MarkerArray()
        life_ns = int(2e8)  # 0.2s

        def setup(ns: str, mid: int, mtype: int):
            m = Marker()
            m.header.frame_id = "/map"
            m.ns = ns
            m.id = mid
            m.type = mtype
            m.action = Marker.ADD
            m.lifetime.nanosec = life_ns
            return m

        # 10) Target sphere (redundant with visualize_target, but grouped here too)
        m_tgt = setup("pp/target", 10, Marker.SPHERE)
        m_tgt.pose.position.x = float(target_point[0])
        m_tgt.pose.position.y = float(target_point[1])
        m_tgt.pose.orientation.w = 1.0
        m_tgt.scale.x = m_tgt.scale.y = m_tgt.scale.z = 0.18
        m_tgt.color.a = 1.0; m_tgt.color.r = 1.0; m_tgt.color.g = 0.0; m_tgt.color.b = 0.0
        ma.markers.append(m_tgt)

        # 11) Lookahead circle (LINE_STRIP)
        m_circ = setup("pp/lookahead_circle", 11, Marker.LINE_STRIP)
        m_circ.scale.x = 0.03
        m_circ.color.a = 1.0; m_circ.color.r = 1.0; m_circ.color.g = 0.5; m_circ.color.b = 0.0
        cx, cy = float(curr_pos[0]), float(curr_pos[1])
        num = 64
        for k in range(num + 1):
            th = 2.0 * math.pi * k / num
            px = cx + Ld * math.cos(th)
            py = cy + Ld * math.sin(th)
            m_circ.points.append(Point(x=px, y=py, z=0.0))
        ma.markers.append(m_circ)

        # 12) Heading arrow (ARROW) as two points
        m_arrow = setup("pp/heading", 12, Marker.ARROW)
        m_arrow.scale.x = 0.35  # shaft diameter
        m_arrow.scale.y = 0.5   # head diameter
        m_arrow.scale.z = 0.2   # head length
        m_arrow.color.a = 1.0; m_arrow.color.r = 0.0; m_arrow.color.g = 0.7; m_arrow.color.b = 1.0
        start = Point(x=cx, y=cy, z=0.02)
        end = Point(x=cx + 0.6 * math.cos(curr_yaw), y=cy + 0.6 * math.sin(curr_yaw), z=0.02)
        m_arrow.points = [start, end]
        ma.markers.append(m_arrow)

        # 13) Line from car to target (LINE_LIST)
        m_line = setup("pp/line_to_target", 13, Marker.LINE_LIST)
        m_line.scale.x = 0.035
        m_line.color.a = 1.0; m_line.color.r = 0.2; m_line.color.g = 1.0; m_line.color.b = 0.2
        m_line.points = [Point(x=cx, y=cy, z=0.01), Point(x=float(target_point[0]), y=float(target_point[1]), z=0.01)]
        ma.markers.append(m_line)

        # 14) Text (Ld / steer / speed)
        m_txt = setup("pp/text", 14, Marker.TEXT_VIEW_FACING)
        m_txt.pose.position.x = cx
        m_txt.pose.position.y = cy + 0.4
        m_txt.pose.position.z = 0.4
        m_txt.scale.z = 0.25
        m_txt.color.a = 1.0; m_txt.color.r = 1.0; m_txt.color.g = 1.0; m_txt.color.b = 1.0
        steer_deg = math.degrees(steer)
        m_txt.text = f"Ld={Ld:.2f} m | steer={steer_deg:.1f}° | v={speed:.2f} m/s"
        ma.markers.append(m_txt)

        self.pp_vis_pub_.publish(ma)
    # <<<

    def get_L_w_speed(self, speed, corner=False):
        if corner:
            maxL = self.get_parameter('maxL_corner').get_parameter_value().double_value
            minL = self.get_parameter('minL_corner').get_parameter_value().double_value
            Lscale = self.get_parameter('Lscale_corner').get_parameter_value().double_value
        else:
            maxL = self.get_parameter('maxL').get_parameter_value().double_value
            minL = self.get_parameter('minL').get_parameter_value().double_value
            Lscale = self.get_parameter('Lscale').get_parameter_value().double_value
        interp_L_scale = (maxL-minL) / Lscale
        return interp_L_scale * speed + minL

    def get_steer_w_speed(self, speed, error, corner=False):
        # Kept for backward compatibility; not used after Pure Pursuit change
        if corner:
            maxP = self.get_parameter('maxP_corner').get_parameter_value().double_value
            minP = self.get_parameter('minP_corner').get_parameter_value().double_value
            Pscale = self.get_parameter('Pscale_corner').get_parameter_value().double_value
        else:
            maxP = self.get_parameter('maxP').get_parameter_value().double_value
            minP = self.get_parameter('minP').get_parameter_value().double_value
            Pscale = self.get_parameter('Pscale').get_parameter_value().double_value

        interp_P_scale = (maxP-minP) / Pscale
        cur_P = maxP - speed * interp_P_scale
        max_control = self.get_parameter("max_steer").get_parameter_value().double_value
        kd = self.get_parameter('D').get_parameter_value().double_value

        d_error = error - self.prev_steer_error
        if not self.real_test:
            if d_error == 0:
                d_error = self.prev_ditem
            else:
                self.prev_ditem = d_error
                self.prev_steer_error = error
        else:
            self.prev_ditem = d_error
            self.prev_steer_error = error
        if corner:
            steer = cur_P * error
        else:
            steer = cur_P * error + kd * d_error
        new_steer = np.clip(steer, -max_control, max_control)
        return new_steer


def main(args=None):
    rclpy.init(args=args)
    print("Lane Follow Initialized (Pure Pursuit)")
    lane_follow_node = LaneFollow()
    rclpy.spin(lane_follow_node)
    lane_follow_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

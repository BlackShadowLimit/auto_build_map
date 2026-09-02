#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan


class AutoGroundCalibrator:
    def __init__(self, calib_frames: int = 25):
        self.target_frames = calib_frames
        self.collected_samples = []
        self.is_calibrated = False
        self.lower_hsv = np.array([0, 0, 0], dtype=np.uint8)
        self.upper_hsv = np.array([180, 255, 255], dtype=np.uint8)

    def step(self, bgr_frame: np.ndarray, logger) -> bool:
        if self.is_calibrated:
            return True

        h, w, _ = bgr_frame.shape
        # 擷取車頭正下方 15% 處（絕對安全的地板信任區）
        patch = bgr_frame[int(h * 0.85):h, int(w * 0.35):int(w * 0.65)]
        hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        self.collected_samples.append(hsv_patch)

        if len(self.collected_samples) >= self.target_frames:
            all_pixels = np.concatenate(self.collected_samples, axis=0).reshape(-1, 3)
            means = np.mean(all_pixels, axis=0)
            stds = np.std(all_pixels, axis=0)

            k = 2.0
            lower = means - k * stds
            upper = means + k * stds

            # 收緊 HSV 容忍度，避免把灰藍色相近的物體誤判為地板
            self.lower_hsv = np.array([
                np.clip(lower[0] - 10, 0, 180),
                np.clip(lower[1] - 20, 0, 255),
                np.clip(lower[2] - 15, 0, 255)
            ], dtype=np.uint8)

            self.upper_hsv = np.array([
                np.clip(upper[0] + 10, 0, 180),
                np.clip(upper[1] + 20, 0, 255),
                np.clip(upper[2] + 15, 0, 255)
            ], dtype=np.uint8)

            self.is_calibrated = True
            logger.info("=== 地板自動校準完成 ===")
            logger.info(f"HSV Lower: {self.lower_hsv.tolist()}")
            logger.info(f"HSV Upper: {self.upper_hsv.tolist()}")

        return self.is_calibrated


class GroundScannerNode(Node):
    def __init__(self):
        super().__init__('ground_scanner_node')
        self.bridge = CvBridge()
        self.calibrator = AutoGroundCalibrator(calib_frames=25)

        # 幾何參數（相機下傾 45 度）
        self.cam_height = 0.13   # 安裝離地高度 (m)
        self.pitch = 0.785398    # 45 度 (rad)
        self.hfov = 1.085        # 水平視角 (rad)
        self.num_readings = 50   # 虛擬雷達射線數量

        self.sub = self.create_subscription(Image, '/camera/image_raw', self._on_image, 10)
        self.scan_pub = self.create_publisher(LaserScan, '/camera_scan', 10)
        self.get_logger().info("GroundScannerNode 已就緒，等待相機影像並進行地面校準...")

    def _on_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge 轉換失敗: {e}")
            return

        # 進行開機自動採樣
        if not self.calibrator.step(frame, self.get_logger()):
            return

        h, w, _ = frame.shape
        
        # --- 防線一：精準 HSV 色彩遮罩 ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        floor_mask = cv2.inRange(hsv, self.calibrator.lower_hsv, self.calibrator.upper_hsv)
        
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, kernel_small)
        color_obstacle_mask = cv2.bitwise_not(floor_mask)

        # --- 防線二：Canny 幾何結構與邊緣檢測（解決同色系立體底座盲區） ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 5, 50, 50)
        edges = cv2.Canny(blur, 40, 100)
        
        edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges_dilated = cv2.dilate(edges, edge_kernel)

        # --- 雙重融合：非地板顏色 OR 有明顯立體結構線條，皆視為障礙物 ---
        combined_obstacle_mask = cv2.bitwise_or(color_obstacle_mask, edges_dilated)

        # 膨脹填滿，確保物體內部與邊緣連成紮實區塊
        fill_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        obstacle_mask = cv2.dilate(combined_obstacle_mask, fill_kernel)

        # 封裝 LaserScan
        scan = LaserScan()
        scan.header.stamp = msg.header.stamp
        scan.header.frame_id = "camera_link"
        scan.angle_min = -self.hfov / 2.0
        scan.angle_max = self.hfov / 2.0
        scan.angle_increment = self.hfov / self.num_readings
        scan.range_min = 0.10
        scan.range_max = 1.80

        ranges = []
        col_step = w // self.num_readings
        vfov = self.hfov * (h / w)

        for i in range(self.num_readings):
            strip = obstacle_mask[:, i * col_step:(i + 1) * col_step]
            y_indices = np.where(strip > 0)[0]

            if len(y_indices) > 0:
                # 找縱向最靠近底部 (v 最大，距離最近) 的障礙點
                v = np.max(y_indices)
                alpha = ((v - (h / 2.0)) / (h / 2.0)) * (vfov / 2.0)
                total_pitch = self.pitch + alpha

                # 透過三角函數反推水平距離
                if total_pitch > 0.05:
                    dist = self.cam_height / math.tan(total_pitch)
                    ranges.append(float(np.clip(dist, scan.range_min, scan.range_max)))
                else:
                    ranges.append(float('inf'))
            else:
                ranges.append(float('inf'))

        scan.ranges = ranges
        self.scan_pub.publish(scan)


def main(args=None):
    rclpy.init(args=args)
    node = GroundScannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan


class GroundScannerNode(Node):
    def __init__(self):
        super().__init__('ground_scanner_node')
        self.bridge = CvBridge()

        # 幾何參數（實體車安裝高度與俯角）
        self.cam_height = 0.10   # 相機離地高度 (m)
        self.pitch = 0.52        # 約 30 度俯角 (rad)
        self.hfov = 2.09         # 魚眼鏡頭視野較廣，設為約 120 度有效視角 (rad)
        self.num_readings = 60   # 射線數量

        self.sub = self.create_subscription(Image, '/camera/image_raw', self._on_image, 10)
        self.scan_pub = self.create_publisher(LaserScan, '/camera_scan', 10)
        self.get_logger().info("實體魚眼 GroundScannerNode 已就緒...")

    def _on_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge 轉換失敗: {e}")
            return

        h, w, _ = frame.shape
        
        # 1. 消除魚眼黑邊遮罩（圓形遮罩，遮掉外圍黑框）
        mask_circle = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        radius = int(min(h, w) * 0.48)
        cv2.circle(mask_circle, center, radius, 255, -1)

        # 2. 轉灰階並執行重度模糊（抹平磨石子斑點與反光噪點）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)  # 保留大障礙物邊界，平滑碎石紋理

        # 3. 邊緣檢測（提高門檻，忽略細微磨石子紋理）
        edges = cv2.Canny(blurred, 60, 160)
        edges = cv2.bitwise_and(edges, edges, mask=mask_circle)

        # 4. 只關注地板有效掃描區域（畫面下半部，避開天花板與日光燈）
        roi_mask = np.zeros_like(edges)
        roi_mask[int(h * 0.45):int(h * 0.90), :] = 255
        obstacle_edges = cv2.bitwise_and(edges, roi_mask)

        # 5. 形態學膨脹：將桌腳、電線、插座連成可檢測實體
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        obstacle_mask = cv2.dilate(obstacle_edges, kernel)

        # 6. 封裝 LaserScan
        scan = LaserScan()
        scan.header.stamp = msg.header.stamp
        scan.header.frame_id = "camera_link"
        scan.angle_min = -self.hfov / 2.0
        scan.angle_max = self.hfov / 2.0
        scan.angle_increment = self.hfov / self.num_readings
        scan.range_min = 0.10
        scan.range_max = 2.00

        ranges = []
        col_step = w // self.num_readings
        vfov = self.hfov * (h / w)

        for i in range(self.num_readings):
            strip = obstacle_mask[:, i * col_step:(i + 1) * col_step]
            y_indices = np.where(strip > 0)[0]

            if len(y_indices) > 0:
                v = np.max(y_indices)
                # 近似魚眼鏡頭的弧度投影反推
                alpha = ((v - (h / 2.0)) / (h / 2.0)) * (vfov / 2.0)
                total_pitch = self.pitch + alpha

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

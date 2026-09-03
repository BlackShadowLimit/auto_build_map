#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, LaserScan

class GroundScannerNode(Node):
    def __init__(self):
        super().__init__('ground_scanner_node')
        self.bridge = CvBridge()

        # --- 使用者提供的額外資訊 ---
        self.cam_height = 0.15   # 相機離地高度 15 cm (0.15 m)
        # 地磚尺寸: 300mm x 300mm (0.3m x 0.3m)
        # 已知: 畫面正下方 (y = height) 對應距離為 0m (鏡頭正下方)
        
        self.hfov = 2.09         # 水平視角 (約120度，與原掃描節點一致)
        self.num_readings = 60   # 輸出的雷射射線數量
        self.max_detect_dist = 2.0

        # 建立影像訂閱與虛擬雷射發布
        self.sub = self.create_subscription(CompressedImage, '/camera/image_raw/compressed', self._on_image, 10)
        self.scan_pub = self.create_publisher(LaserScan, '/camera_scan', 10)
        
        self.get_logger().info("GroundScannerNode (針對磨石子地磚優化版) 已就緒...")

    def pixel_to_distance(self, y, h):
        """
        將影像 y 座標轉換為前方物理距離 (m)。
        已知：畫面最底部 (y = h) 是鏡頭正下方 (距離 0m)。
        未來若需更精確距離，可利用 30cm 地磚網格的像素座標進行多項式曲線擬合 (Polynomial Fitting)。
        目前採用基於相機高度與垂直視角的三角幾何推導。
        """
        if y >= h - 1:
            return 0.0
        
        # 假設下半部畫面的垂直視角分佈為 90 度 (從正下方的 -90 度到水平的 0 度)
        vfov_bottom_half = 1.5708 
        ratio = (h - y) / (h / 2.0)  # y 在下半部時的比例 (0 ~ 1)
        
        if ratio >= 1.0: # 超過畫面下半部，視為無限遠
            return float('inf')
            
        angle_from_vertical = ratio * vfov_bottom_half
        
        if angle_from_vertical < 0.01:
            return 0.0
            
        # 距離 = 相機高度 * tan(與垂直線的夾角)
        dist = self.cam_height * math.tan(angle_from_vertical)
        return float(np.clip(dist, 0.0, self.max_detect_dist))

    def _on_image(self, msg: CompressedImage):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge 轉換失敗: {e}")
            return

        h, w, _ = frame.shape
        
        # 1. 魚眼邊界遮罩 (過濾圓形以外的無效黑邊)
        mask_circle = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        radius = int(min(h, w) * 0.48)
        cv2.circle(mask_circle, center, radius, 255, -1)

        # 2. 動態擷取「鏡頭正下方」的地板顏色 (畫面最底中央)
        # 因為已知最下方必定是安全地面 (若無障礙物擋住鏡頭)
        ref_roi = frame[h-30:h-5, w//2-30:w//2+30]
        ref_hsv = cv2.cvtColor(ref_roi, cv2.COLOR_BGR2HSV)
        
        median_h = np.median(ref_hsv[:, :, 0])
        median_s = np.median(ref_hsv[:, :, 1])
        median_v = np.median(ref_hsv[:, :, 2])

        # 3. HSV 色彩空間分割
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 設定針對磨石子地磚的動態寬容度 (容許斑點的亮暗變化)
        # 放寬亮度的下限，容許更暗的接縫 (從 -70 改為 -120)
        lower_bound = np.array([max(0, median_h - 25), max(0, median_s - 40), max(0, median_v - 120)])
        upper_bound = np.array([min(179, median_h + 25), min(255, median_s + 60), min(255, median_v + 60)])
        
        # 產生「是地板」的二值化遮罩
        floor_mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

        # === 新增：建立紫紅色干擾遮罩 ===
        lower_purple = np.array([135, 50, 50])
        upper_purple = np.array([165, 255, 255])
        purple_mask = cv2.inRange(hsv_frame, lower_purple, upper_purple)

        # 將紫紅色遮罩「聯集 (OR)」加入地板遮罩中，強迫程式將紫色視為安全區域
        floor_mask = cv2.bitwise_or(floor_mask, purple_mask)
        
        # 最後再套用魚眼邊界遮罩，濾除圓形外的黑邊
        floor_mask = cv2.bitwise_and(floor_mask, mask_circle)

        # 4. 形態學處理：消除磨石子黑斑造成的偽障礙物破洞
        # 放大 kernel 尺寸 (11x11 改為 21x21)，以跨越/填補磁磚接縫
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        # 閉運算：將地板中的磁磚接縫、黑洞(斑點)填滿 (增加 iterations=3)
        floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)
        # 開運算：消除散落的雜訊
        floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

        # 5. 生成 LaserScan 掃描資料
        scan = LaserScan()
        scan.header.stamp = msg.header.stamp
        scan.header.frame_id = "camera_link"
        scan.angle_min = -self.hfov / 2.0
        scan.angle_max = self.hfov / 2.0
        scan.angle_increment = self.hfov / self.num_readings
        scan.range_min = 0.05
        scan.range_max = self.max_detect_dist

        ranges = []
        col_step = w // self.num_readings

        for i in range(self.num_readings):
            strip = floor_mask[:, i * col_step:(i + 1) * col_step]
            col_profile = np.max(strip, axis=1) # 將該區塊水平壓縮成一條線
            
            # 從畫面底部 (離車體最近) 往上 (遠處) 尋找第一個「非地板 (0)」的像素
            obstacle_y = -1
            for y in range(h - 1, h // 2, -1):
                if col_profile[y] == 0:
                    obstacle_y = y
                    break
                    
            if obstacle_y != -1:
                dist = self.pixel_to_distance(obstacle_y, h)
                ranges.append(dist)
            else:
                ranges.append(float('inf'))

        scan.ranges = ranges

        valid_ranges = [r for r in ranges if r < float('inf')]
        if valid_ranges:
            min_dist = min(valid_ranges)
            self.get_logger().info(f"偵測到障礙物，最近距離: {min_dist:.2f} 公尺")

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

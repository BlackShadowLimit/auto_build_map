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

        # --- 使用者提供的額外資訊 ---
        self.cam_height = 0.15   # 相機離地高度 15 cm (0.15 m)
        
        self.hfov = 2.09         # 水平視角 (約120度，與原掃描節點一致)
        self.num_readings = 60   # 輸出的雷射射線數量
        self.max_detect_dist = 2.0

        # 建立影像訂閱與虛擬雷射發布
        # 修改：為了在樹莓派上避免壓縮雜訊，直接訂閱無損的原生 Image
        self.sub = self.create_subscription(Image, '/camera/image_raw', self._on_image, 10)
        self.scan_pub = self.create_publisher(LaserScan, '/camera_scan', 10)
        
        self.get_logger().info("GroundScannerNode (樹莓派無損影像優化版) 已就緒...")

    def pixel_to_distance(self, y, h):
        """
        將影像 y 座標轉換為前方物理距離 (m)。
        前向相機模型：畫面正中心 (h/2) 為地平線 (無限遠)。
        """
        pixel_dy = y - (h / 2.0)
        
        # 如果像素在畫面上半部或正中心，代表看向上方或地平線，距離無限遠
        if pixel_dy <= 0:
            return float('inf')
            
        # 假設垂直視角為 90 度 (vfov = 1.5708 rad)，從中心到最底部的角度為 vfov/2
        vfov = 1.5708
        angle_down_from_horizon = (pixel_dy / (h / 2.0)) * (vfov / 2.0)
        
        # 距離 = 相機高度 / tan(俯角)
        dist = self.cam_height / math.tan(angle_down_from_horizon)
        return float(np.clip(dist, 0.0, self.max_detect_dist))

    def _on_image(self, msg: Image):
        try:
            # 修改：使用 imgmsg_to_cv2 轉換 Raw Image
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge 轉換失敗: {e}")
            return

        h, w, _ = frame.shape
        
        # 1. 魚眼邊界遮罩 (過濾圓形以外的無效黑邊)
        mask_circle = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        radius = int(min(h, w) * 0.48)
        cv2.circle(mask_circle, center, radius, 255, -1)
        
        # --- 切除畫面最底部（車體陰影區與黑邊） ---
        bottom_crop_y = int(h / 2 + radius) - 60  # 擴大裁切範圍，徹底避開邊緣 0.05m 的雜訊
        if bottom_crop_y < h:
            mask_circle[bottom_crop_y:, :] = 0

        # 2. 動態擷取「鏡頭正下方」的地板顏色 (避開被切除的底部陰影區)
        ref_y_end = bottom_crop_y - 5
        ref_y_start = ref_y_end - 25
        ref_roi = frame[ref_y_start:ref_y_end, w//2-30:w//2+30]
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
        # 大幅放寬紫色的認定範圍，涵蓋所有偏紫/粉紅的顏色
        lower_purple = np.array([120, 30, 30])
        upper_purple = np.array([175, 255, 255])
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
        scan.header.frame_id = "base_footprint"  # 修改為水平座標系，避免因 camera_link 傾斜 45 度導致訊號被當成地板以下而丟棄
        scan.angle_min = -self.hfov / 2.0
        scan.angle_max = self.hfov / 2.0
        scan.angle_increment = self.hfov / self.num_readings
        scan.range_min = 0.10  # 忽略 10 公分以內的盲區 (車體邊緣/陰影)
        scan.range_max = self.max_detect_dist

        ranges = []
        col_step = w // self.num_readings

        for i in range(self.num_readings):
            strip = floor_mask[:, i * col_step:(i + 1) * col_step]
            col_profile = np.max(strip, axis=1) # 將該區塊水平壓縮成一條線
            
            strip_valid = mask_circle[:, i * col_step:(i + 1) * col_step]
            col_valid = np.max(strip_valid, axis=1) # 檢查是否在有效視角內
            
            # 從畫面底部 (離車體最近) 往上 (遠處) 尋找第一個「非地板 (0)」的像素
            obstacle_y = -1
            for y in range(h - 1, h // 2, -1):
                # 必須在有效範圍內 (col_valid > 0) 且偵測到非地板 (col_profile == 0)
                if col_valid[y] > 0 and col_profile[y] == 0:
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

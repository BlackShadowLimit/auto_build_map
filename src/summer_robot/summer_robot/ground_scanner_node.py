#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, CompressedImage
import threading

class GroundScannerNode(Node):
    def __init__(self):
        super().__init__('ground_scanner_node')
        
        # --- 使用者提供的額外資訊 ---
        self.cam_height = 0.15   # 相機離地高度 15 cm (0.15 m)
        self.hfov = 2.09         # 水平視角 (約120度)
        self.num_readings = 60   # 輸出的雷射射線數量
        self.max_detect_dist = 2.0

        # 直接發布 LaserScan
        self.scan_pub = self.create_publisher(LaserScan, '/camera_scan', 10)
        # 發布壓縮過的 Debug 影像，方便在 PC 端監控與除錯 (低頻寬消耗)
        self.debug_pub = self.create_publisher(CompressedImage, '/camera_debug/compressed', 2)
        
        # 開啟硬體攝影機 (強制使用 V4L2 避免 GStreamer 錯誤)
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            self.get_logger().error("無法開啟硬體相機 /dev/video0！")
        else:
            self.get_logger().info("GroundScannerNode (硬體直讀極速版) 已就緒...")
            # 開啟背景執行緒不斷擷取影像，避免阻塞 ROS spin
            self.capture_thread = threading.Thread(target=self._capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()

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

    def _capture_loop(self):
        fail_count = 0
        while rclpy.ok():
            ret, frame = self.cap.read()
            if not ret:
                fail_count += 1
                if fail_count % 30 == 0:
                    self.get_logger().error(f"相機讀取失敗！(已失敗 {fail_count} 次)，請確認 /dev/video0 是否被佔用。")
                continue
            
            fail_count = 0    
            # 建立假的 Header 提供給 LaserScan
            stamp = self.get_clock().now().to_msg()
            self._process_frame(frame, stamp)

    def _process_frame(self, frame, stamp):
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
        scan.header.stamp = stamp
        scan.header.frame_id = "base_footprint"  # 修改為水平座標系，避免因 camera_link 傾斜 45 度導致訊號被當成地板以下而丟棄
        scan.angle_min = -self.hfov / 2.0
        scan.angle_max = self.hfov / 2.0
        scan.angle_increment = self.hfov / self.num_readings
        scan.range_min = 0.10  # 忽略 10 公分以內的盲區 (車體邊緣/陰影)
        scan.range_max = self.max_detect_dist

        ranges = []
        obstacle_pixels = []
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
                obstacle_pixels.append((int((i + 0.5) * col_step), obstacle_y))
            else:
                ranges.append(float('inf'))
                obstacle_pixels.append(None)

        scan.ranges = ranges

        valid_ranges = [r for r in ranges if r < float('inf')]
        if valid_ranges:
            min_dist = min(valid_ranges)
            self.get_logger().info(f"偵測到障礙物，最近距離: {min_dist:.2f} 公尺")

        self.scan_pub.publish(scan)

        # 6. 生成並發布 Debug 影像 (僅當有人訂閱時，或直接發送以利 RViz 隨時查看)
        # 繪製半透明的綠色遮罩代表「被判定為地板的安全區域」
        debug_frame = frame.copy()
        
        # 使用 OpenCV 安全的方法繪製半透明遮罩
        green_overlay = np.zeros_like(debug_frame)
        green_overlay[:] = (0, 255, 0)
        mask_bool = floor_mask > 0
        debug_frame[mask_bool] = cv2.addWeighted(debug_frame[mask_bool], 0.6, green_overlay[mask_bool], 0.4, 0)
        
        # 畫出障礙物的掃描紅點與距離
        for i, pt in enumerate(obstacle_pixels):
            if pt is not None:
                x, y = pt
                cv2.circle(debug_frame, (x, y), 5, (0, 0, 255), -1)
                dist_str = f"{ranges[i]:.2f}m"
                cv2.putText(debug_frame, dist_str, (x-15, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # 編碼成 JPEG 壓縮影像
        success, encoded_image = cv2.imencode('.jpg', debug_frame)
        if success:
            msg_img = CompressedImage()
            msg_img.header.stamp = stamp
            msg_img.header.frame_id = "camera_link"
            msg_img.format = "jpeg"
            msg_img.data = encoded_image.tobytes()
            self.debug_pub.publish(msg_img)

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

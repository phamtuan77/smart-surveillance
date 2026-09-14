import cv2
import os
import numpy as np

class Visualizer:
    def __init__(self, output_dir="data/output/"):
        """
        Khởi tạo bộ vẽ trực quan và tạo cấu trúc thư mục.
        :param output_dir: Đường dẫn thư mục lưu kết quả video/ảnh.
        """
        self.output_dir = output_dir
        # Tự động tạo thư mục data/output/ nếu chưa tồn tại
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Cấu hình hiển thị chung
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.color_box = (0, 255, 0)      # Xanh lá cho đối tượng bình thường (B, G, R)
        self.color_alert = (0, 0, 255)    # Đỏ cho cảnh báo bất thường
        self.color_text = (255, 255, 255) # Trắng cho văn bản
        
    def draw_detection_tracking_reid(self, frame, tracks):
        """
        Vẽ Bounding Box và ID của người/vật thể (tích hợp Detection, Tracking và ReID).
        """
        for obj in tracks:
            # Lấy bounding box, bỏ qua nếu không có
            bbox = obj.get('bbox')
            if bbox is None:
                continue
                
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ưu tiên hiển thị reid_id, nếu chưa có thì dùng track_id
            obj_id = obj.get('reid_id') or obj.get('track_id', 'Unknown')
            
            # Vẽ Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.color_box, 2)
            
            # Vẽ nhãn ID
            label = f"ID: {obj_id}"
            (w, h), _ = cv2.getTextSize(label, self.font, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), self.color_box, -1)
            cv2.putText(frame, label, (x1, y1 - 5), self.font, 0.6, self.color_text, 1, cv2.LINE_AA)
            
        return frame

    def draw_motion_segmentation(self, frame, motion_mask):
        """
        Phủ lớp màu đánh dấu vùng có chuyển động (Motion Segmentation).
        :param frame: Khung hình gốc.
        :param motion_mask: Ảnh nhị phân (Binary mask) với điểm ảnh 255 là chuyển động.
        :return: Khung hình đã hòa trộn overlay chuyển động.
        """
        # Tạo overlay màu vàng cho vùng chuyển động
        overlay = frame.copy()
        overlay[motion_mask > 0] = (0, 255, 255) # Màu vàng (B, G, R)
        
        # Trộn overlay vào frame gốc (Alpha blending)
        alpha = 0.35
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    def draw_anomaly_alert(self, frame, is_anomalous):
        """
        Vẽ cảnh báo lên màn hình nếu phát hiện hành vi bất thường (Anomaly Detection).
        :param frame: Khung hình hiện tại.
        :param is_anomalous: Boolean (True nếu có bất thường).
        :return: Khung hình có cảnh báo.
        """
        if is_anomalous:
            # Vẽ viền đỏ dày quanh toàn bộ màn hình camera
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, h), self.color_alert, 10)
            
            # Chèn chữ CẢNH BÁO lớn ở góc trái
            cv2.putText(frame, "WARNING: ANOMALY DETECTED!", (30, 50), 
                        self.font, 1.2, self.color_alert, 3, cv2.LINE_AA)
        return frame

    def save_frame(self, frame, filename="output_frame.jpg"):
        """
        Lưu khung hình đã xử lý ra thư mục data/output/
        :param frame: Khung hình cần lưu.
        :param filename: Tên file.
        """
        save_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(save_path, frame)

    def draw(self, frame, tracks=None, motion_mask=None, is_abnormal=False, **kwargs):
        """
        Hàm tổng hợp để pipeline.py gọi tới, thực hiện vẽ toàn bộ kết quả lên frame.
        """
        # 1. Phủ màu vùng chuyển động
        if motion_mask is not None:
            frame = self.draw_motion_segmentation(frame, motion_mask)
        
        # 2. Vẽ Bounding Box và ID đối tượng
        if tracks:
            frame = self.draw_detection_tracking_reid(frame, tracks)
            
        # 3. Vẽ cảnh báo bất thường
        if is_abnormal:
            frame = self.draw_anomaly_alert(frame, is_abnormal)
            
        return frame

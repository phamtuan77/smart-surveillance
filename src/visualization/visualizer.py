import cv2
import numpy as np

class Visualizer:
    def __init__(self):
        # Tạo sẵn bảng màu ngẫu nhiên cho các ID khác nhau
        self.colors = np.random.randint(0, 255, (100, 3), dtype=np.uint8)

    def draw(self, frame, detections, anomaly_score=0.0, is_abnormal=False):
        """
        frame: Ảnh gốc từ video (numpy array)
        detections: list các dictionary chứa {'box': [x1, y1, x2, y2], 'id': int, 'mask': numpy array}
        anomaly_score: float
        is_abnormal: bool
        """
        annotated_frame = frame.copy()

        # 1. Vẽ cảnh báo Anomaly lên góc trái màn hình
        if is_abnormal:
            cv2.putText(annotated_frame, f"ALARM! ANOMALY DETECTED ({anomaly_score:.2f})", 
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            cv2.putText(annotated_frame, "STATUS: NORMAL", 
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 2. Duyệt qua từng đối tượng để vẽ Box, ID và Mask
        for det in detections:
            box = det.get('box')
            obj_id = det.get('id', -1)
            mask = det.get('mask')

            # Trích xuất màu theo ID
            color = self.colors[obj_id % 100].tolist() if obj_id != -1 else (255, 255, 255)

            # Vẽ vùng chuyển động (Segmentation Mask) trước để nó nằm dưới box và text
            if mask is not None:
                colored_mask = np.zeros_like(annotated_frame, dtype=np.uint8)
                colored_mask[mask > 0] = color  # Tô màu cho vùng có chuyển động
                # Trộn mask vào frame gốc với độ trong suốt 0.4
                cv2.addWeighted(colored_mask, 0.4, annotated_frame, 1, 0, annotated_frame)

            # Vẽ Bounding Box và ID
            if box is not None:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                if obj_id != -1:
                    label = f"ID: {obj_id}"
                    # Nền đen chữ trắng cho ID dễ nhìn
                    cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + 80, y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1 + 5, y1 - 7), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated_frame
import cv2
import numpy as np
import sys
import os

# Trỏ đường dẫn để import được module trong src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.visualization.visualizer import Visualizer

def run_test():
    # Khởi tạo frame giả (màn hình đen 720p)
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Khởi tạo mask giả (một hình vuông trắng đại diện cho vùng đang chuyển động)
    dummy_mask = np.zeros((720, 1280), dtype=np.uint8)
    dummy_mask[200:500, 300:500] = 255 

    # Dữ liệu giả lập đầu vào từ các module 1, 2, 4
    mock_detections = [
        {'box': [300, 200, 500, 500], 'id': 1, 'mask': dummy_mask},
        {'box': [700, 300, 850, 600], 'id': 2, 'mask': None}
    ]

    # Test Visualizer
    vis = Visualizer()
    result = vis.draw(dummy_frame, mock_detections, anomaly_score=0.92, is_abnormal=True)

    # Hiển thị
    cv2.imshow("Visualization Test", result)
    print("Bấm phím bất kỳ trên cửa sổ ảnh để thoát...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_test()
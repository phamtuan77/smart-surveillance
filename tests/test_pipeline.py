import sys
import os
import cv2
import numpy as np
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipeline import SurveillancePipeline, FrameContext

class MockDetector:
    def detect(self, image):
        return [{"bbox": [50, 50, 150, 200], "conf": 0.95, "cls": 0}]

class MockTracker:
    def update(self, detections, image):
        return [{"track_id": 1, "bbox": [50, 50, 150, 200]}]

class MockReID:
    def extract(self, image, tracks):
        return {1: np.random.rand(512)}

class MockSegmentor:
    def get_motion_mask(self, image):
        h, w = image.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)

class MockAnomalyDetector:
    def evaluate(self, ctx: FrameContext):
        score = 0.35
        is_abnormal = score > 0.7
        return score, is_abnormal

class MockVisualizer:
    def draw(self, ctx: FrameContext):
        img = ctx.image.copy()
        for t in ctx.tracks:
            x1, y1, x2, y2 = t["bbox"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"ID: {t['track_id']}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        status = "ABNORMAL" if ctx.is_abnormal else "NORMAL"
        cv2.putText(img, f"Score: {ctx.anomaly_score:.2f} [{status}]", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if ctx.is_abnormal else (0, 255, 0), 2)
        return img


def test_pipeline_flow():
    print("Bắt đầu kiểm thử pipeline")
    
    pipeline = SurveillancePipeline(
        detector=MockDetector(),
        tracker=MockTracker(),
        reid=MockReID(),
        segmentor=MockSegmentor(),
        anomaly_detector=MockAnomalyDetector(),
        visualizer=MockVisualizer()
    )

    num_frames = 30
    start_time = time.time()
    
    for frame_id in range(num_frames):
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ctx = pipeline.process_frame(dummy_frame, frame_id)
        
        assert len(ctx.detections) > 0, "Lỗi: Không nhận được detection"
        assert len(ctx.tracks) > 0, "Lỗi: Không nhận được tracks"
        assert ctx.annotated_image is not None, "Lỗi: Chưa vẽ được annotated_image"

    fps = num_frames / (time.time() - start_time)
    print(f"Test thành công {num_frames} frames!")
    print(f"Tốc độ xử lý pipeline giả lập: {fps:.2f} FPS")

if __name__ == "__main__":
    test_pipeline_flow()
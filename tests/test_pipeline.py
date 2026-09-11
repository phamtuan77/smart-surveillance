import numpy as np

from src.pipeline import SurveillancePipeline


class FakeDetector:
    def detect(self, frame):
        return [
            {
                "class": "person",
                "confidence": 0.9,
                "bbox": [50, 50, 150, 250]
            }
        ]


class FakeTracker:
    def update(self, frame):
        return [
            {
                "track_id": 1,
                "bbox": [50, 50, 150, 250],
                "confidence": 0.9,
                "class": "person"
            }
        ]


class FakeReID:
    def identify(self, crop, person_id=None):
        return {
            "feature": np.array([1.0, 2.0, 3.0]),
            "person_id": 1,
            "score": 0.95
        }


class FakeSegmenter:
    def segment(self, frame):
        mask = np.zeros(
            (frame.shape[0], frame.shape[1]),
            dtype=np.uint8
        )

        regions = []

        return mask, regions


class FakeAnomalyDetector:
    def detect(self, frame):
        return {
            "score": 0.01,
            "is_abnormal": False
        }


class FakeVisualizer:
    def draw(
        self,
        frame,
        detections,
        tracks=None,
        anomaly_score=0.0,
        is_abnormal=False
    ):
        return frame.copy()


def test_pipeline_process_frame():

    pipeline = SurveillancePipeline(
        detector=FakeDetector(),
        tracker=FakeTracker(),
        reid=FakeReID(),
        segmentor=FakeSegmenter(),
        anomaly_detector=FakeAnomalyDetector(),
        visualizer=FakeVisualizer()
    )

    frame = np.zeros(
        (480, 640, 3),
        dtype=np.uint8
    )

    context = pipeline.process_frame(
        frame,
        frame_id=0
    )

    # Kiểm tra context
    assert context is not None

    # Detection
    assert len(context.detections) == 1

    # Tracking
    assert len(context.tracks) == 1
    assert context.tracks[0]["track_id"] == 1

    # ReID
    assert 1 in context.reid_features
    assert 1 in context.reid_ids
    assert 1 in context.reid_scores

    assert context.reid_ids[1] == 1
    assert context.reid_scores[1] == 0.95

    # Segmentation
    assert context.motion_mask is not None
    assert context.motion_mask.shape == (480, 640)

    # Anomaly
    assert context.anomaly_score == 0.01
    assert context.is_abnormal is False

    # Visualization
    assert context.annotated_image is not None
    assert context.annotated_image.shape == frame.shape
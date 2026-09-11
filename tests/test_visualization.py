import numpy as np

from src.visualization.visualizer import Visualizer


def test_visualizer_normal():
    visualizer = Visualizer()

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = [
        {
            "class": "person",
            "confidence": 0.95,
            "bbox": [100, 100, 200, 300]
        }
    ]

    tracks = [
        {
            "track_id": 1,
            "bbox": [100, 100, 200, 300],
            "confidence": 0.95,
            "class": "person"
        }
    ]

    output = visualizer.draw(
        frame=frame,
        detections=detections,
        tracks=tracks,
        anomaly_score=0.01,
        is_abnormal=False
    )

    assert output is not None
    assert output.shape == frame.shape
    assert output.dtype == np.uint8


def test_visualizer_abnormal():
    visualizer = Visualizer()

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    output = visualizer.draw(
        frame=frame,
        detections=[],
        tracks=[],
        anomaly_score=0.05,
        is_abnormal=True
    )

    assert output is not None
    assert output.shape == frame.shape
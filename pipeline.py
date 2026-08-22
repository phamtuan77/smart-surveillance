import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class FrameContext:
    frame_id: int
    image: np.ndarray
    detections: List[Dict[str, Any]] = field(default_factory=list)      # Người 1: YOLOv8
    tracks: List[Dict[str, Any]] = field(default_factory=list)          # Người 2: ByteTrack
    reid_features: Dict[int, np.ndarray] = field(default_factory=dict)  # Người 3: ReID
    motion_mask: Optional[np.ndarray] = None                            # Người 4: Segmentation
    anomaly_score: float = 0.0                                          # Người 5: Anomaly
    is_abnormal: bool = False
    annotated_image: Optional[np.ndarray] = None                        # Người 6: Visualization

class SurveillancePipeline:
    def __init__(self, detector=None, tracker=None, reid=None, 
                 segmentor=None, anomaly_detector=None, visualizer=None):
        self.detector = detector
        self.tracker = tracker
        self.reid = reid
        self.segmentor = segmentor
        self.anomaly_detector = anomaly_detector
        self.visualizer = visualizer

    def process_frame(self, frame: np.ndarray, frame_id: int) -> FrameContext:
        ctx = FrameContext(frame_id=frame_id, image=frame)

        # 1. Detection
        if self.detector is not None:
            ctx.detections = self.detector.detect(ctx.image)

        # 2. Tracking
        if self.tracker is not None:
            ctx.tracks = self.tracker.update(ctx.detections, ctx.image)

        # 3. ReID
        if self.reid is not None:
            ctx.reid_features = self.reid.extract(ctx.image, ctx.tracks)

        # 4. Motion Segmentation
        if self.segmentor is not None:
            ctx.motion_mask = self.segmentor.get_motion_mask(ctx.image)

        # 5. Anomaly Detection
        if self.anomaly_detector is not None:
            ctx.anomaly_score, ctx.is_abnormal = self.anomaly_detector.evaluate(ctx)

        # 6. Visualization
        if self.visualizer is not None:
            ctx.annotated_image = self.visualizer.draw(ctx)
        else:
            ctx.annotated_image = ctx.image.copy()

        return ctx
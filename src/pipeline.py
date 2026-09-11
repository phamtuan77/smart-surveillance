import cv2
import numpy as np

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class FrameContext:
    """
    Lưu toàn bộ kết quả xử lý của một frame.
    """

    frame_id: int
    image: np.ndarray

    # ==================================================
    # Detection
    # ==================================================

    detections: List[Dict[str, Any]] = field(
        default_factory=list
    )

    # ==================================================
    # Tracking
    # ==================================================

    tracks: List[Dict[str, Any]] = field(
        default_factory=list
    )

    # ==================================================
    # ReID
    # ==================================================

    reid_features: Dict[int, np.ndarray] = field(
        default_factory=dict
    )

    reid_ids: Dict[int, Any] = field(
        default_factory=dict
    )

    reid_scores: Dict[int, float] = field(
        default_factory=dict
    )

    # ==================================================
    # Segmentation
    # ==================================================

    motion_mask: Optional[np.ndarray] = None

    motion_regions: List = field(
        default_factory=list
    )

    # ==================================================
    # Anomaly Detection
    # ==================================================

    anomaly_score: float = 0.0

    is_abnormal: bool = False

    # ==================================================
    # Final image
    # ==================================================

    annotated_image: Optional[np.ndarray] = None


class SurveillancePipeline:
    """
    Pipeline chính của hệ thống Smart Surveillance.

    Detection
        ↓
    Tracking
        ↓
    ReID
        ↓
    Motion Segmentation
        ↓
    Anomaly Detection
        ↓
    Visualization
    """

    def __init__(
        self,
        detector=None,
        tracker=None,
        reid=None,
        segmentor=None,
        anomaly_detector=None,
        visualizer=None
    ):

        self.detector = detector
        self.tracker = tracker
        self.reid = reid
        self.segmentor = segmentor
        self.anomaly_detector = anomaly_detector
        self.visualizer = visualizer

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int
    ):
        """
        Xử lý toàn bộ pipeline trên một frame.

        Args:
            frame:
                Frame ảnh từ camera/video.

            frame_id:
                Số thứ tự của frame.

        Returns:
            FrameContext:
                Chứa kết quả của toàn bộ module.
        """

        # ==================================================
        # TẠO CONTEXT
        # ==================================================

        ctx = FrameContext(
            frame_id=frame_id,
            image=frame
        )

        # ==================================================
        # 1. DETECTION
        # ==================================================

        if self.detector is not None:

            ctx.detections = self.detector.detect(
                frame
            )

        # ==================================================
        # 2. TRACKING
        # ==================================================

        if self.tracker is not None:

            ctx.tracks = self.tracker.update(
                frame
            )

        # ==================================================
        # 3. REID
        # ==================================================

        if (
            self.reid is not None
            and len(ctx.tracks) > 0
        ):

            height, width = frame.shape[:2]

            for track in ctx.tracks:

                track_id = track.get(
                    "track_id"
                )

                bbox = track.get(
                    "bbox"
                )

                # ------------------------------------------
                # Kiểm tra bounding box
                # ------------------------------------------

                if bbox is None:
                    continue

                # ------------------------------------------
                # Lấy tọa độ bounding box
                # ------------------------------------------

                x1, y1, x2, y2 = map(
                    int,
                    bbox
                )

                # ------------------------------------------
                # Giới hạn bbox trong frame
                # ------------------------------------------

                x1 = max(
                    0,
                    x1
                )

                y1 = max(
                    0,
                    y1
                )

                x2 = min(
                    width,
                    x2
                )

                y2 = min(
                    height,
                    y2
                )

                # ------------------------------------------
                # Kiểm tra bbox hợp lệ
                # ------------------------------------------

                if (
                    x2 <= x1
                    or y2 <= y1
                ):
                    continue

                # ------------------------------------------
                # Cắt người ra khỏi frame
                # ------------------------------------------

                person_crop = frame[
                    y1:y2,
                    x1:x2
                ]

                if person_crop.size == 0:
                    continue

                # ------------------------------------------
                # Trích xuất + nhận diện ReID
                # ------------------------------------------

                try:

                    result = self.reid.identify(
                        person_crop,
                        person_id=track_id
                    )

                    feature = result.get(
                        "feature"
                    )

                    person_id = result.get(
                        "person_id"
                    )

                    score = result.get(
                        "score",
                        0.0
                    )

                    # --------------------------------------
                    # Lưu feature
                    # --------------------------------------

                    if (
                        track_id is not None
                        and feature is not None
                    ):

                        ctx.reid_features[
                            track_id
                        ] = feature

                    # --------------------------------------
                    # Lưu ReID ID
                    # --------------------------------------

                    if (
                        track_id is not None
                        and person_id is not None
                    ):

                        ctx.reid_ids[
                            track_id
                        ] = person_id

                    # --------------------------------------
                    # Lưu similarity score
                    # --------------------------------------

                    if track_id is not None:

                        ctx.reid_scores[
                            track_id
                        ] = score

                    # --------------------------------------
                    # Lưu trực tiếp vào track
                    # --------------------------------------

                    track["reid_id"] = (
                        person_id
                    )

                    track["reid_score"] = (
                        score
                    )

                except Exception as error:

                    print(
                        f"[ReID] Lỗi xử lý track "
                        f"{track_id}: {error}"
                    )

        # ==================================================
        # 4. MOTION SEGMENTATION
        # ==================================================

        if self.segmentor is not None:

            motion_mask, regions = (
                self.segmentor.segment(frame)
            )

            ctx.motion_mask = (
                motion_mask
            )

            ctx.motion_regions = (
                regions
            )

        # ==================================================
        # 5. ANOMALY DETECTION
        # ==================================================

        if self.anomaly_detector is not None:

            result = (
                self.anomaly_detector.detect(
                    frame
                )
            )

            ctx.anomaly_score = (
                result.get(
                    "score",
                    0.0
                )
            )

            ctx.is_abnormal = (
                result.get(
                    "is_abnormal",
                    False
                )
            )

        # ==================================================
        # 6. VISUALIZATION
        # ==================================================

        if self.visualizer is not None:

            ctx.annotated_image = (
                self.visualizer.draw(
                    frame=frame,
                    detections=ctx.detections,
                    tracks=ctx.tracks,
                    anomaly_score=ctx.anomaly_score,
                    is_abnormal=ctx.is_abnormal
                )
            )

        else:

            ctx.annotated_image = (
                frame.copy()
            )

        # ==================================================
        # RETURN
        # ==================================================

        return ctx
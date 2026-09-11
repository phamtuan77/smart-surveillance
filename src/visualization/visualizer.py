import cv2
import numpy as np


class Visualizer:
    def __init__(self):
        self.colors = np.random.randint(
            0, 255, (100, 3), dtype=np.uint8
        )

    def draw(
        self,
        frame,
        detections,
        tracks=None,
        anomaly_score=0.0,
        is_abnormal=False
    ):
        annotated_frame = frame.copy()

        height, width = annotated_frame.shape[:2]

        # ==================================================
        # 1. STATUS / ANOMALY
        # ==================================================

        if is_abnormal:

            # Banner cảnh báo
            cv2.rectangle(
                annotated_frame,
                (0, 0),
                (width, 90),
                (0, 0, 180),
                -1
            )

            # Dòng cảnh báo
            cv2.putText(
                annotated_frame,
                "ANOMALY DETECTED",
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # Score
            cv2.putText(
                annotated_frame,
                f"Score: {anomaly_score:.4f}",
                (20, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        else:

            # Banner Normal
            cv2.rectangle(
                annotated_frame,
                (0, 0),
                (width, 55),
                (0, 130, 0),
                -1
            )

            cv2.putText(
                annotated_frame,
                f"NORMAL | Score: {anomaly_score:.4f}",
                (20, 37),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        # ==================================================
        # 2. TRACKING + REID
        # ==================================================

        if tracks is not None:

            for track in tracks:

                bbox = track.get("bbox")

                if bbox is None:
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    bbox
                )

                track_id = track.get(
                    "track_id",
                    -1
                )

                reid_id = track.get(
                    "reid_id",
                    None
                )

                reid_score = track.get(
                    "reid_score",
                    0.0
                )

                # Màu riêng cho từng Track ID
                color = self.colors[
                    track_id % 100
                ].tolist()

                # ------------------------------------------
                # Bounding Box
                # ------------------------------------------

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                # ------------------------------------------
                # Track ID
                # ------------------------------------------

                label_y = max(
                    110,
                    y1 - 10
                )

                if reid_id is not None:

                    label = (
                        f"Track: {track_id} "
                        f"ReID: {reid_id}"
                    )

                else:

                    label = (
                        f"Track: {track_id}"
                    )

                # ------------------------------------------
                # Label background
                # ------------------------------------------

                (text_width, text_height), _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    2
                )

                cv2.rectangle(
                    annotated_frame,
                    (x1, label_y - text_height - 8),
                    (x1 + text_width + 8, label_y + 2),
                    color,
                    -1
                )

                # ------------------------------------------
                # Label text
                # ------------------------------------------

                cv2.putText(
                    annotated_frame,
                    label,
                    (x1 + 4, label_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

        # ==================================================
        # 3. THÔNG TIN HỆ THỐNG
        # ==================================================

        info_y = height - 45

        cv2.putText(
            annotated_frame,
            f"Persons: {len(detections)}",
            (15, info_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            annotated_frame,
            f"Tracks: {len(tracks) if tracks else 0}",
            (15, info_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        return annotated_frame
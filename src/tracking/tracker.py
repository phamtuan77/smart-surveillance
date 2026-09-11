from ultralytics import YOLO


class PersonTracker:

    def __init__(self, model_name="yolov8n.pt"):
        self.model = YOLO(model_name)

    def update(self, frame):
        """
        Nhận 1 frame và trả về danh sách người đang được tracking.
        """

        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            verbose=False
        )

        tracks = []

        if not results:
            return tracks

        result = results[0]

        if result.boxes is None:
            return tracks

        for box in result.boxes:

            # ByteTrack chưa cấp ID
            if box.id is None:
                continue

            track_id = int(box.id[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            confidence = float(box.conf[0])

            tracks.append({
                "track_id": track_id,
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence,
                "class": "person"
            })

        return tracks
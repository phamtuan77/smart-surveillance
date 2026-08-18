from ultralytics import YOLO


class PersonDetector:

    def __init__(self, model_name="yolov8n.pt"):
        self.model = YOLO(model_name)

    def detect(self, frame):
        results = self.model(frame, verbose=False)

        detections = []

        for result in results:
            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # Chỉ lấy người
                if class_id != 0:
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                detections.append({
                    "class": "person",
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]
                })

        return detections
import cv2

from src.detection.detector import PersonDetector


VIDEO_PATH = "data/test_videos/test.mp4"


def main():

    detector = PersonDetector()

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Không thể mở video!")
        return

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        detections = detector.detect(frame)

        for detection in detections:

            x1, y1, x2, y2 = detection["bbox"]
            confidence = detection["confidence"]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = f"Person {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        cv2.imshow(
            "Smart Surveillance - Detection",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
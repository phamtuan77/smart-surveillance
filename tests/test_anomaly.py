import cv2

from src.anomaly.detector import AnomalyDetector


VIDEO_PATH = "data/test_videos/test.mp4"


def main():

    detector = AnomalyDetector()

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Không thể mở video!")
        return

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        result = detector.detect(frame)

        color = (0, 0, 255) if result["is_abnormal"] else (0, 255, 0)
        label = f"{result['status']} ({result['score']:.4f})"

        cv2.putText(
            frame,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv2.imshow(
            "Smart Surveillance - Anomaly",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

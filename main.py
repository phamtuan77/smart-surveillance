import os
import cv2

from src.detection.detector import PersonDetector
from src.tracking.tracker import PersonTracker
from src.reid.reid import PersonReID
from src.segmentation.motion import MotionSegmenter
from src.anomaly.detector import AnomalyDetector
from src.visualization.visualizer import Visualizer
from src.pipeline import SurveillancePipeline


VIDEO_PATH = "data/test_videos/test.mp4"
OUTPUT_DIR = "outputs"
OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "smart_surveillance_output.mp4"
)


def create_pipeline():
    """
    Khởi tạo tất cả module của Smart Surveillance.
    """

    print("[System] Khởi tạo Detection...")

    detector = PersonDetector()

    print("[System] Khởi tạo Tracking...")

    tracker = PersonTracker()

    print("[System] Khởi tạo ReID...")

    reid = PersonReID(
        device="cpu",
        threshold=0.7
    )

    print("[System] Khởi tạo Motion Segmentation...")

    segmentor = MotionSegmenter(
        min_area=150
    )

    print("[System] Khởi tạo Anomaly Detection...")

    anomaly_detector = AnomalyDetector(
        model_path="models/anomaly/autoencoder.pt",
        threshold_path="models/anomaly/threshold.txt",
        frame_size=64
    )

    print("[System] Khởi tạo Visualization...")

    visualizer = Visualizer()

    pipeline = SurveillancePipeline(
        detector=detector,
        tracker=tracker,
        reid=reid,
        segmentor=segmentor,
        anomaly_detector=anomaly_detector,
        visualizer=visualizer
    )

    return pipeline


def main():

    print("=" * 60)
    print("SMART SURVEILLANCE SYSTEM")
    print("=" * 60)

    # ==========================================
    # 1. KIỂM TRA VIDEO
    # ==========================================

    if not os.path.exists(VIDEO_PATH):
        print(
            f"[Error] Không tìm thấy video: {VIDEO_PATH}"
        )
        return

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("[Error] Không thể mở video!")
        return

    # ==========================================
    # 2. LẤY THÔNG TIN VIDEO
    # ==========================================

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print()
    print("[Video]")
    print(f"  Resolution : {width} x {height}")
    print(f"  FPS        : {fps:.2f}")
    print(f"  Frames     : {total_frames}")

    # ==========================================
    # 3. TẠO THƯ MỤC OUTPUT
    # ==========================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ==========================================
    # 4. TẠO VIDEO WRITER
    # ==========================================

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        print(
            "[Error] Không thể tạo video output!"
        )

        cap.release()
        return

    # ==========================================
    # 5. KHỞI TẠO PIPELINE
    # ==========================================

    pipeline = create_pipeline()

    print()
    print("=" * 60)
    print("BẮT ĐẦU XỬ LÝ VIDEO")
    print("Nhấn Q để dừng")
    print("=" * 60)

    frame_id = 0

    # ==========================================
    # 6. XỬ LÝ TỪNG FRAME
    # ==========================================

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # --------------------------------------
        # Chạy toàn bộ pipeline
        # --------------------------------------

        context = pipeline.process_frame(
            frame,
            frame_id
        )

        # --------------------------------------
        # Lấy frame đã visualization
        # --------------------------------------

        output_frame = context.annotated_image

        if output_frame is None:
            output_frame = frame

        # --------------------------------------
        # Ghi frame vào video output
        # --------------------------------------

        writer.write(output_frame)

        # --------------------------------------
        # Hiển thị frame
        # --------------------------------------

        cv2.imshow(
            "Smart Surveillance",
            output_frame
        )

        # --------------------------------------
        # In thông tin mỗi 30 frame
        # --------------------------------------

        if frame_id % 30 == 0:

            print(
                f"[Frame {frame_id}] "
                f"Detection={len(context.detections)} | "
                f"Tracks={len(context.tracks)} | "
                f"ReID={len(context.reid_features)} | "
                f"Motion={len(context.motion_regions)} | "
                f"Anomaly={context.anomaly_score:.4f} | "
                f"Abnormal={context.is_abnormal}"
            )

        frame_id += 1

        # --------------------------------------
        # Nhấn Q để thoát
        # --------------------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print()
            print("[System] Người dùng dừng chương trình.")
            break

    # ==========================================
    # 7. GIẢI PHÓNG TÀI NGUYÊN
    # ==========================================

    cap.release()
    writer.release()

    cv2.destroyAllWindows()

    # ==========================================
    # 8. THÔNG BÁO KẾT QUẢ
    # ==========================================

    print()
    print("=" * 60)
    print("HOÀN THÀNH")
    print("=" * 60)

    print(
        f"Số frame đã xử lý: {frame_id}"
    )

    print(
        f"Video kết quả: {OUTPUT_VIDEO}"
    )


if __name__ == "__main__":
    main()
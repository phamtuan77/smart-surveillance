import os

import cv2
import numpy as np
import torch

from src.anomaly.autoencoder import ConvAutoencoder


VIDEO_PATH = "data/test_videos/test.mp4"
MODEL_PATH = "models/anomaly/autoencoder.pt"
THRESHOLD_PATH = "models/anomaly/threshold.txt"

FRAME_SIZE = 64
FRAME_STEP = 5

# Số độ lệch chuẩn dùng để xác định threshold
STD_MULTIPLIER = 3.0


def preprocess(frame):
    """
    Chuyển frame thành tensor grayscale 64x64.
    """

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    resized = cv2.resize(
        gray,
        (FRAME_SIZE, FRAME_SIZE)
    )

    normalized = (
        resized.astype(np.float32) / 255.0
    )

    tensor = torch.from_numpy(
        normalized
    )

    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor


def calculate_scores(model, device):
    """
    Chạy toàn bộ video và tính reconstruction error
    cho các frame được lấy mẫu.
    """

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(
            f"Không thể mở video: {VIDEO_PATH}"
        )

    scores = []

    frame_id = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_id % FRAME_STEP == 0:

            input_tensor = preprocess(frame)
            input_tensor = input_tensor.to(device)

            with torch.no_grad():

                reconstructed = model(
                    input_tensor
                )

                error = torch.mean(
                    (input_tensor - reconstructed) ** 2
                )

            scores.append(
                error.item()
            )

        frame_id += 1

    cap.release()

    return np.array(
        scores,
        dtype=np.float32
    )


def main():

    print("=" * 60)
    print("CALIBRATE ANOMALY THRESHOLD")
    print("=" * 60)

    # ==========================================
    # 1. KIỂM TRA MODEL
    # ==========================================

    if not os.path.exists(MODEL_PATH):

        print(
            f"[Error] Không tìm thấy model: {MODEL_PATH}"
        )

        print()
        print(
            "Hãy chạy trước:"
        )

        print(
            "python -m src.anomaly.train"
        )

        return

    # ==========================================
    # 2. KIỂM TRA VIDEO
    # ==========================================

    if not os.path.exists(VIDEO_PATH):

        print(
            f"[Error] Không tìm thấy video: {VIDEO_PATH}"
        )

        return

    # ==========================================
    # 3. DEVICE
    # ==========================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[Calibrate] Device: {device}"
    )

    # ==========================================
    # 4. LOAD MODEL
    # ==========================================

    model = ConvAutoencoder().to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.eval()

    print(
        f"[Calibrate] Đã load model: {MODEL_PATH}"
    )

    # ==========================================
    # 5. TÍNH ANOMALY SCORES
    # ==========================================

    print()
    print(
        "[Calibrate] Đang tính reconstruction error..."
    )

    scores = calculate_scores(
        model,
        device
    )

    if len(scores) == 0:

        print(
            "[Error] Không tính được score nào."
        )

        return

    # ==========================================
    # 6. TÍNH THỐNG KÊ
    # ==========================================

    mean_score = float(
        np.mean(scores)
    )

    std_score = float(
        np.std(scores)
    )

    min_score = float(
        np.min(scores)
    )

    max_score = float(
        np.max(scores)
    )

    threshold = (
        mean_score
        + STD_MULTIPLIER * std_score
    )

    # ==========================================
    # 7. HIỂN THỊ KẾT QUẢ
    # ==========================================

    print()
    print("=" * 60)
    print("KẾT QUẢ CALIBRATION")
    print("=" * 60)

    print(
        f"Số frame được dùng: {len(scores)}"
    )

    print(
        f"Min score          : {min_score:.6f}"
    )

    print(
        f"Mean score         : {mean_score:.6f}"
    )

    print(
        f"Std score          : {std_score:.6f}"
    )

    print(
        f"Max score          : {max_score:.6f}"
    )

    print(
        f"Std multiplier     : {STD_MULTIPLIER}"
    )

    print(
        f"New threshold      : {threshold:.6f}"
    )

    # ==========================================
    # 8. LƯU THRESHOLD
    # ==========================================

    os.makedirs(
        "models/anomaly",
        exist_ok=True
    )

    with open(
        THRESHOLD_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            f"{threshold:.10f}"
        )

    print()
    print(
        f"Threshold đã lưu tại: {THRESHOLD_PATH}"
    )


if __name__ == "__main__":
    main()
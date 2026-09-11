import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.anomaly.autoencoder import ConvAutoencoder


VIDEO_PATH = "data/test_videos/test.mp4"
MODEL_PATH = "models/anomaly/autoencoder.pt"

FRAME_SIZE = 64
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.001
FRAME_STEP = 5


def load_frames(video_path):
    """
    Đọc frame từ video và chuyển thành ảnh grayscale 64x64.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Không thể mở video: {video_path}"
        )

    frames = []

    frame_id = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_id % FRAME_STEP == 0:

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

            frames.append(normalized)

        frame_id += 1

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(
            "Không lấy được frame nào từ video."
        )

    data = np.array(frames, dtype=np.float32)

    data = torch.from_numpy(data)

    data = data.unsqueeze(1)

    return data


def train():

    print("=" * 60)
    print("TRAIN ANOMALY AUTOENCODER")
    print("=" * 60)

    # ==========================================
    # 1. KIỂM TRA VIDEO
    # ==========================================

    if not os.path.exists(VIDEO_PATH):

        print(
            f"[Error] Không tìm thấy video: {VIDEO_PATH}"
        )

        return

    # ==========================================
    # 2. LOAD DATA
    # ==========================================

    print()
    print("[Train] Đang đọc video...")

    data = load_frames(VIDEO_PATH)

    print(
        f"[Train] Số lượng frame: {len(data)}"
    )

    print(
        f"[Train] Kích thước: {tuple(data.shape)}"
    )

    # ==========================================
    # 3. DATASET
    # ==========================================

    dataset = TensorDataset(data, data)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # ==========================================
    # 4. DEVICE
    # ==========================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[Train] Device: {device}"
    )

    # ==========================================
    # 5. MODEL
    # ==========================================

    model = ConvAutoencoder().to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # ==========================================
    # 6. TRAIN
    # ==========================================

    print()
    print("[Train] Bắt đầu training...")

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        for inputs, targets in loader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(
                outputs,
                targets
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss / len(loader)
        )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {average_loss:.6f}"
        )

    # ==========================================
    # 7. LƯU MODEL
    # ==========================================

    os.makedirs(
        "models/anomaly",
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print()
    print("=" * 60)
    print("TRAINING HOÀN THÀNH")
    print("=" * 60)

    print(
        f"Model đã lưu tại: {MODEL_PATH}"
    )


if __name__ == "__main__":
    train()
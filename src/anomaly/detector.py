import os

import cv2
import numpy as np
import torch

from src.anomaly.autoencoder import ConvAutoencoder


class AnomalyDetector:
    
    def __init__(self, model_path="models/anomaly/autoencoder.pt", frame_size=64, threshold=0.02):

        self.frame_size = frame_size
        self.threshold = threshold  # TODO: hiệu chỉnh lại khi có model đã train + dữ liệu validation

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ConvAutoencoder().to(self.device)

        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"[Anomaly] Đã load model từ '{model_path}'")
        else:
            print("[Anomaly] Chưa có model đã train, dùng model khởi tạo ngẫu nhiên (chỉ để test pipeline).")

        self.model.eval()

    def preprocess(self, frame):
      

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.frame_size, self.frame_size))
        normalized = resized.astype(np.float32) / 255.0

        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        return tensor.to(self.device)

    def compute_score(self, frame):
       
        input_tensor = self.preprocess(frame)

        with torch.no_grad():
            reconstructed = self.model(input_tensor)
            error = torch.mean((input_tensor - reconstructed) ** 2)

        return error.item()

    def detect(self, frame):
       
        score = self.compute_score(frame)
        is_abnormal = score > self.threshold

        return {
            "score": score,
            "status": "Abnormal" if is_abnormal else "Normal",
            "is_abnormal": is_abnormal
        }

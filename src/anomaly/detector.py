import os

import cv2
import numpy as np
import torch

from src.anomaly.autoencoder import ConvAutoencoder


class AnomalyDetector:
    """
    Phát hiện bất thường bằng Convolutional Autoencoder.
    """

    def __init__(
        self,
        model_path="models/anomaly/autoencoder.pt",
        threshold_path="models/anomaly/threshold.txt",
        frame_size=64
    ):
        self.frame_size = frame_size

        # ==========================================
        # DEVICE
        # ==========================================

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # ==========================================
        # LOAD MODEL
        # ==========================================

        if not os.path.exists(model_path):

            raise FileNotFoundError(
                f"Không tìm thấy model Anomaly: {model_path}\n"
                f"Hãy chạy:\n"
                f"python -m src.anomaly.train"
            )

        self.model = ConvAutoencoder().to(
            self.device
        )

        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device
            )
        )

        self.model.eval()

        print(
            f"[Anomaly] Đã load model từ '{model_path}'"
        )

        # ==========================================
        # LOAD THRESHOLD
        # ==========================================

        if not os.path.exists(threshold_path):

            raise FileNotFoundError(
                f"Không tìm thấy threshold: {threshold_path}\n"
                f"Hãy chạy:\n"
                f"python -m src.anomaly.calibrate"
            )

        with open(
            threshold_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.threshold = float(
                file.read().strip()
            )

        print(
            f"[Anomaly] Device: {self.device}"
        )

        print(
            f"[Anomaly] Threshold: {self.threshold:.6f}"
        )

    # ==========================================
    # PREPROCESS
    # ==========================================

    def preprocess(self, frame):

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        resized = cv2.resize(
            gray,
            (
                self.frame_size,
                self.frame_size
            )
        )

        normalized = (
            resized.astype(np.float32) / 255.0
        )

        tensor = torch.from_numpy(
            normalized
        )

        tensor = tensor.unsqueeze(0).unsqueeze(0)

        return tensor.to(self.device)

    # ==========================================
    # COMPUTE SCORE
    # ==========================================

    def compute_score(self, frame):

        input_tensor = self.preprocess(
            frame
        )

        with torch.no_grad():

            reconstructed = self.model(
                input_tensor
            )

            error = torch.mean(
                (input_tensor - reconstructed) ** 2
            )

        return error.item()

    # ==========================================
    # DETECT
    # ==========================================

    def detect(self, frame):

        score = self.compute_score(
            frame
        )

        is_abnormal = (
            score > self.threshold
        )

        return {
            "score": score,
            "status": (
                "Abnormal"
                if is_abnormal
                else "Normal"
            ),
            "is_abnormal": is_abnormal
        }
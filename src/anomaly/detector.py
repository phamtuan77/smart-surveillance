import cv2
import numpy as np
import torch

from .autoencoder import AutoEncoder


class AnomalyDetector:

    def __init__(
        self,
        frame_size=(64, 64),
        latent_size=128,
        threshold=0.10
    ):
        self.width = frame_size[0]
        self.height = frame_size[1]

        self.input_size = (
            self.width * self.height
        )

        self.threshold = threshold

        self.model = AutoEncoder(
            input_size=self.input_size,
            latent_size=latent_size
        )

        self.model.eval()

    # =========================================================
    # FRAME -> TENSOR
    # =========================================================

    def preprocess_frame(self, frame):

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        resized = cv2.resize(
            gray,
            (self.width, self.height)
        )

        normalized = resized.astype(
            np.float32
        ) / 255.0

        tensor = torch.tensor(
            normalized,
            dtype=torch.float32
        )

        tensor = tensor.flatten()

        return tensor

    # =========================================================
    # RECONSTRUCTION ERROR
    # =========================================================

    def calculate_reconstruction_error(
        self,
        frame
    ):

        input_tensor = self.preprocess_frame(
            frame
        )

        input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():

            reconstructed = self.model(
                input_tensor
            )

        error = torch.mean(
            (input_tensor - reconstructed) ** 2
        )

        return float(error.item())

    # =========================================================
    # ERROR -> ANOMALY SCORE
    # =========================================================

    def calculate_anomaly_score(
        self,
        reconstruction_error
    ):
        """
        Chuyển Reconstruction Error
        thành Anomaly Score từ 0 -> 1.

        Công thức:

            score = error / (error + threshold)

        Score càng lớn -> càng bất thường.
        """

        score = (
            reconstruction_error
            / (
                reconstruction_error
                + self.threshold
            )
        )

        score = min(
            max(score, 0.0),
            1.0
        )

        return float(score)

    # =========================================================
    # PREDICT
    # =========================================================

    def predict(self, frame):

        reconstruction_error = (
            self.calculate_reconstruction_error(
                frame
            )
        )

        anomaly_score = (
            self.calculate_anomaly_score(
                reconstruction_error
            )
        )

        if reconstruction_error >= self.threshold:
            label = "ABNORMAL"
        else:
            label = "NORMAL"

        return {
            "reconstruction_error":
                reconstruction_error,

            "anomaly_score":
                anomaly_score,

            "label":
                label
        }

    # =========================================================
    # PROCESS VIDEO
    # =========================================================

    def process_video(
        self,
        input_path,
        output_path=None
    ):
        """
        Phân tích toàn bộ video.

        Nếu output_path được truyền vào,
        chương trình sẽ lưu video có kết quả.
        """

        cap = cv2.VideoCapture(
            input_path
        )

        if not cap.isOpened():
            raise FileNotFoundError(
                f"Không thể mở video: {input_path}"
            )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 25

        frame_width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        frame_height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        writer = None

        if output_path is not None:

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (
                    frame_width,
                    frame_height
                )
            )

        results = []

        frame_number = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            result = self.predict(frame)

            results.append({
                "frame": frame_number,
                **result
            })

            # Hiển thị thông tin
            text_score = (
                f"Anomaly Score: "
                f"{result['anomaly_score']:.3f}"
            )

            text_error = (
                f"Reconstruction Error: "
                f"{result['reconstruction_error']:.5f}"
            )

            text_label = (
                f"Status: {result['label']}"
            )

            cv2.putText(
                frame,
                text_score,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                text_error,
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                text_label,
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255)
                if result["label"] == "ABNORMAL"
                else (0, 255, 0),
                2
            )

            if writer is not None:
                writer.write(frame)

            cv2.imshow(
                "Anomaly Detection",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        cap.release()

        if writer is not None:
            writer.release()

        cv2.destroyAllWindows()

        return results

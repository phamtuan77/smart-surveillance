from src.reid.features_extractor import FeatureExtractor
from src.reid.reid_model import ReIDModel


class PersonReID:
    """
    Kết hợp FeatureExtractor và ReIDModel thành một module ReID hoàn chỉnh.

    FeatureExtractor:
        - Nhận ảnh người.
        - Trích xuất vector đặc trưng bằng ResNet18.

    ReIDModel:
        - Lưu các vector đặc trưng vào gallery.
        - So sánh vector mới với gallery bằng Cosine Similarity.
        - Trả về ID phù hợp nếu độ tương đồng vượt threshold.
    """

    def __init__(self, device="cpu", threshold=0.7):
        """
        Khởi tạo module ReID.

        Args:
            device: Thiết bị chạy model, "cpu" hoặc "cuda".
            threshold: Ngưỡng Cosine Similarity để xác định cùng một người.
        """

        self.extractor = FeatureExtractor(device=device)

        self.matcher = ReIDModel(
            threshold=threshold
        )

    def extract(self, image):
        """
        Trích xuất vector đặc trưng từ ảnh một người.

        Args:
            image: Ảnh người dạng numpy array.

        Returns:
            numpy.ndarray:
                Vector đặc trưng của người.
        """

        feature = self.extractor.extract(image)

        return feature

    def match(self, feature):
        """
        So sánh feature với gallery hiện tại.

        Args:
            feature: Vector đặc trưng cần tìm kiếm.

        Returns:
            tuple:
                (person_id, similarity_score)

            Nếu không tìm thấy người phù hợp:
                (None, similarity_score)
        """

        person_id, score = self.matcher.match(feature)

        return person_id, score

    def add_to_gallery(self, person_id, feature):
        """
        Thêm vector đặc trưng của một người vào gallery.

        Args:
            person_id: ID của người.
            feature: Vector đặc trưng của người.
        """

        self.matcher.add_to_gallery(
            person_id,
            feature
        )

    def identify(self, image, person_id=None):
        """
        Trích xuất feature và xác định người.

        Quy trình:

            Ảnh người
                ↓
            FeatureExtractor
                ↓
            Feature vector
                ↓
            ReIDModel
                ↓
            So sánh Gallery
                ↓
            Person ID

        Args:
            image:
                Ảnh người dạng numpy array.

            person_id:
                ID hiện tại từ Tracking.
                Nếu không tìm thấy người trong gallery,
                ID này sẽ được dùng để tạo entry mới.

        Returns:
            dict:
                {
                    "person_id": ID người,
                    "score": độ tương đồng,
                    "feature": vector đặc trưng,
                    "matched": True/False
                }
        """

        feature = self.extract(image)

        matched_id, score = self.match(feature)

        if matched_id is not None:
            return {
                "person_id": matched_id,
                "score": score,
                "feature": feature,
                "matched": True
            }

        if person_id is not None:
            self.add_to_gallery(
                person_id,
                feature
            )

            return {
                "person_id": person_id,
                "score": score,
                "feature": feature,
                "matched": False
            }

        return {
            "person_id": None,
            "score": score,
            "feature": feature,
            "matched": False
        }
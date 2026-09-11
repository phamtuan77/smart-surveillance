import numpy as np

class ReIDModel:
    def __init__(self, threshold=0.7):
        self.gallery = {}          # {id: [feature1, feature2, ...]}
        self.threshold = threshold # ngưỡng để coi là "cùng 1 người"

    def _cosine_similarity(self, a, b):
        a = a / (np.linalg.norm(a) + 1e-8)
        b = b / (np.linalg.norm(b) + 1e-8)
        return np.dot(a, b)

    def add_to_gallery(self, person_id, feature):
        """Lưu đặc trưng của 1 người vào kho (khi track bị mất dấu)"""
        if person_id not in self.gallery:
            self.gallery[person_id] = []
        self.gallery[person_id].append(feature)

    def match(self, feature):
        """
        So sánh feature mới với gallery.
        Trả về: (matched_id, similarity) nếu tìm thấy người cũ,
                 hoặc (None, 0) nếu là người mới.
        """
        best_id = None
        best_score = 0

        for person_id, features in self.gallery.items():
            for f in features:
                score = self._cosine_similarity(feature, f)
                if score > best_score:
                    best_score = score
                    best_id = person_id

        if best_score >= self.threshold:
            return best_id, best_score
        return None, best_score
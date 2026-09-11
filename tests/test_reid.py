import numpy as np

from src.reid.reid_model import ReIDModel


def test_reid_match_same_feature():
    model = ReIDModel(threshold=0.7)

    feature = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    model.add_to_gallery(1, feature)

    person_id, score = model.match(feature)

    assert person_id == 1
    assert score >= 0.7


def test_reid_reject_different_feature():
    model = ReIDModel(threshold=0.7)

    feature_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    feature_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    model.add_to_gallery(1, feature_a)

    person_id, score = model.match(feature_b)

    assert person_id is None
    assert score < 0.7


def test_reid_gallery():
    model = ReIDModel()

    feature = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    model.add_to_gallery(5, feature)

    assert 5 in model.gallery
    assert len(model.gallery[5]) == 1
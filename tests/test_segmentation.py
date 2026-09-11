import cv2
import numpy as np

from src.segmentation.motion import MotionSegmenter


def test_motion_segmenter_returns_mask_and_regions():
    segmenter = MotionSegmenter()

    frame1 = np.zeros((240, 320, 3), dtype=np.uint8)
    frame2 = frame1.copy()

    cv2.rectangle(
        frame2,
        (100, 80),
        (180, 160),
        (255, 255, 255),
        -1
    )

    segmenter.segment(frame1)

    motion_mask, regions = segmenter.segment(frame2)

    assert motion_mask is not None
    assert isinstance(regions, list)
    assert motion_mask.shape == (240, 320)


def test_motion_mask_is_valid():
    segmenter = MotionSegmenter()

    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    motion_mask, regions = segmenter.segment(frame)

    assert motion_mask.dtype == np.uint8
    assert motion_mask.shape == (240, 320)
    assert isinstance(regions, list)
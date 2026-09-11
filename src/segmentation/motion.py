import cv2

from src.segmentation.mask import (
    clean_motion_mask,
    create_background_subtractor,
    find_motion_regions,
)


class MotionSegmenter:
    """Detect and mark moving regions in successive video frames."""

    def __init__(self, min_area=150, history=300, var_threshold=16):
        self.min_area = min_area
        self.background_subtractor = create_background_subtractor(
            history=history,
            var_threshold=var_threshold,
        )

    def segment(self, frame):
        """Return the cleaned motion mask and its bounding boxes."""
        foreground_mask = self.background_subtractor.apply(frame)
        motion_mask = clean_motion_mask(foreground_mask)
        regions = find_motion_regions(motion_mask, self.min_area)
        return motion_mask, regions

    def annotate(self, frame):
        """Draw a box and label around every detected moving region."""
        motion_mask, regions = self.segment(frame)
        annotated_frame = frame.copy()
        for x, y, width, height in regions:
            cv2.rectangle(
                annotated_frame,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                annotated_frame,
                "Moving",
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        return annotated_frame, motion_mask
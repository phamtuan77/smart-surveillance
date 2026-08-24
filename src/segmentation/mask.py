import cv2


def create_background_subtractor(history=300, var_threshold=16):
    """Create the MOG2 background model used for foreground segmentation."""
    return cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=False,
    )


def clean_motion_mask(foreground_mask, kernel_size=5):
    """Remove small noise and close small gaps in a foreground mask."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    cleaned_mask = cv2.morphologyEx(
        foreground_mask,
        cv2.MORPH_OPEN,
        kernel,
    )
    return cv2.morphologyEx(
        cleaned_mask,
        cv2.MORPH_CLOSE,
        kernel,
    )


def find_motion_regions(motion_mask, min_area=150):
    """Return bounding boxes for connected moving regions."""
    contours, _ = cv2.findContours(
        motion_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    regions = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        regions.append(cv2.boundingRect(contour))
    return regions

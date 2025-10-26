import cv2
import numpy as np
from typing import Tuple

def calculate_border_irregularity(mask: np.ndarray, result_image: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Calculates border irregularity using multiple metrics:
    - Circularity
    - Convexity defects
    - Solidity
    - Extent
    Returns a composite irregularity score and an annotated image.
    """
    # Preprocess mask
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return result_image, 0.0

    # Select largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    if area < 100:
        return result_image, 0.0

    # Convex hull & defects
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    defects_score = 1 - (area / hull_area) if hull_area > 0 else 0

    # Circularity = 4πA / P²
    perimeter = cv2.arcLength(largest_contour, True)
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    circularity_score = 1 - circularity  # higher = more irregular

    # Solidity = area / hull area
    solidity = area / hull_area if hull_area > 0 else 0
    solidity_score = 1 - solidity

    # Extent = area / bounding rect area
    x, y, w, h = cv2.boundingRect(largest_contour)
    bounding_area = w * h
    extent = area / bounding_area if bounding_area > 0 else 0
    extent_score = 1 - extent

    # Combine all scores
    total_score = np.clip((circularity_score + defects_score + solidity_score + extent_score) / 4, 0, 1)

    # Annotate and return
    annotated = result_image.copy()
    cv2.drawContours(annotated, [largest_contour], -1, (0, 255, 0), 1)
    cv2.drawContours(annotated, [hull], -1, (0, 0, 255), 1)
    text = f"Irregularity: {total_score:.2f}"
    cv2.putText(annotated, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return annotated, total_score

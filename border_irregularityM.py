import cv2
import numpy as np
from typing import Tuple, Optional

def calculate_border_irregularity(result_image: np.ndarray) -> Tuple[np.ndarray, Optional[float], Optional[float]]:
    """
    Analyze the border irregularity of the mole from a filtered result image.
    
    Returns: output_image, irregularity_score, convexity_defect_ratio
    """
    # Create a binary mask from the result image
    gray = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Smooth and detect edges
    blurred = cv2.GaussianBlur(binary_mask, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found.")
        return result_image, None, None

    largest_contour = max(contours, key=cv2.contourArea)

    # Compute border metrics
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, closed=True)
    if perimeter == 0 or area == 0:
        print("Invalid contour.")
        return result_image, None, None
    
    # Circularity. Perfect circle has a circularity of 1
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    # Convexity defect ratio
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    convexity_defect_ratio = (hull_area - area) / hull_area if hull_area > 0 else 0
    # Irregularity score
    irregularity_score = 1 - circularity

    # Annotate the image
    output_image = result_image.copy()
    cv2.drawContours(output_image, [largest_contour], -1, (0, 255, 0), 2)  # green
    cv2.drawContours(output_image, [hull], -1, (255, 0, 0), 2)             # blue

    # Overlay text
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    text_color = (255, 255, 255)
    cv2.putText(output_image, f"Circularity: {circularity:.3f}", (10, 20), font, scale, text_color, 1)
    cv2.putText(output_image, f"Convexity Defect: {convexity_defect_ratio:.3f}", (10, 40), font, scale, text_color, 1)
    cv2.putText(output_image, f"Irregularity Score: {irregularity_score:.3f}", (10, 60), font, scale, text_color, 1)

    # Optional: Resize for consistent layout
    output_image = cv2.resize(output_image, (300, 300))

    return output_image, irregularity_score, convexity_defect_ratio

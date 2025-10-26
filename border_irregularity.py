import cv2
import numpy as np

def calculate_border_irregularity(result_image):
    """Analyzes border irregularity of the mole from the HSV-filtered result image (BGR format)."""

    # Convert image to binary mask
    # Background is black, but mole is in color; so detect non-black areas
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

    # Calculate border metrics
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, closed=True)
    if perimeter == 0 or area == 0:
        print("Invalid contour.")
        return result_image, None, None

    circularity = (4 * np.pi * area) / (perimeter ** 2)
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    convexity_defect_ratio = (hull_area - area) / hull_area if hull_area > 0 else 0
    irregularity_score = 1 - circularity

    # Draw contours and overlay text
    output_image = result_image.copy()
    cv2.drawContours(output_image, [largest_contour], -1, (0, 255, 0), 2)  # green = contour
    cv2.drawContours(output_image, [hull], -1, (255, 0, 0), 2)             # blue = convex hull

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    cv2.putText(output_image, f"Circularity: {circularity:.3f}", (10, 20), font, scale, (255, 255, 255), 1)
    cv2.putText(output_image, f"Convexity Defect: {convexity_defect_ratio:.3f}", (10, 40), font, scale, (255, 255, 255), 1)
    cv2.putText(output_image, f"Irregularity Score: {irregularity_score:.3f}", (10, 60), font, scale, (255, 255, 255), 1)

    # Optional: Resize for consistent display
    output_image = cv2.resize(output_image, (300, 300))

    return output_image, irregularity_score, convexity_defect_ratio















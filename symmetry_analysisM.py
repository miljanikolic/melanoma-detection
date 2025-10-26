import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def analyze_symmetry(result_image):
    """
    Analyze vertical and horizontal symmetry of a mole region.

    Returns: average_symmetry, vertical_symmetry, horizontal_symmetry, [comparison_images])
    If no contour is found, all values return as None.
    """
    blurred = cv2.bilateralFilter(result_image, 15, 75, 75)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found.")
        return None, None, None, [edges]

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    mole = result_image[y:y+h, x:x+w]

    # Split and flip for symmetry comparison
    left_half = mole[:, :w//2]
    right_half = mole[:, w//2:]
    right_half_flipped = cv2.flip(right_half, 1)
    right_half_flipped = cv2.resize(right_half_flipped, (left_half.shape[1], left_half.shape[0]))

    top_half = mole[:h//2, :]
    bottom_half = mole[h//2:, :]
    bottom_half_flipped = cv2.flip(bottom_half, 0)
    bottom_half_flipped = cv2.resize(bottom_half_flipped, (top_half.shape[1], top_half.shape[0]))

    # Compute SSIM - Structural Similarity Index (works on color with channel_axis=-1)
    vertical_symmetry, _ = ssim(left_half, right_half_flipped, full=True, channel_axis=-1)
    horizontal_symmetry, _ = ssim(top_half, bottom_half_flipped, full=True, channel_axis=-1)
    avg_symmetry = (vertical_symmetry + horizontal_symmetry) / 2

    Horiz1 = np.concatenate((left_half, right_half_flipped), axis=1)
    Horiz2 = np.concatenate((top_half, bottom_half_flipped), axis=1)

    return avg_symmetry, vertical_symmetry, horizontal_symmetry, [Horiz1, Horiz2, edges]

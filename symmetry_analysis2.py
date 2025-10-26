import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def analyze_symmetry(result_image):
    """
    Analyze vertical and horizontal symmetry of a mole region.

    Returns: avg_symmetry, vertical_symmetry, horizontal_symmetry, [comparison_images]
    """
    # Convert to grayscale for edge detection
    #gray = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to preserve edges while reducing noise
    #blurred = cv2.bilateralFilter(gray, 15, 75, 75)
    #edges = cv2.Canny(blurred, 50, 150)
    gray = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)


    # Find contours to isolate the mole region
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found.")
        return None, None, None, [edges]

    # Use the largest contour as the mole
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    mole_crop = gray[y:y+h, x:x+w]  # cropped grayscale mole

    # Split vertically
    left_half = mole_crop[:, :w//2]
    right_half = mole_crop[:, w//2:]
    right_half_flipped = cv2.flip(right_half, 1)
    right_half_flipped = cv2.resize(right_half_flipped, (left_half.shape[1], left_half.shape[0]))

    # Split horizontally
    top_half = mole_crop[:h//2, :]
    bottom_half = mole_crop[h//2:, :]
    bottom_half_flipped = cv2.flip(bottom_half, 0)
    bottom_half_flipped = cv2.resize(bottom_half_flipped, (top_half.shape[1], top_half.shape[0]))

    # Compute SSIM (grayscale)
    vertical_symmetry, _ = ssim(left_half, right_half_flipped, full=True)
    horizontal_symmetry, _ = ssim(top_half, bottom_half_flipped, full=True)
    avg_symmetry = (vertical_symmetry + horizontal_symmetry) / 2

    # Visualization for debugging
    Horiz1 = np.concatenate((left_half, right_half_flipped), axis=1)
    Horiz2 = np.concatenate((top_half, bottom_half_flipped), axis=1)

    # Return scores and comparison images
    return avg_symmetry, vertical_symmetry, horizontal_symmetry, [Horiz1, Horiz2, edges]

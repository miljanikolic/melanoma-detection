import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def ensure_color(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def analyze_symmetry(result_image):
    blurred = cv2.bilateralFilter(result_image, 15, 75, 75)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found.")
        return None, None, None, [edges]

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Make width and height even
    if w % 2 != 0:
        w -= 1
    if h % 2 != 0:
        h -= 1

    mole = result_image[y:y+h, x:x+w]

    # Split and flip
    left_half = mole[:, :w//2]
    right_half = mole[:, w//2:]
    right_half_flipped = cv2.flip(right_half, 1)
    right_half_flipped = cv2.resize(right_half_flipped, (left_half.shape[1], left_half.shape[0]))

    top_half = mole[:h//2, :]
    bottom_half = mole[h//2:, :]
    bottom_half_flipped = cv2.flip(bottom_half, 0)
    bottom_half_flipped = cv2.resize(bottom_half_flipped, (top_half.shape[1], top_half.shape[0]))

    # Ensure color
    left_half = ensure_color(left_half)
    right_half_flipped = ensure_color(right_half_flipped)
    top_half = ensure_color(top_half)
    bottom_half_flipped = ensure_color(bottom_half_flipped)

    # Compute SSIM
    vertical_symmetry, _ = ssim(left_half, right_half_flipped, full=True, channel_axis=-1)
    horizontal_symmetry, _ = ssim(top_half, bottom_half_flipped, full=True, channel_axis=-1)

    # Clamp SSIM scores
    vertical_symmetry = max(0, min(vertical_symmetry, 1))
    horizontal_symmetry = max(0, min(horizontal_symmetry, 1))
    avg_symmetry = (vertical_symmetry + horizontal_symmetry) / 2

    Horiz1 = np.concatenate((left_half, right_half_flipped), axis=1)
    Horiz2 = np.concatenate((top_half, bottom_half_flipped), axis=1)

    return avg_symmetry, vertical_symmetry, horizontal_symmetry, [Horiz1, Horiz2, edges]

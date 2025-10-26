import cv2
import numpy as np

def auto_isolate_mole_from_center(image_bgr, patch_size=20, margin=10, show_steps=False):
    """
    Isolates mole by sampling HSV values from the center region of the image.

    Parameters:
    - image_bgr: BGR input image (from cv2.imread)
    - patch_size: size of the square patch to sample at the center
    - margin: amount added/subtracted to min/max HSV to allow tolerance
    - show_steps: if True, show intermediate images

    Returns:
    - mole_only: Mole region in color
    - mole_mask: Binary mask of mole
    """
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    height, width = image_hsv.shape[:2]

    # Get center patch
    cx, cy = width // 2, height // 2
    half = patch_size // 2
    patch = image_hsv[cy - half:cy + half, cx - half:cx + half]

    # Reshape to list of pixels
    pixels = patch.reshape(-1, 3)
    h_vals, s_vals, v_vals = pixels[:, 0], pixels[:, 1], pixels[:, 2]

    # Compute min/max with margin
    h_min = max(0, np.min(h_vals) - margin)
    h_max = min(179, np.max(h_vals) + margin)
    s_min = max(0, np.min(s_vals) - margin)
    s_max = min(255, np.max(s_vals) + margin)
    v_min = max(0, np.min(v_vals) - margin)
    v_max = min(255, np.max(v_vals) + margin)

    lower_bound = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upper_bound = np.array([h_max, s_max, v_max], dtype=np.uint8)

    # Create mole mask
    mole_mask = cv2.inRange(image_hsv, lower_bound, upper_bound)

    # Optional: clean the mask
    kernel = np.ones((5, 5), np.uint8)
    mole_mask = cv2.morphologyEx(mole_mask, cv2.MORPH_OPEN, kernel)
    mole_mask = cv2.morphologyEx(mole_mask, cv2.MORPH_CLOSE, kernel)

    # Apply mask to original image
    mole = cv2.bitwise_and(image_bgr, image_bgr, mask=mole_mask)

    if show_steps:
        cv2.imshow("Original", image_bgr)
        cv2.imshow("Mole Mask", mole_mask)
        cv2.imshow("Isolated Mole", mole)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return mole, mole_mask

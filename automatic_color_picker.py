#on edges
import cv2
import numpy as np
import random

def auto_isolate_mole(image_bgr, edge_width=20, samples_per_edge=30, show_steps=False, save_hsv_bounds_path=None):
    """
    Automatically isolates the mole from the skin using edge-based HSV sampling.

    Parameters:
    - image_bgr: Original image in BGR color space
    - edge_width: Width of border to sample skin color from
    - samples_per_edge: Number of random pixels to sample per edge
    - show_steps: Show intermediate image steps
    - save_hsv_bounds_path: Optional file path to save HSV bounds

    Returns:
    - mole_only: Image with mole isolated
    - mole_mask: Binary mask of the mole
    - skin_mask: Binary mask of skin region
    """
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    height, width = image_hsv.shape[:2]

    hsv_samples = []

    def sample_edge(region):
        h, w = region.shape[:2]
        for _ in range(samples_per_edge):
            y = random.randint(0, h - 1)
            x = random.randint(0, w - 1)
            hsv = region[y, x]
            if not (hsv[0] < 10 and hsv[1] < 30 and hsv[2] < 30):  # Ignore black
                hsv_samples.append(hsv)

    # Sample HSV from all edges
    sample_edge(image_hsv[:edge_width, :])         # top
    sample_edge(image_hsv[-edge_width:, :])        # bottom
    sample_edge(image_hsv[:, :edge_width])         # left
    sample_edge(image_hsv[:, -edge_width:])        # right

    if not hsv_samples:
        raise ValueError("No valid skin samples found on edges.")

    hsv_samples = np.array(hsv_samples)
    lower_skin = np.min(hsv_samples, axis=0).astype(np.uint8)
    upper_skin = np.max(hsv_samples, axis=0).astype(np.uint8)

    # Optional: Save bounds
    if save_hsv_bounds_path:
        with open(save_hsv_bounds_path, "w") as f:
            f.write(f"{lower_skin[0]},{lower_skin[1]},{lower_skin[2]}\n")
            f.write(f"{upper_skin[0]},{upper_skin[1]},{upper_skin[2]}\n")

    # Step 2: Skin mask
    skin_mask = cv2.inRange(image_hsv, lower_skin, upper_skin)
    kernel = np.ones((7, 7), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

    # Step 3: Mole mask = NOT skin
    mole_mask = cv2.bitwise_not(skin_mask)

    # Step 4: Remove very bright spots (likely skin glare)
    v_channel = image_hsv[:, :, 2]
    v_nonzero = v_channel[v_channel > 0]
    brightness_threshold = np.percentile(v_nonzero, 90)
    bright_mask = cv2.inRange(v_channel, brightness_threshold, 255)
    mole_mask = cv2.bitwise_and(mole_mask, cv2.bitwise_not(bright_mask))

    # Clean and binarize mask
    mole_mask = cv2.GaussianBlur(mole_mask, (5, 5), 0)
    _, mole_mask = cv2.threshold(mole_mask, 127, 255, cv2.THRESH_BINARY)

    # Step 5: Apply mask to original image
    mole_only = cv2.bitwise_and(image_bgr, image_bgr, mask=mole_mask)

    if show_steps:
        cv2.imshow("Original", image_bgr)
        cv2.imshow("Skin Mask", skin_mask)
        cv2.imshow("Mole Mask", mole_mask)
        cv2.imshow("Isolated Mole", mole_only)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return mole_only, mole_mask, skin_mask

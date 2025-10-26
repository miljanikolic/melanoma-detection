import cv2
import numpy as np

def create_mole_mask(clustered_bgr, original_image, margin=20):
    """
    Create a mask selecting the mole region from clustered image by
    automatically selecting the darkest cluster color.

    Parameters:
        clustered_bgr (np.array): Clustered image in BGR color space.
        original_image (np.array): Original input image.
        margin (int): Margin to apply around the darkest cluster HSV values.

    Returns:
        mask (np.array): Binary mask for the mole region.
        mole_isolated (np.array): Original image masked with the mole region.
    """
    # Convert clustered image to HSV for color thresholding
    clustered_hsv = cv2.cvtColor(clustered_bgr, cv2.COLOR_BGR2HSV)

    # Find unique cluster colors (rows) in clustered_bgr
    # Reshape to Nx3 array of colors
    pixels = clustered_bgr.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)

    # Convert unique colors to HSV
    unique_colors_hsv = cv2.cvtColor(unique_colors.reshape(-1,1,3), cv2.COLOR_BGR2HSV).reshape(-1,3)

    # Find the darkest cluster by V channel (brightness)
    darkest_index = np.argmin(unique_colors_hsv[:,2])
    darkest_hsv = unique_colors_hsv[darkest_index]

    # Define lower and upper HSV bounds with margin (clamp between 0-255)
    lower_bound = np.clip(darkest_hsv - [margin, margin, margin], 0, 255)
    upper_bound = np.clip(darkest_hsv + [margin, margin, margin], 0, 255)

    lower_bound = lower_bound.astype(np.uint8)
    upper_bound = upper_bound.astype(np.uint8)

    # Create mask by thresholding clustered_hsv with these bounds
    mask = cv2.inRange(clustered_hsv, lower_bound, upper_bound)

    # Clean the mask with morphological operations
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Use mask to isolate the mole from original image
    mole_isolated = cv2.bitwise_and(original_image, original_image, mask=mask)

    return mask, mole_isolated

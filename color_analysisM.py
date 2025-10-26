import cv2
import numpy as np

def analyze_color(image):
    """Analyze color characteristics of a mole in the image, using HSV filtering and morphology."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Load HSV values from file
    try:
        with open("hsv_range.txt", "r") as file:
            h1, s1, v1 = map(int, file.readline().split())
            h2, s2, v2 = map(int, file.readline().split())
    except FileNotFoundError:
        print("HSV range file not found. Please run color_picker.py first.")
        return None, None

    lower_brown = np.array([min(h1, h2), min(s1, s2), min(v1, v2)])
    upper_brown = np.array([max(h1, h2), max(s1, s2), max(v1, v2)])

    print(f"Using HSV Range: Lower: {lower_brown}, Upper: {upper_brown}")

    # Create mask
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

    # Morphological closing to remove black pixels inside the mole
    # dilation followed by erosion
    kernel = np.ones((7, 7), np.uint8)
    mask_brown_cleaned = cv2.morphologyEx(mask_brown, cv2.MORPH_CLOSE, kernel)

    # Calculate the percentage
    brown_pixels = cv2.countNonZero(mask_brown_cleaned)
    total_pixels = image.shape[0] * image.shape[1]
    brown_percentage = (brown_pixels / total_pixels) * 100

    result_image = cv2.bitwise_and(image, image, mask=mask_brown_cleaned)

    #return brown_percentage, result_image
    return brown_percentage, result_image, lower_brown, upper_brown

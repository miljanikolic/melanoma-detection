import cv2
import numpy as np
import re

def analyze_color(image):
    """Analyze color characteristics of a mole in the image using HSV filtering and morphology."""

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Load HSV values from hsv_ranges.txt (mole range)
    try:
        with open("hsv_ranges.txt", "r") as file:
            lines = file.readlines()
            mole_lower = list(map(int, re.findall(r'\d+', lines[2])))
            mole_upper = list(map(int, re.findall(r'\d+', lines[3])))
    except FileNotFoundError:
        print("HSV range file not found. Please run the auto HSV extractor script first.")
        return None, None, None, None

    lower_brown = np.array(mole_lower)
    upper_brown = np.array(mole_upper)

    print(f"Using Mole HSV Range: Lower: {lower_brown}, Upper: {upper_brown}")

    # Create mask
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

    # Morphological closing to remove black pixels inside the mole
    kernel = np.ones((7, 7), np.uint8)
    mask_brown_cleaned = cv2.morphologyEx(mask_brown, cv2.MORPH_CLOSE, kernel)

    # Calculate the percentage of brown pixels
    brown_pixels = cv2.countNonZero(mask_brown_cleaned)
    total_pixels = image.shape[0] * image.shape[1]
    brown_percentage = (brown_pixels / total_pixels) * 100

    result_image = cv2.bitwise_and(image, image, mask=mask_brown_cleaned)

    return brown_percentage, result_image, lower_brown, upper_brown

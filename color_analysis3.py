import cv2
import numpy as np

def read_hsv_range(file_path="hsv_range.txt"):
    """Read HSV lower and upper bounds from file."""
    try:
        with open(file_path, "r") as file:
            lines = file.readlines()
            lower = np.array([int(x) for x in lines[0].strip().split()])
            upper = np.array([int(x) for x in lines[1].strip().split()])
            return lower, upper
    except FileNotFoundError:
        print("HSV range file not found. Please run color_picker.py first.")
        return None, None

def analyze_color(image):
    """Analyze color characteristics of a mole in the image using HSV range from file."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_brown, upper_brown = read_hsv_range()
    if lower_brown is None or upper_brown is None:
        return None, None

    print(f"Using HSV Range:\nLower: {lower_brown}\nUpper: {upper_brown}")
    
    # Create a mask based on the HSV range
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

    # ✨ Apply morphological closing to fill black holes inside the mole
    kernel = np.ones((7, 7), np.uint8)
    mask_brown_cleaned = cv2.morphologyEx(mask_brown, cv2.MORPH_CLOSE, kernel)

    # Calculate brown pixel percentage
    brown_pixels = cv2.countNonZero(mask_brown_cleaned)
    total_pixels = image.shape[0] * image.shape[1]
    brown_percentage = (brown_pixels / total_pixels) * 100

    # Visualize the cleaned mask
    result_image = cv2.bitwise_and(image, image, mask=mask_brown_cleaned)

    return brown_percentage, result_image

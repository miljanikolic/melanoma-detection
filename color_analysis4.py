










import cv2
import numpy as np

def analyze_mole_color(image):
    """
    Analyzes the color distribution of the mole in an image.

    Args:
        image (np.ndarray): The input image (BGR format).

    Returns:
        dict: A dictionary with color percentages (brown, black, red).
    """
    # Convert image to HSV for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Updated color ranges using the HSV value you identified
    lower_brown = np.array([170, 100, 30])  # Adjusted lower bound
    upper_brown = np.array([180, 200, 100])  # Adjusted upper bound

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])

    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for each color
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    # Calculate color percentages
    total_pixels = hsv.size // 3

    brown_percentage = np.sum(mask_brown > 0) / total_pixels * 100
    black_percentage = np.sum(mask_black > 0) / total_pixels * 100
    red_percentage = np.sum(mask_red > 0) / total_pixels * 100

    # Return the results
    return {
        'brown_percentage': round(brown_percentage, 2),
        'black_percentage': round(black_percentage, 2),
        'red_percentage': round(red_percentage, 2)
    }

# Test
if __name__ == '__main__':
    img = cv2.imread(r"D:\1) FAKS\Diplomski pocetna verzija\data\test\benign\2.jpg")  # Replace with your test image
    color_analysis = analyze_mole_color(img)
    print("Color Analysis:", color_analysis)



import cv2
import numpy as np
import matplotlib.pyplot as plt
from color_analysis import analyze_color  # This should return HSV image and HSV bounds

def calculate_brown_color_variation(image):
    # Use the analyze_color function to isolate the mole and get HSV bounds
    try:
        _, _, lower_brown, upper_brown = analyze_color(image)
    except Exception as e:
        print(f"Error in color analysis: {e}")
        return 0.0, None, None, None

    # Ensure image is in HSV format
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Create mask using HSV bounds
    mask = cv2.inRange(hsv, lower_brown, upper_brown)
    # Extract brown pixels
    brown_pixels = hsv[mask > 0]

    if brown_pixels.size == 0:
        print("No brown pixels found.")
        return 0.0, None, None, None

    # Extract hue channel
    h_values = brown_pixels[:, 0]

    # Calculate histogram (not normalized)
    hist, bins = np.histogram(h_values, bins=20, range=(0, 180), density=True)

    # Print histogram info
    for i in range(len(hist)):
        print(f"Hue range {int(bins[i])}-{int(bins[i+1])}: {hist[i]} pixels")

    # Calculate variation score
    variation_score = np.std(hist)  #racuna standardnu devijaciju

    return variation_score, hist, bins, mask

def plot_brown_histogram(hist, bins):
    bin_width = bins[1] - bins[0]
    plt.bar(bins[:-1], hist, width=bin_width, align='edge', color='brown', edgecolor='black')
    plt.xlabel('Hue Value')
    plt.ylabel('Pixel Count')
    plt.title('Histogram of Brown Hue Values')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

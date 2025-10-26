import cv2
import numpy as np

def analyze_color_features(mole_isolated):
    #hsv = cv2.cvtColor(mole_isolated, cv2.COLOR_BGR2HSV)
    h, w = mole_isolated.shape[:2]
    total_pixels = h * w

    # 1. Brown color ratio
    lower_brown = np.array([5, 50, 50])
    upper_brown = np.array([30, 255, 255])
    mask_brown = cv2.inRange(mole_isolated, lower_brown, upper_brown)
    brown_ratio = np.sum(mask_brown > 0) / total_pixels

    # 2. Number of dominant colors (K-means)
    pixels = mole_isolated.reshape(-1, 3)
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, K=5, bestLabels=None, criteria=criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
    num_colors = len(np.unique(labels))

    # 3. Percentage of black pixels
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 40])
    mask_black = cv2.inRange(mole_isolated, lower_black, upper_black)
    percent_black = np.sum(mask_black > 0) / total_pixels

    # 4. Percentage of red pixels
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(mole_isolated, lower_red1, upper_red1) | cv2.inRange(mole_isolated, lower_red2, upper_red2)
    percent_red = np.sum(mask_red > 0) / total_pixels

    # 5. Color symmetry (hue histogram correlation)
    hue = mole_isolated[:, :, 0]
    left_half = hue[:, :w // 2]
    right_half = hue[:, w // 2:]
    hist_left = cv2.calcHist([left_half], [0], None, [180], [0, 180])
    hist_right = cv2.calcHist([right_half], [0], None, [180], [0, 180])
    color_symmetry = cv2.compareHist(hist_left, hist_right, cv2.HISTCMP_CORREL)

    # 6. Scoring system
    score = 0
    if brown_ratio > 0.8: score += 1
    if num_colors <= 2: score += 1
    if percent_black < 0.1: score += 1
    if percent_red < 0.1: score += 1
    if color_symmetry > 0.8: score += 1

    classification = "Benign" if score >= 4 else "Malignant"

    return {
        "brown_ratio": round(brown_ratio, 3),
        "num_colors": num_colors,
        "percent_black": round(percent_black, 3),
        "percent_red": round(percent_red, 3),
        "color_symmetry": round(color_symmetry, 3),
        "color_score": score,
        "color_classification": classification
    }

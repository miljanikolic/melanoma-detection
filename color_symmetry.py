import cv2
import numpy as np
from plot_symmetry import plot_color_symmetry_histograms


def analyze_color_symmetry(rotated_cropped_mole):
    left_hist, right_hist, top_hist, bottom_hist = halves(rotated_cropped_mole)
    plot_color_symmetry_histograms(left_hist, right_hist, top_hist, bottom_hist)
    correlation_lr = cv2.compareHist(left_hist, right_hist, cv2.HISTCMP_CORREL)
    correlation_tb = cv2.compareHist(top_hist, bottom_hist, cv2.HISTCMP_CORREL)
    
    print("Color Symmetry Scores:")
    print(f"Correlation lr color: {correlation_lr:.4f} (higher → more symmetric)")
    print(f"Correlation tb color: {correlation_tb:.4f} (higher → more symmetric)")
    return correlation_lr, correlation_tb

def halves(rotated_cropped_mole):
    #DELIM KROPOVANU MASKU UZDUZNO NA DVA JEDNAKA DELA
    hsv = cv2.cvtColor(rotated_cropped_mole, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]  # Extract Hue
    height, width = hue.shape
    left_half = hue[:, :width // 2]
    right_half = hue[:, width // 2:]
    right_half_flipped = cv2.flip(right_half, 1)
    left_hist = cv2.calcHist([left_half], [0], None, [30], [0, 180])
    right_hist = cv2.calcHist([right_half_flipped], [0], None, [30], [0, 180])
    left_hist = cv2.normalize(left_hist, None, 0, 1, cv2.NORM_MINMAX)
    right_hist = cv2.normalize(right_hist, None, 0, 1, cv2.NORM_MINMAX)

    #DELIM KROPOVANU MASKU HORIZONTALNO NA DVA JEDNAKA DELA
    top_half = hsv[:height // 2, :]
    bottom_half = hsv[height // 2:, :]
    bottom_half_flipped = cv2.flip(bottom_half, 0)

    top_hue = top_half[:, :, 0]
    bottom_hue = bottom_half_flipped[:, :, 0]
    
    top_hist = cv2.calcHist([top_hue], [0], None, [30], [0, 180])
    bottom_hist = cv2.calcHist([bottom_hue], [0], None, [30], [0, 180])

    top_hist = cv2.normalize(top_hist, None, 0, 1, cv2.NORM_MINMAX)
    bottom_hist = cv2.normalize(bottom_hist, None, 0, 1, cv2.NORM_MINMAX)

    
    cv2.imshow("Left half", left_half)
    cv2.moveWindow("Left half", 100, 100)  # Move to (100,100)
    cv2.imshow("Right half flipped", right_half_flipped)
    cv2.moveWindow("Right half flipped", 450, 100) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return left_hist, right_hist, top_hist, bottom_hist

import cv2
import numpy as np

def analyze_symmetry(rotated_mask):
    left_hist, right_hist, top_hist, bottom_hist = halves(rotated_mask)
    correlation1 = cv2.compareHist(left_hist, right_hist, cv2.HISTCMP_CORREL)
    correlation2 = cv2.compareHist(top_hist, bottom_hist, cv2.HISTCMP_CORREL)
    avg_correlation = (correlation1 + correlation2) / 2
    
    print("Symmetry (histograms) Scores:")
    print(f"Correlation lr: {correlation1:.4f} (higher → more symmetric)")
    print(f"Correlation tb: {correlation2:.4f} (higher → more symmetric)")
    print(f"Average Correlation: {avg_correlation:.4f} (higher → more symmetric)")

    return correlation1, correlation2, avg_correlation

def halves(rotated_mask):
    # Getting two halves - left and right from cropped picture (they are equal size)
    height, width = rotated_mask.shape
    left_half = rotated_mask[:, :width // 2]
    right_half = rotated_mask[:, width // 2:]
    right_half_flipped = cv2.flip(right_half, 1)
    sum_left = np.sum(left_half > 0, axis=0).astype(np.float32)
    sum_right = np.sum(right_half_flipped > 0, axis=0).astype(np.float32)
    min_len1 = min(len(sum_left), len(sum_right))
    sum_left = sum_left[:min_len1]
    sum_right = sum_right[:min_len1]
    left_hist = cv2.normalize(sum_left, None, 0, 1, cv2.NORM_MINMAX)
    right_hist = cv2.normalize(sum_right, None, 0, 1, cv2.NORM_MINMAX)

    # Getting two halves - top and bottom from cropped picture (they are equal size)
    top_half = rotated_mask[:height // 2, :]
    bottom_half = rotated_mask[height // 2:, :]
    bottom_half_flipped = cv2.flip(bottom_half, 0)
    sum_top = np.sum(top_half > 0, axis=1).astype(np.float32)
    sum_bottom = np.sum(bottom_half_flipped > 0, axis=1).astype(np.float32)

    min_len2 = min(len(sum_top), len(sum_bottom))
    sum_top = sum_top[:min_len2]
    sum_bottom = sum_bottom[:min_len2]
    top_hist = cv2.normalize(sum_top, None, 0, 1, cv2.NORM_MINMAX)
    bottom_hist = cv2.normalize(sum_bottom, None, 0, 1, cv2.NORM_MINMAX)
    
    return left_hist, right_hist, top_hist, bottom_hist

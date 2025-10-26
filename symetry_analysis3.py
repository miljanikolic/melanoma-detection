import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def analyze_symmetry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 15, 75, 75)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0, image  # Fallback

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    mole = gray[y:y+h, x:x+w]

    left = mole[:, :w//2]
    right = mole[:, w//2:]
    right_flipped = cv2.flip(right, 1)
    right_flipped = cv2.resize(right_flipped, (left.shape[1], left.shape[0]))

    top = mole[:h//2, :]
    bottom = mole[h//2:, :]
    bottom_flipped = cv2.flip(bottom, 0)
    bottom_flipped = cv2.resize(bottom_flipped, (top.shape[1], top.shape[0]))

    vertical_score, _ = ssim(left, right_flipped, full=True)
    horizontal_score, _ = ssim(top, bottom_flipped, full=True)
    symmetry_score = (vertical_score + horizontal_score) / 2

    annotated = cv2.cvtColor(mole, cv2.COLOR_GRAY2BGR)
    return symmetry_score, annotated

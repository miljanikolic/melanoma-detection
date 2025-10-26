import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_hue_entropy(rotated_cropped_mole):
    hsv = cv2.cvtColor(rotated_cropped_mole, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    hist, _ = np.histogram(hue, bins=30, range=(0, 179), density=False)
    probabilities = hist / np.sum(hist)

    print(hist)
    print(probabilities)

    hue_entropy = 0
    for p in probabilities:
        if p > 0:
            hue_entropy += p * np.log2(1/p)
            #entropy -= p * np.log2(p)
    return hue_entropy 


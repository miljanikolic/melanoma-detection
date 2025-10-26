import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from border_irregularity import calculate_border_irregularity

# Load the image
img_path = r"D:\1) FAKS\Diplomski pocetna verzija\data\test\benign\8.jpg"  
img_bgr = cv2.imread(img_path)

# Analyze border irregularity
border_scores = calculate_border_irregularity(img_bgr)
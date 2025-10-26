import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Load the image
img_path = r"D:\1) FAKS\Diplomski pocetna verzija\data\test\benign\8.jpg"  
img_bgr = cv2.imread(img_path)

# Convert to grayscale
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


#clahe = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(5,5))
#contrast_img = clahe.apply(gray)
#cv2.imshow("CLAHE Enhanced", contrast_img)

alpha = 1.8  # Contrast control (1.0-3.0)
beta = 10    # Brightness control (0-100)

contrast_img = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)            # treba da nadjem prave vrednosti za alfu i betu, ali problem je sto mi na 2. slici
cv2.imshow("Increased Contrast", contrast_img)                              # vidi crne uglove, takodje slika 5 sa dve crtice sa strane



# Apply Gaussian blur
#blurred_gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
blurred_bilateral = cv2.bilateralFilter(contrast_img, 3, 50, 50)        #isto nije lose
blurred_bilatera2 = cv2.bilateralFilter(contrast_img, 5, 50, 50)        #ovo mi je najbolje

edges1 = cv2.Canny(blurred_bilateral, 50, 150)
edges2 = cv2.Canny(blurred_bilatera2, 50, 150)


#cv2.imshow("Gaus", blurred_gaussian)


cv2.imshow('Bilateral', blurred_bilateral)
cv2.imshow('Bilatera2', blurred_bilatera2)

cv2.imshow('Bilatera edge1', edges1)
cv2.imshow('Bilateraedge 2', edges2)
cv2.waitKey(0)
cv2.destroyAllWindows()
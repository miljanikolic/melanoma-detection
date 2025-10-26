import cv2
import numpy as np

# Load the image
img_path = r"D:\1) FAKS\Diplomski pocetna verzija\data\test\benign\8.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Different blur strengths
blur_3x3 = cv2.GaussianBlur(img, (3, 3), 0)
blur_5x5 = cv2.GaussianBlur(img, (5, 5), 1)
blur_7x7 = cv2.GaussianBlur(img, (7, 7), 2)
blurred = cv2.bilateralFilter(img, 9, 75, 75)

# Show results
cv2.imshow("Original", img)
cv2.imshow("Blur 3x3", blur_3x3)
cv2.imshow("Blur 5x5", blur_5x5)
cv2.imshow("Blur 7x7", blur_7x7)
cv2.imshow("Bilateral Filter", blurred)


cv2.waitKey(0)
cv2.destroyAllWindows()


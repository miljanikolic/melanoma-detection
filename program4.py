#### OVDE SAM PROBALA DIREKTNO S CV2 FUNKCIJOM

import cv2
import numpy as np

# Load the image
img_path = r"D:\1) FAKS\Diplomski pocetna verzija\data\test\benign\2.jpg"  
img_bgr = cv2.imread(img_path)

# Convert to grayscale
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
gray_enhanced = clahe.apply(gray)

# Apply Bilateral Filtering to reduce noise while preserving edges
blurred = cv2.bilateralFilter(gray_enhanced, d=9, sigmaColor=75, sigmaSpace=75)

# Use Sobel filter to detect gradient changes
sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
sobel_edges = cv2.magnitude(sobel_x, sobel_y)

# Normalize Sobel output for better visualization
sobel_edges = np.uint8(255 * sobel_edges / np.max(sobel_edges))

# Apply Laplacian for additional edge detection
laplacian_edges = cv2.Laplacian(blurred, cv2.CV_64F)
laplacian_edges = np.uint8(np.absolute(laplacian_edges))

# Combine Sobel and Laplacian edges
combined_edges = cv2.addWeighted(sobel_edges, 0.5, laplacian_edges, 0.5, 0)

# Apply Canny Edge Detection with adaptive thresholding
edges = cv2.Canny(combined_edges, 30, 150)  # Adjust thresholds as needed

# Apply morphological closing to refine edges
kernel = np.ones((3,3), np.uint8)
edges_refined = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Display results
cv2.imshow('Original Image', img_bgr)
cv2.imshow('Enhanced Grayscale', gray_enhanced)
cv2.imshow('Sobel + Laplacian Edges', combined_edges)
cv2.imshow('Canny Edge Detection', edges)
cv2.imshow('Refined Edges', edges_refined)

cv2.waitKey(0)
cv2.destroyAllWindows()




#import cv2
#import numpy as np
#import matplotlib.pyplot as plt
#from scipy import ndimage

# Load the image
#img_path = r"D:\1) FAKS\Diplomski pocetna verzija\data\test\benign\1.jpg"  # Change this to your image path
#img = cv2.imread(img_path)

# Convert to grayscale
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
#blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Resize the image (optional, but helps for consistent processing)
#gray_resized = cv2.resize(gray, (400, 400))

# Apply Canny edge detection
#edges = cv2.Canny(gray_resized, threshold1=100, threshold2=200)

# Display results 1
#fig, axes = plt.subplots(1, 3, figsize=(12, 4))
#axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#axes[0].set_title("Original Image")
#axes[0].axis("off")

#axes[1].imshow(gray, cmap="gray")
#axes[1].set_title("Grayscale Image")
#axes[1].axis("off")

#axes[2].imshow(blurred, cmap="gray")
#axes[2].set_title("Blurred Image")
#axes[2].axis("off")

# Display results 2
#plt.figure(figsize=(10, 5))
#plt.subplot(1, 2, 1)
#plt.imshow(gray_resized, cmap="gray")
#plt.title("Grayscale Image")
#plt.subplot(1, 2, 2)
#plt.imshow(edges, cmap="gray")
#plt.title("Canny Edge Detection")

# Find contours in the edges image
#contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create an empty image to draw the contours
#contour_img = cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2BGR)

# Draw all contours
#cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

# Find the largest contour by area
#largest_contour = max(contours, key=cv2.contourArea)

# Create a blank mask (same size as the image) filled with zeros (black)
#mask = np.zeros_like(gray_resized, dtype=np.uint8)
#mask = np.uint8(mask)  # Ensure the mask is of type uint8

# Draw the largest contour on the mask (white color)
#cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

# Mask the original image to keep only the mole (using the mask)
#mole_image = cv2.bitwise_and(img, img, mask=mask)

# Display the result
#plt.figure(figsize=(10, 5))
#plt.subplot(1, 2, 1)
#plt.imshow(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))
#plt.title("Mask of Mole")

#plt.subplot(1, 2, 2)
#plt.imshow(cv2.cvtColor(mole_image, cv2.COLOR_BGR2RGB))
#plt.title("Mole Isolated")



#plt.figure(figsize=(10, 5))
#plt.imshow(contour_img)
#plt.title("Contours of Mole")

# The function cv2.imshow() is used to display an image in a window.
#cv2.imshow("Grayscale Image", gray)
#cv2.imshow('Canny Edge Detection', edges)
# waitKey() waits for a key press to close the window and 0 specifies indefinite loop
#cv2.waitKey(0)
# cv2.destroyAllWindows() simply destroys all the windows we created.
#cv2.destroyAllWindows()

# Apply Canny edge detection
#edges = cv2.Canny(gray_resized, threshold1=100, threshold2=200)

# Find contours in the edges image
#ontours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour by area
#largest_contour = max(contours, key=cv2.contourArea)

# Create a blank mask (same size as the image) filled with zeros (black)
#mask = np.zeros_like(gray_resized, dtype=np.uint8)

# Draw the largest contour on the mask (white color)
#cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

# Resize mask to match original image dimensions
#mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

# Ensure the mask is uint8
#mask = np.uint8(mask)

# Mask the original image to keep only the mole (using the mask)
#mole_image = cv2.bitwise_and(img, img, mask=mask)

# Display the result
#plt.figure(figsize=(10, 5))

#plt.subplot(1, 2, 1)
#plt.imshow(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))
#plt.title("Mask of Mole")

#plt.subplot(1, 2, 2)
#plt.imshow(cv2.cvtColor(mole_image, cv2.COLOR_BGR2RGB))
#plt.title("Mole Isolated")

#plt.show()





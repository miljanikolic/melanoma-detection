import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from border_irregularity import calculate_border_irregularity

# Load the image
img_path = r"D:\1) FAKS\Diplomski pocetna verzija\data\test\benign\8.jpg"  
img_bgr = cv2.imread(img_path)

# Analyze border irregularity
border_scores = calculate_border_irregularity(img_bgr)

# Convert to grayscale
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
#blurred = cv2.GaussianBlur(gray, (5, 5), 0)
blurred = cv2.bilateralFilter(gray, 15, 75, 75)

# Apply Canny Edge Detection
edges = cv2.Canny(blurred, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Select the largest contour
if contours:
    largest_contour = max(contours, key=cv2.contourArea)

    # Get bounding box for the contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the mole region
    mole = gray[y:y+h, x:x+w]           #x i y su pocetne koord. u gornjem levom uglu, w-sirina, h-visina

    # Split into left and right halves
    left_half = mole[:, :w//2]
    right_half = mole[:, w//2:]

    # Flip the right half for comparison
    right_half_flipped = cv2.flip(right_half, 1)
    right_half_flipped = cv2.resize(right_half_flipped, (left_half.shape[1], left_half.shape[0]))


    top_half = mole[:h//2, :]
    bottom_half = mole[h//2:, :]
    bottom_half_flipped = cv2.flip(bottom_half, 0) 
    bottom_half_flipped = cv2.resize(bottom_half_flipped, (top_half.shape[1], top_half.shape[0]))     #ovo resava problem kada nisu istih dimenzija dve polovine slike(neparni br piksela)


    # Compare halves using Structural Similarity Index (SSIM)
    vertical_symmetry_score, _ = ssim(left_half, right_half_flipped, full=True)
    horizontal_symmetry_score, _ = ssim(top_half, bottom_half_flipped, full=True)
    
    # Final asymmetry score (lower score = more asymmetry)
    symmetry_score = (vertical_symmetry_score + horizontal_symmetry_score)/2

    #print(f"Asymmetry Score (0 to 1, higher means more symmetric): {asymmetry_score}")
    print(f"Vertical Symmetry Score: {vertical_symmetry_score}")
    print(f"Horizontal Symmetry Score: {horizontal_symmetry_score}")
    print(f"Symmetry Score: {symmetry_score}")

    # concatenate image Horizontally 
    Horiz1 = np.concatenate((left_half, right_half_flipped), axis=1) 
    Horiz2 = np.concatenate((top_half, bottom_half_flipped), axis=1) 
    
    # Display results
    cv2.imshow("Left Half and Right Half Flipped", Horiz1)
    #cv2.imshow("Flipped Right Half", right_half_flipped)
    cv2.imshow("Top Half and Bottom Half Flipped", Horiz2)
    #cv2.imshow("Bottom Half Flipped", bottom_half_flipped)

    cv2.imshow("Grayscale Image", gray)
    cv2.imshow('Canny Edge Detection', edges)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No contours found.")


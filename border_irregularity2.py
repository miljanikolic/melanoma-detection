
import cv2
import numpy as np

img_path = r"D:\1) FAKS\Diplomski pocetna verzija\data\test\benign\8.jpg"  
img_bgr = cv2.imread(img_path)

def calculate_border_irregularity(image):
    """Analyzes the border irregularity of a mole from an image."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to smooth the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No contours found.")
        return None

    # Select the largest contour (assuming it's the mole)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate area and perimeter
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, closed=True)
    
    # Avoid division by zero
    if perimeter == 0 or area == 0:
        print("Invalid contour detected.")
        return None
    
    # **Circularity (compactness) Score**
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    
    # **Convex Hull Analysis (Roughness Score)**
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    
    # Convexity defect ratio
    if hull_area == 0:
        convexity_defect_ratio = 0
    else:
        convexity_defect_ratio = (hull_area - area) / hull_area
    
    # **Irregularity Score (1 - circularity)**
    irregularity_score = 1 - circularity  # Higher values indicate more irregular borders

    # **Display results**
    output_image = image.copy()
    cv2.drawContours(output_image, [largest_contour], -1, (0, 255, 0), 2)  # Green contour
    cv2.drawContours(output_image, [hull], -1, (255, 0, 0), 2)  # Blue convex hull

    cv2.imshow("Mole Border", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # **Print and return the scores**
    print(f"Circularity Score: {circularity:.3f} (Closer to 1 = More Circular)")
    print(f"Convexity Defect Ratio: {convexity_defect_ratio:.3f} (Higher = More Irregular)")
    print(f"Border Irregularity Score: {irregularity_score:.3f} (Higher = More Irregular)")
    
    return irregularity_score, convexity_defect_ratio


# Analyze border irregularity
border_scores = calculate_border_irregularity(img_bgr)



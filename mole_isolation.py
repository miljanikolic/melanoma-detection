import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def isolate(loaded_image):
    cv2.imshow("Original image", loaded_image)
    gray =cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7,7), 0)
    (T, threshInv) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cv2.imshow("Threshold", threshInv)
    print("Otsu's thresholding value: {}".format(T))
    cv2.imshow("Gray", gray)
    plot_otsu_histogram(gray, T)

    kernel = np.ones((7, 7), np.uint8)
    mole_mask = cv2.morphologyEx(threshInv, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mole_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)     
    
    if not contours:
        print("No contours found!")
        return loaded_image, mole_mask
    
    max_contour = max(contours, key=cv2.contourArea) 

    # Maybe useful later
    #M = cv2.moments(max_contour)
    #print(M)
    #cx = int(M['m10']/M['m00'])
    #cy = int(M['m01']/M['m00'])
    #print(f'Coordinates: cx={cx:.2f}, cy={cy:.2f}')

    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    (center_x, center_y), (width, height), angle = rect
    print(f"Bounding Box Size: width={width:.2f}, height={height:.2f}")
    print(f"Bounding Box Center: x={center_x:.2f}, y={center_y:.2f}")
    print(f"Rotation Angle: {angle:.2f} degrees")
    
    # For later work
    #area = cv2.contourArea(max_contour)
    
    contour_image = np.zeros_like(mole_mask)
    cv2.drawContours(contour_image, [max_contour], 0, color=255, thickness=1)
    cv2.imshow("Max Contour", contour_image)
    
    # Black canvas used for drawing the largest contour and filling the region inside
    # New mask is being created to ensure that only mole will be isolated
    cleaned_mask = np.zeros_like(mole_mask) 
    cv2.drawContours(cleaned_mask, [max_contour], 0, color=255, thickness=-1)  
    cv2.imshow("Cleaned mask", cleaned_mask)

    # Apply the mask to isolate mole
    masked = cv2.bitwise_and(loaded_image, loaded_image, mask=cleaned_mask)
    cv2.imshow("Output", masked)

    print(type(masked))
    print(type(cleaned_mask))

    #Rotated mask
    pil_mask = Image.fromarray(cleaned_mask)
    rotated_mole_mask = pil_mask.rotate(angle, expand=False)
    rotated_mole_mask = np.array(rotated_mole_mask)
    cv2.imshow("Rotated mask", rotated_mole_mask)

    #Rotated isolated mole
    pil_masked = Image.fromarray(masked)
    rotated_mole_masked = pil_masked.rotate(angle, expand=False)
    rotated_mole_masked = np.array(rotated_mole_masked)
    cv2.imshow("Output rotated", rotated_mole_masked)

    cv2.drawContours(masked, [box], 0, color = (0, 0, 255), thickness = 1)
    cv2.circle(masked, (int(center_x), int(center_y)), 4, color = (0, 0, 255), thickness = -1)
    cv2.imshow("Isolated Mole with, red center of rectangle and rectangle", masked)
    center = (int(center_x), int(center_y))  #from minAreaRect
    width = int(width)
    height = int(height)

    cropped_mole = crop_centered(rotated_mole_masked, center, width, height)
    cv2.imshow("Cropped Mole", cropped_mole)

    cropped_mask = crop_centered(rotated_mole_mask, center, width, height)
    cv2.imshow("Cropped Mole Maks", cropped_mask)
    cv2.waitKey(0)
    return cropped_mole, cropped_mask


def plot_otsu_histogram(gray_image, threshold_value):
    plt.figure(figsize=(8, 5))
    plt.hist(gray_image.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7, label='Pixel Intensity')
    plt.axvline(x=threshold_value, color='red', linestyle='--', label=f'Otsu Threshold = {threshold_value:.2f}')
    plt.title('Grayscale Histogram with Otsu Threshold')
    plt.xlabel('Pixel Intensity (0–255)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def crop_centered(matrix, center, width, height):
    cx, cy = center
    start_x = cx - width // 2
    start_y = cy - height // 2
    
    # To ensure the crop stays inside the image
    start_x = max(0, start_x)
    start_y = max(0, start_y)
    end_x = min(matrix.shape[1], start_x + width)
    end_y = min(matrix.shape[0], start_y + height)

    cropped = matrix[start_y:end_y, start_x:end_x]
    return cropped





#Ovde biram dve nijanse, prvo lower, pa higher value, koristim za color_analysis3 i main3

import cv2
import numpy as np

# Store selected HSV values
selected_hsv_values = []

# Mouse callback function
def pick_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_value = hsv[y, x]
        selected_hsv_values.append(hsv_value)
        print(f"Picked HSV Value {len(selected_hsv_values)}: {hsv_value}")

        # When two points are selected, save them to a file
        if len(selected_hsv_values) == 2:
            lower = np.minimum(selected_hsv_values[0], selected_hsv_values[1])
            upper = np.maximum(selected_hsv_values[0], selected_hsv_values[1])
            with open("hsv_range.txt", "w") as file:
                file.write(f"{lower[0]} {lower[1]} {lower[2]}\n")
                file.write(f"{upper[0]} {upper[1]} {upper[2]}\n")
            print("HSV range saved to hsv_range.txt.")
            cv2.destroyAllWindows()

# Load image
img = cv2.imread(r"D:\1) FAKS\Diplomski pocetna verzija\data\test\benign\8.jpg")

if img is None:
    raise ValueError("Image not found. Check the path.")

# Convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Create window and set callback
cv2.namedWindow('HSV Color Picker')
cv2.setMouseCallback('HSV Color Picker', pick_color)

print("Click two pixels in the image to select HSV lower and upper bounds.")

while True:
    cv2.imshow('HSV Color Picker', img)
    if cv2.waitKey(1) & 0xFF == 27 or len(selected_hsv_values) == 2:  # ESC to exit
        break

cv2.destroyAllWindows()



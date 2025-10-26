#Ovde biram vise tacaka

import cv2
import numpy as np

hsv_values = []

def pick_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_value = hsv[y, x]
        hsv_values.append(hsv_value)
        print(f"Picked HSV: {hsv_value}")

# Load image
img = cv2.imread(r"D:\1) FAKS\Diplomski pocetna verzija\data\test\benign\2.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.namedWindow('HSV Color Picker')
cv2.setMouseCallback('HSV Color Picker', pick_color)

print("Click on different parts of the mole to collect HSV values. Press ESC when done.")

while True:
    cv2.imshow('HSV Color Picker', img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cv2.destroyAllWindows()

# Save to file
if hsv_values:
    hsv_values = np.array(hsv_values)
    lower = np.min(hsv_values, axis=0)
    upper = np.max(hsv_values, axis=0)
    print(f"Lower HSV: {lower}, Upper HSV: {upper}")
    with open("hsv_range.txt", "w") as file:
        file.write(f"{lower[0]} {lower[1]} {lower[2]}\n")
        file.write(f"{upper[0]} {upper[1]} {upper[2]}")

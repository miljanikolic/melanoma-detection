import cv2
import numpy as np

# Function to update HSV range based on trackbar positions
def update_hsv_range(val):
    global lower_hue, upper_hue, lower_sat, upper_sat, lower_val, upper_val

    lower_hue = cv2.getTrackbarPos("Lower Hue", "HSV Picker")
    upper_hue = cv2.getTrackbarPos("Upper Hue", "HSV Picker")
    lower_sat = cv2.getTrackbarPos("Lower Saturation", "HSV Picker")
    upper_sat = cv2.getTrackbarPos("Upper Saturation", "HSV Picker")
    lower_val = cv2.getTrackbarPos("Lower Value", "HSV Picker")
    upper_val = cv2.getTrackbarPos("Upper Value", "HSV Picker")

# Load the image
#img_path = r"images/test_image.jpg"  # Change this path
#image = cv2.imread(img_path)

image = cv2.imread(r"D:\1) FAKS\Diplomski pocetna verzija\data\test\benign\8.jpg")

if image is None:
    raise FileNotFoundError("Image not found. Please check the path.")

# Convert image to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Create a window with trackbars for HSV adjustment
cv2.namedWindow("HSV Picker")
cv2.createTrackbar("Lower Hue", "HSV Picker", 0, 179, update_hsv_range)
cv2.createTrackbar("Upper Hue", "HSV Picker", 179, 179, update_hsv_range)
cv2.createTrackbar("Lower Saturation", "HSV Picker", 0, 255, update_hsv_range)
cv2.createTrackbar("Upper Saturation", "HSV Picker", 255, 255, update_hsv_range)
cv2.createTrackbar("Lower Value", "HSV Picker", 0, 255, update_hsv_range)
cv2.createTrackbar("Upper Value", "HSV Picker", 255, 255, update_hsv_range)

# Default HSV range
lower_hue, upper_hue = 0, 179
lower_sat, upper_sat = 0, 255
lower_val, upper_val = 0, 255

while True:
    # Apply mask with current HSV range
    lower_bound = np.array([lower_hue, lower_sat, lower_val])
    upper_bound = np.array([upper_hue, upper_sat, upper_val])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Display masked image with HSV values
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    display_image = masked_image.copy()
    hsv_text = f"HSV Range: {lower_bound} - {upper_bound}"
    cv2.putText(display_image, hsv_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("HSV Picker", display_image)

    # Save HSV values if 's' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        with open("hsv_values.txt", "w") as f:
            f.write(f"Lower HSV: {lower_bound.tolist()}\n")
            f.write(f"Upper HSV: {upper_bound.tolist()}\n")
        print("HSV values saved to hsv_values.txt")
    elif key == 27:  # ESC key to exit
        break

cv2.destroyAllWindows()

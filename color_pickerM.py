import cv2
import numpy as np

def pick_and_save_hsv(image, save_path="hsv_range.txt", window_name="HSV Color Picker"):
    """
    Opens an interactive window to pick multiple HSV colors from an image.
    Saves min and max HSV range to a file if any values are selected.
    
    Returns:
        (lower_brown, upper_brown values): Tuple of HSV numpy arrays, or (None, None) if nothing picked.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_values = []

    def pick_color(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            hsv_val = hsv[y, x]
            hsv_values.append(hsv_val)
            print(f"Picked HSV: {hsv_val}")

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, pick_color)

    print("Click on different parts of the mole to collect HSV values. Press ESC when done.")
    while True:
        cv2.imshow(window_name, image)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cv2.destroyAllWindows()

    if hsv_values:
        hsv_values = np.array(hsv_values)
        lower = np.min(hsv_values, axis=0)
        upper = np.max(hsv_values, axis=0)

        print(f"Lower HSV: {lower}, Upper HSV: {upper}")
        with open(save_path, "w") as file:
            file.write(f"{lower[0]} {lower[1]} {lower[2]}\n")
            file.write(f"{upper[0]} {upper[1]} {upper[2]}")
        return lower, upper
    else:
        print("No HSV values were selected.")
        return None, None

import cv2
import numpy as np

def keep_largest_contour(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return binary_mask
    largest = max(contours, key=cv2.contourArea)
    clean_mask = np.zeros_like(binary_mask)
    cv2.drawContours(clean_mask, [largest], -1, 255, cv2.FILLED)
    return clean_mask

def remove_small_regions(mask, min_area=1000):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(filtered, [cnt], -1, 255, cv2.FILLED)
    return filtered

def auto_isolate_mole(image_bgr, edge_width=20, min_blob_area=1000, show_steps=False):
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Step 1: Sample skin HSV from image edges
    def sample_skin_color():
        h_vals, s_vals, v_vals = [], [], []
        borders = [
            image_hsv[:edge_width, :], image_hsv[-edge_width:, :],
            image_hsv[:, :edge_width], image_hsv[:, -edge_width:]
        ]
        for region in borders:
            for row in region:
                for pixel in row:
                    h, s, v = pixel
                    if not (h < 10 and s < 30 and v < 30):  # Skip near-black
                        h_vals.append(h)
                        s_vals.append(s)
                        v_vals.append(v)
        if not h_vals:
            return None, None
        lower = (
            np.percentile(h_vals, 5),
            np.percentile(s_vals, 5),
            np.percentile(v_vals, 5)
        )
        upper = (
            np.percentile(h_vals, 95),
            np.percentile(s_vals, 95),
            np.percentile(v_vals, 95)
        )
        return np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8)

    lower_skin, upper_skin = sample_skin_color()
    if lower_skin is None:
        raise ValueError("No valid skin pixels found on edges.")

    # Step 2: Create skin mask and invert for mole mask
    skin_mask = cv2.inRange(image_hsv, lower_skin, upper_skin)
    kernel = np.ones((7, 7), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    mole_mask = cv2.bitwise_not(skin_mask)
    mole_mask = cv2.morphologyEx(mole_mask, cv2.MORPH_OPEN, kernel)

    # Step 3: Filter out small noisy blobs
    mole_mask = remove_small_regions(mole_mask, min_blob_area)

    # Step 4: Use Canny edge detection to enhance mole shape
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Step 5: Combine mole mask with edge mask
    edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    final_mask = cv2.bitwise_and(mole_mask, cv2.bitwise_not(edges_dilated))

    # Step 6: Keep only largest blob (likely the mole)
    #final_mask = keep_largest_contour(enhanced_mask)

    # Step 7: Apply to original image
    mole_only = cv2.bitwise_and(image_bgr, image_bgr, mask=final_mask)

    if show_steps:
        cv2.imshow("Original", image_bgr)
        cv2.imshow("Skin Mask", skin_mask)
        cv2.imshow("Initial Mole Mask", mole_mask)
        cv2.imshow("Canny Edges", edges)
        cv2.imshow("Enhanced Mole Mask", final_mask)
        cv2.imshow("Isolated Mole", mole_only)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return mole_only, final_mask, skin_mask


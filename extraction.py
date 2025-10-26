import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_mole(image, k=3, show_steps=True):
    blurred = cv2.medianBlur(image, 5)
    hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    Z = hsv_img.reshape(-1, 3).astype(np.float32)
    _, labels, centers = cv2.kmeans(Z, k, None,
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0),
                                    attempts=10, flags=cv2.KMEANS_PP_CENTERS)

    labels = labels.flatten()
    skin_index = -1
    min_distance = float('inf')

    for i in range(k):
        cluster_mask = (labels == i)
        cluster_pixels = hsv_img.reshape(-1, 3)[cluster_mask]

        if cluster_pixels.size == 0:
            continue

        h_mean = np.mean(cluster_pixels[:, 0])
        s_mean = np.mean(cluster_pixels[:, 1])
        v_mean = np.mean(cluster_pixels[:, 2])

        # Skin is usually high value, medium saturation
        distance = abs(s_mean - 60) + abs(v_mean - 200)         #60 i 200
        if distance < min_distance:
            min_distance = distance
            skin_index = i

    # Create skin mask
    skin_mask = np.zeros(labels.shape, dtype=np.uint8)
    skin_mask[labels == skin_index] = 255
    skin_mask = skin_mask.reshape(image.shape[:2])

    # Invert to get mole mask
    mole_mask = cv2.bitwise_not(skin_mask)

    # Clean up small regions
    kernel = np.ones((7, 7), np.uint8)
    mole_mask = cv2.morphologyEx(mole_mask, cv2.MORPH_OPEN, kernel)
    mole_mask = cv2.morphologyEx(mole_mask, cv2.MORPH_CLOSE, kernel)
    mole_mask = cv2.dilate(mole_mask, kernel, iterations=1)

    # Apply mask to original image
    mole = cv2.bitwise_and(image, image, mask=mole_mask)

    #skin = cv2.bitwise_and(image, image, mask=skin_mask)

    if show_steps:
        clustered = centers[labels.astype(int)].reshape(image.shape).astype(np.uint8)

        plt.figure(figsize=(18, 5))
        plt.subplot(1, 5, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original")

        plt.subplot(1, 5, 2)
        plt.imshow(clustered)
        plt.title("Clustered")

        plt.subplot(1, 5, 3)
        plt.imshow(skin_mask, cmap='gray')
        plt.title("Skin Mask")

        plt.subplot(1, 5, 4)
        plt.imshow(mole_mask, cmap='gray')
        plt.title("Mole Mask")

        plt.subplot(1, 5, 5)
        plt.imshow(cv2.cvtColor(mole, cv2.COLOR_BGR2RGB))
        plt.title("Extracted Mole")
        plt.tight_layout()
        plt.show()

    return mole, mole_mask                                                                  #, skin_mask

"""
def auto_get_brown_hsv_bounds(image_bgr, mole_mask, margin=10):
    
    #Automatically computes lower and upper HSV bounds for brown from the mole region.
    #Adds a margin to include nearby brown shades.
    
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mole_pixels = hsv[mole_mask > 0]

    if mole_pixels.size == 0:
        print("No mole pixels found in mask.")
        return np.array([0, 0, 0]), np.array([179, 255, 255])

    # Get min and max with optional margin
    h_min = max(np.min(mole_pixels[:, 0]) - margin, 0)
    s_min = max(np.min(mole_pixels[:, 1]) - margin, 0)
    v_min = max(np.min(mole_pixels[:, 2]) - margin, 0)

    h_max = min(np.max(mole_pixels[:, 0]) + margin, 179)
    s_max = min(np.max(mole_pixels[:, 1]) + margin, 255)
    v_max = min(np.max(mole_pixels[:, 2]) + margin, 255)

    lower_brown = np.array([h_min, s_min, v_min])
    upper_brown = np.array([h_max, s_max, v_max])

    return np.array([h_min, s_min, v_min], dtype=np.uint8), np.array([h_max, s_max, v_max], dtype=np.uint8)

    #return lower_brown, upper_brown
"""
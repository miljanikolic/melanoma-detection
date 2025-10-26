import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_hsv_histograms(hsv_image):
    h, s, v = cv2.split(hsv_image)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.hist(h.ravel(), bins=180, range=[0, 180], color='r')
    plt.title('Hue Histogram')

    plt.subplot(1, 3, 2)
    plt.hist(s.ravel(), bins=256, range=[0, 256], color='g')
    plt.title('Saturation Histogram')

    plt.subplot(1, 3, 3)
    plt.hist(v.ravel(), bins=256, range=[0, 256], color='b')
    plt.title('Value Histogram')

    plt.tight_layout()
    plt.show()

def plot_cluster_histograms(hsv_img, labels, num_clusters):
    for i in range(num_clusters):
        cluster_mask = (labels == i)
        cluster_pixels = hsv_img.reshape(-1, 3)[cluster_mask.flatten()]
        if cluster_pixels.size == 0:
            continue
        h_vals = cluster_pixels[:, 0]
        s_vals = cluster_pixels[:, 1]
        v_vals = cluster_pixels[:, 2]

        plt.figure(figsize=(12, 3))
        plt.suptitle(f'Cluster {i} HSV Histogram')

        plt.subplot(1, 3, 1)
        plt.hist(h_vals, bins=180, range=[0, 180], color='r')
        plt.title('Hue')

        plt.subplot(1, 3, 2)
        plt.hist(s_vals, bins=256, range=[0, 256], color='g')
        plt.title('Saturation')

        plt.subplot(1, 3, 3)
        plt.hist(v_vals, bins=256, range=[0, 256], color='b')
        plt.title('Value')

        plt.tight_layout()
        plt.show()

def create_mole_mask(hsv_image, original_image, v_threshold=120, min_size=300):
    h, s, v = cv2.split(hsv_image)
    
    _, mask = cv2.threshold(v, v_threshold, 255, cv2.THRESH_BINARY_INV)

    # Morphological operations
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Remove small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    new_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            new_mask[labels == i] = 255

    mole = cv2.bitwise_and(original_image, original_image, mask=new_mask)
    return new_mask, mole

def auto_isolate_mole(image, k=3, show_steps=True):
    blurred = cv2.medianBlur(image, 5)
    hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    Z = hsv_img.reshape((-1, 3)).astype(np.float32)
    _, labels, centers = cv2.kmeans(Z, k, None,
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0),
                                    attempts=10, flags=cv2.KMEANS_PP_CENTERS)

    clustered = centers[labels.flatten().astype(int)].reshape(image.shape).astype(np.uint8)
    #clustered_bgr = cv2.cvtColor(clustered, cv2.COLOR_HSV2BGR)  # Use HSV since k-means was done in HSV

    mole_mask, mole = create_mole_mask(clustered, image, v_threshold=120)
    
    if np.count_nonzero(mole_mask) == 0:
        print("Mole isolation failed or mask is empty.")

    skin_mask = cv2.bitwise_not(mole_mask)
    skin = cv2.bitwise_and(image, image, mask=skin_mask)
    clustered = cv2.cvtColor(clustered, cv2.COLOR_HSV2BGR)
    #clustered = cv2.cvtColor(clustered, cv2.COLOR_BGR2GRAY)
    #gray = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
    if show_steps:
        plt.figure(figsize=(16, 4))
        plt.subplot(1, 4, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")

        plt.subplot(1, 4, 2)
        plt.imshow(clustered)
        plt.title("Clustered HSV Image")

        plt.subplot(1, 4, 3)
        plt.imshow(mole_mask, cmap='gray')
        plt.title("Mole Mask")

        plt.subplot(1, 4, 4)
        plt.imshow(cv2.cvtColor(mole, cv2.COLOR_BGR2RGB))
        plt.title("Isolated Mole")

        plt.tight_layout()
        plt.show()

        plot_cluster_histograms(hsv_img, labels.flatten(), k)

    return mole, mole_mask, skin, labels, hsv_img, k




def plot_cluster_histograms(hsv_img, labels, num_clusters):
    for i in range(num_clusters):
        cluster_mask = (labels == i)
        cluster_pixels = hsv_img.reshape(-1, 3)[cluster_mask.flatten()]
        if cluster_pixels.size == 0:
            continue
        h_vals = cluster_pixels[:, 0]
        s_vals = cluster_pixels[:, 1]
        v_vals = cluster_pixels[:, 2]





"""
def visualize_clusters(labels, image_shape, k):
    
    #Create a visualization showing which pixel belongs to which cluster.
    
    label_image = labels.reshape(image_shape[:2])
    cluster_vis = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)

    # Assign a distinct color to each cluster index
    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(k)]

    for i in range(k):
        cluster_vis[label_image == i] = colors[i]

    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(cluster_vis, cv2.COLOR_BGR2RGB))
    plt.title("Cluster Assignments")
    plt.axis('off')
    plt.show()
"""

"""def fill_holes(mask):
    
    Fill small holes (black regions) inside the white mask.
    
    # Invert mask to get holes as foreground

    inverted = cv2.bitwise_not(mask)
    
    # Flood fill from the edges
    h, w = mask.shape
    floodfill_mask = np.zeros((h + 2, w + 2), np.uint8)
    floodfilled = inverted.copy()
    cv2.floodFill(floodfilled, floodfill_mask, (0, 0), 255)

    # Invert floodfilled image and combine with original mask
    filled_holes = cv2.bitwise_not(floodfilled)
    result = cv2.bitwise_or(mask, filled_holes)

    return result"""

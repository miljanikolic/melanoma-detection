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

def find_central_surfaces(labels, cluster_idx, image_shape, num_surfaces=1):
    """
    Finds the most central connected components (surfaces) inside a cluster.
    Returns masks of selected surfaces.
    """
    h, w = image_shape[:2]
    image_center = np.array([w / 2, h / 2])

    # 1. Binary mask for this cluster
    cluster_mask = (labels.flatten() == cluster_idx).astype(np.uint8)
    cluster_mask = cluster_mask.reshape(h, w)

    # 2. Find connected components
    num_labels, label_img = cv2.connectedComponents(cluster_mask)

    surface_scores = []

    for i in range(1, num_labels):  # Skip label 0 (background)
        component_mask = (label_img == i).astype(np.uint8)
        coords = np.column_stack(np.where(component_mask == 1))
        centroid = np.mean(coords, axis=0)[::-1]  # (x, y)

        dist = np.linalg.norm(np.array(centroid) - image_center)
        area = np.sum(component_mask)

        # Add score (closer and larger is better)
        score = dist - 0.05 * area
        surface_scores.append((score, component_mask))

    # 3. Sort and return top `num_surfaces` masks
    surface_scores.sort(key=lambda x: x[0])
    selected_masks = [m for _, m in surface_scores[:num_surfaces]]

    return selected_masks

def auto_isolate_mole(image, k=2, top_n_clusters=2, show_steps=True):
    blurred = cv2.medianBlur(image, 5)
    hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    Z = hsv_img.reshape((-1, 3)).astype(np.float32)
    _, labels, centers = cv2.kmeans(Z, k, None,
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0),
                                    attempts=10, flags=cv2.KMEANS_PP_CENTERS)

    h, w = image.shape[:2]
    image_center = np.array([w / 2, h / 2])

    cluster_scores = []
    for i in range(k):
        cluster_mask = (labels.flatten() == i).astype(np.uint8).reshape(h, w)
        coords = np.column_stack(np.where(cluster_mask == 1))
        if len(coords) == 0:
            continue
        centroid = np.mean(coords, axis=0)[::-1]  # (x, y)
        dist = np.linalg.norm(np.array(centroid) - image_center)
        area = np.sum(cluster_mask)
        score = dist - 0.01 * area  # lower score = better
        cluster_scores.append((score, i))

    # Select top N clusters closest to center
    cluster_scores.sort(key=lambda x: x[0])
    selected_clusters = [idx for _, idx in cluster_scores[:top_n_clusters]]

    final_mask = np.zeros((h, w), dtype=np.uint8)
    for cluster_idx in selected_clusters:
        surfaces = find_central_surfaces(labels, cluster_idx, (h, w), num_surfaces=1)
        for mask in surfaces:
            final_mask = cv2.bitwise_or(final_mask, mask)

    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mole = cv2.bitwise_and(image, image, mask=final_mask)
    skin_mask = cv2.bitwise_not(final_mask)
    skin = cv2.bitwise_and(image, image, mask=skin_mask)

    if show_steps:
        clustered = centers[labels.flatten().astype(int)].reshape(image.shape).astype(np.uint8)
        #label_mask = (labels.flatten() == 0).astype(np.uint8)
        #label_mask = label_mask.reshape(image.shape[:2]) * 255  # Scale for binary mask
        #cluster_0_image = cv2.bitwise_and(image, image, mask=label_mask)
        plt.figure(figsize=(16, 4))
        plt.subplot(1, 4, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")

        plt.subplot(1, 4, 2)
        plt.imshow(clustered)
        plt.title("Clustered HSV Image")

        plt.subplot(1, 4, 3)
        plt.imshow(final_mask, cmap='gray')
        plt.title("Mole Mask (from center surfaces)")

        plt.subplot(1, 4, 4)
        plt.imshow(cv2.cvtColor(mole, cv2.COLOR_BGR2RGB))
        plt.title("Isolated Mole")

        plt.tight_layout()
        plt.show()

    return mole, final_mask, skin, labels, hsv_img, k

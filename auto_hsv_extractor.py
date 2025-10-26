import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def auto_isolate_mole(image, k=3, show_steps=True):
    """
    Automatically isolate mole region using K-means and HSV histogram stats.
    Returns: isolated mole image, mole mask, skin image, cluster labels, HSV image, and number of clusters.
    """
    # Step 1: Preprocess
    blurred = cv2.medianBlur(image, 5)
    #lab_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Step 2: Flatten and K-means
    Z = hsv_img.reshape((-1, 3)).astype(np.float32)
    _, labels, centers = cv2.kmeans(Z, k, None,
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0),
                                    attempts=10, flags=cv2.KMEANS_PP_CENTERS)

    # Step 3: Convert cluster centers for debugging (optional)
    #centers_bgr = cv2.cvtColor(centers.reshape((1, k, 3)).astype(np.uint8), cv2.COLOR_LAB2BGR)
    #centers_hsv = cv2.cvtColor(centers_bgr, cv2.COLOR_BGR2HSV)[0]
        # Step 3.5: Select two clusters closest to image center
    image_center = np.array([image.shape[1] / 2, image.shape[0] / 2])  # (x, y)

    coordinates = np.indices((image.shape[0], image.shape[1])).transpose(1, 2, 0)  # shape: (H, W, 2)
    flat_coords = coordinates.reshape(-1, 2)

    distances = []
    for i in range(k):
        cluster_mask = (labels.flatten() == i)
        if not np.any(cluster_mask):
            continue
        cluster_coords = flat_coords[cluster_mask]
        centroid = np.mean(cluster_coords, axis=0)
        dist_to_center = np.linalg.norm(centroid - image_center)
        distances.append((i, dist_to_center))

    # Sort clusters by distance to center
    distances.sort(key=lambda x: x[1])
    mole_indices = [idx for idx, _ in distances[:2]]  # Take two closest clusters

    print("Selected central mole clusters:", mole_indices)

    # Step 4: Select mole clusters based on HSV histogram stats
    mole_indices = []
    hsv_pixels = hsv_img.reshape(-1, 3)

    print("Cluster HSV Statistics:")
    for i in range(k):
        cluster_mask = (labels.flatten() == i)
        cluster_pixels = hsv_pixels[cluster_mask]

        if cluster_pixels.size == 0:
            continue
        
        h_max = np.max(cluster_pixels[:, 0])
        h_mean = np.mean(cluster_pixels[:, 0])
        s_mean = np.mean(cluster_pixels[:, 1])
        v_mean = np.mean(cluster_pixels[:, 2])

        print(f"Cluster {i}: H={h_mean:.1f}, S={s_mean:.1f}, V={v_mean:.1f}")

        # Rule: likely mole = high S, low V, and brown/dark hue
        if s_mean > 60 and v_mean < 160 and (h_mean <= 30 or h_mean >= 160):
            mole_indices.append(i)

    print("Selected mole clusters:", mole_indices)

    # Step 5: Build binary mask from selected clusters
    mask = np.zeros(labels.shape, dtype=np.uint8)
    for idx in mole_indices:
        mask[labels.flatten() == idx] = 255
    mask = mask.reshape((image.shape[:2]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Step 6: Extract mole and skin
    mole = cv2.bitwise_and(image, image, mask=mask)
    skin_mask = cv2.bitwise_not(mask)
    skin = cv2.bitwise_and(image, image, mask=skin_mask)

    # Step 7: Visualizations
    if show_steps:
        clustered = centers[labels.flatten().astype(int)].reshape(image.shape).astype(np.uint8)
        #clustered_bgr = cv2.cvtColor(clustered, cv2.COLOR_LAB2BGR)

        plt.figure(figsize=(16, 4))
        plt.subplot(1, 4, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")

        plt.subplot(1, 4, 2)
        plt.imshow(clustered)         #(clustered_bgr, cv2.COLOR_BGR2RGB))
        plt.title("Clustered HSV Image")

        plt.subplot(1, 4, 3)
        plt.imshow(mask, cmap='gray')
        plt.title("Mole Mask")

        plt.subplot(1, 4, 4)
        plt.imshow(cv2.cvtColor(mole, cv2.COLOR_BGR2RGB))
        plt.title("Isolated Mole")

        plt.tight_layout()
        plt.show()

        # Cluster histograms
        plot_cluster_histograms(hsv_img, labels.flatten(), k)

    return mole, mask, skin, labels, hsv_img, k

#mole, mask, skin, labels.flatten()


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
#def apply_mask(img, mask):
#    result = cv2.bitwise_and(img, img, mask=mask)
#    return result


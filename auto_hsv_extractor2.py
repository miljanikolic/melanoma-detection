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

def auto_isolate_mole(image, k=3, show_steps=True):
    """
    Automatically isolate mole region using K-means with histogram analysis.
    Returns the isolated mole image, mask, skin image, and cluster labels.
    """
    blurred = cv2.medianBlur(image, 5)
    #lab_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Flatten for K-means
    Z = hsv_img.reshape((-1, 3)).astype(np.float32)
    _, labels, centers = cv2.kmeans(Z, k, None,
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0),
                                    attempts=10, flags=cv2.KMEANS_PP_CENTERS)

    # Convert cluster centers to HSV
    centers_bgr = cv2.cvtColor(centers.reshape((1, k, 3)).astype(np.uint8), cv2.COLOR_LAB2BGR)
    centers_hsv = cv2.cvtColor(centers_bgr, cv2.COLOR_BGR2HSV)[0]

   #OVO MENJAM
    mole_indices = []
    print("Cluster HSV Centers:")
    for i, hsv in enumerate(centers_hsv):
        h, s, v = hsv
        print(f"Cluster {i}: H={h}, S={s}, V={v}")
        if (0 <= h <= 25) and (s >= 60) and (v <= 180):
            mole_indices.append(i)

    print("Selected mole clusters:", mole_indices)
    

    # Step 5: Automatically pick mole-like clusters based on HSV histogram stats
    mole_indices = []
    print("Cluster HSV Centers and Statistics:")
    hsv_pixels = hsv_img.reshape(-1, 3)

    for i in range(k):
        cluster_mask = (labels.flatten() == i)
        cluster_pixels = hsv_pixels[cluster_mask]

        if cluster_pixels.size == 0:
            continue

        h_mean = np.mean(cluster_pixels[:, 0])
        s_mean = np.mean(cluster_pixels[:, 1])
        v_mean = np.mean(cluster_pixels[:, 2])

        print(f"Cluster {i}: H-mean={h_mean:.1f}, S-mean={s_mean:.1f}, V-mean={v_mean:.1f}")

        # Automatic rule: likely mole if value is darker than average and saturation is medium/high
        if s_mean > 60 and v_mean < 160 and (h_mean <= 30 or h_mean >= 160):  # supports both brown and very dark
            mole_indices.append(i)

    print("Selected mole clusters:", mole_indices)

    mask = np.zeros(labels.shape, dtype=np.uint8)
    for idx in mole_indices:
        mask[labels.flatten() == idx] = 255
    mask = mask.reshape((image.shape[:2]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    mole = cv2.bitwise_and(image, image, mask=mask)
    skin_mask = cv2.bitwise_not(mask)
    skin = cv2.bitwise_and(image, image, mask=skin_mask)

    if show_steps:
        clustered = centers[labels.flatten().astype(int)].reshape(image.shape).astype(np.uint8)


        plt.figure(figsize=(16, 4))
        plt.subplot(1, 4, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")

        plt.subplot(1, 4, 2)
        plt.imshow(cv2.cvtColor(clustered, cv2.COLOR_BGR2RGB))
        plt.title("Clustered LAB Image")

        plt.subplot(1, 4, 3)
        plt.imshow(mask, cmap='gray')
        plt.title("Mole Mask")

        plt.subplot(1, 4, 4)
        plt.imshow(cv2.cvtColor(mole, cv2.COLOR_BGR2RGB))
        plt.title("Isolated Mole")
        plt.tight_layout()
        plt.show()

        # Show histograms per cluster
        plot_cluster_histograms(hsv_img, labels.flatten(), k)

    return mole, mask, skin, labels, hsv_img, k
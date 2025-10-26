import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import math


def analyze_symmetry(result_image):
    debug_images = []

    # Smooth and detect edges
    blurred = cv2.bilateralFilter(result_image, 15, 75, 75)
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found.")
        return None, None, None, [edges]

    largest_contour = max(contours, key=cv2.contourArea)

    # Get rotated rectangle and angle
    rotated_rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rotated_rect)
    box = np.intp(box)

    # Calculate rotation angle from bottom edge
    angle = calculate_bottom_edge_angle(box)
    print(f"Rotating mole by {angle:.2f} degrees to align")

    # Draw rectangle on original image for debug
    debug_img_box = result_image.copy()
    if len(debug_img_box.shape) == 2:
        debug_img_box = cv2.cvtColor(debug_img_box, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(debug_img_box, [box], 0, (0, 255, 0), 2)
    debug_images.append(debug_img_box)

    # Rotate whole image by -angle
    (h, w) = result_image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(result_image, rotation_matrix, (w, h),
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    debug_images.append(rotated_image.copy())

    #blurred_rot = cv2.bilateralFilter(rotated_image, 15, 75, 75)
    #edges_rot = cv2.Canny(blurred_rot, 50, 150)

    # Find contours again on rotated image
    edges_rot = cv2.Canny(rotated_image, 50, 150)
    contours_rot, _ = cv2.findContours(edges_rot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_rot:
        print("No contours found after rotation.")
        return None, None, None, debug_images

    largest_contour_rot = max(contours_rot, key=cv2.contourArea)
    x, y, w_box, h_box = cv2.boundingRect(largest_contour_rot)

    if w_box % 2 != 0:
        w_box -= 1
    if h_box % 2 != 0:
        h_box -= 1

    mole = rotated_image[y:y + h_box, x:x + w_box]

    left_half = mole[:, :w_box // 2]
    right_half = mole[:, w_box // 2:]
    right_half_flipped = cv2.flip(right_half, 1)

    top_half = mole[:h_box // 2, :]
    bottom_half = mole[h_box // 2:, :]
    bottom_half_flipped = cv2.flip(bottom_half, 0)

    vertical_symmetry, _ = ssim(left_half, right_half_flipped, full=True, channel_axis=-1)
    horizontal_symmetry, _ = ssim(top_half, bottom_half_flipped, full=True, channel_axis=-1)

    vertical_symmetry = max(0, min(vertical_symmetry, 1))
    horizontal_symmetry = max(0, min(horizontal_symmetry, 1))
    avg_symmetry = (vertical_symmetry + horizontal_symmetry) / 2

    Horiz1 = np.concatenate((left_half, right_half_flipped), axis=1)
    Horiz2 = np.concatenate((top_half, bottom_half_flipped), axis=1)

    debug_images.append(Horiz1)
    debug_images.append(Horiz2)

    return avg_symmetry, vertical_symmetry, horizontal_symmetry, debug_images


def calculate_bottom_edge_angle(box):
    box = sorted(box, key=lambda p: p[1])                               #top_points = box[:2]
    bottom_points = box[2:]
    bottom_points = sorted(bottom_points, key=lambda p: p[0])
    (x1, y1) = bottom_points[0]
    (x2, y2) = bottom_points[1]
    dx = x2 - x1
    dy = y2 - y1
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    return angle_deg


#def ensure_color(img):
#    if len(img.shape) == 2:
#        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#    return img


def show_debug_images(images, window_name="Symmetry Debug View"):
    titles = [
        "Original with Rotated Rectangle",
        "Aligned Mole",
        "Vertical Symmetry",
        "Horizontal Symmetry"
    ]

    labeled_images = []
    for i, img in enumerate(images):
        img_resized = cv2.resize(img, (300, 300))
        label = titles[i] if i < len(titles) else f"Step {i+1}"
        img_labeled = img_resized.copy()
        cv2.putText(img_labeled, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)
        labeled_images.append(img_labeled)

    rows = []
    for i in range(0, len(labeled_images), 2):
        row = np.hstack(labeled_images[i:i+2])
        rows.append(row)

    combined = np.vstack(rows)
    cv2.imshow(window_name, combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



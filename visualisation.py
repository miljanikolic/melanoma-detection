import cv2
import numpy as np
from classificationM import classify_mole

    #Convert grayscale to BGR and resize image
def resize_and_convert_image(img, size=(300, 300)):

    if img is None:
        return None
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.resize(img, size)

    #Create a black image with title and optional score
def create_title_image(width, title, score=None, font=cv2.FONT_HERSHEY_SIMPLEX):
    title_img = np.zeros((60, width, 3), dtype=np.uint8)
    
    # Title text
    title_size = cv2.getTextSize(title, font, 0.6, 1)[0]
    title_x = (width - title_size[0]) // 2
    cv2.putText(title_img, title, (title_x, 22), font, 0.6, (255, 255, 255), 1)

    # Score text
    if score is not None:
        score_text = f"Score: {score:.2f}"
        score_size = cv2.getTextSize(score_text, font, 0.5, 1)[0]
        score_x = (width - score_size[0]) // 2
        cv2.putText(title_img, score_text, (score_x, 45), font, 0.5, (255, 255, 255), 1)

    return title_img

def show_analysis_results(image_list, title_list, score_list=None, classification_result=None, resize_dims_list=None):
    if not image_list or not title_list:
        print("Error: image_list and title_list must not be empty.")
        return

    if len(image_list) != len(title_list):
        print("Warning: image_list and title_list lengths do not match.")

    # Filter out None images
    filtered_images = []
    filtered_titles = []
    filtered_scores = []
    filtered_dims = []

    for i, img in enumerate(image_list):
        if img is None:
            print(f"Warning: Skipping None image at index {i} (title: {title_list[i]})")
            continue
        filtered_images.append(img)
        filtered_titles.append(title_list[i])
        if score_list: filtered_scores.append(score_list[i])
        if resize_dims_list: filtered_dims.append(resize_dims_list[i] if i < len(resize_dims_list) else None)

    if not filtered_images:
        print("No valid images to display.")
        return

    # Resize images
    resized_images = []
    for i, img in enumerate(filtered_images):
        resize_dims = filtered_dims[i] if resize_dims_list and i < len(filtered_dims) and filtered_dims[i] else (300, 300)
        resized_images.append(resize_and_convert_image(img, size=resize_dims))

    # Create title overlays
    title_images = []
    for i, img in enumerate(resized_images):
        title = filtered_titles[i]
        score = filtered_scores[i] if score_list and i < len(filtered_scores) else None
        title_img = create_title_image(img.shape[1], title, score)
        title_images.append(title_img)

    # Combine titles and images
    full_images = [np.vstack((title, img)) for title, img in zip(title_images, resized_images)]

    # Equalize heights
    max_height = max(img.shape[0] for img in full_images)
    equalized_images = [
        cv2.copyMakeBorder(img, 0, max_height - img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        if img.shape[0] < max_height else img
        for img in full_images
    ]

    result_row = np.hstack(equalized_images)

    # Add space for classification
    bottom_space = 50
    final_img = cv2.copyMakeBorder(result_row, 0, bottom_space, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Draw classification result
    if classification_result:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Final Classification: {classification_result}"
        text_size = cv2.getTextSize(text, font, 0.8, 2)[0]
        text_x = (final_img.shape[1] - text_size[0]) // 2
        text_y = result_row.shape[0] + 35
        cv2.putText(final_img, text, (text_x, text_y), font, 0.8, (255, 255, 255), 2)

    # Show final result
    cv2.imshow("Melanoma Analysis Results", final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



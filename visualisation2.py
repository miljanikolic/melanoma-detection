import cv2
import numpy as np
from classificationM import classify_mole

def show_analysis_results(image_list, title_list, score_list=None, classification_result=None, resize_dims_list=None):
    if not image_list or not title_list:
        print("Error: image_list and title_list must not be empty.")
        return

    if len(image_list) != len(title_list):
        print("Warning: image_list and title_list lengths do not match.")
    
    if score_list and len(score_list) != len(image_list):
        print("Warning: score_list length does not match image_list.")

    resized_images = []

    # Resize and prepare images
    for i, img in enumerate(image_list):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if resize_dims_list and i < len(resize_dims_list) and resize_dims_list[i] is not None:
            width, height = resize_dims_list[i]
        else:
            width, height = 300, 300  # Default size

        resized = cv2.resize(img, (width, height))
        resized_images.append(resized)

    font = cv2.FONT_HERSHEY_SIMPLEX
    title_height = 60
    title_images = []

    for i, img in enumerate(resized_images):
        width = img.shape[1]
        title_img = np.zeros((title_height, width, 3), dtype=np.uint8)

        # Safely get title and score
        title = title_list[i] if i < len(title_list) else f"Image {i+1}"
        score = score_list[i] if score_list and i < len(score_list) else None

        # Title
        title_size = cv2.getTextSize(title, font, 0.6, 1)[0]
        title_x = (width - title_size[0]) // 2
        cv2.putText(title_img, title, (title_x, 22), font, 0.6, (255, 255, 255), 1)

        # Score
        """
        if score is not None:
            score_text = f"Score: {score:.2f}"
            if score and score > 0.33:
                score_color = (0, 255, 0)
            elif score < 0.66:
                score_color = (0, 165, 255)
            else:
                score_color = (0, 0, 255)

            score_size = cv2.getTextSize(score_text, font, 0.5, 1)[0]
            score_x = (width - score_size[0]) // 2
            cv2.putText(title_img, score_text, (score_x, 45), font, 0.5, score_color, 1)

        title_images.append(title_img)
        """

        if score is not None:
            score_text = f"Score: {score:.2f}"
            score_color = (255, 255, 255)  # Always white

            score_size = cv2.getTextSize(score_text, font, 0.5, 1)[0]
            score_x = (width - score_size[0]) // 2
            cv2.putText(title_img, score_text, (score_x, 45), font, 0.5, score_color, 1)

        title_images.append(title_img)


    # Stack title and image
    full_images = [np.vstack((t, img)) for t, img in zip(title_images, resized_images)]

    # Equalize heights
    max_height = max(img.shape[0] for img in full_images)
    equalized_images = []
    for img in full_images:
        h, w = img.shape[:2]
        if h < max_height:
            pad = max_height - h
            img = cv2.copyMakeBorder(img, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        equalized_images.append(img)

    result_row = np.hstack(equalized_images)

    # Add space for classification
    bottom_space = 50
    final_img = cv2.copyMakeBorder(result_row, 0, bottom_space, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    
    # Classification text
    if classification_result:
        text = f"Final Classification: {classification_result}"
        color = (255, 255, 255)  # Always white
        text_size = cv2.getTextSize(text, font, 0.8, 2)[0]
        text_x = (final_img.shape[1] - text_size[0]) // 2
        text_y = result_row.shape[0] + 35
        cv2.putText(final_img, text, (text_x, text_y), font, 0.8, color, 2)

    # Show result
    cv2.imshow("Melanoma Analysis Results", final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

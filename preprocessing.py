import cv2
import numpy as np

def create_nonblack_mask(image, threshold=10):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold  # True for non-black pixels
    return mask.astype(np.uint8) * 255  # Convert to binary mask



def crop_black_borders(image, threshold=10):
    """
    Automatically crops black borders from an image.
    threshold: pixel intensity below which it's considered black
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    coords = cv2.findNonZero(thresh)  # Find all non-black pixels
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = image[y:y+h, x:x+w]
        return cropped
    else:
        return image  # Return original if no non-black found


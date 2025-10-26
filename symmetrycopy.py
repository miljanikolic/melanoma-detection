#https://stackoverflow.com/questions/53301086/how-to-identify-if-the-shapes-of-an-image-are-symmetric-or-asymmetric-using-open
import cv2
import numpy as np
from plot_symmetry import plot_symmetry_histograms



def analyze_symmetry(rotated_mask):
    left_hist, right_hist, top_hist, bottom_hist = halves(rotated_mask)
    #plot_symmetry_histograms(left_hist, right_hist, top_hist, bottom_hist)
    correlation1 = cv2.compareHist(left_hist, right_hist, cv2.HISTCMP_CORREL)
    correlation2 = cv2.compareHist(top_hist, bottom_hist, cv2.HISTCMP_CORREL)
    avg_correlation = (correlation1 + correlation2) / 2

    print("Symmetry (histograms) Scores:")
    print(f"Correlation lr: {correlation1:.4f} (higher → more symmetric)")
    print(f"Correlation tb: {correlation2:.4f} (higher → more symmetric)")
    print(f"Average Correlation: {avg_correlation:.4f} (higher → more symmetric)")

    return correlation1, correlation2, avg_correlation

def halves(rotated_mask):
        #DELIM KROPOVANU MASKU UZDUZNO NA DVA JEDNAKA DELA
    height, width = rotated_mask.shape
    left_half = rotated_mask[:, :width // 2]
    right_half = rotated_mask[:, width // 2:]
    right_half_flipped = cv2.flip(right_half, 1)
    sum_left = np.sum(left_half > 0, axis=0).astype(np.float32)
    sum_right = np.sum(right_half_flipped > 0, axis=0).astype(np.float32)

    min_len1 = min(len(sum_left), len(sum_right))
    sum_left = sum_left[:min_len1]
    sum_right = sum_right[:min_len1]

    left_hist = cv2.normalize(sum_left, None, 0, 1, cv2.NORM_MINMAX)
    right_hist = cv2.normalize(sum_right, None, 0, 1, cv2.NORM_MINMAX)


    #DELIM KROPOVANU MASKU HORIZONTALNO NA DVA JEDNAKA DELA
    top_half = rotated_mask[:height // 2, :]
    bottom_half = rotated_mask[height // 2:, :]
    bottom_half_flipped = cv2.flip(bottom_half, 0)
    sum_top = np.sum(top_half > 0, axis=1).astype(np.float32)
    sum_bottom = np.sum(bottom_half_flipped > 0, axis=1).astype(np.float32)

    min_len2 = min(len(sum_top), len(sum_bottom))
    sum_top = sum_top[:min_len2]
    sum_bottom = sum_bottom[:min_len2]
    top_hist = cv2.normalize(sum_top, None, 0, 1, cv2.NORM_MINMAX)
    bottom_hist = cv2.normalize(sum_bottom, None, 0, 1, cv2.NORM_MINMAX)
    
    #cv2.imshow("Left half", left_half)
    #cv2.moveWindow("Left half", 100, 100)  # Move to (100,100)
    #cv2.imshow("Right half flipped", right_half_flipped)
    #cv2.moveWindow("Right half flipped", 450, 100) 
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return left_hist, right_hist, top_hist, bottom_hist



    #chi_square1 = cv2.compareHist(left_hist, right_hist, cv2.HISTCMP_CHISQR)
    #bhattacharyya1 = cv2.compareHist(left_hist, right_hist, cv2.HISTCMP_BHATTACHARYYA)
    #chi_square2 = cv2.compareHist(top_hist, bottom_hist, cv2.HISTCMP_CHISQR)
    #bhattacharyya2 = cv2.compareHist(top_hist, bottom_hist, cv2.HISTCMP_BHATTACHARYYA)

    #print(f"Chi-Square lr: {chi_square1:.4f} (lower → more symmetric)")
    #print(f"Bhattacharyya Distance lr: {bhattacharyya1:.4f} (lower → more symmetric)")
    #print(f"Chi-Square tb: {chi_square2:.4f} (lower → more symmetric)")
    #print(f"Bhattacharyya Distance tb: {bhattacharyya2:.4f} (lower → more symmetric)")

"""
def analyze_symmetry(binary_image):
    #binary_image = binary_image.astype(np.uint8)  

    # Compute projections directly on whole binary image
    #G_X = np.sum(binary_image, axis=1).astype(np.float32)  # Horizontal projection
    #G_Y = np.sum(binary_image, axis=0).astype(np.float32)  # Vertical projection

    G_X = np.sum(binary_image, axis=1)  # Horizontal projection
    G_Y = np.sum(binary_image, axis=0)  # Vertical projection


    #G_X = cv2.reduce(binary_image, 0 ,cv2.REDUCE_SUM)
    #G_Y = cv2.reduce(binary_image, 1 ,cv2.REDUCE_SUM)

    #G_X = G_X.astype(np.float32).flatten().reshape(-1, 1)
    #G_Y = G_Y.astype(np.float32).flatten().reshape(-1, 1)

    # Normalize projections
    G_X = cv2.normalize(G_X, None, 0, 1, cv2.NORM_MINMAX)
    G_Y = cv2.normalize(G_Y, None, 0, 1, cv2.NORM_MINMAX)
    G_X = G_X.astype(np.float32).reshape(-1, 1)
    G_Y = G_Y.astype(np.float32).reshape(-1, 1)

    #chi_square = 0.5 * np.sum(((G_X - G_Y) ** 2) / (G_X + G_Y + 1e-10))

    # Poredimo histograme
    correlation = cv2.compareHist(G_X, G_Y, cv2.HISTCMP_CORREL)
    chi_square = cv2.compareHist(G_X, G_Y, cv2.HISTCMP_CHISQR)
    bhattacharyya = cv2.compareHist(G_X, G_Y, cv2.HISTCMP_BHATTACHARYYA)

    print("Symmetry Scores:")
    print(f"Correlation: {correlation:.4f} (higher → more symmetric)")
    print(f"Chi-Square: {chi_square:.4f} (lower → more symmetric)")
    print(f"Bhattacharyya Distance: {bhattacharyya:.4f} (lower → more symmetric)")

    return correlation, chi_square, bhattacharyya
"""
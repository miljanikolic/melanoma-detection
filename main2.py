import cv2
import numpy as np
#from color_analysis2 import analyze_color
from symmetry_analysis2 import analyze_symmetry
from visualisation2 import show_analysis_results
from classification5 import classify_mole
from brown_color_variation5 import calculate_brown_color_variation, plot_brown_histogram
from auto_hsv_extractor2 import auto_isolate_mole, plot_hsv_histograms  

def main():
    # === Step 0: Load Image ===
    image_path = r"D:\1) FAKS\Diplomski pocetna verzija\data\test\benign\8.jpg"
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load image.")
        return

    # === Step 1: Automatically Isolate Mole ===
    print("Isolating mole using K-means clustering...")
    mole_only, mask, skin, labels, hsv_img, k = auto_isolate_mole(img, k=3, show_steps=True)
    #mole, mole_mask, skin_mask = auto_isolate_mole(img, show_steps=True)
    #mole_masked = apply_mask(img, mask)  # use original image and final mask

    if mole_only is None or cv2.countNonZero(mask) == 0:
        print("Mole isolation failed or mask is empty.")
        return

    # === Step 2: Color Analysis ===
    print("Performing color analysis...")
    #brown_percentage, mole_result, lower_brown, upper_brown = analyze_color(mole_hsv)
    #if mole_result is None or cv2.countNonZero(cv2.cvtColor(mole_result, cv2.COLOR_BGR2GRAY)) == 0:
    #    print("Color analysis failed or detected region is empty.")
    #    return
    #print(f"Brown Percentage: {brown_percentage:.2f}%")

    # === Step 3: Brown Color Variation Analysis ===
    #print("Calculating brown color variation...")
    variation_score, hist, bins, mask = calculate_brown_color_variation(mole_only)
    #variation_score, hist, bins, mask = calculate_brown_color_variation(mole_masked, lower_brown, upper_brown)
    #print(f"Variation Score: {variation_score:.2f}")
    #print(f"Number of brown pixels: {np.count_nonzero(mask)}")
    #if hist is not None:
    #   plot_brown_histogram(hist, bins)

        #hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)   
    mole_only = cv2.cvtColor(mole_only, cv2.COLOR_BGR2HSV)
    plot_hsv_histograms(hsv_img)
    #plot_hsv_histograms(mole)

    # === Step 4: Symmetry Analysis ===
    print("Performing symmetry analysis...")
    try:
        symmetry_score, vert_score, horz_score, symmetry_images = analyze_symmetry(mole_only)
        #symmetry_score, vert_score, horz_score, symmetry_images = analyze_symmetry(mole_masked)
    except Exception as e:
        print(f"Symmetry analysis failed: {e}")
        symmetry_score = vert_score = horz_score = 0.0
        symmetry_images = [mole_only] * 3

    # === Step 5: Classification ===
    classification_result = classify_mole(symmetry_score, variation_score)
    print(f"Final Classification: {classification_result}")


    

    mole_only = cv2.cvtColor(mole_only, cv2.COLOR_HSV2BGR)
    # === Step 6: Visualization ===
    image_list = [
        symmetry_images[0],  # Left vs Right
        symmetry_images[1],  # Top vs Bottom
        symmetry_images[2],  # Canny Edges
        mole_only          # Isolated Mole
    ]
    title_list = [
        "Left vs Right",
        "Top vs Bottom",
        "Canny Edges",
        "Isolated Mole"
    ]
    score_list = [
        vert_score,
        horz_score,
        None,
        #brown_percentage / 100.0
    ]
    resize_dims = [
        (150, 150),
        (150, 150),
        (300, 300),
        (300, 300)
    ]

    show_analysis_results(
        image_list=image_list,
        title_list=title_list,
        score_list=score_list,
        classification_result=classification_result,
        resize_dims_list=resize_dims
    )

if __name__ == "__main__":
    main()

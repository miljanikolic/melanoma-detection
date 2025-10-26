import numpy as np
import cv2
import os
from color_analysisM import analyze_color
from symmetry_analysisM import analyze_symmetry
from visualisationM import show_analysis_results
from classificationM import classify_mole
from brown_color_variation import calculate_brown_color_variation, plot_brown_histogram
from automatic_color_picker import auto_isolate_mole

def main():
    # === Step 0: Load image ===
    image_path = r"D:\1) FAKS\Diplomski pocetna verzija\data\test\benign\11.jpg"
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load image.")
        return
    
    mole, mole_mask, skin_mask = auto_isolate_mole(
    img,
    edge_width=30,
    samples_per_edge=40,
    show_steps=True,
    save_hsv_bounds_path="last_auto_skin_bounds.txt"
    )
    cv2.imwrite("output_mole.jpg", mole)
    cv2.imwrite("output_mole_mask.jpg", mole_mask)
    # === Step 1: HSV Color Picking ===

    """pick_and_save_hsv(img)
    if not os.path.exists("hsv_range.txt"):
        print("HSV range file not found. Exiting.")
        return"""

    # === Step 2: Color Analysis ===
    brown_percentage, mole_result, lower_brown, upper_brown = analyze_color(img)
    if mole_result is None or cv2.countNonZero(cv2.cvtColor(mole_result, cv2.COLOR_BGR2GRAY)) == 0:
        print("Color analysis failed or detected area is empty.")
        return
    print(f"Brown Percentage: {brown_percentage:.2f}%")

    # === Step 3: Brown Color Variation Analysis ===
    variation_score, hist, bins, mask = calculate_brown_color_variation(mole_result)
    print(f"Brown Color Variation Score: {variation_score:.2f}")
    print(f"Number of brown pixels: {np.count_nonzero(mask)}")

    if hist is not None:
        plot_brown_histogram(hist, bins)

    # === Step 4: Symmetry Analysis ===
    try:
        symmetry_score, vert_score, horz_score, symmetry_images = analyze_symmetry(mole_result)
    except Exception as e:
        print(f"Symmetry analysis failed: {e}")
        symmetry_score = vert_score = horz_score = 0.0
        symmetry_images = [mole_result] * 3

    # === Step 5: Classification ===
    classification_result = classify_mole(symmetry_score, brown_percentage)
    print(f"Final Classification: {classification_result}")

    # === Step 6: Visualization ===
    image_list = [
        symmetry_images[0],  # Left vs Right
        symmetry_images[1],  # Top vs Bottom
        symmetry_images[2],  # Canny Edges
        mole_result          # Isolated Mole
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
        brown_percentage / 100.0
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
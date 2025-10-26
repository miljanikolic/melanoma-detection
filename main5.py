import cv2
#import numpy as np
from symmetry_analysis import analyze_symmetry
from visualisation import show_analysis_results
from classification import classify_mole
from brown_color_variation5 import calculate_brown_color_variation, plot_brown_histogram
from extraction import extract_mole_by_skin_subtraction        #, auto_get_brown_hsv_bounds  

def main():
    # === Step 0: Load Image ===
    image_path = r"D:\1) FAKS\Diplomski pocetna verzija\data\train\benign\1165.jpg"
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load image.")
        return

    # Automatically Isolate Mole (via skin subtraction) ===
    print("Isolating mole by subtracting skin...")
    mole, mole_mask, skin_mask = extract_mole_by_skin_subtraction(img, k=3, show_steps=True)

    if mole is None or cv2.countNonZero(mole_mask) == 0:
        print("Mole isolation failed or mask is empty.")
        return

    # Color Analysis 
    print("Performing color analysis...")
    brown_score, hist, bins = calculate_brown_color_variation(mole)       
    print(f"Brown color variation score: {brown_score:.4f}")

    # Optionally show histogram
    if hist is not None:
        plot_brown_histogram(hist, bins)

    # Symmetry Analysis 
    print("Performing symmetry analysis...")
    try:
        symmetry_score, vert_score, horz_score, symmetry_images = analyze_symmetry(mole)
    except Exception as e:
        print(f"Symmetry analysis failed: {e}")
        symmetry_score = vert_score = horz_score = 0.0
        symmetry_images = [mole] * 3

    # Classification 
    classification_result = classify_mole(symmetry_score, brown_score)
    print(f"Final Classification: {classification_result}")

    # Visualization 
    image_list = [
    symmetry_images[0],
    symmetry_images[1],
    symmetry_images[2],
    mole
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
    None
]

    resize_dims = [
    (150, 150),
    (150, 150),
    (300, 300),
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

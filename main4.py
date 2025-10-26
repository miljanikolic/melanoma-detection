import cv2
import numpy as np
#from color_analysis2 import analyze_color
from symmetry_analysis2 import analyze_symmetry
from visualisation2 import show_analysis_results
from classification5 import classify_mole
from brown_color_variation5 import calculate_brown_color_variation, plot_brown_histogram
from auto_extractor import auto_isolate_mole


def main():
    # === Step 0: Load Image ===
    image_path = r"D:\1) FAKS\Diplomski pocetna verzija\data\test\benign\8.jpg"
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load image.")
        return

    # === Step 1: Automatically Isolate Mole ===
    print("Isolating mole using K-means clustering...")
    mole, mask, skin, labels, hsv_img, k = auto_isolate_mole(img, k=3, show_steps=True)

    if mole is None or cv2.countNonZero(mask) == 0:
        print("Mole isolation failed or mask is empty.")
        return

    # === Step 2: Color Analysis ===
    print("Performing color analysis...")


    # === Step 4: Symmetry Analysis ===
    print("Performing symmetry analysis...")
    try:
        symmetry_score, vert_score, horz_score, symmetry_images = analyze_symmetry(mole)
        #symmetry_score, vert_score, horz_score, symmetry_images = analyze_symmetry(mole_masked)
    except Exception as e:
        print(f"Symmetry analysis failed: {e}")
        symmetry_score = vert_score = horz_score = 0.0
        symmetry_images = [mole] * 3

    # === Step 5: Classification ===
    classification_result = classify_mole(symmetry_score, mole)
    print(f"Final Classification: {classification_result}")

    # === Step 6: Visualization ===
    image_list = [
        symmetry_images[0],  # Left vs Right
        symmetry_images[1],  # Top vs Bottom
        symmetry_images[2],  # Canny Edges
        mole          # Isolated Mole
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
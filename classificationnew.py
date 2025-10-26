import cv2
import os
import pandas as pd
from color_analysis import analyze_color
from symmetry_analysis import analyze_symmetry
from border_irregularity import calculate_border_irregularity
from classification import classify_mole  # your classify_mole function

def analyze_dataset(base_path):
    results = []
    categories = ["benign", "malignant"]

    for category in categories:
        folder = os.path.join(base_path, category)
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(folder, filename)
                image = cv2.imread(path)

                # Step 1: Color analysis
                brown_percentage, result_img, mask = analyze_color(image)
                if result_img is None:
                    print(f"Skipping {filename}, color analysis failed.")
                    continue

                # Step 2: Symmetry analysis
                avg_symmetry, vert_sym, horiz_sym, _ = analyze_symmetry(result_img)

                # Step 3: Border irregularity
                border_img, irregularity_score, convexity_defect = calculate_border_irregularity(result_img)

                # Step 4: Classification
                prediction = classify_mole(avg_symmetry, irregularity_score, brown_percentage)

                results.append({
                    "filename": filename,
                    "category": category,
                    "symmetry": avg_symmetry,
                    "irregularity": irregularity_score,
                    "brown_percentage": brown_percentage,
                    "prediction": prediction
                })

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv("mole_analysis_results.csv", index=False)
    print("Analysis complete. Results saved to mole_analysis_results.csv.")

# Example usage
analyze_dataset("dataset")

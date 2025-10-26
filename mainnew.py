import cv2
import os
import pandas as pd
from color_analysis import analyze_color
from symmetry_analysisM import analyze_symmetry
from border_irregularity import calculate_border_irregularity
from classification import classify_mole


def analyze_single_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None

    brown_percentage, mole_result = analyze_color(img)
    if mole_result is None:
        print(f"Color analysis failed: {image_path}")
        return None

    symmetry_score, vert_score, horz_score, _ = analyze_symmetry(mole_result)
    border_image, irregularity_score, convexity_defect = calculate_border_irregularity(mole_result)
    classification_result = classify_mole(symmetry_score, irregularity_score, brown_percentage)

    return {
        'filename': os.path.basename(image_path),
        'symmetry_score': symmetry_score,
        'vertical_symmetry': vert_score,
        'horizontal_symmetry': horz_score,
        'irregularity_score': irregularity_score,
        'convexity_defect': convexity_defect,
        'brown_percentage': brown_percentage,
        'prediction': classification_result
    }


def analyze_dataset(base_path):
    results = []
    categories = ['benign', 'malignant']

    for category in categories:
        folder = os.path.join(base_path, category)
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(folder, filename)
                result = analyze_single_image(image_path)
                if result:
                    result['category'] = category
                    results.append(result)

    df = pd.DataFrame(results)
    df.to_csv("classification_results.csv", index=False)
    print("Saved results to classification_results.csv")


if __name__ == "__main__":
    base_path = r"C:\\Users\\PC\\Desktop\\Vezbanje python\\Images\\diplomski_pocetno\\dataset"
    analyze_dataset(base_path)

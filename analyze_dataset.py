import os
import csv
from main import analyze_image

def analyze_folder(folder_path, label):
    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            file_path = os.path.join(folder_path, filename)
            result = analyze_image(file_path)
            if result:
                result["filename"] = filename
                result["label"] = label
                results.append(result)
    return results

def main():
    benign_folder = r"C:\Users\PC\Desktop\Vezbanje python\Images\diplomski_pocetno\dataset\benign"
    #benign_folder = r"C:\Users\PC\Desktop\Vezbanje python\Images\diplomski_pocetno\dataset\benign"
    malignant_folder = r"C:\Users\PC\Desktop\Vezbanje python\Images\diplomski_pocetno\dataset\malignant"

    benign_results = analyze_folder(benign_folder, "benign")
    malignant_results = analyze_folder(malignant_folder, "malignant")
    all_results = benign_results + malignant_results

    csv_filename = "mole_analysis_results.csv"
    with open(csv_filename, mode='w', newline='') as file:
        fieldnames = [
            "filename", "label", "hue_entropy", "gray_entropy", "average_symmetry", "result"
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    print(f"Results saved to {csv_filename}")


if __name__ == "__main__":
    main()

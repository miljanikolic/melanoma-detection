#https://www.geeksforgeeks.org/python/numpy-histogram-method-in-python/
import cv2
from open_file_dialog import open_file_dialog
from mole_isolation import isolate
from symmetry_ssim import symmetry_ssim
from entropy import calculate_hue_entropy                               #, calculate_gray_entropy, calculate_gray_entropy1, calculate_gray_entropy2
from final_classification import classification

def main():
    # Open file dialog koji omogucava odabir slike
    file_path = open_file_dialog()
    original_image = cv2.imread(file_path)
    #print("Image shape:", original_image.shape)           # (height, width, channels)
    #print("Data type:", original_image.dtype)             # usually 'uint8' for 8-bit images
    
    if original_image is None:
        print("Failed to load image.")
        return  # Zaustavlja se dalje odvijanje programa ako nije ucitana slika

    # Izolovanje mladeza
    isolated_rotated_mole, mole_rotated_mask = isolate(original_image)
    
    # Entropije
    hue_entropy = calculate_hue_entropy(isolated_rotated_mole)

    #gray_entropy = calculate_gray_entropy(isolated_rotated_mole)
    #gray_entropy1 = calculate_gray_entropy1(isolated_rotated_mole)
    #gray_entropy2 = calculate_gray_entropy2(isolated_rotated_mole)

    # Simetrija SSIM
    ssim_symmetry_scores = symmetry_ssim(mole_rotated_mask)

    # Konacni rezultat
    result = classification(hue_entropy, ssim_symmetry_scores)
    print(f"Hue Entropy (color): {hue_entropy:.4f} bits")
    print(f"Average Symmetry score: {ssim_symmetry_scores:.4f}")
    print(f"Final classification: {result}")

    #print(f"Gray Entropy: {gray_entropy:.4f} bits")
    #print(f"Gray1 Entropy: {gray_entropy1:.4f} bits")
    #print(f"Gray2 Entropy: {gray_entropy2:.4f} bits")

if __name__ == "__main__":
    main()


    
"""


import cv2
from mole_isolation import isolate
from symmetry_ssim import symmetry_ssim
from entropy import calculate_hue_entropy, calculate_gray_entropy, calculate_gray_entropy1, calculate_gray_entropy2
from final_classification import classification

def analyze_image(file_path):
    original_image = cv2.imread(file_path)
    if original_image is None:
        print(f"Failed to load image: {file_path}")
        return None

    isolated_rotated_mole, mole_rotated_mask = isolate(original_image, show_steps=False)

    hue_entropy = calculate_hue_entropy(isolated_rotated_mole)
    gray_entropy = calculate_gray_entropy2(isolated_rotated_mole)
    ssim_scores = symmetry_ssim(mole_rotated_mask)
    result = classification(hue_entropy, ssim_scores)

    return {
        "hue_entropy": float(hue_entropy),
        "gray_entropy": float(gray_entropy),
        "average_symmetry": float(ssim_scores),
        "result": result
    }

if __name__ == "__main__":
    from open_file_dialog import open_file_dialog
    file_path = open_file_dialog()
    result = analyze_image(file_path)
    if result:
        print(result)


"""

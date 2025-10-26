import cv2
import numpy as np
#from scipy.stats import entropy
import matplotlib.pyplot as plt


def calculate_hue_entropy(rotated_cropped_mole):
    hsv = cv2.cvtColor(rotated_cropped_mole, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]

    hist, _ = np.histogram(hue, bins=30, range=(0, 179), density=False)
    probabilities = hist / np.sum(hist)
    #probabilities = hist

    print(hist)
    print(probabilities)
    #hist, _ = np.histogram(hue.flatten(), bins=30, range=(0, 180), density=True)

    #hist_sum = np.sum(hist)
    #probabilities = hist / hist_sum
    # Plot the histogram

    #plt.figure(figsize=(6, 4))
    #plt.bar(np.arange(30) + 0.5, hist, width=1.0, edgecolor='black')
    #plt.xlabel('Hue Bin Index (0 to 29)')
    #plt.ylabel('Pixel Count')
    #plt.title('Hue Histogram (30 bins)')
    #plt.tight_layout()
    #plt.show()

    hue_entropy = 0
    for p in probabilities:
        if p > 0:
            hue_entropy += p * np.log2(1/p)
            #entropy -= p * np.log2(p)
    return hue_entropy 

    #probabilities = probabilities[probabilities > 0]
        #hue_entropy = entropy(probabilities, base=2)  # Izračunato ovde
"""
def calculate_gray_entropy(rotated_cropped_mole, bins=30):
    gray = cv2.cvtColor(rotated_cropped_mole, cv2.COLOR_BGR2GRAY)
    hist, _ = np.histogram(gray.flatten(), bins=bins, range=(0, 256), density=False)

    hist_sum = np.sum(hist)
    if hist_sum == 0:
        gray_entropy = 0
    else:
        probabilities = hist / hist_sum
        probabilities = probabilities[probabilities > 0]
        gray_entropy = entropy(probabilities, base=2)

    return gray_entropy

import cv2
import numpy as np

def calculate_gray_entropy1(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Izračunavanje histograma (256 vrednosti, od 0 do 255)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    # Normalizuj histogram da dobiješ verovatnoće
    hist_sum = np.sum(hist)
    probabilities = hist / hist_sum

    # Računanje entropije (izbegavaj log(0))
    entropy = 0
    for p in probabilities:
        if p > 0:
            #entropy -= p * np.log2(p)
            entropy += p * np.log2(1/p)

    # Pošto je p niz dužine 1x1, koristi p[0] da izvučeš vrednost
    return float(entropy[0])


def calculate_gray_entropy2(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Histogram (256 podeoka za 256 nivoa sive)
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))

    # Ukupan broj piksela
    total_pixels = np.sum(hist)

    # ver = (br piksela koji imaju vrednost i) / (ukupan broj piksela)
    probabilities = hist / total_pixels

    print(len(probabilities))
    
    #entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy

# Primer:
# img = cv2.imread('putanja_do_slike.jpg')
# entropy_value = calculate_gray_entropy(img)
# print(f'Gray entropy: {entropy_value:.4f}')


"""
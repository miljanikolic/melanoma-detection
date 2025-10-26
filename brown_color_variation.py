import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Calculates hue variation and entropy of brown hue pixels in the mole image
def calculate_brown_color_variation(mole_isolated):           
    hue_channel = mole_isolated[:, :, 0]
    hue_pixels = hue_channel.flatten()

    # Histogram of hue values (normalized)
    hist, bins = np.histogram(hue_pixels, bins=20, range=(0, 180), density=True)
    hist = hist.flatten()

    # Standard deviation (variation score)
    variation_score = np.std(hist)

    # Entropy (measures spread/unpredictability)
    hist_normalized = hist / np.sum(hist)  # convert to probability distribution
    entropy_score = entropy(hist_normalized + 1e-10)  # avoid log(0)

    return variation_score, entropy_score, hist, bins


#Plots the histogram of brown hue values.


def plot_brown_histogram(hist, bins, entropy_score=None):
    bin_width = bins[1] - bins[0]
    plt.bar(bins[:-1], hist, width=bin_width, align='edge', color='brown', edgecolor='black')
    plt.xlabel('Hue Value')
    plt.ylabel('Normalized Frequency')

    title = 'Histogram of Brown Hue Values'
    if entropy_score is not None:
        title += f' (Entropy: {entropy_score:.4f})'

    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


"""
import numpy as np
import matplotlib.pyplot as plt

#Calculates hue variation (standard deviation of histogram) of brown pixels in the image
def calculate_brown_color_variation(mole_isolated):           

    hue_channel = mole_isolated[:, :, 0]
    hue_pixels = hue_channel.flatten()

    # Histogram of hue values
    hist, bins = np.histogram(hue_pixels, bins=20, range=(0, 180), density=True)
    hist = hist.flatten()
    # Debug print
    #for i in range(len(hist)):
    #    print(f"Hue range {int(bins[i])}-{int(bins[i+1])}: {hist[i]:.4f}")

    # Calculate variation as standard deviation
    variation_score = np.std(hist)

    return variation_score, hist, bins              #, mask

#Plots the histogram of brown hue values.
def plot_brown_histogram(hist, bins):

    bin_width = bins[1] - bins[0]
    plt.bar(bins[:-1], hist, width=bin_width, align='edge', color='brown', edgecolor='black')
    plt.xlabel('Hue Value')
    plt.ylabel('Normalized Frequency')
    plt.title('Histogram of Brown Hue Values')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
"""















import matplotlib.pyplot as plt

def plot_symmetry_histograms(left_hist, right_hist, top_hist, bottom_hist):
    plt.figure(figsize=(12, 5))

    # Vertical Symmetry Histogram (Left/Right)
    plt.subplot(1, 2, 1)
    plt.plot(left_hist, label="Left Half", color="blue")
    plt.plot(right_hist, label="Right Half (Flipped)", color="red")
    plt.title("Vertical Symmetry (Left-Right)")
    plt.xlabel("Hue Bin")
    plt.ylabel("Normalized Frequency")
    plt.legend()

    # Horizontal Symmetry Histogram (Top/Bottom)
    plt.subplot(1, 2, 2)
    plt.plot(top_hist, label="Top Half", color="green")
    plt.plot(bottom_hist, label="Bottom Half (Flipped)", color="orange")
    plt.title("Horizontal Symmetry (Top-Bottom)")
    plt.xlabel("Hue Bin")
    plt.ylabel("Normalized Frequency")
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_color_symmetry_histograms(left_hist, right_hist, top_hist, bottom_hist):
    plt.figure(figsize=(12, 5))

    # Vertical Symmetry Histogram (Left/Right)
    plt.subplot(1, 2, 1)
    plt.plot(left_hist, label="Left Half", color="blue")
    plt.plot(right_hist, label="Right Half (Flipped)", color="red")
    plt.title("Vertical Symmetry Color (Left-Right)")
    plt.xlabel("Hue Bin")
    plt.ylabel("Normalized Frequency")
    plt.legend()

    # Horizontal Symmetry Histogram (Top/Bottom)
    plt.subplot(1, 2, 2)
    plt.plot(top_hist, label="Top Half", color="green")
    plt.plot(bottom_hist, label="Bottom Half (Flipped)", color="orange")
    plt.title("Horizontal Color Symmetry (Top-Bottom)")
    plt.xlabel("Hue Bin")
    plt.ylabel("Normalized Frequency")
    plt.legend()

    plt.tight_layout()
    plt.show()
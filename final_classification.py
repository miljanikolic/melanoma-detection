def classification(hue_entropy, ssim_symmetry_scores):
    if ssim_symmetry_scores > 0.7:
        return "Benign"
    elif ssim_symmetry_scores < 0.5:
        return "Malignant"
    else:
        if hue_entropy > 1.2:
            return "Malignant"
        else:
            return "Benign"
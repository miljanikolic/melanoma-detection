def classify_mole(symmetry_score, variation_score):
    """
    Classifies the mole as benign or malignant based on analysis scores.
    Returns a string: "Benign (High Confidence)", "Possibly Benign",
    "Malignant (High Confidence)" or "Possibly Malignant"
    """

    if symmetry_score > 0.3 and variation_score < 0.05:
        return "Benign (High Confidence)"
    elif symmetry_score >= 0.2 and variation_score < 0.5:
        return "Possibly Benign. More details needed"
    elif symmetry_score < 0.2 and variation_score > 0.5:
        return "Malignant (High Confidence)"
    else:
        return "Possibly Malignant. More details needed"
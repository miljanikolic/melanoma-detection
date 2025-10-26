
def classify_mole(symmetry_score, brown_percentage):
    """
    Classifies the mole as benign or malignant based on analysis scores.
    Returns a string: "Benign (High Confidence)", "Possibly Benign",
    "Malignant (High Confidence)" or "Possibly Malignant"
    """

    #if symmetry_score is None or irregularity_score is None:
    #    return "Unknown"

    #if symmetry_score >= 0.4 and irregularity_score <= 0.4 and brown_percentage > 10:
    #    return "Benign"
    #else:
        #return "Malignant"
    #    return "More details needed"


    if symmetry_score > 0.4 and  brown_percentage < 10:
        return "Benign (High Confidence)"
    elif symmetry_score >= 0.3 :
        return "Possibly Benign"
    elif symmetry_score < 0.25 :
        return "Malignant (High Confidence)"
    else:
        return "Possibly Malignant"

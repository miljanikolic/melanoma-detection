"""
def classify_mole(symmetry_score, variation_score):
    
    Classifies a mole as benign or malignant based on symmetry and color variation.

    Parameters:
        symmetry_score (float): The symmetry score (0 to 1, higher is more symmetrical)
        variation_score (float): The color variation score (standard deviation of hue)

    Returns:
        str: "benign" or "malignant"
    

    if symmetry_score >= 0.37:
        return "benign"
    elif symmetry_score < 0.30:
        return "malignant"
    else:
        if variation_score >= 0.017:
            return "benign"
        else:
            return "malignant" 
"""
def classify_mole(shape_symmetry, entropy):
    if entropy > 1.2:
        return "malignant"
    elif entropy < 0.9:
        return "benign"
    else:
        if shape_symmetry > 0.85:
            return "malignant"
        else:
            return "benign"


"""
def classify_mole(symmetry_score, variation_score, entropy_score):
    if symmetry_score >= 0.37:
        return "benign"
    elif symmetry_score < 0.30:
        return "malignant"
    else:
        if variation_score < 0.019 and entropy_score < 1.4:
            return "benign"
        else:
            return "malignant"
"""
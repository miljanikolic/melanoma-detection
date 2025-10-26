import pandas as pd
import os

def save_to_excel(filename, image_name, symmetry_score, irregularity_score, brown_percentage, classification):
    # Create the dataframe row
    new_data = {
        "Image Name": image_name,
        "Symmetry Score": symmetry_score,
        "Irregularity Score": irregularity_score,
        "Brown Percentage": brown_percentage,
        "Classification": classification
    }

    # Load existing Excel file or create a new one
    if os.path.exists(filename):
        df = pd.read_excel(filename)
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    else:
        df = pd.DataFrame([new_data])

    # Save back to Excel
    df.to_excel(filename, index=False)

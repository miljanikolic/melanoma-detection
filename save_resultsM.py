import pandas as pd

def save_to_excel(filename, image_name, symmetry_score, irregularity_score, brown_percentage, classification):
    from openpyxl import load_workbook
    from os.path import exists

    data = {
        "Image Name": [image_name],
        "Symmetry Score": [symmetry_score],
        "Irregularity Score": [irregularity_score],
        "Brown Percentage": [brown_percentage],
        "Classification": [classification]
    }

    new_df = pd.DataFrame(data)

    try:
        if exists(filename):
            existing_df = pd.read_excel(filename)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        combined_df.to_excel(filename, index=False)

    except PermissionError:
        print(f"Cannot write to '{filename}' — is it open in Excel? Please close it and try again.")

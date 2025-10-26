#RESENJE !
import tkinter as tk
from tkinter import filedialog

def open_image_file():
    root = tk.Tk()
    root.withdraw()  
    file_path = filedialog.askopenfilename(
        title="Select Image File",
        filetypes=[("Images Files", "*.png *.jpg *.jpeg *.bmp")]
    )
    return file_path





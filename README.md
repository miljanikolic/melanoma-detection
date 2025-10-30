# melanoma-detection
By analyzing the symmetry and color variation of the mole from the uploaded image, Python-based program concludes whether it is melanoma or whether the mole is benign.
Symmetry is analyzed using SSIM, while color variation is analyzed using entropy.

How to run the program
1. *Clone the repository*:
   ```bash
   git clone https://github.com/miljanikolic/melanoma-detection.git
   cd melanoma-detection

2. *(Optional) *Create and activate a virtual environment*:
    python -m venv venv
    venv\Scripts\activate     # On Windows
    # or
    source venv/bin/activate  # On macOS/Linux

3. *Install required libraries*:
    pip install -r requirements.txt

4. *Run the program*:
python main.py

5. *Select an image to analyze*:
When you start the program, a file-open dialog appears. Choose the mole image you want to analyze.

6. *View results*:
The program automatically performs color, symmetry, and border analysis. Processed images and results are displayed on screen.

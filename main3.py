import cv2
from color_analysis3 import analyze_color  # Import the function

if __name__ == '__main__':
    # Load the image you want to analyze
    img_path = r"D:\1) FAKS\Diplomski pocetna verzija\data\test\benign\8.jpg"
    img = cv2.imread(img_path)

    if img is None:
        print(f"Could not load image from {img_path}")
    else:
        brown_percentage, result_image = analyze_color(img)

        if brown_percentage is not None:
            print(f"Mole Color Coverage: {brown_percentage:.2f}%")
            cv2.imshow("Mole Mask", result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
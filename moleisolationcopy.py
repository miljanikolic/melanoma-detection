#OVO JE KOD KOJI SAM KORISTILA I ZA ISPITIVANJE SIMEGTRIJE POMOCU HISTOGRAMA
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def isolate(loaded_image):
    cv2.imshow("Original image", loaded_image)
    gray =cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7,7), 0)
    (T, threshInv) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cv2.imshow("Threshold", threshInv)
    print("Otsu's thresholding value: {}".format(T))
    cv2.imshow("Gray", gray)
    plot_otsu_histogram(gray, T)


    
    kernel = np.ones((7, 7), np.uint8)
    #####mole_mask = cv2.morphologyEx(threshInv, cv2.MORPH_OPEN, kernel)
    mole_mask = cv2.morphologyEx(threshInv, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mole_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)     #contours je lista svih kontura,
    
    if not contours:
        print("No contours found!")
        return loaded_image, mole_mask
    
    max_contour = max(contours, key=cv2.contourArea) 

        #OVO MI MOZDA BUDE KORISTILO
    #M = cv2.moments(max_contour)
    #print(M)
    #cx = int(M['m10']/M['m00'])
    #cy = int(M['m01']/M['m00'])
    #print(f'Coordinates: cx={cx:.2f}, cy={cy:.2f}')

    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    (center_x, center_y), (width, height), angle = rect
    print(f"Bounding Box Size: width={width:.2f}, height={height:.2f}")
    print(f"Bounding Box Center: x={center_x:.2f}, y={center_y:.2f}")
    print(f"Rotation Angle: {angle:.2f} degrees")
    #max_len = max(width, height)
    #("Largest Side Length: {max_len:.2f} pixels")

    #OVO MI MOZDA BUDE KORISTILO
    #area = cv2.contourArea(max_contour)
    
    contour_image = np.zeros_like(mole_mask)
    cv2.drawContours(contour_image, [max_contour], 0, color=255, thickness=1)
    cv2.imshow("Max Contour", contour_image)
    
    cleaned_mask = np.zeros_like(mole_mask) #maska koja se sastoji sve od nula, "crni prazni kanvas", na kojoj crtamo najvecu konturu iz liste kontura
    cv2.drawContours(cleaned_mask, [max_contour], 0, color=255, thickness=-1)  #kada je thickness=-1 onda se popunjava citava regija unutar konture, a ne crta se samo kontura
                                                                                # kontura se crta kada je thickness pozitivna vrednost
                                                                                #popunjava se belom bojom color=255
                                                                                #-1 oznacava da se crtaju sve konture (ja svakako ovde u kodu izdvajam jednu maksimalnu)
                                                                                #koristim oznaku 0 da crtam prvu konturu
    cv2.imshow("Cleaned mask", cleaned_mask)

    # Apply the mask to isolate mole
    masked = cv2.bitwise_and(loaded_image, loaded_image, mask=cleaned_mask)
    cv2.imshow("Output", masked)

    print(type(masked))
    print(type(cleaned_mask))
    #ROTIRANA MASKA
    pil_mask = Image.fromarray(cleaned_mask)
    rotated_mole_mask = pil_mask.rotate(angle, expand=False)
    rotated_mole_mask = np.array(rotated_mole_mask)
    cv2.imshow("Rotated mask", rotated_mole_mask)

    #ROTIRANI IZDVOJENI MLADEZ
    pil_masked = Image.fromarray(masked)
    rotated_mole_masked = pil_masked.rotate(angle, expand=False)
    rotated_mole_masked = np.array(rotated_mole_masked)
    cv2.imshow("Output rotated", rotated_mole_masked)

    #black_matrix = np.zeros((int(max_len), int(max_len)), dtype=np.uint8)

    cv2.drawContours(masked, [box], 0, color = (0, 0, 255), thickness = 1)
    cv2.circle(masked, (int(center_x), int(center_y)), 4, color = (0, 0, 255), thickness = -1)
    cv2.imshow("Isolated Mole with, red center of rectangle and rectangle", masked)
    center = (int(center_x), int(center_y))  #Iz funkcije minAreaRect
    #width = int(max_len)
    #height = int(max_len)
    width = int(width)
    height = int(height)

    cropped_mole = crop_centered(rotated_mole_masked, center, width, height)
    cv2.imshow("Cropped Mole", cropped_mole)

    cropped_mask = crop_centered(rotated_mole_mask, center, width, height)
    cv2.imshow("Cropped Mole Maks", cropped_mask)
    cv2.waitKey(0)
    
    return cropped_mole, cropped_mask
#Syntax: cv2.threshold(source, thresholdValue, maxVal, thresholdingTechnique)

def plot_otsu_histogram(gray_image, threshold_value):
    plt.figure(figsize=(8, 5))
    plt.hist(gray_image.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7, label='Pixel Intensity')
    plt.axvline(x=threshold_value, color='red', linestyle='--', label=f'Otsu Threshold = {threshold_value:.2f}')
    plt.title('Grayscale Histogram with Otsu Threshold')
    plt.xlabel('Pixel Intensity (0–255)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def crop_centered(matrix, center, width, height):
    cx, cy = center
    #start_x = int(cx - width // 2)
    #start_y = int(cy - height // 2)

    start_x = cx - width // 2
    start_y = cy - height // 2
    
    # Ensure the crop stays inside the image
    start_x = max(0, start_x)
    start_y = max(0, start_y)
    end_x = min(matrix.shape[1], start_x + width)
    end_y = min(matrix.shape[0], start_y + height)

    cropped = matrix[start_y:end_y, start_x:end_x]
    return cropped



#https://pyimagesearch.com/2021/04/28/opencv-thresholding-cv2-threshold/
#https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html




"""
#OVO JE ZA UPISIVANJE U TABELU
import cv2
import numpy as np
from PIL import Image

def isolate(loaded_image, show_steps=False):
    if show_steps:
        cv2.imshow("Original image", loaded_image)
    gray = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    (T, threshInv) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if show_steps:
        cv2.imshow("Threshold", threshInv)
    print("Otsu's thresholding value: {}".format(T))
    
    
    kernel = np.ones((7, 7), np.uint8)
    mole_mask = cv2.morphologyEx(threshInv, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mole_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found!")
        return loaded_image, mole_mask

    max_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(max_contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    (center_x, center_y), (width, height), angle = rect
    max_len = max(width, height)

    area = cv2.contourArea(max_contour)

    contour_image = np.zeros_like(mole_mask)
    cv2.drawContours(contour_image, [max_contour], 0, color=255, thickness=2)
    if show_steps:
        cv2.imshow("Max Contour", contour_image)

    cleaned_mask = np.zeros_like(mole_mask)
    cv2.drawContours(cleaned_mask, [max_contour], 0, color=255, thickness=-1)
    if show_steps:
        cv2.imshow("Cleaned mask", cleaned_mask)

    masked = cv2.bitwise_and(loaded_image, loaded_image, mask=cleaned_mask)
    if show_steps:
        cv2.imshow("Output", masked)

    pil_mask = Image.fromarray(cleaned_mask)
    rotated_mole_mask = pil_mask.rotate(angle, expand=False)
    rotated_mole_mask = np.array(rotated_mole_mask)
    if show_steps:
        cv2.imshow("Rotated mask", rotated_mole_mask)

    pil_masked = Image.fromarray(masked)
    rotated_mole_masked = pil_masked.rotate(angle, expand=False)
    rotated_mole_masked = np.array(rotated_mole_masked)
    if show_steps:
        cv2.imshow("Output rotated", rotated_mole_masked)

    if show_steps:
        black_matrix = np.zeros((int(max_len), int(max_len)), dtype=np.uint8)
        cv2.imshow("Black Matrix", black_matrix)
        cv2.drawContours(masked, [box], 0, (0, 0, 255), 2)
        cv2.circle(masked, (cx, cy), 4, (255, 0, 0), -1)
        cv2.circle(masked, (int(center_x), int(center_y)), 4, (0, 0, 255), -1)
        cv2.imshow("Isolated Mole with Blue Centroid, red center of rectangle and rectangle", masked)

    center = (int(center_x), int(center_y))
    width = int(width)
    height = int(height)

    cropped_mole = crop_centered(rotated_mole_masked, center, width, height)
    cropped_mask = crop_centered(rotated_mole_mask, center, width, height)

    if show_steps:
        cv2.imshow("Cropped Mole", cropped_mole)
        cv2.imshow("Cropped Mole Mask", cropped_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return cropped_mole, cropped_mask

def crop_centered(matrix, center, width, height):
    cx, cy = center
    start_x = int(cx - width // 2)
    start_y = int(cy - height // 2)
    
    start_x = max(0, start_x)
    start_y = max(0, start_y)
    end_x = min(matrix.shape[1], start_x + width)
    end_y = min(matrix.shape[0], start_y + height)
    
    return matrix[start_y:end_y, start_x:end_x]


"""



"""
import cv2
import numpy as np
from PIL import Image

def isolate(loaded_image):
    cv2.imshow("Original image", loaded_image)
    gray =cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7,7), 0)
    (T, threshInv) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cv2.imshow("Threshold", threshInv)
    print("Otsu's thresholding value: {}".format(T))
    
    
    kernel = np.ones((7, 7), np.uint8)
    #####mole_mask = cv2.morphologyEx(threshInv, cv2.MORPH_OPEN, kernel)
    mole_mask = cv2.morphologyEx(threshInv, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mole_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)     #contours je lista svih kontura,
    
    if not contours:
        print("No contours found!")
        return loaded_image, mole_mask
    


        #OVO MI MOZDA BUDE KORISTILO
    #M = cv2.moments(max_contour)
    #print(M)
    #cx = int(M['m10']/M['m00'])
    #cy = int(M['m01']/M['m00'])
    #print(f'Coordinates: cx={cx:.2f}, cy={cy:.2f}')
    max_contour = max(contours, key=cv2.contourArea) 
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    (center_x, center_y), (width, height), angle = rect
    print(f"Bounding Box Size: width={width:.2f}, height={height:.2f}")
    print(f"Bounding Box Center: x={center_x:.2f}, y={center_y:.2f}")
    print(f"Rotation Angle: {angle:.2f} degrees")
    #max_len = max(width, height)
    #("Largest Side Length: {max_len:.2f} pixels")

    contour_image = np.zeros_like(mole_mask)
    cv2.drawContours(contour_image, [max_contour], 0, color=255, thickness=2)
    cv2.imshow("Max Contour", contour_image)

    cleaned_mask = np.zeros_like(mole_mask)
    cv2.drawContours(cleaned_mask, [max_contour], 0, color=255, thickness=-1)
    cv2.imshow("Cleaned mask", cleaned_mask)

    masked = cv2.bitwise_and(loaded_image, loaded_image, mask=cleaned_mask)
    cv2.imshow("Masked Mole", masked)

    cv2.drawContours(masked, [box], 0, (0, 0, 255), 2)
    cv2.circle(masked, (int(center_x), int(center_y)), 4, (0, 0, 255), -1)
    cv2.imshow("Mole with Bounding Box", masked)

    # Crop precisely using perspective transform
    cropped_mole = crop_minAreaRect(loaded_image, rect)
    cropped_mask = crop_minAreaRect(cleaned_mask, rect)

    cv2.imshow("Cropped Mole", cropped_mole)
    cv2.imshow("Cropped Mask", cropped_mask)

    cv2.waitKey(0)
    return cropped_mole, cropped_mask
#Syntax: cv2.threshold(source, thresholdValue, maxVal, thresholdingTechnique)

def crop_minAreaRect(img, rect):
    """"""
    Seče tačno pravougaonik oko konture na osnovu minAreaRect.
    :param img: Ulazna slika
    :param rect: Rezultat cv2.minAreaRect (centar, dimenzije, ugao)
    :return: Izdvojeni pravougaonik kao slika
    """"""

    #box = cv2.boxPoints(rect)
    #box = np.array(box, dtype="float32")
    box = cv2.boxPoints(rect)  # 4 tačke pravougaonika
    box = np.int8(box)         # zaokruži koordinate na int

    width = int(rect[1][0])
    height = int(rect[1][1])

    # Definiši destinacione tačke: (0,0), (width,0), (width,height), (0,height)
    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    # Dobij matricu transformacije
    M = cv2.getPerspectiveTransform(np.array(box, dtype="float32"), dst_pts)

    # Primeni perspektivnu transformaciju
    warped = cv2.warpPerspective(img, M, (width, height))

    return warped


#https://pyimagesearch.com/2021/04/28/opencv-thresholding-cv2-threshold/
#https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
#https://stackoverflow.com/questions/37177811/crop-rectangle-returned-by-minarearect-opencv-python
#link iznad je za secenje slike





"""
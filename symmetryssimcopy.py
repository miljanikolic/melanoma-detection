from skimage.metrics import structural_similarity as ssim
import cv2


def symmetry_ssim(rotated_cropped_mask):
    #gray_mole = cv2.cvtColor(rotated_cropped_mask, cv2.COLOR_BGR2GRAY)

    left_half, right_half_flipped, top_half, bottom_half_flipped = halves(rotated_cropped_mask)

    vertical_symmetry = ssim(left_half, right_half_flipped)
    horizontal_symmetry = ssim(top_half, bottom_half_flipped)

    #vertical_symmetry, _ = ssim(left_half, right_half_flipped, full=True)
    #horizontal_symmetry, _ = ssim(top_half, bottom_half_flipped, full=True)

    vertical_symmetry = max(0, min(vertical_symmetry, 1))
    horizontal_symmetry = max(0, min(horizontal_symmetry, 1))
    avg_symmetry = (vertical_symmetry + horizontal_symmetry) / 2
    print("Symmetry (SSIM) Scores:")
    print(f"Vertical symmetry: {vertical_symmetry:.4f}")
    print(f"Horizontal symmetry: {horizontal_symmetry:.4f}")
    print(f"Average symmetry: {avg_symmetry:.4f}")
    #scores = (vertical_symmetry, horizontal_symmetry, avg_symmetry)

    return avg_symmetry #scores               #vertical_symmetry, horizontal_symmetry, avg_symmetry


def halves(rotated_cropped_mask):
    height, width = rotated_cropped_mask.shape
    left_half = rotated_cropped_mask[:, :width // 2]
    right_half = rotated_cropped_mask[:, width // 2:]
    right_half_flipped = cv2.flip(right_half, 1)

    # Izjednaci sirine
    min_width = min(left_half.shape[1], right_half_flipped.shape[1])
    left_half = left_half[:, :min_width]
    right_half_flipped = right_half_flipped[:, :min_width]

    top_half = rotated_cropped_mask[:height // 2, :]
    bottom_half = rotated_cropped_mask[height // 2:, :]
    bottom_half_flipped = cv2.flip(bottom_half, 0)

    # Izjednaci visine
    min_height = min(top_half.shape[0], bottom_half_flipped.shape[0])
    top_half = top_half[:min_height, :]
    bottom_half_flipped = bottom_half_flipped[:min_height, :]
    cv2.imshow("Left half", left_half)
    cv2.imshow("Right half flipped",right_half_flipped)
    cv2.imshow("Top half", top_half)
    cv2.imshow("Bottom half flipped", bottom_half_flipped)
    cv2.waitKey(0)

    return left_half, right_half_flipped, top_half, bottom_half_flipped


"""
#prosledjujem masku za ssim analizu
def halves(rotated_cropped_mask):
    #DELIM KROPOVANU MASKU UZDUZNO NA DVA JEDNAKA DELA
    height, width = rotated_cropped_mask.shape
    left_half = rotated_cropped_mask[:, :width // 2]
    right_half = rotated_cropped_mask[:, width // 2:]
    right_half_flipped = cv2.flip(right_half, 1)

    #DELIM KROPOVANU MASKU HORIZONTALNO NA DVA JEDNAKA DELA
    top_half = rotated_cropped_mask[:height // 2, :]
    bottom_half = rotated_cropped_mask[height // 2:, :]
    bottom_half_flipped = cv2.flip(bottom_half, 0)
    
    
    cv2.imshow("Left half", left_half)
    cv2.moveWindow("Left half", 100, 100)  # Move to (100,100)
    cv2.imshow("Right half flipped", right_half_flipped)
    cv2.moveWindow("Right half flipped", 450, 100) 
    cv2.imshow("Top half", top_half)
    cv2.moveWindow("Top half", 100, 500)  # Move to (100,100)
    cv2.imshow("Bottom half flipped", bottom_half_flipped)
    cv2.moveWindow("Bottom half flipped", 450, 500) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return left_half, right_half_flipped, top_half, bottom_half_flipped
"""
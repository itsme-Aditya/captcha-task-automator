import numpy as np
import os, cv2, time
from skimage.metrics import structural_similarity as ssim

def extract_captcha_elements(full_img, master_w, master_h):
    header_crop = full_img[0:142, 0:master_w]
    grid_crop = full_img[147:629, 0:master_w]
    return header_crop, grid_crop

def mask_checkmark(img):
    """
    Masks top-left region where the select-checkmark appears.
    """
    h, w = img.shape[:2]
    masked = img.copy()
    # Mask top-left 25% area
    masked[0:int(h*0.25), 0:int(w*0.25)] = 0
    return masked

def preprocess_for_match(img):
    """
    Light Gaussian blur to reduce pixel-perfect rendering noise.
    """
    return cv2.GaussianBlur(img, (3, 3), 0)

def normalize_cell(img, target_size=(160, 160), border=15):
    """
    Shrinks image and adds white border to normalize layout.
    """
    inner_size = (target_size[0] - 2*border, target_size[1] - 2*border)
    resized = cv2.resize(img, inner_size, interpolation=cv2.INTER_AREA)
    bordered = cv2.copyMakeBorder(resized, border, border, border, border, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return bordered

def arrays_match(cells_old, cells_new):
    if len(cells_old) != len(cells_new):
        return False
    time.sleep(2)
    print("\n\n\n\narrays_match called") 
    # Save all cells for debugging
    debug_dir = "logs/matching_logs"
    os.makedirs(debug_dir, exist_ok=True)

    if ssim is None:
        print("Warning: scikit-image not found. Falling back to exact comparison.")
        return all(np.array_equal(o, n) for o, n in zip(cells_old, cells_new))

    i = 0
    for old_img, new_img in zip(cells_old, cells_new):
        if old_img.shape != new_img.shape:
            return False

        i += 1
        old_m = mask_checkmark(old_img)
        new_m = mask_checkmark(new_img)
        old_p = preprocess_for_match(old_m)
        new_p = preprocess_for_match(new_m)
        old_gray = cv2.cvtColor(old_p, cv2.COLOR_BGR2GRAY)
        new_gray = cv2.cvtColor(new_p, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(old_gray, new_gray, full=True)
        print(f"  Cell {i} SSIM: {score:.3f}")
        if score < 0.5:
            old_m2 = mask_checkmark(normalize_cell(old_img))
            old_p2 = preprocess_for_match(old_m2)
            old_gray2 = cv2.cvtColor(old_p2, cv2.COLOR_BGR2GRAY)
            new_gray2 = cv2.resize(new_gray, (old_gray2.shape[1], old_gray2.shape[0]))
            score2, _ = ssim(old_gray2, new_gray2, full=True)
            print(f"  Cell {i} SSIM (normalized): {score2:.3f}")
            cv2.imwrite(os.path.join(debug_dir, f"cell{i}_old_normalized.png"), old_gray2)
            cv2.imwrite(os.path.join(debug_dir, f"cell{i}_new_for_normalized_cmp.png"), new_gray2)
            score = max(score, score2)  # take the better score

        if score < 0.60:
            return False

    return True

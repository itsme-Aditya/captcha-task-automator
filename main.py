import os
import mss
import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
import pyautogui
import time
import re
import sys
import tkinter as tk
from threading import Thread
from skimage.metrics import structural_similarity as ssim

# LOGS_DIR = "logs"
# os.makedirs(LOGS_DIR, exist_ok=True)

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# MODEL_PATH = r".\assets\final_model.pt"

def get_base_path():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(__file__)

base_path = get_base_path()

try:
    import nltk
    nltk.data.path.append(os.path.join(base_path, "assets", "nltk_data"))
    from nltk.tokenize import word_tokenize
except Exception:
    def word_tokenize(text):
        return re.findall(r'\b\w+\b', text)
    
MODEL_PATH = os.path.join(base_path, "assets", "final_model.pt")

tesseract_path = os.path.join(base_path, "assets", "tesseract", "tesseract.exe")
pytesseract.pytesseract.tesseract_cmd = tesseract_path

try:
    print(f"Loading YOLO model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Warning: Failed to load YOLO model: {e}")
    model = None

MAX_ITERATIONS = 5
CONF_THRESHOLD = 0.30

# Fixed button centers (local to master TL) from map files
REFRESH_NORMAL = (28, 679)
VERIFY_NORMAL  = (418, 679)
REFRESH_ERROR  = (28, 718)
VERIFY_ERROR   = (418, 718)

running = False
# ==========================================
# Screen Capture & Segmentation
# ==========================================
def find_master_box():
    """Finds captcha area: header TL for origin, header width for W,
       Verify button bottom for H. Width is always the header width."""
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        scan_region = {
            "left": monitor["left"],
            "top": monitor["top"],
            "width": monitor["width"] // 2,
            "height": monitor["height"] - 80
        }
        raw_img = sct.grab(scan_region)
        bgr_img = cv2.cvtColor(np.array(raw_img), cv2.COLOR_BGRA2BGR)

        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([95, 100, 100]), np.array([125, 255, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        header_contours = [c for c in contours if cv2.contourArea(c) > 15000]
        button_contours = [c for c in contours if 200 < cv2.contourArea(c) < 5000]

        if not header_contours:
            return None, None, None, None

        # Header = largest blue contour → gives us TL and authoritative W
        header = max(header_contours, key=cv2.contourArea)
        hx, hy, hw, hh = cv2.boundingRect(header)
        tl_x = hx
        tl_y = hy
        calc_w = hw  # header width IS the captcha width (always 482)

        # Find Verify button: right-aligned below header
        header_right = hx + hw
        valid_buttons = []
        for c in button_contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            br_x = bx + bw
            if (header_right - 40) < br_x < (header_right + 10):
                if by > (hy + hh):
                    valid_buttons.append(c)

        if valid_buttons:
            verify = max(valid_buttons, key=lambda c: cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3])
            vx, vy, vw, vh = cv2.boundingRect(verify)
            calc_h = (vy + vh) - tl_y
        else:
            calc_h = 706

        # Captcha is always at least 706px tall (744 with error). If we got less,
        # a spurious blue element inside the grid was misdetected as the button.
        if calc_h < 706:
            calc_h = 706

        global_x = scan_region["left"] + tl_x
        global_y = scan_region["top"] + tl_y
        print(f"  [find_master_box] TL=({global_x},{global_y}) W={calc_w} H={calc_h}")
        return global_x, global_y, calc_w, calc_h

def capture_master_region(master_x, master_y, master_w, master_h):
    region = {"left": master_x, "top": master_y, "width": master_w, "height": master_h}
    with mss.mss() as sct:
        raw_img = sct.grab(region)
        return cv2.cvtColor(np.array(raw_img), cv2.COLOR_BGRA2BGR)

def extract_captcha_elements(full_img, master_w, master_h):
    header_crop = full_img[0:142, 0:master_w]
    grid_crop = full_img[147:629, 0:master_w]
    return header_crop, grid_crop

def is_error_state(master_h):
    return master_h >= 720

def get_buttons(master_h):
    if is_error_state(master_h):
        return REFRESH_ERROR, VERIFY_ERROR
    return REFRESH_NORMAL, VERIFY_NORMAL

# ==========================================
# OCR
# ==========================================
def analyze_instruction(header_crop):
    scaled = cv2.resize(header_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    text = pytesseract.image_to_string(inv, config='--psm 6').lower()
    tokens = word_tokenize(text)

    if "square" in tokens or "squares" in tokens:
        return ("REFRESH", None)

    classes = {
        "car", "bus", "motorcycle", "bicycle", "crosswalk", "boat",
        "chimney", "bridge", "hydrant", "stair", "stairs", "palm"
    }
    if "traffic" in tokens:
        return ("SOLVE", "traffic light")
    if "fire" in tokens and "hydrant" in tokens:
        return ("SOLVE", "hydrant")

    for token in tokens:
        singular = token[:-1] if token.endswith('s') else token
        if token in classes:
            return ("SOLVE", token)
        if singular in classes:
            return ("SOLVE", singular)

    return ("UNKNOWN", text)

def read_error_message(full_img):
    """Crop error text row (between grid bottom and buttons) and OCR it."""
    h = full_img.shape[0]
    if h < 640:
        return None
    footer = full_img[629:min(670, h), :]
    if footer.size == 0:
        return None
    scaled = cv2.resize(footer, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config='--psm 7').lower()
    tokens = word_tokenize(text)
    print(f"  [Error OCR] text='{text.strip()}' tokens={tokens}")
    if "select" in tokens or "matching" in tokens or "new" in tokens:
        return "NEW_IMAGES"
    if "try" in tokens or "again" in tokens:
        return "TRY_AGAIN"
    return None

# ==========================================
# Helpers
# ==========================================
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
    # debug_dir = "debug_cells"
    # os.makedirs(debug_dir, exist_ok=True)
    # for idx, (old_img, new_img) in enumerate(zip(cells_old, cells_new), start=1):
    #     cv2.imwrite(os.path.join(debug_dir, f"cell{idx}_old.png"), old_img)
    #     cv2.imwrite(os.path.join(debug_dir, f"cell{idx}_new.png"), new_img)

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
            # cv2.imwrite(os.path.join(debug_dir, f"cell{i}_old_normalized.png"), old_gray2)
            # cv2.imwrite(os.path.join(debug_dir, f"cell{i}_new_for_normalized_cmp.png"), new_gray2)
            score = max(score, score2)  # take the better score

        if score < 0.60:
            return False

    return True

def slow_click(gx, gy):
    pyautogui.moveTo(gx, gy + 15, duration=0.8)
    time.sleep(0.1)
    pyautogui.click()

def get_positive_cells(grid, target):
    """Split grid into 9 cells, run classifier on each.
       Cell centers are calculated dynamically from grid dimensions."""
    h, w = grid.shape[:2]
    cell_h = h // 3
    cell_w = w // 3
    cells = []
    positive_cells = []
    cell_centers = {}

    for row in range(3):
        for col in range(3):
            y1 = row * cell_h
            y2 = y1 + cell_h if row < 2 else h
            x1 = col * cell_w
            x2 = x1 + cell_w if col < 2 else w

            cell_img = grid[y1:y2, x1:x2]
            cells.append(cell_img)
            cell_id = f"cell_{row}_{col}"

            # Dynamic center: grid offset (147) + cell center within grid
            cx_local = x1 + (x2 - x1) // 2
            cy_local = 147 + y1 + (y2 - y1) // 2
            cell_centers[cell_id] = (cx_local, cy_local)

            if model and target:
                results = model(cell_img, verbose=False)
                for r in results:
                    if r.probs is None:
                        continue
                    top5 = r.probs.top5
                    top5_conf = r.probs.top5conf
                    target_lower = target.lower()
                    is_positive = False
                    for rank, (idx, conf) in enumerate(zip(top5[:2], top5_conf[:2])):
                        class_name = model.names[int(idx)].lower()
                        conf_val = float(conf)
                        print(f"  {cell_id} rank{rank+1}: {class_name} {conf_val:.2f}")
                        if (target_lower in class_name or class_name in target_lower) and conf_val >= CONF_THRESHOLD:
                            is_positive = True
                    if is_positive:
                        positive_cells.append(cell_id)

    return cells, positive_cells, cell_centers

# ==========================================
# Main Loop
# ==========================================
def observe_loop():
    print("Starting Observation Loop...")
    previous_cells = []

    master_x, master_y, master_w, master_h = find_master_box()
    if master_x is None:
        print("Could not find captcha. Aborting.")
        return

    iteration = 0

    while iteration < MAX_ITERATIONS and running:
        iteration += 1
        print(f"\n--- Iteration {iteration}/{MAX_ITERATIONS} ---")

        full_img = capture_master_region(master_x, master_y, master_w, master_h)
        # cv2.imwrite(os.path.join(LOGS_DIR, "captcha_area.png"), full_img)

        header, grid = extract_captcha_elements(full_img, master_w, master_h)
        # cv2.imwrite(os.path.join(LOGS_DIR, "grid.png"), grid)

        action, target = analyze_instruction(header)

        if action == "REFRESH":
            refresh_btn, _ = get_buttons(master_h)
            print("Target: segmentation (squares) — clicking Refresh")
            slow_click(master_x + refresh_btn[0], master_y + refresh_btn[1])
            time.sleep(1)
            previous_cells = []
            # Re-locate after refresh
            master_x2, master_y2, master_w2, master_h2 = find_master_box()
            if master_x2 is not None:
                master_x, master_y, master_w, master_h = master_x2, master_y2, master_w2, master_h2
            continue

        if action == "UNKNOWN":
            print(f"Unknown target text. Waiting...")
            time.sleep(3)
            continue

        print(f"Target: {target}")

        current_cells, positive_cells, cell_centers = get_positive_cells(grid, target)
        print(f"Positive cells: {positive_cells}")

        if previous_cells and arrays_match(current_cells, previous_cells):
            print("Cells unchanged. Aborting.")
            return
        previous_cells = current_cells

        # --- CLICK POSITIVE CELLS (dynamic centers) ---
        if positive_cells:
            print("Clicking positive cells...")
            for cell_id in positive_cells:
                lx, ly = cell_centers[cell_id]
                global_cx = master_x + lx
                global_cy = master_y + ly
                print(f"  Clicking {cell_id} at ({global_cx}, {global_cy})")
                slow_click(global_cx, global_cy)
                time.sleep(0.3)

        # --- CLICK VERIFY (fixed from map files) ---
        refresh_btn, verify_btn = get_buttons(master_h)
        print(f"Clicking Verify at local {verify_btn}...")
        slow_click(master_x + verify_btn[0], master_y + verify_btn[1])
        time.sleep(2)

        # --- RE-LOCATE after verify ---
        master_x2, master_y2, master_w2, master_h2 = find_master_box()
        if master_x2 is None:
            print("Captcha disappeared — assuming solved. Aborting.")
            return
        master_x, master_y, master_w, master_h = master_x2, master_y2, master_w2, master_h2

        # --- READ ERROR MESSAGE ---
        full_img_after = capture_master_region(master_x, master_y, master_w, master_h)
        error_type = read_error_message(full_img_after)

        if error_type == "NEW_IMAGES":
            time.sleep(2)
            print("'Please select all matching images' — checking for new cells...")
            # Extract fresh cells from the current grid after the click
            _, grid_after = extract_captcha_elements(full_img_after, master_w, master_h)
            h, w = grid_after.shape[:2]
            ch, cw = h // 3, w // 3
            current_cells_after = []
            for row in range(3):
                for col in range(3):
                    y1, x1 = row * ch, col * cw
                    y2, x2 = (y1 + ch if row < 2 else h), (x1 + cw if col < 2 else w)
                    current_cells_after.append(grid_after[y1:y2, x1:x2])
            
            if arrays_match(previous_cells, current_cells_after):
                print("No new cells detected (YOLO missed some). Clicking Refresh.")
                time.sleep(2)
                m_x, m_y, m_w, m_h = find_master_box()
                if m_x is not None:
                    master_x, master_y, master_w, master_h = m_x, m_y, m_w, m_h
                refresh_btn, _ = get_buttons(master_h)
                slow_click(master_x + refresh_btn[0], master_y + refresh_btn[1])
                time.sleep(1)
            else:
                print("New cells detected (fading images). Continuing to solve...")
                time.sleep(4)
            
            previous_cells = []
            continue
        elif error_type == "TRY_AGAIN":
            print("'Please try again' — re-locating and clicking Refresh.")
            time.sleep(1)  # wait for UI expansion to settle
            m_x, m_y, m_w, m_h = find_master_box()
            if m_x is not None:
                master_x, master_y, master_w, master_h = m_x, m_y, m_w, m_h
            refresh_btn, _ = get_buttons(master_h)
            slow_click(master_x + refresh_btn[0], master_y + refresh_btn[1])
            time.sleep(2)
            previous_cells = []
            continue
        else:
            print("No error detected — assuming solved. Aborting.")
            return

    print(f"Reached max iterations ({MAX_ITERATIONS}). Aborting.")
    return

def on_bot_finished():
    global running
    running = False
    solve_btn.config(text="Solve")
    status_label.config(text="Idle")

def toggle_bot():
    global running

    if not running:
        running = True
        solve_btn.config(text="Stop")
        status_label.config(text="Running...")

        def run_bot():
            try:
                observe_loop()
            finally:
                root.after(0, on_bot_finished)

        Thread(target=run_bot, daemon=True).start()

    else:
        running = False
        solve_btn.config(text="Solve")
        status_label.config(text="Stopped")

# --- UI ---
root = tk.Tk()
root.resizable(False, False)
root.title("Captcha Solver")

root.geometry("400x150")

solve_btn = tk.Button(root, text="Solve", command=toggle_bot, width=12, height=2)
solve_btn.pack(pady=20)

status_label = tk.Label(root, text="Idle")
status_label.pack(pady=10)

root.mainloop()

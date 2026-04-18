import os, cv2, time, sys
from ultralytics import YOLO
from config import MODEL_PATH, MAX_ITERATIONS, CONF_THRESHOLD, LOGS_DIR, USE_SEGMENTATION, API_KEY
from ocr import analyze_instruction, read_error_message
from vision import extract_captcha_elements, arrays_match
from ui import find_master_box, capture_master_region, get_buttons, slow_click
from segmentation import handle_segmentation

try:
    print(f"Loading YOLO model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    if not API_KEY:
        print("SAM3 not detected. Segmentations will be skipped.")
except Exception as e:
    print(f"Warning: Failed to load YOLO model: {e}")
    model = None

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
        sys.exit(1)

    iteration = 0

    while iteration < MAX_ITERATIONS:
        iteration += 1
        print(f"\n--- Iteration {iteration}/{MAX_ITERATIONS} ---")

        full_img = capture_master_region(master_x, master_y, master_w, master_h)
        cv2.imwrite(os.path.join(LOGS_DIR, "captcha_area.png"), full_img)

        header, grid = extract_captcha_elements(full_img, master_w, master_h)
        cv2.imwrite(os.path.join(LOGS_DIR, "grid.png"), grid)

        action, target = analyze_instruction(header)

        if action == "SEGMENT":
            if USE_SEGMENTATION and API_KEY:
                print(f"Target (segmentation): {target}")
                handle_segmentation(grid, target, master_x, master_y, iteration)

                # Click Verify
                refresh_btn, verify_btn = get_buttons(master_h)
                print(f"Clicking Verify at local {verify_btn}...")
                slow_click(master_x + verify_btn[0], master_y + verify_btn[1])
                time.sleep(2)
            else:
                refresh_btn, _ = get_buttons(master_h)
                print("Target (segmentation): skipping (USE_SEGMENTATION=False) — clicking Refresh")
                slow_click(master_x + refresh_btn[0], master_y + refresh_btn[1])
                time.sleep(1)

            # Re-locate after action
            master_x2, master_y2, master_w2, master_h2 = find_master_box()
            if master_x2 is None:
                print("Captcha disappeared — assuming solved.")
                sys.exit(0)
            master_x, master_y, master_w, master_h = master_x2, master_y2, master_w2, master_h2
            previous_cells = []
            continue

        if action == "UNKNOWN":
            refresh_btn, _ = get_buttons(master_h)
            print(f"Unknown target text ({target}). Clicking Refresh...")
            slow_click(master_x + refresh_btn[0], master_y + refresh_btn[1])
            time.sleep(3)
            previous_cells = []
            # Re-locate after refresh
            m_x, m_y, m_w, m_h = find_master_box()
            if m_x is not None:
                master_x, master_y, master_w, master_h = m_x, m_y, m_w, m_h
            continue

        print(f"Target: {target}")

        current_cells, positive_cells, cell_centers = get_positive_cells(grid, target)
        print(f"Positive cells: {positive_cells}")

        if previous_cells and arrays_match(current_cells, previous_cells):
            print("Cells unchanged. Aborting.")
            sys.exit(0)
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
            sys.exit(0)
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
                time.sleep(2)
            
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
            sys.exit(0)

    print(f"Reached max iterations ({MAX_ITERATIONS}). Aborting.")
    sys.exit(0)
import numpy as np
import os, cv2, requests, time, base64
from config import LOGS_DIR, API_KEY
from ui import slow_click

GRID_SIZE = 4
OVERLAP_THRESHOLD = 0.1
HEADER_OFFSET = 147  # grid starts 147px below master TL

def handle_segmentation(grid_img, target_class, master_x, master_y, iteration, on_update=None):
    """
    Full segmentation pipeline:
      1. Call SAM3 API with grid image + target class
      2. Build mask, check 4x4 cells for overlap
      3. Click positive cells
      4. Save annotated debug image

    Returns list of positive cell_ids (for logging).
    """
    h, w = grid_img.shape[:2]

    # Encode grid to base64─
    _, buf = cv2.imencode(".png", grid_img)
    img_b64 = base64.b64encode(buf).decode("utf-8")

    # Call SAM3 API
    print(f"  [SAM3] Using SAM3  |  prompt: '{target_class}'")
    url = f"https://serverless.roboflow.com/sam3/concept_segment?api_key={API_KEY}"
    resp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json={
            "format": "polygon",
            "image": {"type": "base64", "value": img_b64},
            "prompts": [{"text": target_class}],
        },
    )
    print(f"  [SAM3] Status: {resp.status_code}")
    data = resp.json()

    # Build binary mask
    predictions = data["prompt_results"][0]["predictions"]
    print(f"  [SAM3] Predictions: {len(predictions)}")

    mask = np.zeros((h, w), dtype=np.uint8)
    for pred in predictions:
        for poly_pts in pred["masks"]:
            pts = np.array(poly_pts, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

    # Analyse cells & click
    cell_h = h // GRID_SIZE
    cell_w = w // GRID_SIZE
    positive = []

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w

            covered = np.count_nonzero(mask[y1:y2, x1:x2])
            ratio = covered / (cell_h * cell_w)

            if ratio >= OVERLAP_THRESHOLD:
                cell_id = f"cell_{row}_{col}"
                # center of cell in grid-local coords
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                # convert to global screen coords
                gx = master_x + cx
                gy = master_y + HEADER_OFFSET + cy
                positive.append((cell_id, gx, gy))
                print(f"  [SAM3]   ({row},{col})  {ratio:5.1%}  << POSITIVE")
            else:
                print(f"  [SAM3]   ({row},{col})  {ratio:5.1%}")

    # SHOW RESULT BEFORE CLICKING
    annotated = grid_img.copy()
    overlay = annotated.copy()
    overlay[mask > 0] = (0, 255, 0)
    annotated = cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0)

    if on_update:
        on_update(annotated)
        time.sleep(0.5)

    # Click positive cells
    print(f"  [SAM3] Positive: {len(positive)} cells")
    for cell_id, gx, gy in positive:
        print(f"  [SAM3]   Clicking {cell_id} at ({gx}, {gy})")
        slow_click(gx, gy)
        time.sleep(0.3)

    # Save annotated debug image
    annotated = grid_img.copy()
    overlay = annotated.copy()
    overlay[mask > 0] = (0, 250, 0)
    cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            y1, y2 = row * cell_h, (row + 1) * cell_h
            x1, x2 = col * cell_w, (col + 1) * cell_w
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 1)

    for cell_id, _, _ in positive:
        # parse row,col back from cell_id for drawing
        parts = cell_id.split("_")
        r, c = int(parts[1]), int(parts[2])
        cx = (c * cell_w + (c + 1) * cell_w) // 2
        cy = (r * cell_h + (r + 1) * cell_h) // 2
        cv2.circle(annotated, (cx, cy), 8, (0, 0, 255), -1)
        cv2.circle(annotated, (cx, cy), 10, (255, 255, 255), 2)

    out_path = os.path.join(LOGS_DIR, f"sam3_annotated_{iteration}.png")
    cv2.imwrite(out_path, annotated)
    print(f"  [SAM3] Saved: {out_path}")

    return [cid for cid, _, _ in positive]

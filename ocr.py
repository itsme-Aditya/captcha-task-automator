import os, cv2, nltk, pytesseract
from config import base_path

nltk.data.path.append(os.path.join(base_path, "assets", "nltk_data"))
from nltk.tokenize import word_tokenize

tesseract_path = os.path.join(base_path, "assets", "tesseract", "tesseract.exe")
pytesseract.pytesseract.tesseract_cmd = tesseract_path

def analyze_instruction(header_crop):
    scaled = cv2.resize(header_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    inv = cv2.bitwise_not(gray)
    text = pytesseract.image_to_string(inv, config='--psm 6').lower()
    tokens = word_tokenize(text)

    classes = {
        "car", "bus", "motorcycle", "bicycle", "crosswalk", "boat",
        "chimney", "bridge", "hydrant", "stair", "stairs", "palm"
    }

    is_segmentation = "square" in tokens or "squares" in tokens

    # Extract target class
    if "traffic" in tokens:
        target = "traffic light"
    elif "fire" in tokens and "hydrant" in tokens:
        target = "hydrant"
    else:
        target = None
        for token in tokens:
            singular = token[:-1] if token.endswith('s') else token
            if token in classes:
                target = token
                break
            if singular in classes:
                target = singular
                break

    if is_segmentation:
        return ("SEGMENT", target)

    if target:
        return ("SOLVE", target)

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

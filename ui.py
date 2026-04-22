import numpy as np
import mss, cv2, pyautogui, time
from config import REFRESH_NORMAL, VERIFY_NORMAL, REFRESH_ERROR, VERIFY_ERROR

pyautogui.FAILSAFE = True

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

def is_error_state(master_h):
    return master_h >= 720

def get_buttons(master_h):
    if is_error_state(master_h):
        return REFRESH_ERROR, VERIFY_ERROR
    return REFRESH_NORMAL, VERIFY_NORMAL

def slow_click(gx, gy):
    pyautogui.moveTo(gx, gy + 15, duration=1.25)
    time.sleep(0.1)
    pyautogui.click()

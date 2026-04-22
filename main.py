import os
import sys
import tkinter as tk
import customtkinter as ctk
from threading import Thread, Lock
from PIL import Image, ImageTk
from config import MODEL_PATH, API_KEY, base_path
from captcha_solver import observe_loop

LOG_LEVEL = "USER"  # USER or DEBUG

gif_completed_once = False

running = False
sct = None
model = None
cv2 = None
np = None
mss = None
pytesseract = None
ssim = None

BG = "#121212"
PANEL = "#1E1E1E"
ACCENT = "#00FF9C"
ACCENT_HOVER = "#00cc7a"
TEXT = "#E5E5E5"

if not API_KEY:
    print("SAM3 not detected. Segmentations will be skipped.")


def log_user(msg):
    print(msg)

def log_debug(msg):
    if LOG_LEVEL == "DEBUG":
        print(msg)

model_lock = Lock()
def load_resources():
    global model, cv2, np, mss, pytesseract, ssim

    if model is not None:
        return
    
    with model_lock:
        if model is not None:
            return  # already loaded

        log_user("Initializing model...")

        import cv2 as _cv2
        import numpy as _np
        import mss as _mss
        import pytesseract as _pytesseract
        from ultralytics import YOLO as _YOLO
        from skimage.metrics import structural_similarity as _ssim

        cv2 = _cv2
        np = _np
        mss = _mss
        pytesseract = _pytesseract
        ssim = _ssim

        # setup tesseract again AFTER import
        tesseract_path = os.path.join(base_path, "assets", "tesseract", "tesseract.exe")
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

        # load model
        try:
            model = _YOLO(MODEL_PATH)
            log_user("Model ready")
        except Exception as e:
            print(f"Failed to load model: {e}")
            model = None

# ----------------------------------------------------
# Functions for Tkinter Graphical User Interface
# ----------------------------------------------------
def on_bot_finished():
    global running
    running = False
    spinner.stop()
    spinner.set(0)
    solve_btn.configure(text="Solve")
    status_label.configure(text="Idle")
    preview_label.configure(image=None, text="Preview")
    preview_label.image = None

def toggle_bot():
    global running

    if not running:
        running = True
        solve_btn.configure(text="Stop")

        def run_bot():
            global sct
            try:
                # Start spinner (for solving)
                def start_running():
                    spinner.start()
                    # status_label.configure(text="Running...", text_color="#00FF9C")
                    status_label.configure(text="Running...", text_color= ACCENT)

                root.after(0, start_running)
                load_resources()
                import mss as _mss
                sct = _mss.mss()
                observe_loop(model, on_update=update_preview)

            finally:
                root.after(0, on_bot_finished)

        Thread(target=run_bot, daemon=True).start()

    else:
        running = False
        solve_btn.configure(text="Solve")
        status_label.configure(text="Stopped", text_color="#FF4C4C")
        spinner.stop()
        spinner.set(0)

def log(message):
    def append():
        log_box.configure(state="normal")
        log_box.insert("end", message + "\n")

        # limit log size
        if int(log_box.index("end-1c").split(".")[0]) > 200:
            log_box.delete("1.0", "2.0")

        log_box.configure(state="disabled")
        log_box.see("end")

    root.after(0, append)

class PrintRedirector:
    def write(self, message):
        if message.strip():
            log(message.strip())

    def flush(self):
        pass

sys.stdout = PrintRedirector()
sys.stderr = PrintRedirector()


def clear_logs():
    log_box.configure(state='normal')
    log_box.delete(1.0, tk.END)
    log_box.configure(state='disabled')

def animate_gif():
    global gif_index, gif_running, gif_completed_once

    if not gif_running or not gif_frames:
        return

    frame = gif_frames[gif_index]
    gif_label.configure(image=frame)
    gif_label.lift()

    gif_index += 1

    if gif_index >= len(gif_frames):
        gif_completed_once = True
        gif_index = 0  # loop if still running

    root.after(100, animate_gif)

def preload_model():
    global gif_running

    try:
        root.after(0, lambda: status_label.configure(text="Preloading model..."))

        # START GIF
        gif_running = True
        root.after(0, animate_gif)

        # LOAD MODEL (blocking)
        load_resources()

        # AFTER LOADING → wait for GIF to finish current loop
        def stop_when_ready():
            if not gif_completed_once:
                root.after(100, stop_when_ready)
                return

            # NOW stop
            global gif_running
            gif_running = False
            gif_label.place_forget()

            solve_btn.configure(state="normal")
            status_label.configure(text="Idle", text_color="#888888")

        root.after(0, stop_when_ready)

    except Exception as e:
        log(f"Preload failed: {e}")

gif_frames = []
gif_index = 0
gif_label = None
gif_running = False

def load_gif(path):
    global gif_frames
    gif = Image.open(path)

    gif_frames = []
    try:
        while True:
            frame = gif.copy().resize((window_width, window_height))
            gif_frames.append(ImageTk.PhotoImage(frame))
            gif.seek(len(gif_frames))  # next frame
    except EOFError:
        pass

def update_preview(img):
    def update():
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        pil_img = pil_img.resize((220, 220), Image.LANCZOS)

        tk_img = ImageTk.PhotoImage(pil_img)

        preview_label.configure(image=tk_img, text="")
        preview_label.image = tk_img  # keep reference

    root.after(0, update)

# ------------------- UI -------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
icon_path = os.path.join(base_path, "assets", "icon.ico")
root.iconbitmap(icon_path)
root.configure(fg_color=BG)
root.title("Captcha Task Automator")

control_frame = ctk.CTkFrame(root, fg_color="transparent")
control_frame.pack(pady=15)

window_width = 500
window_height = 600

root.update_idletasks()

screen_width = root.winfo_screenwidth()

x = screen_width - window_width - 300
y = 200

root.geometry(f"{window_width}x{window_height}+{x}+{y}")

root.attributes("-topmost", True)

gif_label = tk.Label(root, bd=0)
gif_label.place(x=0, y=0, relwidth=1, relheight=1)
gif_label.lift()

gif_path = os.path.join(base_path, "assets", "intro1.gif")
load_gif(gif_path)

gif_running = True
animate_gif()

# ------------------- Solve Button -------------------
solve_btn = ctk.CTkButton(
    control_frame,
    text="Solve",
    command=toggle_bot,
    width=140,
    height=40,
    fg_color=ACCENT,
    hover_color=ACCENT_HOVER,
    text_color="black"
)
solve_btn.pack()
solve_btn.configure(state="disabled")

# ------------------- Spinner (Progress Bar) -------------------
spinner = ctk.CTkProgressBar(
    control_frame,
    width=180,
    progress_color=ACCENT,
    fg_color="#2a2a2a",
    corner_radius=10
)
spinner.pack(pady=8)
spinner.set(0)

# ------------------- Status Label -------------------
status_label = ctk.CTkLabel(
    control_frame,
    text="Idle",
    text_color=ACCENT
)
status_label.pack()

# ------------------- Preview -------------------
preview_frame = ctk.CTkFrame(
    root,
    width=240,
    height=240,
    fg_color=PANEL,
    corner_radius=12
)
preview_frame.pack(pady=10)
preview_frame.pack_propagate(False)

preview_label = ctk.CTkLabel(preview_frame, text="Preview")
preview_label.place(relx=0.5, rely=0.5, anchor="center")

# ------------------- Log Box -------------------
log_box = ctk.CTkTextbox(
    root,
    width=460,
    height=140,
    fg_color=PANEL,
    text_color=TEXT,
    corner_radius=10,
    font=("Cascadia Code", 11)
)
log_box.pack(padx=15, pady=10, fill="both", expand=True)

log_box.configure(state="disabled")

# ------------------- Clear Button -------------------
clear_btn = ctk.CTkButton(
    root,
    text="Clear Logs",
    command=clear_logs,
    width=120,
    fg_color="#333333",
    hover_color="#444444"
)
clear_btn.pack(pady=(0, 10))

Thread(target=preload_model, daemon=True).start()
root.mainloop()
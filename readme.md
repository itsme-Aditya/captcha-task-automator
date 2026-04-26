# Captcha Task Automator

An advanced, AI-driven automation suite designed to solve complex visual captchas. This system integrates Object Detection (**YOLOv8**), Computer Vision (**OpenCV**), and Cloud-based Segmentation (**SAM3**) to automate grid-based and selection-based tasks with high precision.

## Demo

[![Watch the demo](https://img.youtube.com/vi/PdfPxPpi-E0/0.jpg)](https://youtu.be/PdfPxPpi-E0)

## 🚀 Key Features

* **Intelligent Instruction Parsing**: Uses Tesseract OCR and NLTK to interpret captcha headers and identify target objects like traffic lights, hydrants, or stairs.
* **Hybrid Solving Engine**:
    * **Classification Mode**: Utilizes a YOLO model to analyze 3x3 grids for specific object matches.
    * **Segmentation Mode**: Leverages the Roboflow SAM3 API for pixel-level accuracy in "click the squares" style tasks.
* **Dynamic UI Detection**: Automatically locates the captcha widget on the screen by scanning for specific color signatures (HSV masking) and UI boundaries.
* **Advanced Error Handling**: Detects state changes such as "Please try again" or "Select matching images" and manages fading images by comparing Structural Similarity (SSIM) between frames.
* **Real-time Dashboard**: A CustomTkinter GUI featuring a live processing preview, log console, and automated resource preloading.

## 📁 Project Structure

* `main.py`: The entry point. Manages the GUI, threading, and resource initialization.
* `captcha_solver.py`: The core logic controller. Orchestrates the observation loop and decision-making flow.
* `segmentation.py`: Handles API communication with SAM3 and maps masks to screen coordinates.
* `vision.py`: Responsible for image preprocessing, cell extraction, and SSIM-based change detection.
* `ocr.py`: Extracts and tokenizes text from the captcha header to determine the required action.
* `ui.py`: Manages screen capturing and human-like mouse interactions.
* `config.py`: Centralized configuration for model paths, API keys, and UI coordinate offsets.

## 🛠 Tech Stack

| Component | Technology |
| :--- | :--- |
| **GUI** | CustomTkinter, Pillow |
| **AI/ML** | Ultralytics YOLOv8, Roboflow SAM3 |
| **Vision** | OpenCV, Scikit-Image (SSIM) |
| **OCR/NLP** | PyTesseract, NLTK |
| **Automation** | PyAutoGUI, MSS |

## ⚙️ Setup & Installation

### Prerequisites
1.  **Python 3.9+**
2.  **Tesseract OCR**: Install Tesseract on your system and ensure the executable path is correctly mapped in `ocr.py`.
3.  **Roboflow API Key**: Required for segmentation features.

### Installation
1.  Clone the repository to your local machine.
2.  Install the required dependencies:
    ```bash
    pip install ultralytics opencv-python nltk pytesseract mss pyautogui customtkinter pillow scikit-image requests python-dotenv
    ```
3.  Create a `.env` file in the root directory and add your API key:
    ```env
    ROBOFLOW_API_KEY=your_actual_key_here
    ```
4.  Ensure your trained model is placed at `assets/final_model.pt`.

## 🖥 Usage

1.  **Launch**: Run `python main.py`.
2.  **Preload**: The application will automatically initialize the YOLO model and OCR assets while playing an intro animation.
3.  **Position**: Ensure the captcha is visible on your primary monitor.
4.  **Solve**: Click the **"Solve"** button. The bot will take over mouse control to complete the task.
5.  **Failsafe**: To abort the bot at any time, move your mouse to any corner of the screen to trigger the PyAutoGUI failsafe.

---
**Disclaimer**: *This project is intended for educational and research purposes only. Users are responsible for ensuring their use of this software complies with the Terms of Service of any websites accessed.*
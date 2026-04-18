import os, sys
from dotenv import load_dotenv

LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

def get_base_path():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    return os.path.dirname(__file__)

base_path = get_base_path()

MODEL_PATH = os.path.join(base_path, "assets", "final_model.pt")

MAX_ITERATIONS = 5
CONF_THRESHOLD = 0.30

# Fixed button centers (local to master TL) from map files
REFRESH_NORMAL = (28, 679)
VERIFY_NORMAL  = (418, 679)
REFRESH_ERROR  = (28, 718)
VERIFY_ERROR   = (418, 718)
USE_SEGMENTATION = True

load_dotenv()
API_KEY = os.getenv("ROBOFLOW_API_KEY")

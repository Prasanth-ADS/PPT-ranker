import os
from dotenv import load_dotenv

load_dotenv()

# Ollama Configuration
# Using Llama 3.1 8B (quantized) for better quality evaluations
OLLAMA_MODEL = "llama3.1:8b-instruct-q4_K_M"
OLLAMA_HOST = "http://localhost:11434"  # Default Ollama server

# Tesseract Path Configuration
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")

# Directory Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOAD_DIR = os.path.join(BASE_DIR, os.getenv("DOWNLOAD_DIR", "downloads"))
OUTPUT_DIR = os.path.join(BASE_DIR, os.getenv("OUTPUT_DIR", "output"))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Ensure directories exist
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Processing Config
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 5))
MAX_WORKERS = 8  # For parallel downloads (increased from 4)

# Cache Configuration
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"

# OCR Configuration
MIN_IMAGE_SIZE = 100  # Skip images smaller than this (pixels) - likely icons/logos
OCR_CONFIG = "--psm 6"  # Optimized for block text detection (faster than default)

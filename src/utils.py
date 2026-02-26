from pathlib import Path

# Project root = folder that contains "src"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR  / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR =  PROJECT_ROOT / "models"

# Auto-create folders if they dont already exists
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = os.getenv("PROJ_ROOT")
if PROJ_ROOT is None:
    logger.error(
        "PROJ_ROOT environment variable not set. Please set it in your .env file."
    )
    raise ValueError(
        "PROJ_ROOT environment variable not set. Please set it in your .env file."
    )
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

PROJ_ROOT = Path(PROJ_ROOT)

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    logger.warning(
        "Hugging Face token not found. Please set the HF_TOKEN environment variable in your .env file."
    )

SEED = 2025
TRAIN_SPLIT = 0.7
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

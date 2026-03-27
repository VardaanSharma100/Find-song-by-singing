from datasets import load_dataset
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.logger import get_logger
logger = get_logger(__name__)

try:
    print("Starting download....")
    logger.info("Starting download in make_data.")

    dataset = load_dataset("amanteur/CHAD_hummings")

    print("Download complete.")
    print(dataset)
    logger.info("Download complete in make_data.")
except Exception as e:
    logger.error(f"Error in make_data: {e}")

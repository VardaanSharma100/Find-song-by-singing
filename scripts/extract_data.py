import os
import sys
import soundfile as sf
from datasets import load_dataset


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.logger import get_logger
logger = get_logger(__name__)

try:
    print("Loading Datasets...")
    logger.info("Loading Datasets in extract_data...")

    dataset = load_dataset('amanteur/CHAD_hummings')

    train_data = dataset['train']

    output_dir = 'data/raw/hummings'

    os.makedirs(output_dir, exist_ok = True)

    for i, item in enumerate(train_data):
        audio_array = item['wav']['array']
        sample_rate = item['wav']['sampling_rate']

        file_path = os.path.join(output_dir, f"hum_{i}.wav")

        sf.write(file_path, audio_array, sample_rate)

        if i > 0 and i % 500 == 0:
            print(f"Successfully saved {i} files")
            logger.info(f"Successfully saved {i} files")
    
    logger.info("Extraction complete.")
except Exception as e:
    logger.error(f"Error in extract_data: {e}")

import os 
import sys
import librosa
import numpy as np
from tqdm import tqdm


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.logger import get_logger
logger = get_logger(__name__)

RAW_HUM_DIR = "data/raw/hummings"
RAW_SONG_DIR = "data/raw/songs"
PROCESSED_DIR = "data/processed"

try:
    os.makedirs(f"{PROCESSED_DIR}/hummings/mel", exist_ok=True)
    os.makedirs(f"{PROCESSED_DIR}/hummings/pitch", exist_ok=True)
    os.makedirs(f"{PROCESSED_DIR}/songs/mel", exist_ok=True)
    os.makedirs(f"{PROCESSED_DIR}/songs/pitch", exist_ok=True)

    SR = 8000
    DURATION = 16
    N_MELS = 64
    HOP_LENGTH = 256

    def extract_features(file_path):
        try:
            y, sr = librosa.load(file_path, sr=SR, duration=DURATION)

            target_len = SR * DURATION
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)))
            else:
                y = y[:target_len]
            
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH
            )
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)

            f0, voiced_flag, voiced_probs = librosa.pyin(
                y,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr,
                hop_length=HOP_LENGTH
            )

            f0 = np.nan_to_num(f0)

            min_len = min(mel_db.shape[1], f0.shape[0])
            mel_db = mel_db[:, :min_len]
            f0 = f0[:min_len]

            return mel_db, f0
        except Exception as e:
            logger.error(f"Error in extract_features for {file_path}: {e}")
            raise e

    for category, folder in [("hummings", RAW_HUM_DIR),("songs", RAW_SONG_DIR)]:
        print(f"\nProcessing {category}...")
        logger.info(f"Processing category {category}...")
        files = [f for f in os.listdir(folder) if f.endswith(".wav")]

        for filename in tqdm(files):
            file_id = filename.split('_')[1].split('.')[0]
            input_path = os.path.join(folder, filename)

            try:
                mel, pitch = extract_features(input_path)
                np.save(f"{PROCESSED_DIR}/{category}/mel/{file_id}.npy", mel)
                np.save(f"{PROCESSED_DIR}/{category}/pitch/{file_id}.npy", pitch)
            except Exception as e:
                print(f"Error on {filename}: {e}")
                logger.error(f"Error saving features for {filename}: {e}")
    print("done......")
    logger.info("Preprocessing complete.")
except Exception as e:
    logger.error(f"Critical error in top-level preprocess script: {e}")
            

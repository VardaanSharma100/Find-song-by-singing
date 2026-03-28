import librosa
import numpy as np
import torch
import sys
import os
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger
logger = get_logger(__name__)

def process_single_audio(file_path, offset=0.0, duration=16.0, sr=22050, n_mels=64, max_pitch=500.0):
    try:
        y, sr = librosa.load(file_path, sr=sr, offset=offset, duration=duration)

        target_length = int(sr * duration)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        mel_tensor = torch.tensor(mel_spec_db, dtype=torch.float32).unsqueeze(0) 

        f0 = librosa.yin(y=y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0 = np.nan_to_num(f0) 
        
        f0 = np.clip(f0, 0, max_pitch) / max_pitch
        
        pitch_tensor = torch.tensor(f0, dtype=torch.float32)

        return mel_tensor, pitch_tensor
    except Exception as e:
        logger.error(f"Error processing audio in {file_path}: {e}")
        
        return torch.zeros((1, n_mels, int(sr * duration // 512))), torch.zeros((1, int(sr * duration // 512)))

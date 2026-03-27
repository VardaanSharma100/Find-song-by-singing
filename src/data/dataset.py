import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import Dataset


sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger
logger = get_logger(__name__)

class HummingDataset(Dataset):
    def __init__(self, processed_dir):
        try:
            self.hum_mel_dir = os.path.join(processed_dir, 'hummings', 'mel')       
            self.hum_pitch_dir = os.path.join(processed_dir, 'hummings', 'pitch')   
            self.song_mel_dir = os.path.join(processed_dir, 'songs', 'mel')
            self.song_pitch_dir = os.path.join(processed_dir, 'songs', 'pitch')     

            self.file_ids = self._get_valid_ids()
        except Exception as e:
            logger.error(f"Error initializing HummingDataset: {e}")
            self.file_ids = []

    def _get_valid_ids(self):
        try:
            hum_mels = set([f.split('.')[0] for f in os.listdir(self.hum_mel_dir)]) 
            hum_pitches = set([f.split('.')[0] for f in os.listdir(self.hum_pitch_dir)])
            song_mels = set([f.split('.')[0] for f in os.listdir(self.song_mel_dir)])
            song_pitches = set([f.split('.')[0] for f in os.listdir(self.song_pitch_dir)])

            return list(hum_mels.intersection(hum_pitches, song_mels, song_pitches))
        except Exception as e:
            logger.error(f"Error finding valid dataset IDs: {e}")
            return []

    def __len__(self):
        return len(self.file_ids)

    def _load_tensors(self, path):
        try:
            return torch.from_numpy(np.load(path)).float()
        except Exception as e:
            logger.error(f"Error loading tensor from {path}: {e}")
            return torch.tensor([])

    def __getitem__(self, idx):
        try:
            anchor_id = self.file_ids[idx]

            hum_mel = self._load_tensors(os.path.join(self.hum_mel_dir,f'{anchor_id}.npy'))
            hum_pitch = self._load_tensors(os.path.join(self.hum_pitch_dir,f'{anchor_id}.npy'))

            pos_mel = self._load_tensors(os.path.join(self.song_mel_dir,f'{anchor_id}.npy'))
            pos_pitch = self._load_tensors(os.path.join(self.song_pitch_dir,f'{anchor_id}.npy'))

            negative_id = random.choice(self.file_ids)
            while negative_id == anchor_id:
                negative_id = random.choice(self.file_ids)

            neg_mel = self._load_tensors(os.path.join(self.song_mel_dir,f'{negative_id}.npy'))
            neg_pitch = self._load_tensors(os.path.join(self.song_pitch_dir,f'{negative_id}.npy'))

            return {
                'anchor': {
                    'mel': hum_mel.unsqueeze(0),
                    'pitch': hum_pitch
                },
                'positive': {
                    'mel': pos_mel.unsqueeze(0),
                    'pitch': pos_pitch
                },
                'negative': {
                    'mel': neg_mel.unsqueeze(0),
                    'pitch': neg_pitch
                }
            }
        except Exception as e:
            logger.error(f"Error getting item at index {idx}: {e}")
            
            return {'anchor': {}, 'positive': {}, 'negative': {}}

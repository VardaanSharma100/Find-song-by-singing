import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class HummingDataset(Dataset):
    def __init__(self, processed_dir):

        self.hum_mel_dir = os.path.join(processed_dir, 'hummings', 'mel')
        self.hum_pitch_dir = os.path.join(processed_dir, 'hummings', 'pitch')
        self.song_mel_dir = os.path.join(processed_dir, 'songs', 'mel')
        self.song_pitch_dir = os.path.join(processed_dir, 'songs', 'pitch')

        self.file_ids = self._get_valid_ids()

    def _get_valid_ids(self):

        hum_mels = set([f.split('.')[0] for f in os.listdir(self.hum_mel_dir)])
        hum_pitches = set([f.split('.')[0] for f in os.listdir(self.hum_pitch_dir)])
        song_mels = set([f.split('.')[0] for f in os.listdir(self.song_mel_dir)])
        song_pitches = set([f.split('.')[0] for f in os.listdir(self.song_pitch_dir)])
        
        return list(hum_mels.intersection(hum_pitches, song_mels, song_pitches))

    def __len__(self):
        return len(self.file_ids)

    def _load_tensors(self, path):
        return torch.from_numpy(np.load(path)).float()
    
    def __getitem__(self, idx):
        
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

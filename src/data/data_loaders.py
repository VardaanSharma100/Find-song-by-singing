import torch
import sys
import os
from torch.utils.data import DataLoader, random_split
from .dataset import HummingDataset


sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger
logger = get_logger(__name__)

def get_dataloaders(data_dir="data/processed", batch_size=32, seed=42):
    try:
        print("Initializing Data Loaders and splitting dataset...")
        logger.info("Initializing Data Loaders and splitting dataset...")
        
        full_dataset = HummingDataset(data_dir)
        
        total_size = len(full_dataset)
        train_size = int(0.80 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size
        
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, 
            [train_size, val_size, test_size], 
            generator=generator
        )
        
        print(f" -> Data Split: {train_size} Train | {val_size} Val | {test_size} Test")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    except Exception as e:
        logger.error(f"Error initializing dataloaders: {e}")
        return None, None, None

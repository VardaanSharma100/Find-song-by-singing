import torch
import os

from src.data.data_loaders import get_dataloaders
from src.models.siamese import SiameseNetwork
from src.training.loss import TripletLoss
from src.training.train_loop import train_model
from src.utils.config import EMBEDDING_DIM

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs("models", exist_ok=True)

    train_loader, val_loader, _ = get_dataloaders(data_dir="data/processed", batch_size=32)

    model = SiameseNetwork(embedding_dim=EMBEDDING_DIM)
    

    checkpoint_path = "models/best_hum_model.pth" 
    
    if os.path.exists(checkpoint_path):
        print(f"[*] Found existing model at {checkpoint_path}. Loading weights...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("[*] No existing model found. Starting with a fresh, untrained brain.")

    criterion = TripletLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    print("Starting the production training engine...")
    trained_model = train_model(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,  
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=50, 
        device=device,
        patience=5
    )

if __name__ == "__main__":
    main()
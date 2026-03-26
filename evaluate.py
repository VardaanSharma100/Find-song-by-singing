import torch
from tqdm import tqdm

from src.data.data_loaders import get_dataloaders
from src.models.siamese import SiameseNetwork
from src.training.loss import TripletLoss
from src.utils.config import EMBEDDING_DIM

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading Final Exam...\n")

    _, _, test_loader = get_dataloaders(data_dir="data/processed", batch_size=32)

    model = SiameseNetwork(embedding_dim=EMBEDDING_DIM)
    model.load_state_dict(torch.load("models/best_hum_model.pth", map_location=device))
    model.to(device)
    model.eval()

    criterion = TripletLoss(margin=1.0)
    total_test_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing on unseen data"):
            a_mel, a_pitch = batch['anchor']['mel'].to(device), batch['anchor']['pitch'].to(device)
            p_mel, p_pitch = batch['positive']['mel'].to(device), batch['positive']['pitch'].to(device)
            n_mel, n_pitch = batch['negative']['mel'].to(device), batch['negative']['pitch'].to(device)

            anchor_out = model(a_mel, a_pitch)
            positive_out = model(p_mel, p_pitch)
            negative_out = model(n_mel, n_pitch)

            loss = criterion(anchor_out, positive_out, negative_out)
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"\n======================================")
    print(f" (TEST LOSS): {avg_test_loss:.4f}")
    print(f"======================================")

if __name__ == "__main__":
    main()
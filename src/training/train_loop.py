import torch
from tqdm import tqdm

def train_model(model, dataloader, criterion, optimizer, num_epochs=10, device='cuda'):

    model.to(device)

    model.train(device)

    model.train()

    for epoch in range(num_epochs):
        tota_loss = 0.0
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

        for batch in tqdm(dataloader, desc='Training'):

            a_mel = batch['anchor']['mel'].to(device)
            a_pitch = batch['anchor']['pitch'].to(device)

            p_mel = batch['positive']['mel'].to(device)
            p_pitch = batch['positive']['pitch'].to(device)

            n_mel = batch['negative']['mel'].to(device)
            n_pitch = batch['negative']['pitch'].to(device)


            optimizer.zero_grad()

            anchor_out = model(a_mel, a_pitch)
            positive_out = model(p_mel, p_pitch)
            negative_out = model(n_mel, n_pitch)

            loss = criterion(anchor_out, positive_out, negative_out)

            loss.backwards()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
    
    print("\n Training Complete")

    return model
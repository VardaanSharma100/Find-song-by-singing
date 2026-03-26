import torch
from tqdm import tqdm

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler=None, num_epochs=50, device='cuda', patience=5):
    model.to(device)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train() 
        total_train_loss = 0.0
        
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        for batch in tqdm(train_dataloader, desc="Training"):
            
            a_mel, a_pitch = batch['anchor']['mel'].to(device), batch['anchor']['pitch'].to(device)
            p_mel, p_pitch = batch['positive']['mel'].to(device), batch['positive']['pitch'].to(device)
            n_mel, n_pitch = batch['negative']['mel'].to(device), batch['negative']['pitch'].to(device)

            optimizer.zero_grad()
            anchor_out = model(a_mel, a_pitch)
            positive_out = model(p_mel, p_pitch)
            negative_out = model(n_mel, n_pitch)

            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)

        model.eval() 
        total_val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validating"):
                a_mel, a_pitch = batch['anchor']['mel'].to(device), batch['anchor']['pitch'].to(device)
                p_mel, p_pitch = batch['positive']['mel'].to(device), batch['positive']['pitch'].to(device)
                n_mel, n_pitch = batch['negative']['mel'].to(device), batch['negative']['pitch'].to(device)

                anchor_out = model(a_mel, a_pitch)
                positive_out = model(p_mel, p_pitch)
                negative_out = model(n_mel, n_pitch)

                val_loss = criterion(anchor_out, positive_out, negative_out)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if scheduler is not None:
            scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "models/best_hum_model.pth")
            print(" -> Checkpoint: New best validation model saved!")
        else:
            patience_counter += 1
            print(f" -> No improvement. Early stopping patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print("\n[!] Early Stopping Triggered! The AI has started overfitting.")
                break

    print("\nTraining Complete! Best weights are saved in 'models/best_hum_model.pth'")
    return model
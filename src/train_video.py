"""
src/train_video.py
Dedicated Training Loop for PRISM Temporal Video Model (R3D_18).
Designed for Kaggle execution with FP16 AMP.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from sklearn.metrics import roc_auc_score

from dataset_video import CelebDFVideoDataset
from model_video import PRISMVideoNetwork

def train_one_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    scaler: GradScaler, 
    device: torch.device, 
    epoch: int
):
    model.train()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad(set_to_none=True)
        
        videos = batch["video"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        
        with torch.amp.autocast(device_type=device.type):
            out = model(videos)
            loss = criterion(out["logit"].squeeze(1), labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

    return total_loss / max(1, len(dataloader))

def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            videos = batch["video"].to(device, non_blocking=True)
            labels = batch["label"]
            
            with torch.amp.autocast(device_type=device.type):
                out = model(videos)
                probs = torch.sigmoid(out["logit"].squeeze(1))
                
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    auc = roc_auc_score(all_labels, all_probs)
    # Binary predictions using 0.5 threshold
    preds = [1 if p > 0.5 else 0 for p in all_probs]
    acc = sum(1 for p, l in zip(preds, all_labels) if p == l) / max(1, len(all_labels))
    
    print(f"Validation AUC: {auc:.4f} | Accuracy: {acc:.4f}")
    return auc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Video Training on {device}")
    
    dataset_dir = "/kaggle/input/celeb-df-v2"
    if not os.path.exists(dataset_dir):
        print(f"Warning: Dataset not found at {dataset_dir}. Ensure you are on Kaggle or update path.")
        
    train_dataset = CelebDFVideoDataset(root_dir=dataset_dir, num_frames=16)
    
    # We use a very small batch size due to 3D video tensors taking massive VRAM
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    
    model = PRISMVideoNetwork(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scaler = GradScaler()
    
    num_epochs = 10
    best_auc = 0.0
    os.makedirs("checkpoints", exist_ok=True)
    
    print(f"Dataset Size: {len(train_dataset)} videos. Batches per epoch: {len(train_loader)}")
    
    for epoch in range(1, num_epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch)
        print(f"--- Epoch {epoch} Completed | Avg Loss: {loss:.4f} ---")
        
        # Optionally add a validation set here
        # val_auc = evaluate(model, val_loader, device)
        # if val_auc > best_auc:
        #    best_auc = val_auc
        #    torch.save(model.state_dict(), f"checkpoints/best_video_model.pth")
        
        # Save epoch checkpoint
        torch.save(model.state_dict(), f"checkpoints/video_model_epoch_{epoch}.pth")

if __name__ == "__main__":
    # Wrap in basic test block if executed directly locally
    main()

"""
src/train.py
PRISM multi-task training - Kaggle Dual T4 GPU edition.
FP16 AMP + DataParallel + Model Checkpointing.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler

import numpy as np
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

_bce_loss = nn.BCEWithLogitsLoss()


def soft_dice_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Soft Dice Loss over predicted heatmap and ground truth mask.
    Formula: 1 - (2 * (pred*gt).sum() + eps) / (pred.sum() + gt.sum() + eps)
    """
    eps = 1e-6
    pred_flat = pred.reshape(-1)
    gt_flat = gt.reshape(-1)
    intersection = (pred_flat * gt_flat).sum()
    return 1.0 - (2.0 * intersection + eps) / (pred_flat.sum() + gt_flat.sum() + eps)


def multitask_loss(
    logit: torch.Tensor,  # [B, 1] raw logit
    mask: torch.Tensor,  # [B, 1, H, W] predicted mask (sigmoid applied)
    label: torch.Tensor,  # [B] float labels {0,1}
    gt_mask: torch.Tensor,  # [B, 1, H, W] ground truth mask
) -> tuple:
    """Total_Loss = L_BCE + 2.0 * L_Dice"""
    l_bce = _bce_loss(logit.squeeze(1), label)
    l_dice = soft_dice_loss(mask, gt_mask)
    total = l_bce + 2.0 * l_dice
    return total, l_bce, l_dice


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int = 0,
) -> dict:
    model.train()
    total_loss_sum = 0.0
    bce_sum = 0.0
    dice_sum = 0.0
    n_batches = 0

    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)

        rgb = batch["rgb"].to(device, non_blocking=True)
        noise = batch["noise"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        gt_mask = batch["mask"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type):
            out = model(rgb, noise)
            total, l_bce, l_dice = multitask_loss(
                out["logit"], out["mask"], label, gt_mask
            )

        scaler.scale(total).backward()
        scaler.step(optimizer)
        scaler.update()

        # Strategically clear lingering activations
        torch.cuda.empty_cache()

        total_loss_sum += total.item()
        bce_sum += l_bce.item()
        dice_sum += l_dice.item()
        n_batches += 1

    mean_total = total_loss_sum / max(n_batches, 1)
    mean_bce = bce_sum / max(n_batches, 1)
    mean_dice = dice_sum / max(n_batches, 1)

    print(
        f"  Epoch {epoch:03d} | loss={mean_total:.4f}  bce={mean_bce:.4f}  "
        f"dice={mean_dice:.4f}"
    )

    return {"loss": mean_total, "bce": mean_bce, "dice": mean_dice}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    all_probs = []
    all_labels = []
    iou_list = []

    with torch.no_grad():
        for batch in dataloader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            noise = batch["noise"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            gt_mask = batch["mask"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type):
                out = model(rgb, noise)

            prob = torch.sigmoid(out["logit"].squeeze(1))
            all_probs.append(prob.cpu().float())
            all_labels.append(label.cpu().float())

            pred_bin = (out["mask"] >= 0.5).float()
            gt_bin = (gt_mask >= 0.5).float()

            for i in range(pred_bin.size(0)):
                p = pred_bin[i].reshape(-1)
                g = gt_bin[i].reshape(-1)
                intersection = (p * g).sum().item()
                union = (p + g).clamp(max=1).sum().item()
                iou = (intersection / union) if union > 0 else 1.0
                iou_list.append(iou)

    all_probs = torch.cat(all_probs).float().numpy()
    all_labels = torch.cat(all_labels).float().numpy().astype(int)

    try:
        auc = float(roc_auc_score(all_labels, all_probs))
    except ValueError:
        auc = 0.5

    miou = float(np.mean(iou_list)) if iou_list else 0.0

    print(f"  Eval  | AUC-ROC={auc:.4f}  mIoU={miou:.4f}")
    return {"auc_roc": auc, "miou": miou}


# ---------------------------------------------------------------------------
# __main__ - Kaggle execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))

    from model import PRISMNetwork
    from dataset import CelebDFDataset
    from preprocessor import PRISMPreprocessor

    # Kaggle paths
    # Dynamically find the exact dataset path in Kaggle
    import glob

    possible_dirs = glob.glob("/kaggle/input/**/Celeb-real", recursive=True)
    if possible_dirs:
        DATASET_ROOT = os.path.dirname(possible_dirs[0])
        print(f"✅ Dynamically resolved Celeb-DF dataset at: {DATASET_ROOT}")
    else:
        DATASET_ROOT = "/kaggle/input/celeb-df-v2"

    OUTPUT_DIR = "/kaggle/working"
    BEST_PTH = os.path.join(OUTPUT_DIR, "prism_model_best.pth")
    FINAL_PTH = os.path.join(OUTPUT_DIR, "prism_model_final.pth")
    EPOCHS = 20
    GRAD_ACCUM_STEPS = 4

    # Check device and GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count() if device.type == "cuda" else 0
    print(f"Device: {device} | GPUs available: {n_gpu}")

    # Fallback to a dummy dataset if running locally without the Celeb-DF folder
    if not os.path.exists(DATASET_ROOT):
        print(f"WARN: {DATASET_ROOT} not found. Ensure you are running on Kaggle.")
        # For the sake of syntax/smoke testing without the 100GB dataset:
        import tempfile, cv2

        tmp_root = tempfile.mkdtemp()
        for folder in ["Celeb-real", "YouTube-real", "Celeb-synthesis"]:
            os.makedirs(os.path.join(tmp_root, folder), exist_ok=True)
            # Make a dummy video
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                os.path.join(tmp_root, folder, "dummy.mp4"), fourcc, 10.0, (224, 224)
            )
            out.write(np.zeros((224, 224, 3), dtype=np.uint8))
            out.release()
        DATASET_ROOT = tmp_root

    # Setup Preprocessor & Dataset
    prep = PRISMPreprocessor()
    dataset = CelebDFDataset(DATASET_ROOT, prep, split="train")

    # Kaggle DataLoader specs
    loader = DataLoader(
        dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True
    )

    # Model (avoid download in basic test if running outside Kaggle)
    # Using pretrained=False for local testing safety, but normally True on Kaggle.
    # The prompt doesn't explicitly mandate False, but let's assume we want actual weights if possible.
    # Actually, the user asked to "Rewrite src/train.py to train the PRISM model on Kaggle's dual T4 GPUs."
    # We should probably use pretrained=True since it's a real training script.
    model = PRISMNetwork(pretrained=True).to(device)

    # Dual GPU support
    if n_gpu > 1:
        print("Wrapping model in DataParallel for dual GPUs.")
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-4,
        weight_decay=1e-2,
    )

    # Add Cosine Annealing Learning Rate Scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    best_auc = -1.0

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(
        f"Starting Kaggle {EPOCHS}-epoch training loop with advanced learning rate scheduling..."
    )
    for epoch in range(1, EPOCHS + 1):
        # 1. Train
        train_metrics = train_one_epoch(
            model, loader, optimizer, scaler, device, epoch=epoch
        )

        # Step the scheduler
        scheduler.step()

        # 2. Evaluate (in-sample for this concise pipeline)
        eval_metrics = evaluate(model, loader, device)
        auc = eval_metrics["auc_roc"]

        # 3. Checkpoint tracking
        if auc > best_auc:
            best_auc = auc
            print(f"  -> New best AUC ({best_auc:.4f})! Saving to {BEST_PTH}")
            # Unpack DataParallel if needed before saving
            state_dict = (
                model.module.state_dict()
                if isinstance(model, nn.DataParallel)
                else model.state_dict()
            )
            torch.save(state_dict, BEST_PTH)

    # Final save
    print(f"Training complete. Saving final model to {FINAL_PTH}")
    state_dict = (
        model.module.state_dict()
        if isinstance(model, nn.DataParallel)
        else model.state_dict()
    )
    torch.save(state_dict, FINAL_PTH)

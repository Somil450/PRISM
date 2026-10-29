"""
scripts/train_kaggle_vit_dora.py
PRISM Spatial ViT-B/16 (DoRA) Official Kaggle Retraining Script.

Enforces:
  1. Strict exclusion of official Celeb-DF v2 test list (List_of_testing_videos.txt).
  2. Identity-disjoint Train / Validation partitioning (zero identity overlap).
  3. Explicit logging and export of train_ids, val_ids, test_ids, and split manifests.
  4. Best checkpoint tracking on held-out validation AUC (prism_model_best.pth).
"""

import os
import sys
import time
import glob
import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler
from sklearn.metrics import roc_auc_score

# Ensure src is on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from model import PRISMNetwork
from preprocessor import PRISMPreprocessor
from dataset import TARGET_SIZE


def parse_celeb_df_test_list(test_list_path: str) -> Tuple[Set[str], Set[str]]:
    """
    Parses official Celeb-DF test list.
    Returns: (test_filenames, test_identities)
    """
    test_filenames = set()
    test_identities = set()

    if not os.path.isfile(test_list_path):
        return test_filenames, test_identities

    with open(test_list_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            rel_p = parts[1] if len(parts) >= 2 else parts[0]
            fname = os.path.basename(rel_p)
            test_filenames.add(fname)
            stem = os.path.splitext(fname)[0]
            for sp in stem.split("_"):
                if sp.startswith("id") and len(sp) > 2 and sp[2:].isdigit():
                    test_identities.add(sp)

    return test_filenames, test_identities


def build_identity_disjoint_splits(
    dataset_root: str,
    test_list_path: Optional[str] = None,
    val_ratio: float = 0.20,
    seed: int = 42,
    min_train_identities: int = 30,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Scans Celeb-DF, excludes official test set and test identities, and partitions
    remaining identities into Train and Validation splits with zero identity leakage.
    Strictly tracks BOTH target and source identities in synthesis videos.
    """
    tl_path = test_list_path or os.path.join(dataset_root, "List_of_testing_videos.txt")
    test_filenames, test_identities = parse_celeb_df_test_list(tl_path)
    print(f"[+] Loaded official test list: {len(test_filenames)} test videos across {len(test_identities)} test identities.")

    categories = {
        "Celeb-real": 0.0,
        "YouTube-real": 0.0,
        "Celeb-synthesis": 1.0,
    }

    test_records = []
    pool_records = []

    for folder, label in categories.items():
        folder_path = os.path.join(dataset_root, folder)
        if not os.path.isdir(folder_path):
            continue

        for fname in sorted(os.listdir(folder_path)):
            if not fname.endswith(".mp4"):
                continue

            stem = os.path.splitext(fname)[0]
            entry_ids = set()
            for sp in stem.split("_"):
                if sp.startswith("id") and len(sp) > 2 and sp[2:].isdigit():
                    entry_ids.add(sp)

            primary_id = sorted(list(entry_ids))[0] if entry_ids else f"yt_{stem}"
            # Dual identity check: if ANY identity in the video is in test_identities, quarantine to test
            is_test = (fname in test_filenames) or bool(entry_ids.intersection(test_identities))

            record = {
                "path": os.path.join(folder_path, fname),
                "filename": fname,
                "label": label,
                "identity": primary_id,
                "all_ids": list(entry_ids),
                "category": folder,
            }

            if is_test:
                test_records.append(record)
            else:
                pool_records.append(record)

    # Collect all unique celebrity identities in the pool
    pool_celeb_ids = set()
    for r in pool_records:
        for ident in r["all_ids"]:
            pool_celeb_ids.add(ident)
    pool_identities = sorted(list(pool_celeb_ids))

    if not pool_identities:
        # Fallback for youtube or custom sets without idX tags
        pool_identities = sorted(list(set(r["identity"] for r in pool_records)))

    rng = np.random.RandomState(seed)
    rng.shuffle(pool_identities)

    n_val_ids = max(1, int(round(len(pool_identities) * val_ratio)))
    val_identities = set(pool_identities[:n_val_ids])
    train_identities = set(pool_identities[n_val_ids:])

    # Check minimum train identity threshold per Prompt 13 Rule 7
    if len(train_identities) < min_train_identities:
        raise ValueError(
            f"Identity-disjoint train set has only {len(train_identities)} identities, which is under "
            f"minimum threshold ({min_train_identities}). Stopping execution. Proposed fix: Use an explicit "
            f"38-Train / 10-Val / 14-Test identity partition instead of the default testing list."
        )

    # Filter records ensuring BOTH identities in fakes belong strictly to the respective partition
    train_records = []
    val_records = []
    quarantined_cross_split = 0

    for r in pool_records:
        e_ids = set(r["all_ids"])
        if not e_ids:
            # YouTube-real or untagged
            if r["identity"] in train_identities:
                train_records.append(r)
            elif r["identity"] in val_identities:
                val_records.append(r)
        else:
            if e_ids.issubset(train_identities):
                train_records.append(r)
            elif e_ids.issubset(val_identities):
                val_records.append(r)
            else:
                # Video spans across the train/val boundary (e.g. target in train, source in val)
                # Discard to guarantee 100% strict identity disjointness
                quarantined_cross_split += 1

    # Verify no split is empty
    if len(train_records) == 0 or len(val_records) == 0 or len(test_records) == 0:
        raise ValueError(
            f"One or more splits is empty! Train={len(train_records)} vids, "
            f"Val={len(val_records)} vids, Test={len(test_records)} vids."
        )

    # Verify Strict Disjointness
    train_vids = set(r["filename"] for r in train_records)
    val_vids = set(r["filename"] for r in val_records)
    test_vids = set(r["filename"] for r in test_records)

    assert len(train_vids.intersection(val_vids)) == 0, "Train-Val video leakage!"
    assert len(train_vids.intersection(test_vids)) == 0, "Train-Test video leakage!"
    assert len(val_vids.intersection(test_vids)) == 0, "Val-Test video leakage!"

    assert len(train_identities.intersection(val_identities)) == 0, "Train-Val identity leakage!"
    assert len(train_identities.intersection(test_identities)) == 0, "Train-Test identity leakage!"
    assert len(val_identities.intersection(test_identities)) == 0, "Val-Test identity leakage!"

    train_real_count = sum(1 for r in train_records if r["label"] == 0.0)
    train_fake_count = sum(1 for r in train_records if r["label"] == 1.0)
    val_real_count = sum(1 for r in val_records if r["label"] == 0.0)
    val_fake_count = sum(1 for r in val_records if r["label"] == 1.0)
    test_real_count = sum(1 for r in test_records if r["label"] == 0.0)
    test_fake_count = sum(1 for r in test_records if r["label"] == 1.0)

    protocol_label = "custom identity-disjoint split, not the official protocol"

    summary = {
        "protocol_label": protocol_label,
        "dataset_root": dataset_root,
        "seed": seed,
        "train_video_count": len(train_records),
        "train_real_count": train_real_count,
        "train_fake_count": train_fake_count,
        "val_video_count": len(val_records),
        "val_real_count": val_real_count,
        "val_fake_count": val_fake_count,
        "test_video_count": len(test_records),
        "test_real_count": test_real_count,
        "test_fake_count": test_fake_count,
        "quarantined_cross_split_videos": quarantined_cross_split,
        "train_identity_count": len(train_identities),
        "val_identity_count": len(val_identities),
        "test_identity_count": len(test_identities),
        "total_identities": len(train_identities) + len(val_identities) + len(test_identities),
        "train_ids": sorted(list(train_identities)),
        "val_ids": sorted(list(val_identities)),
        "test_ids": sorted(list(test_identities)),
        "disjoint_verified": True,
    }

    print("\n" + "=" * 95)
    print(f"🏷️  PROTOCOL: '{protocol_label}'")
    print("=" * 95)
    print(f"Total Unique Identities Tracked : {summary['total_identities']}")
    print(f"Train Split Partition           : {len(train_records)} vids (Real={train_real_count}, Fake={train_fake_count}) across {len(train_identities)} IDs")
    print(f"Validation Split Partition      : {len(val_records)} vids (Real={val_real_count}, Fake={val_fake_count}) across {len(val_identities)} IDs")
    print(f"Quarantined Cross-Split Fakes   : {quarantined_cross_split} videos (quarantined to prevent identity leakage)")
    print(f"Official Held-Out Test Set      : {len(test_records)} vids (Real={test_real_count}, Fake={test_fake_count}) across {len(test_identities)} IDs")
    print("=" * 95 + "\n")

    return train_records, val_records, test_records, summary


class CelebDFSplitDataset(Dataset):
    """
    Fast PyTorch Dataset consuming pre-partitioned record lists.
    """

    def __init__(self, records: List[Dict[str, Any]], preprocessor: PRISMPreprocessor) -> None:
        self.records = records
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        p = rec["path"]
        label = float(rec["label"])

        cap = cv2.VideoCapture(p)
        frame = None
        if cap.isOpened():
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total > 0:
                target_idx = np.random.randint(0, total)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, f = cap.read()
            if ret and f is not None:
                frame = f
            cap.release()

        if frame is None:
            frame = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)

        rgb_t, noise_t = self.preprocessor.process_frame(frame)

        # Synthetic weakly-supervised mask for fakes
        if label > 0.5:
            mask = np.zeros((1, TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
            mask[:, 56:168, 56:168] = 1.0
            mask_t = torch.from_numpy(mask)
        else:
            mask_t = torch.zeros((1, TARGET_SIZE, TARGET_SIZE), dtype=torch.float32)

        return {
            "rgb": rgb_t,
            "noise": noise_t,
            "label": torch.tensor(label, dtype=torch.float32),
            "mask": mask_t,
        }


def soft_dice_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    pred_flat = pred.reshape(-1)
    gt_flat = gt.reshape(-1)
    intersection = (pred_flat * gt_flat).sum()
    return 1.0 - (2.0 * intersection + eps) / (pred_flat.sum() + gt_flat.sum() + eps)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    bce_fn: nn.Module,
    epoch: int = 1,
) -> Dict[str, float]:
    model.train()
    total_loss, bce_loss_sum, dice_loss_sum = 0.0, 0.0, 0.0
    n_batches = 0

    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)
        rgb = batch["rgb"].to(device, non_blocking=True)
        noise = batch["noise"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        gt_mask = batch["mask"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type):
            out = model(rgb, noise)
            l_bce = bce_fn(out["logit"].squeeze(1), label)
            l_dice = soft_dice_loss(out["mask"], gt_mask)
            loss = l_bce + 2.0 * l_dice

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        bce_loss_sum += l_bce.item()
        dice_loss_sum += l_dice.item()
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "bce": bce_loss_sum / max(n_batches, 1),
        "dice": dice_loss_sum / max(n_batches, 1),
    }


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
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
                inter = (p * g).sum().item()
                union = (p + g).clamp(max=1).sum().item()
                iou = (inter / union) if union > 0 else 1.0
                iou_list.append(iou)

    if not all_probs:
        return {"auc_roc": 0.5, "miou": 0.0}

    all_probs = torch.cat(all_probs).float().numpy()
    all_labels = torch.cat(all_labels).float().numpy().astype(int)

    if len(np.unique(all_labels)) > 1:
        auc = float(roc_auc_score(all_labels, all_probs))
    else:
        auc = 0.5

    miou = float(np.mean(iou_list)) if iou_list else 0.0
    return {"auc_roc": auc, "miou": miou}


def run_training_pipeline(
    dataset_root: str,
    output_dir: str = "/kaggle/working",
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 2e-4,
    device_str: Optional[str] = None,
    mock_test: bool = False,
) -> Dict[str, Any]:
    """
    Executes identity-disjoint ViT-DoRA retraining.
    """
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    n_gpu = torch.cuda.device_count() if device.type == "cuda" else 0
    print(f"[+] Execution Device: {device} (Available GPUs: {n_gpu})")

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    splits_dir = out_p / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    if mock_test or not os.path.exists(dataset_root):
        print("[!] Running in mock test mode: synthesizing miniature identity-labeled test dataset...")
        import tempfile
        tmp = tempfile.mkdtemp()
        for fol in ["Celeb-real", "YouTube-real", "Celeb-synthesis"]:
            os.makedirs(os.path.join(tmp, fol), exist_ok=True)
            for i in range(4):
                ident = f"id{i}"
                vpath = os.path.join(tmp, fol, f"{ident}_{i:03d}.mp4")
                writer = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (128, 128))
                for _ in range(5):
                    writer.write(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
                writer.release()
        tl_path = os.path.join(tmp, "List_of_testing_videos.txt")
        with open(tl_path, "w") as f:
            f.write("0 Celeb-real/id3_003.mp4\n1 Celeb-synthesis/id3_003.mp4\n")
        dataset_root = tmp

    # Build identity-disjoint splits (minimum 30 train identities enforced in production)
    train_recs, val_recs, test_recs, summary = build_identity_disjoint_splits(
        dataset_root, min_train_identities=2 if mock_test else 30
    )

    # Save split manifests
    with open(splits_dir / "celeb_df_train_split.json", "w", encoding="utf-8") as f:
        json.dump(train_recs, f, indent=2)
    with open(splits_dir / "celeb_df_val_split.json", "w", encoding="utf-8") as f:
        json.dump(val_recs, f, indent=2)
    with open(splits_dir / "celeb_df_test_split.json", "w", encoding="utf-8") as f:
        json.dump(test_recs, f, indent=2)
    with open(splits_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[+] Split manifests exported to: {splits_dir.as_posix()}")

    # Setup Preprocessor & DataLoaders
    prep = PRISMPreprocessor()
    train_ds = CelebDFSplitDataset(train_recs, prep)
    val_ds = CelebDFSplitDataset(val_recs, prep)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2 if sys.platform != "win32" else 0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2 if sys.platform != "win32" else 0)

    # Initialize ViT-DoRA
    model = PRISMNetwork(pretrained=False).to(device)
    if n_gpu > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))
    bce_fn = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_pth = str(out_p / "prism_model_best.pth")
    final_pth = str(out_p / "prism_model_final.pth")

    print(f"[+] Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, bce_fn, epoch=epoch)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)
        dt = time.time() - t0

        v_auc = val_metrics["auc_roc"]
        v_miou = val_metrics["miou"]
        print(f"Epoch {epoch:02d}/{epochs:02d} ({dt:.1f}s) | Train Loss: {tr_metrics['loss']:.4f} | "
              f"Val AUC: {v_auc:.4f} | Val mIoU: {v_miou:.4f}")

        if v_auc > best_auc:
            best_auc = v_auc
            print(f"  [*] New Best Validation AUC: {best_auc:.4f}! Checkpoint saved to {best_pth}")
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state, best_pth)

    # Save final model
    state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state, final_pth)
    print(f"[+] Retraining Complete. Final model saved to: {final_pth}")

    return {
        "best_val_auc": best_auc,
        "best_checkpoint": best_pth,
        "final_checkpoint": final_pth,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description="PRISM Kaggle ViT-DoRA Disjoint Retraining")
    parser.add_argument("--dataset_root", type=str, default="/kaggle/input/celeb-df-v2")
    default_out = "/kaggle/working" if os.path.exists("/kaggle") else str(REPO_ROOT / "data" / "kaggle_working")
    parser.add_argument("--output_dir", type=str, default=default_out)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--mock_test", action="store_true", help="Run rapid smoke test with synthetic miniature fixtures")
    args = parser.parse_args()

    run_training_pipeline(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        mock_test=args.mock_test,
    )


if __name__ == "__main__":
    main()

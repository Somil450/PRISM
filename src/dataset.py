"""
src/dataset.py
PRISM PyTorch Dataset - Celeb-DF-v2 ingestion for Kaggle training.
"""

import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Callable, Optional

from preprocessor import PRISMPreprocessor

TARGET_SIZE = 224  # keep in sync with preprocessor


class CelebDFDataset(Dataset):
    """
    Ingests the Celeb-DF-v2 dataset.
    Crawls:
        Celeb-real/ (Label 0)
        YouTube-real/ (Label 0)
        Celeb-synthesis/ (Label 1)
        
    Samples 1 random frame per video during __getitem__ to save VRAM/CPU.
    Generates synthetic weakly-supervised masks for fakes.
    """

    def __init__(
        self,
        root_dir: str,
        preprocessor: PRISMPreprocessor,
        split: str = 'train',
        max_frames_per_video: int = 4,
        test_list_path: Optional[str] = None,
    ) -> None:
        """
        Args:
            root_dir: Path to Celeb-DF-v2 root (e.g., /kaggle/input/celeb-df-v2/).
            preprocessor: Instance of PRISMPreprocessor.
            split: 'train', 'test', or 'all'.
            max_frames_per_video: Provided for API compatibility.
            test_list_path: Path to List_of_testing_videos.txt.
        """
        self.root_dir = root_dir
        self.preprocessor = preprocessor
        self.split = split.lower()
        self.max_frames_per_video = max_frames_per_video

        # Discover official test list
        tl_file = test_list_path or os.path.join(root_dir, "List_of_testing_videos.txt")
        test_filenames = set()
        test_identities = set()

        if os.path.isfile(tl_file):
            with open(tl_file, "r", encoding="utf-8", errors="replace") as f:
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

        self.videos = []

        categories = {
            "Celeb-real": 0.0,
            "YouTube-real": 0.0,
            "Celeb-synthesis": 1.0
        }

        for folder, label in categories.items():
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            for fname in os.listdir(folder_path):
                if not fname.endswith(".mp4"):
                    continue

                stem = os.path.splitext(fname)[0]
                entry_ids = set()
                for sp in stem.split("_"):
                    if sp.startswith("id") and len(sp) > 2 and sp[2:].isdigit():
                        entry_ids.add(sp)

                is_test = fname in test_filenames
                has_test_id = bool(entry_ids.intersection(test_identities))

                if self.split == "test":
                    if is_test:
                        self.videos.append({
                            "path": os.path.join(folder_path, fname),
                            "label": label
                        })
                elif self.split == "train" and test_filenames:
                    # Strict Zero-Identity Leakage: exclude test videos AND test identities
                    if not is_test and not has_test_id:
                        self.videos.append({
                            "path": os.path.join(folder_path, fname),
                            "label": label
                        })
                else:
                    self.videos.append({
                        "path": os.path.join(folder_path, fname),
                        "label": label
                    })

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, idx: int) -> dict:
        video_info = self.videos[idx]
        v_path = video_info["path"]
        label = video_info["label"]

        cap = cv2.VideoCapture(v_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {v_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1

        # Randomly sample exactly 1 frame
        random_frame_idx = random.randint(0, max(0, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_idx)

        ret, bgr_frame = cap.read()
        if not ret:
            # Fallback to first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, bgr_frame = cap.read()
            if not ret:
                cap.release()
                raise IOError(f"Could not read any frame from {v_path}")

        cap.release()

        # Process the frame using the preprocessor
        rgb_tensor, noise_tensor = self.preprocessor.process_frame(bgr_frame)

        # Generate weakly-supervised mask [1, 224, 224]
        mask_tensor = torch.zeros(1, TARGET_SIZE, TARGET_SIZE, dtype=torch.float32)
        if label == 1.0:
            # Fake: central rectangular mask spanning middle 60%
            start = int(TARGET_SIZE * 0.2)
            end = int(TARGET_SIZE * 0.8)
            mask_tensor[0, start:end, start:end] = 1.0

        return {
            "rgb": rgb_tensor,
            "noise": noise_tensor,
            "label": torch.tensor(label, dtype=torch.float32),
            "mask": mask_tensor
        }


# ---------------------------------------------------------------------------
# Quick shape-verification smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    # Build tiny fake Celeb-DF structure
    with tempfile.TemporaryDirectory() as tmpdir:
        for folder in ["Celeb-real", "YouTube-real", "Celeb-synthesis"]:
            os.makedirs(os.path.join(tmpdir, folder))

        # Create dummy mp4 files (we'll just use cv2.VideoWriter)
        def create_dummy_video(path, frames=5):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(path, fourcc, 10.0, (640, 480))
            for _ in range(frames):
                frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
                out.write(frame)
            out.release()

        create_dummy_video(os.path.join(tmpdir, "Celeb-real", "real1.mp4"))
        create_dummy_video(os.path.join(tmpdir, "YouTube-real", "real2.mp4"))
        create_dummy_video(os.path.join(tmpdir, "Celeb-synthesis", "fake1.mp4"))

        prep = PRISMPreprocessor()
        ds = CelebDFDataset(tmpdir, prep)
        print(f"Dataset length: {len(ds)}")

        for i in range(len(ds)):
            item = ds[i]
            rgb   = item["rgb"]
            noise = item["noise"]
            label = item["label"]
            mask  = item["mask"]

            print(
                f"  [{i}] rgb={tuple(rgb.shape)} noise={tuple(noise.shape)} "
                f"label={label.item():.0f} mask={tuple(mask.shape)}"
            )

            assert rgb.shape   == (3, 224, 224), f"rgb shape wrong: {rgb.shape}"
            assert noise.shape == (1, 224, 224), f"noise shape wrong: {noise.shape}"
            assert mask.shape  == (1, 224, 224), f"mask shape wrong: {mask.shape}"
            assert rgb.dtype   == torch.float32
            assert noise.dtype == torch.float32
            assert label.dtype == torch.float32
            assert mask.dtype  == torch.float32
            
            if label.item() == 1.0:
                assert mask.sum() > 0, "Fake mask should not be all zeros"
            else:
                assert mask.sum() == 0, "Real mask should be all zeros"

        print("All dataset shape assertions passed.")

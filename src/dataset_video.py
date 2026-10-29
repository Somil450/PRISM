"""
src/dataset_video.py
PyTorch Dataset for Temporal Video Deepfake Training.
Extracts contiguous frame sequences [C, T, H, W] from videos.
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class CelebDFVideoDataset(Dataset):
    def __init__(self, root_dir: str, num_frames: int = 16, target_size: int = 224) -> None:
        """
        Args:
            root_dir: Path to Celeb-DF-v2 root.
            num_frames: Number of contiguous frames to extract per video.
            target_size: Height and Width to resize frames.
        """
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.target_size = target_size
        self.videos = []

        categories = {"Celeb-real": 0.0, "YouTube-real": 0.0, "Celeb-synthesis": 1.0}

        for folder, label in categories.items():
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            for fname in os.listdir(folder_path):
                if fname.endswith(".mp4"):
                    self.videos.append({"path": os.path.join(folder_path, fname), "label": label})

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, idx: int) -> dict:
        v_path = self.videos[idx]["path"]
        label = self.videos[idx]["label"]

        cap = cv2.VideoCapture(v_path)
        if not cap.isOpened():
            return self._get_empty_tensor(label)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= self.num_frames:
            start_frame = 0
        else:
            # Pick a random starting frame ensuring we can get a full contiguous chunk
            start_frame = np.random.randint(0, total_frames - self.num_frames)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        for _ in range(self.num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize
            frame = cv2.resize(frame, (self.target_size, self.target_size))
            frames.append(frame)
            
        cap.release()

        # Pad if video was too short
        while len(frames) < self.num_frames:
            frames.append(np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8))

        # Shape: [T, H, W, C] -> Normalize to [0,1]
        frames_np = np.stack(frames).astype(np.float32) / 255.0
        
        # PyTorch Video models expect [C, T, H, W]
        # Transpose: from [T, H, W, C] to [C, T, H, W]
        frames_t = torch.from_numpy(frames_np).permute(3, 0, 1, 2)

        return {
            "video": frames_t,
            "label": torch.tensor(label, dtype=torch.float32)
        }

    def _get_empty_tensor(self, label: float) -> dict:
        # Fallback for corrupted videos
        return {
            "video": torch.zeros((3, self.num_frames, self.target_size, self.target_size)),
            "label": torch.tensor(label, dtype=torch.float32)
        }

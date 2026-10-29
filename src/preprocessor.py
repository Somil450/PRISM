"""
src/preprocessor.py
PRISM Preprocessing - CPU-bound. Keeps VRAM free for model inference.
Outputs float32 tensors ready for DataLoader batching.
"""

import subprocess
from typing import Optional, Tuple, List
import cv2
import numpy as np
import torch

# ImageNet stats for RGB normalization
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

TARGET_SIZE = 224  # fixed spatial resolution


class PRISMPreprocessor:
    """
    Stateless preprocessing helper. All ops run on CPU.
    No model weights - instantiate freely, no cleanup needed.
    """

    def extract_sensor_noise(self, bgr_img: np.ndarray) -> torch.Tensor:
        """
        Extracts high-frequency PRNU camera noise via Laplacian residual.

        Args:
            bgr_img: HxWx3 uint8 BGR array (OpenCV native format).

        Returns:
            float32 tensor [1, 224, 224] in [0, 1].
        """
        # grayscale → float32 for Laplacian
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Laplacian highlights high-freq noise (PRNU signal)
        noise = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)

        # abs so both positive/negative residuals register
        noise = np.abs(noise)

        # normalize to [0, 1]; guard div-by-zero on flat patches
        n_min, n_max = noise.min(), noise.max()
        if n_max > n_min:
            noise = (noise - n_min) / (n_max - n_min)
        else:
            noise = np.zeros_like(noise)

        # resize to fixed spatial resolution
        noise = cv2.resize(noise, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)

        # [H, W] → [1, H, W] tensor
        return torch.from_numpy(noise).unsqueeze(0)  # float32 by default

    def purify_frame(self, bgr_frame: np.ndarray, quality: int = 65) -> np.ndarray:
        """
        Adversarial Purification via fast DCT thresholding (JPEG quantization).
        Destroys high-frequency adversarial gradients without losing semantic meaning.
        """
        _, encoded = cv2.imencode('.jpg', bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    def process_frame(self, bgr_frame: np.ndarray) -> tuple:
        """
        Center-crops to square, resizes to 224x224, produces both streams.
        Applies Modality-Aware Adversarial Purification.

        Args:
            bgr_frame: HxWx3 uint8 BGR array.

        Returns:
            (rgb_tensor [3, 224, 224], noise_tensor [1, 224, 224])
        """
        h, w = bgr_frame.shape[:2]

        # center crop to square - no distortion on non-square inputs
        crop_size = min(h, w)
        top  = (h - crop_size) // 2
        left = (w - crop_size) // 2
        cropped_raw = bgr_frame[top:top + crop_size, left:left + crop_size]
        
        # --- Modality-Aware Adversarial Purification ---
        # The raw crop goes to the noise stream (so PRNU/ELA artifacts aren't destroyed).
        # The purified crop goes to the semantic stream (ViT/Gaze) to block adversarial noise.
        cropped_purified = self.purify_frame(cropped_raw)

        # resize once, reuse for both streams
        resized = cv2.resize(cropped_purified, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)

        # --- RGB stream (Purified) ---
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # [3,H,W]
        mean = torch.tensor(_IMAGENET_MEAN, dtype=rgb_tensor.dtype).view(-1, 1, 1)
        std = torch.tensor(_IMAGENET_STD, dtype=rgb_tensor.dtype).view(-1, 1, 1)
        rgb_tensor = (rgb_tensor - mean) / std

        # --- Noise stream (Raw) ---
        noise_tensor = self.extract_sensor_noise(cropped_raw)

        return rgb_tensor, noise_tensor

    def extract_video_frames(
        self,
        video_path: str,
        max_frames: int = 16,
    ) -> list:
        """
        Uniformly samples up to `max_frames` frames from a video.

        Args:
            video_path: Path to any OpenCV-readable video file.
            max_frames:  Maximum frames to sample.

        Returns:
            List of (rgb_tensor, noise_tensor) tuples, one per sampled frame.
            Returns empty list on unreadable video.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []

        # uniform indices across full duration
        n = min(max_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, n, dtype=int)

        results = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            results.append(self.process_frame(frame))

        cap.release()
        return results

    def extract_raw_video_frames(
        self,
        video_path: str,
        max_frames: int = 16,
    ) -> list:
        """
        Extracts up to `max_frames` contiguous raw BGR frames from a video file.
        Using consecutive adjacent frames is required for differential optical flow.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []

        n = min(max_frames, total_frames)
        # Sample consecutive burst starting around video midpoint
        start_idx = max(0, (total_frames - n) // 2)

        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        for _ in range(n):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        return frames

    def get_video_fps(self, video_path: str) -> float:
        """
        Retrieves actual frame rate of a video file.
        Defaults to 30.0 if unreadable or non-positive.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 30.0
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            return fps if (fps > 0.0 and not np.isnan(fps) and not np.isinf(fps)) else 30.0
        except Exception:
            return 30.0

    def extract_raw_video_frames_with_meta(
        self,
        video_path: str,
        max_frames: int = 32,
    ) -> dict:
        """
        Extracts up to `max_frames` contiguous raw BGR frames and precise temporal metadata.
        Returns dict with keys: 'frames', 'fps', 'start_frame', 'start_sec', 'total_frames'.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"frames": [], "fps": 30.0, "start_frame": 0, "start_sec": 0.0, "total_frames": 0}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        raw_fps = float(cap.get(cv2.CAP_PROP_FPS))
        fps = raw_fps if (raw_fps > 0.0 and not np.isnan(raw_fps) and not np.isinf(raw_fps)) else 30.0

        if total_frames <= 0:
            cap.release()
            return {"frames": [], "fps": fps, "start_frame": 0, "start_sec": 0.0, "total_frames": 0}

        n = min(max_frames, total_frames)
        start_idx = max(0, (total_frames - n) // 2)

        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        for _ in range(n):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        start_sec = float(start_idx) / fps if fps > 0 else 0.0

        return {
            "frames": frames,
            "fps": fps,
            "start_frame": start_idx,
            "start_sec": start_sec,
            "total_frames": total_frames,
        }

    def extract_audio_track(
        self,
        video_path: str,
        sample_rate: int = 16000,
        max_duration_sec: float = 30.0,
    ) -> Optional[Tuple[np.ndarray, int]]:
        """
        Extracts mono audio waveform from a video container using ffmpeg streaming pipe.
        Returns (waveform_float32, sample_rate) normalized in [-1.0, 1.0], or None if silent/no audio.
        """
        try:
            cmd = [
                "ffmpeg",
                "-v", "error",
                "-i", video_path,
                "-t", str(max_duration_sec),
                "-vn",
                "-f", "s16le",
                "-acodec", "pcm_s16le",
                "-ar", str(sample_rate),
                "-ac", "1",
                "-",
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
            if proc.returncode != 0 or len(proc.stdout) < sample_rate * 2 * 0.2:  # at least 0.2 sec
                return None

            raw_pcm = np.frombuffer(proc.stdout, dtype=np.int16)
            if raw_pcm.size == 0:
                return None

            waveform = raw_pcm.astype(np.float32) / 32768.0
            max_amp = float(np.max(np.abs(waveform)))
            if max_amp > 1e-5:
                waveform = waveform / max_amp

            return waveform, sample_rate
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Quick shape-verification smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    prep = PRISMPreprocessor()

    # dummy BGR frame - random 480x640 like a typical webcam feed
    dummy_bgr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    rgb_t, noise_t = prep.process_frame(dummy_bgr)
    print(f"process_frame -> rgb:   {tuple(rgb_t.shape)}  dtype={rgb_t.dtype}")
    print(f"process_frame -> noise: {tuple(noise_t.shape)} dtype={noise_t.dtype}")
    assert rgb_t.shape   == (3, 224, 224), f"Bad rgb shape: {rgb_t.shape}"
    assert noise_t.shape == (1, 224, 224), f"Bad noise shape: {noise_t.shape}"
    assert rgb_t.dtype   == torch.float32
    assert noise_t.dtype == torch.float32
    assert 0.0 <= float(noise_t.min()) and float(noise_t.max()) <= 1.0, "Noise out of [0,1]"

    noise_only = prep.extract_sensor_noise(dummy_bgr)
    print(f"extract_sensor_noise -> {tuple(noise_only.shape)} dtype={noise_only.dtype}")
    assert noise_only.shape == (1, 224, 224)

    print("All preprocessor shape assertions passed.")

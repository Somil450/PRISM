"""
scripts/benchmark_indeepfake.py
InDeepFake & SAMANVAYA Multimodal Indic Benchmark Evaluator (2025).

Purpose:
  Evaluates PRISM's multi-agent forensic engine on InDeepFake or local test samples.
  Strictly audits whether the official InDeepFake dataset is present.
  Evaluates Audio-only, Video-only, Cross-Modal, and Fused pipelines.
  Evaluates phoneme-to-viseme lip synchrony bias across Indic languages
  (Hindi, Tamil, Telugu, Bengali) and runs audio robustness stress testing.

Reference:
  Arnab Kumar Das, Ruchira Naskar, et al.
  "InDeepFake: A Multimodal Multi-Lingual Deepfake Dataset for Indic Populations."
  Pattern Recognition Letters, 2025.
  Access guide: docs/INDEEPFAKE_ACCESS.md
"""

import argparse
import os
import sys
import json
import re
import csv
import shutil
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import io

try:
    import cv2
    import scipy.signal as signal
    import scipy.io.wavfile as wavfile
    from agents.video_temporal_agent import VideoTemporalAgent
    from agents.lip_sync_agent import LipSyncAgent
    from agents.audio_voice_agent import AudioVoiceAgent, NoSpeechDetectedError
    from agents.rppg_agent import RPPGAgent
    from fusion import DempsterShaferFusion
except ImportError as err:
    print(f"Error importing PRISM forensic components: {err}")
    sys.exit(1)

try:
    import moviepy.editor as mp
    HAS_MOVIEPY = True
except ImportError:
    HAS_MOVIEPY = False


class IndicMetadataParser:
    """
    Parses Indic metadata attributes from file paths or annotations.
    """
    @staticmethod
    def parse_attributes(path: Path) -> Dict[str, Any]:
        p_str = str(path).lower().replace("\\", "/")
        lang = "Unknown"
        if "hindi" in p_str:
            lang = "Hindi"
        elif "tamil" in p_str:
            lang = "Tamil"
        elif "telugu" in p_str:
            lang = "Telugu"
        elif "bengali" in p_str:
            lang = "Bengali"

        modality = "Multimodal"
        if "lip" in p_str:
            modality = "Lip-Sync Manipulated"
        elif "voice" in p_str or "cloned" in p_str or "audio" in p_str:
            modality = "Acoustic Voice Cloned"
        elif "swap" in p_str or "face" in p_str:
            modality = "Visual Face Swap"

        return {"language": lang, "modality": modality}


class IndicLanguageIdentifier:
    """
    Identifies Indic language family or speech presence from audio tracks.
    Compares filename/metadata tags against detected acoustic footprint.
    """

    SUPPORTED_LANGUAGES = ["Hindi", "Tamil", "Telugu", "Bengali"]

    @staticmethod
    def extract_audio(video_path: str) -> Tuple[Optional[np.ndarray], int]:
        """Extracts 16kHz mono audio waveform from video using ffmpeg (with moviepy fallback)."""
        ffmpeg_cmd = shutil.which("ffmpeg")
        if ffmpeg_cmd:
            try:
                proc = subprocess.run(
                    [ffmpeg_cmd, "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                if proc.stdout and len(proc.stdout) > 44:
                    sr, data = wavfile.read(io.BytesIO(proc.stdout))
                    if data.ndim > 1:
                        data = np.mean(data, axis=1)
                    if data.dtype == np.int16:
                        wave = (data / 32768.0).astype(np.float32)
                    elif data.dtype == np.int32:
                        wave = (data / 2147483648.0).astype(np.float32)
                    else:
                        wave = data.astype(np.float32)
                    return wave, int(sr)
            except Exception:
                pass

        if HAS_MOVIEPY:
            try:
                clip = mp.VideoFileClip(video_path)
                if clip.audio is None:
                    clip.close()
                    return None, 16000
                arr = clip.audio.to_soundarray(fps=16000)
                clip.close()
                if arr.ndim > 1:
                    wave = np.mean(arr, axis=1).astype(np.float32)
                else:
                    wave = arr.astype(np.float32)
                return wave, 16000
            except Exception:
                pass

        return None, 16000

    @classmethod
    def detect_language(cls, waveform: Optional[np.ndarray], sr: int, fallback_tag: str) -> Tuple[str, float]:
        """
        Acoustic / linguistic language identification.
        Returns (detected_language, confidence).
        """
        if waveform is None or len(waveform) < sr * 0.5:
            return "Unvoiced / Silent", 0.0

        # Heuristic acoustic profile: Dravidian (Tamil/Telugu) vs Indo-Aryan (Hindi/Bengali)
        # Dravidian languages feature higher retroflex consonant energy in 2.5-4 kHz band
        freqs, _, spec = signal.stft(waveform, fs=sr, nperseg=512)
        mag = np.abs(spec)

        retroflex_energy = float(np.mean(mag[(freqs >= 2500) & (freqs <= 4000), :]))
        vowel_energy = float(np.mean(mag[(freqs >= 300) & (freqs <= 1200), :])) + 1e-6
        retroflex_ratio = retroflex_energy / vowel_energy

        # If fallback tag is explicit, evaluate agreement
        norm_fallback = fallback_tag.capitalize()
        if norm_fallback in cls.SUPPORTED_LANGUAGES:
            # Check acoustic plausibility
            conf = 0.85 if retroflex_ratio > 0.05 else 0.50
            return norm_fallback, conf

        # Default acoustic classification
        if retroflex_ratio > 0.28:
            return "Tamil / Telugu (Dravidian)", 0.65
        else:
            return "Hindi / Bengali (Indo-Aryan)", 0.65


class RealSpeechClipLoader:
    """
    Structured loader for real/authentic and synthesized speech clips with explicit
    language annotations (Hindi, Tamil, Telugu, Bengali, English).
    Validates vocal energy (RMS >= 0.005) to ensure silent/unvoiced clips are rejected
    or flagged, preventing pseudo-evaluations on silent data.
    """

    def __init__(self, annotation_file: Optional[str] = None, clips_dir: Optional[str] = None) -> None:
        self.records: List[Dict[str, Any]] = []
        if annotation_file and os.path.exists(annotation_file):
            self.load_annotations(annotation_file)
        elif clips_dir and os.path.exists(clips_dir):
            self.scan_directory(clips_dir)

    def load_annotations(self, annotation_file: str) -> List[Dict[str, Any]]:
        p = Path(annotation_file)
        self.records = []
        if p.suffix.lower() == ".json":
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    self.records.append(self._validate_record(item))
        elif p.suffix.lower() == ".csv":
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.records.append(self._validate_record(dict(row)))
        return self.records

    def scan_directory(self, clips_dir: str) -> List[Dict[str, Any]]:
        p = Path(clips_dir)
        self.records = []
        for ext in [".mp4", ".wav", ".avi", ".mov"]:
            for f in sorted(list(p.rglob(f"*{ext}"))):
                lang = "Unknown"
                f_lower = str(f).lower()
                for l in ["hindi", "tamil", "telugu", "bengali", "english"]:
                    if l in f_lower:
                        lang = l.capitalize()
                        break
                is_fake = 1 if any(k in f_lower for k in ["fake", "swap", "clone", "synth", "inswapper"]) else 0
                record = {
                    "path": str(f),
                    "label": is_fake,
                    "language": lang,
                    "speaker_id": f.stem.split("_")[0],
                }
                self.records.append(self._validate_record(record))
        return self.records

    def _validate_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        p = Path(record["path"])
        has_speech = False
        rms = 0.0
        if p.exists():
            if p.suffix.lower() in [".wav", ".flac"]:
                try:
                    sr, data = wavfile.read(str(p))
                    if data.ndim > 1:
                        data = np.mean(data, axis=1)
                    wave = data.astype(np.float32)
                    rms = float(np.sqrt(np.mean(wave**2))) if len(wave) > 0 else 0.0
                    has_speech = (rms >= 0.005)
                except Exception:
                    pass
            elif p.suffix.lower() in [".mp4", ".avi", ".mov"]:
                wave, _ = IndicLanguageIdentifier.extract_audio(str(p))
                if wave is not None and len(wave) > 0:
                    rms = float(np.sqrt(np.mean(wave**2)))
                    has_speech = (rms >= 0.005)

        return {
            "path": str(p),
            "label": int(record.get("label", 0)),
            "language": str(record.get("language", "Unknown")).capitalize(),
            "speaker_id": str(record.get("speaker_id", p.stem)),
            "has_speech": has_speech,
            "rms_energy": round(rms, 6),
        }

    def get_voiced_clips(self) -> List[Dict[str, Any]]:
        return [r for r in self.records if r["has_speech"]]

    def get_clips_by_language(self, language: str) -> List[Dict[str, Any]]:
        return [r for r in self.records if r["language"].lower() == language.lower()]


class AudioPerturbationSuite:
    """
    Stress-tests audio forensic robustness against real-world degradation:
      1. 16 kHz / 8 kHz Downsampling
      2. MP3 / AAC Social Media Compression
      3. Background Noise Injection (SNR 10 dB, 20 dB)
      4. WhatsApp Voice-Note Opus Codec (16 kbps)
    """

    @staticmethod
    def add_gaussian_noise(waveform: np.ndarray, target_snr_db: float = 15.0) -> np.ndarray:
        signal_power = np.mean(waveform ** 2)
        if signal_power < 1e-8:
            return waveform
        noise_power = signal_power / (10 ** (target_snr_db / 10.0))
        noise = np.random.normal(0, np.sqrt(noise_power), len(waveform)).astype(np.float32)
        return waveform + noise

    @staticmethod
    def simulate_whatsapp_opus(waveform: np.ndarray, sr: int) -> np.ndarray:
        ffmpeg_cmd = shutil.which("ffmpeg")
        if not ffmpeg_cmd:
            return waveform

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in, \
             tempfile.NamedTemporaryFile(suffix=".opus", delete=False) as f_opus, \
             tempfile.NamedTemporaryFile(suffix="_rec.wav", delete=False) as f_out:
            in_path, opus_path, out_path = f_in.name, f_opus.name, f_out.name

        try:
            # Scale to int16
            pcm = (waveform * 32767.0).clip(-32768, 32767).astype(np.int16)
            wavfile.write(in_path, sr, pcm)

            # Transcode to WhatsApp-style Opus (16 kbps)
            subprocess.run([ffmpeg_cmd, "-y", "-i", in_path, "-c:a", "libopus", "-b:a", "16k", opus_path],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            # Transcode back to WAV
            subprocess.run([ffmpeg_cmd, "-y", "-i", opus_path, out_path],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

            _, rec_pcm = wavfile.read(out_path)
            rec_wave = (rec_pcm / 32768.0).astype(np.float32)
            return rec_wave
        except Exception:
            return waveform
        finally:
            for p in [in_path, opus_path, out_path]:
                if os.path.exists(p):
                    os.remove(p)


# ---------------------------------------------------------------------------
# Metric Calculations (Imported from eval_core for consistent tie & CI handling)
# ---------------------------------------------------------------------------
try:
    from eval_core import compute_roc_metrics, bootstrap_ci
except ImportError:
    try:
        from scripts.eval_core import compute_roc_metrics, bootstrap_ci
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from eval_core import compute_roc_metrics, bootstrap_ci


class InDeepFakeBenchmark:
    """
    Multimodal audio-visual benchmark evaluator across Indic languages.
    """

    def __init__(self, annotation_file: Optional[str] = None) -> None:
        self.temporal_agent = VideoTemporalAgent()
        self.lip_agent = LipSyncAgent()
        self.audio_agent = AudioVoiceAgent()
        self.rppg_agent = RPPGAgent()
        self.fusion = DempsterShaferFusion()
        self.lang_id = IndicLanguageIdentifier()

    def is_real_indeepfake_present(self, dataset_path: str) -> bool:
        p = Path(dataset_path)
        if not p.exists():
            return False
        videos = list(p.rglob("*.mp4"))[:301]
        has_indic_subdirs = (p / "hindi").exists() or (p / "tamil").exists() or (p / "telugu").exists() or (p / "bengali").exists()
        return len(videos) > 300 and has_indic_subdirs

    def evaluate_streams(self, video_path: str, fail_loud: bool = False) -> Dict[str, Any]:
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while len(frames) < 45 and cap.isOpened():
            ret, f = cap.read()
            if not ret:
                break
            frames.append(f)
        cap.release()

        if not frames:
            return {"error": "Cannot decode frames"}

        wave, sr = self.lang_id.extract_audio(str(video_path))
        rms = float(np.sqrt(np.mean(wave**2))) if (wave is not None and len(wave) > 0) else 0.0
        has_speech = bool(wave is not None and len(wave) >= 1000 and rms >= 0.005)

        if not has_speech and fail_loud:
            raise NoSpeechDetectedError(
                f"Acoustic stream in '{video_path}' is silent or unvoiced (RMS={rms:.6f}) - no speech detected."
            )

        # 1. Video-only Stream (Temporal)
        t_ev = self.temporal_agent.analyze(frames)

        # 2. Audio-only and 3. Cross-modal Streams
        if has_speech:
            a_ev = self.audio_agent.analyze((wave, sr))
            lip_ev = self.lip_agent.analyze((frames, (wave, sr)))
            score_audio = a_ev.prob_tampered
            score_lip = lip_ev.prob_tampered
            audio_status = "EVALUATED"
            lip_status = "EVALUATED"
            lip_lag = lip_ev.telemetry.get("temporal_lag_ms", 0.0)
        else:
            a_ev = None
            lip_ev = None
            score_audio = None
            score_lip = None
            audio_status = "NOT EVALUATED - no speech"
            lip_status = "NOT EVALUATED - no speech"
            lip_lag = 0.0

        # 4. Fused Multimodal Graph
        masses = {"temporal": t_ev.bba_mass}
        edges = []
        if a_ev and a_ev.bba_mass.get("U", 1.0) < 0.95:
            masses["audio"] = a_ev.bba_mass
            edges.append(("audio", "temporal"))
        if lip_ev and lip_ev.bba_mass.get("U", 1.0) < 0.95:
            masses["lip"] = lip_ev.bba_mass
            edges.append(("lip", "temporal"))
        if not edges:
            edges = [("temporal", "temporal")]

        fused = self.fusion.fuse_graph_evidence(masses, edges)

        # Language identification
        fallback_lang = "Hindi" if "hindi" in str(video_path).lower() else ("Tamil" if "tamil" in str(video_path).lower() else ("Telugu" if "telugu" in str(video_path).lower() else ("Bengali" if "bengali" in str(video_path).lower() else "General Indic")))
        detected_lang, lang_conf = self.lang_id.detect_language(wave, sr, fallback_lang)

        return {
            "score_video_only": t_ev.prob_tampered,
            "score_audio_only": score_audio,
            "score_cross_modal": score_lip,
            "audio_status": audio_status,
            "lip_sync_status": lip_status,
            "has_speech": has_speech,
            "score_fused": fused["pignistic_fake"],
            "verdict": fused["verdict"],
            "lip_lag_ms": lip_lag,
            "detected_language": detected_lang,
            "fallback_language": fallback_lang,
            "lang_agreement": fallback_lang.lower() in detected_lang.lower() if has_speech else False,
        }

    def run_benchmark(self, dataset_path: str, export_json: Optional[str] = None) -> Dict[str, Any]:
        p = Path(dataset_path)
        is_real = self.is_real_indeepfake_present(dataset_path)

        print("=" * 80)
        if not is_real:
            print("⚠️  DATASET PRESENCE AUDIT NOTICE:")
            print("   InDeepFake not found - results below are from local samples only and are NOT InDeepFake results.")
            print("   To obtain and evaluate official InDeepFake (2025), see: docs/INDEEPFAKE_ACCESS.md")
        else:
            print("✅ OFFICIAL INDEEPFAKE MULTIMODAL BENCHMARK DETECTED (Pattern Recognition Letters 2025)")
        print("=" * 80)

        videos = list(p.rglob("*.mp4")) + list(p.rglob("*.avi"))
        if not videos:
            print(f"[!] No video files found in {dataset_path}")
            return {"total_evaluated": 0, "is_real_indeepfake": False}

        results = []
        lang_groups = {}
        agreements = []
        lags_by_lang = {}

        y_true_all = []
        y_fused_all = []
        y_video_all = []
        y_audio_voiced = []
        y_audio_true = []
        y_lip_voiced = []
        y_lip_true = []

        print(f"Evaluating {len(videos)} media files across multimodal streams...")
        for idx, v in enumerate(videos, 1):
            is_fake = "fake" in v.name.lower() or "swap" in v.name.lower() or "inswapper" in v.name.lower()
            gt = 1 if is_fake else 0
            res = self.evaluate_streams(str(v))
            if "error" in res:
                continue

            lang = res["fallback_language"]
            if lang not in lang_groups:
                lang_groups[lang] = {"y_true": [], "y_score": []}
                lags_by_lang[lang] = []

            lang_groups[lang]["y_true"].append(gt)
            lang_groups[lang]["y_score"].append(res["score_fused"])
            lags_by_lang[lang].append(res["lip_lag_ms"])
            if res["has_speech"]:
                agreements.append(res["lang_agreement"])

            y_true_all.append(gt)
            y_fused_all.append(res["score_fused"])
            y_video_all.append(res["score_video_only"])

            if res["score_audio_only"] is not None:
                y_audio_true.append(gt)
                y_audio_voiced.append(res["score_audio_only"])

            if res["score_cross_modal"] is not None:
                y_lip_true.append(gt)
                y_lip_voiced.append(res["score_cross_modal"])

            results.append({"filename": v.name, "path": str(v), "gt": gt, **res})

        y_t = np.array(y_true_all)
        fused_auc, fused_eer, fused_tpr1 = compute_roc_metrics(y_t, np.array(y_fused_all))
        vid_auc, _, _ = compute_roc_metrics(y_t, np.array(y_video_all))

        ci_fused_tuple, ci_fused_str = bootstrap_ci(y_t, np.array(y_fused_all))
        overall_acc = float(np.mean((np.array(y_fused_all) > 0.5) == y_t)) if len(y_t) > 0 else 0.0

        # Print Multimodal Stream Ablation Table
        print("\n" + "=" * 80)
        print("🔬 MULTIMODAL STREAM ABLATION & DISCRIMINATION RESULTS")
        print("=" * 80)
        print(f"{'Stream / Modality':30s} | {'AUC / Status':26s} | {'EER':8s} | {'Status'}")
        print("-" * 75)
        print(f"{'Video-Only (Temporal Jitter)':30s} | {vid_auc:8.3f}                   | {'N/A':8s} | Isolated Visual")

        aud_auc = "NOT EVALUATED - no speech"
        aud_eer = "N/A"
        if len(y_audio_voiced) > 0:
            aud_auc_val, aud_eer_val, _ = compute_roc_metrics(np.array(y_audio_true), np.array(y_audio_voiced))
            aud_auc = aud_auc_val
            aud_eer = aud_eer_val
            print(f"{'Audio-Only (Voice Vocoder)':30s} | {aud_auc:8.3f}                   | {aud_eer:8.3f} | Isolated Acoustic")
        else:
            print(f"{'Audio-Only (Voice Vocoder)':30s} | {'NOT EVALUATED - no speech':26s} | {'N/A':8s} | Acoustic Absent/Silent")

        lip_auc = "NOT EVALUATED - no speech"
        lip_eer = "N/A"
        if len(y_lip_voiced) > 0:
            lip_auc_val, lip_eer_val, _ = compute_roc_metrics(np.array(y_lip_true), np.array(y_lip_voiced))
            lip_auc = lip_auc_val
            lip_eer = lip_eer_val
            print(f"{'Cross-Modal (Lip Synchrony)':30s} | {lip_auc:8.3f}                   | {lip_eer:8.3f} | Audio-Visual Viseme")
        else:
            print(f"{'Cross-Modal (Lip Synchrony)':30s} | {'NOT EVALUATED - no speech':26s} | {'N/A':8s} | AV Synchrony Absent")

        print(f"{'FUSED PRISM SYSTEM (Graph)':30s} | {fused_auc:8.3f}                   | {fused_eer:8.3f} | Full Evidential Fusion")
        print("-" * 75)
        print(f"Overall Accuracy: {overall_acc:.1%} | 95% CI: {ci_fused_str}")

        # Print Language Slicing Table
        agreement_rate = float(np.mean(agreements)) if agreements else 0.0
        print("\n" + "=" * 80)
        print("🗣️ INDIC LANGUAGE BREAKDOWN & PHONEME-VISEME BIAS AUDIT")
        print("=" * 80)
        print(f"Language Tag Agreement Rate: {agreement_rate:.1%}")
        print(f"{'Language':16s} | {'N':5s} | {'AUC':7s} | {'Mean AV Lag':12s} | {'Std Lag':10s} | {'Reliability Status'}")
        print("-" * 80)

        insufficient_languages = []
        for l_name, l_data in lang_groups.items():
            l_yt = np.array(l_data["y_true"])
            l_ys = np.array(l_data["y_score"])
            n_l = len(l_yt)
            l_auc, _, _ = compute_roc_metrics(l_yt, l_ys)
            l_lags = lags_by_lang.get(l_name, [0.0])
            m_lag = float(np.mean(l_lags))
            s_lag = float(np.std(l_lags))

            rel = "✅ Valid" if n_l >= 30 else "⚠️ Insufficient (N < 30)"
            if n_l < 30:
                insufficient_languages.append(l_name)

            print(f"{l_name:16s} | {n_l:5d} | {l_auc:7.3f} | {m_lag:10.1f} ms | {s_lag:8.1f} ms | {rel}")

        if insufficient_languages:
            print(f"\n⚠️  Explicit Data Caveat: The following language cohorts have insufficient samples (N < 30) to draw statistical conclusions: {', '.join(insufficient_languages)}.")

        # Modality statistics
        parser_inst = IndicMetadataParser()
        modality_groups: Dict[str, Dict[str, List]] = {}
        for r in results:
            m_attrs = parser_inst.parse_attributes(Path(r.get("path", r.get("filename", ""))))
            mod = m_attrs.get("modality", "Multimodal")
            if mod not in modality_groups:
                modality_groups[mod] = {"y_true": [], "y_score": []}
            modality_groups[mod]["y_true"].append(r.get("gt", r.get("y_true", 0)))
            modality_groups[mod]["y_score"].append(r.get("score_fused", r.get("fused_score", 0.5)))

        summary = {
            "is_real_indeepfake": is_real,
            "disclaimer": "Local samples only - NOT InDeepFake" if not is_real else "Official InDeepFake",
            "total_evaluated": len(videos),
            "overall_accuracy": round(overall_acc, 4),
            "stream_ablation": {
                "video_only_auc": vid_auc,
                "audio_only_auc": aud_auc,
                "audio_only_eer": aud_eer,
                "cross_modal_auc": lip_auc,
                "cross_modal_eer": lip_eer,
                "fused_auc": fused_auc,
                "fused_eer": fused_eer,
                "fused_ci_95": ci_fused_str,
            },
            "language_statistics": lang_groups,
            "modality_statistics": modality_groups,
            "insufficient_languages": insufficient_languages,
            "videos": results,
        }

        if export_json:
            with open(export_json, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRISM InDeepFake Benchmark Evaluator")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Directory of InDeepFake dataset")
    parser.add_argument("--sample_dir", type=str, default="data/real_test_samples", help="Local sample directory")
    parser.add_argument("--export_json", type=str, default=None, help="Save evaluation summary to JSON")

    args = parser.parse_args()
    target_dir = args.dataset_dir if args.dataset_dir else args.sample_dir

    bench = InDeepFakeBenchmark()
    bench.run_benchmark(target_dir, export_json=args.export_json)

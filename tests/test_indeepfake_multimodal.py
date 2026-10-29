"""
tests/test_indeepfake_multimodal.py
Unit tests verifying InDeepFake benchmark components, language identification,
audio perturbation suite, silent fail-loud behavior, and real speech clip loader.
"""

import sys
import os
import unittest
import tempfile
import json
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from benchmark_indeepfake import (
    InDeepFakeBenchmark,
    IndicLanguageIdentifier,
    AudioPerturbationSuite,
    RealSpeechClipLoader,
    NoSpeechDetectedError,
)
from agents.audio_voice_agent import AudioVoiceAgent
from agents.lip_sync_agent import LipSyncAgent


class TestInDeepFakeMultimodal(unittest.TestCase):

    def setUp(self):
        self.bench = InDeepFakeBenchmark()

    def test_01_real_dataset_presence_check(self):
        """InDeepFake dataset presence detector must return False for local sample directory."""
        self.assertFalse(self.bench.is_real_indeepfake_present("data/real_test_samples"))
        self.assertFalse(self.bench.is_real_indeepfake_present("non_existent_folder_abc"))

    def test_02_silent_audio_detection(self):
        """IndicLanguageIdentifier must cleanly detect silent / unvoiced audio."""
        detected_lang, conf = IndicLanguageIdentifier.detect_language(None, 16000, "Hindi")
        self.assertEqual(detected_lang, "Unvoiced / Silent")
        self.assertEqual(conf, 0.0)

        short_wave = np.zeros(100, dtype=np.float32)
        detected_lang2, conf2 = IndicLanguageIdentifier.detect_language(short_wave, 16000, "Tamil")
        self.assertEqual(detected_lang2, "Unvoiced / Silent")

    def test_03_audio_perturbation_gaussian_noise(self):
        """Audio perturbation suite must correctly add Gaussian noise to specified SNR."""
        sr = 16000
        t = np.linspace(0, 1.0, sr)
        clean = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        noisy = AudioPerturbationSuite.add_gaussian_noise(clean, target_snr_db=10.0)

        self.assertEqual(len(noisy), len(clean))
        self.assertFalse(np.allclose(clean, noisy))

        # Check noise power added
        noise_diff = noisy - clean
        snr_meas = 10.0 * np.log10(np.mean(clean**2) / (np.mean(noise_diff**2) + 1e-9))
        self.assertAlmostEqual(snr_meas, 10.0, delta=1.5)

    def test_04_stream_ablation_isolation_and_silent_status(self):
        """Multi-stream evaluation must return NOT EVALUATED for absent audio instead of dummy score."""
        sample_video = "data/real_test_samples/01_real_human_video_v1.mp4"
        if os.path.exists(sample_video):
            res = self.bench.evaluate_streams(sample_video)
            self.assertIn("score_video_only", res)
            self.assertIn("score_audio_only", res)
            self.assertIn("score_cross_modal", res)
            self.assertIn("score_fused", res)
            self.assertGreaterEqual(res["score_video_only"], 0.0)
            self.assertLessEqual(res["score_video_only"], 1.0)
            # Sample video has no audio track -> must report NOT EVALUATED
            self.assertIsNone(res["score_audio_only"])
            self.assertIsNone(res["score_cross_modal"])
            self.assertEqual(res["audio_status"], "NOT EVALUATED - no speech")
            self.assertEqual(res["lip_sync_status"], "NOT EVALUATED - no speech")

    def test_05_silent_audio_fails_loudly_when_requested(self):
        """Audio and lip-sync agents must raise NoSpeechDetectedError when fail_loud=True on silent input."""
        sample_video = "data/real_test_samples/01_real_human_video_v1.mp4"
        if os.path.exists(sample_video):
            with self.assertRaises(NoSpeechDetectedError):
                self.bench.evaluate_streams(sample_video, fail_loud=True)

        # Direct agent tests
        audio_ag = AudioVoiceAgent()
        lip_ag = LipSyncAgent()

        silent_pcm = np.zeros(16000, dtype=np.float32)
        with self.assertRaises(NoSpeechDetectedError):
            audio_ag.analyze((silent_pcm, 16000), fail_loud=True)

        with self.assertRaises(NoSpeechDetectedError):
            lip_ag.analyze(([np.zeros((100, 100, 3), dtype=np.uint8)] * 10, (silent_pcm, 16000)), fail_loud=True)

    def test_06_real_speech_clip_loader(self):
        """RealSpeechClipLoader must load annotations, validate vocal presence, and slice by language."""
        sr = 16000
        t = np.linspace(0, 0.5, int(sr * 0.5))
        speech_pcm = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        silent_pcm = np.zeros(int(sr * 0.5), dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            f_voiced = os.path.join(tmp_dir, "speaker01_hindi_real.wav")
            f_silent = os.path.join(tmp_dir, "speaker02_tamil_silent.wav")

            wavfile.write(f_voiced, sr, (speech_pcm * 32767).astype(np.int16))
            wavfile.write(f_silent, sr, (silent_pcm * 32767).astype(np.int16))

            annotations = [
                {"path": f_voiced, "label": 0, "language": "Hindi", "speaker_id": "spk01"},
                {"path": f_silent, "label": 0, "language": "Tamil", "speaker_id": "spk02"},
            ]
            meta_path = os.path.join(tmp_dir, "meta.json")
            with open(meta_path, "w", encoding="utf-8") as mf:
                json.dump(annotations, mf)

            loader = RealSpeechClipLoader(annotation_file=meta_path)
            self.assertEqual(len(loader.records), 2)

            voiced = loader.get_voiced_clips()
            self.assertEqual(len(voiced), 1)
            self.assertEqual(voiced[0]["language"], "Hindi")
            self.assertTrue(voiced[0]["has_speech"])

            tamil_clips = loader.get_clips_by_language("Tamil")
            self.assertEqual(len(tamil_clips), 1)
            self.assertFalse(tamil_clips[0]["has_speech"])


if __name__ == "__main__":
    unittest.main()

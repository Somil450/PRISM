"""
src/dataset_manifest.py
Standardized Dataset Ingestion Layer and Manifest Schema for PRISM.

Supports:
  1. FaceForensics++ (FF++ c0/c23/c40)
  2. Celeb-DF (v2)
  3. Deepfake Detection Challenge (DFDC / DFDC Preview)
  4. Local generic directories with fallback metadata resolution

Enforces:
  - Strict manifest schema: (path, label, identity, generator, compression)
  - Zero-Identity-Leakage partitioning (identity-disjoint train / calibration / held-out test splits)
"""

import os
import re
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np


MANIFEST_SCHEMA_FIELDS = ["path", "label", "identity", "generator", "compression"]


def create_manifest_entry(
    path: str,
    label: int,
    identity: str,
    generator: str = "original",
    compression: str = "unknown",
) -> Dict[str, Any]:
    """Creates a validated manifest entry adhering to the PRISM schema."""
    assert label in (0, 1), f"Label must be 0 (authentic) or 1 (tampered), got {label}"
    return {
        "path": str(Path(path).as_posix()),
        "label": int(label),
        "identity": str(identity).strip(),
        "generator": str(generator).strip(),
        "compression": str(compression).strip(),
    }


class DatasetManifestLoader:
    """
    Ingests official benchmark datasets into a unified PRISM manifest.
    """

    @staticmethod
    def load_from_file(manifest_path: str) -> List[Dict[str, Any]]:
        """Loads an existing CSV or JSON manifest."""
        p = Path(manifest_path)
        if not p.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        records = []
        if p.suffix.lower() == ".json":
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    records.append(create_manifest_entry(
                        path=item["path"],
                        label=int(item["label"]),
                        identity=item["identity"],
                        generator=item.get("generator", "original" if item["label"] == 0 else "unknown"),
                        compression=item.get("compression", "unknown"),
                    ))
        elif p.suffix.lower() == ".csv":
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(create_manifest_entry(
                        path=row["path"],
                        label=int(row["label"]),
                        identity=row["identity"],
                        generator=row.get("generator", "original" if int(row["label"]) == 0 else "unknown"),
                        compression=row.get("compression", "unknown"),
                    ))
        return records

    @staticmethod
    def save_to_file(manifest: List[Dict[str, Any]], output_path: str) -> None:
        """Saves manifest to CSV or JSON."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() == ".json":
            with open(p, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        else:
            with open(p, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=MANIFEST_SCHEMA_FIELDS)
                writer.writeheader()
                for r in manifest:
                    writer.writerow({k: r.get(k, "") for k in MANIFEST_SCHEMA_FIELDS})

    @staticmethod
    def load_faceforensics(dataset_root: str, compression: str = "c23") -> List[Dict[str, Any]]:
        """
        Loads FaceForensics++ directory structure.
        Standard layout:
          dataset_root/
            original_sequences/youtube/{compression}/videos/*.mp4
            manipulated_sequences/{Deepfakes,Face2Face,FaceSwap,NeuralTextures,FaceShifter}/{compression}/videos/*.mp4
        """
        root = Path(dataset_root)
        manifest = []
        if not root.exists():
            return manifest

        # 1. Authentic originals
        orig_paths = list(root.glob(f"original_sequences/*/{compression}/videos/*.mp4")) + \
                     list(root.glob(f"original_sequences/*/{compression}/*.mp4"))
        for p in orig_paths:
            vid_id = p.stem.split("_")[0]
            manifest.append(create_manifest_entry(
                path=str(p),
                label=0,
                identity=f"ff_{vid_id}",
                generator="original",
                compression=compression,
            ))

        # 2. Manipulated sequences
        manip_dir = root / "manipulated_sequences"
        if manip_dir.exists():
            for method_dir in manip_dir.iterdir():
                if not method_dir.is_dir():
                    continue
                gen_name = method_dir.name
                vids = list(method_dir.glob(f"{compression}/videos/*.mp4")) + \
                       list(method_dir.glob(f"{compression}/*.mp4"))
                for p in vids:
                    # e.g. 001_002.mp4: target 001 was swapped onto 002
                    parts = p.stem.split("_")
                    target_id = parts[0] if parts else p.stem
                    manifest.append(create_manifest_entry(
                        path=str(p),
                        label=1,
                        identity=f"ff_{target_id}",
                        generator=gen_name,
                        compression=compression,
                    ))

        return manifest

    @staticmethod
    def parse_celeb_df_test_list(test_list_file: str) -> Dict[str, Any]:
        """
        Parses official Celeb-DF v2 List_of_testing_videos.txt.
        Returns mapping of relative paths, filenames, identities, and labels.
        """
        p = Path(test_list_file)
        if not p.exists():
            raise FileNotFoundError(f"Celeb-DF test list not found: {test_list_file}")

        test_rel_paths = set()
        test_filenames = set()
        test_identities = set()
        test_labels = {}

        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    lbl = int(parts[0])
                    rel_p = parts[1].replace("\\", "/")
                else:
                    rel_p = parts[0].replace("\\", "/")
                    lbl = 1 if "synth" in rel_p.lower() else 0

                fname = Path(rel_p).name
                test_rel_paths.add(rel_p)
                test_filenames.add(fname)
                test_labels[rel_p] = lbl
                test_labels[fname] = lbl

                # Extract identity tags (idX)
                stem = Path(rel_p).stem
                for sp in stem.split("_"):
                    if sp.startswith("id") and len(sp) > 2 and sp[2:].isdigit():
                        test_identities.add(sp)

        return {
            "test_relative_paths": test_rel_paths,
            "test_filenames": test_filenames,
            "test_identities": test_identities,
            "test_labels": test_labels,
        }

    @staticmethod
    def load_celeb_df_v2(
        dataset_root: str,
        test_list_path: Optional[str] = None,
        split: str = "all",
    ) -> List[Dict[str, Any]]:
        """
        Loads Celeb-DF v2 directory structure adhering to official splits.
        Standard layout:
          dataset_root/
            Celeb-real/idX_ZZZZ.mp4
            YouTube-real/00000.mp4
            Celeb-synthesis/idX_idY_ZZZZ.mp4
            List_of_testing_videos.txt (official 518 test videos)

        Args:
            dataset_root: Path to Celeb-DF-v2 root.
            test_list_path: Path to List_of_testing_videos.txt.
            split: 'all', 'train' (excludes test videos & test IDs), or 'test' (only official test videos).
        """
        root = Path(dataset_root)
        manifest = []
        if not root.exists():
            return manifest

        test_info: Optional[Dict[str, Any]] = None
        tl_path = Path(test_list_path) if test_list_path else root / "List_of_testing_videos.txt"
        if tl_path.exists():
            try:
                test_info = DatasetManifestLoader.parse_celeb_df_test_list(str(tl_path))
            except Exception:
                test_info = None

        all_candidates = []

        # Celeb-real (label = 0)
        celeb_real = root / "Celeb-real"
        if celeb_real.exists():
            for p in celeb_real.glob("*.mp4"):
                parts = p.stem.split("_")
                identity = parts[0] if parts[0].startswith("id") else f"celeb_{parts[0]}"
                all_candidates.append(create_manifest_entry(
                    path=str(p),
                    label=0,
                    identity=identity,
                    generator="original",
                    compression="raw",
                ))

        # YouTube-real (label = 0)
        yt_real = root / "YouTube-real"
        if yt_real.exists():
            for p in yt_real.glob("*.mp4"):
                all_candidates.append(create_manifest_entry(
                    path=str(p),
                    label=0,
                    identity=f"yt_{p.stem}",
                    generator="original",
                    compression="youtube_h264",
                ))

        # Celeb-synthesis (label = 1)
        celeb_synth = root / "Celeb-synthesis"
        if celeb_synth.exists():
            for p in celeb_synth.glob("*.mp4"):
                parts = p.stem.split("_")
                target_id = parts[0] if parts and parts[0].startswith("id") else f"synth_{p.stem}"
                all_candidates.append(create_manifest_entry(
                    path=str(p),
                    label=1,
                    identity=target_id,
                    generator="DeepFaceLab_CelebDF",
                    compression="h264",
                ))

        if split == "all" or test_info is None:
            return all_candidates

        test_filenames = test_info["test_filenames"]
        test_identities = test_info["test_identities"]

        filtered_manifest = []
        for entry in all_candidates:
            fname = Path(entry["path"]).name
            stem = Path(entry["path"]).stem
            entry_ids = set()
            for part in stem.split("_"):
                if part.startswith("id") and len(part) > 2 and part[2:].isdigit():
                    entry_ids.add(part)
            if entry["identity"].startswith("id"):
                entry_ids.add(entry["identity"])

            is_test_video = fname in test_filenames
            has_test_identity = bool(entry_ids.intersection(test_identities))

            if split == "test":
                if is_test_video:
                    filtered_manifest.append(entry)
            elif split == "train":
                # Strict Zero-Identity Leakage: Exclude if video is in test set OR identity appears in test set
                if not is_test_video and not has_test_identity:
                    filtered_manifest.append(entry)

        return filtered_manifest

    @staticmethod
    def verify_celeb_df_zero_leakage(
        train_manifest: List[Dict[str, Any]],
        test_manifest: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Proves no test video or identity was in training manifest:
        1. Checks filename / video ID disjointness.
        2. Checks celebrity identity ID (idX) disjointness.
        Raises AssertionError if any leakage is found.
        """
        def get_vids(m_list):
            return set(Path(m["path"]).name for m in m_list)

        def get_ids(m_list):
            ids = set()
            for m in m_list:
                ident = m.get("identity", "")
                if ident.startswith("id"):
                    ids.add(ident)
                stem = Path(m["path"]).stem
                for part in stem.split("_"):
                    if part.startswith("id") and len(part) > 2 and part[2:].isdigit():
                        ids.add(part)
            return ids

        train_vids = get_vids(train_manifest)
        test_vids = get_vids(test_manifest)
        video_overlap = train_vids.intersection(test_vids)

        train_ids = get_ids(train_manifest)
        test_ids = get_ids(test_manifest)
        identity_overlap = train_ids.intersection(test_ids)

        if len(video_overlap) > 0:
            raise AssertionError(f"Video ID train-test leakage detected ({len(video_overlap)} videos overlap): {sorted(list(video_overlap))[:5]}")

        if len(identity_overlap) > 0:
            raise AssertionError(f"Identity train-test leakage detected ({len(identity_overlap)} identities overlap): {sorted(list(identity_overlap))}")

        return {
            "train_video_count": len(train_vids),
            "test_video_count": len(test_vids),
            "train_identity_count": len(train_ids),
            "test_identity_count": len(test_ids),
            "video_overlap_count": 0,
            "identity_overlap_count": 0,
            "proven_zero_leakage": True,
        }

    @staticmethod
    def load_dfdc(dataset_root: str) -> List[Dict[str, Any]]:
        """
        Loads DFDC (Deepfake Detection Challenge) directory structure.
        Standard layout:
          dataset_root/
            metadata.json (or dfdc_train_part_XX/metadata.json)
            *.mp4
        """
        root = Path(dataset_root)
        manifest = []
        if not root.exists():
            return manifest

        meta_files = list(root.rglob("metadata.json"))
        for mf in meta_files:
            try:
                with open(mf, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
                parent_dir = mf.parent
                for filename, info in meta_data.items():
                    vid_path = parent_dir / filename
                    if not vid_path.exists():
                        continue
                    is_fake = 1 if info.get("label", "").upper() == "FAKE" else 0
                    orig = info.get("original")
                    # Use original filename as cluster identity if fake, else self
                    identity = f"dfdc_{Path(orig).stem}" if orig and is_fake else f"dfdc_{Path(filename).stem}"
                    manifest.append(create_manifest_entry(
                        path=str(vid_path),
                        label=is_fake,
                        identity=identity,
                        generator="DFDC_Pipeline" if is_fake else "original",
                        compression="h264",
                    ))
            except Exception:
                continue

        return manifest

    @staticmethod
    def partition_by_identity(
        manifest: List[Dict[str, Any]],
        calib_ratio: float = 0.50,
        seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Partitions manifest into Calibration and Held-Out Test splits with
        STRICT ZERO IDENTITY LEAKAGE. No identity in calibration split can
        ever appear in the held-out test split.
        """
        if not manifest:
            return [], []

        identities = sorted(list(set(m["identity"] for m in manifest)))
        rng = np.random.RandomState(seed)
        rng.shuffle(identities)

        split_idx = int(round(len(identities) * calib_ratio))
        calib_identities: Set[str] = set(identities[:split_idx])
        test_identities: Set[str] = set(identities[split_idx:])

        # Disjointness verification
        intersection = calib_identities.intersection(test_identities)
        assert len(intersection) == 0, f"Identity leakage detected: {intersection}"

        calib_split = [m for m in manifest if m["identity"] in calib_identities]
        test_split = [m for m in manifest if m["identity"] in test_identities]

        return calib_split, test_split

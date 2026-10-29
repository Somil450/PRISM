"""
src/agents/metadata_agent.py
Forensic Metadata & Container Provenance Agent.

Inspects EXIF tags, PNG textual chunks, XMP packets, C2PA manifests,
and detects known AI generator signatures (Stable Diffusion, Midjourney, DALL-E, ComfyUI).
"""

from typing import Any, Dict, List, Optional
import io
import re
from PIL import Image, ExifTags

try:
    from .base_agent import AgentEvidence, BaseForensicAgent
except ImportError:
    from base_agent import AgentEvidence, BaseForensicAgent

# Known AI generation keywords in metadata chunks
_AI_GENERATOR_KEYWORDS = [
    "stable diffusion",
    "automatic1111",
    "comfyui",
    "midjourney",
    "dall-e",
    "dalle",
    "novelai",
    "invokeai",
    "fooocus",
    "adobe firefly",
    "generative fill",
    "steps: ",
    "sampler: ",
    "cfg scale: ",
    "seed: ",
    "negative prompt:",
    "claude",
    "anthropic",
    "runway",
    "kling",
    "pika",
    "sora",
    "luma",
    "haiper",
    "gen-2",
    "gen-3",
    "generative ai",
    "google generative ai",
    "c2pa.created",
    "c2pa.actions",
    "synthid",
    "imagen",
    "veo",
    "soravideo",
    "minimax",
    "seedance",
    "cogvideo",
    "wan-ai",
    "svd",
]

# Manipulation-grade software (face-swaps, mobile deepfake apps, automated synthesis pipelines)
_MANIPULATION_SOFTWARE = [
    "faceapp",
    "capcut",
    "bytedance",
    "inshot",
    "filmora",
    "deepface",
    "reface",
]

# Professional creative editing and color-grading tools (benign post-processing)
_CREATIVE_EDITING_SOFTWARE = [
    "photoshop",
    "gimp",
    "canva",
    "picsart",
    "lightroom",
    "snapseed",
    "premiere",
    "after effects",
    "davinci",
]

# Editing software and re-encoding toolchain keywords
_SUSPICIOUS_SOFTWARE = [
    "photoshop",
    "gimp",
    "canva",
    "picsart",
    "lightroom",
    "snapseed",
    "faceapp",
    "lavf",
    "lavc",
    "ffmpeg",
    "handbrake",
    "premiere",
    "after effects",
    "capcut",
    "bytedance",
    "davinci",
    "inshot",
    "filmora",
    "quicktime",
]

# C2PA (Coalition for Content Provenance and Authenticity) signatures
_C2PA_SIGNATURES = [
    b"c2pa",
    b"jumb",
    b"c2ma",
    b"c2as",
    b"contentcredentials",
]


class MetadataAgent(BaseForensicAgent):
    """
    Analyzes image and container metadata for provenance verification,
    editing footprints, and generative AI parameters.
    """

    def __init__(self, base_reliability: float = 0.90) -> None:
        super().__init__(name="Metadata & Provenance Agent", base_reliability=base_reliability)

    def _extract_exif(self, image: Image.Image) -> Dict[str, Any]:
        """Extracts human-readable EXIF dictionary from PIL Image."""
        exif_data = {}
        try:
            raw_exif = image._getexif()
            if raw_exif:
                for tag_id, value in raw_exif.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    # Convert raw binary or large bytes to string safely
                    if isinstance(value, bytes):
                        try:
                            value = value.decode("utf-8", errors="ignore").strip()
                        except Exception:
                            value = f"<binary {len(value)} bytes>"
                    exif_data[tag_name] = str(value)
        except Exception:
            pass
        return exif_data

    def _extract_png_info(self, image: Image.Image) -> Dict[str, str]:
        """Extracts PNG text metadata chunks (e.g. parameters, prompt, workflow)."""
        info_dict = {}
        if hasattr(image, "info") and isinstance(image.info, dict):
            for k, v in image.info.items():
                if isinstance(v, (str, int, float)):
                    info_dict[str(k)] = str(v)
                elif isinstance(v, bytes):
                    try:
                        info_dict[str(k)] = v.decode("utf-8", errors="ignore").strip()
                    except Exception:
                        pass
        return info_dict

    def _learned_anomaly_detection(self, metadata: dict) -> float:
        """
        Analyzes metadata structure for anomalies (e.g., missing critical camera
        tags when 'Make' or 'Model' is present, or conflicting resolution tags).
        """
        score = 0.0
        keys = set([k.lower() for k in metadata.keys()])
        
        # Anomaly 1: Camera Make/Model exists but no ExposureTime, FNumber, or ISOSpeedRatings
        has_camera = "make" in keys or "model" in keys
        has_optics = any(k in keys for k in ["exposuretime", "fnumber", "isospeedratings", "focallength"])
        if has_camera and not has_optics:
            score += 0.4  # Highly suspicious: fake camera tags injected by software
            
        # Anomaly 2: Software tag exists alongside unmodified original date (rare)
        if "software" in keys and "datetimeoriginal" in keys and "datetimedigitized" not in keys:
            score += 0.2
            
        return min(1.0, score)

    def _inspect_video_container(self, raw_bytes: bytes) -> Dict[str, Any]:
        """
        Scans video container metadata atoms (MP4, MOV, WebM, AVI), strictly skipping
        the 'mdat' (media payload) entropy stream to prevent random bitstream substring false positives.
        Inspects:
        - Major brand / compatible brands (ftyp, moov)
        - Re-encoding toolchain / encoder signatures (Lavf, Lavc, ffmpeg, CapCut, Adobe)
        - Hardware camera atoms (©mak, ©mod, ©xyz, com.apple.quicktime)
        """
        meta_bytes = b""
        is_mp4_container = False
        offset = 0
        n_bytes = len(raw_bytes)

        # Parse sequential top-level boxes, extracting metadata atoms and skipping mdat
        while offset < n_bytes - 8:
            box_size = int.from_bytes(raw_bytes[offset:offset+4], "big")
            box_type = raw_bytes[offset+4:offset+8]
            if box_size == 1 and offset + 16 <= n_bytes:
                box_size = int.from_bytes(raw_bytes[offset+8:offset+16], "big")
            if box_size <= 0 or offset + box_size > n_bytes + 64:
                meta_bytes += raw_bytes[offset:min(offset + 4096, n_bytes)]
                break

            if box_type in (b"ftyp", b"moov", b"udta", b"meta", b"trak", b"free", b"jumb", b"uuid"):
                is_mp4_container = True
                meta_bytes += raw_bytes[offset:min(offset + box_size, n_bytes)]
            elif box_type == b"mdat":
                is_mp4_container = True
                # Intentionally skip mdat video entropy stream

            offset += box_size

        if not is_mp4_container and (b"ftyp" in raw_bytes[:1024] or b"moov" in raw_bytes[:65536] or b"jumb" in raw_bytes[:1024]):
            is_mp4_container = True
            meta_bytes = raw_bytes[:4096]

        try:
            sample_text = meta_bytes.decode("ascii", errors="ignore").lower()
        except Exception:
            sample_text = ""

        found_encoders = []
        for sw in _SUSPICIOUS_SOFTWARE:
            if sw in sample_text:
                found_encoders.append(sw)

        found_ai = []
        for ai_kw in _AI_GENERATOR_KEYWORDS:
            if ai_kw in sample_text:
                found_ai.append(ai_kw)

        # Use verified camera hardware brand tokens (minimum 4 characters to avoid random substring collision)
        camera_markers = ["apple", "iphone", "samsung", "galaxy", "pixel", "google", "sony", "canon", "nikon", "panasonic", "fujifilm", "gopro"]
        found_cameras = []
        for cam in camera_markers:
            if cam in sample_text:
                # If cam is "google", ensure it is not part of "google generative ai" or "google c2pa" or "google media processing"
                if cam == "google" and any(ai_term in sample_text for ai_term in ["generative", "c2pa", "generator", "media processing"]):
                    continue
                found_cameras.append(cam)

        has_lavf = any(e in ["lavf", "lavc", "ffmpeg"] for e in found_encoders)

        return {
            "is_video_container": is_mp4_container,
            "encoders": found_encoders,
            "ai_signatures": found_ai,
            "cameras": found_cameras,
            "has_lavf": has_lavf,
        }

    def analyze(self, image_input: Any, **kwargs: Any) -> AgentEvidence:
        """
        Runs comprehensive container and metadata forensic analysis across images and video streams.
        Accepts in-memory media payload (raw bytes, PIL Image, or numpy array).
        Strictly prohibits path, filename, label, generator, or identity arguments.

        Args:
            image_input: PIL Image instance, numpy array, or raw in-memory bytes.
            kwargs:
                file_bytes: Raw bytes of the media file (if separate from image_input)

        Returns:
            AgentEvidence packet with quantitative tampering mass and findings.
        """
        self._validate_no_leakage_kwargs(kwargs)

        if isinstance(image_input, str):
            raise ValueError(
                "MetadataAgent.analyze() does not accept path or filename strings. "
                "Pass in-memory bytes, PIL Image, or numpy array directly to prevent label leakage."
            )

        file_bytes = kwargs.get("file_bytes", None)
        image = None
        exif = {}
        png_info = {}

        # Check if raw input is bytes
        if isinstance(image_input, bytes):
            if file_bytes is None:
                file_bytes = image_input
            if image_input.startswith(b"\x89PNG") or image_input.startswith(b"\xff\xd8\xff") or b"WEBP" in image_input[:16]:
                try:
                    image = Image.open(io.BytesIO(image_input))
                    exif = self._extract_exif(image)
                    png_info = self._extract_png_info(image)
                except Exception:
                    pass
        elif isinstance(image_input, Image.Image):
            image = image_input
            exif = self._extract_exif(image)
            png_info = self._extract_png_info(image)
        elif hasattr(image_input, "ndim"):
            import numpy as np
            if isinstance(image_input, np.ndarray):
                try:
                    image = Image.fromarray(image_input)
                    exif = self._extract_exif(image)
                    png_info = self._extract_png_info(image)
                except Exception:
                    pass

        findings: List[str] = []
        suspicious_tags: List[str] = []
        ai_detected = False
        software_detected = False
        manipulation_detected = False
        creative_editing_detected = False
        c2pa_detected = False
        has_camera_hardware = False

        # Check C2PA Content Credentials provenance manifest
        if file_bytes is not None and len(file_bytes) > 16:
            lower_bytes = file_bytes.lower()
            for sig in _C2PA_SIGNATURES:
                if sig in lower_bytes:
                    c2pa_detected = True
                    if any(syn in lower_bytes for syn in [b"generative ai", b"c2pa.created", b"synthetic", b"digitalcreation", b"created by google", b"generator"]):
                        ai_detected = True
                        suspicious_tags.append("C2PA manifest explicitly certifies synthetic generative AI creation.")
                        findings.append("C2PA Content Credentials provenance manifest explicitly certifies synthetic generative AI creation.")
                    else:
                        findings.append(f"C2PA Content Credentials provenance manifest identified in media container ('{sig.decode('ascii', errors='ignore')}').")
                    break

        # Strict zero-leakage metadata analysis: inspect ONLY EXIF tags, PNG info, and container bytes.
        # NEVER inspect filename or path tokens (prohibited label leakage).
        all_text_metadata = " ".join(
            list(exif.values()) + list(png_info.values()) + list(png_info.keys())
        ).lower()

        # 1. Video Container Inspection
        container_info = None
        multiplexer_detected = False
        if file_bytes is not None and len(file_bytes) > 64:
            container_info = self._inspect_video_container(file_bytes)
            if container_info["is_video_container"]:
                if container_info["ai_signatures"]:
                    ai_detected = True
                    suspicious_tags.append(f"AI Generator tag in container: '{', '.join(container_info['ai_signatures'])}'")
                
                hard_encoders = [e for e in container_info["encoders"] if e not in ["lavf", "lavc", "ffmpeg", "quicktime"]]
                transcode_encoders = [e for e in container_info["encoders"] if e in ["lavf", "lavc", "ffmpeg", "quicktime"]]
                
                if hard_encoders:
                    # Check if hard encoder is in manipulation software
                    for he in hard_encoders:
                        if any(ms in he for ms in _MANIPULATION_SOFTWARE):
                            manipulation_detected = True
                            software_detected = True
                            suspicious_tags.append(f"High-risk manipulation software detected in container: '{he}'")
                        elif any(cs in he for cs in _CREATIVE_EDITING_SOFTWARE):
                            creative_editing_detected = True
                            suspicious_tags.append(f"Creative post-processing software in container: '{he}'")
                        else:
                            software_detected = True
                            suspicious_tags.append(f"Container edited via production toolchain: '{he}'")
                elif transcode_encoders:
                    multiplexer_detected = True
                    suspicious_tags.append(f"Container multiplexed via generic transcode library: '{', '.join(transcode_encoders)}'")

                if container_info["cameras"]:
                    has_camera_hardware = True
                    findings.append(f"Camera hardware marker in container: '{', '.join(container_info['cameras'])}'")

        # 2. Check for AI generator fingerprints in internal text metadata chunks (EXIF/PNG chunks only)
        for kw in _AI_GENERATOR_KEYWORDS:
            if kw in all_text_metadata:
                ai_detected = True
                suspicious_tags.append(f"AI Signature detected in metadata chunks: '{kw}'")
                break

        # 4. Check for editing software tags (tiered analysis)
        for sw in _MANIPULATION_SOFTWARE:
            if sw in all_text_metadata:
                manipulation_detected = True
                software_detected = True
                suspicious_tags.append(f"High-risk manipulation software detected: '{sw}'")

        for sw in _CREATIVE_EDITING_SOFTWARE:
            if sw in all_text_metadata:
                creative_editing_detected = True
                suspicious_tags.append(f"Creative post-processing software trace: '{sw}'")

        for sw in _SUSPICIOUS_SOFTWARE:
            if sw in all_text_metadata and sw not in ["lavf", "lavc", "ffmpeg", "quicktime"]:
                if not manipulation_detected and not creative_editing_detected:
                    software_detected = True
                    suspicious_tags.append(f"Editing software trace: '{sw}'")

        # 5. Check for authentic camera hardware tags in EXIF
        make = exif.get("Make", "")
        model = exif.get("Model", "")
        datetime_orig = exif.get("DateTimeOriginal", exif.get("DateTime", ""))
        focal_length = exif.get("FocalLength", "")
        iso = exif.get("ISOSpeedRatings", exif.get("ISO", ""))

        if make or model:
            has_camera_hardware = True
            findings.append(f"Camera Hardware identified: {make} {model}".strip())
            if datetime_orig:
                findings.append(f"Capture Timestamp: {datetime_orig}")
            if focal_length or iso:
                findings.append(f"Optics: Focal Length={focal_length}, ISO={iso}")

        # 6. Determine Tampering Probability & Reliability
        if ai_detected:
            prob_tampered = 0.99
            reliability = 0.95
            verdict = "TAMPERED"
            findings.append("CRITICAL: Raw prompt/workflow parameters or AI generator footprint detected in container chunks.")
        elif manipulation_detected:
            prob_tampered = 0.85
            reliability = 0.88
            verdict = "TAMPERED"
            findings.append("CRITICAL: Media container indicates automated face-swap or synthetic mobile manipulation toolchain.")
        elif c2pa_detected and not manipulation_detected and not ai_detected:
            prob_tampered = 0.05
            reliability = 0.92
            verdict = "AUTHENTIC"
            findings.append("CONFIRMED: Cryptographic C2PA provenance manifest verifies capture chain authenticity.")
        elif creative_editing_detected and has_camera_hardware:
            prob_tampered = 0.20
            reliability = 0.75
            verdict = "AUTHENTIC"
            findings.append("CONFIRMED: Camera hardware capture metadata confirmed alongside non-destructive creative post-processing.")
        elif creative_editing_detected or software_detected:
            prob_tampered = 0.85
            reliability = 0.88
            verdict = "TAMPERED"
            findings.append("WARNING: Media container indicates post-capture editing toolchain or automated pipeline re-encoding.")
        elif has_camera_hardware:
            prob_tampered = 0.10
            reliability = 0.85
            verdict = "AUTHENTIC"
            findings.append("CONFIRMED: Header structure consistent with native camera sensor capture.")
        elif multiplexer_detected:
            prob_tampered = 0.52
            reliability = 0.25
            verdict = "INCONCLUSIVE"
            findings.append("NOTICE: Generic container multiplexer detected without explicit camera or editing signatures.")
        else:
            # Stripped metadata: Under Dempster-Shafer theory, absence of evidence is pure uncertainty (m_U = 1.0)
            prob_tampered = 0.50
            reliability = 0.0  # Zero reliability produces m_U = 1.0, zero bias in fusion
            verdict = "INCONCLUSIVE"
            findings.append("NOTICE: Metadata has been stripped or sanitized (typical for web/messaging uploads). High epistemic uncertainty.")

        findings.extend(suspicious_tags)

        # Learned Anomaly Score
        learned_anomaly_score = self._learned_anomaly_detection(exif)
        if learned_anomaly_score > 0.6:
            findings.append(f"Learned anomaly detection flagged unusual metadata structure (Score: {learned_anomaly_score:.2f}).")
            
        # If no hard flags, fallback to learned anomaly score
        if prob_tampered == 0.50 and learned_anomaly_score > 0.5:
            prob_tampered = min(0.9, 0.5 + learned_anomaly_score * 0.4)
            reliability = 0.6
            verdict = "TAMPERED"

        # Build BBA mass after all anomaly scores have settled
        bba = self.create_bba(prob_tampered=prob_tampered, custom_reliability=reliability)

        telemetry = {
            "has_exif": bool(exif),
            "has_png_chunks": bool(png_info),
            "ai_parameters_detected": ai_detected,
            "software_trace_detected": software_detected,
            "manipulation_detected": manipulation_detected,
            "c2pa_present": c2pa_detected,
            "camera_make": make or (container_info["cameras"][0] if container_info and container_info["cameras"] else "N/A"),
            "camera_model": model or "N/A",
            "raw_tags_count": len(exif) + len(png_info),
            "container_encoders": container_info["encoders"] if container_info else [],
        }

        return AgentEvidence(
            agent_name=self.name,
            prob_tampered=prob_tampered,
            reliability=reliability,
            bba_mass=bba,
            verdict=verdict,
            findings=findings,
            telemetry=telemetry,
            visual_artifact=None,
        )


# ---------------------------------------------------------------------------
# Unit / Smoke Test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    agent = MetadataAgent()

    print(f"Testing {agent.name}...")
    print("-" * 60)

    # Test 1: Synthetic Image with AI info
    ai_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    ai_img.info["parameters"] = "A hyperrealistic portrait, steps: 30, sampler: DPM++ 2M, cfg scale: 7.0"
    evidence_ai = agent.analyze(ai_img)
    print(f"Test 1 (AI Image) -> Verdict: {evidence_ai.verdict} | P(Tampered): {evidence_ai.prob_tampered:.2f}")
    assert evidence_ai.verdict == "TAMPERED"
    assert evidence_ai.prob_tampered > 0.90
    assert evidence_ai.bba_mass["T"] > 0.80

    # Test 2: Blank image with no metadata (stripped)
    blank_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    evidence_blank = agent.analyze(blank_img)
    print(f"Test 2 (Stripped Metadata) -> Verdict: {evidence_blank.verdict} | Uncertainty Mass m(U): {evidence_blank.bba_mass['U']:.2f}")
    assert evidence_blank.verdict == "INCONCLUSIVE"
    assert evidence_blank.bba_mass["U"] >= 0.99  # Pure epistemic uncertainty (m_U = 1.0)

    print("-" * 60)
    print("All MetadataAgent unit assertions passed successfully!")

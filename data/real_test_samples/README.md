# PRISM Curated Evaluation Probes

This directory contains verified benchmark media probes for end-to-end multi-agent evaluation:

| Probe File | Modality | Ground Truth | Forensic Artifacts |
| :--- | :--- | :--- | :--- |
| `01_real_human_video_v1.mp4` | Video (H.264) | AUTHENTIC | Natural camera sensor noise, authentic facial micro-motion |
| `01_real_human_video_v1_authentic_source.mp4` | Video (H.264) | AUTHENTIC | Benchmark ground truth source recording |
| `01_real_human_video_v1_inswapper_faceswap.mp4` | Video (H.264) | TAMPERED | Inswapper autoencoder smoothing, PRNU boundary drop |
| `02_real_human_video_v2.mp4` | Video (H.264) | AUTHENTIC | Real camera mobile phone capture |
| `03_deepfake_face_swap_vs1.mp4` | Video (H.264) | TAMPERED | High PRNU facial noise divergence (0.610), seam boundary |
| `04_deepfake_face_swap_vs2.mp4` | Video (H.264) | TAMPERED | High PRNU facial noise divergence (0.504), optical jitter |
| `05_real_camera_canon_powershot.jpg` | Image (JPEG) | AUTHENTIC | Canon PowerShot hardware EXIF & consistent CFA PRNU |
| `06_real_camera_nikon_coolpix.jpg` | Image (JPEG) | AUTHENTIC | Nikon Coolpix physical sensor fingerprint |
| `07_real_camera_sony_cybershot.jpg` | Image (JPEG) | AUTHENTIC | Sony CyberShot authentic CFA grid |
| `08_whatsapp_stripped_photo.jpg` | Image (JPEG) | AUTHENTIC | Metadata-stripped camera photo; physical sensor noise verified |
| `09_whatsapp_recompressed_crf40_video.mp4` | Video (H.264) | AUTHENTIC | Heavy CRF 40 recompression; verified physical provenance |
| `10_gemini_generative_ai_video.mp4` | Video (H.264) | TAMPERED | Google Generative AI C2PA container manifest & temporal jitter |

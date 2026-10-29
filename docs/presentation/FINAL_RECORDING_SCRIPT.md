# PRISM: Final 7-Minute Video Recording Script
**Target Duration:** Exactly 7 Minutes (420 Seconds)  
**Speakers & Roles:**  
1. **Tanush Bhootra** (Start: 0:00 - 2:15) — Intro, Real-World Problem, Why Hashes & Legacy CNNs Fail, PRISM Overview  
2. **Somil Jain** (Mid: 2:15 - 4:30) — Dual-Stream Backbone (ViT-DoRA + PRNU) & The 8 Forensic Micro-Agents  
3. **Kulsum Malik** (Dempster-Shafer, Results & End: 4:30 - 7:00) — Dempster-Shafer Fusion, Benchmark Results (Table 1), Laptop Hardware Speed (RTX 3050), Heatmap Localization, & Conclusion  

---

## 🎬 PART 1: TANUSH BHOOTRA (0:00 – 2:15)
*Role: Opening, Motivation, The Cryptographic Illusion, Baseline Collapse, System Concept*  
*Duration: ~2 Minutes 15 Seconds | Word Count: ~295 Words*

---

### [0:00 - 0:40] Opening & Greeting
*(On-Screen: Slide 1 - Title Slide)*

**What Tanush Speaks:**  
"Good morning, respected committee members and sir.  
I am Tanush Bhootra, and along with my team members Somil Jain and Kulsum Malik, we are presenting our project: **PRISM**—an evidential multi-agent framework for multimedia deepfake and tampering detection.

Today, generative AI tools allow anyone to clone human voices, swap faces, and fabricate photorealistic videos in seconds. These synthetic media files are now actively exploited for financial fraud, targeted defamation, and political disinformation. Our objective in this project was to design a forensic system that moves beyond blind guessing, authenticates media using real physical camera laws, and explains *why* an asset is authentic or forged."

> 💡 **Bakar / Thought for Tanush:** *Sir ko seedha clear bolo: AI deepfakes banana aasan ho gaya hai, fraud badh raha hai. Humne PRISM banaya hai jo physics ke dam pe fake pakadta hai.*

---

### [0:40 - 1:30] Why Legacy Systems Fail (The Real Problem)
*(On-Screen: Slide 2 - Problem Statement & Motivation)*

**What Tanush Speaks:**  
"To understand why this is challenging, we must examine why existing forensic tools fail.

First is what we call the **Cryptographic Illusion**. Legal evidence protocols like NIST rely on SHA-256 file hashes to prove chain of custody. But a hash only confirms that bits were not corrupted during download. It gives zero assurance about physical reality. It is like a sealed postal envelope—the wax seal is intact, but someone put counterfeit money inside before sealing it! A cryptographically untampered container can easily wrap a fully synthetic deepfake face.

Second, existing AI classifiers like XceptionNet or MesoNet look at raw pixel grids. While they achieve 99% accuracy on the specific benchmark they were trained on, their accuracy collapses from 99% down to 65% when tested on new generators! They overfit to video compression quirks instead of understanding immutable physical principles."

> 💡 **Bakar / Thought for Tanush:** *Do batein pakad lo: Hash se sirf transmission check hota hai, AI face nahi. Aur puraane models train data pe 99% laate hain, par naye fakes pe 65% pe crash ho jaate hain.*

---

### [1:30 - 2:15] The PRISM Solution & Handover
*(On-Screen: Slide 3 - System Architecture)*

**What Tanush Speaks:**  
"To resolve this, we built PRISM. Rather than forcing a single neural network to output a random percentage, PRISM operates like an independent forensic crime lab.  

We deploy eight specialist micro-agents. Each agent evaluates a distinct physical clue—such as eye pupil reflections, dermal heartbeat pulse, 3D lighting angles, and microscopic camera sensor noise.  

To walk you through our dual-stream backbone and how these eight agents work, I will now hand over to Somil."

> 💡 **Bakar / Thought for Tanush:** *Smooth handover do: Bolo hum ek model pe bharosa nahi karte, humne 8 alag detectives banaye hain. Fir bolo: "I now hand over to Somil."*

---

## 🎬 PART 2: SOMIL JAIN (2:15 – 4:30)
*Role: Dual-Stream Representation (ViT-DoRA + PRNU) and The 8 Forensic Micro-Agents*  
*Duration: ~2 Minutes 15 Seconds | Word Count: ~310 Words*

---

### [2:15 - 3:00] Dual-Stream Representation: AI + Hardware Physics
*(On-Screen: Slide 4 - Dual-Stream Representation)*

**What Somil Speaks:**  
"Thank you, Tanush.  
Respected sir, when media enters PRISM, our core architecture processes it through two simultaneous, complementary streams:

The first stream is semantic. We employ an ImageNet-initialized Vision Transformer adapted with **DoRA**—Directional Low-Rank Adaptation. Retraining an 86-million-parameter transformer causes catastrophic forgetting of visual priors. DoRA decouples magnitude from direction, training only 1.24 million parameters—just 1.43% of the network! This makes it extremely lightweight while preserving edge and texture representations.

The second stream is purely physical. Due to microscopic silicon imperfections, every real camera sensor imprints a permanent hardware fingerprint called **PRNU**—Photo-Response Non-Uniformity. We extract this noise using Daubechies wavelets and Wiener filtering. Generative AI models synthesize pixels mathematically; they cannot replicate this physical silicon lattice noise. Spliced face patches immediately reveal missing sensor noise!"

> 💡 **Bakar / Thought for Somil:** *Sir ko batao 2 raste hain: Ek ViT model jo DoRA se sirf 1.2M weights train karta hai (super lightweight), aur dusra camera sensor ka noise (PRNU) jo AI copy nahi kar sakta.*

---

### [3:00 - 4:30] The Eight Forensic Micro-Agents
*(On-Screen: Slide 5 - The Eight Micro-Agents)*

**What Somil Speaks:**  
"Surrounding this dual-stream core, we deploy eight independent micro-agents inspecting distinct physical domains:

1. **The Lighting Agent:** computes 3D light vectors for the face and background. If the lighting angle contradicts by more than 35 degrees, it flags tampering.
2. **The Corneal Eye Glint Agent:** checks pupil specular reflections. Real eyes reflect the same light bulb at stereo angles; deepfakes frequently show mismatched glints originating from different source photos.
3. **The Heartbeat Pulse Agent:** uses rPPG technology to measure subtle facial skin color fluctuations caused by blood circulation across 32 frames. Real humans exhibit a 1-to-2 Hertz pulse; AI-generated faces have a flat, biologically dead signal!
4. **The Lip-Sync Agent:** correlates mouth aspect ratio with acoustic audio energy. Bilabial sounds like 'P' and 'B' physically require closed lips; voice clones often show open lips during these phonemes.
5. In addition, our **Temporal Flow**, **Compression ELA**, **Container Metadata**, and **Spatial Decoder** agents independently check motion continuity, JPEG boundary seams, and file atom trees without label leakage.

Now, to explain how we combine these eight diverse agents using Dempster-Shafer theory, our benchmark results, and hardware speed, I pass the floor to Kulsum."

> 💡 **Bakar / Thought for Somil:** *Ye 4 main points bold bolo: Lighting mismatch, aankhon ki reflection, skin me heartbeat pulse, aur lip-sync! Fir Kulsum ko handover do: "I pass the floor to Kulsum."*

---

## 🎬 PART 3: KULSUM MALIK (4:30 – 7:00)
*Role: Dempster-Shafer Fusion, Benchmark Comparison (Table 1), Laptop GPU Speed (RTX 3050), Heatmap, & Conclusion*  
*Duration: ~2 Minutes 30 Seconds | Word Count: ~340 Words*

---

### [4:30 - 5:15] Decision Fusion: Dempster-Shafer Evidential Theory
*(On-Screen: Slide 6 - Dempster-Shafer Fusion & Zadeh's Paradox)*

**What Kulsum Speaks:**  
"Thank you, Somil.  
A critical question in multi-agent forensics is: how do we combine eight different outputs?  
Most systems use simple score averaging. But averaging is dangerous in forensics! If a video is compressed on WhatsApp, one agent might get degraded evidence. A simple average allows that confused agent to corrupt the whole score.

Instead, PRISM employs **Dempster-Shafer Evidential Theory**. Each agent maps its findings into a Basic Belief Assignment over Authentic, Tampered, or explicit **Epistemic Ignorance**—meaning 'I am uncertain'.  
When agents sharply disagree, classical combination rules can produce degenerate verdicts—known as Zadeh's paradox. PRISM implements a conflict ceiling at 0.99, ensuring mathematical stability and outputting legally calibrated belief-plausibility intervals [Belief, Plausibility] rather than an overconfident guess."

> 💡 **Bakar / Thought for Kulsum:** *Sir ko samjhao: Average nahi karte kyunki ek confused agent sab kharab kar deta hai. Dempster-Shafer 'uncertainty' ko alag rakhta hai aur Zadeh paradox ko 0.99 ceiling se solve karta hai.*

---

### [5:15 - 6:00] Benchmark Results & SOTA Comparison
*(On-Screen: Slide 7 - State-of-the-Art Benchmark Comparison)*

**What Kulsum Speaks:**  
"We evaluated PRISM across four major international benchmark suites with strict subject-disjoint splits, ensuring no individual in training ever appeared in testing.

As shown in Table 1:
- On **FaceForensics++ (C23)**, PRISM attains **99.21% AUC**, matching the best models.
- But the decisive breakthrough is on **Celeb-DF v2** under zero fine-tuning! While XceptionNet collapsed from 99.26% down to 65.27%, PRISM maintained **90.85% AUC** with 76.5% multi-modal accuracy!
- On acoustic speech cloning, our vocoder agent achieved **78.86% AUC**.
Across our verified test cohort of 416 real and manipulated media assets, evidential multi-agent fusion consistently outperformed every single-backbone detector."

> 💡 **Bakar / Thought for Kulsum:** *Table 1 pe ungli ya mouse ghumao aur bolo: Normal data pe 99%, aur sabse tough test (Celeb-DF) pe bina retraining ke 90.85% AUC!*

---

### [6:00 - 6:40] Real-Time Laptop GPU Profiling & Tampering Heatmap
*(On-Screen: Slide 8 - Laptop Speed & Heatmap Localization)*

**What Kulsum Speaks:**  
"Next, we ensured that PRISM can be deployed by real police and forensic investigators without requiring expensive cloud servers.  
We profiled PRISM directly on a standard laptop equipped with an **NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM)**.

PRISM runs with an average inference latency of just **62.76 milliseconds** per frame, consuming only **1,450 MB of VRAM**. This enables a throughput of **24.2 frames per second—true real-time speed**!

Furthermore, as seen in Figure 5, our spatial decoder doesn't just output a percentage. It generates a 224x224 red pixel heatmap pinpointing the exact boundary where the face was swapped or manipulated, providing visual explainability compliant with Federal Rule of Evidence 702."

> 💡 **Bakar / Thought for Kulsum:** *Ye impressive point hai: 62 ms speed on standard RTX 3050 laptop (24 FPS real-time) + Figure 5 ka red pixel heatmap jo court evidence ke liye ready hai!*

---

### [6:40 - 7:00] NIST Compliance, Summary & Conclusion
*(On-Screen: Slide 9 - Conclusion & NIST Compliance)*

**What Kulsum Speaks:**  
"In conclusion, every verification run in PRISM automatically compiles into a tamper-evident, SHA-256 signed JSON and PDF forensic dossier compliant with NIST Special Publication 800-86.  

By combining low-rank vision transformers, physical camera sensor noise, and Dempster-Shafer evidential fusion, PRISM overcomes the catastrophic brittleness of deepfake detection. Our complete codebase, model weights, and 87 automated regression tests are verified.

Thank you so much, respected sir and committee members, for your time and guidance. We are now happy to take your questions."

---

## 🎯 Final Quick Tips for a High-Scoring Video:
1. **Pronunciation Guide:**
   - *DoRA* = "Doh-RA" (Directional Low-Rank Adaptation)
   - *PRNU* = "P-R-N-U" (Photo-Response Non-Uniformity)
   - *rPPG* = "R-P-P-G" (Remote photoplethysmography / pulse)
   - *Dempster-Shafer* = "Demp-ster Shay-fer"
   - *Zadeh's paradox* = "Zah-day's paradox"
2. **Pacing:** Pause for 1 second after saying key figures like **90.85%**, **62.7 ms**, and **RTX 3050**.
3. **Screen Setup:** Open [`PRISM_Final_Presentation.pptx`](file:///D:/PRISM/video/PRISM_Final_Presentation.pptx) or [`PRISM_Final_Presentation.html`](file:///D:/PRISM/video/PRISM_Final_Presentation.html) on full screen.

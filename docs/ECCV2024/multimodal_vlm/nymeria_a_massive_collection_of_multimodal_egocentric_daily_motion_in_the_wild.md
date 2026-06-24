---
title: >-
  [Paper Note] Nymeria: A Massive Collection of Multimodal Egocentric Daily Motion in the Wild
description: >-
  [ECCV 2024][Multimodal VLM][Egocentric perspective] Nymeria is currently the world's largest in-the-wild human motion dataset (300 hours, 264 participants), providing synchronized and co-localized multi-device multimodal egocentric data (Project Aria glasses + wristbands + motion capture suits) for the first time, accompanied by 310.5K hierarchical motion-language descriptions.
tags:
  - "ECCV 2024"
  - "Multimodal VLM"
  - "Egocentric perspective"
  - "human motion dataset"
  - "multimodal"
  - "motion-language"
  - "daily activities"
date: 2026-05-08
content_hash: e7b9c7fdc0a27133
---

# Nymeria: A Massive Collection of Multimodal Egocentric Daily Motion in the Wild

**Conference**: ECCV 2024  
**arXiv**: [2406.09905](https://arxiv.org/abs/2406.09905)  
**Code**: [https://www.projectaria.com/datasets/nymeria](https://www.projectaria.com/datasets/nymeria)  
**Area**: Multimodal VLM  
**Keywords**: Egocentric perspective, human motion dataset, multimodal, motion-language, daily activities

## TL;DR

Nymeria is currently the world's largest in-the-wild human motion dataset (300 hours, 264 participants), providing synchronized and co-localized multi-device multimodal egocentric data (Project Aria glasses + wristbands + motion capture suits) for the first time, accompanied by 310.5K hierarchical motion-language descriptions.

## Background & Motivation

**Background**: With the rise of smart glasses and wearable devices (e.g., Meta, Google Glass, Apple Vision Pro), human-centric contextualized AI leveraging the user's own motion as context has become a vital research direction. However, current research is constrained by data: the scale, diversity, and modalities of real-world data are limited, while simulated data lacks realism.

**Limitations of Prior Work**:
   - **Difficulty in acquiring in-the-wild ground truth**: Optical marker-based motion capture is limited by line-of-sight occlusion and restricted to confined spaces; inertial motion capture suffers from accumulated drift.
   - **Difficulty in multi-device alignment**: Temporal and spatial alignment of different devices requires high-precision synchronization. Existing solutions rely on visual or audio cues, which have limited accuracy and reliability, and require frequent interruptions during collection to correct clock drift.
   - **Sparse language annotations**: Existing motion-language datasets feature brief descriptions and lack scene context, with scales far below what is required for LLM training.
   - Existing datasets are either small-scale, indoor-only, lack parametric human representations, or lack a first-person perspective.

**Key Challenge**: The "under-observation" of full-body motion from an egocentric perspective (only hands and partial body are visible) makes motion estimation inherently under-determined. This necessitates massive real-world data to learn motion priors, but existing datasets are highly insufficient in scale, diversity, and modality richness.

**Goal**: Construct a human motion dataset that substantially surpasses existing ones in multiple dimensions, encompassing: in-the-wild full-body motion capture + multimodal egocentric data + precisely synchronized co-localization + large-scale hierarchical language descriptions.

**Key Insight**: Leverage a multi-device combination consisting of an XSens inertial mocap suit, Project Aria smart glasses, and custom miniAria wristbands, paired with custom sub-millisecond synchronization hardware and globally optimized alignment algorithms.

**Core Idea**: Through a comprehensive integration of hardware and software innovations, collect co-localized multimodal egocentric daily motion data at an unprecedented scale, and build multi-level language annotations ranging from fine-grained motion narrations to activity summarizations.

## Method

### Overall Architecture

Nymeria dataset construction pipeline: Hardware design & synchronization $\rightarrow$ Data collection (20 scenarios, 264 participants, 50 locations) $\rightarrow$ Data processing (motion retargeting + SLAM localization + global alignment) $\rightarrow$ Hierarchical language annotation $\rightarrow$ Benchmarking.

### Key Designs

1. **Multi-device Synchronization and Co-localization System**:

   **Function**: Achieves sub-millisecond level synchronization via a unified time signal across an XSens mocap suit (17 inertial sensors), Project Aria glasses (RGB + grayscale + eye-tracking cameras + IMUs, etc.), and two miniAria wristbands, and co-localizes them into a unified 3D coordinate system using SLAM and optimization.

   **Mechanism**: 
    - A custom synchronization device serves unified timestamps to all devices, achieving an alignment accuracy of $\le 1$ motion frame ($4.2\text{ms}$) between XSens and Aria.
    - Project Aria MPS provides millimeter-level 6DoF localization (VIO + SLAM + visual-inertial bundle-adjustment).
    - Align the drifting XSens trajectory with the precise MPS trajectory via HandEye calibration: assuming a constant rigid transformation $T_{HD}$ from head to device, the trajectory is segmented into short $4.2\text{ms}$ segments to optimize local velocity matching.

   **Design Motivation**: The value of multi-device data depends on the alignment accuracy. Commercial devices lack unified synchronization protocols, and visual or audio cues in existing solutions are not reliable enough. Custom synchronization hardware combined with optimization algorithms is the only way to sustain high precision during long-duration in-the-wild captures.

2. **Full-body Motion Retargeting**:

   **Function**: Retargets XSens skeletal motion to a parametric human model with 159 joints.

   **Mechanism**: Solve an inverse kinematics optimization problem utilizing the global positions of 79 anatomical landmarks output by XSens:

    $$\arg\min_{\phi, \theta_{0\cdots T-1}, \mathbf{v}^{0\cdots K-1}} \sum_{t=0}^{N-1} \sum_{i=0}^{K-1} \|T^i(\phi, \theta_t)\mathbf{v}^i - \mathbf{p}_t^i\|^2$$

   where $\phi$ represents identity parameters (global scale + bone lengths), $\theta$ represents pose parameters (6DoF global + 52 Euler angles), and $\mathbf{v}^i$ denotes landmark local offsets. The problem is solved using the Levenberg-Marquardt algorithm, incorporating parameter constraints and body collision penalties.

   **Design Motivation**: The simplified skeletal model of XSens exhibits issues like self-penetration and unnatural poses. By employing an anatomically-inspired human model combined with collision-constrained optimization, a more realistic motion representation is obtained.

3. **Hierarchical Motion-Language Description**:

   **Function**: Provides natural language annotations at three levels of granularity for motion data.

   **Mechanism**: 25 annotators concurrently viewed three synchronized perspectives—**egocentric video + third-person video + 3D motion rendering**—to annotate across three levels from fine to coarse granularity:
    - **Motion Narration**: $\le 5$-second clips, detailing full-body poses, arm/leg movements, and attention direction.
    - **Atomic Action**: $\le 5$-second clips, succinctly describing actions using verbs.
    - **Activity Summarization**: 30-second clips, summarizing the main activity in a single sentence.

   **Design Motivation**: 
    - Synchronized playback of three perspectives enables annotators to gain a comprehensive understanding of the motion (first-person for hand interactions, third-person/3D rendering for the entire body).
    - Hierarchical annotation supports varied research needs, such as fine-grained motion generation, action recognition, and activity understanding.
    - Contextual descriptions containing object and environmental information (rather than isolated action labels) are more aligned with the needs of the LLM era.

### Data Specifications and Statistics

- **Scale**: 300 hours of daily activities, 1,200 sequences, average 15 minutes/sequence.
- **Participants**: 264 individuals (48.5% female, 51.4% male), diverse ethnicities.
- **Scenes**: 20 activity categories (18 indoor + 5 outdoor ones), 50 locations (47 residential + 3 campus areas).
- **Sensor Data**: 260M body poses, 201.2M images, 11.7B IMU samples, 10.8M gaze points.
- **Trajectories**: Head 399 km + wrist 1,053 km.
- **Language**: 310.5K sentences, 8.64M words, 6,545 vocabulary size.

## Key Experimental Results

### Main Results — Comparison with Existing Datasets

| Dataset | Duration(h) | Pose Frames(M) | Avg Sequence Length(min) | Participants | Language Sentences(K) | Vocabulary | Head-worn | Wrist-worn | Outdoor | Gaze |
|-------|--------|----------|--------------|-------|-----------|-------|------|------|------|------|
| AMASS | 42 | 0.9 | 0.22 | 346 | - | - | ✗ | ✗ | ✗ | ✗ |
| HPS | 4.5 | 0.5 | 8.2 | 7 | - | - | ✓ | ✗ | ✓ | ✗ |
| EgoBody | 2 | 0.4 | 1 | 36 | - | - | ✓ | ✗ | ✗ | ✓ |
| HumanML3D | 28.6 | 2.9 | 0.12 | - | 45.0 | 5371 | ✗ | ✗ | ✗ | ✗ |
| EgoExo4D | 88.8 | 9.6 | 2.6 | 740 | 432 | 4405 | ✓ | ✗ | ✓ | ✗ |
| **Nymeria** | **300** | **260** | **15** | **264** | **310.5** | **6545** | **✓** | **✓** | **✓** | **✓** |

### Baseline Evaluations — Tracking/Generation/Language

| Task | Method | Data | MPJPE(cm)↓ | Hand PE(cm)↓ | MPJVE(cm/s)↓ | FID↓ |
|------|------|------|-----------|-------------|-------------|------|
| 3-point tracking | AvatarPoser | AMASS | 4.20 | 2.34 | 28.23 | - |
| 3-point tracking | AvatarPoser | Nymeria(real) | 7.97 | 6.25 | 16.71 | - |
| 3-point tracking | AvatarPoser | Nymeria(synth) | 7.31 | 3.47 | 16.63 | - |
| 3-point generation | BoDiffusion | AMASS | 3.63 | - | - | - |
| 3-point generation | BoDiffusion | Nymeria | 7.98 | - | - | 2.32 |
| 1-point generation | EgoEgo | Nymeria | 13.22 | - | - | 5.14 |
| VQ-VAE | 2PQ-16384CB | Nymeria | 3.449cm(mm) | - | - | - |

### Key Findings

- **Nymeria's motion is more challenging**: AvatarPoser's MPJPE rises from 4.20 cm on AMASS to 7.97 cm on Nymeria due to complex movements like climbing stairs, sports, and uneven terrains.
- **Small gap between real vs. synthetic inputs**: real (7.97) vs. synthetic (7.31) demonstrates the high tracking quality of the hardware.
- **3-point > 1-point**: BoDiffusion (7.98) vs. EgoEgo (13.22) shows that wristband data offers critical physical constraints.
- **Motion tokenization is feasible**: VQ-VAE's reconstruction quality on Nymeria is close to AMASS, supporting LLM-style motion generation.
- **Initial exploration of motion-to-text**: MotionGPT on a Nymeria subset achieves BLEU@1=42.22 and CIDEr=37.27. Since the description complexity is higher than in HumanML3D, its performance is lower than in the original paper, but it indicates the data is sufficient to train viable models.
- **An average of 27.8 words per sentence**, far exceeding the description length of existing motion-language datasets.

## Highlights & Insights

1. **Engineering feat**: 264 participants, 50 locations, 300 hours, and sub-millisecond multi-device synchronization. The dataset construction itself is a monumental technical and operational achievement.
2. **Unprecedented modal completeness**: Parametric full-body motion, egocentric RGB/grayscale/eye-tracking, wristbands, IMUs, gaze, 3D scene point clouds, and 6DoF localization—all precisely synchronized.
3. **Elaborate hierarchical language annotation design**: Tri-perspective synchronized annotation ensures quality, and the granularity hierarchy from motion narration to activity summarization supports various research directions.
4. **20 scenarios cover rich daily routines**: Ranging from cooking, fitness, and hiking to party preparation and Simon Says—not just generic walking/running.
5. **Ingenious global alignment algorithm**: Successfully applies the HandEye calibration formulation to align the drifting XSens trajectory with the SLAM trajectory.

## Limitations & Future Work

1. **Mocap suits affect naturalness**: Wearing inertial mocap suits and wristbands alters appearance and restricts certain ranges of motion.
2. **XSens accuracy is sensitive to calibration and body shape measurements**: Inaccurate shape measurements lead to self-penetrating or unnatural poses.
3. **Incomplete scene coverage**: Lacks public spaces (e.g., buses, stores, hospitals). The 15% proportion of outdoor activities is relatively low.
4. **No fine-grained hand motion**: Unlike Motion-X, Nymeria does not capture individual fingers or facial expressions.
5. **High cost of language annotation**: The 25 annotators only completed fine-grained annotations for a fraction of the data (motion narrations cover only 38.6h out of 300h).
6. **Privacy restrictions**: De-identification (face blurring) restricts applicability for face-related research.

## Related Work & Insights

- **AMASS**: A pioneering large-scale motion dataset that unifies multiple optical motion capture sources into SMPL, but lacks scene context and an egocentric perspective—Nymeria fills this gap.
- **EgoExo4D**: A large-scale egocentric + third-person dataset, but lacks parametric full-body motion and wristband data—Nymeria provides the complete motion ground truth.
- **Motion-X**: A large-scale full-body motion dataset (including face/fingers), but derived from monocular video estimation which is less precise than inertial motion capture.
- **HumanML3D**: The most prominent motion-language dataset, but its scale and description complexity are far below Nymeria.
- **Insights**: The core value of a dataset paper lies in filling blank dimensions, achieving extreme scale, and providing comprehensive baseline evaluations.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Multiple "firsts": the largest in-the-wild motion dataset, the first multi-device synchronized egocentric dataset, and the largest motion-language dataset.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Mentions four baseline task types (tracking, generation, VQ-VAE, motion-to-text) but with only 1-2 methods evaluated per category.
- **Writing Quality**: ⭐⭐⭐⭐ The dataset paper is well-structured with thorough statistics, though it is quite long.
- **Value**: ⭐⭐⭐⭐⭐ Open-source dataset + complete toolchain, which will significantly drive the field of egocentric motion understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] LifeEval: A Multimodal Benchmark for Assistive AI in Egocentric Daily Life Tasks](../../CVPR2026/multimodal_vlm/lifeeval_a_multimodal_benchmark_for_assistive_ai_in_egocentric_daily_life_tasks.md)
- [\[ECCV 2024\] FreeMotion: MoCap-Free Human Motion Synthesis with Multimodal Large Language Models](freemotion_mocap-free_human_motion_synthesis_with_multimodal_large_language_mode.md)
- [\[ACL 2025\] Donate or Create? Comparing Data Collection Strategies for Emotion-labeled Multimodal Social Media Posts](../../ACL2025/multimodal_vlm/donate_or_create_comparing_data_collection.md)
- [\[NeurIPS 2025\] Reading Recognition in the Wild](../../NeurIPS2025/multimodal_vlm/reading_recognition_in_the_wild.md)
- [\[ICLR 2026\] Figma2Code: Automating Multimodal Design to Code in the Wild](../../ICLR2026/multimodal_vlm/figma2code_automating_multimodal_design_to_code_in_the_wild.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] MOSPA: Human Motion Generation Driven by Spatial Audio
description: >-
  [NeurIPS 2025][Human Understanding][Spatial audio] This work introduces the novel task of spatial-audio-driven human motion generation, constructs the SAM dataset comprising 9+ hours, 27 scenes…
tags:
  - "NeurIPS 2025"
  - "Human Understanding"
  - "Spatial audio"
  - "motion generation"
  - "diffusion model"
  - "SAM dataset"
  - "binaural audio"
date: 2026-05-08
content_hash: c4b4fd209ffd54e4
---

# MOSPA: Human Motion Generation Driven by Spatial Audio

**Conference**: NeurIPS 2025
**arXiv**: [2507.11949](https://arxiv.org/abs/2507.11949)  
**Code**: Available (public project page)  
**Area**: Human Motion Generation / Spatial Audio
**Keywords**: Spatial audio, motion generation, diffusion model, SAM dataset, binaural audio

## TL;DR

This work introduces the novel task of spatial-audio-driven human motion generation, constructs the SAM dataset comprising 9+ hours, 27 scenes, and 12 subjects with paired binaural audio and motion data, and proposes the MOSPA diffusion model. By fusing audio features including MFCC, tempogram, and RMS with sound source position and motion style conditions, MOSPA achieves an FID of 7.98, substantially outperforming music/dance baselines such as EDGE (14.0) and POPDG (21.0).

## Background & Motivation

**Background**: Conditional motion generation has been widely explored across tasks such as text-to-motion (MDM, MoMask), music-to-dance (EDGE, Bailando), and speech-to-gesture (GestureDiffuCLIP). These methods extract semantic and rhythmic information from audio to drive motion, yet all neglect the spatial properties of sound — direction, distance, and intensity.

**Limitations of Prior Work**: Human responses to sound are naturally driven by spatial perception — a sharp sound from the left prompts one to cover the ear and dodge rightward, while gentle music from the front may draw one closer. Existing music/speech-to-motion methods encode only temporal semantic features, rendering them incapable of modeling such spatial dependencies. More critically, no dedicated spatial-audio–motion paired dataset exists for training and evaluating such models.

**Key Challenge**: Spatial audio encodes not only semantics (what sound) but also spatial properties (where it comes from, how loud it is), the latter of which significantly influences human body motion. However, this spatial information is entirely absent from existing audio feature extraction and motion generation pipelines. Furthermore, individual variation in responsiveness to the same sound (ranging from sluggish to sensitive) also requires explicit modeling.

**Goal**: (1) Construct the first spatial-audio–motion paired dataset; (2) Design a generative model that leverages both semantic and spatial features of spatial audio to synthesize plausible human motion.

**Key Insight**: The paper adopts binaural audio — the form of spatial audio closest to human auditory experience — which naturally encodes sound source direction and distance via inter-aural signal differences. MFCC captures temporal semantics, RMS energy encodes distance information, and motion style labels (sluggish / neutral / sensitive) control response intensity.

**Core Idea**: Replace conventional audio features with binaural spatial audio features that incorporate sound source position as the diffusion model's conditioning signal, enabling spatially-aware human motion generation.

## Method

### Overall Architecture

MOSPA is built upon a diffusion model (DDPM). The inputs are binaural spatial audio features $\mathbf{a}$, sound source position $\mathbf{s}$, and motion style $g$; the output is a SMPL-X human motion sequence of $T=240$ frames (30 fps, 8 seconds). The denoiser $\mathcal{G}$ directly predicts the clean motion $\hat{\mathbf{x}_0} = \mathcal{G}(\mathbf{x}_t, t; \mathbf{a}, \mathbf{s}, g)$ (rather than predicting noise), implemented with an encoder-only Transformer architecture.

### Key Designs

1. **Binaural Spatial Audio Feature Extraction**:

    - Function: Extract feature vectors from binaural audio that jointly encode semantic, temporal, and spatial information.
    - Mechanism: A 1136-dimensional feature vector is extracted independently for each ear, comprising: MFCC and its delta (spectral envelope and dynamic variation, 40 dims), constant-Q and STFT chromagram (pitch information, 24 dims), onset strength and tempogram (rhythm and beat, 1069 dims), RMS energy $E_{\text{rms}}$ (signal intensity → proxy for distance, 1 dim), and active frame indicator $F_{\text{active}} = E_{\text{rms}} > 0.01$ (1 dim). The left and right ear features are concatenated to yield $\mathbf{a} \in \mathbb{R}^{T \times 2272}$. The critical role of RMS is that a closer sound source produces higher RMS, and inter-aural RMS differences implicitly encode directional information.
    - Design Motivation: Conventional music-to-dance methods primarily use MFCC and beat features. This work additionally introduces RMS energy to capture distance information, and preserves spatial differences through binaural concatenation — a key component for modeling "where the sound comes from."

2. **Motion Representation and Residual Feature Fusion**:

    - Function: Enhance motion generation accuracy through a redundant representation combining position, rotation, and velocity.
    - Mechanism: Each motion vector $\mathbf{x}$ contains global positions $\mathbf{p} \in \mathbb{R}^{T \times 75}$, local rotations in 6D format $\mathbf{r} \in \mathbb{R}^{T \times 150}$, and joint velocities $\mathbf{v} \in \mathbb{R}^{T \times 75}$ for $J=25$ SMPL-X joints, yielding 300 dimensions per frame. Velocity serves as residual information to help the model capture subtle spatial-audio-induced motion variations such as changes in reaction speed.
    - Design Motivation: Rotation-only representations are compact but lose global displacement information, which is critical for spatial audio responses (i.e., knowing which direction the character moves). The redundant position-velocity features provide more direct supervision signals for the model.

3. **Condition Fusion and Transformer Denoiser**:

    - Function: Fuse multiple conditioning signals to guide the diffusion denoising process.
    - Mechanism: Timestep $t$, motion vector $\mathbf{x}_t$, audio features $\mathbf{a}$, sound source position $\mathbf{s}$, and motion style $g$ are each projected to a shared latent dimension (512) via independent FFNs. Random masking is applied to $\mathbf{a}$ and $\mathbf{s}$ (analogous to classifier-free guidance), and the results are concatenated into a complete token sequence $\mathbf{z}$. After adding positional encoding, the sequence is fed into an encoder Transformer (4 layers, 8 heads, 512 dims). The final $T$ tokens are decoded via an FFN to produce the predicted clean motion $\hat{\mathbf{x}_0}$.
    - Design Motivation: Random condition masking enables guidance strength control during inference; the self-attention mechanism of the encoder-only Transformer allows free cross-temporal interaction between audio and motion tokens.

### Loss & Training

The total loss comprises five terms: $\mathcal{L} = \lambda_{\text{data}}\mathcal{L}_{\text{data}} + \lambda_{\text{geo}}\mathcal{L}_{\text{geo}} + \lambda_{\text{foot}}\mathcal{L}_{\text{foot}} + \lambda_{\text{traj}}\mathcal{L}_{\text{traj}} + \lambda_{\text{rot}}\mathcal{L}_{\text{rot}}$

- $\mathcal{L}_{\text{data}}$: MSE between predicted and ground-truth motion + MSE of inter-frame variation (temporal smoothness).
- $\mathcal{L}_{\text{geo}}$: Joint position MSE after FK forward kinematics + velocity MSE (physical consistency).
- $\mathcal{L}_{\text{foot}}$: Foot contact consistency loss (preventing foot sliding).
- $\mathcal{L}_{\text{traj}}$ and $\mathcal{L}_{\text{rot}}$: Emphasis losses on trajectory and rotation (accelerating convergence and improving directional accuracy).

Training strategy: All $\lambda$ are initialized to 1; at epochs 5000 and 6000, $\lambda_{\text{traj}}$ and $\lambda_{\text{rot}}$ are increased to 3 to emphasize late-stage trajectory and rotation accuracy. A 1000-step cosine noise schedule is used with AdamW optimizer (lr=$10^{-4}$), batch size 128, training for approximately 18 hours on a single RTX 4090.

## Key Experimental Results

### Main Results

Quantitative evaluation on the SAM dataset (8:1:1 split, 2400/300/300 sequences):

| Method | R-prec Top1↑ | R-prec Top3↑ | FID↓ | Diversity→ | APD→ |
|--------|-------------|-------------|------|-----------|------|
| Real Motion | 1.000 | 1.000 | 0.001 | 23.62 | 59.44 |
| Bailando | 0.077 | 0.182 | 168.4 | 17.35 | 23.12 |
| LODGE | 0.444 | 0.679 | 102.3 | 21.10 | 11.80 |
| POPDG | 0.762 | 0.934 | 21.0 | 22.54 | 35.00 |
| EDGE | 0.886 | 0.977 | 14.0 | 23.10 | 43.88 |
| **MOSPA** | **0.937** | **0.996** | **7.98** | **23.58** | **53.92** |

### Ablation Study

| Configuration | FID | R-prec Top1 | Note |
|--------------|-----|-------------|------|
| Full model (512d, 8h, 1000 steps, +genre) | **7.98** | **0.937** | Optimal configuration |
| 256d | 9.23 | 0.891 | Lower dimensionality degrades quality |
| 4 heads | 9.28 | 0.923 | Fewer heads mildly affect performance |
| 100 diffusion steps | 8.46 | 0.930 | Slight quality drop with fewer steps |
| 4 diffusion steps | 8.39 | 0.934 | 4 steps remain competitive |
| w/o genre condition | 10.93 | 0.889 | Genre is important for response intensity modeling |
| w/o MFCC | 9.07 | 0.907 | MFCC is critical for semantic modeling |
| w/o tempogram | 10.79 | 0.917 | Tempogram has the largest impact on quality |

### Key Findings

- **MOSPA comprehensively outperforms all baselines**: FID of 7.98 is less than half that of EDGE (14.0), and R-precision Top1 exceeds it by 5.1 percentage points. Diversity and APD are closest to real motion.
- **Bailando and LODGE fail dramatically**: FIDs of 168 and 102, respectively, as they are designed for music or long sequences and cannot handle abruptly changing spatial audio.
- **Genre condition is critical**: Removing it degrades FID by 37% (10.93 vs. 7.98), indicating that response intensity modeling is essential for spatial-audio-driven motion generation.
- **Tempogram contributes most**: Its removal degrades FID by 35% (10.79 vs. 7.98), demonstrating the importance of beat and rhythm information for motion timing.
- **Diffusion steps can be substantially reduced**: 4-step diffusion still yields a competitive FID of 8.39, suggesting feasibility for real-time applications.
- User study (25 participants): MOSPA receives the most votes across all three criteria — intent alignment, motion quality, and similarity to ground truth.

## Highlights & Insights

- **Pioneering definition of a new task with a dedicated dataset**: The spatial audio-to-motion direction was entirely unexplored prior to this work. The SAM dataset — spanning 27 scenes, 3 response intensities, 16 sound source positions, and 12 subjects — is thoughtfully designed and establishes a benchmark for future research. The data collection pipeline (Vicon motion capture + synchronized dual-microphone recording + Deity PR-2 recorder) is reproducible.
- **Elegant use of RMS energy as a distance proxy**: Among all audio features, RMS energy and the active frame indicator are the simplest, yet they encode precisely what distinguishes spatial audio from conventional audio — distance and loudness. Inter-aural RMS differences further implicitly encode directionality.
- **Three-tier motion style classification is intuitive and effective**: Categorizing human responsiveness to sound into sluggish / neutral / sensitive is an intuitively sound modeling choice. The genre ablation confirms its importance, while simultaneously providing a natural control interface for generating diverse motions.
- **High practical value from a VR/gaming perspective**: Generated virtual characters that react to spatial sounds with directional awareness (e.g., fleeing in the opposite direction upon hearing a gunshot) represent a foundational capability for immersive experiences.

## Limitations & Future Work

- **Lack of physical constraints**: Generated motions may contain physically implausible poses (e.g., ground penetration, unbalanced stances), as no physics simulation engine (e.g., Isaac Gym) is integrated for constraint enforcement.
- **Body motion only**: Finger and facial motions — both supported by SMPL-X and relevant to fine-grained sound responses (e.g., finger details when covering ears, expressions of fear) — are excluded.
- **No scene environment awareness**: Motion generation is not conditioned on surrounding objects or scene geometry, precluding environment-interactive motions (e.g., ducking behind a table upon hearing a sound).
- **Limited dataset scale**: 9 hours of motion data is relatively small for deep generative models, potentially constraining generalization. Expanding to more scenes and subjects would improve robustness.
- **Limitations of evaluation metrics**: R-precision and FID rely on a trained bidirectional GRU feature extractor; these metrics may fail to capture spatial correctness (e.g., whether the character moves in the correct direction relative to the sound source).

## Related Work & Insights

- **vs. EDGE / Bailando (music-to-dance)**: These methods exploit only beat and melody information and cannot perceive sound source direction or distance. Their poor adaptation to spatial audio demonstrates that spatial features require dedicated modeling.
- **vs. GestureDiffuCLIP (speech-to-gesture)**: Speech-to-gesture focuses on semantic correspondence (gestures during speech), whereas spatial audio-to-motion concerns whole-body spatial reactions (responding in the direction of the sound source) — fundamentally different problems.
- **vs. prior spatial audio research**: Works such as Sound-of-Pixels and Self-supervised Moving Vehicle Tracking learn spatial information from video/audio but address perception tasks rather than generation. This paper is the first to apply spatial audio to motion generation.
- Inspiration: Future work could incorporate scene awareness and physics simulation for more realistic spatial-audio-responsive motion generation, and extend to robotics — enabling robots to produce spatially-aware reactions based on perceived sounds.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to define the spatial-audio-driven motion generation task and construct a dedicated dataset and method, opening an entirely new research direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comparisons against 4 baselines, multi-dimensional ablations, user study, and OOD testing are fairly comprehensive; however, dedicated evaluation of spatial correctness is absent.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated, dataset construction is described in detail, method diagrams are intuitive, and the appendix is informative.
- Value: ⭐⭐⭐⭐⭐ The contribution of the dataset and task definition exceeds that of the method itself, opening new research directions for VR, robotics, gaming, and related applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] VASA-3D: Lifelike Audio-Driven Gaussian Head Avatars from a Single Image](vasa-3d_lifelike_audio-driven_gaussian_head_avatars_from_a_single_image.md)
- [\[NeurIPS 2025\] VimoRAG: Video-based Retrieval-augmented 3D Motion Generation for Motion Language Models](vimorag_video-based_retrieval-augmented_3d_motion_generation_for_motion_language.md)
- [\[ICCV 2025\] KinMo: Kinematic-Aware Human Motion Understanding and Generation](../../ICCV2025/human_understanding/kinmo_kinematic-aware_human_motion_understanding_and_generation.md)
- [\[CVPR 2026\] UniLS: End-to-End Audio-Driven Avatars for Unified Listening and Speaking](../../CVPR2026/human_understanding/unils_end-to-end_audio-driven_avatars_for_unified_listening_and_speaking.md)
- [\[CVPR 2026\] MoLingo: Motion-Language Alignment for Text-to-Human Motion Generation](../../CVPR2026/human_understanding/molingo_motion-language_alignment_for_text-to-motion_generation.md)

</div>

<!-- RELATED:END -->

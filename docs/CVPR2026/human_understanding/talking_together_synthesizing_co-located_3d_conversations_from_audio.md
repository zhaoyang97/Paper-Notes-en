---
title: >-
  [Paper Note] Talking Together: Synthesizing Co-Located 3D Conversations from Audio
description: >-
  [CVPR 2026][Human Understanding][dyadic conversation] This work presents the first method for generating complete facial animations of two participants **sharing the same 3D physical space** from a single mixed audio str…
tags:
  - "CVPR 2026"
  - "Human Understanding"
  - "dyadic conversation"
  - "3D facial animation"
  - "diffusion model"
  - "co-located space"
  - "eye gaze interaction"
date: 2026-05-08
content_hash: be88393b63358503
---

# Talking Together: Synthesizing Co-Located 3D Conversations from Audio

**Conference**: CVPR 2026  
**arXiv**: [2603.08674](https://arxiv.org/abs/2603.08674)  
**Code**: N/A  
**Area**: Human Understanding  
**Keywords**: dyadic conversation, 3D facial animation, diffusion model, co-located space, eye gaze interaction

## TL;DR

This work presents the first method for generating complete facial animations of two participants **sharing the same 3D physical space** from a single mixed audio stream. It introduces a dual-stream diffusion architecture (shared U-Net + cross-attention), a two-stage mixed-data training strategy, LLM-driven text-to-spatial-layout control, and an auxiliary eye gaze loss to synthesize natural mutual gaze, head turning, and spatially-aware dyadic 3D conversation animations.

## Background & Motivation

- **Limitations of Prior Work**: Existing audio-driven 3D facial animation methods either focus on a single speaker (CodeTalker, FaceFormer, SelfTalk) or generate conversation participants as independent "video-conference-style" avatars (DualTalk), neglecting the critical physical co-location relationships in real face-to-face conversations—relative position, orientation, and mutual gaze.
- **Core Challenges**: (1) Large-scale co-located dyadic 3D conversation data is extremely scarce; (2) Separating each speaker's voice from a single mixed audio stream and modeling the speaking–listening interaction dynamics is non-trivial; (3) Complete 3D animations must include spatial relationships (translation, rotation) and eye contact.
- **Contributions**: The paper proposes the first conversation generation system that explicitly models dyadic 3D spatial relationships, constructs a large-scale dataset covering 2M+ interaction pairs, and achieves a paradigm shift from "video-conference avatars" to "in-room face-to-face dialogue."
- **Applications**: VR/AR remote co-presence, immersive social interaction, virtual agent conversation.

## Method

### Overall Architecture

The system takes a single mixed audio waveform $\mathbf{A} \in \mathbb{R}^T$ (containing both speakers' voices) as input and outputs 3D facial animation parameter sequences for each participant: expression vector $\bm{\psi} \in \mathbb{R}^{L \times 63}$, skeletal pose $\bm{\theta} \in \mathbb{R}^{L \times 4 \times 3}$ (four joints: neck, head, left eye, right eye), and global translation $\mathbf{t} \in \mathbb{R}^{L \times 3}$. Each participant's output is concatenated as $\mathbf{x} = \text{concat}(\psi, \theta, \mathbf{t}) \in \mathbb{R}^{L \times 78}$, with sequence length $L=250$ (10 seconds @ 25fps). The core is a conditional diffusion model with a dual-stream shared U-Net architecture, augmented with cross-attention, speaker role embeddings, and FiLM modulation.

### Key Designs

#### 1. Audio Speaker Masking

- **Function**: Extracts per-participant speaking probability from the mixed audio to provide temporal "who is speaking" guidance.
- **Mechanism**: Uses the Looking to Listen model for source separation, then generates binary speaking probability masks $\mathbf{m}_A, \mathbf{m}_B \in [0,1]^{T \times 1}$ via WebRTC VAD.
- **Design Motivation**: The masks need not be perfectly accurate—slight imprecision acts as beneficial noise, stabilizing training and improving robustness to real conversational overlaps. Masks are precomputed during training and estimated online at inference.

#### 2. Shared Dual-Stream Diffusion Architecture

- **Function**: Processes the noisy inputs of both participants in parallel to generate their respective denoised 3D animation parameters.
- **Mechanism**: A single U-Net backbone with shared weights processes $\mathbf{x}_{t,A}$ and $\mathbf{x}_{t,B}$ in parallel, promoting a unified speaker-agnostic facial motion representation. Bidirectional cross-attention layers are inserted in the decoder to enable information exchange:
$$\mathbf{h}'_A = \text{Attention}(\mathbf{Q}_A, \mathbf{K}_B, \mathbf{V}_B), \quad \mathbf{h}'_B = \text{Attention}(\mathbf{Q}_B, \mathbf{K}_A, \mathbf{V}_A)$$
- **Design Motivation**: The shared backbone ensures consistency between both output streams, while cross-attention enables each stream to model reactive conversational behaviors (e.g., nodding responses, gaze shifts), while preserving individual speaker characteristics.

#### 3. Speaker Role Embedding & Dynamic Conditioning

- **Function**: Encodes the per-frame "speaking/listening" interaction state and provides multi-modal global conditioning.
- **Mechanism**: Two learnable embeddings $\mathbf{e}_{\text{speak}}$ and $\mathbf{e}_{\text{listen}}$ are introduced and linearly interpolated using the speaking probability $\mathbf{m}^{(k)}$ to produce per-frame role vectors:
$$\mathbf{e}_{\text{role}}^{(k)} = \mathbf{m}^{(k)} \mathbf{e}_{\text{speak}} + (1 - \mathbf{m}^{(k)}) \mathbf{e}_{\text{listen}}$$
The conditioning vector $\mathbf{c}^{(k)}$ concatenates Wav2Vec 2.0 audio features $\mathbf{a}^{(k)}$, both participants' role embeddings, and the speaking probability mask, and is injected via two pathways: (1) concatenation with the noisy input; (2) FiLM modulation of intermediate features:
$$\text{FiLM}(\mathbf{h}, \mathbf{c}^{(k)}) = (\bm{\gamma}(\mathbf{c}^{(k)}) + 1) \odot \mathbf{h} + \bm{\beta}(\mathbf{c}^{(k)})$$
- **Design Motivation**: Continuous role embeddings smoothly represent speaking–listening transitions and simultaneous speech states; FiLM modulation enables per-frame adaptive control.

#### 4. Multi-Stage Training

- **Function**: Balances interaction diversity with lip synchronization accuracy.
- **Mechanism**:
    - **Stage 1 (Pretraining)**: Pretrains the dual-stream model on large-scale conversation data (50,000+ hours, 2M+ interaction pairs) to learn natural interaction dynamics (head turning, nodding, facial reactions); the loss covers expressions, rotations, and translation.
    - **Stage 2 (Fine-tuning)**: Fine-tunes on high-quality single-speaker data and a super-resolution-enhanced conversation subset. For single-speaker data, an L2 loss is applied only to the 20 lip/jaw expression parameters of the speaking participant; all other losses are set to zero.
- **Design Motivation**: Online conversation videos suffer from low resolution and occlusion, yielding imprecise lip annotations, whereas single-speaker frontal videos provide accurate lip motion. The two-stage strategy allows the model to first learn interaction dynamics, then refine lip synchronization.
- **Loss Function**: Total loss = expression reconstruction ($\lambda_{expr}=1$) + rotation reconstruction ($\lambda_{rot}=8$) + translation reconstruction ($\lambda_{trans}=1$) + vertex velocity regularization ($\lambda_{vel}=1$) + auxiliary eye gaze loss ($\lambda_{gaze}=5$).

#### 5. Controllable Spatial Relationship via LLM

- **Function**: Allows users to control the 3D spatial relationship between the two participants through natural language descriptions.
- **Mechanism**: During training, the first-frame ground-truth global translations $\mathbf{t}_A^{(0)}, \mathbf{t}_B^{(0)}$ serve as conditioning; the model learns to predict relative displacements $\Delta\mathbf{t}^{(k)} = \mathbf{t}^{(k)} - \mathbf{t}^{(0)}$. At inference, the Gemini LLM maps user text descriptions (e.g., "intimate conversation," "arguing across a table") to 3D coordinates via few-shot prompting.
- **Design Motivation**: Relative displacement normalization simplifies the learning problem; the model only needs to learn "motion patterns given a spatial layout" (including the relationship between head rotation, gaze direction, and absolute position).

#### 6. Auxiliary Eye Gaze Loss

- **Function**: Encourages realistic mutual gaze and gaze avoidance behavior.
- **Mechanism**: Left and right eye rotation parameters are converted to 3D gaze direction vectors; their mean is computed and a cosine similarity loss is applied between predicted and ground-truth gaze directions.
- **Design Motivation**: The key innovation is **selective application**—a higher weight is applied only to the top 20% of conversation samples ranked by head rotation variance, since clips with large head movements are intuitively more likely to contain meaningful gaze interactions (e.g., turning to look at the other person), making the learned gaze patterns more informative.

### Dataset Construction

Two complementary large-scale datasets are constructed:

- **Dyadic Conversation Dataset (50,000+ hours, 10k+ identities)**: Real co-located dyadic conversations are curated from online videos through scene filtering (excluding video-conference split-screen), quality filtering (occluded/blurry/small faces), facial super-resolution enhancement, and 3D facial reconstruction (with temporal smoothing) to obtain complete 3D parameters.
- **Synthetic Dubbing Dataset (50,000+ hours, 10k+ identities)**: Speech segments are randomly sampled and clipped from high-quality single-speaker frontal videos, then alternately concatenated to form pseudo-conversation audio, yielding perfect speaker mask ground truth and high-precision lip motion.

## Key Experimental Results

### Main Results: Quantitative Comparison (Table 2)

| Method | FD ↓ | P-FD ↓ | MSE-FULL ↓ | MSE-ROT ↓ | MSE-EYE ↓ | MSE-LIP ↓ | vMSE-FULL ↓ | SID-SPE ↑ | SID-LIS ↑ |
|------|------|--------|------------|-----------|-----------|-----------|-------------|-----------|-----------|
| CodeTalker | 47.23 | 70.54 | 10.47 | 14.28 | 3.07 | 2.95 | 12.49 | 0 | 0 |
| SelfTalk | 43.58 | 53.98 | 8.21 | 11.59 | 2.47 | 2.41 | 10.98 | 1.68 | 1.27 |
| FaceFormer | 52.66 | 59.84 | 13.89 | 12.34 | 2.96 | 2.84 | 10.47 | 1.59 | 0.43 |
| DualTalk | 28.41 | 38.29 | 9.91 | 8.42 | 2.11 | 2.50 | 8.32 | 1.57 | 1.95 |
| L2L | 38.92 | 66.13 | 11.32 | 10.15 | 2.35 | 2.94 | 11.21 | 1.78 | 1.58 |
| Ours (Single) | 19.58 | 29.03 | 6.32 | 6.74 | 1.23 | 1.14 | 6.86 | 2.23 | 1.40 |
| **Ours** | **10.43** | **18.24** | **4.03** | **3.50** | **0.98** | **0.35** | **7.99** | **2.28** | **2.48** |

### Ablation Study (Table 3)

| Ablation Configuration | FD ↓ | P-FD ↓ | MSE-EXP ↓ | MSE-TRAN ↓ | MSE-ROT ↓ |
|----------|------|--------|-----------|------------|-----------|
| Single-speaker data only | 50.45 | 50.05 | 10.01 | 2.32 | 3.88 |
| w/o Stage 2 (conversation pretraining only) | 60.12 | 64.44 | 7.73 | 1.71 | 2.94 |
| w/o role embeddings | 35.92 | 35.93 | 7.18 | 1.80 | 2.87 |
| w/o cross-attention | 30.49 | 40.87 | 6.87 | 1.54 | 2.98 |
| w/o gaze loss | 37.46 | 42.90 | 7.33 | 2.59 | 2.77 |
| **Full model** | **21.71** | **22.56** | **5.97** | **1.50** | **2.48** |

### Human Evaluation Preference Rate (Table 4, %)

| Method | Lip Sync | Speaker Motion | Listener Motion | Interaction Quality | Gaze Quality |
|------|--------|-----------|-----------|---------|---------|
| SelfTalk | 0.9 | 0.9 | 1.6 | 1.6 | 2.4 |
| DualTalk | 3.9 | 6.3 | 7.2 | 5.6 | 7.9 |
| Ours (Stage 1 only) | 15.9 | 19.0 | 18.2 | 21.4 | 21.4 |
| **Ours** | **79.3** | **73.8** | **73.0** | **71.4** | **68.3** |

### Key Findings

1. **Dominant FD advantage**: The proposed method achieves FD=10.43, less than one-third of the strongest baseline DualTalk (28.41); MSE-LIP=0.35 represents an 86% reduction compared to DualTalk's 2.50.
2. **Both training stages are indispensable**: Removing Stage 2 (conversation data only) causes FD to surge to 60.12; using single-speaker data alone yields FD=50.45—demonstrating that interaction learning and lip refinement must be combined.
3. **Cross-attention is critical for interaction modeling**: Its removal increases P-FD from 22.56 to 40.87, confirming that bidirectional information exchange is essential for capturing speaking–listening reactions.
4. **Gaze loss significantly improves spatial awareness**: Its removal increases MSE-TRAN from 1.50 to 2.59, indicating that gaze constraints indirectly improve overall spatial position prediction.
5. **Overwhelming human preference**: The proposed method achieves preference rates exceeding 68% on all five dimensions, with lip synchronization preference reaching 79.3%.

## Highlights & Insights

1. **Paradigm shift from "video conference" to "in-room conversation"**: The paper is the first to explicitly model co-located 3D spatial relationships (relative position, orientation, mutual gaze)—the core element overlooked by all prior conversation generation methods.
2. **Dataset engineering of significant value**: The two complementary datasets are cleverly designed—conversation data provides interaction diversity while the synthetic dubbing data provides lip accuracy and perfect mask ground truth; the two-stage training strategy optimally combines both advantages.
3. **Selective application of gaze loss**: Applying higher weights only to the top 20% of samples by head motion variance is an elegant design that avoids learning meaningless gaze patterns from static or low-quality clips.
4. **LLM-driven spatial control**: The few-shot approach of mapping text descriptions to 3D coordinates is concise and effective, providing an elegant controllability interface for generative models.

## Limitations & Future Work

1. The method relies on the quality of source separation and VAD; masks may be unreliable under high-noise or heavily overlapping speech conditions.
2. The 3DMM parameterization limits expressiveness (e.g., micro-expressions, asymmetric expressions); the 63-dimensional expression encoding may be insufficient.
3. Only facial and head animation is modeled; full-body co-located interaction (gestures, body lean, etc.) is not addressed.
4. The LLM-based text-to-spatial-coordinate mapping relies on few-shot prompting, with limited generalization to complex or rare scene descriptions.
5. Training requires substantial compute resources (16×A100, 200K steps), making reproduction costly.

## Related Work & Insights

- **Audio-driven single-speaker 3D avatars**: FaceFormer, CodeTalker, and SelfTalk focus on single-speaker generation without interaction modeling; this work naturally extends to the dyadic setting via a dual-stream architecture with cross-attention.
- **Conversation generation**: DualTalk models two speakers but is limited to a "video-conference" style without spatial relationships; L2L trains separate models per identity and cannot generate both speakers simultaneously—the proposed method unifies speaking and listening roles within a single shared model.
- **Spatially-aware group interaction**: Existing multi-person body motion generation focuses on collision avoidance and gait coordination but lacks high-fidelity facial expressions; this work fills the gap in face-level spatial interaction generation.
- **Insights**: The continuous interpolation strategy for role embeddings (rather than discrete role labels) is worth adopting; the two-stage approach of "learning interactions from noisy data + learning fine details from clean data" has broad applicability.

## Rating

| Dimension | Score (1–10) | Notes |
|------|-------------|------|
| Novelty | 9 | First explicit modeling of co-located 3D dyadic conversation generation; the problem formulation itself is a contribution. |
| Technical Depth | 8 | Dual-stream diffusion + cross-attention + FiLM + two-stage training + gaze loss; modules are well-designed and mutually reinforcing. |
| Experimental Thoroughness | 9 | Comprehensive quantitative comparison (11 baselines), thorough ablation (5 variants), and convincing human evaluation (19 evaluators × 14 groups × 5 dimensions). |
| Engineering Contribution | 9 | The construction pipeline for two large-scale datasets (50,000+ hours each) provides substantial engineering value. |
| Application Potential | 8 | Directly applicable to VR/AR remote co-presence, though computational cost and model complexity may limit practical deployment. |
| **Overall** | **8.6** | Novel problem formulation, complete system design, and strong experimental evidence; a pioneering work in co-located conversation generation. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] PoseSyn: Synthesizing Diverse 3D Pose Data from In-the-Wild 2D Data](../../ICCV2025/human_understanding/posesyn_synthesizing_diverse_3d_pose_data_from_in-the-wild_2d_data.md)
- [\[CVPR 2026\] UniLS: End-to-End Audio-Driven Avatars for Unified Listening and Speaking](unils_end-to-end_audio-driven_avatars_for_unified_listening_and_speaking.md)
- [\[NeurIPS 2025\] VASA-3D: Lifelike Audio-Driven Gaussian Head Avatars from a Single Image](../../NeurIPS2025/human_understanding/vasa-3d_lifelike_audio-driven_gaussian_head_avatars_from_a_single_image.md)
- [\[NeurIPS 2025\] MOSPA: Human Motion Generation Driven by Spatial Audio](../../NeurIPS2025/human_understanding/mospa_human_motion_generation_driven_by_spatial_audio.md)
- [\[AAAI 2026\] Streaming Generation of Co-Speech Gestures via Accelerated Rolling Diffusion](../../AAAI2026/human_understanding/streaming_generation_of_co-speech_gestures_via_accelerated_rolling_diffusion.md)

</div>

<!-- RELATED:END -->

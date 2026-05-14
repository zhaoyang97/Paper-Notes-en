---
title: >-
  [Paper Note] Pixel Motion Diffusion Is What We Need for Robot Control
description: >-
  [CVPR 2026][Image Generation][Pixel motion diffusion] DAWN proposes a two-stage fully diffusion-based framework — a Motion Director that generates dense pixel motion fields as interpretable intermediate representations…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Pixel motion diffusion"
  - "robot control"
  - "vision-language-action"
  - "optical flow representation"
  - "hierarchical diffusion policy"
date: 2026-05-08
content_hash: 8c5d6e740286f4ae
---

# Pixel Motion Diffusion Is What We Need for Robot Control

**Conference**: CVPR 2026
**arXiv**: [2509.22652](https://arxiv.org/abs/2509.22652)
**Code**: [Available](https://eronguyen.github.io/DAWN)
**Area**: Image Generation
**Keywords**: Pixel motion diffusion, robot control, vision-language-action, optical flow representation, hierarchical diffusion policy

## TL;DR

DAWN proposes a two-stage fully diffusion-based framework — a Motion Director that generates dense pixel motion fields as interpretable intermediate representations, and an Action Expert that converts these fields into executable robot action sequences — achieving SOTA on CALVIN (Avg Len 4.00), MetaWorld (Overall 65.4%), and real-world benchmarks, with substantially smaller model capacity and training data than competing methods.

## Background & Motivation

**Background**: Vision-Language-Action (VLA) models leverage large-scale web data to achieve broad generalization, yet remain limited in motion perception and spatial reasoning. Existing motion-guided approaches follow two lines: (1) sparse pixel trajectories (General Flow, FLIP, Track2Act, etc.) that extract motion cues from keypoints or sparse points; (2) future RGB frame prediction (SuSIE, UniPi, VPP, etc.) that uses video diffusion models to generate future observations and then derive actions.

**Limitations of Prior Work**:
- **Insufficient information from sparse trajectories**: Tracking only a small number of keypoints fails to describe full-scene motion, losing critical spatial information in complex manipulation tasks.
- **High cost of RGB frame prediction**: Generating complete video frames in high-dimensional RGB space is computationally expensive and lacks explicit motion structure.
- **Indirect extraction introduces complexity**: Gen2Act first generates video and then tracks pixels to extract motion, introducing unnecessary indirection and error accumulation.

**Key Challenge**: Rather than generating complete RGB frames and indirectly extracting motion, directly predicting dense pixel motion preserves full-scene motion information while greatly reducing generation complexity — pixel motion fields are structurally simpler and more learnable than RGB frames.

**Goal**: DAWN (Diffusion is All We Need) — a two-stage pipeline in which both stages use diffusion models: a Motion Director predicts dense pixel motion fields in latent space, and an Action Expert converts these fields into action sequences, forming a fully trainable, end-to-end, interpretable control pipeline.

## Method

### Overall Architecture

DAWN adopts a dual-diffusion hierarchical structure:

1. **Motion Director (high-level controller)**: Built upon a pretrained latent diffusion model (LDM), it takes multi-view images (static view + gripper view) and language instructions as input, and generates dense pixel motion fields $\mathbf{F}'_{t,k} = [u, v, (u+v)/2]$.
2. **Action Expert (low-level controller)**: Built upon a diffusion Transformer, it jointly encodes pixel motion fields, visual observations, language instructions, and robot states, and denoises to generate executable action sequences.

The two modules are connected via structured pixel motion representations, maintaining modular upgradability while providing an intuitive and interpretable intermediate abstraction.

### Key Designs

#### 1. Dense Pixel Motion Representation

- **Function**: Encodes scene-level motion intent as a structured intermediate representation, bridging high-level language understanding and low-level action generation.
- **Mechanism**: The pixel motion from frame $\mathbf{I}_t$ to $\mathbf{I}_{t+k}$ is defined as $\mathbf{F}_{t,k} = [u, v]$, where $u, v \in \mathbb{R}^{H \times W}$ denote horizontal and vertical displacements, respectively. To reuse pretrained RGB diffusion models, this is encoded as a three-channel image $\mathbf{F}'_{t,k} = [u, v, (u+v)/2]$.
- **Design Motivation**: (1) Dense motion retains more complete scene dynamics than sparse keypoints; (2) three-channel encoding enables direct transfer of RGB-pretrained diffusion models; (3) motion fields are lower-dimensional and more regularly structured than RGB frames, reducing generation difficulty.
- **Training Label Generation**: Ground-truth pixel motion is extracted from frame pairs $(\mathbf{I}_t, \mathbf{I}_{t+k})$ using the RAFT optical flow model.

#### 2. Motion Director Architecture

- **Function**: Conditionally generates dense pixel motion fields.
- **Mechanism**: Built on a pretrained LDM, the current frame's VAE encoding (without added noise) is concatenated with Gaussian noise as U-Net input. Language embeddings (CLIP text encoder), gripper-view embeddings (CLIP visual encoder), and time offset $k$ are injected via cross-attention.
- **Design Motivation**: The noiseless current-frame encoding serves as a spatial alignment condition; zero-initialized weights introduce gripper-view conditioning without disrupting pretrained model behavior at the start of training.
- **Training Strategy**: Only the U-Net denoiser weights are updated; VAE, CLIP, and other pretrained modules are frozen. MSE noise estimation loss is used.

#### 3. Action Expert Architecture

- **Function**: Converts pixel motion fields into low-level executable robot action sequences.
- **Mechanism**: Multimodal encoding → cross-attention conditioned denoising Transformer → iterative denoising to generate action chunks. Encoder configurations: DINOv3 ConvNeXt-S encodes pixel motion and visual observations; T5-small encodes language instructions; a 2-layer MLP encodes robot states.
- **Design Motivation**: (1) A diffusion Transformer, rather than an MLP denoiser, better models complex dependencies among multimodal conditions; (2) freezing visual/text encoders leverages pretrained representations, and training only the denoiser and state encoder reduces data requirements.
- **Loss & Training**: MSE noise estimation loss; action chunks are predicted in action space.

### Training & Inference Pipeline

- **Parallel Training**: The Motion Director and Action Expert can be trained in parallel — the former uses RAFT optical flow as ground truth, the latter uses corresponding GT optical flow and actions. Optionally, the Action Expert can be further fine-tuned on actual Motion Director outputs.
- **Inference Pipeline**: Observation encoding → Motion Director (25 diffusion steps) → pixel motion field → Action Expert denoising → action sequence → execution → observation update → closed-loop.
- **Modular Upgradability**: The two modules can be independently replaced or upgraded, facilitating integration of future stronger visual or control models.

## Key Experimental Results

### Main Results

**CALVIN ABC→D (no external robot data, Table 1)**:

| Method | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Avg Len ↑ |
|--------|--------|--------|--------|--------|--------|-----------|
| Diffusion Policy | 0.40 | 0.12 | 0.03 | 0.01 | 0.00 | 0.56 |
| Robo-Flamingo | 0.82 | 0.62 | 0.47 | 0.33 | 0.24 | 2.47 |
| MoDE | 0.92 | 0.79 | 0.67 | 0.56 | 0.45 | 3.39 |
| RoboUniview | 0.94 | 0.84 | 0.73 | 0.62 | 0.51 | 3.65 |
| Seer-Large | 0.93 | 0.85 | 0.76 | 0.69 | 0.60 | 3.83 |
| VPP | 0.96 | 0.88 | 0.78 | 0.71 | 0.60 | 3.93 |
| Enhanced DP (ours) | 0.82 | 0.67 | 0.53 | 0.41 | 0.35 | 2.78 |
| **DAWN (ours)** | **0.98** | **0.91** | **0.79** | **0.71** | **0.61** | **4.00** |

**CALVIN ABC→D (with external data, Table 2)**:

| Method | Extra Data | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Avg Len ↑ |
|--------|------------|--------|--------|--------|--------|--------|-----------|
| GR-1 | Ego4D | 0.85 | 0.71 | 0.60 | 0.50 | 0.40 | 3.06 |
| LTM | OpenX | 0.97 | 0.82 | 0.73 | 0.67 | 0.61 | 3.81 |
| MoDE | Multiple | 0.96 | 0.89 | 0.81 | 0.72 | 0.65 | 4.01 |
| VPP | Multiple | 0.97 | 0.91 | 0.87 | 0.82 | 0.77 | 4.33 |
| DreamVLA | DROID | 0.98 | 0.95 | 0.90 | 0.83 | 0.78 | 4.44 |
| **DAWN (ours)** | DROID | 0.98 | 0.92 | 0.81 | 0.75 | 0.64 | 4.10 |

**MetaWorld 11-task Success Rate (Table 3)**:

| Method | door-open | door-close | basketball | shelf-place | btn-press | faucet-close | hammer | assembly | Overall |
|--------|-----------|------------|------------|-------------|-----------|--------------|--------|----------|---------|
| Diffusion Policy | 45.3 | 45.3 | 8.0 | 0.0 | 40.0 | 22.7 | 4.0 | 1.3 | 24.1 |
| ATM | 75.3 | 90.7 | 24.0 | 16.3 | 77.3 | 50.0 | 4.3 | 2.0 | 52.0 |
| LTM | 77.3 | 95.0 | 39.0 | 20.3 | 82.7 | 52.3 | 10.3 | 7.7 | 57.7 |
| **DAWN (ours)** | **94.7** | **97.3** | **42.0** | **24.7** | **92.0** | **76.3** | **12.7** | **10.7** | **65.4** |

**Real-world lift-and-place experiment (Table 4, 20 random initializations per task)**:

| Method | Apple | Avocado | Banana | Grape | Kiwi | Orange | Inference Latency (ms) |
|--------|-------|---------|--------|-------|------|--------|------------------------|
| Enhanced DP | 5→4 | 6→6 | 5→4 | 4→3 | 5→5 | 4→4 | 112.77 |
| π₀ | 10→9 | 6→6 | 5→3 | 8→5 | 5→3 | 8→7 | 571.89 |
| VPP | 16→14 | 15→15 | 15→14 | 17→17 | 15→15 | 16→14 | 190.55 |
| **DAWN** | **19→19** | **20→19** | **17→16** | **19→19** | **17→16** | **18→16** | 319.82 |

(→ left: number of successful lifts; right: number of successful placements)

### Ablation Study (CALVIN ABC→D, Table 6)

**(a) Pixel motion vs. other intermediate representations**:

| Setting | Avg Len |
|---------|---------|
| No intermediate representation (Action Expert only) | 2.78 |
| RGB target image | 3.21 |
| Pixel motion (U-Net trained from scratch) | 3.42 |
| **Pixel motion (pretrained LDM)** | **4.00** |

**(b) Gripper-view conditioning**:

| Setting | Avg Len |
|---------|---------|
| VPP without gripper view | 3.58 |
| DAWN without gripper view | 3.74 |
| **DAWN with gripper view** | **4.00** |

**(c) Motion Director diffusion steps**:

| Steps | 2 | 10 | 25 | 40 |
|-------|---|----|----|-----|
| Avg Len | 3.88 | 3.96 | **4.00** | 3.95 |

**Bimanual manipulation (Table 5)**: DAWN achieves an action prediction MSE of 0.117 on Galaxea R1-Lite bimanual manipulation, outperforming Enhanced DP (0.128).

### Key Findings

- DAWN achieves SOTA without external data (4.00 > VPP 3.93), demonstrating high data efficiency.
- Pixel motion substantially outperforms RGB target images (4.00 vs. 3.21), and transfer from pretrained LDM further improves performance (4.00 vs. 3.42).
- Strong semantic understanding: the model performs well on semantically similar but distinct task pairs (door-open 94.7% vs. door-close 97.3%).
- Motion Director achieves Avg Len 3.88 with only 2 diffusion steps, indicating that motion information is highly concentrated in the early denoising steps.
- In real-world settings, reliable transfer is achieved with only 1,000 episodes and 100k fine-tuning steps, with near-zero grasp failure rate for DAWN.
- The framework generalizes effectively to bimanual manipulation scenarios.

## Highlights & Insights

1. **Dense motion fields as a universal intermediate language**: Compared to sparse trajectories and RGB frames, dense pixel motion preserves complete spatial information while reducing generation complexity; the three-channel encoding elegantly reuses RGB pretrained weights.
2. **Surprising effectiveness of pretrained transfer**: An LDM trained on RGB images transfers efficiently to pixel motion generation (from-scratch 3.42 vs. pretrained 4.00), suggesting that motion fields and RGB images share meaningful structure in latent space.
3. **Modular design enables high data efficiency**: Freezing pretrained encoders and training only the denoiser allows the model to match or exceed SOTA with significantly smaller model capacity and less training data than VLA methods.
4. **Interpretable intermediate representation**: Pixel motion fields are directly visualizable, enabling users to understand the model's motion intent — a practically valuable property for safe robot deployment.

## Limitations & Future Work

1. Two-stage sequential inference introduces additional latency (319 ms vs. 113 ms for Enhanced DP), limiting real-time applicability.
2. RAFT optical flow used as training labels may introduce noise in scenes with occlusion or large deformations.
3. Performance with external data is lower than VPP/DreamVLA (4.10 vs. 4.33/4.44), leaving room to improve large-scale data utilization.
4. Real-world experiments cover only lift-and-place and single-category bimanual tasks; complex long-horizon manipulation has not been validated.
5. Single-step motion prediction (predicting offset $k$) lacks multi-step planning capability.

## Related Work & Insights

- **Gen2Act / FLIP**: Indirectly extract motion trajectories from generated videos → DAWN bypasses video generation and directly predicts motion, achieving greater efficiency.
- **Diffusion Policy**: End-to-end diffusion action generation without motion intermediate abstraction → DAWN's Action Expert, conditioned on motion, yields a substantial gain (2.78→4.00).
- **VPP**: Extracts video diffusion features in RGB space as implicit motion representations → DAWN replaces these with explicit pixel motion, performing better without external data.
- **π₀**: Large-scale VLA flow-matching model → DAWN substantially outperforms it in real-world settings with a smaller model and less data.
- **Insight**: Pretrained image diffusion models can be viewed as general-purpose visual prediction engines whose capabilities extend well beyond RGB image generation, transferring to structured output tasks such as motion prediction.

## Rating

| Dimension | Score (1–5) | Note |
|-----------|-------------|------|
| Novelty | 4 | Dense pixel motion + dual diffusion pipeline; three-channel encoding for reusing RGB pretrained weights is elegant |
| Technical Depth | 3.5 | Well-engineered system, though the core principle is relatively straightforward (LDM + optical flow GT) |
| Experimental Thoroughness | 4.5 | Three major benchmarks (CALVIN / MetaWorld / real world) + bimanual + detailed ablations |
| Writing Quality | 4 | Clear structure with thorough motivation and comparative analysis |
| Value | 4 | High data efficiency, interpretable, modular, and deployable |
| Overall | 4.0 | A practical framework achieving SOTA with smaller models and less data |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PixelDiT: Pixel Diffusion Transformers for Image Generation](pixeldit_pixel_diffusion_transformers_for_image_generation.md)
- [\[CVPR 2026\] DiP: Taming Diffusion Models in Pixel Space](dip_taming_diffusion_models_in_pixel_space.md)
- [\[CVPR 2026\] Causal Motion Diffusion Models for Autoregressive Motion Generation](causal_motion_diffusion_models_for_autoregressive_motion_generation.md)
- [\[CVPR 2026\] DeCo: Frequency-Decoupled Pixel Diffusion for End-to-End Image Generation](deco_frequency-decoupled_pixel_diffusion_for_end-to-end_image_generation.md)
- [\[CVPR 2026\] Exploring Conditions for Diffusion Models in Robotic Control](exploring_conditions_for_diffusion_models_in_robotic_control.md)

</div>

<!-- RELATED:END -->

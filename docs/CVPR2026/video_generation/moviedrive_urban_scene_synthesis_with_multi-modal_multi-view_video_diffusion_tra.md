---
title: >-
  [Paper Note] MoVieDrive: Urban Scene Synthesis with Multi-Modal Multi-View Video Diffusion Transformer
description: >-
  [CVPR 2026][Video Generation][multi-modal video generation] MoVieDrive proposes a unified multi-modal multi-view video diffusion Transformer that simultaneously generates RGB video, depth maps, and semantic maps within a single model via a two-level modal-shared + modal-specific architecture. Combined with diverse conditioning inputs (text, layout, contextual reference), it achieves FVD 46.8 (SOTA) on nuScenes while producing cross-modally consistent, high-quality driving scene synthesis.
tags:
  - CVPR 2026
  - Video Generation
  - multi-modal video generation
  - multi-view consistency
  - diffusion Transformer
  - urban scene synthesis
  - autonomous driving data augmentation
date: 2026-05-08
content_hash: 9441811f7907e2fa
---

# MoVieDrive: Urban Scene Synthesis with Multi-Modal Multi-View Video Diffusion Transformer

**Conference**: CVPR 2026
**arXiv**: [2508.14327](https://arxiv.org/abs/2508.14327)
**Code**: None
**Area**: Video Generation
**Keywords**: multi-modal video generation, multi-view consistency, diffusion Transformer, urban scene synthesis, autonomous driving data augmentation

## TL;DR

MoVieDrive proposes a unified multi-modal multi-view video diffusion Transformer that simultaneously generates RGB video, depth maps, and semantic maps within a single model via a two-level modal-shared + modal-specific architecture. Combined with diverse conditioning inputs (text, layout, contextual reference), it achieves FVD 46.8 (SOTA) on nuScenes while producing cross-modally consistent, high-quality driving scene synthesis.

## Background & Motivation

**State of the Field**: Video generation models (SVD, CogVideoX) perform well on general video generation, but applying them to autonomous driving requires multi-view spatiotemporal consistency and high controllability. Methods such as DriveDreamer and MagicDrive have explored multi-view urban scene generation, yet support only RGB single-modality output.

**Limitations of Prior Work**: Autonomous driving requires not only RGB video but also depth maps and semantic maps for comprehensive scene understanding. Existing approaches employ **multiple independent models** to generate different modalities separately, resulting in: (a) high deployment complexity; (b) inability to exploit cross-modal complementary information to improve generation quality; and (c) no guarantee of inter-modal consistency.

**Limitations of UniScene**: UniScene attempts to jointly generate RGB and LiDAR but still relies on multiple independent models, falling short of a truly unified multi-modal generation framework.

**Core Assumption**: Different modalities (RGB, depth, semantic) share a common latent space after encoding through a shared 3D VAE, requiring only a small number of modality-specific components to distinguish them—making it feasible for a single unified model to perform multi-modal generation.

## Method

### Overall Architecture

Conditioning inputs → Condition encoders (text / layout / contextual reference) → Unified diffusion Transformer (modal-shared layers + modal-specific layers) → Noise estimation → Shared 3D VAE decoding → Multi-modal multi-view video output.

Core Idea: Multi-modal multi-view scene generation is decomposed into **modal-shared learning** (temporal consistency + multi-view spatiotemporal consistency) and **modal-specific learning** (cross-modal interaction + projection), all handled within a unified framework.

### Key Designs

1. **Diverse Conditioning Input Encoding**:

    - **Text Condition**: Camera intrinsics and extrinsics (Fourier embedding + MLP) are concatenated with scene text descriptions (frozen T5 encoder) to obtain $f^{text}$, which is injected into the diffusion model via cross-attention.
    - **Layout Condition**: Three fine-grained control signals—a box map $c^b$ projected from 3D bounding boxes, a road map $c^r$ projected from road structures, and a layout map $c^o$ projected from sparse 3D occupancy. The key innovation is a **unified layout encoder** (modality-independent causal ResNet blocks + shared causal ResNet blocks) that fuses all three signals, avoiding multiple independent encoders: $f^{layout} = E_s^l(E_b^l(c^b) \otimes E_r^l(c^r) \otimes E_o^l(c^o))$
    - **Contextual Reference Condition**: Optional initial frames encoded by the shared 3D VAE (temporal dimension = 1), used for future scene prediction.
    - **Design Motivation**: Different conditions operate at different granularities—text controls global style, layout controls fine-grained structure, and reference frames provide initial context.

2. **Modal-Shared Components (Temporal Layers + Multi-View Spatiotemporal Blocks)**:

    - **Temporal Attention Layer $D^{tem}$**: Based on CogVideoX's 3D full attention; learns inter-frame temporal consistency with text conditioning injected via cross-attention.
    - **Multi-View Spatiotemporal Block $D^{st}$** (inserted once every $\alpha_1$ temporal layers): Contains four sub-layers:
        - *3D Spatial Embedding Layer*: Encodes 3D occupancy positions $c^{occ}$ with multi-resolution hash grid encoding to enhance spatial consistency.
        - *3D Spatial Attention*: Reshapes the latent to $\mathcal{R}^{K \times (VHW) \times C}$ to learn the 3D spatial structure across all camera views.
        - *Spatiotemporal Attention*: Reshapes to $\mathcal{R}^{(VKHW) \times C}$ to capture complete multi-view spatiotemporal information.
        - *Feed-Forward Layer*: Further transforms features.
    - Formula: $h = D^{st}(D^{tem}(z', f^{text}, t), c^{occ}, t)$

3. **Modal-Specific Components (Cross-Modal Interaction Layers + Projection Heads)**:

    - **Cross-Modal Interaction Layer $D_m^{cm}$** (inserted every $\alpha_2$ modal-shared layers): self-attention + cross-attention + FFN. The cross-attention query comes from the current modality's latent, while keys/values come from the concatenated latents of **other modalities**: $h'_m = D_m^{cm}(h, h_m^{modal}, t)$
    - **Modality-Specific Projection Head**: Linear layer + adaptive normalization for estimating per-modality noise $\epsilon$.
    - **Design Motivation**: Cross-modal attention enables different modalities to supply complementary information to one another while preserving each modality's distinctive features.

### Loss & Training

- **Training Objective**: DDPM noise estimation loss, summed with per-modality weighting: $\mathcal{L} = \sum_m^M \lambda_m \mathbb{E}_{x_{0,m}, t_m, \epsilon_m, C} \|\epsilon_m - \epsilon_{\theta,m}(x_{t,m}, t_m, C)\|^2$
- **Condition Dropout**: Randomly drops a subset of conditions to improve generalization and output diversity.
- **Inference**: DDIM sampler for accelerated denoising + classifier-free guidance to balance diversity and condition adherence.
- **Pretraining Strategy**: Temporal layers and projection heads are initialized with CogVideoX pretrained weights; all other layers are randomly initialized. The 3D VAE and T5 encoder are frozen.
- **Default Setting**: 6 cameras, 49 frames, resolution 512×256.

## Key Experimental Results

### Main Results (nuScenes)

| Method | FVD↓ | mAP↑ | mIoU↑ | AbsRel↓ | Sem-mIoU↑ |
|--------|------|------|-------|---------|-----------|
| DriveDreamer | 340.8 | - | - | - | - |
| MagicDrive | 236.2 | 9.7 | 15.6 | 0.255 | 23.5 |
| MagicDrive-V2 | 112.7 | 11.5 | 17.4 | 0.280 | 22.4 |
| CogVideoX+SyntheOcc | 60.4 | 15.9 | 28.2 | 0.124 | 32.4 |
| **MoVieDrive** | **46.8** | **22.7** | **35.8** | **0.110** | **37.5** |

- FVD improves ~22% over the strongest baseline (CogVideoX+SyntheOcc).
- Achieves comprehensive superiority in controllability (mAP, mIoU) and multi-modal quality (AbsRel, Sem-mIoU).

### Ablation Study

| Configuration | FVD↓ | AbsRel↓ | Sem-mIoU↑ | Note |
|---------------|------|---------|-----------|------|
| RGB only + external depth/semantic models | 42.0 | 0.121 | 36.4 | Single-modality generation + post-processing |
| RGB+Depth unified + external semantic | 43.4 | 0.111 | 36.0 | Two-modality unification benefits depth |
| RGB+Depth+Semantic fully unified | 46.8 | **0.110** | **37.5** | Three-modality complementarity is optimal |

| Transformer Component | FVD↓ | Note |
|-----------------------|------|------|
| Temporal layers only (L1) | 153.7 | Lacks spatial consistency |
| L1 + modal-specific (L3) | 78.8 | Multi-modal differentiation helps |
| L1 + multi-view spatiotemporal block (L2) + L3 | **46.8** | Full model is optimal |

### Key Findings

- **Unified model outperforms multi-model pipeline**: Jointly generating three modalities yields better depth and semantic quality than a two-stage approach of generating RGB first followed by independent estimation models.
- **Multi-view spatiotemporal block is critical**: Removing it causes FVD to surge from 46.8 to 78.8, with severe degradation in cross-view consistency.
- **Unified layout encoder outperforms independent VAE encoding**: Implicit alignment of conditioning embedding spaces yields performance gains.
- **Waymo generalization**: Achieves FVD 61.6 on Waymo, outperforming CogVideoX+SyntheOcc (82.3).
- **Long video generation**: Capable of generating long videos without reference frames while maintaining consistent scene layout and content.

## Highlights & Insights

- **Pioneering work in unified multi-modal generation**: The first approach in autonomous driving to build a single model that simultaneously generates RGB, depth, and semantic tri-modal multi-view video, filling an important gap.
- **Successful validation of the "shared latent space" hypothesis**: Different modalities can indeed be effectively modeled through a shared 3D VAE with a small number of modal-specific layers, offering architectural insights for multi-modal generation.
- **High engineering quality in condition design**: Three-level conditioning inputs (global text, mid-granularity layout, initial frame reference) combined with a unified layout encoder make generation both controllable and flexible.
- **Scene style editing support**: Modifying the text prompt enables generation of driving scenes under different time-of-day and weather conditions.

## Limitations & Future Work

- **Limited quality of depth and semantic pseudo-labels**: Training depth maps are derived from Depth-Anything-V2 and semantic maps from Mask2Former rather than ground truth; real multi-modal annotations would likely yield further improvements.
- **Poor generation quality in distant regions**: Long video generation exhibits noisy artifacts in distant areas, possibly due to temporal compression in the 3D VAE discarding fine-grained details.
- **High computational cost**: Unifying multiple modalities introduces additional parameters and computation from modal-specific layers; training time and inference speed are not reported.
- **LiDAR modality not addressed**: The method supports only RGB, depth, and semantic visual modalities, without extension to point cloud or other 3D sensor data.
- **Future directions**: (a) More efficient cross-modal information fusion; (b) Extension to additional modalities (optical flow, normal maps); (c) Joint optimization with downstream tasks (3D detection, planning).

## Related Work & Insights

- **vs MagicDrive/MagicDrive-V2**: The MagicDrive series encodes box coordinates and handles conditions independently; MoVieDrive instead uses 2D box map projections and a unified layout encoder, yielding a simpler design with better performance.
- **vs UniScene**: UniScene uses separate models for RGB and LiDAR generation; MoVieDrive achieves genuine single-model multi-modal generation.
- **vs CogVideoX+SyntheOcc**: The direct baseline competitor; MoVieDrive adds multi-view spatiotemporal blocks and cross-modal interaction layers on top, improving FVD by 22%.
- **Insights**: The modal-shared + modal-specific framework design is generalizable to other multi-modal generation tasks; the unified layout encoder's condition fusion strategy is worth referencing.

## Rating

- Novelty: ⭐⭐⭐⭐ First unified multi-modal multi-view driving scene generation framework with a well-motivated architecture.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual-dataset evaluation on nuScenes + Waymo with thorough ablation and visualization analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, highly informative method figures, and a complete notation system.
- Value: ⭐⭐⭐⭐ Significant value for autonomous driving data synthesis; unified multi-modal generation reduces deployment complexity.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Rethinking Position Embedding as a Context Controller for Multi-Reference and Multi-Shot Video Generation](rethinking_position_embedding_as_a_context_controller_for_multi-reference_and_mu.md)
- [\[CVPR 2026\] Let Your Image Move with Your Motion! – Implicit Multi-Object Multi-Motion Transfer](let_your_image_move_with_your_motion_--_implicit_multi-object_multi-motion_trans.md)
- [\[CVPR 2026\] Unified Camera Positional Encoding for Controlled Video Generation](unified_camera_positional_encoding_for_controlled_video_generation.md)
- [\[CVPR 2026\] SwitchCraft: Training-Free Multi-Event Video Generation with Attention Controls](switchcraft_training-free_multi-event_video_generation_with_attention_controls.md)
- [\[CVPR 2026\] StreamDiT: Real-Time Streaming Text-to-Video Generation](streamdit_real-time_streaming_text-to-video_generation.md)

<!-- RELATED:END -->

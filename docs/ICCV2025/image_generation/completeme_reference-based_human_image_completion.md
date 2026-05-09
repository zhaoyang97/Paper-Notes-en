---
title: >-
  [Paper Note] CompleteMe: Reference-based Human Image Completion
description: >-
  [ICCV 2025][Image Generation][Image Completion] This paper proposes the CompleteMe framework, which leverages a dual U-Net architecture and Region-focused Attention (RFA) Block to achieve high-fidelity reference-guided human image completion by exploiting fine-grained person-specific details (clothing textures, tattoos, etc.) from reference images.
tags:
  - ICCV 2025
  - Image Generation
  - Image Completion
  - Reference-based Inpainting
  - Dual U-Net
  - Attention Mechanism
  - Human Body
date: 2026-05-08
content_hash: 583af969aa7182e1
---

# CompleteMe: Reference-based Human Image Completion

**Conference**: ICCV 2025
**arXiv**: [2504.20042](https://arxiv.org/abs/2504.20042)
**Code**: N/A
**Area**: Image Generation / Human Image Completion
**Keywords**: Image Completion, Reference-based Inpainting, Dual U-Net, Attention Mechanism, Human Body

## TL;DR

This paper proposes the CompleteMe framework, which leverages a dual U-Net architecture and Region-focused Attention (RFA) Block to achieve high-fidelity reference-guided human image completion by exploiting fine-grained person-specific details (clothing textures, tattoos, etc.) from reference images.

## Background & Motivation

### State of the Field

Human image completion is an important task in computer vision, with applications in photo editing, virtual try-on, and animation. Existing methods fall into two categories with corresponding limitations:

**Limitations of reference-free methods**:

### Root Cause

Methods such as LOHC and BrushNet can generate plausible human body shapes but fail to recover person-specific details (e.g., particular clothing patterns, tattoo designs, unique accessories).

### Limitations of Prior Work

Without a reference image, such unique information cannot be hallucinated from scratch.

**Limitations of reference-guided methods**:

### Starting Point

Paint-by-Example, AnyDoor, and similar methods primarily focus on object-level insertion or completion.

### Supplementary Note

Methods such as MimicBrush struggle to establish accurate correspondences when the pose difference between the source and reference images is large.

### Supplementary Note

Existing methods cannot effectively capture and integrate fine-grained details from reference images.

Core challenge: How to precisely map local details from reference images to the target completion region under significant pose discrepancy.

## Method

### Overall Architecture

CompleteMe adopts a **dual U-Net architecture** consisting of:
1. **Reference U-Net ($U_{ref}$)**: Extracts detailed visual features from multiple reference images.
2. **Complete U-Net ($U_{comp}$)**: Processes the masked input and leverages reference features to perform completion.
3. **CLIP Image Encoder**: Provides global semantic features.

Reference images are segmented by body region (upper garment, lower garment, hair, face, shoes, etc.), encoded separately, and fused into the Complete U-Net via RFA Blocks.

### Key Designs

**1. Reference U-Net**

- Initialized from Stable Diffusion 1.5 pretrained weights.
- Directly encodes reference images at timestep=0 (without diffusion noise).
- Extracts multi-scale spatial features separately for different body regions (upper-body clothing, lower-body clothing, hair/accessories, face, shoes).
- Processes each reference image sequentially to ensure flexibility.

**2. Region-focused Attention (RFA) Block**

This is the core contribution of CompleteMe:

- **Explicit mask filtering**: Applies reference masks to suppress irrelevant regions in reference features, producing masked reference features.
- **Feature concatenation**: Concatenates masked reference features with input features.
- **Region-focused attention**:

$$\text{RFA}(Q, K, V) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

where $Q = f_{input}$, $K, V = f_{concat}$

- **Decoupled cross-attention**: Inspired by IP-Adapter, cross-attention is performed separately over local reference features and global CLIP features, and the results are summed.

**3. Masking Strategy**

A mixed masking strategy is employed during training:
- 50% probability: random grid masks (1–30 iterations).
- 50% probability: human body shape masks.

### Loss & Training

- **Loss Function**: MSE Loss
- **Optimizer**: Adam, lr=2×10⁻⁵
- **Training Setup**: 8×A100, batch size 64, 30K iterations
- **Random Dropout**: All reference features are dropped with probability 0.2; each reference condition is independently dropped with probability 0.2.
- **Inference**: DDIM 50 steps, guidance scale 7.5
- **Training Data**: Built upon DeepFashion-MultiModal; 40K training pairs.

## Key Experimental Results

### Main Results

| Method | CLIP-I↑ | DINO↑ | DreamSim↓ | LPIPS↓ | PSNR↑ | SSIM↑ |
|--------|---------|-------|-----------|--------|-------|-------|
| BrushNet | 95.90 | 95.08 | 0.0576 | 0.0600 | 28.58 | 0.9224 |
| LeftRefill | 96.33 | 95.12 | 0.0574 | 0.0598 | 28.87 | 0.9283 |
| MimicBrush | 96.98 | 94.37 | 0.0651 | 0.0694 | 28.36 | 0.9174 |
| **CompleteMe** | **97.18** | **96.29** | **0.0419** | **0.0588** | 28.70 | 0.9239 |

CompleteMe achieves state-of-the-art performance on all identity-consistency metrics (CLIP-I, DINO, DreamSim), reducing DreamSim from 0.0574 to 0.0419 (a 27% improvement).

### Ablation Study

| Ablation | Result |
|----------|--------|
| Reference-free vs. reference-guided | Reference-free methods fail to recover person-specific details |
| Without RFA | Difficulty in establishing accurate correspondences |
| Multi-reference vs. single-reference | Multi-region references provide more comprehensive information |

### Key Findings

1. **Explicit region focusing is critical**: Applying cross-attention over the full image performs poorly; explicit masking combined with concatenation enables the model to precisely match corresponding regions.
2. **Part-wise separate encoding**: Decomposing reference images by body region and encoding them separately is more effective than encoding the full image.
3. **Model flexibility**: At inference time, a single reference image can be used, and textual prompts can optionally be incorporated.
4. **User study validation**: A large-scale user study confirms the subjective superiority of CompleteMe.

## Highlights & Insights

1. **Clear task formulation**: The paper explicitly distinguishes between the sub-problems of reference-free and reference-guided completion.
2. **Elegant RFA design**: Explicitly directing attention to relevant regions via masking is more efficient and reliable than implicit learning.
3. **Decoupled global + local**: A dual-track design combining CLIP global semantics with Reference U-Net local details.
4. **Practical benchmark**: A test set of 417 groups with significant pose variation is constructed.

## Limitations & Future Work

1. **Limited training data scale**: Only 40K training pairs are used.
2. **Based on SD1.5**: The backbone model is relatively dated; upgrading to SDXL/SD3 may yield quality improvements.
3. **Extreme pose discrepancy**: Correspondences may still fail when the pose difference between source and reference is very large.
4. **Dependency on body-part parsing**: Body region segmentation is required as part of the reference input pipeline.
5. **Static images only**: The method has not been extended to temporally consistent completion for video sequences.

## Related Work & Insights

- **MimicBrush**: Dual diffusion U-Net with self-supervised video training, but performs poorly under large pose discrepancies.
- **AnyDoor**: A zero-shot object teleportation framework oriented toward object-level operations.
- **IP-Adapter**: Its decoupled cross-attention mechanism is adopted by CompleteMe for global/local feature fusion.
- **BrushNet**: Dual-branch pixel-level mask feature embedding, but lacks reference-guided capability.
- Insight: **The core challenge in reference-guided image editing is correspondence establishment**; explicit region guidance is more reliable than purely implicit learning.

## Rating

| Dimension | Score (1–5) |
|-----------|------------|
| Novelty | 3.5 |
| Technical Depth | 3.5 |
| Experimental Thoroughness | 4 |
| Writing Quality | 3.5 |
| Practicality | 4 |
| Overall | 3.5 |

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] HiFi-Inpaint: Towards High-Fidelity Reference-Based Inpainting for Generating Detail-Preserving Human-Product Images](../../CVPR2026/image_generation/hifi-inpaint_towards_high-fidelity_reference-based_inpainting_for_generating_det.md)
- [\[ICCV 2025\] Cycle Consistency as Reward: Learning Image-Text Alignment without Human Preferences](cycle_consistency_as_reward_learning_image-text_alignment_without_human_preferen.md)
- [\[ICCV 2025\] HPSv3: Towards Wide-Spectrum Human Preference Score](hpsv3_towards_wide-spectrum_human_preference_score.md)
- [\[ICCV 2025\] A Unified Framework for Motion Reasoning and Generation in Human Interaction](a_unified_framework_for_motion_reasoning_and_generation_in_human_interaction.md)
- [\[NeurIPS 2025\] GeneMAN: Generalizable Single-Image 3D Human Reconstruction from Multi-Source Human Data](../../NeurIPS2025/image_generation/geneman_generalizable_single-image_3d_human_reconstruction_from_multi-source_hum.md)

<!-- RELATED:END -->

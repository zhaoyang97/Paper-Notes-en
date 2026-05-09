---
title: >-
  [Paper Note] SplatTalk: 3D VQA with Gaussian Splatting
description: >-
  [3D Vision] This paper proposes SplatTalk, a framework that leverages generalizable 3D Gaussian Splatting to generate LLM-compatible 3D tokens from multi-view RGB images alone, enabling zero-shot 3D visual question answering that surpasses 2D LMM baselines and approaches 3D LMM performance.
tags:
  - 3D Vision
date: 2026-05-08
content_hash: 262e48034788a5fd
---

# SplatTalk: 3D VQA with Gaussian Splatting

## Metadata
- **Conference**: ICCV 2025
- **arXiv**: [2503.06271](https://arxiv.org/abs/2503.06271)
- **Code**: [Project Page](https://splat-talk.github.io/)
- **Area**: 3D Vision
- **Keywords**: 3D VQA, 3D Gaussian Splatting, language-guided 3D understanding, LLM, zero-shot

## TL;DR

This paper proposes SplatTalk, a framework that leverages generalizable 3D Gaussian Splatting to generate LLM-compatible 3D tokens from multi-view RGB images alone, enabling zero-shot 3D visual question answering that surpasses 2D LMM baselines and approaches 3D LMM performance.

## Background & Motivation

### Limitations of Prior Work

**Background**: Language-guided 3D scene understanding is critical for applications such as robotics and AR/VR. Existing approaches face three key challenges:

**Scarcity of 3D data**: Compared to 2D, language-annotated 3D data is extremely limited, constraining the development of 3D LMMs.

**Reliance on explicit 3D input**: Most 3D methods require explicit 3D representations such as point clouds or meshes, making them ill-suited for image-only settings.

**Lack of 3D reasoning in 2D methods**: Feeding multi-view images directly into 2D LMMs organizes tokens by image frame rather than 3D space, preventing cross-view spatial reasoning (e.g., "What is across from the door?").

**Core Idea**: Aggregating tokens in 3D space prior to feeding them into an LLM substantially improves spatial reasoning capability.

## Method

### Overall Architecture

SplatTalk consists of three stages:

**Stage 1: Feature Autoencoder Training**
- High-dimensional visual tokens are extracted from LLaVA-OV's visual encoder and multimodal projector (these tokens are already aligned with the LLM input space).
- An autoencoder is trained to compress the high-dimensional, unbounded, and sparse features into a low-dimensional hyperspherical space.

**Stage 2: Self-Supervised 3D-Language Gaussian Splatting Training**
- The FreeSplat framework is extended to support language Gaussian field learning.
- Multi-view RGB images → CNN multi-scale feature extraction → adaptive cost volume construction for depth prediction → back-projection into 3D Gaussian triplets $(\boldsymbol{\mu}, \boldsymbol{\omega}, \boldsymbol{f})$.
- A Pixel-wise Triplet Fusion (PTF) module merges overlapping Gaussians across views.
- An MLP decoder simultaneously predicts rendering parameters and low-dimensional language features $f$.

**Stage 3: 3D VQA Inference**
- Language features are extracted from 3D Gaussians → mapped back to high-dimensional space via the decoder → fed directly as visual tokens into the LLM.

### Key Designs

**1. Visual Token Selection**
Tokens after the multimodal projector are used as training targets (rather than raw encoder features), since post-projector features are already aligned with the LLM's latent space and can be directly understood and reasoned over.

**2. Theoretical Analysis of Mean Feature Aggregation**
Scene-level tokens are obtained by mean pooling the language features of 3D Gaussians. The paper provides a theoretical justification that mean feature aggregation encodes holistic conceptual information about the 3D scene.

**3. Entropy-Adaptive Token Sampling**
Tokens are sampled non-uniformly, prioritizing regions with higher information entropy, improving 3D VQA performance without additional training cost.

### Loss & Training

The self-supervised training objective comprises:
- RGB reconstruction loss: $\mathcal{L}_{\text{rgb}}$ (ensuring rendering quality)
- Language feature reconstruction loss: $\mathcal{L}_{\text{lang}}$ (aligning low-dimensional features with pseudo ground truth compressed by the autoencoder)
- Optional LoRA fine-tuning loss: $\mathcal{L}_{\text{lora}}$ (fine-tuning the LLM to improve VQA performance)

## Key Experimental Results

### Main Results: Comparison on ScanQA and SQA3D

| Method | Modality | ScanQA CIDEr | ScanQA EM@1 | SQA3D EM@1 |
|--------|----------|--------------|-------------|------------|
| ScanQA | PC | 64.9 | 21.1 | 47.2 |
| 3D-VisTA | PC | 69.6 | 22.4 | 48.5 |
| LEO | PC+I | 101.4 | 24.5 | 50.0 |
| LLaVA-OV | I | 50.0 | 15.6 | - |
| GPT-4V | I | 59.6 | - | - |
| **SplatTalk (Ours)** | **I** | **Surpasses 2D methods** | **Surpasses 2D methods** | **Approaches 3D SOTA** |

### Ablation Study

| Component | CIDEr Change |
|-----------|--------------|
| Without autoencoder compression (direct high-dim features) | Significant drop; training unstable |
| Without entropy-adaptive sampling | Moderate drop |
| Using encoder features only (pre-projector) | Severe drop; projector retraining required |
| Without 3D representation (pure 2D multi-view) | Notably insufficient spatial reasoning |

### Key Findings
1. **Necessity of 3D representation**: SplatTalk demonstrates clear advantages over pure 2D methods on questions requiring cross-view spatial reasoning (e.g., "What is across from the window?").
2. **Importance of post-projector features**: Using raw encoder features requires retraining the projector, whereas post-projector visual tokens are directly compatible with the LLM.
3. **Zero-shot capability**: Competitive 3D VQA is achieved through self-supervised training alone, without any 3D-language annotation data.

## Highlights & Insights

1. **First self-supervised 3D Gaussian language field method for zero-shot 3D VQA**: Removes the dependency of 3D VQA on explicit 3D inputs such as point clouds.
2. **"Aggregate in 3D space first, then feed to LLM" paradigm**: Captures spatial relationships more effectively than directly processing multi-view images.
3. **Strong practicality**: Requires only multi-view RGB images, with no need for depth maps, point clouds, or other additional inputs.

## Limitations & Future Work

- Performance depends on the quality of FreeSplat's generalizable 3DGS; robustness to sparse views and complex scenes remains to be validated.
- Autoencoder compression may discard some fine-grained semantic information.
- Multi-view image input is required at inference time, which may limit real-time applicability.

## Related Work & Insights

- **3D LMMs**: LEO, Chat-Scene, LL3DA
- **2D LMMs**: LLaVA-OV, GPT-4V
- **Semantic 3DGS**: LangSplat, ChatSplat
- **Generalizable 3DGS**: FreeSplat

## Rating

- Novelty: ⭐⭐⭐⭐ (novel combination of 3DGS and LLM)
- Technical Depth: ⭐⭐⭐⭐ (complete theoretical analysis and system design)
- Experimental Thoroughness: ⭐⭐⭐⭐ (comprehensive comparison across multiple benchmarks)
- Value: ⭐⭐⭐⭐ (requires only RGB images; low barrier to deployment)

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] StochasticSplats: Stochastic Rasterization for Sorting-Free 3D Gaussian Splatting](stochasticsplats_stochastic_rasterization_for_sorting-free_3d_gaussian_splatting.md)
- [\[ICCV 2025\] LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos](longsplat_robust_unposed_3d_gaussian_splatting_for_casual_long_videos.md)
- [\[ICCV 2025\] Tune-Your-Style: Intensity-Tunable 3D Style Transfer with Gaussian Splatting](tune-your-style_intensity-tunable_3d_style_transfer_with_gaussian_splatting.md)
- [\[ICCV 2025\] Self-Ensembling Gaussian Splatting for Few-Shot Novel View Synthesis](self-ensembling_gaussian_splatting_for_few-shot_novel_view_synthesis.md)
- [\[ICCV 2025\] ri3d few-shot gaussian splatting with repair and inpainting diffusion priors](ri3d_few-shot_gaussian_splatting_with_repair_and_inpainting_diffusion_priors.md)

<!-- RELATED:END -->

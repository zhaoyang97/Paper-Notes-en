---
title: >-
  [Paper Note] BVINet: Unlocking Blind Video Inpainting with Zero Annotations
description: >-
  [ICCV 2025][Image Generation][blind video inpainting] This paper is the first to formally define and address the task of *blind video inpainting*—simultaneously predicting *where* to restore and *how* to restore, end-to-end, without any annotation of corrupted regions. A mask prediction network and a video completion network mutually reinforce each other via a consistency constraint, achieving strong results on both synthetic data and real-world applications (danmaku removal and scratch repair).
tags:
  - ICCV 2025
  - Image Generation
  - blind video inpainting
  - mask prediction
  - wavelet sparse transformer
  - video completion
  - consistency loss
date: 2026-05-08
content_hash: 8699eb7a17ce27ea
---

# BVINet: Unlocking Blind Video Inpainting with Zero Annotations

**Conference**: ICCV 2025
**arXiv**: [2502.01181](https://arxiv.org/abs/2502.01181)
**Code**: N/A
**Area**: Image Generation / Video Restoration
**Keywords**: blind video inpainting, mask prediction, wavelet sparse transformer, video completion, consistency loss

## TL;DR

This paper is the first to formally define and address the task of *blind video inpainting*—simultaneously predicting *where* to restore and *how* to restore, end-to-end, without any annotation of corrupted regions. A mask prediction network and a video completion network mutually reinforce each other via a consistency constraint, achieving strong results on both synthetic data and real-world applications (danmaku removal and scratch repair).

## Background & Motivation

Existing video inpainting methods are fundamentally *non-blind*—they assume the corruption mask is known in advance, requiring users to manually annotate damaged regions in each frame. This leads to two practical problems:

**High annotation cost**: Boundaries between corrupted and clean regions are often ambiguous, making precise annotation difficult and time-consuming, especially for high-frame-rate or high-resolution videos.

**Limited applicability**: In many scenarios, corrupted regions cannot be anticipated or manually labeled, such as video scratches, watermarks, and danmaku overlays.

The authors categorize video corruption into two types: (1) externally introduced artifacts that disrupt the original video structure (scratches, watermarks, danmaku, etc.); and (2) unwanted content originally present in the video (object removal). This paper focuses on the first category.

A naive approach of applying blind image inpainting frame-by-frame ignores inter-frame motion continuity, leading to flickering artifacts. An end-to-end video-level solution is therefore necessary.

## Method

### Overall Architecture

BVINet consists of two mutually constrained sub-networks: a **Mask Prediction Network** (MPNet) that predicts corrupted regions, and a **Video Completion Network** (VCNet) that uses the predicted masks to restore corrupted content by aggregating information from valid regions. Both networks are jointly optimized via a consistency loss.

### Key Designs

1. **Mask Prediction Network (MPNet)**:

    - Two-stage structure: Short-Term Prediction module + Long-Term Refinement module
    - **Short-Term Prediction (STP)**:
        - Encoder-decoder architecture; processes each frame independently
        - Detects intra-frame semantic discontinuities to predict corruption masks: $m_i^s = STP(x_i)$
        - Replaces conventional downsampling (max-pooling/strided-conv) with Discrete Wavelet Transform (DWT) to improve robustness against noise
    - **Long-Term Refinement (LTR)**:
        - Exploits temporal consistency priors to refine the predicted mask sequence
        - Core component: sequence-to-sequence transformer
        - Maps deep features to Q/K/V, partitioned into $N$ groups along the channel dimension; computes spatial-temporal affinity matrices within a $T$-frame response window
        - Soft-attention aggregates multi-group affinity matrices and aggregated features: $\hat{E} = E + Conv(D) \odot G$

2. **Video Completion Network (VCNet) — Wavelet Sparse Transformer**:

    - Innovation: isolates noise in the frequency domain and uses sparse attention to aggregate the most relevant features
    - Frequency decomposition: applies DWT to Q/K/V, isolating noise into high-frequency components $Q_i^H, K_i^H, V_i^H$, while low-frequency components $Q_i^L, K_i^L, V_i^L$ retain only clean, structural features
    - **Dual-branch attention mechanism**:
        - Dense Self-Attention (DSA): $DSA = Softmax(\frac{Q^L \cdot (K^L)^T}{\sqrt{d}} + B)$
        - Sparse Self-Attention (SSA): $SSA = Softmax(ReLU(\frac{Q^L \cdot (K^L)^T}{\sqrt{d}}) + B)$, using ReLU to suppress negative similarities
        - Adaptive weighted fusion: $\hat{V}^L = (\omega_1 \odot DSA + \omega_2 \odot SSA) V^L$
    - Attention values at corrupted positions are set to zero in both DSA and SSA, ensuring information is borrowed exclusively from valid regions
    - Inverse Wavelet Transform (IDWT) reconstructs the final feature representation

3. **Consistency Loss**:

    - Core idea: if both mask prediction and video completion are accurate, the difference between the corrupted frame $x_i$ and the completed result $\hat{y}_i$ should reside exclusively within the corrupted region
    - Consistency relation: $m_i^l = \mathcal{B}(\hat{y}_i - x_i)$
    - Consistency loss: $\mathcal{L}_c = \|m_i^l - \mathcal{B}(\hat{y}_i - x_i)\|_1 + \|m_i - \mathcal{B}(\hat{y}_i - x_i)\|_1$
    - Enforces precise correspondence between the two networks, enabling mutual regularization

### Loss & Training

Total loss: $\mathcal{L} = \lambda_m \mathcal{L}_m + \lambda_v \mathcal{L}_v + \lambda_c \mathcal{L}_c$

- $\mathcal{L}_m$: mask prediction loss
- $\mathcal{L}_v$: video completion loss
- $\mathcal{L}_c$: consistency loss
- Hyperparameters: $\lambda_m = 3$, $\lambda_v = 5$, $\lambda_c = 0.02$ (determined via grid search)

**Dedicated dataset construction**:
- Free-form strokes are used as corruption masks, filled with real image content (rather than constant values or noise)
- Iterative Gaussian blur expands corruption boundaries to eliminate obvious edge priors, forcing the model to infer from semantic context
- Dataset: 2,400 synthetic videos + 1,250 real danmaku removal videos

## Key Experimental Results

### Main Results (Blind vs. Non-Blind Setting)

YouTube-VOS dataset:

| Method | Blind | PSNR↑ | SSIM↑ | $E_{warp}$↓ | LPIPS↓ |
|--------|-------|-------|-------|-------------|--------|
| FGT (non-blind) | ✗ | 30.811 | 0.9258 | 0.1308 | 0.4565 |
| WaveFormer (non-blind) | ✗ | 33.264 | 0.9435 | 0.1184 | 0.2933 |
| **VCNet (non-blind)** | ✗ | **34.107** | **0.9521** | **0.1102** | **0.2145** |
| MPNet+FGT (blind) | ✓ | 27.032 | 0.8755 | 0.1609 | 0.8667 |
| MPNet+WaveFormer (blind) | ✓ | 29.185 | 0.8902 | 0.1508 | 0.7153 |
| **BVINet (blind)** | ✓ | **30.528** | **0.9088** | **0.1362** | **0.6556** |

### Ablation Study

MPNet ablation:

| Configuration | BCE↓ | IOU↑ |
|---------------|------|------|
| STP (strided-conv) | 1.1251 | 0.8437 |
| DWT_STP | 1.0785 | 0.8682 |
| DWT_STP + LTR | 0.9176 | 0.8829 |
| **Full MPNet** | **0.8052** | **0.9017** |

Sparse attention + consistency loss ablation:

| DSA | SSA | $\mathcal{L}_c$ | PSNR↑ | SSIM↑ | $E_{warp}$↓ | LPIPS↓ |
|-----|-----|-----------------|-------|-------|-------------|--------|
| ✓ | ✗ | ✗ | 29.172 | 0.8897 | 0.1529 | 0.7264 |
| ✓ | ✓ | ✗ | 29.885 | 0.8962 | 0.1454 | 0.6891 |
| ✓ | ✓ | ✓ | **30.528** | **0.9088** | **0.1362** | **0.6556** |

### Efficiency Analysis

| Method | FLOPs | Inference Time |
|--------|-------|----------------|
| STTN | 477.91G | 0.22s |
| FuseFormer | 579.82G | 0.30s |
| E2FGVI | 442.18G | 0.26s |
| FGT | 455.91G | 0.39s |
| **VCNet** | **396.35G** | **0.21s** |

### Key Findings

- BVINet in the blind setting achieves performance comparable to the second-best non-blind method E2FGVI (PSNR 30.528 vs. 30.064), validating the feasibility of blind inpainting
- VCNet in the non-blind setting substantially outperforms all existing methods (PSNR 34.107 vs. runner-up 33.264)
- DWT downsampling significantly improves mask prediction quality (IOU 0.8437→0.8682), with long-term refinement further boosting it to 0.9017
- The dual-branch (DSA+SSA) design outperforms either branch alone; the consistency loss contributes an additional ~0.6 dB PSNR gain
- The model is robust to multiple corruption patterns, including Gaussian noise and solid-color fills unseen during training
- BVINet outperforms OGNet and RAVUNet on real-world danmaku removal

## Highlights & Insights

- **Pioneer task definition**: The first work to formally propose blind video inpainting, unifying "where to restore" and "how to restore" in a single framework
- **Elegant mutual constraint design**: Mask prediction and video completion form a closed loop through the consistency loss—the mask guides localization, while the completion result in turn validates mask quality
- **Frequency-domain sparse attention**: The combination of DWT to isolate noise into high-frequency components and ReLU to suppress negative similarities is better suited to inpainting than conventional attention
- **Dataset design rationale**: Filling corrupted regions with real image content (rather than constant values) and blurring boundaries prevents the model from learning distributional priors, instead encouraging semantic understanding

## Limitations & Future Work

- The paper focuses primarily on externally introduced corruption (scratches, danmaku, etc.); performance on the second corruption type (removal of originally present content) is not validated
- The dataset is relatively small (2,400 + 1,250 videos); larger-scale data could further improve generalization
- In the blind setting, a gap of 3–4 dB PSNR remains compared to the strongest non-blind methods, with mask prediction accuracy as the bottleneck
- The potential of diffusion models for blind video inpainting remains unexplored
- The corruption types addressed are primarily additive overlays; applicability to degradation types such as compression artifacts warrants further investigation

## Related Work & Insights

- The transition from blind image inpainting to blind video inpainting is not a trivial frame-by-frame extension—temporal consistency is the central challenge
- The constraint of "borrowing information exclusively from valid regions" in video inpainting stands in contrast to the global aggregation of standard attention; sparse attention's selectivity is better suited to such tasks
- The consistency constraint paradigm can be generalized to other video restoration tasks requiring joint optimization of detection and restoration

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Entirely new task definition; the end-to-end blind inpainting framework is groundbreaking
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validated on both synthetic and real data with 16 baselines, detailed ablations, and efficiency analysis
- Writing Quality: ⭐⭐⭐⭐ — Clear problem formalization and well-structured methodological decomposition
- Value: ⭐⭐⭐⭐ — Opens a zero-annotation paradigm for video inpainting with high practical value for danmaku/watermark removal

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Unlocking the Potential of Diffusion Priors in Blind Face Restoration](unlocking_the_potential_of_diffusion_priors_in_blind_face_restoration.md)
- [\[ICCV 2025\] AnyPortal: Zero-Shot Consistent Video Background Replacement](anyportal_zero-shot_consistent_video_background_replacement.md)
- [\[ICCV 2025\] Less-to-More Generalization: Unlocking More Controllability by In-Context Generation](less-to-more_generalization_unlocking_more_controllability_by_in-context_generat.md)
- [\[ICCV 2025\] Video Motion Graphs](video_motion_graphs.md)
- [\[ICCV 2025\] REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers](repa-e_unlocking_vae_for_end-to-end_tuning_of_latent_diffusion_transformers.md)

<!-- RELATED:END -->

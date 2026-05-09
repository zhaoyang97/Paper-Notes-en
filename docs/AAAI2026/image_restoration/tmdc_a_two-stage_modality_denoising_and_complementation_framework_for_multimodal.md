---
title: >-
  [Paper Note] TMDC: A Two-Stage Modality Denoising and Complementation Framework for Multimodal Sentiment Analysis
description: >-
  [AAAI 2026][Image Restoration][multimodal sentiment analysis] This paper proposes TMDC, a two-stage framework in which the first stage learns denoised modality-specific and modality-common representations on complete data, and the second stage leverages denoised representations from available modalities to reconstruct missing ones — marking the first joint treatment of noise and missing modalities in MSA.
tags:
  - AAAI 2026
  - Image Restoration
  - multimodal sentiment analysis
  - missing modality
  - noisy modality
  - variational information bottleneck
  - denoising
date: 2026-05-08
content_hash: 6a039406a233f00e
---

# TMDC: A Two-Stage Modality Denoising and Complementation Framework for Multimodal Sentiment Analysis

**Conference**: AAAI 2026
**arXiv**: [2511.10325](https://arxiv.org/abs/2511.10325)
**Code**: Not available
**Area**: Multimodal Sentiment Analysis
**Keywords**: multimodal sentiment analysis, missing modality, noisy modality, variational information bottleneck, denoising

## TL;DR

This paper proposes TMDC, a two-stage framework in which the first stage learns denoised modality-specific and modality-common representations on complete data, and the second stage leverages denoised representations from available modalities to reconstruct missing ones — marking the first joint treatment of noise and missing modalities in MSA.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: Multimodal Sentiment Analysis (MSA) fuses text, audio, and video to predict sentiment, but real-world scenarios present two major challenges: **missing modalities** (due to privacy concerns or incomplete collection) and **noisy inputs** (due to sensor noise).

### State of the Field

**Background**: Existing methods address these two problems **separately** — denoising methods assume complete data, while missing-modality methods assume clean inputs.

### Root Cause

**Key Challenge**: When noise and missing modalities co-occur, existing methods (e.g., IMDer, DiCMoR, MoMKE) exhibit significant performance degradation.

### Solution Direction

**Solution Direction**: Noisy inputs → erroneous reconstruction → compounding error propagation across training and inference.

### Solution Direction

**Goal**: How to jointly address noise interference and missing modalities in MSA while avoiding error propagation?

## Method

### Overall Architecture

Two-stage training pipeline: Intra-Modality Denoising (IMD) → Inter-Modality Complementation (IMC)

### Stage 1: Intra-Modality Denoising (IMD)

Trained on **complete data**, comprising two denoising modules:

**1. Modality-Specific Denoising (MSD)**: A per-modality VIB + Attention network
- VIB objective: $\mathcal{L}^m = \mathcal{L}_{TASK}(y^m, y) + \beta \text{KL}(p(e_s^m|e^m) \| \mathcal{N}(0, \mathbf{I}))$
- Reparameterization: $X_s^m = \mu_s^m + \epsilon \sigma_s^m$, where $\mu_s^m = W_1^m e^m + b_1^m$
- Denoised features pass through MHA self-attention + residual FC to yield $\hat{X}_{Spe}^m$

**2. Modality-Common Denoising (MCD)**: A **parameter-shared** VIB + Attention network across all modalities
- Identical architecture to MSD, but Conv1D, VIB, and Attention parameters are shared across modalities
- Extracts modality-invariant representation $\hat{X}_{Com}^m$

### Stage 2: Inter-Modality Complementation (IMC)

During training, certain modalities are randomly zeroed out to simulate missing conditions; available modalities are used for complementation:

**1. Unimodal Enhancement**: Fuses specific and common representations via the attention learned in Stage 1:
$$X_{All}^{m1} = \text{MHA}^{m1}(X_s^{m1}, X_c^{m1})$$

**2. Cross-Modal Compensation**: Bidirectional attention across available modalities with swapped query/key to obtain complementary features $X_{T2V}$ and $X_{V2T}$

**3. Final Fusion**: $X = [X_{Compensate}, \hat{X}_{All}^T, \hat{X}_{All}^V]$, followed by FC for prediction

**Overall Training Objectives**:
$$\mathcal{L}_{IMD} = \sum_{m} \left(\sum_{b \in \{Spe,Com\}} \mathcal{L}_b^m + \sum_{k \in \{s,c\}} \mathcal{L}_k^m \right), \quad \mathcal{L}_{IMC} = \mathcal{L}_{TASK}(y_{All}, y)$$

## Key Experimental Results

### Main Results

| Method | MOSI (Avg ACC/F1) | MOSEI (Avg ACC/F1) | IEMOCAP (Avg WA/UA) |
|--------|------------------|-------------------|---------------------|
| MoMKE | 77.05/76.46 | 80.44/79.98 | 73.35/72.78 |
| **TMDC** | **77.64/77.35** | **81.22/80.76** | **73.77/73.64** |

- Gains on MOSI: +0.59 ACC / +0.89 F1; MOSEI: +0.78/+0.78; IEMOCAP: +0.42/+0.86
- **Noise robustness** (Gaussian noise ε=10): TMDC vs. MoMKE — MOSI 60.8 vs. 53.9, MOSEI 71.2 vs. 61.2, IEMOCAP 51.0 vs. 34.4 (average margin of approximately **10 points**)
- Ablation: Removing the IMC stage yields the largest performance drop (MOSI ACC from 77.64 to 74.17); removing MSD has a larger impact than removing MCD

## Highlights & Insights

- **First joint treatment** of noise and missing modalities in MSA, filling a notable research gap
- **Principled two-stage design**: denoising before complementation prevents noise from propagating through reconstruction
- **Dual-path VIB denoising**: specific representations preserve modality-exclusive information while common representations capture shared semantics, with a clear division of roles
- **Strong noise robustness**: at noise level 10, TMDC still outperforms MoMKE by an average of 10 percentage points
- Comprehensive evaluation across 7 missing-modality combinations, including extreme cases

## Limitations & Future Work

- Validation is limited to three relatively small sentiment datasets (MOSI/MOSEI/IEMOCAP)
- Some redundancy exists in the shared representations (acknowledged by the authors in the conclusion), leaving room for further compression
- Hyperparameters such as the VIB $\beta$ require manual tuning
- Only Gaussian noise is considered; real-world noise is more diverse (e.g., modality misalignment, annotation noise)
- Pretrained multimodal large models (e.g., LLMs/VLMs) are not utilized; feature extraction relies on fixed backbones

## Related Work & Insights

**vs. MoMKE (prior SOTA)**: MoMKE trains multiple modality-specific MoE experts without explicit denoising, resulting in sharp performance degradation under noise. TMDC's explicit VIB denoising combined with its two-stage training strategy confers a clear advantage in noisy settings. **vs. IMDer/DiCMoR**: These methods employ diffusion/flow-based modality reconstruction but assume clean inputs, leading to failure when noise and missing modalities co-occur.

## Insights

- The "denoise-then-complement" two-stage paradigm is generalizable to other incomplete multimodal learning scenarios (e.g., medical image multimodal fusion)
- VIB as a general-purpose denoising tool deserves greater attention in multimodal learning, as its information compression property is naturally suited to filtering noise
- Decoupling modality-specific and modality-common representations is an effective strategy for handling missing modalities

## Rating

⭐⭐⭐⭐ — The problem formulation is precise (joint handling of noise and missing modalities), the method design is well-motivated, and noise robustness is thoroughly validated; however, the evaluation datasets are small-scale and large pretrained models are not incorporated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Blink: Dynamic Visual Token Resolution for Enhanced Multimodal Understanding](../../CVPR2026/image_restoration/blink_dynamic_visual_token_resolution_for_enhanced_multimodal_understanding.md)
- [\[AAAI 2026\] Large Language Models Meet Extreme Multi-label Classification: Scaling and Multi-modal Framework](large_language_models_meet_extreme_multi-label_classification_scaling_and_multi-.md)
- [\[AAAI 2026\] Blur-Robust Detection via Feature Restoration: An End-to-End Framework for Prior-Guided Infrared UAV Target Detection](blur-robust_detection_via_feature_restoration_an_end-to-end_framework_for_prior-.md)
- [\[CVPR 2026\] Towards Universal Computational Aberration Correction in Photographic Cameras: A Comprehensive Benchmark Analysis](../../CVPR2026/image_restoration/unicac_universal_computational_aberration_correction_benchmark.md)
- [\[CVPR 2026\] DRFusion: Degradation-Robust Fusion via Degradation-Aware Diffusion Framework](../../CVPR2026/image_restoration/drfusion_degradation_robust_fusion_via_degradation_aware_diffusion_framework.md)

</div>

<!-- RELATED:END -->

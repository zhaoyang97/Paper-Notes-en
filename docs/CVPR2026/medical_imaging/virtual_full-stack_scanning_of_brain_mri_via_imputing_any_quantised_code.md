---
title: >-
  [Paper Note] Virtual Full-stack Scanning of Brain MRI via Imputing Any Quantised Code
description: >-
  [CVPR 2026][Medical Imaging][MRI modality imputation] This paper proposes CodeBrain, which reformulates the any-to-any brain MRI modality imputation problem as a region-level full-stack quantised code prediction task. Through a two-stage pipeline (scalar quantisation reconstruction + grading-loss code prediction), it achieves unified missing modality synthesis and outperforms five state-of-the-art methods.
tags:
  - CVPR 2026
  - Medical Imaging
  - MRI modality imputation
  - finite scalar quantisation
  - brain MRI
  - cross-modality synthesis
  - any-to-any
date: 2026-05-08
content_hash: d99610881b1ed4e2
---

# Virtual Full-stack Scanning of Brain MRI via Imputing Any Quantised Code

**Conference**: CVPR 2026
**arXiv**: [2501.18328](https://arxiv.org/abs/2501.18328)
**Code**: [Available](https://github.com/ycwu1997/CodeBrain)
**Area**: Medical Imaging
**Keywords**: MRI modality imputation, finite scalar quantisation, brain MRI, cross-modality synthesis, any-to-any

## TL;DR

This paper proposes CodeBrain, which reformulates the any-to-any brain MRI modality imputation problem as a region-level full-stack quantised code prediction task. Through a two-stage pipeline (scalar quantisation reconstruction + grading-loss code prediction), it achieves unified missing modality synthesis and outperforms five state-of-the-art methods.

## Background & Motivation

### 1. Clinical Need
Brain MRI examinations involve multiple acquisition protocols (T1, T2, PD, FLAIR, T1Gd, etc.), with different modalities emphasising distinct anatomical or pathological features. In clinical practice, however, acquiring a complete modality set is often infeasible due to scanning time, cost, and contrast agent risks. Virtual full-stack scanning aims to impute missing modalities from incomplete acquisitions, improving data completeness and clinical utility.

### 2. Limitations of Prior Work
Existing unified imputation methods rely on two types of global conditions to specify available/missing modalities:
- **Global binary vectors**: e.g., M2DN uses [1,1,0] to indicate modality availability, but cannot capture region-level and cross-modality variation.
- **Learnable modality queries**: e.g., MMT uses modality-specific decoders, whose parameter count grows with the number of modalities, resulting in poor generalisation.

Both approaches essentially perform pixel-level modality translation and lack compact modelling of cross-modality spatial relationships.

### 3. Core Insight
Theoretically, different MRI modalities of the same subject share underlying spin properties at the pixel level (transferable); empirically, SynthSeg demonstrates that different modalities share structural priors (shared). It is therefore possible to recast the complex any-to-any imputation problem as a simpler **region-level code prediction** task—predicting a compact full-stack representation rather than synthesising each modality independently.

## Method

### Overall Architecture

CodeBrain adopts a two-stage pipeline:

**Stage I (Compact Representation Learning)**: Learns a compact representation of complete brain MRI by encoding the full modality set into region-level scalar quantised codes plus modality-agnostic common features, then decoding back to the full MRI.

**Stage II (Code Prediction)**: Trains a prior encoder to predict the full-stack quantised codes from incomplete modalities, supervised by a grading loss.

**Inference**: Given incomplete input → prior encoder predicts full-stack codes → concatenated with common features → decoder synthesises missing modalities.

### Key Designs

#### 1. Finite Scalar Quantisation (FSQ)

The core idea is to eliminate the dependency on explicit learnable codebooks required by conventional VQ. The posterior encoder $E_{\text{posterior}}$ encodes the complete MRI $M_{\text{full}}$ into a feature map $F_{\text{full}}$, which is then element-wise quantised:

$$Z_{\text{full},i} = \lfloor L_i/2 \rfloor \times \tanh(F_{\text{full},i})$$
$$\hat{Z}_{\text{full}} = \text{round}(Z_{\text{full}})$$

Each channel $i$ has $L_i$ integer levels in the range $[-\lfloor L_i/2\rfloor, \lfloor L_i/2\rfloor]$. A straight-through estimator is used to enable gradient flow. Experiments use $L=[8,8,8,5,5,5]$ with $d=6$ channels, yielding a total codebook size of $\prod L_i = 64000$.

**Design Advantages**: FSQ requires no explicit codebook learning, avoiding codebook collapse and auxiliary regularisation losses, resulting in efficient and stable training.

#### 2. Dual-Component Bottleneck Representation

A distinctive aspect of Stage I is the decomposition of the complete MRI into two complementary components:
- **Full-stack quantised codes** $\hat{Z}_{\text{full}}$: Capture modality-specific region-level features, extracted from $M_{\text{full}}$ by $E_{\text{posterior}}$.
- **Common features** $F_c$: Capture modality-agnostic anatomical information, extracted from **arbitrary incomplete input** $M_{\text{inc}}$ by a shared encoder $E_c$.

Reconstruction: $\tilde{M}_{\text{full}} = D(\text{Concat}[\hat{Z}_{\text{full}}, F_c])$

During training, random modality masking is applied to $M_{\text{inc}}$ ($K$ modalities set to zero), forcing $F_c$ to learn modality-invariant features.

#### 3. Code Prediction via Grading Loss

In Stage II, the prior encoder $E_{\text{prior}}$ predicts the full-stack codes $\tilde{Z}_{\text{full}}$ from $M_{\text{inc}}$, but does not use a standard classification loss.

**Problem**: Cross-entropy assumes all quantised codes are independent and equidistant, ignoring the clustering structure in quantisation space where adjacent codes correspond to semantically similar patches.

**Solution**: Code prediction is treated as an ordinal regression problem. For the ground-truth label $y_i$ of channel $i$, an ordered grading array $o^i$ is constructed:

$$o^i_j = \begin{cases} 1 & \text{if } j < y_i \\ 0 & \text{else} \end{cases}$$

$y_i$ can be recovered by summing $o^i$. The model is then trained with a binary cross-entropy loss:

$$\mathcal{L}_{\text{grad}} = \mathcal{L}_{\text{bce}}(\tilde{O}_{\text{full}}, \hat{O}_{\text{full}})$$

**Advantage**: This explicitly encodes the clustering structure of quantisation space, enabling smoother code transitions and improving prediction accuracy.

#### 4. Comparison of Conditioning Designs

The paper systematically compares four conditioning designs: fixed binary condition → learnable global condition → **region-level quantised codes (CodeBrain)** → unlimited continuous variables. Results show that continuous variables are overly complex and lead to imputation degradation, while quantised codes achieve the best balance between expressiveness and tractability.

### Loss & Training

**Stage I Loss**:
$$\mathcal{L}_{\text{rec}} = \sum_{i=0}^{N-1} \lambda_{[m,a]} \times \mathcal{L}_{\text{psnr}}(\tilde{M}_i, M_i) + \mathcal{L}_{\text{gan}}(\tilde{M}, M)$$

- $\mathcal{L}_{\text{psnr}}$: Differentiable PSNR approximation loss
- $\mathcal{L}_{\text{gan}}$: LSGAN $\ell_2$ adversarial loss
- $\lambda_m=20$ (missing modality weight), $\lambda_a=5$ (available modality weight)

**Stage II Loss**: $\mathcal{L}_{\text{grad}}$ (grading binary cross-entropy)

**Training Configuration**:
- Backbone: NAFNet
- Optimiser: AdamW, lr=1e-4
- Batch size: 48
- 300 epochs per stage, 8×4090 GPUs, total training time: 2.38 days

## Key Experimental Results

### Main Results

**Table 1: Imputation Results Under Different Scenarios on IXI Dataset (PSNR dB)**

| Scenario | T1 Missing | T2 Missing | PD Missing |
|----------|-----------|-----------|-----------|
| One-to-one range | 23.61–28.51 | 28.08–30.08 | 27.10–33.42 |
| Two-to-one | 28.95 | 31.08 | 34.65 |
| Average imputation | 28.51 | 28.26 | 31.72 |

PD is the easiest to synthesise from other modalities; T2 from T1 is the hardest (reflecting clinical differences). Many-to-one settings consistently outperform one-to-one.

**Table 2: Cross-Method Comparison (IXI + BraTS 2023)**

| Method | IXI PSNR | IXI SSIM(%) | BraTS PSNR | BraTS SSIM(%) |
|--------|----------|-------------|------------|---------------|
| MMGAN | 27.64 | 90.84 | 24.28 | 89.11 |
| MMT | 28.06 | 91.42 | 24.58 | 89.47 |
| M2DN | 28.14 | 91.80 | 24.34 | 89.65 |
| Zhang et al. | 29.00 | 92.63 | 25.01 | 89.98 |
| MMHVAE | 28.11 | 91.20 | 24.29 | 88.83 |
| **CodeBrain** | **29.50** | **93.05** | **25.31** | **90.49** |

CodeBrain leads comprehensively on both datasets, with gains of +0.50 dB PSNR and +0.42% SSIM on IXI (without structural loss supervision).

### Ablation Study

**Table 3: Ablation Study (IXI Mean)**

| Configuration | Reconstruction PSNR | Imputation PSNR |
|---------------|--------------------|--------------------|
| Without common feature $F_c$ | 30.15 | — |
| With $F_c$ | **34.32** (+4.17) | — |
| Classification loss prediction | — | Baseline |
| Grading loss prediction | — | **Superior** |

The common feature contributes +4.17 dB PSNR; the grading loss outperforms the classification loss.

**Downstream Task Validation (BraTS Brain Tumour Segmentation, 3D Dice)**:
- Zero-filling missing modalities: severe performance degradation (complete failure without FLAIR)
- CodeBrain imputation > other method imputation
- CodeBrain imputation ≈ upper bound with all real modalities (row 5 vs. row 6)

### Key Findings

1. **Spontaneous clustering of quantised code distributions**: Without any regularisation, code distributions exhibit clustering that loosely reflects brain anatomical structure.
2. **Region-level conditioning outperforms global conditioning**: Quantised codes achieve the best balance between expressiveness and tractability.
3. **Sensitivity to $\lambda_m/\lambda_a$ ratio**: 20/5 is optimal; performance degrades when the ratio is too large or too small.
4. **Stage II accurately predicts most codes**: Visualisation confirms high agreement between predicted codes and ground-truth codes.

## Highlights & Insights

1. **Paradigm Innovation**: Recasting any-to-any modality translation as region-level code prediction avoids modality-specific design, yielding an elegant and unified framework.
2. **Successful Application of FSQ in Medical Imaging**: Demonstrates the effectiveness of codebook-free scalar quantisation for cross-modality MRI modelling, reducing the training complexity of VQ.
3. **Elegant Introduction of Grading Loss**: Ordinal regression naturally fits the continuous semantic structure of quantisation space, outperforming independent classification.
4. **Imputation Quality Directly Improves Downstream Tasks**: Beyond visual quality, BraTS segmentation performance approaches that of full real modalities, demonstrating genuine clinical value.

## Limitations & Future Work

1. **2D Slice Processing**: The current approach operates at the 2D level and does not exploit 3D volumetric information, potentially losing inter-slice continuity.
2. **Hallucination Risk**: Despite outperforming competitors, synthesised images may still exhibit artefacts, particularly in T1Gd contrast-enhanced regions.
3. **Validated Only on Brain MRI**: Generalisation to other anatomical regions (cardiac, abdominal, etc.) has not been verified.
4. **Fixed Quantisation Levels**: $L=[8,8,8,5,5,5]$ is manually specified; adaptive selection may further improve performance.
5. **MRI Physics Not Incorporated**: Physical priors such as the contrast enhancement mechanism of T1Gd are not exploited; integrating them could improve synthesis of specific modalities.

## Related Work & Insights

- **FSQ → Medical Imaging**: Google's FSQ was originally designed for image generation; CodeBrain demonstrates its effectiveness for cross-modality medical imaging modelling.
- **Extension of the VQGAN Paradigm**: Stage I resembles VQGAN's encode-quantise-decode pipeline, but Stage II replaces autoregressive generation with code prediction.
- **Complementary to SynthSeg**: SynthSeg leverages cross-modality shared structure for robust segmentation; CodeBrain exploits the same prior for modality synthesis.

## Rating

- Novelty: ⭐⭐⭐⭐ Paradigm innovation in recasting any-to-any imputation as code prediction; grading loss design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two datasets, nine scenarios, five baselines, complete ablation and downstream validation.
- Writing Quality: ⭐⭐⭐⭐ Clear figures; coherent logic from motivation to method to experiments.
- Value: ⭐⭐⭐⭐ Provides a practical framework for unified MRI modality imputation that directly improves downstream clinical task performance.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] CodeBrain: Virtual Full-stack Scanning of Brain MRI via Imputing Any Quantised Code](virtual_fullstack_scanning_of_brain_mri_via_imputi.md)
- [\[CVPR 2026\] PGR-Net: Prior-Guided ROI Reasoning Network for Brain Tumor MRI Segmentation](pgr-net_prior-guided_roi_reasoning_network_for_brain_tumor_mri_segmentation.md)
- [\[CVPR 2026\] Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification](diffusionbased_feature_denoising_and_using_nnmf_fo.md)
- [\[CVPR 2026\] Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation](multiscale_structure-guided_latent_diffusion_for_multimodal_mri_translation.md)
- [\[CVPR 2026\] Continual Learning for fMRI-Based Brain Disorder Diagnosis via Functional Connectivity Matrices Generative Replay](forge_continual_learning_for_fmri_based_brain_disorder_diagnosis.md)

<!-- RELATED:END -->

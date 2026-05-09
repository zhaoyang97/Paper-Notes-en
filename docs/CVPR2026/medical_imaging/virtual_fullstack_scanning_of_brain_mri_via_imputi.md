---
title: >-
  [Paper Note] CodeBrain: Virtual Full-stack Scanning of Brain MRI via Imputing Any Quantised Code
description: >-
  [CVPR 2026][Medical Imaging][brain MRI] CodeBrain reformulates any-to-any brain MRI modality imputation as a region-level full-stack quantised code prediction problem. Stage I encodes complete MRI sets into compact code maps and modality-agnostic common features via Finite Scalar Quantisation (FSQ); Stage II predicts code maps from incomplete modalities using a grading loss to preserve the smoothness of the quantisation space. CodeBrain surpasses five SOTA methods on IXI and BraTS 2023, and the synthesised modalities achieve brain tumour segmentation performance approaching that of real data.
tags:
  - CVPR 2026
  - Medical Imaging
  - brain MRI
  - modality imputation
  - scalar quantisation
  - any-to-any synthesis
  - VQ-VAE
date: 2026-05-08
content_hash: e20e75b41a905d78
---

# CodeBrain: Virtual Full-stack Scanning of Brain MRI via Imputing Any Quantised Code

**Conference**: CVPR 2026
**arXiv**: [2501.18328](https://arxiv.org/abs/2501.18328)
**Code**: [GitHub](https://github.com/ycwu1997/CodeBrain)
**Area**: Medical Imaging / MRI Synthesis
**Keywords**: brain MRI, modality imputation, scalar quantisation, any-to-any synthesis, VQ-VAE

## TL;DR
CodeBrain reformulates any-to-any brain MRI modality imputation as a region-level full-stack quantised code prediction problem. Stage I encodes complete MRI sets into compact code maps and modality-agnostic common features via Finite Scalar Quantisation (FSQ); Stage II predicts code maps from incomplete modalities using a grading loss to preserve the smoothness of the quantisation space. CodeBrain surpasses five SOTA methods on IXI and BraTS 2023, and the synthesised modalities achieve brain tumour segmentation performance approaching that of real data.

## Background & Motivation

Brain MRI encompasses multiple modalities (T1/T2/PD/FLAIR/T1Gd), each highlighting distinct anatomical or pathological features. However, acquiring a complete set is clinically impractical due to scan time, cost, and contrast agent risks. Limitations of existing unified imputation methods: (1) global conditioning (binary vectors / learnable queries) fails to capture region-level cross-modal variation; (2) modality-specific modules (one decoder per modality) scale parameter counts linearly with the number of modalities, resulting in poor generalisation; (3) pixel-level translation lacks explicit modelling of cross-modal relationships.

## Core Problem

How to build a unified model capable of imputing any missing MRI modalities from an arbitrary subset of available modalities (any-to-any), without relying on modality-specific modules?

## Method

### Overall Architecture

A two-stage pipeline:

**Stage I (learning compact representations):** Complete MRI set → posterior encoder $E_\text{posterior}$ → FSQ quantisation → full-stack code map $\hat{Z}_\text{full}$ ($d=6$, spatial resolution $h \times w$). In parallel, incomplete input → common encoder $E_c$ → modality-agnostic feature $F_c$. Decoder $D$ reconstructs the complete MRI from $[\hat{Z}_\text{full}, F_c]$.

**Stage II (learning code prediction):** Incomplete input → prior encoder $E_\text{prior}$ → predicted code map $\tilde{Z}_\text{full}$, supervised by a grading loss (targets derived from Stage I).

**Inference:** Incomplete input → $E_\text{prior}$ predicts codes → $E_c$ extracts common features → $D$ decodes and outputs imputed results.

### Key Designs

1. **FSQ-based region-level representation**: The complete MRI is encoded into a $d=6$-channel code map, where the 6-dimensional code at each spatial position represents the full-stack features of the corresponding image patch. Values are quantised to $L=[8,8,8,5,5,5]$ discrete levels (codebook size 64K) using a rounding operation with a straight-through estimator for end-to-end training. Compared to VQ-VAE codebook learning, FSQ requires no explicit codebook, directly rounding activations to finite integers for more stable training. Key advantage: all complete MRI samples are projected into the same finite quantisation space, reducing cross-modal synthesis to a discrete code prediction problem.

2. **Modality-agnostic common feature $F_c$**: Extracted from any incomplete input subset; modalities are randomly masked ($0 < K < N$) during training to ensure $F_c$ does not depend on any specific modality. $F_c$ compensates for high-frequency details lost through quantisation — removing $F_c$ causes a 4.17 dB drop in PSNR. At decoding time, $[\hat{Z}_\text{full}, F_c]$ is concatenated as input to $D$.

3. **Grading loss for code prediction**: Discrete code prediction is formulated as ordinal regression rather than classification — each scalar value is converted into an ordered binary decision sequence $o=[1,1,\dots,1,0,\dots,0]$, trained with BCE loss. Compared to cross-entropy, this preserves the semantic similarity between adjacent codes in the quantisation space (CE assumes all classes are equidistant). Ablation confirms grading outperforms classification (PSNR 29.50 vs. 29.24).

### Loss & Training

**Stage I:** $L_\text{rec} = \lambda_{[m,a]} \times L_\text{psnr} + L_\text{gan}$ (LSGAN), with $\lambda_m=20$ (higher weight for missing modalities) and $\lambda_a=5$.

**Stage II:** $L_\text{grad} = \text{BCE}(\tilde{O}, \hat{O})$.

NAFNet backbone, AdamW with lr $= 10^{-4}$, batch size 48, 300 epochs per stage, trained on 8× RTX 4090 GPUs for a total of 2.38 days.

## Key Experimental Results

### IXI Dataset (T1/T2/PD, averaged over 9 scenarios)

| Method | PSNR↑ | SSIM↑ | MAE↓ |
|--------|-------|-------|------|
| MMGAN | 27.94 | 91.22 | 20.75 |
| MMT | 28.38 | 91.79 | 19.55 |
| M2DN | 28.40 | 92.08 | 19.18 |
| Zhang et al. | 29.31 | 92.96 | 17.83 |
| **CodeBrain** | **29.84** | **93.35** | **16.97** |

### BraTS 2023 (T1/T2/FLAIR/T1Gd, four modalities)

| Method | PSNR↑ | SSIM↑ | MAE↓ |
|--------|-------|-------|------|
| Zhang et al. | 24.95 | 89.90 | 21.68 |
| **CodeBrain** | **25.26** | **90.42** | **21.14** |

### Downstream Brain Tumour Segmentation (3D Dice %)
Imputed T1Gd by CodeBrain yields segmentation Dice of 61.44%, compared to 87.40% with full modalities and only 6.2% with zero-filling. T1Gd is the most challenging modality to impute as it requires contrast enhancement.

### Ablation Study

- **Removing $F_c$**: Reconstruction PSNR drops from 34.32 to 30.15 (−4.17 dB), demonstrating the critical role of common features in recovering fine-grained details.
- **Classification vs. Grading**: Grading outperforms classification (PSNR 29.50 vs. 29.24).
- **PSNR loss vs. L1 loss**: PSNR loss achieves slightly higher PSNR (34.32 vs. 34.08) with a marginal SSIM trade-off.
- **Quantised vs. continuous conditioning**: Discrete codes outperform continuous variables as conditions (infinite continuous conditioning leads to a significant performance drop), suggesting quantisation provides a favourable balance between expressiveness and predictability.

## Highlights & Insights

- **Recasting image translation as discrete code prediction**: This is the core insight — it is simpler and more unified than direct pixel-to-pixel translation. After quantisation, all modalities share the same finite space, eliminating the need for modality-specific modules.
- **FSQ is more robust than VQ-VAE**: No codebook learning is required (avoiding codebook collapse); values are simply rounded to integers with STE, yielding an engineering-clean solution.
- **Grading loss preserves quantisation space structure**: Ordinal regression retains the relational ordering (e.g., "3 is greater than 2 but less than 4"), making it a well-suited approach for ordered discrete prediction.
- **Design of the modality-agnostic common feature**: Random modality masking during training enables $F_c$ extraction from any arbitrary subset — an elegant and simple design.

## Limitations & Future Work

- Processing is performed on 2D slices only, without exploiting 3D spatial context (acknowledged by the authors in the limitations section).
- Imputation quality for T1Gd remains substantially below real data (Dice 61.44% vs. 87.40%); contrast-enhanced modalities cannot be fully inferred from non-enhanced ones.
- Hallucination artefacts may still be present.
- Training requires 2.38 days on 8× RTX 4090 GPUs, representing a non-trivial computational cost.
- Validation is limited to brain MRI; generalisation to other anatomical regions (e.g., cardiac, abdominal) remains unknown.

## Related Work & Insights

- **vs. MMT/M2DN**: These methods rely on global conditioning (binary vectors / modality queries) and cannot capture region-level variation. CodeBrain's quantised codes carry spatially distinct values at each patch position.
- **vs. Zhang et al. [49]**: Uses dual modality-shared and modality-specific encoders, with parameter counts scaling with the number of modalities. CodeBrain is entirely modality-agnostic.
- **vs. VQGAN/VQ-VAE**: These are designed for single-modality generation. CodeBrain extends vector quantisation to cross-modal translation and replaces traditional VQ with FSQ to avoid codebook collapse.

- The paradigm of *recasting cross-modal translation as discrete code prediction* may transfer to other multimodal generation tasks.
- Grading loss is applicable to any scenario requiring prediction of ordered discrete values.
- There is a conceptual connection to tokenisation in vision-language models.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Reframing MRI imputation as code prediction is a novel perspective; the FSQ + grading loss combination is effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Two datasets, multiple scenarios (one-to-one / many-to-one), detailed ablations ($F_c$, grading, loss function, conditioning type, loss weights), and downstream segmentation validation.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear and complete, with informative figures and well-motivated method design.
- **Value**: ⭐⭐⭐ — Medical imaging is not a core research direction, but the idea of using quantised codes for cross-modal translation is insightful.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Virtual Full-stack Scanning of Brain MRI via Imputing Any Quantised Code](virtual_full-stack_scanning_of_brain_mri_via_imputing_any_quantised_code.md)
- [\[CVPR 2026\] PGR-Net: Prior-Guided ROI Reasoning Network for Brain Tumor MRI Segmentation](pgr-net_prior-guided_roi_reasoning_network_for_brain_tumor_mri_segmentation.md)
- [\[CVPR 2026\] Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification](diffusionbased_feature_denoising_and_using_nnmf_fo.md)
- [\[CVPR 2026\] Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation](multiscale_structure-guided_latent_diffusion_for_multimodal_mri_translation.md)
- [\[CVPR 2026\] Continual Learning for fMRI-Based Brain Disorder Diagnosis via Functional Connectivity Matrices Generative Replay](forge_continual_learning_for_fmri_based_brain_disorder_diagnosis.md)

</div>

<!-- RELATED:END -->

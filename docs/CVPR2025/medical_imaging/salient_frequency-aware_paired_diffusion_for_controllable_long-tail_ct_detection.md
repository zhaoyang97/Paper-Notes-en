---
title: >-
  [Paper Note] SALIENT: Frequency-Aware Paired Diffusion for Controllable Long-Tail CT Detection
description: >-
  [CVPR2025][Medical Imaging][Diffusion models] Proposes SALIENT, a mask-conditioned generative framework based on wavelet-domain diffusion. Through frequency-aware, interpretable optimization objectives and paired lesion-mask volume generation, it achieves controllable and efficient synthetic data augmentation and precision recovery in long-tail CT detection. It systematically characterizes the augmentation dose-response curve for the first time.
tags:
  - "CVPR2025"
  - "Medical Imaging"
  - "Diffusion models"
  - "wavelet domain"
  - "long-tail detection"
  - "CT augmentation"
  - "synthetic data"
  - "class imbalance"
  - "dose-response"
date: 2026-05-08
content_hash: 1f24e797acf4d815
---

# SALIENT: Frequency-Aware Paired Diffusion for Controllable Long-Tail CT Detection

**Conference**: CVPR2025  
**arXiv**: [2602.23447](https://arxiv.org/abs/2602.23447)  
**Code**: Planned release (Non-commercial academic research license)  
**Area**: Medical Image  
**Keywords**: Diffusion models, wavelet domain, long-tail detection, CT augmentation, synthetic data, class imbalance, dose-response

## TL;DR

Proposes SALIENT, a mask-conditioned generative framework based on wavelet-domain diffusion. Through frequency-aware, interpretable optimization objectives and paired lesion-mask volume generation, it achieves controllable and efficient synthetic data augmentation and precision recovery in long-tail CT detection. It systematically characterizes the augmentation dose-response curve for the first time.

## Background & Motivation

- **Two Major Failure Modes of Long-Tail Detection**: (1) Intra-patient signal dilution—low target volume ratio (TVR), where lesions occupy a minuscule proportion of the field-of-view in whole-body CT scans; (2) Cross-dataset class imbalance—extreme long-tail distribution of rare lesions.
- **Precision Ceiling**: Even with a high AUROC, models exhibit low precision, poor AUPRC, and unstable F1 scores in detecting rare lesions, limiting clinical credibility.
- **Limitations of Prior Work in Diffusion Models**:
    - Pixel-space DDPMs in 3D incur prohibitive computational costs, requiring aggressive downsampling that leads to loss of details.
    - Existing mask-conditioned methods lack attribute-level controllable adjustment and paired supervision.
    - Frequency-domain diffusion methods rely on manual weight tuning and cannot disentangle interpretable image attributes.
- **Unexplored Augmentation Dose Effect**: Synthetic augmentation typically assumes monotonic benefits, but the optimal "therapeutic dose" and optimal-exceeding "toxic dose" have not been systematically characterized.

## Method

### 1. Wavelet-Domain Diffusion Framework
- Applies a one-level Haar Discrete Wavelet Transform (DWT) to each CT slice to obtain four sub-bands: LL (low-frequency structure/brightness) and LH/HL/HH (high-frequency edges/texture details).
- Conducts diffusion in the wavelet coefficient space rather than the pixel space to explicitly separate global brightness from high-frequency structural details.
- Conditioning signals include: downsampled lesion masks + wavelet features of adjacent slices (2.5D context).

### 2. Mask-Gated Frequency Scaling (FSA)
- Performs mask-gated frequency modulation on noisy wavelet coefficients before UNet input to generate modulated coefficients.

### 3. Loss & Training
- **Wavelet Sub-band Weighted Reconstruction Loss**: Imposed with higher weights near lesion boundaries to suppress HH channel amplification.
- **Low-Frequency Moment Regularization ($L_{LL}$)**: Constrains the mean and variance of the LL sub-band to match the real distribution, preventing brightness drift.
- **High-Frequency Variance Control ($L_{HF}$)**: Constrains the variance of LH/HL/HH sub-bands to match real data, preserving texture fidelity.
- Total loss: $L = L_{\text{wavelet}} + L_{LL} + L_{HF} + L_{\text{aux}}$

### 4. Structured Classifier-Free Guidance (CFG)
- Three forward passes: unconditional, mask-only conditional, and mask + adjacent-slice conditional.
- The adj-slice guidance strength decays along diffusion steps, emphasizing global anatomical consistency in early stages and focusing on lesion refinement in later stages.

### 5. 3D VAE Volumetric Lesion Mask Generation
- Trains MaskVAE3D to learn the latent pathological manifold from limited positive samples, sampling to generate morphologically diverse 3D lesion masks.

### 6. Semi-Supervised Segmentation Pairing
- Generates paired pseudo-segmentation masks for synthetic CT using UCMT (Uncertainty-Aware Cross-Model Training).
- Ultimately outputs paired (synthetic CT, lesion mask) for downstream mask-guided detection training.

## Key Experimental Results

### Dataset
- 5205 contrast-enhanced whole-body CT cases (200 mediastinal hematoma positive, 5005 negative), exhibiting a natural long-tail distribution (~3% positivity rate).
- Unified preprocessing: orientation alignment, resampling, soft-tissue HU windowing, intensity normalization, and anatomical cropping of the mediastinum region.

### Generation Quality

| Method | MS-SSIM↑ | FID↓ |
|------|:---:|:---:|
| MedDDPM (Pixel Space) | 0.63 | 118.4 |
| **SALIENT (Wavelet Domain)** | **0.83** | **46.5** |

- Segmentation Fidelity: Dice $0.72 \pm 0.24$ vs $0.27 \pm 0.16$ for MedDDPM.
- Computational Acceleration: $4\times$ faster than 2.5D MedDDPM and $28\times$ faster than 3D MedDDPM.

### Augmentation Dose-Response Experiments

| Seed Size | Optimal Dose | 1% Prevalence $\Delta$AUPRC | Remarks |
|:---:|:---:|:---:|:---:|
| $n=50$ | $2\times$ | +0.0605 | Stable therapeutic window |
| $n=25$ | $4\times$ | +0.12 | Dose shifts right, larger gain |

- **Key Findings**: As labeled seed size decreases, the optimal dose shifts right ($2\times \to 4\times$) and the AUPRC gain becomes larger, suggesting a seed-dependent augmentation mechanism under low-label conditions.
- AUROC remains consistently high, indicating that the core contribution of SALIENT is precision recovery (AUPRC) rather than mere separability inflation.
- Excessive synthetic augmentation ($10\times$) leads to performance degradation, confirming the existence of a "toxic dose."

### Radiologists Blind Evaluation
- 5-point Likert scale evaluation: SALIENT outperforms MedDDPM in brightness/contrast realism, lesion-to-background fusion, high-frequency artifact suppression, and mask fidelity.

## Highlights & Insights

1. **Elegant Design of Wavelet-Domain Diffusion**: Combining frequency decomposition with diffusion models provides interpretable, attribute-level control "knobs" (brightness, structure, edge, and contrast) while substantially reducing 3D computational cost.
2. **Paired Generation**: Generates paired (synthetic CT, mask) data end-to-end, directly supporting mask-guided detection training workflows without requiring manual annotation.
3. **Augmentation Dose-Response Analysis**: Systematically characterizes the "therapeutic window" and "toxic dose" of synthetic augmentation for the first time, identifying the seed-dependent dose-shift pattern as a methodological contribution.
4. **Full-Pipeline Design**: Houses a complete pipeline from 3D mask generation $\to$ wavelet-domain synthesis $\to$ semi-supervised pairing $\to$ mask-guided detection $\to$ subject-level aggregation.
5. **Remarkable Precision Recovery**: Elevates AUPRC by 0.12 under extreme long-tail conditions (1% prevalence) while keeping AUROC consistently high.

## Limitations & Future Work

1. Evaluated on a single lesion type (mediastinal hematoma); generalization to other rare lesions (e.g., small nodules, micrometastases, microbleeds) remains to be demonstrated.
2. Employs single-level Haar wavelets; more complex multi-level or learnable wavelet transforms might further enhance reconstruction quality.
3. The semi-supervised paired masks rely on the quality of pseudo-labels from UCMT, meaning pseudo-label errors can propagate to downstream detection.
4. The "therapeutic window" of the augmentation dose-response is an empirical discovery, lacking a theoretical explanation of why $2\times$ or $4\times$ is optimal.
5. Evaluated solely on single-center data; generalization across multi-center, multi-device, and multi-protocol setups requires further verification.
6. The morphological diversity of 3D VAE mask generation is limited by the number of positive samples in the training set (only 200 positive cases).

## Rating
- Novelty: 4/5 — The combination of wavelet-domain diffusion, frequency-aware optimization objectives, and dose-response analysis is innovative. The structured CFG is also solid.
- Experimental Thoroughness: 4/5 — Comprehensive coverage of generation quality (MS-SSIM/FID), sub-band analysis, radiologists' blind evaluation, downstream detection, and the dose-response curve, though a single lesion type limits the generalizability of the conclusions.
- Writing Quality: 4/5 — Well-structured and richly illustrated (especially the informative frequency analysis figures), though some notation definitions are scattered across sections.
- Value: 4/5 — Provides a systematic and controllable synthetic augmentation solution for long-tail medical image detection, with the augmentation dose-response analysis framework offering methodological value.

<!-- Experimental hardware: NVIDIA H100, training uses AdamW + cosine lr decay + EMA -->
<!-- All positive samples are included in every epoch, negative samples are balanced-sampled -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] GuideGen: A Text-Guided Framework for Paired Full-Torso Anatomy and CT Volume Generation](../../AAAI2026/medical_imaging/guidegen_a_text-guided_framework_for_paired_full-torso_anatomy_and_ct_volume_gen.md)
- [\[CVPR 2025\] SeaLion: Semantic Part-Aware Latent Point Diffusion Models for 3D Generation](sealion_semantic_part-aware_latent_point_diffusion_models_for_3d_generation.md)
- [\[ICML 2025\] Context Matters: Query-aware Dynamic Long Sequence Modeling of Gigapixel Images](../../ICML2025/medical_imaging/context_matters_query-aware_dynamic_long_sequence_modeling_of_gigapixel_images.md)
- [\[ICCV 2025\] Controllable Latent Space Augmentation for Digital Pathology](../../ICCV2025/medical_imaging/controllable_latent_space_augmentation_for_digital_pathology.md)
- [\[CVPR 2025\] Novel Architecture of RPA In Oral Cancer Lesion Detection](novel_architecture_of_rpa_in_oral_cancer_lesion_detection.md)

</div>

<!-- RELATED:END -->

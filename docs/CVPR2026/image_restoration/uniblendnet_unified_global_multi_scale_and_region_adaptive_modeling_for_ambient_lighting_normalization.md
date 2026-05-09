---
title: >-
  [Paper Note] UniBlendNet: Unified Global, Multi-Scale, and Region-Adaptive Modeling for Ambient Lighting Normalization
description: >-
  [CVPR 2026][Image Restoration][ambient lighting normalization] This paper proposes UniBlendNet, which builds upon IFBlend to unify three complementary modules—global context modeling, multi-scale feature aggregation, and region-adaptive residual refinement—for ambient lighting normalization under complex spatially varying illumination conditions.
tags:
  - CVPR 2026
  - Image Restoration
  - ambient lighting normalization
  - shadow removal
  - multi-scale aggregation
  - mask-guided refinement
  - frequency-spatial restoration
date: 2026-05-08
content_hash: 57f6e0363573f411
---

# UniBlendNet: Unified Global, Multi-Scale, and Region-Adaptive Modeling for Ambient Lighting Normalization

**Conference**: CVPR 2026
**arXiv**: [2604.13383](https://arxiv.org/abs/2604.13383)
**Code**: None
**Area**: Image Restoration
**Keywords**: ambient lighting normalization, shadow removal, multi-scale aggregation, mask-guided refinement, frequency-spatial restoration

## TL;DR

This paper proposes UniBlendNet, which builds upon IFBlend to unify three complementary modules—global context modeling, multi-scale feature aggregation, and region-adaptive residual refinement—for ambient lighting normalization under complex spatially varying illumination conditions.

## Background & Motivation

Ambient Lighting Normalization (ALN) aims to restore images degraded by complex spatially varying illumination arising from interactions among multiple light sources, object geometry, and material properties. Existing methods such as IFBlend leverage frequency-domain priors to model lighting variations, yet exhibit three key limitations: (1) limited global context modeling, preventing the capture of scene-level long-range lighting dependencies; (2) spatially uniform residual correction, leading to over-enhancement in bright regions and under-correction in dark regions; and (3) lack of adaptive multi-scale feature aggregation strategies to handle shadow and illumination inconsistencies at different spatial scales. These three limitations motivate the authors to design a unified framework that simultaneously addresses all three aspects.

## Method

### Overall Architecture

UniBlendNet is built upon the encoder-decoder frequency-spatial joint restoration backbone of IFBlend, introducing three complementary components: (1) a UniConvNet-based global context branch that extracts global features in parallel with the main backbone; (2) a Scale-Aware Aggregation Module (SAAM) inserted at the bottleneck layer; and (3) a mask-guided residual refinement mechanism. The final restoration formula is $\mathbf{I}_r = \mathbf{I}_{inp} + \mathbf{M} \odot \mathbf{R}$, where $\mathbf{M}$ is a soft guidance mask and $\mathbf{R}$ is the predicted residual.

### Key Designs

1. **UniConvNet Global Context Modeling**: UniConvNet is employed to expand the effective receptive field through progressive aggregation with increasing kernel sizes, extracting global context features $\mathbf{F}_g$ directly from the input image and fusing them with the decoder's final features for residual prediction. This supplements the frequency-spatial backbone's limited capacity for long-range dependency modeling.

2. **Scale-Aware Aggregation Module (SAAM)**: A three-level pyramid (original scale, $2\times$ downsampling, $4\times$ downsampling) is constructed at the bottleneck layer, where shared-weight convolutional branches process each scale before upsampling back to the original resolution. Global average pooling computes a global descriptor, and a lightweight MLP predicts dynamic scale weights; residual fusion then yields the final bottleneck feature. This enables the network to dynamically emphasize the most informative scale.

3. **Mask-Guided Residual Refinement**: Two independent prediction heads are designed: a mask prediction head outputs a continuous soft guidance mask $\mathbf{M} \in [0,1]$ via sigmoid to control the residual correction intensity at each spatial location, while a residual prediction head fuses decoder features and global context features to generate the residual $\mathbf{R}$. Pseudo-binary masks are constructed from the relative grayscale difference between degraded and clean images for supervision.

### Loss & Training

A multi-objective loss function is employed for joint training: $\mathcal{L} = \mathcal{L}_{rec} + \alpha_1 \mathcal{L}_{ssim} + \alpha_2 \mathcal{L}_{grad} + \alpha_3 \mathcal{L}_{perc} + \lambda \mathcal{L}_{mask}$, comprising an L1 reconstruction loss, SSIM structural similarity loss, gradient consistency loss, perceptual loss, and mask L1 supervision loss. Pseudo-masks are constructed by thresholding the positive relative grayscale difference.

## Key Experimental Results

### Main Results

Evaluation is conducted on the NTIRE Ambient Lighting Normalization benchmark. UniBlendNet consistently outperforms the IFBlend baseline on both PSNR and SSIM, with more natural and stable visual results.

| Metric | IFBlend (Baseline) | UniBlendNet | Gain |
|--------|-------------------|-------------|------|
| PSNR | Baseline | Higher | Consistent improvement |
| SSIM | Baseline | Higher | Consistent improvement |

### Ablation Study

Ablation studies confirm the individual contributions of the three key components: global illumination modeling, scale-aware feature aggregation, and region-adaptive residual refinement each yield incremental improvements.

### Key Findings

- The three modules collaborate in a complementary manner to jointly improve illumination consistency and structural fidelity.
- The mask-guided mechanism effectively enables spatially adaptive restoration, avoiding the pitfalls of globally uniform correction.
- The dynamic weight learning in SAAM allows the network to adaptively select the most informative scale based on the input.

## Highlights & Insights

- Decomposing the ALN problem into a unified global–multi-scale–local hierarchical framework yields a clear and principled design.
- The mask-guided residual refinement paradigm of "where to restore and how much to restore" is broadly generalizable.
- SAAM's dynamic scale weighting eliminates the need for manually designed scale fusion strategies.

## Limitations & Future Work

- Experiments are conducted on a single ALN benchmark only, leaving generalization capability unverified.
- The inference speed overhead introduced by the increased model complexity is not discussed.
- Light source estimation and separation in multi-light-source scenes remain open problems.

## Related Work & Insights

- IFBlend's frequency-domain priors provide an effective foundation for lighting normalization.
- The large-kernel aggregation strategy of UniConvNet can be extended to other low-level vision tasks requiring large receptive fields.
- The mask-guided region-adaptive idea can be transferred to other restoration tasks involving spatially non-uniform degradation.

## Rating

6/10 — The method is well-designed with three complementary modules, but the contributions are incremental and validation is limited to a single benchmark.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] UniRain: Unified Image Deraining with RAG-based Dataset Distillation and Multi-objective Reweighted Optimization](unirain_unified_image_deraining_with_rag_based_dataset_distillation_and_multi_obje.md)
- [\[NeurIPS 2025\] MS-BART: Unified Modeling of Mass Spectra and Molecules for Structure Elucidation](../../NeurIPS2025/image_restoration/ms-bart_unified_modeling_of_mass_spectra_and_molecules_for_structure_elucidation.md)
- [\[CVPR 2026\] IA-CLAHE: Image-Adaptive Clip Limit Estimation for CLAHE](ia_clahe_image_adaptive_clip_limit.md)
- [\[CVPR 2026\] RAR: Restore, Assess, Repeat - A Unified Framework for Iterative Image Restoration](rar_restore_assess_repeat_a_unified_framework_for_iterative_image_restoration.md)
- [\[ICCV 2025\] Learning Pixel-adaptive Multi-layer Perceptrons for Real-time Image Enhancement](../../ICCV2025/image_restoration/learning_pixel-adaptive_multi-layer_perceptrons_for_real-time_image_enhancement.md)

<!-- RELATED:END -->

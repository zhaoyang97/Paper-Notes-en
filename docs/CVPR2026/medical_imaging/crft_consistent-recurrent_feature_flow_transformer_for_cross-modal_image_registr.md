---
title: >-
  [Paper Note] CRFT: Consistent-Recurrent Feature Flow Transformer for Cross-Modal Image Registration
description: >-
  [CVPR 2026][Medical Imaging][Cross-modal registration] CRFT is a unified coarse-to-fine cross-modal image registration framework that learns modality-agnostic feature flow representations within a Transformer architectur…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "Cross-modal registration"
  - "feature flow learning"
  - "coarse-to-fine"
  - "discrepancy-guided attention"
  - "spatial geometric transformation"
date: 2026-05-08
content_hash: da8d376f47147013
---

# CRFT: Consistent-Recurrent Feature Flow Transformer for Cross-Modal Image Registration

**Conference**: CVPR 2026
**arXiv**: [2604.05689](https://arxiv.org/abs/2604.05689)  
**Code**: [https://github.com/NEU-Liuxuecong/CRFT](https://github.com/NEU-Liuxuecong/CRFT)  
**Area**: Medical Imaging / Image Registration
**Keywords**: Cross-modal registration, feature flow learning, coarse-to-fine, discrepancy-guided attention, spatial geometric transformation

## TL;DR
CRFT is a unified coarse-to-fine cross-modal image registration framework that learns modality-agnostic feature flow representations within a Transformer architecture. It employs 1/8-resolution global correspondence at the coarse stage and multi-scale local refinement at 1/2–1/4 resolution at the fine stage, coupled with iterative discrepancy-guided attention and Spatial Geometric Transform (SGT) to recursively refine flow fields and capture subtle spatial inconsistencies. CRFT outperforms SOTA methods including RAFT, GMFlow, and LoFTR across diverse cross-modal datasets covering optical, infrared, SAR, and multispectral imagery.

## Background & Motivation

**Background**: Cross-modal image registration—establishing spatial correspondences between images acquired by different sensors—is a core problem in computer vision, with applications in 3D reconstruction, visual localization, and remote sensing analysis.

**Limitations of Prior Work**: (1) Hand-crafted features (SIFT/RIFT) are unreliable under strong nonlinear appearance discrepancies; (2) learning-based sparse matching methods (SuperGlue/LoFTR) are optimized for RGB imagery and generalize poorly to cross-modal inputs; (3) optical flow methods (RAFT/GMFlow) assume photometric consistency, which is violated by cross-modal inputs; (4) all existing methods struggle with combined challenges of large affine transformations, scale variation, and modality gaps.

**Core Idea**: (1) Learn modality-agnostic feature flow within a Transformer—establishing feature-space correspondences across modalities without relying on pixel-level consistency; (2) hierarchical coarse-to-fine matching (global + local); (3) iterative discrepancy-guided recurrent flow refinement—leveraging feature discrepancies to actively localize misaligned regions.

## Method

### Overall Architecture
Given an input image pair $(I^A, I^B)$, a shared ResNet CNN extracts features at three scales (1/2, 1/4, 1/8). The **coarse stage** (1/8 resolution) applies self-attention and cross-attention to establish global correspondences and produce an initial flow field. The **fine stage** (1/2 + 1/4 resolution) injects fine-grained spatial details via window attention and cross-attention. **Iterative discrepancy-guided flow optimization** (N rounds) then applies SGT-based feature alignment, computes feature discrepancies, applies discrepancy-weighted attention-guided updates with residual flow estimation, and employs a confidence estimation network for smoothing, ultimately producing a high-accuracy dense flow field.

### Key Designs

1. **Coarse-Stage Flow Estimation (Global Context)**:

    - 1/8-resolution features → self-attention enhancement + cross-attention cross-modal matching → global correlation matrix → initial flow field.
    - Design Motivation: Low-resolution features capture high-level structure, making them more robust to inter-modal spectral and radiometric inconsistencies, thereby stabilizing global matching.

2. **Fine-Stage Flow Refinement (Local Detail)**:

    - 1/2 and 1/4-resolution features → window self-attention for local pattern capture → cross-attention to inject fine-grained spatial details.
    - Design Motivation: High resolution preserves spatial detail, but global attention is computationally infeasible at this scale; window attention combined with hierarchical fusion addresses this limitation.

3. **Iterative Discrepancy-Guided Flow Optimization (Core Contribution)**:

    - N rounds of iterative refinement. Per round:
        - **Fine-Scale Feature Transformation (FSFT)**: Warp features using the current flow field.
        - **Spatial Geometric Transform (SGT)**: Explicitly models affine/scale transformations to handle large deformations.
        - **Discrepancy Computation**: Residual between warped features and target features → discrepancy map.
        - **Discrepancy-Guided Flow Optimization (DGFO)**: Discrepancy-weighted attention → automatically focuses on misaligned regions.
        - **Residual Update (RU)**: Discrepancy-guided residual flow update.
        - **Confidence Estimation Network (CENet)**: Predicts per-pixel confidence to smooth the final flow field.
    - Design Motivation: Single-pass matching cannot handle complex nonlinear and affine deformations; recurrent refinement progressively corrects errors. Discrepancy guidance ensures attention concentrates on the most poorly aligned regions, improving efficiency.

4. **Modality-Agnostic Design**:

    - A shared CNN encoder is used across modalities, learning modality-invariant features.
    - Feature flow formulation replaces pixel-level photometric/optical flow assumptions, eliminating dependence on pixel-level consistency.

## Key Experimental Results

### Main Results

**OSdataset (Optical–SAR Registration)**

| Method | Type | AEPE ↓ | CMR@3px ↑ | CMR@1px ↑ | CMR@0.7px ↑ |
|--------|------|--------|-----------|-----------|-------------|
| RIFT2 | Hand-crafted | 23.61 | 22.9% | 0.0% | 0.0% |
| GMFlow | Optical flow | 11.91 | 17.0% | 0.0% | 0.0% |
| RAFT | Optical flow | 3.51 | 69.6% | 15.9% | 8.7% |
| ADRNet | Dense matching | 1.67 | 90.1% | 35.0% | 20.6% |
| GDROS | Dense matching | 1.34 | 91.1% | 49.2% | 35.5% |
| XoFTR+Flow | Semi-dense | 1.13 | 96.2% | 57.6% | 41.7% |
| **CRFT** | **Ours** | **0.65** | **99.0%** | **95.1%** | **89.9%** |

CRFT is the only method to achieve sub-pixel AEPE (0.65); CMR@0.7px reaches 89.9%, which is **2.15×** the second-best method XoFTR+Flow (41.7%).

**RoadScene (Visible–Infrared Registration)**

| Method | AEPE ↓ | CMR@3px ↑ | CMR@1px ↑ | CMR@0.7px ↑ |
|--------|--------|-----------|-----------|-------------|
| RIFT2 | 17.27 | 36.4% | 0.0% | 0.0% |
| RAFT | 8.92 | 66.6% | 14.1% | 8.0% |
| ADRNet | 4.72 | 50.1% | 9.4% | 4.8% |
| XoFTR+Flow | 4.83 | 27.3% | 0.0% | 0.0% |
| **CRFT** | **2.37** | **68.2%** | **18.2%** | **4.5%** |

On RoadScene, CRFT achieves the lowest AEPE (2.37) and the highest CMR@1px (18.2%).

### Ablation Study

| Configuration | Effect |
|---------------|--------|
| Coarse stage only | Global correspondences established but insufficient spatial precision |
| + Fine stage | Improved local detail and higher accuracy |
| + Discrepancy guidance (N=1) | Further correction of geometric misalignment |
| **+ Iterative refinement (N=3)** | **Best performance with stable convergence** |
| w/o SGT | Degraded—significant drop in registration of large affine transformations |
| w/o discrepancy guidance | Degraded—unfocused attention, lower correction efficiency |
| w/o FSFT | Degraded—cross-modal feature spaces remain misaligned, leading to unstable discrepancy computation |

### Key Findings
- The SGT module is most critical for large affine transformations—without SGT, registration under large rotation/scale changes is nearly infeasible.
- Discrepancy-guided attention outperforms uniform attention by concentrating on regions requiring correction, making iterations more efficient.
- N=3 iterations is sufficient for convergence; additional iterations yield diminishing returns.
- CRFT remains competitive on RGB–RGB scenarios, demonstrating that the modality-agnostic design does not sacrifice same-modality performance.

## Highlights & Insights
- **Modality-agnostic feature flow**: Cross-modal registration is unified as flow estimation in feature space—no modality-specific design is required per sensor pair, yielding strong generalizability.
- **Discrepancy-guided adaptive attention**: Feature discrepancies between warped and target representations serve as attention weights, automatically localizing misaligned regions—substantially more efficient than uniform attention.
- **Explicit geometric modeling via SGT**: Affine transformations are integrated as a learnable module rather than relying on the flow field to implicitly learn large deformations.

## Limitations & Future Work
- N=3 iterative refinement increases inference time.
- The coarse stage employs global attention, requiring resolution control for large images.
- Validation is currently limited to remote sensing and navigation scenarios; application to medical registration (CT–MRI) remains to be explored.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of discrepancy-guided recurrence, SGT, and modality-agnostic flow is effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validated across optical, infrared, SAR, and multispectral scenarios.
- Writing Quality: ⭐⭐⭐⭐ Detailed architectural diagrams.
- Value: ⭐⭐⭐⭐ Offers general-purpose registration utility for remote sensing and navigation applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)
- [\[AAAI 2026\] Radiation-Preserving Selective Imaging for Pediatric Hip Dysplasia: A Cross-Modal Approach](../../AAAI2026/medical_imaging/radiation-preserving_selective_imaging_for_pediatric_hip_dysplasia_a_cross-modal.md)
- [\[AAAI 2026\] FaNe: Towards Fine-Grained Cross-Modal Contrast with False-Negative Reduction and Text-Conditioned Sparse Attention](../../AAAI2026/medical_imaging/fane_towards_fine-grained_cross-modal_contrast_with_false-negative_reduction_and.md)
- [\[CVPR 2026\] Diffusion-Based Feature Denoising and Using NNMF for Robust Brain Tumor Classification](diffusion-based_feature_denoising_and_using_nnmf_for_robust_brain_tumor_classifi.md)
- [\[CVPR 2026\] Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation? A Cross-Dataset Empirical Study](are_general-purpose_vision_models_all_we_need_for_2d_medical_image_segmentation_.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] ReCon-GS: Continuum-Preserved Gaussian Streaming for Fast and Compact Reconstruction
description: >-
  [NeurIPS 2025][Object Detection][3D Gaussian Splatting] This paper proposes ReCon-GS, which achieves incremental 3D reconstruction via continuum-preserved Gaussian streaming…
tags:
  - "NeurIPS 2025"
  - "Object Detection"
  - "3D Gaussian Splatting"
  - "streaming reconstruction"
  - "continuum preservation"
  - "incremental learning"
  - "real-time rendering"
date: 2026-05-08
content_hash: 67a8e2fcb1bca148
---

# ReCon-GS: Continuum-Preserved Gaussian Streaming for Fast and Compact Reconstruction

**Conference**: NeurIPS 2025
**arXiv**: [2509.24325](https://arxiv.org/abs/2509.24325)
**Code**: Available
**Area**: Object Detection / 3D Reconstruction
**Keywords**: 3D Gaussian Splatting, streaming reconstruction, continuum preservation, incremental learning, real-time rendering

## TL;DR

This paper proposes ReCon-GS, which achieves incremental 3D reconstruction via continuum-preserved Gaussian streaming, substantially reducing storage requirements and training time while maintaining rendering quality, and supporting real-time reconstruction of large-scale scenes.

## Background & Motivation

### State of the Field

**Background**: 3D Gaussian Splatting (3DGS) has achieved revolutionary progress in static scene reconstruction, yet standard methods require processing all input images at once.

**Limitations of Prior Work**: One-shot processing of large-scale scenes demands enormous memory; incremental methods suffer from catastrophic forgetting (Gaussians in previously reconstructed regions being overwritten by new data) and cross-view inconsistency.

**Key Challenge**: Incremental efficiency vs. global consistency — per-frame processing is fast but loses global information, while global processing is accurate but not scalable.

**Key Insight**: Preserving the continuity of Gaussians between old and new data so that incremental updates do not corrupt existing reconstructions.

## Method

### Overall Architecture

Input image stream → Local Gaussian initialization → Continuum-preserved incremental update → Global Gaussian scene → Real-time rendering.

### Key Designs

1. **Gaussian Streaming**

    - Function: Divides the input image stream into windows; each window incrementally updates the Gaussian set.
    - Mechanism: Gaussians introduced by a new window are associated with existing Gaussians through spatially overlapping regions.
    - Design Motivation: Avoids the memory bottleneck of processing all images at once.

2. **Continuum Preservation Mechanism**

    - Function: Prevents optimization on new data from corrupting attributes of existing Gaussians.
    - Mechanism: Regularization constraints are imposed on Gaussians in overlapping regions to limit positional and attribute drift.
    - Design Motivation: Addresses catastrophic forgetting in incremental learning.

3. **Adaptive Density Control**

    - Function: Dynamically adjusts Gaussian density according to the coverage of new views.
    - Mechanism: Adds Gaussians in newly covered but unmodeled regions; prunes redundant ones.
    - Design Motivation: Maintains model compactness and prevents unbounded growth of the Gaussian count.

### Loss & Training

$\mathcal{L} = \mathcal{L}_{photo} + \lambda_{reg}\mathcal{L}_{continuity} + \lambda_{ssim}\mathcal{L}_{SSIM}$

## Key Experimental Results

### Main Results

| Method | PSNR↑ | SSIM↑ | Training Time↓ | Storage↓ |
|--------|-------|-------|----------------|----------|
| 3DGS (Global) | 33.14 | 0.969 | 30min | 100% |
| InstantNGP | 31.25 | 0.951 | 5min | 40% |
| StreamRF | 30.89 | 0.942 | 8min | 55% |
| **ReCon-GS** | **32.78** | **0.965** | **12min** | **45%** |

### Ablation Study

| Configuration | PSNR | Note |
|---------------|------|------|
| w/o continuity constraint | 30.45 | Severe forgetting |
| w/o density control | 31.89 | Excessive Gaussians |
| **Full model** | **32.78** | **Best** |

### Key Findings

- Trails global 3DGS by only 0.36 PSNR, while reducing training time and storage by more than 50%.
- Continuum preservation contributes +2.33 PSNR, establishing it as the core module.
- Advantages are more pronounced on large-scale outdoor scenes (Mega-NeRF dataset).

## Highlights & Insights

- **Continuum Preservation**: Addresses the central challenge of incremental 3DGS — catastrophic forgetting. The regularization constraint is both simple and effective.
- **Strong Practicality**: Supports online data acquisition and real-time reconstruction, making it well-suited for robotics and AR applications.

## Limitations & Future Work

- Overlap region detection relies on the accuracy of pose estimation.
- Dynamic scenes are not addressed.
- Global consistency in large-scale scenes still has room for improvement.

## Related Work & Insights

- **vs. 3DGS**: A static global method that does not support incremental processing; ReCon-GS achieves streaming reconstruction at near-comparable quality.
- **vs. StreamRF**: StreamRF is NeRF-based and limited in rendering speed; ReCon-GS leverages Gaussians to enable real-time rendering.

## Rating
- Novelty: ⭐⭐⭐⭐ The continuum-preservation idea is clear and practical.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple datasets.
- Writing Quality: ⭐⭐⭐⭐ Well-structured presentation.
- Value: ⭐⭐⭐⭐⭐ Strong practical demand for real-time reconstruction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ReCon: Region-Controllable Data Augmentation with Rectification and Alignment for Object Detection](recon_region-controllable_data_augmentation_with_rectification_and_alignment_for.md)
- [\[AAAI 2026\] TubeRMC: Tube-conditioned Reconstruction with Mutual Constraints for Weakly-supervised Spatio-Temporal Video Grounding](../../AAAI2026/object_detection/tubermc_tube-conditioned_reconstruction_with_mutual_constraints_for_weakly-super.md)
- [\[NeurIPS 2025\] DitHub: A Modular Framework for Incremental Open-Vocabulary Object Detection](dithub_a_modular_framework_for_incremental_openvocabulary_ob.md)
- [\[NeurIPS 2025\] FlexEvent: Towards Flexible Event-Frame Object Detection at Varying Operational Frequencies](flexevent_towards_flexible_event-frame_object_detection_at_varying_operational_f.md)
- [\[NeurIPS 2025\] CQ-DINO: Mitigating Gradient Dilution via Category Queries for Vast Vocabulary Object Detection](cq-dino_mitigating_gradient_dilution_via_category_queries_for_vast_vocabulary_ob.md)

</div>

<!-- RELATED:END -->

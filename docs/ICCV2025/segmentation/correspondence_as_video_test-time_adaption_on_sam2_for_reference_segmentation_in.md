---
title: >-
  [Paper Note] Correspondence as Video: Test-Time Adaption on SAM2 for Reference Segmentation in the Wild
description: >-
  [ICCV 2025][Segmentation][Reference Segmentation] CAV-SAM represents the correspondence between reference–target image pairs as a pseudo-video sequence…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "Reference Segmentation"
  - "SAM2"
  - "Video Object Segmentation"
  - "Diffusion-Based Semantic Transition"
  - "Test-Time Adaptation"
date: 2026-05-08
content_hash: cf8975f2048128bf
---

# Correspondence as Video: Test-Time Adaption on SAM2 for Reference Segmentation in the Wild

**Conference**: ICCV 2025
**arXiv**: [2508.07759](https://arxiv.org/abs/2508.07759)
**Code**: [https://github.com/wanghr64/cav-sam](https://github.com/wanghr64/cav-sam)
**Area**: Image Segmentation / Few-Shot Segmentation
**Keywords**: Reference Segmentation, SAM2, Video Object Segmentation, Diffusion-Based Semantic Transition, Test-Time Adaptation

## TL;DR

CAV-SAM represents the correspondence between reference–target image pairs as a pseudo-video sequence, bridging semantic gaps via a Diffusion-Based Semantic Transition (DBST) module and aligning geometric variations via a Test-Time Geometric Alignment (TTGA) module. This enables SAM2's video segmentation capability to be adapted to reference segmentation in a training-free manner, surpassing the state of the art by approximately 5% mIoU on cross-domain few-shot segmentation benchmarks.

## Background & Motivation

Large vision models (e.g., SAM) suffer from domain gaps and category novelty issues when applied to downstream tasks. Reference segmentation—conveying new knowledge to the model via reference images and masks—is a promising direction to address this. However:

- **Existing methods rely on meta-learning**: Few-Shot Segmentation (FSS) and Cross-Domain FSS (CD-FSS) methods require extensive meta-training, incurring substantial data and computational costs.
- **Discrete image pairs vs. continuous video**: SAM2 supports interactive video object segmentation (iVOS), propagating segmentation from annotated frames across an entire video sequence. However, reference–target image pairs are discrete and differ from natural continuous video in two key respects: (a) *semantic gap*—iVOS tracks the same instance, whereas FSS recognizes different instances of the same category; (b) *geometric variation*—iVOS assumes smooth inter-frame transformations, while geometric changes in reference segmentation can be large.

A key observation: simply concatenating reference and target images into a pseudo-video allows SAM2 to achieve near-SOTA performance on CD-FSS benchmarks (mIoU 60.68 vs. SOTA ~61), demonstrating substantial latent potential of iVOS models for reference segmentation.

The core idea of this paper: **represent the correspondences between reference and target images as a video**, using diffusion-generated semantic transition sequences to bridge the semantic gap and lightweight test-time fine-tuning to align geometric variations.

## Method

### Overall Architecture

Given a reference image $I_r$, its mask $M_r$, and a target image $I_t$, CAV-SAM comprises two modules:
1. **DBST**: generates a semantic transition sequence $I_v^1, \ldots, I_v^{N_v}$ between the reference and target images using a diffusion model.
2. **TTGA**: performs lightweight test-time fine-tuning using prototype vectors and augmented images, generating pseudo-labels as additional prompts for SAM2.

SAM2 then performs segmentation on the pseudo-video sequence in an iVOS manner.

### Key Designs

1. **Diffusion-Based Semantic Transition (DBST)**:

    - **Function**: Generates pseudo-video frames with semantically smooth transitions between the reference and target images.
    - **Mechanism**: Building on DiffMorpher, LoRA parameters $\Delta\theta_r$ and $\Delta\theta_t$ are trained separately on the reference and target images, then fused via linear interpolation $\Delta\theta_\alpha = (1-\alpha)\Delta\theta_r + \alpha\Delta\theta_t$. Latent noise obtained from DDIM inversion is also interpolated using spherical linear interpolation: $\mathbf{z}_{T\alpha}=\frac{\sin((1-\alpha)\phi)}{\sin\phi}\mathbf{z}_{Tr}+\frac{\sin(\alpha\phi)}{\sin\phi}\mathbf{z}_{Tt}$.
    - **Design Motivation**: iVOS models track the same instance, whereas FSS requires recognizing different instances of the same category. The diffusion-generated semantic transition enables SAM2 to track object categories along a continuous variation trajectory.
    - **Optimization**: The unnecessary visual refinement module from DiffMorpher is removed, substantially reducing inference cost.

2. **Test-Time Geometric Alignment (TTGA)**:

    - **Function**: Adapts SAM2's image encoder to geometric variations of the target object and generates pseudo-label prompts.
    - **Mechanism**: Only the FPN layers of the SAM2 image encoder are fine-tuned. A prototype vector $\boldsymbol{p}_r = \text{MAP}(F_r, M_r)$ extracted from the reference image activates the target region via cosine similarity. The training loss is Augmentative Cyclic Consistency (ACC): $\mathcal{L} = \mathcal{L}_{aug} + \mathcal{L}_{cyc}$, where $\mathcal{L}_{aug}$ supervises predictions on augmented images, and $\mathcal{L}_{cyc}$ enforces cyclic consistency by predicting the original image using pseudo-labels from the augmented image.
    - **Design Motivation**: Features extracted directly from SAM2 fail to effectively activate target regions in the semantic transition sequence. Lightweight fine-tuning with a single reference image (100 steps, FPN layers only) avoids the high cost of meta-training.

3. **Augmentative Cyclic Consistency (ACC)**:

    - **Function**: Maximizes the learning signal from limited annotated data.
    - **Mechanism**: $I_r \rightarrow \boldsymbol{p}_r \rightarrow \hat{M}_r^{aug}$ (forward) $\rightarrow I_r^{aug} \rightarrow \hat{\boldsymbol{p}}_r^{aug} \rightarrow \hat{M}_r$ (cyclic). Predicted pseudo-labels rather than ground-truth masks are used to compute the augmented prototype, providing a more robust self-supervised signal.
    - **Design Motivation**: Compared with the ABC strategy that uses GT masks, ACC employs predicted masks as an intermediate step, yielding an implicit data augmentation effect.

### Loss & Training

- No meta-training is required; only 100 steps of lightweight fine-tuning per reference image at test time.
- SAM2 tiny is used; DBST trains LoRA for 200 steps with rank=16 and 20-step DDIM inversion.
- Semantic transition sequence length $N_v=9$, with $\alpha$ uniformly distributed from 0.2 to 0.8.
- TTGA learning rate $1\times10^{-3}$ with cosine annealing.

## Key Experimental Results

### Main Results

**CD-FSS 1-shot/5-shot mIoU across 4 datasets:**

| Method | Type | Deepglobe | ISIC | Chest X-Ray | FSS-1000 | Average |
|--------|------|-----------|------|-------------|----------|---------|
| IFA (CVPR24) | CD-FSS | 37.73 | 44.55 | 80.03 | 79.97 | 60.57 |
| DR-Adaptor (CVPR24) | CD-FSS | 41.29 | 40.77 | 82.35 | 79.05 | 60.86 |
| APSeg (CVPR24) | SAM-based | 35.94 | 45.43 | 84.10 | 79.71 | 61.30 |
| **CAV-SAM (Ours)** | **iVOS-based** | **39.11** | **50.36** | **86.97** | **79.78** | **64.06** |

5-shot: the proposed method achieves an average mIoU of 69.14, surpassing all compared methods (APSeg: 65.09, DR-Adaptor: 65.42).

The most significant improvement is observed on the challenging Chest X-Ray medical dataset (1-shot: 86.97 vs. 84.10, +2.87).

### Ablation Study

| iVOS Model | DBST | TTGA | mIoU |
|-----------|------|------|------|
| SAM2 | ✗ | ✗ | 60.68 |
| SAM2 | ✓ | ✗ | 62.37 |
| SAM2 | ✗ | ✓ | 62.52 |
| SAM2 | ✓ | ✓ | **64.06** |

**ACC vs. ABC**: The Augmentative Cyclic Consistency (ACC) strategy substantially outperforms the ABC strategy that relies on GT masks.

### Key Findings
- Simply concatenating reference and target images as a pseudo-video already achieves near-SOTA performance (60.68), confirming that iVOS models are naturally well-suited for reference segmentation.
- The DBST and TTGA modules each contribute approximately +1.7 and +1.8 mIoU, respectively, with their combination yielding additive gains.
- Performance on the Deepglobe dataset is relatively weaker, likely because SAM is oriented toward object segmentation whereas Deepglobe involves region segmentation.
- Although the visual quality of DBST-generated frames is lower than that of the original DiffMorpher outputs, segmentation performance remains strong—SAM2's robust iVOS capability compensates for the reduced visual fidelity.

## Highlights & Insights
- **Elegant perspective shift**: Reformulating reference segmentation as video object segmentation elegantly leverages SAM2's iVOS capability.
- **Meta-training-free**: The approach entirely bypasses the large-scale meta-training required by conventional FSS methods, relying solely on 100 steps of test-time fine-tuning.
- **Natural safeguard for semantic consistency**: When the reference and target images do not share the same category, DBST generates meaningless sequences and TTGA prototypes fail to activate, naturally preventing erroneous segmentation.
- **General framework**: Compatible with any iVOS model and diffusion model; future advances in either component can be directly incorporated.

## Limitations & Future Work
- The DBST module still requires per-pair LoRA training (200 steps), resulting in non-trivial inference overhead.
- Performance on region segmentation (e.g., Deepglobe remote sensing scenes) is suboptimal, reflecting inherent limitations of SAM/SAM2.
- The quality of generated pseudo-video sequences depends on the diffusion model's capacity and may be limited for cross-domain pairs with large style discrepancies.
- Only 1-shot and 5-shot settings are evaluated; the benefit of using more reference images remains unexplored.
- The use of SAM2 tiny may impose an upper bound on performance; larger SAM2 variants could yield further improvements.

## Related Work & Insights
- **SAM2**: The core foundation model providing iVOS capability for segmenting objects in video.
- **DiffMorpher**: A diffusion-based image morphing method; its LoRA and noise interpolation strategies are adopted here to generate semantic transition sequences.
- **APSeg / VRP-SAM**: Reference segmentation methods built on SAM that still rely on meta-learning to introduce additional prompt encoders.
- **PATNet / IFA / DR-Adaptor**: CD-FSS methods that require a complete meta-training pipeline.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Recasting reference segmentation as video segmentation is a highly elegant perspective shift; the DBST+TTGA design is well-motivated and clean.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across four datasets with sufficient ablations, though comparisons with more iVOS models are lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clear, problem analysis is thorough, and figures are intuitive and accessible.
- Value: ⭐⭐⭐⭐ Proposes a new paradigm for reference segmentation with important implications for adapting foundation models to downstream tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] TopoTTA: Topology-Enhanced Test-Time Adaptation for Tubular Structure Segmentation](topotta_topology-enhanced_test-time_adaptation_for_tubular_structure_segmentatio.md)
- [\[ICCV 2025\] Hybrid-TTA: Continual Test-time Adaptation via Dynamic Domain Shift Detection](hybrid-tta_continual_test-time_adaptation_via_dynamic_domain_shift_detection.md)
- [\[ICCV 2025\] Online Reasoning Video Segmentation with Just-in-Time Digital Twins](online_reasoning_video_segmentation_with_just-in-time_digital_twins.md)
- [\[ICCV 2025\] WildSeg3D: Segment Any 3D Objects in the Wild from 2D Images](wildseg3d_segment_any_3d_objects_in_the_wild_from_2d_images.md)
- [\[ICCV 2025\] A Plug-and-Play Physical Motion Restoration Approach for In-the-Wild High-Difficulty Motions](a_plugandplay_physical_motion_restoration_approach_for_inthe.md)

</div>

<!-- RELATED:END -->

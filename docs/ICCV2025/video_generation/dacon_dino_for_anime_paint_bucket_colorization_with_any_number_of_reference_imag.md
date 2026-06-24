---
title: >-
  [Paper Note] DACoN: DINO for Anime Paint Bucket Colorization with Any Number of Reference Images
description: >-
  [ICCV 2025][Video Generation][Anime colorization] This paper proposes DACoN, which fuses semantic features from the DINOv2 foundation model with high-resolution spatial features from a U-Net to enable automatic anime line art colorization with an arbitrary number of reference images, surpassing existing methods on both key-frame and sequential-frame colorization tasks.
tags:
  - "ICCV 2025"
  - "Video Generation"
  - "Anime colorization"
  - "DINOv2"
  - "line art coloring"
  - "multi-reference images"
  - "semantic correspondence"
date: 2026-05-08
content_hash: def96538c23f88c9
---

# DACoN: DINO for Anime Paint Bucket Colorization with Any Number of Reference Images

**Conference**: ICCV 2025
**arXiv**: [2509.14685](https://arxiv.org/abs/2509.14685)  
**Code**: [https://github.com/kzmngt/DACoN](https://github.com/kzmngt/DACoN)  
**Area**: Video Generation
**Keywords**: Anime colorization, DINOv2, line art coloring, multi-reference images, semantic correspondence

## TL;DR

This paper proposes DACoN, which fuses semantic features from the DINOv2 foundation model with high-resolution spatial features from a U-Net to enable automatic anime line art colorization with an arbitrary number of reference images, surpassing existing methods on both key-frame and sequential-frame colorization tasks.

## Background & Motivation

In anime production, line art colorization is a time-consuming and repetitive manual task. Existing methods face three major bottlenecks:

**Limited number of reference images**: Prior methods (e.g., AnT) rely on a Multiplex Transformer with a fixed input structure, supporting at most 1–2 reference images and failing to exploit the full information available in multi-angle color design sheets.

**Low accuracy for key-frame colorization**: Key frames exhibit large compositional differences (pose, viewpoint, and occlusion variations), and line art contains only contour information, making it difficult to establish semantic correspondences and resulting in accuracy far below that of sequential-frame colorization.

**Limitations of generative approaches**: Diffusion-model-based methods may generate colors absent from the reference images, and their outputs are composite images rather than segmented layers, hampering post-production editing.

A key finding of this paper is that **DINOv2 can capture part-level semantic information from line art** (e.g., arms, hair), such that PCA visualizations of DINO features on line art still reveal clear semantic partitions even when only contour information is present. This provides a strong prior for correspondence-based colorization.

## Method

### Overall Architecture

The DACoN pipeline: given $K$ reference images and a target line art → extract features from each image independently (the model does not observe other images, imposing no constraint on their count) → obtain per-segment features via segment pooling → compute cosine similarities between reference and target segments → propagate the color of the most similar reference segment to each target segment.

Let the reference images be $L_k \in \mathbb{R}^{H \times W \times 3}$ ($k=1,...,K$) and the target image be $L_t$, with corresponding segment masks $m_k \in \mathbb{R}^{M_k \times H \times W}$ and $m_t \in \mathbb{R}^{N \times H \times W}$, and reference color information $c_k \in \mathbb{R}^{M_k \times 3}$.

### Key Designs

1. **Dual-stream feature extraction with DINOv2 + U-Net**:

    - **DINOv2 encoder** (frozen): extracts low-resolution semantic features $D \in \mathbb{R}^{H_d \times W_d \times C_d}$ with $C_d=1024$ dimensions at an input resolution of 518×518.
    - **U-Net encoder** (trainable): extracts high-resolution spatial features $U \in \mathbb{R}^{H \times W \times C_u}$ with $C_u=128$ dimensions at an input resolution of 512×512.
    - Design motivation: DINOv2 provides global semantic understanding ("this is an arm"), while the U-Net provides local fine-grained features ("the precise shape and location of this segment").

2. **Segment pooling and feature fusion**:

    - Feature maps are resized to the original resolution, element-wise multiplied by the segment mask, and spatially averaged:
    $d = \text{avg}(D \odot m), \quad u = \text{avg}(U \odot m)$
    - DINO features are projected via an MLP from $C_d$ to $C_u$, then concatenated with the CNN features.
    - A second MLP fuses the concatenated features into the final per-segment representation $f_k \in \mathbb{R}^{M_k \times C_u}$.
    - Key advantage: **each image is processed independently**, so feature extraction does not depend on other images, removing any constraint on the number of reference images.

3. **Segment correspondence and color propagation**:

    - Per-segment features from all $K$ reference images are concatenated along the segment dimension: $f_r \in \mathbb{R}^{M \times C_u}$, where $M = \sum_k M_k$.
    - A cosine similarity matrix $\hat{S} \in \mathbb{R}^{M \times N}$ is computed between reference and target features.
    - Argmax identifies the most similar reference segment for each target segment, whose color is then propagated.
    - A softmax with temperature $T=0.1$ is applied to sharpen the distribution.

### Loss & Training

The total loss is a weighted sum of two terms:
$$\mathcal{L}_{\text{final}} = \lambda_{ce} \mathcal{L}_{ce} + \lambda_{dc} \mathcal{L}_{dc}$$

- **Cross-entropy loss** $\mathcal{L}_{ce}$: treats colorization as a classification problem, computing cross-entropy over softmax color probabilities; colors absent from the ground truth are excluded.
- **DINO-guided feature consistency loss** $\mathcal{L}_{dc}$: encourages the model's final similarity matrix $\hat{S}$ to align with the similarity matrix $\hat{S}'$ computed from pure DINO features:
$$\mathcal{L}_{dc} = \frac{1}{MN} \sum_{m,n} |\hat{S}_{m,n} - \hat{S}'_{m,n}|$$

Hyperparameters: $\lambda_{ce}=0.5$, $\lambda_{dc}=0.2$. The model is trained for 5 epochs with batch size 2, Adam optimizer, learning rate $1 \times 10^{-4}$, on a single RTX 4090 for 14 hours.

## Key Experimental Results

### Main Results (Key-frame colorization, 3D-rendered test set, 8 character subsets)

| Method | # Refs | Acc | Acc-Thresh | Pix-Acc | Pix-F-Acc |
|--------|--------|-----|------------|---------|-----------|
| ColorFlow | 1 | 12.10 | 13.13 | 53.75 | 7.51 |
| MangaNinja | 1 | 17.39 | 19.52 | 8.68 | 34.51 |
| AniDoc | 1 | 23.98 | 27.06 | 78.71 | 50.25 |
| BasicPBC-Ref | 1 | - | 63.46 | 95.31 | 82.69 |
| **DACoN (Ours)** | 1 | **70.61** | **75.15** | **97.82** | **92.16** |
| **DACoN (Ours)** | 5 | **75.49** | **79.55** | **98.43** | **94.72** |
| **DACoN (Ours)** | max | **76.66** | **80.68** | **98.60** | **95.04** |

### Ablation Study

| Configuration | Acc (1-ref) | Acc (5-ref) | Description |
|---------------|-------------|-------------|-------------|
| DACoN (full) | 67.87 | 73.25 | Full test set |
| No color info (Mono) | 64.32 | 70.26 | Line colors removed |
| w/o $\mathcal{L}_{dc}$ | 63.95 | 72.57 | DINO consistency loss removed |

Sequential-frame colorization (3D-rendered):

| Method | Acc | Acc-Thresh | Pix-Acc |
|--------|-----|------------|---------|
| AniDoc | 31.20 | 36.36 | 92.01 |
| LVCD | 37.63 | 43.77 | 66.20 |
| AnT (Cadmium) | 66.34 | 77.13 | 97.65 |
| BasicPBC | 82.66 | 87.26 | 99.05 |
| **DACoN (Ours)** | **84.76** | **88.23** | **99.27** |

### Key Findings

- **Benefit of multiple reference images**: Increasing from 1 to 5 references improves key-frame accuracy by approximately 5 percentage points (67.87→73.25), demonstrating the value of multi-angle color information.
- **Selective reference selection is unnecessary**: Adding reference images with low similarity to the target does not degrade accuracy, simplifying the overall pipeline.
- **Bidirectional temporal references are most effective**: In sequential-frame colorization, using both the preceding and following frames as references (±1 frame) improves accuracy from 84.71 to 88.93.
- **DINO consistency loss is critical for key frames**: Removing this loss reduces key-frame accuracy by approximately 4 percentage points, with comparatively little effect on sequential frames.
- **Line color information is valuable**: Removing line colors (Mono mode) causes a 3–5 percentage point accuracy drop, as shadow and highlight regions rely on colored lines for disambiguation.

## Highlights & Insights

- The discovery of DINOv2's semantic capability on line art is an important empirical contribution, opening a new direction for introducing foundation models into the animation domain.
- Eliminating the reference-count constraint by processing each image independently is an elegantly simple yet effective design choice.
- A unified model handles both key-frame and sequential-frame colorization tasks, avoiding the need to maintain separate models.
- In practical animation production, segment-based colorization preserves editability, making it more practical than generative approaches.

## Limitations & Future Work

- When a character is too close to the camera and body parts extend beyond the frame, the U-Net cannot capture complete line art information.
- Performance degrades under extreme poses, as color design sheets primarily depict upright stances.
- Small background regions surrounded by foreground elements may be misclassified as foreground.
- The frozen DINOv2 encoder may limit adaptability to diverse anime styles.

## Related Work & Insights

- Compared to BasicPBC, DACoN's gain on sequential-frame colorization is modest (+2%), but the substantial leap in key-frame colorization accuracy (63.46→75.15) constitutes the core contribution.
- The feature extraction capability of foundation models in creative domains (anime, illustration) is greatly underestimated; this paper provides compelling evidence to the contrary.
- Insight: other creative tasks requiring semantic correspondence, such as manga translation and style transfer, may similarly benefit from DINO features.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Introducing DINOv2 into anime colorization is a novel cross-domain application, though the overall technical framework is relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers both key-frame and sequential-frame tasks, multiple baselines, ablation studies, and failure case analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Method description is clear; DINO feature visualizations are intuitive and persuasive.
- **Value**: ⭐⭐⭐⭐⭐ Directly applicable to practical anime production workflows; open-sourced code enhances usability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DIVE: Taming DINO for Subject-Driven Video Editing](dive_taming_dino_for_subject-driven_video_editing.md)
- [\[ICLR 2026\] MAGREF: Masked Guidance for Any-Reference Video Generation with Subject Disentanglement](../../ICLR2026/video_generation/magref_masked_guidance_for_any-reference_video_generation_with_subject_disentang.md)
- [\[ICCV 2025\] SteerX: Creating Any Camera-Free 3D and 4D Scenes with Geometric Steering](steerx_creating_any_camera-free_3d_and_4d_scenes_with_geometric_steering.md)
- [\[CVPR 2026\] Composing Concepts from Images and Videos via Concept-prompt Binding](../../CVPR2026/video_generation/composing_concepts_from_images_and_videos_via_concept-prompt_binding.md)
- [\[CVPR 2026\] Scaling Zero-Shot Reference-to-Video Generation](../../CVPR2026/video_generation/scaling_zero-shot_reference-to-video_generation.md)

</div>

<!-- RELATED:END -->

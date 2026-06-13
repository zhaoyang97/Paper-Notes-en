---
title: >-
  [Paper Note] Rewis3d: Reconstruction Improves Weakly-Supervised Semantic Segmentation
description: >-
  [CVPR 2026][3D Vision][Weakly-supervised segmentation] This paper proposes Rewis3d, the first framework to integrate feedforward 3D scene reconstruction as an auxiliary supervision signal for weakly-supervised semantic s…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Weakly-supervised segmentation"
  - "3D reconstruction"
  - "cross-modal consistency"
  - "sparse annotation"
  - "Mean Teacher"
date: 2026-05-08
content_hash: 3b0b25d3af179fdf
---

# Rewis3d: Reconstruction Improves Weakly-Supervised Semantic Segmentation

**Conference**: CVPR 2026
**arXiv**: [2603.06374](https://arxiv.org/abs/2603.06374)  
**Code**: Coming soon  
**Area**: 3D Vision / Semantic Segmentation
**Keywords**: Weakly-supervised segmentation, 3D reconstruction, cross-modal consistency, sparse annotation, Mean Teacher

## TL;DR

This paper proposes Rewis3d, the first framework to integrate feedforward 3D scene reconstruction as an auxiliary supervision signal for weakly-supervised semantic segmentation. Through a dual student-teacher architecture and dual confidence-weighted cross-modal consistency loss, Rewis3d improves mIoU by 2–7% under sparse annotation, while using only 2D images at inference time.

## Background & Motivation

Semantic segmentation has achieved remarkable progress but relies heavily on dense pixel-level annotations, which are extremely costly to obtain. Weakly-supervised semantic segmentation (WSSS) reduces annotation burden by exploiting sparse labels such as point annotations, scribbles, or coarse labels, yet a performance gap with fully supervised methods persists.

**Key limitations of existing approaches:**
1. **Ceiling of pure 2D methods**: Methods such as SASFormer and TreeEnergy design dedicated architectures and losses to propagate annotation information within the 2D image plane, but struggle to compensate for the lack of supervision in geometrically complex outdoor scenes.
2. **Underutilization of 3D geometry**: 3D structure naturally provides cross-view consistency constraints — when an object is annotated with a scribble in one view, its 3D geometry can propagate labels to all other views where it appears.

**Core insight**: Recent breakthroughs in feedforward 3D reconstruction (e.g., MapAnything) enable high-fidelity 3D point clouds to be recovered directly from ordinary 2D video sequences without specialized sensors such as LiDAR. This motivates a novel strategy: **leveraging reconstructed 3D geometry as auxiliary supervision to enhance 2D weakly-supervised segmentation**, while maintaining a purely 2D inference pipeline.

## Method

### Overall Architecture

Rewis3d comprises three key components working in concert:

1. **2D segmentation branch**: SegFormer-B4 + Mean Teacher architecture
2. **3D segmentation branch**: Point Transformer V3 + Mean Teacher architecture
3. **Cross-Modal Consistency (CMC)**: Bidirectional knowledge transfer — the teacher of one modality supervises the student of the other

Training proceeds in two stages:
- **Base training** (15 epochs): Each modality independently establishes its own student-teacher framework.
- **CMC training**: Cross-modal consistency loss is introduced, with linear warm-up over 5 epochs to a maximum weight of $\lambda = 0.1$.

**Key property**: 3D reconstruction serves only as a preprocessing step; final inference is performed entirely in 2D — requiring no 3D sensors and incurring no additional inference overhead.

### Key Designs

1. **3D Scene Reconstruction and View-Aware Sampling**:
    - MapAnything reconstructs dense point clouds with per-point reconstruction confidence via a single forward pass over 2D video sequences.
    - Processing the full scene point cloud (200+ images → 60M+ points) directly is infeasible.
    - View-aware sampling is proposed: a dedicated 120K-point sub-sample is generated for each target image.
    - **60% of points are sampled from the current view** (ensuring dense 2D–3D correspondences, approximately 72K corresponding points).
    - **40% are sampled from the surrounding scene** (providing contextual information to maintain global scene understanding in the 3D branch).
    - In contrast, random sampling yields only approximately 140 correspondences per image; view-aware sampling guarantees effective training of the CMC loss.

2. **Dual Student-Teacher Architecture**:
    - Both the 2D and 3D branches maintain independent Mean Teacher student-teacher structures.
    - Teacher weights are updated via EMA: $\boldsymbol{\theta}_t^{\text{teacher}} \leftarrow \alpha \boldsymbol{\theta}_{t-1}^{\text{teacher}} + (1-\alpha) \boldsymbol{\theta}_t^{\text{student}}$, where $\alpha = 0.99$.
    - Each branch applies a supervised cross-entropy loss $\mathcal{L}_S$ on annotated regions and an unsupervised consistency loss $\mathcal{L}_U$ on unannotated regions using teacher pseudo-labels.
    - Confidence filtering retains only pixels where the teacher's maximum class probability exceeds a threshold $\tau$.

3. **Dual Confidence-Weighted Cross-Modal Consistency (CMC)**:
    - Loss for the 3D teacher guiding the 2D student: $\mathcal{L}_C^{2D} = -\sum_j w_i \cdot \log(S_{2D}^{y_i}(I_j))$
    - Weights combine two confidence sources: $w_i = \underbrace{\max(\text{softmax}(T_{3D}(p_i)))}_{\text{prediction confidence}} \cdot \underbrace{c_i^{\text{rec}}}_{\text{reconstruction confidence}}$
    - Prediction confidence is derived from the 3D teacher's output probabilities; reconstruction confidence is derived from MapAnything's per-point reconstruction quality.
    - This dual filtering ensures that supervision signals originate primarily from "reliable predictions on high-quality reconstructed geometry."
    - Symmetrically, the 2D teacher also supervises the 3D student ($\mathcal{L}_C^{3D}$).

### Loss & Training

Total loss function:

$$\mathcal{L}_{\text{Total}} = \sum_{m \in \{2D, 3D\}} (\mathcal{L}_S^m + \mathcal{L}_U^m) + \lambda_{2D} \mathcal{L}_C^{2D} + \lambda_{3D} \mathcal{L}_C^{3D}$$

Training details:
- 2D branch: SegFormer-B4, learning rate $5 \times 10^{-5}$
- 3D branch: Point Transformer V3, learning rate $10^{-3}$
- Optimizer: AdamW, batch size 12, two H100 GPUs
- 50 epochs (250 for NYUv2), CMC weight $\lambda = 0.1$ with linear warm-up
- Students use stronger augmentations (Cutout, Blur, AugMix / RandomRotation, RandomScale, RandomJitter); teachers use weak augmentations

## Key Experimental Results

### Main Results: Semantic Segmentation under Scribble Annotations

| Method | 3D Supervision | Backbone | Waymo mIoU | SS/FS% | KITTI-360 mIoU | SS/FS% | NYUv2 mIoU | SS/FS% |
|--------|---------------|----------|-----------|--------|---------------|--------|-----------|--------|
| Fully Supervised | — | SegFormer-B4 | 59.0 | — | 68.4 | — | 51.1 | — |
| EMA (baseline) | — | SegFormer-B4 | 49.4 | 83.7 | 60.3 | 88.2 | 42.9 | 84.0 |
| SASFormer | — | SegFormer-B4 | 37.8 | 64.1 | 46.4 | 67.8 | 44.7 | 87.5 |
| TEL | — | DeepLabV3+ | 42.4 | 71.9 | 59.2 | 86.6 | 38.3 | 75.0 |
| **Ours (Real 3D)** | LiDAR/Depth | SegFormer-B4 | 51.8 | 87.8 | 61.7 | 90.2 | 44.7 | 87.6 |
| **Ours (Recon)** | Reconstruction | SegFormer-B4 | **53.3** | **90.3** | **63.4** | **93.4** | **46.1** | **90.2** |

### Ablation Study (Waymo Dataset)

| Configuration | Confidence Filtering | Sampling Strategy | 3D Source | mIoU |
|---------------|---------------------|------------------|-----------|------|
| EMA baseline (2D only) | — | — | — | 49.4 |
| No filtering | ❌ | View-aware | Multi-view reconstruction | 51.9 |
| + Prediction confidence | Prediction | View-aware | Multi-view reconstruction | 52.7 |
| + Reconstruction confidence | Reconstruction | View-aware | Multi-view reconstruction | 52.1 |
| + Dual confidence (Ours) | Dual | View-aware | Multi-view reconstruction | **53.3** |
| Random sampling | Dual | Random | Multi-view reconstruction | 51.9 |
| Single-frame reconstruction | Dual | View-aware | Single-frame | 52.1 |

### Generalization Across Annotation Types (Cityscapes)

| Method | Point | Scribble | Coarse Label |
|--------|-------|---------|-------------|
| Fully Supervised | 77.6 | 77.6 | 77.6 |
| TEL | 53.1 | 64.4 | 64.9 |
| SASFormer | 42.7 | 55.6 | 42.8 |
| EMA (baseline) | 50.5 | 61.2 | 66.5 |
| **Ours** | **56.5** (+6.0) | **68.1** (+6.9) | **68.6** (+2.1) |

### Key Findings

1. **Reconstructed 3D outperforms real 3D**: This counterintuitive result stems from two factors — reconstructed point clouds are typically denser and more complete than LiDAR, and dual confidence filtering suppresses reconstruction noise (real LiDAR lacks a reconstruction confidence metric).
2. **View-aware sampling is critical**: Compared to random sampling (~140 correspondences per image), view-aware sampling guarantees ~72K correspondences, yielding a 1.4% mIoU improvement.
3. **Dual confidence outperforms single confidence**: Prediction confidence and reconstruction confidence capture complementary aspects of reliability.
4. **Multi-view reconstruction outperforms single-frame**: Multi-view provides richer geometric context and more reliable depth estimation (+1.2 mIoU).
5. **The method generalizes across annotation types**: Consistent gains are observed across point, scribble, and coarse label annotations, with the largest improvements under the sparsest supervision.
6. **The supervision gap is substantially closed**: On KITTI-360, the method bridges 93.4% of the gap between weakly-supervised and fully-supervised performance.

## Highlights & Insights

- **Paradigm innovation**: Rewis3d is the first to employ feedforward 3D reconstruction as auxiliary supervision for weakly-supervised segmentation — unlike methods that directly use LiDAR or perform segmentation in 3D, it leverages reconstructed geometry to enhance 2D segmentation while maintaining a purely 2D inference pipeline.
- **Counterintuitive finding**: Reconstructed 3D (derived from 2D video) surpasses real LiDAR/depth 3D, owing to advantages in point cloud density and filterability.
- **Elegant cross-modal design**: The dual student-teacher structure combined with dual confidence weighting ensures reliable cross-modal knowledge transfer without over-relying on either modality.
- **General-purpose framework**: The approach is not tied to a specific segmentation architecture (validated with SegFormer and EoMT), annotation type (points/scribbles/coarse labels), or scene type (outdoor/indoor).
- **Comprehensive experimental coverage**: Evaluated on four datasets (Waymo, KITTI-360, Cityscapes, NYUv2), three annotation types, with thorough ablation studies.

## Limitations & Future Work

1. **Reconstruction noise in dynamic scenes**: The 3D reconstruction model (MapAnything) is not optimized for dynamic content; moving objects in driving scenes introduce geometric noise and depth uncertainty.
2. **Reconstruction computational overhead**: Although inference incurs no additional cost, the 3D reconstruction preprocessing during training (200+ images → 60M+ point clouds) is computationally non-trivial.
3. **Limited gains for single-frame scenes**: Cityscapes is processed frame-by-frame without multi-view video, which constrains performance improvements.
4. **No integration of dynamic-aware reconstruction**: Incorporating reconstruction models that explicitly handle dynamic scenes is a clear direction for improvement.
5. **Underutilization of the 3D branch at inference**: A fully trained 3D branch is discarded at inference; its potential value remains unexplored.

## Related Work & Insights

- **MapAnything / DUSt3R / VGGT**: Recent advances in feedforward multi-view reconstruction that make high-quality 3D geometry obtainable from 2D images — the foundational infrastructure for Rewis3d.
- **Mean Teacher**: Boettcher et al. (2024) demonstrate that Mean Teacher remains competitive under sparse annotations; Rewis3d builds on this by incorporating 3D geometric supervision.
- **SASFormer / TEL**: Current WSSS state of the art, but limited to propagating information within the 2D plane without exploiting geometric consistency.
- **2DPASS**: Distills knowledge from 2D to 3D for LiDAR segmentation — the reverse direction of Rewis3d (3D → 2D).
- **Insight**: Cross-modal consistency is a powerful self-supervisory signal, especially when different modalities (2D appearance vs. 3D geometry) provide complementary information.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐⭐⭐ |
| Theoretical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐⭐ |
| Overall | ⭐⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VirPro: Visual-referred Probabilistic Prompt Learning for Weakly-Supervised Monocular 3D Detection](virpro_visual-referred_probabilistic_prompt_learning_for_weakly-supervised_monoc.md)
- [\[CVPR 2026\] JOPP-3D: Joint Open Vocabulary Semantic Segmentation on Point Clouds and Panoramas](jopp3d_joint_open_vocabulary_semantic_segmentation.md)
- [\[AAAI 2026\] Enhancing Generalization of Depth Estimation Foundation Model via Weakly-Supervised Adaptation with Regularization](../../AAAI2026/3d_vision/enhancing_generalization_of_depth_estimation_foundation_model_via_weakly-supervi.md)
- [\[CVPR 2026\] E-RayZer: Self-supervised 3D Reconstruction as Spatial Visual Pre-training](e-rayzer_self-supervised_3d_reconstruction_as_spatial_visual_pre-training.md)
- [\[CVPR 2026\] Regularizing INR with Diffusion Prior for Self-Supervised 3D Reconstruction of Neutron Computed Tomography Data](regularizing_inr_with_diffusion_prior_self-supervised_3d_reconstruction_of_neutr.md)

</div>

<!-- RELATED:END -->

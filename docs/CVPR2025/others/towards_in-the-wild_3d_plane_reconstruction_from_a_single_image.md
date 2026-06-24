---
title: >-
  [Paper Note] Towards In-the-Wild 3D Plane Reconstruction from a Single Image
description: >-
  [CVPR 2025][Plane Reconstruction] ZeroPlane proposes the first cross-domain zero-shot 3D plane reconstruction framework. By constructing a large-scale plane benchmark dataset with 14 datasets and 560k annotations, and designing a normal-offset decoupled classification-regression paradigm along with a pixel-level geometric enhanced embedding module, it achieves generalization performance significantly outperforming existing methods in diverse indoor and outdoor scenes.
tags:
  - "CVPR 2025"
  - "Plane Reconstruction"
  - "Zero-shot Generalization"
  - "Single-image 3D Reconstruction"
  - "Cross-domain Learning"
  - "Transformer"
date: 2026-05-08
content_hash: 02fb1313fdb1e7a8
---

# Towards In-the-Wild 3D Plane Reconstruction from a Single Image

**Conference**: CVPR 2025  
**arXiv**: [2506.02493](https://arxiv.org/abs/2506.02493)  
**Code**: [https://github.com/jcliu0428/ZeroPlane](https://github.com/jcliu0428/ZeroPlane)  
**Area**: LLM Evaluation  
**Keywords**: Plane Reconstruction, Zero-shot Generalization, Single-image 3D Reconstruction, Cross-domain Learning, Transformer

## TL;DR

ZeroPlane proposes the first cross-domain zero-shot 3D plane reconstruction framework. By constructing a large-scale plane benchmark dataset with 14 datasets and 560k annotations, and designing a normal-offset decoupled classification-regression paradigm along with a pixel-level geometric enhanced embedding module, it achieves generalization performance significantly outperforming existing methods in diverse indoor and outdoor scenes.

## Background & Motivation

**Background**: Single-image 3D plane reconstruction is a fundamental task in 3D vision, applied in AR, localization and mapping, and robotics. Existing methods (PlaneNet, PlaneRCNN, PlaneTR, PlaneRecTR) are mostly trained and tested on a single dataset, focusing on either indoor scenes (ScanNet) or outdoor scenes (Synthia), performing well in their respective domains.

**Limitations of Prior Work**: (1) No existing method has attempted cross-domain generalizable plane reconstruction, where the geometric scales and plane distributions of indoor and outdoor scenes differ vastly; (2) High-quality outdoor plane annotation data is scarce, and existing outdoor datasets lack dense plane mask annotations; (3) Existing methods are trained and tested at low resolutions (256×192), which limits reconstruction quality.

**Key Challenge**: The geometric parameter distributions of planes in indoor and outdoor scenes vary enormously (indoor offsets are typically <5m, while outdoor offsets can reach tens or even hundreds of meters). Existing methods couple the normal and offset into a single scaled vector $\mathbf{n}/d$ for regression, failing to adapt to such massive geometric scale variations during joint training.

**Goal**: To construct a unified cross-domain 3D plane reconstruction system capable of zero-shot generalization across diverse indoor, outdoor, and even in-the-wild scenarios.

**Key Insight**: Inspired by zero-shot depth estimation (MiDaS, Depth Anything), training an integrated model on mixed large-scale datasets can significantly boost generalization ability. The authors extend this idea to the more challenging task of plane reconstruction.

**Core Idea**: Decouple the representation of normal and offset, applying a "classification-regression" paradigm to both (first classifying to the nearest cluster center/exemplar, then regressing the residual) to alleviate the difficulty of learning cross-domain geometric parameters.

## Method

### Overall Architecture

ZeroPlane is built upon the detection-segmentation framework of Mask2Former and PlaneRecTR. Given a single input image, multi-scale features are extracted using a DINOv2-Base encoder and passed to a DPT pixel decoder to obtain multi-resolution feature maps. Learnable plane queries interact with image features within a Transformer decoder to output instance-level predictions: segmentation masks, classification scores, normals, and offsets. Additional pixel-level depth/normal predictions serve as auxiliary tasks, and their features are injected into plane queries via a geometric enhancement module.

### Key Designs

1. **Normal-Offset Decoupled + Classification-Regression Paradigm**:

    - **Function**: Solves the core challenge of learning difficult geometric parameter regression under cross-domain mixed training.
    - **Mechanism**: Decouples the plane parameter $\mathbf{n}/d$ into normal $\mathbf{n}$ and offset $d$ for separate learning. A classification-regression strategy is applied to each parameter: first clustering mixed training data via K-Means to obtain $K_n=7$ normal exemplars and $K_d=20$ offset exemplars (offsets are grouped into indoor/outdoor sets with a 20m threshold, clustering 10 exemplars per group). An MLP classification head predicts the corresponding exemplar, and a regression head predicts the relative residual. The final predictions are: $\mathbf{n} = \hat{\mathbf{n}}^{(i)} + \mathbf{r_n}^{(i)}$, $d = \hat{d}^{(j)} + r_d^{(j)}$
    - **Design Motivation**: Direct regression requires the network to simultaneously handle drastically different geometric scales (indoor <5m vs. outdoor >50m), which is highly challenging. Classification first brings the problem to a coarse, appropriate range, while regression only needs to fine-tune the residual, vastly reducing learning difficulty.

2. **Pixel-Level Geometric Enhanced Plane Embedding Module**:

    - **Function**: Injects low-level pixel-level geometric cues (depth, normals) into plane queries to enhance instance-level predictions.
    - **Mechanism**: Two CNN modules are appended after the encoder to predict pixel-level object depth maps $\mathbf{D}$ and normal maps $\mathbf{N}$ (as auxiliary tasks). They are projected into embedding spaces $\mathbf{F_D}$ and $\mathbf{F_N}$, after which plane queries interact with the geometric features via cross-attention: $\mathbf{X_D} = \text{Attn}(\mathbf{Q}, \mathbf{F_D})$, $\mathbf{X_N} = \text{Attn}(\mathbf{Q}, \mathbf{F_N})$. The final embedding is $\mathbf{X} = \mathbf{X_F} + \mathbf{X_D} + \mathbf{X_N}$.
    - **Design Motivation**: Simply training pixel depth/normals as auxiliary tasks provides limited improvement because plane queries cannot directly leverage this low-level geometric information. The attention mechanism enables queries to discover fine-grained contextual cues (such as plane boundaries) from pixel-level geometric predictions.

3. **Large-scale Cross-domain Plane Benchmark Dataset**:

    - **Function**: Provides high-resolution dense plane annotations across diverse indoor and outdoor environments.
    - **Mechanism**: Consolidates 14 datasets (7 indoor + 4 outdoor), totaling 560k high-resolution (640×480) annotations. For datasets lacking semantic annotations, Mask2Former is used to obtain panoptic segmentation results as pseudo-ground truths, and planes are then fitted on back-projected point clouds for each object/region. For outdoor scenes, synthetic datasets (with accurate depth maps) or stereo camera data are utilized to generate plane annotations.
    - **Design Motivation**: Existing plane datasets only cover indoor environments with low resolutions, which cannot support cross-domain training. High-resolution annotations are a prerequisite for high-quality plane reconstruction.

### Loss & Training

- **Total Loss**: $L = \lambda_c L_c + \lambda_m L_m + \lambda_{n_c} L_{n_c} + \lambda_{n_r} L_{n_r} + \lambda_{d_c} L_{d_c} + \lambda_{d_r} L_{d_r} + \lambda_{p_d} L_{p_d} + \lambda_{p_n} L_{p_n}$
- Cross-entropy loss is used for classification, L1 loss for regression, and a combined cross-entropy and Dice loss for masks.
- Bipartite matching strategy (Hungarian matching) is employed to match predictions with ground truths.
- **Training**: AdamW optimizer, lr=1e-4, batch size=16, 50K steps, with a 10x decay at 40K and 47K steps.

## Key Experimental Results

### Main Results — Zero-shot Evaluation

| Method | NYUv2 Recall@0.1m | NYUv2 Recall@5° | 7-Scenes Recall@0.1m |
|------|-------------------|-----------------|---------------------|
| PlaneRecTR (S) | 10.13 | 16.42 | - |
| PlaneRecTR (M) | 14.29 | 24.97 | 10.97 |
| **ZeroPlane-DINO-B (M)** | **17.86** | **37.29** | **17.19** |
| ZeroPlane-Dust3R (M) | 21.20 | 38.32 | 19.14 |

### Ablation Study

| Design | NYUv2 Recall@0.1m |
|------|-------------------|
| Coupled representation ($\mathbf{n}/d$ direct regression) | ~12 |
| Decoupled + Direct Regression | ~14 |
| Decoupled + Classification-Regression | **17.86** |
| W/O Geometric Enhancement Module | Dec. by ~1-2% |
| Auxiliary task without attention injection | Limited gain |

### Key Findings

- Mixed training (M) significantly outperforms single dataset training (S), validating the complementary value of cross-domain data.
- Decoupled normal/offset + classification-regression yields huge improvements compared to direct regression, especially in depth reconstruction accuracy.
- Inserting the geometric enhanced embedding module via the attention mechanism is effective, whereas treating it purely as an auxiliary task offers limited help.
- Stronger encoders (DINOv2-L, Dust3R) further improve performance, though the base version already leads by a large margin.
- The advantage on outdoor zero-shot data is even more pronounced, reflecting strong cross-domain generalization capability.

## Highlights & Insights

- First to introduce the concept of zero-shot generalization to the field of 3D plane reconstruction, opening up a new research direction.
- The "classification-regression" paradigm elegantly solves the problem of vast distribution differences in cross-domain geometric parameters, showing good generalizability.
- The constructed large-scale cross-domain benchmark dataset itself is a significant contribution, serving as a foundation for future research.
- The introduction of the DINOv2 encoder significantly enhances the cross-domain robustness of features.

## Limitations & Future Work

- Outdoor data mostly originates from synthetic datasets or stereo cameras; obtaining high-quality annotations for real-world outdoor scenes remains challenging.
- The indoor/outdoor segmentation threshold (20m) for offset exemplars is manually set, which may lack flexibility.
- Reconstruction of non-planar regions (e.g., curved surfaces) is not addressed, focusing only on planes.
- Integrating depth foundation models (e.g., Depth Anything) could be explored to further improve geometric estimation accuracy.

## Related Work & Insights

- **PlaneRecTR**: The previous SOTA Transformer plane detector; ZeroPlane introduces cross-domain training strategies on top of it.
- **MiDaS/Depth Anything**: The success of zero-shot depth estimation directly inspired this work.
- **Mask2Former**: Provided the overall architecture and bipartite matching strategy.
- **Dust3R**: Served as an alternative encoder to further enhance geometric awareness.

## Rating

- **Novelty**: 8/10 — First to propose cross-domain zero-shot plane reconstruction, with an elegantly designed classification-regression paradigm.
- **Experimental Thoroughness**: 9/10 — 14 datasets, multiple encoders, comprehensive ablation studies, with zero-shot evaluation covering both indoor and outdoor environments.
- **Writing Quality**: 8/10 — Thorough problem analysis and clear motivation for methods.
- **Value**: 8/10 — Dual contribution of both dataset and methodology, opening a new avenue of research in 3D plane reconstruction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Image Reconstruction from Readout-Multiplexed Single-Photon Detector Arrays](image_reconstruction_from_readout-multiplexed_single-photon_detector_arrays.md)
- [\[CVPR 2025\] Regor: Progressive Correspondence Regenerator for Robust 3D Registration](progressive_correspondence_regenerator_for_robust_3d_registration.md)
- [\[CVPR 2025\] MagicArticulate: Make Your 3D Models Articulation-Ready](magicarticulate_make_your_3d_models_articulation-ready.md)
- [\[CVPR 2025\] Rooftop Wind Field Reconstruction Using Sparse Sensors: From Deterministic to Generative Learning Methods](rooftop_wind_field_reconstruction_using_sparse_sensors_from_deterministic_to_gen.md)
- [\[CVPR 2025\] LATTE-MV: Learning to Anticipate Table Tennis Hits from Monocular Videos](latte-mv_learning_to_anticipate_table_tennis_hits_from_monocular_videos.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] MARCO: Navigating the Unseen Space of Semantic Correspondence
description: >-
  [CVPR 2026][3D Vision][Semantic Correspondence] This paper proposes MARCO, a semantic correspondence model built on a single DINOv2 backbone. It progressively improves spatial precision via a coarse-to-fine Gaussian RBF loss, and expands sparse keypoint supervision into dense pseudo-correspondence labels through a self-distillation framework. MARCO achieves state-of-the-art performance on standard benchmarks as well as on unseen keypoints and categories, while being 3× smaller and 10× faster than dual-encoder approaches.
tags:
  - CVPR 2026
  - 3D Vision
  - Semantic Correspondence
  - DINOv2
  - Self-Distillation
  - Coarse-to-Fine
  - Generalization
date: 2026-05-08
content_hash: b09be64aea9493e0
---

# MARCO: Navigating the Unseen Space of Semantic Correspondence

**Conference**: CVPR 2026
**arXiv**: [2604.18267](https://arxiv.org/abs/2604.18267)
**Code**: [https://visinf.github.io/MARCO](https://visinf.github.io/MARCO)
**Area**: 3D Vision
**Keywords**: Semantic Correspondence, DINOv2, Self-Distillation, Coarse-to-Fine, Generalization

## TL;DR

This paper proposes MARCO, a semantic correspondence model built on a single DINOv2 backbone. It progressively improves spatial precision via a coarse-to-fine Gaussian RBF loss, and expands sparse keypoint supervision into dense pseudo-correspondence labels through a self-distillation framework. MARCO achieves state-of-the-art performance on standard benchmarks as well as on unseen keypoints and categories, while being 3× smaller and 10× faster than dual-encoder approaches.

## Background & Motivation

**State of the Field**: Semantic correspondence aims to establish pixel-level matches between semantically equivalent regions. Recent dominant methods adopt dual-encoder architectures combining DINOv2 (for robust semantic alignment) with Stable Diffusion (for rich spatial detail), such as Geo-SC and SD+DINO. These methods perform well on benchmarks but have parameter counts approaching one billion.

**Limitations of Prior Work**: (1) Dual-encoder designs incur substantial computational cost, requiring feature extraction from two separate encoders. (2) More critically, models trained on sparse keypoints generalize poorly to unseen keypoints and unseen categories at test time, since query points rarely coincide with annotated positions in practice. This exposes a fundamental gap between benchmark performance and real-world applicability.

**Root Cause**: Sparse keypoint supervision causes models to overfit to annotated locations. Fine-tuned DINOv2 achieves higher accuracy near annotated keypoints, but the broader surface-level consistency that originally spanned the entire object is degraded—representations collapse toward keypoint neighborhoods.

**Paper Goals**: (1) Improve precision on standard benchmarks, particularly at fine-grained localization thresholds. (2) Substantially enhance generalization to unseen keypoints and unseen categories. (3) Preserve the efficiency advantage of a single-backbone design.

**Starting Point**: Although frozen DINOv2 features exhibit limited spatial consistency, they already encode sparse but reliable correspondence cues. These cues can be exploited during training to automatically discover and propagate dense correspondences, extending supervision from a handful of keypoints to the full object surface.

**Core Idea**: A coarse-to-fine supervision objective improves spatial precision, while a self-distillation framework combined with flow anchoring expands sparse keypoints into dense pseudo-labels covering the entire object surface, encouraging features to remain smooth across the whole object rather than collapsing near annotated points.

## Method

### Overall Architecture

MARCO builds on a DINOv2 backbone with only two lightweight components: a bottleneck adapter (AdaptFormer, <5% parameter overhead) and a compact upsampling head (transposed convolution + depthwise convolution, 4× feature resolution increase). Training employs two complementary objectives: a coarse-to-fine supervision loss and a self-distillation dense correspondence loss.

### Key Designs

1. **Coarse-to-Fine Gaussian RBF Loss**:

    - **Function**: Guides correspondence matching progressively from coarse region-level alignment to sub-patch-level precise localization.
    - **Mechanism**: A cross-entropy loss supervises the predicted probability map against a Gaussian RBF kernel centered at the GT keypoint. The key innovation is cosine annealing of the bandwidth $\sigma$: $\sigma(t) = \sigma_{min} + \frac{1}{2}(\sigma_{max} - \sigma_{min})(1 + \cos(\pi t/T))$. A large $\sigma$ (wide kernel) early in training encourages region-level alignment; a small $\sigma$ (narrow kernel) later enforces precise localization.
    - **Design Motivation**: Training directly with a small $\sigma$ yields precise matches at a few high-confidence locations but degrades overall accuracy. Training with a large $\sigma$ produces broad but coarse matches. The annealing strategy first establishes stable regional alignment before progressively tightening the target, capturing the benefits of both regimes.

2. **Dense Self-Distillation via Flow Anchoring**:

    - **Function**: Expands sparse keypoint supervision into dense pseudo-correspondence labels covering the object surface.
    - **Mechanism**: (a) Mutual nearest-neighbor matches $\mathcal{P}_{MNN}$ are extracted from the EMA teacher network's features and merged with GT keypoints to form a seed set. (b) A Delaunay triangulation is constructed on the source endpoints of the seed set; piecewise affine transformations between triangle pairs yield a dense flow field $\mathbf{D}(\mathbf{u})$. (c) K-means clustering in displacement space identifies coherent motion regions, with $k$ automatically selected via BIC. (d) Only clusters containing GT keypoint pairs are retained as reliable pseudo-labels—i.e., regions whose flow direction is consistent with GT correspondences.
    - **Design Motivation**: Direct dense matching using DINOv2 features introduces substantial errors (due to symmetry, occlusion, etc.). The flow anchoring strategy cleverly uses GT keypoints as anchors to validate the reliability of discovered correspondences.

3. **Lightweight Architectural Enhancements**:

    - **Function**: Improves feature quality and spatial resolution without substantially increasing parameter count.
    - **Mechanism**: AdaptFormer inserts bottleneck adapters ($\mathbf{W}_{down} \in \mathbb{R}^{D \times d}$, $d \ll D$) into upper Transformer blocks in a residual manner. The upsampling head achieves 4× upsampling via 2× transposed convolution + GELU + 3×3 depthwise convolution, lifting 14×14 patch-level features to sub-patch resolution.
    - **Design Motivation**: Keeping the backbone frozen and training only the adapters fully exploits DINOv2's pretrained representations while avoiding the overfitting risks associated with large-scale fine-tuning.

### Loss & Training

The total loss is $\mathcal{L} = \mathcal{L}_{sup} + \mathcal{L}_{self}$. The supervised loss uses CE with Gaussian RBF annealing; the self-distillation loss uses L2 regression, which is more robust to noisy pseudo-labels. The teacher network is an EMA of the student.

## Key Experimental Results

### Main Results

| Dataset | Threshold | MARCO | Prev. SOTA (Geo-SC) | Gain |
|--------|------|-------|----------------|------|
| SPair-71k | PCK@0.10 | **Best** | 2nd | +4.0 |
| SPair-71k | PCK@0.01 | **Best** | 2nd | **+8.9** |
| AP-10K (Intra) | PCK@0.10 | **Best** | 2nd | +2.9 |
| PF-PASCAL | PCK@0.10 | **Best** | 2nd | Gain |

### Generalization Results

| Setting | MARCO | Prev. SOTA (Jamais Vu) | Gain |
|------|-------|-------------------|------|
| SPair-U (Unseen Keypoints) | **Best** | 2nd | **+5.1** |
| MP-100 (Unseen Categories) | **Best** | 2nd | **+5.6** |

### Ablation Study

| Configuration | SPair PCK@0.10 | SPair-U | Notes |
|------|---------------|---------|------|
| Full MARCO | Best | Best | Complete method |
| w/o coarse-to-fine annealing | Drops | Drops | Localization precision impaired |
| w/o self-distillation | Drops | Drops significantly | Generalization degrades sharply |
| w/o upsampling head | Drops | — | Sub-patch precision limited |

### Key Findings

- MARCO's advantage at the fine-grained threshold PCK@0.01 (+8.9) substantially exceeds its advantage at PCK@0.10 (+4.0), demonstrating the effectiveness of the coarse-to-fine strategy for precise localization.
- Self-distillation is the decisive factor for generalization—without it, fine-tuned DINOv2 performs even worse than the frozen model on unseen keypoints.
- The single-backbone approach surpasses dual-encoder methods while being 3× smaller and 10× faster, indicating that training strategy rather than architectural scale is the critical factor.

## Highlights & Insights

- The flow-anchoring self-distillation design is particularly elegant: mining sparse reliable matches from frozen encoder features → densification via Delaunay triangulation → displacement clustering with GT-anchor filtering. Each step has a clear purpose and the stages connect seamlessly.
- The observation that sparse supervision causes representation collapse is incisive—fine-tuning improves performance near keypoints but degrades object-level consistency (the flow visualizations in Figure 2 are highly illustrative). Self-distillation directly addresses this pathology.
- The paper introduces new generalization benchmarks (unseen keypoint and unseen category evaluations based on MP-100), providing more rigorous evaluation standards for the field.

## Limitations & Future Work

- Self-distillation relies on the existence of sparse, reliable correspondences already present in DINOv2's feature space; the method may be limited for object categories where the pretrained representations lack such structure.
- Delaunay triangulation cannot generate pseudo-labels for regions outside the convex hull of the seed points.
- While independence from 3D priors is an advantage, it also limits the method's capacity to handle severely deformed objects.
- Future directions include incorporating video temporal consistency to provide richer dense correspondence signals.

## Related Work & Insights

- **vs. Geo-SC / dual-encoder methods**: MARCO surpasses them with a single backbone, demonstrating that a well-designed training strategy can compensate for architectural simplicity.
- **vs. Jamais Vu**: Both target generalization to unseen keypoints, but Jamais Vu relies on 3D templates and is constrained to trained categories. MARCO's self-distillation requires no category priors or 3D information.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Flow-anchoring self-distillation is a highly original training paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Standard benchmarks and generalization benchmarks with detailed ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem analysis is incisive and methodological derivation is elegant.
- Value: ⭐⭐⭐⭐⭐ Simultaneously achieves large gains in precision and generalization with high efficiency; a significant advance in correspondence estimation.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Do It Yourself: Learning Semantic Correspondence from Pseudo-Labels](../../ICCV2025/3d_vision/do_it_yourself_learning_semantic_correspondence_from_pseudo-labels.md)
- [\[CVPR 2026\] MimiCAT: Mimic with Correspondence-Aware Cascade-Transformer for Category-Free 3D Pose Transfer](mimicat_mimic_with_correspondence-aware_cascade-transformer_for_category-free_3d.md)
- [\[CVPR 2026\] RayNova: Scale-Temporal Autoregressive World Modeling in Ray Space](raynova_scale-temporal_autoregressive_world_modeling_in_ray_space.md)
- [\[CVPR 2026\] DuoMo: Dual Motion Diffusion for World-Space Human Reconstruction](duomo_dual_motion_diffusion_for_world-space_human_reconstruction.md)
- [\[ICLR 2026\] A Genetic Algorithm for Navigating Synthesizable Molecular Spaces](../../ICLR2026/3d_vision/a_genetic_algorithm_for_navigating_synthesizable_molecular_spaces.md)

<!-- RELATED:END -->

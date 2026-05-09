---
title: >-
  [Paper Note] ReLaGS: Relational Language Gaussian Splatting
description: >-
  [CVPR2026][3D Vision][3D Gaussian Splatting] This paper proposes ReLaGS, the first training-free framework that unifies multi-level language Gaussian fields with open-vocabulary 3D scene graphs. It improves scene representation via Maximum Weight Pruning and Robust Outlier-aware Feature Aggregation, and achieves efficient structured 3D scene understanding through GNN-based relation prediction.
tags:
  - CVPR2026
  - 3D Vision
  - 3D Gaussian Splatting
  - Open-Vocabulary
  - 3D Scene Graph
  - Hierarchical Semantics
  - Relation Reasoning
  - Training-Free
date: 2026-05-08
content_hash: a1424555b2fa6d2f
---

# ReLaGS: Relational Language Gaussian Splatting

**Conference**: CVPR2026
**arXiv**: [2603.17605](https://arxiv.org/abs/2603.17605)
**Code**: [Project Page](https://dfki-av.github.io/ReLaGS/)
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, Open-Vocabulary, 3D Scene Graph, Hierarchical Semantics, Relation Reasoning, Training-Free

## TL;DR

This paper proposes ReLaGS, the first training-free framework that unifies multi-level language Gaussian fields with open-vocabulary 3D scene graphs. It improves scene representation via Maximum Weight Pruning and Robust Outlier-aware Feature Aggregation, and achieves efficient structured 3D scene understanding through GNN-based relation prediction.

## Background & Motivation

**Radiance fields lack semantics**: Although NeRF/3DGS excel at geometric and photometric reconstruction, they lack scene semantic information and cannot support high-level reasoning tasks such as navigation, editing, and question answering.

**Limitations of language field distillation**: Existing language field methods (LangSplat, LERF, etc.) only encode "what objects exist" and cannot handle queries involving spatial relations such as "select the cup next to the laptop," as they are **single-level and isolated**—lacking hierarchical semantics and inter-entity relationships.

**Absence of hierarchical granularity**: Users may describe whole objects ("ramen") or their parts ("noodles"); a single semantic granularity cannot distinguish part-level from object-level queries, making it difficult to accommodate the ambiguity of natural language.

**High cost of relation modeling**: RelationField learns relations via ray pairs but requires hours of per-scene training and renders below 10 fps; SplatTalk requires LLM tokenization and LoRA fine-tuning, incurring substantial cost.

**Multi-view feature inconsistency**: SAM masks are inconsistent across viewpoints, CLIP features are noisy, and direct average aggregation causes object embeddings to be corrupted by outliers.

**Limitations of scene graph methods**: ConceptGraphs relies on expensive LLM inference and produces text graphs; GaussianGraph requires per-scene training; Open3DSG is restricted to pre-segmented point clouds—none provides a unified, efficient open-vocabulary 3D scene graph solution.

## Method

### Overall Architecture

ReLaGS consists of three stages, **all without per-scene training**:

1. **Maximum Weight Pruning (MWP)**: Removes floater Gaussians that contribute negligibly to all training views from the reconstructed Gaussian field, purifying the geometric structure.
2. **Multi-level Gaussian field construction**: Adopts the gradient-free hierarchical strategy of THGS to cluster Gaussians from superpoints → sub-parts → parts → objects, combined with Robust Outlier-aware Feature Aggregation (ROFA) to produce reliable language embeddings.
3. **3D scene graph construction**: Builds an explicit open-vocabulary scene graph atop the hierarchical representation, supporting two relation acquisition modes: LLM-augmented labeling and GNN-based prediction.

### Key Designs

**Maximum Weight Pruning (MWP)**:

- For each Gaussian $G_i$, compute its maximum contribution weight across all views and pixels: $\omega_i^{\max} = \max_{c,p} w_{i,p}^{(c)}$
- Prune Gaussians with $\omega_i^{\max} < \tau_{contrib}$, which typically correspond to floaters at boundaries or in occluded regions.
- Ablation studies confirm this step contributes most to performance gain (+6.16 mIoU).

**Robust Outlier-aware Feature Aggregation (ROFA)**:

- For an object's CLIP features $\{f_i\}$ observed across $\mathcal{C}_{obj}$ views, compute the average cosine similarity $s_i$ of each feature against all others.
- Apply Z-score normalization $z_i = (s_i - \mu_s)/\sigma_s$ and filter features with $z_i < -\tau_{lang}$.
- Average only the remaining consistent features to obtain a stable object language embedding.
- The threshold $\tau_{lang}=3$ yields optimal performance.

**Multi-level Gaussian representation**:

- Defines $L$ nested hierarchical levels $\mathcal{S}^{(1)}, \dots, \mathcal{S}^{(L)}$, with finer-grained parts at lower levels and complete objects at higher levels.
- Establishes consistent 2D–3D correspondences via a pixel-to-dominant-Gaussian mapping ($G^*_{(u,v)} = \arg\max_i w_i$).
- Designs a tree-search query algorithm that descends from the root node when a child node yields higher similarity to the query, automatically inferring the appropriate query granularity.

**Scene graph construction — dual-path design**:

- **LLM augmentation path**: Render view-consistent cluster ID maps → Set-of-Mark (SoM) annotation → GPT-4V relation predicate inference → select top-$k_p$ frequency predicates → encode with Jina and average as edge embeddings (semantically rich but sparse).
- **GNN prediction path**: Construct a neighbor graph within a distance threshold, then apply a residual graph neural network $\mathcal{F}_\theta$ to predict relation embeddings $\hat{f}_{ij} = f'_{ij} + \mathcal{F}_\theta(f_v^{src}, f_v^{dst}, f'_{ij})$; pretrained on 3RScan via contrastive learning and directly generalized to new scenes (efficient and scalable).

### Loss & Training

- The GNN is pretrained using a **contrastive learning loss** that aligns predicted relation embeddings with ground-truth relation embeddings in the Jina embedding space.
- The overall framework requires no per-scene gradient optimization.

## Key Experimental Results

### 3D Scene Graph Prediction (3DSSG/RIO10)

| Method | Object R@5 | Object R@10 | Predicate R@3 | Predicate R@5 | Scene-agnostic |
|--------|-----------|------------|--------------|--------------|----------------|
| ConceptGraphs | 0.37 | 0.46 | 0.74 | 0.79 | ✗ |
| RelationField | 0.69 | 0.80 | 0.76 | 0.82 | ✗ |
| Open3DSG | 0.56 | 0.61 | 0.58 | 0.65 | ✓ |
| **ReLaGS (GNN)** | **0.68** | **0.79** | **0.79** | **0.87** | **✓** |

- ReLaGS outperforms RelationField in relation prediction by +0.3 R@3 / +0.5 R@5, without any per-scene training.
- **4.7× faster** and **7.6× more memory-efficient** than RelationField (7.5 GB vs. 32 GB GPU memory).

### Relation-guided 3D Instance Segmentation (ScanNet++)

| Method | mIoU | Scene-agnostic |
|--------|------|----------------|
| LERF | 0.25 | ✗ |
| OpenNeRF | 0.45 | ✗ |
| LangSplat | 0.49 | ✗ |
| RelationField | 0.53 | ✗ |
| THGS | 0.29 | ✓ |
| **ReLaGS** | **0.56** | **✓** |

### Open-Vocabulary Segmentation (LERF-OVS)

| Method | Figurines | Ramen | Teatime | Waldo | Mean | Training-free |
|--------|-----------|-------|---------|-------|------|--------------|
| LAGA | 64.1 | 55.6 | 70.9 | 65.6 | 64.0 | ✗ |
| THGS | 57.3 | 43.5 | 68.3 | 50.7 | 54.9 | ✓ |
| VALA | 59.9 | 51.5 | 70.2 | 65.1 | 61.7 | ✓ |
| **ReLaGS** | **64.7** | 51.2 | **81.0** | 60.6 | **64.4** | **✓** |

### Ablation Study

| Configuration | Figurines | Ramen | Teatime | Kitchen | Mean |
|--------------|-----------|-------|---------|---------|------|
| Baseline | 52.05 | 47.19 | 76.77 | 47.50 | 55.88 |
| +MWP | 59.16 | 47.41 | 80.98 | 60.59 | 62.04 |
| +MWP+ROFA (full) | 64.69 | 51.15 | 80.98 | 60.60 | 64.36 |

### Key Findings

- MWP contributes the most (+6.16 mIoU); removing floater Gaussians is critical for geometry quality and downstream clustering.
- ROFA yields significant gains in densely occluded scenes (Figurines +5.53, Ramen +3.74).
- $\tau_{lang}=3$ is the optimal threshold; both lower and higher values degrade performance.
- The GNN generalizes well across datasets (3RScan → ScanNet++), indicating a small modality gap between language Gaussians and point cloud features.
- The full pipeline completes in approximately 12.6 minutes (scene reconstruction 11 min + language distillation 1.5 min + scene graph 0.1 min), with rendering at 200+ fps.

## Highlights & Insights

- **First unified framework**: Simultaneously supports multi-level semantic hierarchy and open-vocabulary relation reasoning, addressing "what exists," "how it is composed," and "how entities relate."
- **Fully training-free**: 4.7× faster and 7.6× more memory-efficient than RelationField, enabling truly scalable 3D scene understanding.
- **Elegant MWP + ROFA combination**: Purifies geometry and semantics respectively; simple, effective, and thoroughly validated by ablation.
- **Dual-path scene graph design**: LLM augmentation provides semantically rich edges; GNN prediction provides efficient coverage; the two paths are highly complementary.
- **Tree-search querying**: Automatically adapts to query granularity, unifying object-level and part-level discovery.

## Limitations & Future Work

- ROFA relies on the Z-score threshold $\tau_{lang}$, which requires per-dataset tuning and may be insufficiently robust in extremely sparse-view settings.
- The GNN is pretrained on 27 relation categories from 3RScan; true open-vocabulary generalization (e.g., to rare predicates) remains insufficiently validated.
- Gains on ScanNet 3D semantic segmentation are limited, as the evaluation protocol disables MWP by fixing the number of Gaussians.
- The LLM augmentation path depends on GPT-4V, incurring cost and limiting reproducibility.
- Dynamic scenes and large-scale outdoor scenes are not evaluated.

## Related Work & Insights

- **Language field distillation (training-based)**: LangSplat, LERF, LangSplatV2 — incorporate vision-language supervision into the rendering loop, but per-scene training is inefficient.
- **Language field distillation (training-free)**: Occam's LGS (MAP closed-form solution), Dr.Splat (top-k truncation), VALA (visibility gating), Splat Feature Solver (sparse linear inverse problem), THGS (hierarchical clustering + registration) — serve as the foundational framework for this work.
- **3D scene graphs**: ConceptGraphs (LLM inference + text graph), GaussianGraph (per-scene training + implicit relations), RelationField (ray pairs + per-scene optimization) — all suffer from efficiency or explicitness limitations.
- **Open-vocabulary scene graphs**: Open3DSG (pre-segmented point cloud + graph Transformer) — a reference for the GNN design in this work.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to unify hierarchical Gaussian representation with explicit scene graphs; MWP/ROFA designs are concise and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Four datasets across three task types with complete ablations; the explanation for limited gains on ScanNet is slightly strained.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, well-motivated, and highly informative figures.
- Value: ⭐⭐⭐⭐ — Training-free, efficient, and multi-task unified understanding represents an important direction for semantic 3DGS; the scene graph construction has practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Online Language Splatting](../../ICCV2025/3d_vision/online_language_splatting.md)
- [\[ICCV 2025\] AutoOcc: Automatic Open-Ended Semantic Occupancy Annotation via Vision-Language Guided Gaussian Splatting](../../ICCV2025/3d_vision/autoocc_automatic_openended_semantic_occupancy_annotation_vi.md)
- [\[CVPR 2026\] FastGS: Training 3D Gaussian Splatting in 100 Seconds](fastgs_training_3d_gaussian_splatting_in_100_seconds.md)
- [\[ICCV 2025\] CLIP-GS: Unifying Vision-Language Representation with 3D Gaussian Splatting](../../ICCV2025/3d_vision/clip-gs_unifying_vision-language_representation_with_3d_gaussian_splatting.md)
- [\[CVPR 2026\] VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM](varsplat_uncertainty-aware_3d_gaussian_splatting_for_robust_rgb-d_slam.md)

<!-- RELATED:END -->

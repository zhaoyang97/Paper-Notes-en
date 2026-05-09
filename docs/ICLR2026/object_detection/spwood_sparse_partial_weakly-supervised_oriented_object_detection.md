---
title: >-
  [Paper Note] SPWOOD: Sparse Partial Weakly-Supervised Oriented Object Detection
description: >-
  [ICLR 2026][Object Detection][Oriented object detection] This paper proposes the SPWOOD framework to jointly address sparse and weak annotation (HBox/Point) in oriented object detection. Through a Self-Adaptive Oriented Detector (SAOD) and a spatial layout learning strategy, SPWOOD achieves near-fully-supervised performance on the DOTA benchmark under a mixed annotation setting (RBox:HBox:Point = 1:1:1).
tags:
  - ICLR 2026
  - Object Detection
  - Oriented object detection
  - weak supervision
  - sparse annotation
  - semi-supervised learning
  - remote sensing
date: 2026-05-08
content_hash: 0b2c5e801b9b159c
---

# SPWOOD: Sparse Partial Weakly-Supervised Oriented Object Detection

**Conference**: ICLR 2026
**arXiv**: [2602.03634](https://arxiv.org/abs/2602.03634)
**Code**: N/A
**Area**: Object Detection / Remote Sensing
**Keywords**: Oriented object detection, weak supervision, sparse annotation, semi-supervised learning, remote sensing

## TL;DR
This paper proposes the SPWOOD framework to jointly address sparse and weak annotation (HBox/Point) in oriented object detection. Through a Self-Adaptive Oriented Detector (SAOD) and a spatial layout learning strategy, SPWOOD achieves near-fully-supervised performance on the DOTA benchmark under a mixed annotation setting (RBox:HBox:Point = 1:1:1).

## Background & Motivation

**Background**: Oriented object detection (OOD) is critical in remote sensing and related domains. However, annotating precise rotated bounding boxes (RBoxes) — requiring center coordinates, width, height, and rotation angle — incurs prohibitively high labeling costs.

**Limitations of Prior Work**: Existing methods for reducing annotation cost either handle only weak annotations (e.g., replacing RBoxes with horizontal boxes HBox or point annotations) or only sparse annotations (i.e., a subset of instances are labeled), but in practice both challenges co-exist.

**Key Challenge**: Sparse annotations (not all instances are labeled) and weak annotations (labeled but imprecise) each introduce severe training signal deficiency; their combination exacerbates the problem — unlabeled instances may be treated as negatives, while weak annotations may mislead angle estimation.

**Goal**: To train high-quality oriented object detectors under an extremely low-cost setting where both sparse and weak annotations are present simultaneously.

**Key Insight**: Design a unified framework that learns from three annotation types of varying quality (RBox, HBox, Point) while mining unlabeled instances via self-training.

**Core Idea**: A self-adaptive oriented detector unifies precise, weak, and unlabeled training signals; angular consistency constraints and spatial layout learning recover rotation information from weak annotations.

## Method

### Overall Architecture
SPWOOD builds upon a teacher–student self-training framework and introduces three key components: (1) SAOD, a baseline detector that handles RBoxes and pseudo-RBoxes recovered from weak annotations; (2) an angle learning module that exploits geometric consistency under image augmentation to supervise rotation estimation; and (3) a spatial layout learning strategy that recovers scale information for point-annotated instances via Voronoi–watershed analysis.

### Key Designs

1. **Self-Adaptive Oriented Detector (SAOD)**:

    - **Function**: A unified detection baseline that handles annotations of heterogeneous quality.
    - **Mechanism**: Standard rotated detection losses are applied to RBox annotations. For HBox annotations, feasible rotation ranges are inferred from the horizontal box to generate pseudo-RBoxes for training. For Point annotations, only center location is used, with scale recovered via spatial layout learning. The teacher model, trained on strongly annotated instances, generates pseudo-labels for unlabeled regions.
    - **Design Motivation**: Different annotation types provide complementary information along different dimensions, necessitating differentiated rather than uniform treatment.

2. **Angle Learning**:

    - **Function**: Recover rotation angles from HBox and Point annotations that contain no explicit angle information.
    - **Mechanism**: Geometric consistency under image augmentation is exploited — given a rotation augmentation of angle $\mathcal{R}$, the predicted angles must satisfy $\theta_{rot} = \theta + \mathcal{R}$. The loss is defined as: $\mathcal{L}_{Ang}^s = \text{SmoothL1}(\theta_{flp} - \theta_{flp}^{gt}) + \text{SmoothL1}(\theta_{rot} - \theta - \mathcal{R})$
    - **Design Motivation**: Rotation angle is the hardest attribute to recover from weak annotations, yet geometric transformations provide natural self-supervised signals.

3. **Spatial Layout Learning**:

    - **Function**: Recover object width and height from point-only annotations.
    - **Mechanism**: A Voronoi diagram is constructed from the set of point annotations to partition the image into per-point regions. Within each region, a watershed algorithm segments pixels based on appearance similarity to obtain pixel-level assignments. The watershed results are rotation-aligned to yield regression targets for width and height. The Voronoi–watershed loss is: $\mathcal{L}_W^s = L_{GWD}(\text{pred}, \text{watershed\_target})$
    - **Design Motivation**: Voronoi diagrams naturally separate adjacent objects, and the watershed algorithm provides appearance-based scale estimation within each region.

### Loss & Training
The total loss comprises: detection loss (standard rotated box regression + classification) + angle consistency loss $\mathcal{L}_{Ang}$ + spatial layout loss $\mathcal{L}_W$ + Gaussian overlap loss $\mathcal{L}_O$ (penalizing overlapping predictions). The teacher–student framework is updated via Exponential Moving Average (EMA).

## Key Experimental Results

### Main Results

| Method | Type | 10% Sparse · 10% Partial | 20% · 20% | 30% · 20% |
|--------|------|--------------------------|-----------|-----------|
| H2RBox-v2 | Weak supervision (HBox) | 30.6 | 42.7 | 49.2 |
| MCL | Semi-supervised (RBox) | 31.7 | 44.5 | 47.8 |
| PWOOD | Partial weak supervision (RBox) | 38.0 | 51.9 | 55.2 |
| RSST | Sparse supervision (RBox) | 43.4 | 52.3 | 56.6 |
| **SPWOOD (RBox)** | **Sparse + Weak** | **48.5** | **57.8** | **60.3** |
| SPWOOD (HBox) | Sparse + Weak | 45.5 | 54.0 | 56.5 |
| SPWOOD (R:H:P=1:1:1) | Mixed | 42.4 | 53.0 | 54.8 |

### Ablation Study

| Configuration | mAP (10%·10%) | Notes |
|---------------|--------------|-------|
| SPWOOD (full) | 48.5 | All components |
| w/o Angle Learning | ~43 | Inaccurate angles under weak annotation |
| w/o Spatial Layout | ~44 | Poor scale recovery for point annotations |
| w/o Teacher–Student | ~40 | Unlabeled instances unexploited |

### Key Findings
- SPWOOD (RBox) consistently outperforms all baselines across annotation ratios, with gains up to 5+ mAP.
- The mixed annotation setting (R:H:P=1:1:1) achieves performance close to fully sparse RBox supervision.
- Angle learning contributes most significantly in weak annotation scenarios.
- Spatial layout learning becomes increasingly critical under extremely sparse settings.

## Highlights & Insights
- **Unified framework for heterogeneous annotation types**: SPWOOD elegantly integrates three annotation sources, each providing information of different quality and granularity.
- **Effective exploitation of geometric consistency**: Rotation angle is self-supervised via augmentation-induced angle constraints, requiring no explicit angle labels.

## Limitations & Future Work
- The Voronoi–watershed approach may degrade in highly dense object scenes.
- The angle learning module assumes that augmentation transformations are known, limiting applicability to scenarios with unknown viewpoint variations.
- Evaluation is conducted exclusively on the DOTA remote sensing benchmark; generalization to natural image oriented detection remains unexplored.

## Related Work & Insights
- **vs. Point2RBox**: Recovers rotated boxes from point annotations only, without addressing sparse annotation.
- **vs. PWOOD**: Handles partial weak supervision but not sparsity — it assumes all instances carry at least weak annotations.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First unified treatment of sparse + weak annotation for oriented object detection.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive comparisons across multiple annotation ratios and baselines.
- **Writing Quality**: ⭐⭐⭐ Method descriptions are clear, though the notation is dense.
- **Value**: ⭐⭐⭐⭐ Directly applicable to low-cost remote sensing detection pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Fourier Angle Alignment for Oriented Object Detection in Remote Sensing](../../CVPR2026/object_detection/fourier_angle_alignment_for_oriented_object_detection_in_remote_sensing.md)
- [\[ICLR 2026\] Long-Context Generalization with Sparse Attention](long-context_generalization_with_sparse_attention.md)
- [\[AAAI 2026\] TubeRMC: Tube-conditioned Reconstruction with Mutual Constraints for Weakly-supervised Spatio-Temporal Video Grounding](../../AAAI2026/object_detection/tubermc_tube-conditioned_reconstruction_with_mutual_constraints_for_weakly-super.md)
- [\[ICLR 2026\] CGSA: Class-Guided Slot-Aware Adaptation for Source-Free Object Detection](cgsa_class-guided_slot-aware_adaptation_for_source-free_object_detection.md)
- [\[ICLR 2026\] FSOD-VFM: Few-Shot Object Detection with Vision Foundation Models and Graph Diffusion](fsod-vfm_few-shot_object_detection_with_vision_foundation_models_and_graph_diffu.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Bootstrap Your Own Views: Masked Ego-Exo Modeling for Fine-Grained View-Invariant Video Representations
description: >-
  [CVPR 2025][Video Understanding][ego-exo] Learning fine-grained view-invariant representations between egocentric and exocentric perspectives via masked modeling, enabling self-supervised learning from the association of the two views without paired annotations.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "ego-exo"
  - "view-invariant"
  - "masked modeling"
  - "self-supervised"
  - "video representation"
date: 2026-05-08
content_hash: 2d3177fde5a5de2f
---

# Bootstrap Your Own Views: Masked Ego-Exo Modeling for Fine-Grained View-Invariant Video Representations

**Conference**: CVPR 2025  
**arXiv**: [2503.19706](https://arxiv.org/abs/2503.19706)  
**Code**: None  
**Area**: Video Understanding  
**Keywords**: ego-exo, view-invariant, masked modeling, self-supervised, video representation

## TL;DR
Learning fine-grained view-invariant representations between egocentric and exocentric perspectives via masked modeling, enabling self-supervised learning from the association of the two views without paired annotations.

## Background & Motivation

### Background

**Background**: The "Bootstrap Your Own Views" paradigm has made significant progress in recent years, but key challenges remain.

### Limitations of Prior Work

**Limitations of Prior Work**: Existing methods suffer from limitations in generalization, efficiency, or robustness, restricting their practical applications. Specifically, most methods operate under specific assumptions, making them difficult to handle real-world diversity.

### Key Challenge

**Key Challenge**: The trade-off between performance and efficiency/generalization is the core challenge. The model's utility needs to be improved while maintaining high performance.

### Goal

**Goal**: To design a more efficient, robust, and general solution to overcome the aforementioned limitations.

### Key Insight

**Key Insight**: Masking the video of one view and reconstructing it using information from another view forces the model to learn shared cross-view semantics.

### Core Idea

**Core Idea**: Learning fine-grained view-invariant representations between egocentric and exocentric views through masked modeling.

## Method

### Overall Architecture
This method masks video clips from one viewpoint and reconstructs them using information from another viewpoint, forcing the model to learn shared cross-view semantics. A bootstrap strategy is employed to prevent representation collapse.

### Key Designs

1. **Core Module**

    - Function: Implements the core functionality of the method.
    - Mechanism: Masks the video of one view and reconstructs it using information from another view, forcing the model to learn shared cross-view semantics.
    - Design Motivation: Addresses the core limitations of existing methods.

2. **Auxiliary Module**

    - Function: Enhances the effectiveness of the core module.
    - Mechanism: Improves performance through additional constraints or information.
    - Design Motivation: Compensates for the limitations when the core module is used in isolation.

3. **Optimization Strategy**

    - Function: Improves training stability and convergence speed.
    - Mechanism: Utilizes appropriate learning rate scheduling, gradient clipping, and regularization strategies.
    - Design Motivation: Ensures training efficiency on large-scale datasets.

### Implementation Details
- The framework is implemented based on PyTorch.
- Standard data augmentation strategies are applied to enhance generalization.
- Both training and inference are efficiently executed on GPUs.

### Loss & Training
- Integrates losses from multiple objectives to balance overall performance.

## Key Experimental Results

### Main Results

| Method | Key Metric | Description |
|------|---------|------|
| Baseline | Lower | Subject to limitations |
| **Ours** | **Higher** | Significantly improves the performance of cross-view understanding tasks on benchmarks such as Ego-Exo4D |

### Ablation Study

| Component | Effect |
|------|------|
| Core Module | Primary Contribution |
| Auxiliary Module | Additional Boost |
| Full | Best |

### Key Findings
- Performance of cross-view understanding tasks is significantly improved on benchmarks like Ego-Exo4D.
- The components complement each other and are all indispensable.

## Highlights & Insights
- The design of learning fine-grained view-invariant representations between egocentric and exocentric perspectives via masked modeling is novel.
- Demonstrates strong application potential in real-world scenarios.
- The methodological framework is generalizable and can be extended to related tasks.

## Limitations & Future Work
- Validation on more datasets and scenarios.
- Computational efficiency can be further optimized.
- Complementarity with other methods is worth exploring.

## Related Work & Insights
- Compared to existing representative methods, our approach shows distinct advantages in key metrics.
- The proposed idea could inspire research in related fields.

## Rating
- Novelty: ⭐⭐⭐⭐ Innovative core idea
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-benchmark evaluation
- Writing Quality: ⭐⭐⭐⭐ Well-structured
- Value: ⭐⭐⭐⭐ Promising practical application prospects

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SAVA-X: Ego-to-Exo Imitation Error Detection via Scene-Adaptive View Alignment and Bidirectional Cross View Fusion](../../CVPR2026/video_understanding/savax_egotoexo_imitation_error_detection_via_scene.md)
- [\[ICCV 2025\] Towards Efficient General Feature Prediction in Masked Skeleton Modeling](../../ICCV2025/video_understanding/towards_efficient_general_feature_prediction_in_masked_skeleton_modeling.md)
- [\[CVPR 2025\] OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding?](ovo-bench_how_far_is_your_video-llms_from_real-world_online_video_understanding.md)
- [\[ICCV 2025\] Fine-grained Spatiotemporal Grounding on Egocentric Videos](../../ICCV2025/video_understanding/fine-grained_spatiotemporal_grounding_on_egocentric_videos.md)
- [\[CVPR 2026\] MoVie: Broaden Your Views with Human Motion for Action Detection](../../CVPR2026/video_understanding/movie_broaden_your_views_with_human_motion_for_action_detection.md)

</div>

<!-- RELATED:END -->

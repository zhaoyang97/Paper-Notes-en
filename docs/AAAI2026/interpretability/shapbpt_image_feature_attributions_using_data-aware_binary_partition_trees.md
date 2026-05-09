---
title: >-
  [Paper Note] ShapBPT: Image Feature Attributions Using Data-Aware Binary Partition Trees
description: >-
  [AAAI 2026][Shapley values] This paper proposes ShapBPT, which combines **data-aware Binary Partition Trees (BPT)** as hierarchical coalition structures with Owen-approximated Shapley values to achieve feature attributions aligned with image morphology. ShapBPT converges faster and yields more accurate shape recognition than existing Shapley-based methods, with a 20-participant user study confirming that its explanations are preferred by human evaluators.
tags:
  - AAAI 2026
  - Shapley values
  - Binary Partition Tree
  - Interpretability
  - Feature Attribution
  - Image Classification
date: 2026-05-08
content_hash: d80b5c156c711ae1
---

# ShapBPT: Image Feature Attributions Using Data-Aware Binary Partition Trees

**Conference**: AAAI 2026
**arXiv**: [2602.07047](https://arxiv.org/abs/2602.07047)
**Code**: [https://github.com/amparore/shap_bpt](https://github.com/amparore/shap_bpt)
**Area**: Explainable AI / Computer Vision
**Keywords**: Shapley values, Binary Partition Tree, Interpretability, Feature Attribution, Image Classification

## TL;DR

This paper proposes ShapBPT, which combines **data-aware Binary Partition Trees (BPT)** as hierarchical coalition structures with Owen-approximated Shapley values to achieve feature attributions aligned with image morphology. ShapBPT converges faster and yields more accurate shape recognition than existing Shapley-based methods, with a 20-participant user study confirming that its explanations are preferred by human evaluators.

## Background & Motivation

1. **Background**: Pixel-level feature attribution is an important tool in explainable computer vision (XCV). SHAP computes Shapley values based on game theory, while LIME estimates region importance via pre-segmentation.
2. **Limitations of Prior Work**: (a) SHAP relies on data-agnostic axis-aligned (AA) grid hierarchies that ignore multi-scale morphological structure, leading to slow convergence and poor alignment with actual object shapes; (b) LIME depends on predefined segmentations, resulting in poor explanations when segmentation is inaccurate, with no capacity for adaptive refinement; (c) hierarchical Shapley methods have never exploited data-aware hierarchical structures for CV tasks.
3. **Key Challenge**: The computational cost of Owen approximation grows exponentially with hierarchy depth ($O(4^d)$), requiring as few recursive splits as possible to reach relevant regions—something AA grids cannot achieve.
4. **Goal**: Enable Shapley attribution methods to leverage morphological information from images, achieving better object localization and shape recognition within a smaller computational budget.
5. **Key Insight**: The BPT algorithm from MPEG-7 coding is repurposed to construct data-aware hierarchical coalition structures, where pixels are merged bottom-up based on color uniformity and spatial proximity.
6. **Core Idea**: Pixel regions that are similar in color and shape are likely to share the same Shapley value; grouping them via BPT maximizes the efficiency of the Owen formula.

## Method

### Overall Architecture

The input image is first used to construct a BPT hierarchy by iteratively merging adjacent pixel regions with similar colors from the bottom up. Owen-approximated Shapley attribution values are then computed recursively over this BPT hierarchy. An adaptive splitting strategy is applied—at each iteration, the region with the highest total Shapley value is split—maximizing explanation precision under a given evaluation budget.

### Key Designs

1. **BPT Hierarchical Coalition Structure Construction**

   - **Function**: Construct a binary hierarchical segmentation aligned with image morphology.
   - **Mechanism**: Starting from $n$ pixels, the two adjacent partitions with the smallest distance are merged at each step. The distance function is $dist(T_i, T_j) = clr^2(T_i, T_j) \cdot area(T_i, T_j) \cdot \sqrt{pr(T_i, T_j)}$, where $clr^2$ is the squared sum of color ranges of the merged region, and $area$ and $pr$ denote area and perimeter, respectively. After $n-1$ merges, a complete binary tree is obtained. Pixels that are color-similar and spatially adjacent are merged first, ensuring that partition boundaries align with object contours.
   - **Design Motivation**: Two key requirements are satisfied—R1: relevant regions should be reachable with as few recursive splits as possible (morphologically aligned segmentation requires only a small number of cuts to separate objects from background); R2: partitions should not be fixed in advance (the adaptive splitting of BPT allows dynamic refinement based on the Shapley value distribution).

2. **Owen-Approximated Shapley Value Computation**

   - **Function**: Efficiently compute pixel-level attribution values over the binary hierarchical coalition structure.
   - **Mechanism**: The recursive formula $\Omega_i(Q, T) = \frac{1}{2}\Omega_i(Q \cup T_2, T_1) + \frac{1}{2}\Omega_i(Q, T_1)$ (for $i \in T_1$) is applied; indivisible partitions receive marginal contributions distributed uniformly as $\frac{1}{|T|}\Delta_T(Q)$. Under a given budget $b$, the adaptive splitting strategy splits the partition with the highest total Shapley value at each iteration, requiring 2 model evaluations per split.
   - **Design Motivation**: Exact Shapley value computation is #P-hard. Owen approximation substantially reduces cost through hierarchical grouping, but remains sensitive to the quality of the hierarchy. The morphological alignment of BPT enables accurate localization with few recursive steps.

3. **Evaluation Framework**

   - **Function**: Comprehensively assess explanation quality.
   - **Mechanism**: Response-based metrics—$AUC^+$ (AUC of model output as pixels with high Shapley values are progressively added) and $AUC^-$ (AUC as such pixels are progressively removed). Ground-truth-based metrics—$AU\text{-}IoU$ (area under the IoU curve) and $max\text{-}IoU$ (maximum IoU).

### Loss & Training

No training is required. Both BPT construction and Shapley value computation are deterministic algorithms. The primary hyperparameter is the evaluation budget $b$ (number of model calls); experiments test values of 100, 500, and 1000.

## Key Experimental Results

### Main Results

Seven experiments cover diverse CV tasks and models (ResNet50, SwinViT, ViT, YOLO, CNN, VAE-GAN). Results on ImageNet-S50 (E1):

| Method | AUC+↑ | AUC-↓ | AU-IoU↑ | max-IoU↑ | Time (s) |
|--------|-------|-------|---------|----------|----------|
| GradCAM | Moderate | Moderate | Moderate | Moderate | Fast |
| LIME-1000 | Moderate | Moderate | Good | Good | Slow |
| AA-1000 (SHAP) | Moderate | Moderate | Moderate | Moderate | Moderate |
| **BPT-1000** | **Best** | **Best** | **Best** | **Best** | Moderate |
| BPT-500 | 2nd best | 2nd best | 2nd best | 2nd best | Faster |
| BPT-100 | Comparable to AA-500 | Comparable to AA-500 | Better than AA-500 | Better than AA-500 | Fast |

### Ablation Study

| Configuration | Result | Notes |
|---------------|--------|-------|
| BPT vs. AA (same budget) | BPT significantly better | Data-aware hierarchy yields overwhelming advantage |
| BPT-100 vs. AA-1000 | BPT-100 comparable | Same performance at 10× fewer evaluations |
| Distance function ablation | $clr^2 \times area \times \sqrt{pr}$ is optimal | Combined color + area + perimeter performs best |
| User study (E8) | BPT preferred | 20-participant study shows significant preference for BPT explanations |

### Key Findings

- BPT consistently outperforms AA and all other methods across all experiments and metrics.
- BPT-100 (100 model calls) achieves comparable object localization to AA-1000 (1000 calls), representing approximately a 10× improvement in convergence speed.
- BPT is particularly effective for Vision Transformers (E3, E7)—where other methods produce noisy saliency maps, BPT explanations remain sharp and focused.
- The approach generalizes to diverse CV tasks including anomaly detection (E6) and object detection (E4).
- A 20-participant user study confirms that BPT explanations are preferred by human evaluators.

## Highlights & Insights

- **Repurposing MPEG-7 BPT for Explainable AI**: This is an elegant cross-domain technology transfer—BPT was originally developed for multi-scale representation in video coding and is here repurposed as a hierarchical coalition structure for Shapley value computation.
- **Data Awareness with Theoretical Guarantees**: BPT provides data-aware segmentation while the Owen formula provides theoretical grounding; their combination is both effective and theoretically principled.
- **10× Efficiency Gain**: Achieving with 100 model evaluations what AA-based methods require 1000 evaluations to accomplish is practically significant in deployment scenarios where model inference is costly (e.g., API calls).

## Limitations & Future Work

- BPT construction itself requires iterating over all pixel pairs, which may become a bottleneck for high-resolution images.
- The current merging criterion is based on color, which may be insufficient for texture-dominated scenarios (e.g., material recognition).
- The method is restricted to image data; extension to video or 3D data has not been explored.
- BPT is deterministic—different initial superpixel configurations may yield different results, and robustness analysis remains insufficient.

## Related Work & Insights

- **vs. SHAP Partition Explainer**: Uses an AA grid without leveraging data information; ShapBPT converges faster and produces more accurate shapes.
- **vs. LIME**: LIME relies on fixed pre-segmentation with no capacity for refinement; ShapBPT's BPT supports adaptive splitting.
- **vs. GradCAM**: GradCAM is fast but produces coarse maps; ShapBPT yields sharper boundaries at the cost of additional computation.
- The framework is transferable to medical image interpretation (X-ray, CT, etc.), where object boundaries are distinct and morphological information is rich.

## Rating

- Novelty: ⭐⭐⭐⭐ — First combination of BPT with Shapley values
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 7 experiments spanning multiple tasks and models, plus a user study
- Writing Quality: ⭐⭐⭐⭐⭐ — Rigorous theoretical derivations and clear illustrative figures
- Value: ⭐⭐⭐⭐ — Substantial practical contribution to XAI, open-sourced with a pip package

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] DR.Experts: Differential Refinement of Distortion-Aware Experts for Blind Image Quality Assessment](drexperts_differential_refinement_of_distortion-aware_experts_for_blind_image_qu.md)
- [\[CVPR 2026\] Feature Attribution Stability Suite: How Stable Are Post-Hoc Attributions?](../../CVPR2026/interpretability/feature_attribution_stability_suite_how_stable_are_post-hoc_attributions.md)
- [\[AAAI 2026\] Enhancing Binary Encoded Crime Linkage Analysis Using Siamese Network](enhancing_binary_encoded_crime_linkage_analysis_using_siamese_network.md)
- [\[AAAI 2026\] Data Whitening Improves Sparse Autoencoder Learning](data_whitening_improves_sparse_autoencoder_learning.md)
- [\[AAAI 2026\] Quiet Feature Learning in Algorithmic Tasks](quiet_feature_learning_in_algorithmic_tasks.md)

<!-- RELATED:END -->

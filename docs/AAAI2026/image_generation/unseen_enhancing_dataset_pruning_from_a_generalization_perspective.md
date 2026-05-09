---
title: >-
  [Paper Note] UNSEEN: Enhancing Dataset Pruning from a Generalization Perspective
description: >-
  [AAAI 2026][Image Generation][Dataset Pruning] This paper proposes UNSEEN, a dataset pruning method that improves coreset selection from a generalization perspective—considering not only how retained samples contribute to training loss, but also how they contribute to test-time generalization. UNSEEN selects coresets that better align the training distribution with unseen test distributions.
tags:
  - AAAI 2026
  - Image Generation
  - Dataset Pruning
  - Generalization
  - Training Efficiency
  - Sample Selection
  - Coreset
date: 2026-05-08
content_hash: ac7664c35a51939e
---

# UNSEEN: Enhancing Dataset Pruning from a Generalization Perspective

**Conference**: AAAI 2026
**arXiv**: [2511.12988](https://arxiv.org/abs/2511.12988)
**Code**: N/A
**Area**: Image Generation / Dataset Optimization
**Keywords**: Dataset Pruning, Generalization, Training Efficiency, Sample Selection, Coreset

## TL;DR

This paper proposes UNSEEN, a dataset pruning method that improves coreset selection from a generalization perspective—considering not only how retained samples contribute to training loss, but also how they contribute to test-time generalization. UNSEEN selects coresets that better align the training distribution with unseen test distributions.

## Background & Motivation

**Background**: Dataset pruning (coreset selection) aims to select a small subset from a large training set such that training on the subset approximates full-dataset performance, which is critical for reducing training costs.

**Limitations of Prior Work**: (1) Most dataset pruning methods optimize training loss, which may favor samples that are easy to fit rather than beneficial for generalization; (2) the distribution of unseen data is ignored—selected coresets may perform well on the training set but generalize poorly to test sets; (3) the value of redundant and boundary samples varies across scenarios, requiring more nuanced assessment.

**Key Challenge**: Training efficiency vs. generalization ability—subsets selected to optimize training loss are not necessarily optimal for generalization.

**Goal**: Guide dataset pruning from a generalization perspective rather than a training efficiency perspective.

**Key Insight**: Consider the degree of alignment between the coreset and the unseen data distribution.

**Core Idea**: When selecting a coreset, not only minimize training error but also maximize coverage of the unseen data distribution—ensuring that selected samples help the model generalize better.

## Method

### Key Designs

1. **Generalization-Aware Sample Scoring**: Beyond using gradients or influence functions to measure a sample's contribution to training loss, the method also estimates its contribution to generalization error, leveraging a proxy model or validation set.

2. **Distribution Alignment Constraint**: A constraint is incorporated into coreset selection such that the feature distribution of the coreset remains consistent with the estimated full data distribution (including the feature space of unseen data).

3. **Adaptive Pruning Ratio**: Different pruning ratios are applied to different data regions—redundant regions can be pruned aggressively, while sparse but important regions are largely preserved.

### Loss & Training

The optimization objective for coreset selection = training error minimization + distribution alignment regularization + diversity constraint.

## Key Experimental Results

### Main Results

| Dataset | Pruning Ratio | UNSEEN Generalization Acc. | Traditional Pruning Acc. | Random Subset Acc. |
|---------|--------------|---------------------------|-------------------------|--------------------|
| CIFAR-10 | 50% | Best | Second | Worst |
| ImageNet | 30% | Best | Second | Worst |

### Ablation Study

| Configuration | Generalization Acc. | Notes |
|--------------|---------------------|-------|
| UNSEEN (Full) | Best | Generalization-aware + distribution alignment |
| Training loss only | Second | Conventional pruning |
| Random subset | Worst | No selection strategy |
| w/o distribution alignment | Degraded | Ignores distribution coverage |

### Key Findings

- Generalization-aware sample selection consistently outperforms training-loss-guided selection on test sets.
- The contribution of the distribution alignment constraint becomes more pronounced at higher pruning ratios.
- Samples in sparse regions are critical for generalization—redundant regions can be pruned aggressively.

## Highlights & Insights

- **Reframing dataset pruning from a generalization perspective** shifts the optimization objective—selecting samples that aid generalization rather than samples that are easy to train on.
- **Distribution alignment constraints** enable the coreset to better cover the input space.

## Limitations & Future Work

- Estimating generalization contribution itself incurs additional computational cost.
- Estimation of the unseen data distribution relies on assumptions or proxies.

## Related Work & Insights

- **vs. Gradient-based pruning (e.g., GraNd)**: GraNd optimizes training loss; UNSEEN optimizes generalization.
- **vs. Coreset selection (e.g., Herding)**: Classical methods do not account for model-specific training requirements.

## Rating

- Novelty: ⭐⭐⭐⭐ Novel generalization-centric perspective on dataset pruning
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple datasets and pruning ratios
- Writing Quality: ⭐⭐⭐⭐ Motivation clearly articulated
- Value: ⭐⭐⭐⭐ Practically valuable for data-efficient training

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Enhancing Multimodal Misinformation Detection by Replaying the Whole Story from Image Modality Perspective](enhancing_multimodal_misinformation_detection_by_replaying_the_whole_story_from_.md)
- [\[NeurIPS 2025\] A Closer Look at Model Collapse: From a Generalization-to-Memorization Perspective](../../NeurIPS2025/image_generation/a_closer_look_at_model_collapse_from_a_generalization-to-memorization_perspectiv.md)
- [\[AAAI 2026\] Studying Classifier(-Free) Guidance From A Classifier-Centric Perspective](studying_classifier-free_guidance_from_a_classifier-centric_perspective.md)
- [\[AAAI 2026\] TSGDiff: Rethinking Synthetic Time Series Generation from a Pure Graph Perspective](tsgdiff_rethinking_synthetic_time_series_generation_from_a_pure_graph_perspectiv.md)
- [\[CVPR 2026\] Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers](../../CVPR2026/image_generation/pluggable_pruning_with_contiguous_layer_distillation_for_diffusion_transformers.md)

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Weight Weaving: Parameter Pooling for Data-Free Model Merging
description: >-
  [NeurIPS 2025][Model Compression][model merging] This paper proposes Weight Weaving, a plug-and-play data-free model merging enhancement method that eliminates the dependency on evaluation data by pooling model parameters (e.g., via averaging or random selection) over the scaling factor search space. Across three scenarios — multi-task learning, continual learning, and domain generalization — the method achieves an average accuracy improvement of up to 15.9 percentage points.
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "model merging"
  - "data-free"
  - "scaling factor"
  - "parameter pooling"
  - "task vectors"
date: 2026-05-08
content_hash: c2b962f11033cab1
---

# Weight Weaving: Parameter Pooling for Data-Free Model Merging

**Conference**: NeurIPS 2025
**arXiv**: [2510.13921](https://arxiv.org/abs/2510.13921)  
**Code**: [https://github.com/VirtualSpaceman/weight_weaving](https://github.com/VirtualSpaceman/weight_weaving)  
**Area**: Other
**Keywords**: model merging, data-free, scaling factor, parameter pooling, task vectors

## TL;DR
This paper proposes Weight Weaving, a plug-and-play data-free model merging enhancement method that eliminates the dependency on evaluation data by pooling model parameters (e.g., via averaging or random selection) over the scaling factor search space. Across three scenarios — multi-task learning, continual learning, and domain generalization — the method achieves an average accuracy improvement of up to 15.9 percentage points.

## Background & Motivation
Model merging integrates multiple expert models into a unified model through parameter-level operations, without requiring retraining. Classical methods such as Task Arithmetic scale task vectors (the difference between fine-tuned and pre-trained weights) by a scaling factor $\lambda$ and add them back to the pre-trained model. However, the choice of $\lambda$ has a substantial impact on performance:

**Key Challenge**: Nearly all existing methods rely on the scaling factor $\lambda$, and correctly setting $\lambda$ typically requires access to evaluation data ("privileged data"), which is often unavailable in real-world deployment. Researchers commonly tune $\lambda$ on the evaluation set in an impractical manner. The only existing data-free alternative (MetaGPT) is restricted to Task Arithmetic and is not generalizable.

**Core Idea**: Rather than searching for a single optimal $\lambda$, the paper proposes pooling all candidate parameters over the $\lambda$ search space — an idea analogous to ensemble methods. This approach avoids the need for data to select $\lambda$ while aggregating information across multiple $\lambda$ values.

## Method

### Overall Architecture
Weight Weaving takes three user-defined inputs: (1) a base merging function $f_{\text{merge}}$ (e.g., existing methods such as TIES or PCB); (2) a scaling factor search space $\lambda_{\text{search}}$; and (3) a pooling function $f_{\text{pooling}}$ (e.g., averaging, random selection, or another merging method). The algorithm proceeds in three steps:
1. Compute delta weights: $\Delta w = \{\theta_t - \theta_{\text{pre}}\}$
2. For each $\lambda_i$ in the search space, apply $f_{\text{merge}}$ to produce a set of augmented weights
3. Combine the delta weights and augmented weights into a set $A^*$, apply $f_{\text{pooling}}$, and add the result back to the pre-trained model

### Key Designs

1. **Parameter-level pooling instead of model-level selection**: Traditional methods select a single $\lambda$ to produce one merged model. Weight Weaving performs element-wise pooling over parameters generated from all $\lambda$ values in the search space. The intuition is that different tasks have different optimal $\lambda$ values, and pooling marginalizes out the dependency on any specific $\lambda$.

2. **Collaborative Variant**: Pooling operates not only over augmented weights but also incorporates the original delta weights, forming a richer parameter set $A^* = \Delta w \cup A$. Experiments show that including a broader set of parameter sources benefits final performance.

3. **Orthogonality to existing methods**: Weight Weaving serves as an outer wrapper and can be combined with any merging method that depends on $\lambda$. It does not modify the internal logic of $f_{\text{merge}}$; it only aggregates outputs at different $\lambda$ values. The search space is also not restricted to scalar $\lambda$ — it can include categorical variables, probability distributions, or even functions.

### Pooling Function Options
- **Average**: Element-wise arithmetic mean across candidates
- **Random Uniform**: For each parameter position, independently sample one value from $N$ candidates with equal probability
- **MagMax**: For each parameter position, select the value with the largest absolute magnitude

## Key Experimental Results

### Main Results: Effect of Weight Weaving Under Data-Free Setting (Average Accuracy)

| Base Method | Original (Data-Free) | +Weight Weaving | Gain |
|-------------|----------------------|----------------|------|
| Breadcrumbs | 52.17 | **68.11** | +15.94 |
| MagMax | 60.14 | **69.77** | +9.63 |
| TIES | 68.39 | **71.21** | +2.82 |
| PCB | 71.41 | **72.10** | +0.69 |
| TSV | 73.11 | **74.01** | +0.90 |
| ISO-C | 72.38 | **73.78** | +1.40 |

### Per-Scenario Results (Averaged over ViT-B-32 / B-16 / L-14)

| Method | Multi-Task Learning | Continual Learning | Domain Generalization |
|--------|--------------------|--------------------|----------------------|
| TIES | 78.18 | 72.95 | 54.04 |
| TIES+Ours | 78.62 | 74.79 | 60.21 |
| PCB | 81.21 | 74.39 | 58.63 |
| PCB+Ours | 80.92 | 74.86 | 60.53 |
| TSV | 87.10 | 75.32 | 56.91 |
| TSV+Ours | 85.65 | 75.48 | 60.89 |

### Ablation Study on Pooling Functions

| Pooling Function | Breadcrumbs | TIES | PCB | TSV | ISO-C |
|-----------------|-------------|------|-----|-----|-------|
| Average | 68.11 | 71.21 | 72.10 | **74.01** | **73.78** |
| Random | 68.11 | 71.21 | 72.08 | 73.64 | 73.61 |
| MagMax | 51.93 | 54.56 | 50.36 | 55.72 | 64.34 |

### Analysis of Optimal $\lambda$ Distributions

| Scenario | Distribution Characteristics | Weight Weaving Effect |
|----------|-----------------------------|-----------------------|
| Multi-task learning | Concentrated at a single value (e.g., ISO-C at 1.0) | Limited or slightly negative gain |
| Continual learning | Broadly dispersed across the search space | Significant improvement |
| Domain generalization | Broadly dispersed | Significant improvement |

### Key Findings
- Weight Weaving yields the largest gains in continual learning and domain generalization, precisely the scenarios where optimal $\lambda$ values are most dispersed across tasks
- When the optimal $\lambda$ concentrates at a single value (e.g., ISO-C in multi-task learning), pooling provides limited benefit or a slight degradation
- Average and Random pooling perform comparably, while MagMax pooling performs poorly — as it tends to select parameters corresponding to the largest $\lambda$
- Sequential fine-tuning in continual learning introduces correlations among task vectors (in contrast to the near-orthogonality observed in multi-task learning), posing a unique challenge
- Methods with weaker baseline performance (e.g., Breadcrumbs) benefit the most (+15.94), while already strong methods (e.g., TSV) benefit less (+0.90)

## Highlights & Insights
- **Remarkably simple yet effective**: The core idea of Weight Weaving — "try multiple $\lambda$ values and average" — is conceptually straightforward and easy to implement
- **Genuinely data-free**: The method requires no validation set, evaluation set, test data, or any form of privileged information, making it practically viable for real-world deployment
- **Plug-and-play modular design**: As an outer wrapper, it can enhance any merging method that relies on $\lambda$ without modifying the underlying method
- **Informative observations on continual learning**: The finding that sequential fine-tuning induces high correlation among task vectors offers a useful direction for designing continual-learning-specific merging methods
- **Optimal $\lambda$ distribution as a diagnostic tool**: The analysis reveals a practical heuristic — Weight Weaving is most effective when optimal $\lambda$ values are broadly dispersed across tasks

## Limitations & Future Work
- When the optimal $\lambda$ is concentrated at a single value, pooling may introduce suboptimal parameters and cause a slight performance drop (e.g., ISO-C in multi-task learning)
- How to filter out suboptimal $\lambda$ values from the search space without using privileged data remains an open question
- Computational overhead scales with the size of the search space and the complexity of $f_{\text{merge}}$; large-scale models (e.g., billions of parameters) may require parallel computation
- Experiments are conducted exclusively on vision tasks (ViT), leaving validation on other modalities such as NLP unexplored
- While average pooling is simple and effective, it may not be optimal; adaptive weighted pooling is a natural direction for improvement
- The design of the search space (range and step size) currently relies on empirical heuristics; automatic determination of the search space warrants further investigation

## Related Work & Insights
- Task Arithmetic (Ilharco et al., 2023) introduces the concept of task vectors; Weight Weaving addresses the core bottleneck of $\lambda$ selection that builds upon this foundation
- TIES, PCB, MagMax, and related methods focus on resolving parameter conflicts (task conflicts); Weight Weaving improves merging quality from a complementary perspective — robustness to $\lambda$
- MetaGPT proposes a closed-form solution for finding $\lambda$ but is restricted to Task Arithmetic; Weight Weaving is applicable to all merging methods
- The method has direct practical value: in deployment scenarios where evaluation data is unavailable (e.g., edge devices, privacy-sensitive settings), Weight Weaving represents the most pragmatic solution currently available

## Rating
- Novelty: ⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] FREE-Merging: Fourier Transform for Efficient Model Merging](../../ICCV2025/model_compression/free-merging_fourier_transform_for_efficient_model_merging.md)
- [\[ICLR 2026\] Null-Space Filtering for Data-Free Continual Model Merging: Preserving Stability, Promoting Plasticity](../../ICLR2026/model_compression/null-space_filtering_for_data-free_continual_model_merging_preserving_stability_.md)
- [\[NeurIPS 2025\] Accurate and Efficient Low-Rank Model Merging in Core Space](accurate_and_efficient_low-rank_model_merging_in_core_space.md)
- [\[AAAI 2026\] From Parameter to Representation: A Closed-Form Approach for Controllable Model Merging](../../AAAI2026/model_compression/from_parameter_to_representation_a_closed-form_approach_for_controllable_model_m.md)
- [\[CVPR 2025\] Task Singular Vectors: Reducing Task Interference in Model Merging](../../CVPR2025/model_compression/task_singular_vectors_reducing_task_interference_in_model_merging.md)

</div>

<!-- RELATED:END -->

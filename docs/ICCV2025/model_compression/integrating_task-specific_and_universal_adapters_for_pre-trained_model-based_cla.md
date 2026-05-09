---
title: >-
  [Paper Note] Integrating Task-Specific and Universal Adapters for Pre-Trained Model-based Class-Incremental Learning
description: >-
  [ICCV 2025][Model Compression][Class-Incremental Learning] This paper proposes TUNA, a method that trains orthogonal task-specific adapters for each incremental task and merges them into a universal adapter. Combined with an entropy-based adapter selection mechanism and a dual-adapter ensemble inference strategy, TUNA achieves state-of-the-art performance in exemplar-free PTM-based class-incremental learning.
tags:
  - ICCV 2025
  - Model Compression
  - Class-Incremental Learning
  - Adapter
  - Pre-Trained Model
  - Model Merging
  - Continual Learning
date: 2026-05-08
content_hash: 36898a3ddff5c648
---

# Integrating Task-Specific and Universal Adapters for Pre-Trained Model-based Class-Incremental Learning

**Conference**: ICCV 2025
**arXiv**: [2508.08165](https://arxiv.org/abs/2508.08165)
**Code**: [https://github.com/LAMDA-CL/ICCV2025-TUNA](https://github.com/LAMDA-CL/ICCV2025-TUNA)
**Area**: Model Compression
**Keywords**: Class-Incremental Learning, Adapter, Pre-Trained Model, Model Merging, Continual Learning

## TL;DR

This paper proposes TUNA, a method that trains orthogonal task-specific adapters for each incremental task and merges them into a universal adapter. Combined with an entropy-based adapter selection mechanism and a dual-adapter ensemble inference strategy, TUNA achieves state-of-the-art performance in exemplar-free PTM-based class-incremental learning.

## Background & Motivation

Class-Incremental Learning (CIL) requires models to continuously learn new categories without forgetting old ones. In the era of pre-trained models (PTMs), the dominant paradigm freezes PTM weights and performs incremental adaptation via lightweight modules such as prompts or adapters. However, existing methods suffer from two key problems:

**Inaccurate module selection**: Methods such as L2P rely on key-query matching to select task-specific prompts. This matching process is fragile and prone to selecting incorrect modules, leading to performance degradation.

**Neglect of cross-task shared knowledge**: Existing methods focus exclusively on task-specific knowledge and fail to distinguish highly similar categories across tasks (e.g., cats and dogs learned in different tasks may share similar visual appearances).

These two issues result in potential adapter misselection at inference time and an inability to leverage universal knowledge to assist classification.

## Method

### Overall Architecture

TUNA consists of three core components: (1) task-specific adapter training with orthogonality constraints; (2) multi-stage adapter merging to construct a universal adapter; and (3) entropy-based adapter selection with dual-adapter ensemble inference.

### Key Designs

1. **Orthogonal Task-Specific Adapter Training**: For each incremental task $t$, a new adapter $\mathcal{A}_t$ is initialized with a bottleneck structure ($W_{down} \in \mathbb{R}^{d \times r}$, ReLU, $W_{up} \in \mathbb{R}^{r \times d}$) and injected into the MLP layers via residual connections. To prevent feature redundancy across tasks, an orthogonality constraint is imposed on the up-projection weights:
    $\mathcal{L}_{orth} = \sum_{i=1}^{t-1} \|W_{up}^t \cdot {W_{up}^i}^\top\|_1$
   The constraint is applied only to the up-projection weights (ablations show that constraining the down-projection as well is harmful), as the up-projection is responsible for projecting features into the high-dimensional space and encoding task-specific information.

2. **Multi-Stage Adapter Merging (Universal Adapter)**: After training on $t$ tasks, all adapter weights are flattened into vectors $\{\mathbf{v}^1, \ldots, \mathbf{v}^t\}$ and merged via two operations:

    - **Sign aggregation**: For each parameter position, a sign vote is computed across all task vectors: $\mathbf{s}^{uni} = \text{sgn}(\sum_i \mathbf{v}^i)$
    - **Magnitude selection**: For each parameter position, the maximum absolute value is selected among those consistent with the consensus sign direction.
    - The final universal task vector is $\mathbf{v}^{uni} = \epsilon^{uni} \odot \mathbf{s}^{uni}$, reshaped back to the adapter structure.

   This merging strategy draws on two operations from model merging: sign aggregation as a voting system to preserve dominant feature directions, and maximum absolute value selection to suppress noise while retaining discriminative feature magnitudes.

3. **Entropy-Based Adapter Selection and Dual-Adapter Ensemble**:

    - **Selection mechanism**: Empirical analysis reveals a strong correlation between low entropy (high confidence) and high accuracy. At inference, the predictive entropy of each task-specific adapter is computed and the adapter with minimum entropy is selected:
    $\mathcal{A}^* = \arg\min_{\mathcal{A}_i} \left(-\sum_c f_c(\mathbf{x}; \mathcal{A}_i) \log f_c(\mathbf{x}; \mathcal{A}_i)\right)$
    - **Ensemble inference**: The predicted probabilities from the selected task-specific adapter and the universal adapter are summed: $y^* = \arg\max_y (f_y(\mathbf{x}; \mathcal{A}^*) + f_y(\mathbf{x}; \mathcal{A}_{uni}))$

### Loss & Training

- Classification loss: standard cross-entropy $\mathcal{L}_{cls}$
- Orthogonal loss: $\mathcal{L}_{orth}$ applied only to up-projection weights
- Total loss: $\mathcal{L} = \mathcal{L}_{cls} + \lambda \mathcal{L}_{orth}$, with $\lambda$ initialized to 1e-3 and decayed exponentially
- After training each task, class means and variances are computed for use in classifier calibration via replay in subsequent tasks
- Training details: SGD with momentum, lr=0.01 with cosine annealing, 20 epochs, batch size=48, adapter projection dim $r$=16

## Key Experimental Results

### Main Results

| Method | CIFAR $\bar{\mathcal{A}}$ | CIFAR $\mathcal{A}_B$ | IN-R $\bar{\mathcal{A}}$ | IN-R $\mathcal{A}_B$ | IN-A $\bar{\mathcal{A}}$ | IN-A $\mathcal{A}_B$ | ObjectNet $\bar{\mathcal{A}}$ | ObjectNet $\mathcal{A}_B$ |
|------|------|------|------|------|------|------|------|------|
| L2P | 85.94 | 79.93 | 75.46 | 69.77 | 49.39 | 41.71 | 63.78 | 52.19 |
| CODA-Prompt | 89.11 | 81.96 | 77.97 | 72.27 | 53.54 | 42.73 | 66.07 | 53.29 |
| SLCA | 92.49 | 88.55 | 81.17 | 77.00 | 68.66 | 58.74 | 72.55 | 61.30 |
| RanPAC | 94.00 | 90.62 | 82.98 | 77.94 | 69.32 | 61.82 | 72.76 | 62.02 |
| EASE | 91.51 | 85.80 | 81.74 | 76.17 | 65.34 | 55.04 | 70.84 | 57.86 |
| **TUNA** | **94.44** | **90.74** | **84.22** | **79.42** | **73.78** | **64.78** | **76.46** | **66.32** |

ViT-B/16-IN21K backbone, B0 Inc setting. TUNA achieves state-of-the-art performance on all four datasets, with particularly notable gains on the more challenging ImageNet-A and ObjectNet benchmarks.

### Ablation Study

| Configuration | ImageNet-A B0 Inc20 $\mathcal{A}_B$ |
|------|------|
| Baseline (multi-adapter max logit) | ~55 |
| + entropy-based adapter selection | ~59 |
| + orthogonal loss | ~62 |
| + universal adapter (full TUNA) | ~65 |

Each component contributes positively. Comparison of three inference strategies (ImageNet-A):
- Variation-1 (full TUNA strategy): best performance
- Variation-2 (entropy-selected task-specific only): lacks shared cross-task knowledge
- Variation-3 (universal adapter only): lacks task-specific fine-grained discrimination

Ablation on the orthogonal loss (ObjectNet B0 Inc20): constraining only the up-projection (Variation-1) yields the best results; jointly constraining up+down introduces excessive rigidity and leads to underfitting.

### Key Findings

- TUNA surpasses traditional CIL methods using 20 exemplars per class (e.g., iCaRL, DER, FOSTER) under an exemplar-free setting.
- Performance is robust to hyperparameter choices ($r \in \{8,16,32,64,128\}$, $\lambda \in \{0.001,...,0.1\}$), indicating good stability.
- Visualizations show that task-specific adapters provide strong within-domain discrimination but suffer cross-domain confusion (e.g., golden retriever → lion), while the universal adapter captures shared cross-task features and corrects such errors.

## Highlights & Insights

- **Simple yet effective adapter merging**: The sign-voting and maximum absolute value selection strategy extracts universal knowledge from multiple adapters without any additional training.
- **Entropy as a proxy for adapter selection**: This approach is more stable and reliable than key-query matching; experiments confirm a strong correlation between low entropy and high accuracy.
- **Complementary dual-adapter inference**: The task-specific adapter provides fine-grained discrimination while the universal adapter offers cross-task generalization.
- **Refined orthogonality constraint design**: Restricting the constraint to the up-projection alone reflects a deep understanding of the functional roles within the adapter architecture.

## Limitations & Future Work

- **Inference efficiency**: Selecting the optimal adapter requires multiple forward passes (equal to the number of tasks), with computational cost growing linearly as the number of tasks increases.
- The universal adapter merging is static (recomputed after each new task); online merging or dynamic adjustment strategies remain unexplored.
- Integration with other PEFT methods (e.g., LoRA, VPT) has not been thoroughly investigated, although the paper notes that the framework is in principle compatible with them.

## Related Work & Insights

- TUNA has a natural connection to model merging methods; the sign-voting strategy is analogous to TIES-Merging.
- The adapter merging paradigm can be generalized to multi-task learning, federated learning, and other scenarios requiring aggregation of multiple models.
- The key distinction from EASE (which concatenates features from multiple adapters) is that TUNA merges adapters into a single universal adapter rather than simple feature concatenation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The dual-adapter inference strategy and adapter merging approach are elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluations span 4 datasets with multiple baselines, detailed ablations, and comprehensive analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Logically structured with clear motivation and complete experimental presentation.
- **Value**: ⭐⭐⭐⭐ Provides an effective adapter management solution for PTM-based CIL.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Mixture of Noise for Pre-Trained Model-Based Class-Incremental Learning](../../NeurIPS2025/model_compression/mixture_of_noise_for_pre-trained_model-based_class-incremental_learning.md)
- [\[ICCV 2025\] Achieving More with Less: Additive Prompt Tuning for Rehearsal-Free Class-Incremental Learning](achieving_more_with_less_additive_prompt_tuning_for_rehearsal-free_class-increme.md)
- [\[ICCV 2025\] Efficient Adaptation of Pre-Trained Vision Transformer Underpinned by Approximation Theory](efficient_adaptation_of_pre-trained_vision_transformer_underpinned_by_approximat.md)
- [\[AAAI 2026\] Compensating Distribution Drifts in Class-incremental Learning of Pre-trained Vision Transformers](../../AAAI2026/model_compression/compensating_distribution_drifts_in_class-incremental_learning_of_pre-trained_vi.md)
- [\[ICCV 2025\] Task Vector Quantization for Memory-Efficient Model Merging](task_vector_quantization_for_memory-efficient_model_merging.md)

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Distill Gold from Massive Ores: Bi-level Data Pruning towards Efficient Dataset Distillation
description: >-
  [ECCV 2024][Causal Inference][Dataset Distillation] This paper proposes a bi-level data pruning strategy, BiLP, which combines static pruning based on empirical loss and dynamic pruning based on individual treatment effect (ITE) to efficiently select the most valuable real samples for dataset distillation. It consistently improves the performance of existing distillation methods in a plug-and-play manner while reducing computational overhead.
tags:
  - "ECCV 2024"
  - "Causal Inference"
  - "Dataset Distillation"
  - "Data Pruning"
  - "Individual Treatment Effect"
  - "Neural Tangent Kernel"
date: 2026-05-08
content_hash: b0a7ed63cc466e61
---

# Distill Gold from Massive Ores: Bi-level Data Pruning towards Efficient Dataset Distillation

**Conference**: ECCV 2024  
**arXiv**: [2305.18381](https://arxiv.org/abs/2305.18381)  
**Code**: [silicx/GoldFromOres-BiLP](https://github.com/silicx/GoldFromOres-BiLP)  
**Area**: Causal Inference  
**Keywords**: Dataset Distillation, Data Pruning, Causal Inference, Individual Treatment Effect, Neural Tangent Kernel

## TL;DR

This paper proposes a bi-level data pruning strategy, BiLP, which combines static pruning based on empirical loss and dynamic pruning based on individual treatment effect (ITE) to efficiently select the most valuable real samples for dataset distillation. It consistently improves the performance of existing distillation methods in a plug-and-play manner while reducing computational overhead.

## Background & Motivation

**Core Goal of Dataset Distillation**: Learning a small synthetic dataset from a large-scale real dataset so that models trained on the synthetic data achieve performance close to those trained on the full dataset, which is an important direction for data-efficient learning.

**Significant Redundancy in Real Data**: Experiments show that when using the DC method (IPC=1) on CIFAR-10, pruning 90% of the real data does not degrade distillation performance. In fact, distillation can be achieved with only 0.04% of the samples, demonstrating that conventional distillation algorithms waste significant computation on redundant samples.

**Prior Work Ignores Real Data Selection**: Existing distillation methods mainly focus on matching strategies (gradient matching, distribution matching, trajectory matching, etc.) or optimization acceleration, while rarely analyzing which real samples are truly valuable to the distillation process itself.

**Limited Capacity of Synthetic Data**: Since the scale of synthetic data is much smaller than real data, the NTK matrix is rank-deficient. Small-scale synthetic data cannot memorize the patterns of all real samples. Selecting "important yet easy" samples is essential to assist distillation.

**Insufficiency of Static Pruning**: Pure preprocessing-based pruning fails in certain high IPC scenarios, indicating that the value of samples changes dynamically during training, which necessitates an adaptive dynamic selection mechanism.

**Bottleneck of Large-Scale Dataset Distillation**: Distillations of large-scale datasets such as ImageNet-1K and Kinetics-400 are extremely time-consuming, urgently requiring efficient data utilization strategies to reduce computational costs.

## Method

### Overall Architecture: Bi-level Data Pruning (BiLP)

BiLP consists of a two-level data selection mechanism that can be embedded into existing distillation algorithms in a plug-and-play manner:

- **Level 1 — Preemptive Pruning**: Before distillation begins, redundant samples are statically pruned based on empirical loss values.
- **Level 2 — Adaptive Pruning**: During distillation, samples that currently contribute the most are dynamically selected based on individual treatment effect (ITE).

### Key Designs 1: Static Pruning Based on Empirical Loss

By analyzing distillation dynamics through the Neural Tangent Kernel (NTK), the authors find that: since the size of synthetic data satisfies $M_s \ll M_r$, the NTK matrix $\boldsymbol{K}_r$ is rank-deficient, making it difficult for the training dynamics of synthetic data to approximate those of the full real dataset. Samples with large gradients $\|\frac{\partial \ell}{\partial u_r}\|$ (i.e., "hard samples" with large empirical losses) contribute little to distillation and can be safely pruned. Since the NTK of wide neural networks is approximately constant, this criterion is valid at the beginning of training, enabling one-time preemptive pruning.

Specific Operation: The network is first trained on the full dataset for several epochs to record the average loss of each sample. Samples with the smallest loss values representing a fraction of $(1-\alpha)$ are kept according to a threshold $\tau$.

### Key Designs 2: Dynamic Pruning Based on Individual Treatment Effect (ITE)

The "presence or absence" of each real sample is viewed as a binary treatment, and the meta-gradient of the synthetic data is viewed as the effect. The sample contribution is measured via the Individual Treatment Effect (ITE):

$$ITE(x_r) = \frac{\partial \mathcal{L}_{meta}(\mathcal{D}_r, \mathcal{D}_s)}{\partial \mathcal{D}_s} - \frac{\partial \mathcal{L}_{meta}(\mathcal{D}_r \setminus \{x_r\}, \mathcal{D}_s)}{\partial \mathcal{D}_s}$$

Using $\|ITE(x_r)\|_2$ as the selection criterion, **samples with both the largest and smallest ITE values are pruned simultaneously**: samples with the smallest ITE contribute little to synthetic data learning, while samples with the largest ITE might be outliers that increase training instability. Retaining samples with intermediate ITE values yields the best results.

### Key Designs 3: ITE Computation Acceleration (Triple Optimization)

The original ITE computation requires computing the meta-gradient for each sample individually, causing enormous overhead. The paper proposes three acceleration techniques:

1. **Taylor Approximation**: For additively separable meta loss functions, first-order Taylor expansion is leveraged to simplify the ITE computation into a single Hessian-vector product plus sample-wise feature vectors, reducing computation time by 98.8%.
2. **Global ITE Distribution Estimation**: By maintaining the running mean $\hat{\mu}$ and variance $\hat{\sigma}^2$ of ITE and assuming a normal distribution, pruning thresholds are determined using quantiles, bypassing the need to traverse the entire dataset.
3. **Lazy Selection**: The pruning decision is updated only once every few iterations.

These three optimizations compress the ITE computational overhead to 1/60,000 of the original, with the total extra training time increasing by only 7%.

## Loss & Training

BiLP does not introduce new loss functions, but instead serves as a plug-and-play module embedded on top of the meta loss of existing distillation methods (e.g., DC, IDC). Training workflow: First, pre-train the classification network to obtain empirical loss $\rightarrow$ pre-prune with ratio $\alpha$ $\rightarrow$ compute ITE within each mini-batch during distillation $\rightarrow$ adaptively prune with ratio $\beta$ $\rightarrow$ update synthetic data using the pruned samples.

## Experiments

### Main Results: Comparison of Distillation Performance (Table 4)

| Method | CIFAR10 IPC=1 | CIFAR10 IPC=10 | CIFAR10 IPC=50 | CIFAR100 IPC=1 | CIFAR100 IPC=10 |
|------|:---:|:---:|:---:|:---:|:---:|
| DC | 28.3 | 44.9 | 53.9 | 12.8 | 25.2 |
| IDC | 50.6 | 67.5 | 74.5 | — | 45.1 |
| MTT | 46.3 | 65.6 | 71.6 | 24.3 | 40.1 |
| DREAM | 51.1 | 69.4 | 74.8 | 29.5 | 46.8 |
| **BiLP+DC** | **30.5** | **45.2** | **54.9** | **13.7** | **26.0** |
| **BiLP+IDC** | **51.5** | **69.4** | **75.4** | **30.1** | **47.2** |
| **BiLP+IDC (x3)** | **55.9** | **69.8** | **76.9** | **34.0** | **48.0** |

BiLP consistently improves DC (+0.8%) and IDC (+1.2%), with more significant improvements on diverse datasets like CIFAR-100; BiLP+IDC (x3) outperforms most SOTA methods.

### Large-Scale Dataset Distillation (Table 7)

| Dataset | IPC | Method | Full Data Acc | BiLP Acc | BiLP Training Time |
|--------|:---:|------|:---:|:---:|:---:|
| ImageNet-1K | 1 | DC | 1.79 | **2.02** | 17.3h (↓27%) |
| ImageNet-1K | 10 | DM | 3.86 | **5.21** | 15.3h (↓38%) |
| ImageNet-1K | 50 | DM | 8.22 | **9.41** | 20.6h (↓41%) |
| Kinetics-400 | 1 | DM | 2.78 | **2.92** | 29.6h (↓21%) |
| Kinetics-400 | 10 | DM | 9.48 | **9.70** | 32.2h (↓26%) |

On ImageNet-1K and Kinetics-400, BiLP simultaneously improves accuracy and reduces training time by up to 60%, making it particularly suitable for large-scale scenarios.

### Ablation Study Key Points

- Preemptive pruning and adaptive pruning both contribute independently to performance gains, and their combination achieves the best results.
- Simultaneously pruning samples with large and small ITE values outperforms single-direction pruning.
- The sample ranking bias introduced by the Taylor approximation is only about 3%, which is negligible.
- Empirical loss can be accurately estimated with only a few epochs (5 epochs are sufficient) and a small number of training runs.
- The triple optimization of ITE compresses the extra computational overhead to only 7%.

## Highlights & Insights

- **Novel Perspective**: This work is the first to systematically analyze the redundancy of real data in dataset distillation, providing theoretical support from both the NTK theory and causal inference perspectives.
- **Plug-and-Play**: BiLP can be directly embedded into various distillation methods such as DC and IDC, demonstrating high versatility.
- **Dual Benefits of Efficiency and Effectiveness**: It improves distillation accuracy while significantly reducing training time (by 27%–60% on large-scale datasets).
- **Dual Validation of Theory and Method**: The NTK rank-deficiency theory explains dataset redundancy, while the ITE causal framework characterizes the dynamic contribution of samples. Experiments cover 6 datasets and 7 distillation methods.

## Limitations & Future Work

- **Unmodeled High-Order Sample Interactions**: The current method relies on independent sample-wise evaluation without considering interactions (e.g., diversity) among samples, leading to suboptimal performance on datasets with low diversity (e.g., MNIST).
- **Limited Margin of Improvement for Distillation Methods**: The average improvement is 0.8%–1.2%, which is a relatively small absolute gain.
- **Taylor Approximation Relies on Additive Separability**: It is only applicable to distillation methods where the meta loss can be decomposed into a sum of sample-wise contributions, making it not directly applicable to trajectory matching methods such as MTT.
- **Preemptive Pruning Requires Extra Pre-training**: Although only a few epochs are needed, it still introduces some upfront overhead.

## Related Work & Insights

- **Dataset Distillation Methods**: Methods like DC (gradient matching), DM (distribution matching), MTT (trajectory matching), IDC (multi-formation factor), and DREAM (feature distribution pruning) are complementary to BiLP.
- **Data Selection/Pruning**: While traditional methods filter samples based on predefined scalar scores, BiLP is the first to introduce causal inference into data selection for distillation scenarios.
- **Neural Tangent Kernel (NTK)**: Used to model training dynamics, NTK's stability is leveraged by BiLP to support early static pruning.
- **Individual Treatment Effect (ITE) in Causal Inference**: Originally used in observational studies, ITE is innovatively applied by BiLP to measure the effect of real samples on the gradients of synthetic data.

## Rating

- Novelty: ⭐⭐⭐⭐ — Analysing data redundancy in distillation from the dual perspectives of causal inference and NTK, offering a novel framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers 6 datasets and 7 distillation methods, including large-scale ImageNet-1K and video dataset Kinetics-400, with detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear theoretical derivations, progressing logically from observations to theory to methodology.
- Value: ⭐⭐⭐⭐ — A plug-and-play general framework with high practical value for improving efficiency in large-scale distillation scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Bi-Level Decision-Focused Causal Learning for Large-Scale Marketing Optimization](../../NeurIPS2025/causal_inference/bi-level_decision-focused_causal_learning_for_large-scale_marketing_optimization.md)
- [\[ICLR 2026\] Meta-Router: Bridging Gold-standard and Preference-based Evaluations in LLM Routing](../../ICLR2026/causal_inference/meta-router_bridging_gold-standard_and_preference-based_evaluations_in_llm_routi.md)
- [\[ICLR 2026\] Query-Specific Causal Graph Pruning under Tiered Knowledge](../../ICLR2026/causal_inference/query-specific_causal_graph_pruning_under_tiered_knowledge.md)
- [\[AAAI 2026\] Sparse Additive Model Pruning for Order-Based Causal Structure Learning](../../AAAI2026/causal_inference/sparse_additive_model_pruning_for_order-based_causal_structure_learning.md)
- [\[ICLR 2026\] Efficient Ensemble Conditional Independence Test Framework for Causal Discovery](../../ICLR2026/causal_inference/efficient_ensemble_conditional_independence_test_framework_for_causal_discovery.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Gradient-Regularized Out-of-Distribution Detection
description: >-
  [ECCV 2024][LLM Evaluation][OOD Detection] This paper proposes GReg/GReg+, which learns the local smoothness of the scoring manifold by regularizing the input gradient norm of the OOD scoring function, and incorporates an energy-score-based clustering sampling strategy to select highly informative auxiliary samples, achieving SOTA on CIFAR and ImageNet OOD detection benchmarks.
tags:
  - "ECCV 2024"
  - "LLM Evaluation"
  - "OOD Detection"
  - "Gradient Regularization"
  - "Energy Score"
  - "Lipschitz Analysis"
  - "Auxiliary Data Sampling"
date: 2026-05-08
content_hash: a54027bdee553409
---

# Gradient-Regularized Out-of-Distribution Detection

**Conference**: ECCV 2024  
**arXiv**: [2404.12368](https://arxiv.org/abs/2404.12368)  
**Code**: [https://github.com/o4lc/Greg-OOD](https://github.com/o4lc/Greg-OOD)  
**Area**: LLM Evaluation  
**Keywords**: OOD Detection, Gradient Regularization, Energy Score, Lipschitz Analysis, Auxiliary Data Sampling

## TL;DR

This paper proposes GReg/GReg+, which learns the local smoothness of the scoring manifold by regularizing the input gradient norm of the OOD scoring function, and incorporates an energy-score-based clustering sampling strategy to select highly informative auxiliary samples, achieving SOTA on CIFAR and ImageNet OOD detection benchmarks.

## Background & Motivation

Neural networks often generate overconfident false predictions when encountering out-of-distribution (OOD) data. OOD detection aims to distinguish between in-distribution (ID) data and OOD data.

Limitations of prior work:
- **Post-hoc methods** (MSP, ODIN, ReAct, DICE, LINe): Do not use auxiliary data and distinguish ID/OOD through inference-time statistics, which limits their performance.
- **Methods using auxiliary data** (OE, Energy Loss, POEM, DOS): Train utilizing auxiliary OOD data, but **only focus on the values of the scoring function**, neglecting to utilize the local structural information of the scoring function.
- **Sampling issues**: The auxiliary dataset can be much larger than the ID data, and greedy sampling tends to bias toward specific regions, leading to insufficient generalization.

The core motivation stems from a simple observation (as shown in Figure 1 of the paper): even if two OOD samples $x_1, x_2$ have the same score, if the gradient of the scoring function at $x_2$ is large (steep surface), samples in its neighborhood are highly susceptible to being misdetected as ID. Therefore, optimizing only the scoring values is insufficient; the local smoothness of the scoring function must also be controlled.

## Method

### Overall Architecture

GReg+ consists of two core components:
1. **Gradient Regularization (GReg)**: Adds a regularization term penalizing the input gradient norm of the scoring function on top of the energy loss.
2. **Energy-based Clustering Sampling**: Conducts feature-space clustering on auxiliary OOD data and selects the most informative samples from each cluster based on energy scores.

### Key Designs

1. **Gradient Regularization Loss**: Based on the energy scoring function $S_\text{En}(x) = -\text{LSE}(f(x))$, a gradient norm regularization term is added to the standard energy loss $\mathcal{L}_{S_\text{En}}$:

    $\mathcal{L}_{\nabla S_\text{En}} = \mathbb{E}_{x_\text{in}} \| \mathbb{I}_{S_\text{En}(x_\text{in}) \leq m_\text{in}} \nabla_{x_\text{in}} S_\text{En}(x_\text{in}) \| + \mathbb{E}_{x_\text{aux}} \| \mathbb{I}_{S_\text{En}(x_\text{aux}) \geq m_\text{aux}} \nabla_{x_\text{aux}} S_\text{En}(x_\text{aux}) \|$

   Key design aspect: **Only apply gradient regularization to correctly detected samples**—for ID samples, the gradient is penalized only when their energy is already sufficiently low (correctly classified as ID); for OOD samples, the gradient is penalized only when their energy is already sufficiently high (correctly classified as OOD). This allows the energy loss to focus on "learning" detection, while the gradient loss concentrates on "stabilizing" the learned detection.

   Total loss function: $\mathcal{L} = \mathcal{L}_\text{CE} + \lambda_S \mathcal{L}_S + \lambda_{\nabla S} \mathcal{L}_{\nabla S}$

2. **Energy-based Clustering Sampling**: When the auxiliary dataset is very large (e.g., 300K RandomImages), sample selection must be efficient. Algorithm workflow:

    - Extract features $z_i = h(x_i)$ for all OOD samples and L2-normalize them to $\hat{z}_i$.
    - Use K-Means to cluster $n_\text{OOD}$ samples into $k = n_\text{ID}$ clusters.
    - Select the **lowest energy samples** from each cluster (for the energy loss $\mathcal{L}_S$, which need to be improved) and the **highest energy samples** (for the gradient loss $\mathcal{L}_{\nabla S}$, which need to be stabilized).

   Clustering ensures spatial diversity while energy selection guarantees sample informativeness, making them complementary.

3. **Theoretical Guarantees (Lipschitz Analysis)**: If the scoring function satisfies the local Lipschitz condition within the $\varepsilon$-neighborhood of $x$, a correctly classified ID sample $x$ has a certified radius:

    $$\varepsilon^* = \min\left\{\varepsilon, \frac{\gamma - S(x)}{L_S}\right\}$$

   Minimizing the local Lipschitz constant $L_S$ expands the certified radius. For ReLU networks, the local Lipschitz constant equals the gradient norm, meaning that **penalizing the gradient norm is equivalent to controlling the local Lipschitz constant**, which provides a theoretical foundation of robustness for this method.

### Loss & Training

- GReg (without sampling): Fine-tune on the pre-trained model for 20 epochs using SGD + cosine annealing.
- GReg+ (with sampling): Train from scratch for 50 epochs (lr=0.1) + fine-tune for 10 epochs (lr=0.01).
- Hyperparameter settings: $\lambda_S = 0.1$, $\lambda_{\nabla S} = 1$.
- CIFAR auxiliary data: 300K RandomImages, clustering is performed per mini-batch each epoch.
- ImageNet auxiliary data: Randomly sample 600K images from the remaining 990 classes each epoch.

## Key Experimental Results

### Main Results

**CIFAR-10 (Average FPR95↓ / AUROC↑):**

| Method | ResNet FPR95 | ResNet AUROC | WRN FPR95 | DenseNet FPR95 |
|------|-------------|-------------|-----------|--------------|
| Energy Loss | 11.14 | 97.53 | 13.11 | 11.26 |
| OpenMix | 22.24 | 96.26 | 21.92 | 22.86 |
| **GReg** | **7.90** | **97.95** | **7.95** | **7.93** |

**ImageNet (10 ID classes, average FPR95↓ / AUROC↑):**

| Method | FPR95↓ | AUROC↑ |
|------|--------|--------|
| LINe | 39.48 | 91.29 |
| DOS | 49.31 | 91.97 |
| Energy Loss | 45.23 | 88.63 |
| GReg | 47.26 | 90.13 |
| **GReg+** | **35.08** | **92.06** |

GReg+ reduces the FPR95 of the best baseline method by over 4% on ImageNet.

### Ablation Study

| Configuration | CIFAR-10 FPR95↓ | CIFAR-10 AUROC↑ | Description |
|------|-----------------|-----------------|------|
| OE | 21.76 | 95.80 | Baseline |
| OE + Grad | 18.79 | 96.32 | + Gradient Regularization |
| Energy | 11.26 | 97.43 | Baseline |
| Energy + Grad | 7.93 | 98.12 | + Gradient Regularization, **30% relative improvement** |
| POEM | 29.15 | 94.18 | Baseline |
| POEM + Grad | 23.87 | 95.41 | + Gradient Regularization |

### Key Findings

- **Generality of Gradient Regularization**: It can be applied in a plug-and-play manner to boost the performance of various methods such as OE, Energy, and POEM.
- **Complementarity of Energy Loss and Gradient Loss**: Energy loss is responsible for widening the ID/OOD score gap, while gradient loss is responsible for smoothing the local manifold.
- **Sampling Significantly Boosts CIFAR-100 and ImageNet Performance**: GReg+ makes a larger contribution than GReg in challenging scenarios.
- **No Harm to ID Accuracy**: Gradient regularization simultaneously improves OOD detection capability and ID classification accuracy.
- **Regularization Suppresses Gradient Norm Growth During Training** (Figure 3): Without GReg, the gradient norm rises rapidly; with GReg, the growth is gentle.

## Highlights & Insights

1. **From Score Values to Scoring Manifold**: Expanding the focus from the output value of the scoring function to its local geometric structure (gradient norm) is a meaningful conceptual innovation.
2. **Unified Theory and Practice**: Lipschitz analysis provides certified robustness theoretical support for gradient regularization, which holds exactly for ReLU networks.
3. **Exquisitely Designed Clustering Sampling**: The two-ended sampling strategy targeting the lowest energy (to be improved) and highest energy (to be stabilized) precisely aligns with the two loss terms.
4. **Simple and General Method**: As a regularization term, GReg can be seamlessly integrated into any scoring-function-based OOD training method.

## Limitations & Future Work

- Gradient calculation introduces extra backpropagation overhead (requiring gradients with respect to the input), which can be slow in large-scale training.
- The ImageNet experiments only utilize 10 classes as ID, failing to validate on the full 1K class scenario.
- The theoretical analysis is based on the local Lipschitz assumption, which is not fully applicable to networks with non-piecewise linear activation functions.
- Clustering sampling requires pre-extracting features of all auxiliary data, which may present scalability issues for extremely large auxiliary sets (such as ImageNet-21K).
- Integration with more recent, stronger post-hoc methods (such as ASH, SHE, etc.) has not been discussed.

## Related Work & Insights

- **Energy Score** (Liu et al., NeurIPS 2020): GReg is directly built upon energy scores.
- **DOS** (ICLR 2024): A SOTA method using K-Means diversity sampling; GReg+'s sampling strategy offers improvements over it.
- **Certified Robustness**: Gradient regularization establishes connections with Lipschitz analysis from certified robustness literature.
- **Core Insight**: Smoothness as a design philosophy can be generalized to other detection tasks (e.g., anomaly detection, novelty detection).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Regularizing the gradient norm of the scoring function is an innovative perspective, and the theoretical analysis adds depth.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive experiments across multiple architectures, datasets, and ablations, further demonstrating generality across different methods.
- **Writing Quality**: ⭐⭐⭐⭐ — Intuitive motivation (the 2D demonstration in Figure 1), with clear theoretical derivations.
- **Value**: ⭐⭐⭐⭐ — Gradient regularization serves as a plug-and-play improvement with high practical value, though the ImageNet experimental scale is relatively small.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DisCoPatch: Taming Adversarially-driven Batch Statistics for Improved Out-of-Distribution Detection](../../ICCV2025/llm_evaluation/discopatch_taming_adversarially-driven_batch_statistics_for_improved_out-of-dist.md)
- [\[ICCV 2025\] ODP-Bench: Benchmarking Out-of-Distribution Performance Prediction](../../ICCV2025/llm_evaluation/odp-bench_benchmarking_out-of-distribution_performance_prediction.md)
- [\[ECCV 2024\] Distribution Alignment for Fully Test-Time Adaptation with Dynamic Online Data Streams](distribution_alignment_for_fully_test-time_adaptation_with_dynamic_online_data_s.md)
- [\[ICML 2025\] G-Sim: Generative Simulations with Large Language Models and Gradient-Free Calibration](../../ICML2025/llm_evaluation/g-sim_generative_simulations_with_large_language_models_and_gradient-free_calibr.md)
- [\[ECCV 2024\] Versatile Incremental Learning: Towards Class and Domain-Agnostic Incremental Learning](versatile_incremental_learning_towards_class_and_domain-agnostic_incremental_lea.md)

</div>

<!-- RELATED:END -->

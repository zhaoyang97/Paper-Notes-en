---
title: >-
  [Paper Note] VKDNW: Training-free Neural Architecture Search through Variance of Knowledge of Deep Network Weights
description: >-
  [CVPR 2025][Neural Architecture Search] VKDNW proposes a training-free NAS proxy based on the spectral entropy of the eigenvalues of the Fisher Information Matrix (FIM). It successfully applies Fisher information theory to large-scale deep network architecture search for the first time, evaluating network classification accuracy potential without any training. Additionally, it introduces the nDCG evaluation metric, which is better suited for NAS tasks.
tags:
  - "CVPR 2025"
  - "Neural Architecture Search"
  - "Training-free proxy"
  - "Fisher Information Matrix"
  - "Cramér-Rao Bound"
  - "Evaluation Metrics"
date: 2026-05-08
content_hash: 0e88589c0cf38e98
---

# VKDNW: Training-free Neural Architecture Search through Variance of Knowledge of Deep Network Weights

**Conference**: CVPR 2025  
**arXiv**: [2502.04975](https://arxiv.org/abs/2502.04975)  
**Code**: [GitHub](https://www.github.com/ondratybl/VKDNW)  
**Area**: Others (Neural Architecture Search)  
**Keywords**: Neural Architecture Search, Training-free proxy, Fisher Information Matrix, Cramér-Rao Bound, Evaluation Metrics

## TL;DR

VKDNW proposes a training-free NAS proxy based on the spectral entropy of the eigenvalues of the Fisher Information Matrix (FIM). It successfully applies Fisher information theory to large-scale deep network architecture search for the first time, evaluating network classification accuracy potential without any training. Additionally, it introduces the nDCG evaluation metric, which is better suited for NAS tasks.

## Background & Motivation

Neural Architecture Search (NAS) aims to systematically search for optimal network architectures, but its core bottleneck is the astronomical computational cost—the basic setup requires training thousands of different architectures.

Key challenges in training-free NAS (TF-NAS):
- **Weak theoretical foundation of existing proxies**: Methods such as GradNorm, NASWOT, and ZenNAS are mostly based on empirical heuristics and lack rigorous statistical theoretical support.
- **Failure of Fisher Information in large networks**: Although the FIM is the "obvious choice" for analyzing networks, it faces numerical instability and computational infeasibility on large-scale, over-parameterized models.
- **Evaluation metrics unsuitable for NAS objectives**: Kendall's $\tau$ and Spearman's $\rho$ treat all networks equally, whereas NAS in practice only cares about whether the best networks can be identified.

Key insight: The quality of a network architecture can be measured by the difficulty of the weight estimation process—the more uniform the FIM eigenvalues, the more consistent the uncertainty of weight estimation in all directions, making training easier and the architecture better.

## Method

### Overall Architecture

VKDNW operates in three steps: (1) calculate the empirical Fisher Information Matrix using random inputs on randomly initialized networks (without ground-truth labels); (2) extract the deciles of the FIM eigenvalue spectrum; (3) compute the entropy of the eigenvalue distribution as an architecture quality proxy, combining it with the number of trainable layers for final ranking.

### Key Designs

**Design 1: Stable and Efficient Estimation of FIM Eigenvalue Spectrum**

- **Function**: Reliably computes the eigenvalues of the Fisher Information Matrix on large-scale, over-parameterized deep networks.
- **Mechanism**: Decomposes the empirical FIM as $\hat{F}(\theta) = \frac{1}{n}\sum_{n=1}^N A_n^T A_n$, utilizing analytical formulas of the softmax function to handle numerically unstable matrix decomposition. It extracts a small number of representative parameters from each trainable layer to construct a low-dimensional FIM, and employs SVD instead of eigenvalue decomposition to handle the ill-conditioned spectrum of the semi-definite matrix.
- **Design Motivation**: Directly computing a $p \times p$ (where $p$ is the number of parameters) FIM is infeasible. Since network outputs are highly imbalanced at initialization, the $\text{diag}(\sigma) - \sigma\sigma^T$ term must be handled correctly to avoid overflow/underflow. A small number of representative parameters is sufficient to capture the essential structural properties of the architecture.

**Design 2: VKDNW — Architecture Proxy Based on FIM Eigenvalue Spectral Entropy**

- **Function**: Quantifies the training feasibility of network architectures independently of network size.
- **Mechanism**: Defines $\text{VKDNW}(f) = -\sum_{k=1}^9 \tilde{\lambda}_k \log \tilde{\lambda}_k$, where $\tilde{\lambda}_k = \lambda_k / \sum_j \lambda_j$ is the normalized deciles of the FIM eigenvalue spectrum. VKDNW reaches its maximum when all eigenvalues are equal, and decreases as the eigenvalues become less uniform. Combined with the number of trainable layers $\aleph(f)$, it forms: $\text{VKDNW}_\text{single}(f) = \aleph(f) + \text{VKDNW}(f)$.
- **Design Motivation**: Based on the Cramér-Rao bound, the uniformity of FIM eigenvalues reflects the balance of weight estimation uncertainty in all directions. More uniform eigenvalues indicate more consistent training difficulty across parameter directions, and more balanced sensitivity of predictions to weight perturbations. Normalization makes the proxy independent of network size, while the combination with the layer count captures both structural quality and network capacity.

**Design 3: nDCG Evaluation Metric — Focusing on Top-Ranking Quality**

- **Function**: More accurately evaluates the ability of TF-NAS proxies to identify the best network.
- **Mechanism**: Inspired by the NDCG metric in information retrieval, it defines $\text{nDCG}_P = \frac{1}{Z} \sum_{j=1}^P \frac{2^{\text{acc}_{k_j}} - 1}{\log_2(1+j)}$, where $P$ is the number of top networks considered. High-accuracy networks receive higher weights when they appear at the top of the ranking.
- **Design Motivation**: Both Kendall's $\tau$ and Spearman's $\rho$ treat all ranking pairs equally, failing to distinguish between a proxy that performs poorly at the top but well at the bottom versus one that performs well at the top but poorly at the bottom. Since NAS only cares about finding the best network, nDCG serves as the exact metric to measure this capability.

### Loss & Training

VKDNW is a training-free scoring method that does not involve any loss function. The empirical FIM is computed using the initialized weights $\theta_\text{init}$ and random inputs, requiring no ground-truth data labels.

## Key Experimental Results

### Main Results: NAS-Bench-201 Search Space

| Method | CIFAR-10 KT | CIFAR-100 KT | ImageNet16 KT | CIFAR-10 nDCG | ImageNet16 nDCG |
|------|------------|-------------|--------------|--------------|----------------|
| FLOPs | 0.623 | 0.586 | 0.545 | 0.745 | 0.403 |
| GradNorm | 0.328 | 0.341 | 0.310 | 0.509 | 0.265 |
| NASWOT | Med | Med | Med | Med | Med |
| **VKDNW** | **Highest** | **Highest** | **Highest** | **Highest** | **Highest** |

### Key Findings

- VKDNW achieves SOTA performance across three datasets and two search spaces, leading comprehensively across KT, SPR, and nDCG metrics.
- Interesting finding: The simple "number of trainable layers" $\aleph$ as a proxy significantly outperforms FLOPs on multiple metrics.
- The information captured by VKDNW is orthogonal to model size, and their combination performs better than either alone.
- The empirical FIM doesn't rely on ground-truth labels (using random inputs of the correct shape is sufficient), illustrating that intrinsic structural properties of the architecture can be captured at initialization.
- The nDCG metric reveals significant performance gaps between prior methods that were previously obscured by traditional metrics.

## Highlights & Insights

1. **Theoretical Depth**: Successfully applies Fisher information theory to large-scale NAS for the first time, overcoming long-standing numerical and computational barriers.
2. **Elegant Proxy Design**: The entropy of the FIM eigenvalue spectrum elegantly captures the intuition of "balanced training difficulty", backed by statistical theory and remaining simple to compute.
3. **Contribution of Evaluation Metric**: The introduction of nDCG fills a gap in the evaluation of TF-NAS, with a highly convincing toy example demonstration.

## Limitations & Future Work

- The Cramér-Rao bound-based motivation assumes the FIM is evaluated at the correct weights, whereas in practice it is evaluated at random initialization, leaving a gap in theoretical guarantees.
- The choice of deciles and the setting of 9 representative eigenvalues are somewhat empirical.
- Validation is primarily performed on classification tasks; generalization to other tasks like object detection or segmentation remains to be verified.
- Future work can explore the dynamic variation of FIM during the training process.

## Related Work & Insights

- **NASWOT/ZenNAS**: Empirical TF-NAS proxies lacking solid theoretical foundations.
- **GradNorm/GraSP**: Gradient-based proxies to which VKDNW provides complementary information.
- **Applications of Fisher Information in ML**: EWC (continual learning) and natural gradient optimization; this work extends it to architecture search.
- Insight: Classical theoretical tools from statistics (e.g., FIM, Cramér-Rao bound) still hold immense application potential in deep learning, where the key lies in addressing computational feasibility issues.

## Rating

⭐⭐⭐⭐ — Solid theoretical contribution; the first successful application of FIM to NAS is impressive. The proposed nDCG evaluation metric holds independent value. The experiments are comprehensive with highly consistent results. This work provides a fresh theoretical perspective to the TF-NAS field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Subnet-Aware Dynamic Supernet Training for Neural Architecture Search](subnet-aware_dynamic_supernet_training_for_neural_architecture_search.md)
- [\[ICCV 2025\] Loss Functions for Predictor-based Neural Architecture Search](../../ICCV2025/others/loss_functions_for_predictor-based_neural_architecture_search.md)
- [\[ECCV 2024\] Auto-GAS: Automated Proxy Discovery for Training-Free Generative Architecture Search](../../ECCV2024/others/auto-gas_automated_proxy_discovery_for_training-free_generative_architecture_sea.md)
- [\[CVPR 2026\] InTrain: Intrinsic Trainability for Zero-Cost Neural Architecture Search](../../CVPR2026/others/intrain_intrinsic_trainability_for_zero-cost_neural_architecture_search.md)
- [\[ICLR 2026\] Training Deep Normalization-Free Spiking Neural Networks with Lateral Inhibition](../../ICLR2026/others/training_deep_normalization-free_spiking_neural_networks_with_lateral_inhibition.md)

</div>

<!-- RELATED:END -->

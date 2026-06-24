---
title: >-
  [Paper Note] Subnet-Aware Dynamic Supernet Training for Neural Architecture Search
description: >-
  [CVPR 2025][Neural Architecture Search] This paper proposes a dynamic supernet training strategy (CaLR + MS). It addresses the subnet training unfairness problem via complexity-aware learning rate scheduling, and alleviates the gradient noise issue through momentum separation, significantly improving the search performance of N-shot NAS with minimal computational overhead.
tags:
  - "CVPR 2025"
  - "Neural Architecture Search"
  - "Supernet Training"
  - "Learning Rate Scheduling"
  - "Momentum Separation"
  - "Subnet Fairness"
date: 2026-05-08
content_hash: 5e4669551fe3a57b
---

# Subnet-Aware Dynamic Supernet Training for Neural Architecture Search

**Conference**: CVPR 2025  
**arXiv**: [2503.10740](https://arxiv.org/abs/2503.10740)  
**Code**: [Project Page](https://cvlab.yonsei.ac.kr/projects/DYNAS/)  
**Area**: Others  
**Keywords**: Neural Architecture Search, Supernet Training, Learning Rate Scheduling, Momentum Separation, Subnet Fairness

## TL;DR

This paper proposes a dynamic supernet training strategy (CaLR + MS). It addresses the subnet training unfairness problem via complexity-aware learning rate scheduling, and alleviates the gradient noise issue through momentum separation, significantly improving the search performance of N-shot NAS with minimal computational overhead.

## Background & Motivation

N-shot NAS methods use a supernet containing all candidate subnets and train the supernet to estimate subnet performance. Existing methods employ static training strategies (where all subnets share the same learning rate scheduler and optimizer), thereby ignoring individual subnet characteristics, which leads to two critical limitations:

(1) **Unfairness**: High-complexity subnets possess more parameters and require more training iterations to fully converge, yet static strategies treat all subnets equally. Consequently, low-complexity subnets converge prematurely and rank high, while high-complexity subnets are insufficiently trained and underestimated, even if their true potential is superior.

(2) **Noisy Momentum**: Subnets randomly sampled from the supernet produce highly divergent gradients at each step. Accumulating these gradients into a single momentum buffer leads to noisy momentum directions and unstable training.

**Core Problem**: Existing methods fail to consider individual subnet characteristics (complexity and structure), resulting in poor supernet ranking consistency and subsequently yielding suboptimal searched architectures.

## Method

### Overall Architecture

The dynamic supernet training framework consists of two plug-and-play components: (1) Complexity-aware Learning Rate Scheduler (CaLR), which adjusts the LR decay rate based on subnet complexity; and (2) Momentum Separation (MS), which groups structurally similar subnets and assigns an independent momentum buffer to each group. These two components complement each other and can be readily applied to various NAS methods, such as SPOS, FairNAS, and FSNAS.

### Key Design 1: Complexity-aware Learning Rate Scheduler (CaLR)

**Function**: Adjusts the LR decay rate according to subnet complexity to balance the training sufficiency of different subnets.

**Mechanism**: A polynomial LR scheduler $\eta^t = \eta^0 \cdot (1 - t/T)^{\gamma(\alpha)}$ is employed, where the decay rate is $\gamma(\alpha) = \omega \log(\mathcal{C}(\alpha)) + \tau$, and $\mathcal{C}(\alpha)$ represents the subnet parameter count. High-complexity subnets have $\gamma < 1$ (slower LR decay, maintaining a larger learning rate for a longer duration), while low-complexity subnets have $\gamma > 1$ (faster LR decay to prevent overtraining). Medium-complexity subnets align with $\gamma = 1$, which is equivalent to linear decay.

**Design Motivation**: High-complexity subnets contain more parameters that require tuning, necessitating more effective training iterations. Directly increasing training epochs is computationally expensive, whereas adjusting the LR decay rate mathematically provides high-complexity subnets with extra exploration opportunities in the parameter space. The logarithmic function ensures that medium-complexity subnets fall back to standard linear decay.

### Key Design 2: Momentum Separation (MS)

**Function**: Reduces momentum noise during supernet training to stabilize the training process.

**Mechanism**: Subnets are clustered based on the operation types selected at a specific edge/layer: $S_i = \{\alpha \in \mathcal{A} | \text{op}(\alpha, e) = o_i\}$. An independent momentum buffer $\mu_i$ is allocated to each cluster: $\mu_i^t = \beta \cdot \mu_i^{t-1} + g^t$. A sampled subnet $\alpha$ updates its corresponding momentum buffer $\mu_i$ based on the cluster $S_i$ it belongs to. Weights are still shared across all subnets.

**Design Motivation**: This is built upon the empirical observation that structurally similar subnets generate similar gradients. By grouping such subnets, high gradient consistency is maintained within each group, yielding more stable momentum updates. Clustering is based on operation types on a single edge/layer, meaning the number of groups equals the number of candidate operations (e.g., 7 groups), which introduces negligible extra memory overhead for the momentum buffers.

### Key Design 3: Evaluation Metrics CB and C3

**Function**: Quantifies the unfairness issue during supernet training.

**Mechanism**: This work introduces Complexity Bias (CB) to measure the degree to which the supernet ranking biases towards low-complexity subnets, and Complexity-Convergence Correlation (C3) to measure the correlation between complexity and convergence level. These two metrics directly detect the unfairness problem and validate the efficacy of CaLR.

**Design Motivation**: Previously, quantitative metrics for the unfairness issue were lacking, making it only indirectly evaluable through final search results. CB and C3 offer direct diagnostic tools.

### Loss & Training

Standard task training losses (such as cross-entropy) are used without introducing any auxiliary loss terms. CaLR and MS only modify the optimization procedure (LR scheduling and momentum updates), exerting no impact on the loss function design.

## Key Experimental Results

### ImageNet MobileNet Search Space

| Method | Params(M) | FLOPs(M) | Top-1(%) | GPU Hours |
|------|-----------|----------|----------|-----------|
| SPOS-L | 4.5 | 471 | 76.6 | 157 |
| **SPOS-L + Ours** | **4.7** | **459** | **77.1** | **159** |
| FairNAS-L | 4.7 | 472 | 76.7 | 364 |
| **FairNAS-L + Ours** | **4.7** | **471** | **77.0** | **369** |
| FSNAS-L | 4.7 | 464 | 76.8 | 740 |
| **FSNAS-L + Ours** | **4.5** | — | **Gain** | — |

### NAS-Bench-201 Ranking Consistency (Kendall's Tau)

| Method | CIFAR-10 | CIFAR-100 | ImageNet-16 |
|------|----------|-----------|-------------|
| SPOS | Baseline | Baseline | Baseline |
| SPOS + CaLR | +Gain | +Gain | +Gain |
| SPOS + MS | +Gain | +Gain | +Gain |
| **SPOS + CaLR + MS** | **Best** | **Best** | **Best** |

### Key Findings

1. **Consistency Improvement**: CaLR + MS significantly improves supernet ranking consistency (Kendall's Tau) across all tested NAS methods and datasets.
2. **ImageNet Top-1 Gains**: SPOS-L improves from 76.6% to 77.1% (+0.5%), and FairNAS-L improves from 76.7% to 77.0% (+0.3%), while adding only about 1% GPU time.
3. **Strong Complementarity**: CaLR and MS target distinct issues (fairness vs. stability), achieving optimal performance when used jointly.
4. **High Versatility**: The proposed method can be seamlessly integrated into both one-shot (e.g., SPOS, FairNAS) and few-shot (e.g., FSNAS) NAS frameworks.
5. **Negligible Overhead**: Peak memory usage increases by less than 1%, and GPU time increases by only 1-5%.

## Highlights & Insights

- **Profound Insights**: The study precisely identifies the two core vulnerabilities of static supernet training and proposes targeted, effective solutions.
- **Plug-and-Play**: Both components can be applied independently or jointly to any N-shot NAS method without requiring modifications to the search space or sampling strategy.
- **Combining Theory with Practice**: Visual analysis of the unfairness problem alongside the introduced CB/C3 metrics provides clear motivational support.

## Limitations & Future Work

- **Choice of Complexity Metrics**: The study only utilizes parameter size as the complexity score, neglecting other dimensions such as FLOPs and memory consumption.
- **Simplistic Clustering Strategy**: MS clusters subnets based on operational types at a single edge/layer, which may not fully capture gradient similarities between complex subnets.
- **Insufficient Validation on Grand Search Spaces**: Validations within the $7^{21}$ scale search space are limited to the MobileNet space.
- Future research can explore adaptive clustering, multi-dimensional complexity scoring, and combination with non-uniform sampling strategies.

## Related Work & Insights

- **FairNAS**: Balances training by sampling multiple subnets but ignores the discrepancy in complexities. PPA tackles fairness through the lens of the optimization process.
- **PA&DA**: Alleviates search difficulty by reducing gradient variance but focuses on sampling strategies, whereas MS concentrates on optimizer momentum.
- **Insights**: Individual differences among subnets in supernet training constitute a highly overlooked factor, where dynamic training strategies yield an elegant, generalized solution.

## Rating

⭐⭐⭐⭐ — This work offers a precise diagnosis of current limitations combined with simple yet effective solutions. The plug-and-play design holds significant practical value. Achieving consistent improvements across various NAS methodologies with negligible overhead stands out as the biggest strength. The simplistic clustering strategy is a minor drawback.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] VKDNW: Training-free Neural Architecture Search through Variance of Knowledge of Deep Network Weights](training-free_neural_architecture_search_through_variance_of_knowledge_of_deep_n.md)
- [\[ICCV 2025\] Loss Functions for Predictor-based Neural Architecture Search](../../ICCV2025/others/loss_functions_for_predictor-based_neural_architecture_search.md)
- [\[CVPR 2026\] InTrain: Intrinsic Trainability for Zero-Cost Neural Architecture Search](../../CVPR2026/others/intrain_intrinsic_trainability_for_zero-cost_neural_architecture_search.md)
- [\[CVPR 2025\] EVOS: Efficient Implicit Neural Training via EVOlutionary Selector](evos_efficient_implicit_neural_training_via_evolutionary_selector.md)
- [\[CVPR 2025\] Tuning the Frequencies: Robust Training for Sinusoidal Neural Networks](tuning_the_frequencies_robust_training_for_sinusoidal_neural_networks.md)

</div>

<!-- RELATED:END -->

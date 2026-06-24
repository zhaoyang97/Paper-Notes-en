---
title: >-
  [Paper Note] FedSWA: Improving Generalization in Federated Learning with Highly Heterogeneous Data via Momentum-Based Stochastic Controlled Weight Averaging
description: >-
  [ICML2025][Optimization][Federated Learning] To address the generalization failure of FedSAM under high data heterogeneity, this paper proposes FedSWA (cyclic learning rate + EMA aggregation) and FedMoSWA (momentum-based variance reduction control variables). Both theoretical analysis and empirical results demonstrate their superiority over FedSAM and its variants, achieving a 21.8% accuracy improvement over FedSAM on CIFAR-100 with Dirichlet-0.1.
tags:
  - "ICML2025"
  - "Optimization"
  - "Federated Learning"
  - "Generalization"
  - "Stochastic Weight Averaging"
  - "Data Heterogeneity"
  - "Variance Reduction"
  - "Loss Landscape Flatness"
date: 2026-05-08
content_hash: eff6bcde1deb25f5
---

# FedSWA: Improving Generalization in Federated Learning with Highly Heterogeneous Data via Momentum-Based Stochastic Controlled Weight Averaging

**Conference**: ICML2025  
**arXiv**: [2507.20016](https://arxiv.org/abs/2507.20016)  
**Code**: [junkangLiu0/FedSWA](https://github.com/junkangLiu0/FedSWA)  
**Area**: Optimization / Federated Learning  
**Keywords**: Federated Learning, Generalization, Stochastic Weight Averaging, Data Heterogeneity, Variance Reduction, Loss Landscape Flatness

## TL;DR

To address the generalization failure of FedSAM under high data heterogeneity, this paper proposes FedSWA (cyclic learning rate + EMA aggregation) and FedMoSWA (momentum-based variance reduction control variables). Both theoretical analysis and empirical results demonstrate their superiority over FedSAM and its variants, achieving a 21.8% accuracy improvement over FedSAM on CIFAR-100 with Dirichlet-0.1.

## Background & Motivation

- **Generalization Challenges in Federated Learning**: Data heterogeneity (Non-IID) in FL causes the global model to bias toward sharp local minima, leading to poor generalization performance.
- **Failure of FedSAM**: SAM seeks **locally flat minima** rather than **globally flat minima** on local clients. In highly heterogeneous scenarios (Dirichlet-0.1), it performs even worse than FedAvg (CIFAR-100: FedSAM 40.1% vs. FedAvg 45.8%).
- **Extra Computational Overhead of SAM**: SAM requires extra forward and backward passes to compute perturbations, making it less efficient than SWA.
- **Key Observation**: By averaging different weights at later training stages, SWA naturally locates the center of flat regions in the loss landscape, making it more suitable for federated scenarios.

## Method

### Overall Architecture

Two progressive algorithms are proposed: FedSWA, which improves the server aggregation strategy, and FedMoSWA, which further aligns local and global models using control variables.

### FedSWA: Cyclic Learning Rate + EMA Aggregation

**Local Learning Rate Decay Strategy**: Within each communication round, the learning rate decays linearly from $\eta_l$ to $\rho \eta_l$:

$$\eta_k^t = \eta_l \left(1 - \frac{k}{K}\right) + \frac{k}{K} \rho \eta_l$$

Local update: $\theta_{i,k+1}^{(t)} = \theta_{i,k}^{(t)} - \eta_k^t \cdot g_i(\theta_{i,k}^{(t)})$

**Server EMA Aggregation** (inspired by LookAhead):

$$v_t = \frac{1}{s} \sum_{i=1}^{s} \theta_{i,K}^t, \quad \theta_t = \theta_{t-1} + \alpha (v_t - \theta_{t-1})$$

- The initial large learning rate is restored at the end of each communication round (learning rate restart) to help escape poor local minima.
- Difference from FedAvg: FedAvg employs a constant learning rate combined with simple averaging, whereas FedSWA integrates a cyclic learning rate with EMA aggregation.

### FedMoSWA: Momentum-Based Variance Reduction Control Variables

Building upon FedSWA, a momentum-based variance reduction mechanism is introduced, modifying the local update to:

$$\theta_{i,k+1}^{(t)} = \theta_{i,k}^{(t)} - \eta_k^t \left( g_i(\theta_{i,k}^{(t)}) - c_i + m \right)$$

where $c_i$ denotes the client control variable and $m$ denotes the server control variable.

**Client Control Variable Update** (two options, Option II is used in the experiments):

$$c_i^+ \leftarrow c_i - m + \frac{1}{\sum_k \eta_k^t} (\theta_{t-1} - \theta_{i,K}^{(t)})$$

**Server Control Variable Update** (momentum mechanism):

$$m \leftarrow m + \gamma \frac{1}{s} \sum_{i \in \mathcal{S}} (c_i^+ - m)$$

**Key Difference from SCAFFOLD**: SCAFFOLD's global variable $c$ assigns equal weight to new and old $c_i$, leading to an update delay issue. The momentum update of FedMoSWA assigns higher weight to the most recently uploaded $c_i$, effectively mitigating the delay caused by low client participation rates.

### Theoretical Analysis of Generalization Error

A generalization analysis framework for FL is established based on uniform stability:

- **FedSWA Generalization Bound**: $\mathcal{O}\left(\frac{L}{mn\beta} e^{1/T+1} (\tilde{c}L + \tilde{c}\sigma_g + \tilde{c}\sigma)\right)$
- **FedMoSWA Generalization Bound**: $\mathcal{O}\left(\frac{L}{mn\beta} e^{1/T+1} (\tilde{c}L + \sigma_g + \tilde{c}\sigma)\right)$
- **FedSAM Generalization Bound**: $\mathcal{O}\left(\frac{L}{mn\beta} e^{1/T+1} (\bar{c}L + \bar{c}\sigma_g + \bar{c}\sigma)\right)$

where $\tilde{c} = 1 + (2+1/KT)^{K-1}/T \gg 1$ and $\bar{c} > \tilde{c}$. In FedMoSWA, the coefficient before the data heterogeneity term $\sigma_g$ is reduced from $\tilde{c}$ to 1, significantly suppressing the impact of heterogeneity on generalization.

**FedMoSWA Optimization Error (Non-convex)**: $\mathcal{O}\left(\frac{\sigma\sqrt{F}}{\sqrt{TKs}} \sqrt{1+s/\alpha^2} + \frac{\beta F}{T}(m/s)^{2/3}\right)$, which is independent of the data heterogeneity parameter $\sigma_g$.

## Key Experimental Results

### Main Results (Table 2, Dirichlet-0.6)

| Method | CIFAR-10 ResNet-18 | CIFAR-100 ResNet-18 |
|------|-------------------|-------------------|
| FedAvg | 86.0% | 54.2% |
| FedSAM | 83.6% | 51.9% |
| SCAFFOLD | 85.9% | 54.1% |
| MoFedSAM | 87.0% | 60.1% |
| FedSWA | 89.5% | 59.8% |
| **FedMoSWA** | **91.2%** | **67.9%** |

### Performance Under Different Heterogeneity Levels (Table 3, CIFAR-100 ResNet-18)

| Method | Dir-0.1 | Dir-0.3 | Dir-0.6 |
|------|---------|---------|---------|
| FedSAM | 40.1% | 49.0% | 51.9% |
| MoFedSAM | 51.5% | 57.5% | 60.1% |
| FedSWA | 50.3% | 55.5% | 59.8% |
| **FedMoSWA** | **61.9%** | **66.2%** | **67.9%** |

### Tiny ImageNet (ViT-Base)

| Method | Dir-0.1 | Dir-0.3 | Dir-0.6 |
|------|---------|---------|---------|
| FedAvg | 70.9% | 71.8% | 72.8% |
| SCAFFOLD | 71.6% | 72.5% | 73.1% |
| FedSWA | 71.9% | 72.6% | 73.2% |
| **FedMoSWA** | **73.8%** | **74.4%** | **74.7%** |

### Ablation Study

| Ablated Element | Findings |
|--------|------|
| Learning Rate Decay $\rho$ | $\rho=0.1$ is optimal; when $\rho=1, \alpha=1$, FedSWA degenerates to FedAvg, verifying the effectiveness of SWA. |
| EMA Coefficient $\alpha$ | $\alpha=1.5$ is optimal, with excessive values degrading performance. |
| Momentum Parameter $\gamma$ | $\gamma=0.2$ is optimal; $\gamma=0.05$ leads to extremely slow convergence. |
| Control Variable Ablation | When $\rho=1, \alpha=1$, FedMoSWA degenerates to FedAvg with variance reduction (65.9%), which outperforms SCAFFOLD (52.3%). |

### Key Findings

1. **Greater Advantages Under Higher Heterogeneity**: Under Dir-0.1, FedMoSWA outperforms MoFedSAM by 10.4%, compared to a 6.2% advantage under Dir-0.6.
2. **Higher Communication Efficiency**: FedMoSWA achieves the target accuracy with fewer communication rounds (CIFAR-100 Dir-0.6: 330 rounds vs. 603 rounds for MoFedSAM).
3. **Flatter Loss Landscape**: The global training loss surface of FedMoSWA is visibly flatter than all baselines.

## Highlights & Insights

1. **Precise Diagnosis of FedSAM's Failure Mechanism**: It clearly points out that FedSAM identifies locally flat instead of globally flat minima under high heterogeneity, presenting a valuable empirical discovery.
2. **Novel Paradigm of Replacing SAM with SWA**: It successfully migrates SWA from centralized learning to FL, yielding superior computational efficiency (no extra forward/backward passes required).
3. **Theoretical Completeness**: Both generalization bounds and optimization bounds are derived, proving that FedMoSWA strictly outperforms FedSAM under both metrics.
4. **Momentum Update Solving SCAFFOLD's Delay Issue**: The analysis and improvement of the update delay in SCAFFOLD's global variable carry great practical significance.
5. **Comprehensive Experimental Coverage**: Evaluations across 3 datasets, 4 network architectures, 3 levels of heterogeneity, and 10 baselines, totaling over 55 experimental configurations.

## Limitations & Future Work

1. **Limited to Image Classification**: Evaluation is confined to image classification, leaving other FL application domains (such as NLP or recommendation systems) unverified.
2. **Fixed Client Participation Rate of 10%**: The performance under extremely low client participation rates (e.g., 1%) has not been explored.
3. **Communication Overhead**: FedMoSWA requires transmitting extra control variables $c_i^+ - m$, incurring approximately double the communication cost of FedAvg (comparable to SCAFFOLD).
4. **Strong Convexity Assumption**: Certain theoretical generalization bound results rely on the strong convexity assumption, which deviates from the non-convex reality of deep learning.
5. **Hyperparameter Sensitivity**: Tuning three additional hyperparameters ($\alpha, \gamma, \rho$) increases the deployment complexity.

## Related Work & Insights

- **FedSAM / MoFedSAM** (Qu et al., 2022): Pioneering work introducing SAM to local training in FL.
- **SCAFFOLD** (Karimireddy et al., 2020): The origin of the variance reduction control variables concept; FedMoSWA improves its global variable update using momentum.
- **SWA** (Izmailov et al., 2018): The original work on Stochastic Weight Averaging, which FedSWA adapts to FL settings.
- **FedACG** (Kim et al., 2024): An active-gradient-consensus method that serves as a strong runner-up baseline in certain settings.
- **LookAhead** (Zhang et al., 2019): The inspiration behind the EMA aggregation strategy.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to combine SWA and momentum-based variance reduction in FL.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Multi-dataset, multi-network, multi-heterogeneity, thorough ablated studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear theoretical and experimental frameworks, intuitive loss surface visualization.
- Value: ⭐⭐⭐⭐ — Significant generalization improvements in highly heterogeneous FL settings, with better computational efficiency than SAM schemes.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] The Panaceas for Improving Low-Rank Decomposition in Communication-Efficient Federated Learning](the_panaceas_for_improving_low-rank_decomposition_in_communication-efficient_fed.md)
- [\[AAAI 2026\] SMoFi: Step-wise Momentum Fusion for Split Federated Learning on Heterogeneous Data](../../AAAI2026/optimization/smofi_step-wise_momentum_fusion_for_split_federated_learning_on_heterogeneous_da.md)
- [\[AAAI 2026\] Tackling Resource-Constrained and Data-Heterogeneity in Federated Learning with Double-Weight Sparse Pack](../../AAAI2026/optimization/tackling_resource-constrained_and_data-heterogeneity_in_federated_learning_with_.md)
- [\[ICCV 2025\] Federated Prompt-Tuning with Heterogeneous and Incomplete Multimodal Client Data](../../ICCV2025/optimization/federated_prompt-tuning_with_heterogeneous_and_incomplete_multimodal_client_data.md)
- [\[NeurIPS 2025\] Streaming Federated Learning with Markovian Data](../../NeurIPS2025/optimization/streaming_federated_learning_with_markovian_data.md)

</div>

<!-- RELATED:END -->

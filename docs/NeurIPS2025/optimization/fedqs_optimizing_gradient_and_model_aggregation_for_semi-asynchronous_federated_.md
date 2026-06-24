---
title: >-
  [Paper Note] FedQS: Optimizing Gradient and Model Aggregation for Semi-Asynchronous Federated Learning
description: >-
  [NeurIPS 2025][Optimization][Federated Learning] This paper proposes FedQS, the first framework to simultaneously optimize both gradient aggregation and model aggregation strategies in semi-asynchronous federated learning (SAFL). By partitioning clients into four categories and adaptively adjusting training strategies, FedQS achieves comprehensive improvements over baselines in accuracy, convergence speed, and stability.
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "Federated Learning"
  - "Semi-Asynchronous"
  - "Gradient Aggregation"
  - "Model Aggregation"
  - "Divide-and-Conquer"
date: 2026-05-08
content_hash: 9fc71bc24eccdc3b
---

# FedQS: Optimizing Gradient and Model Aggregation for Semi-Asynchronous Federated Learning

**Conference**: NeurIPS 2025
**arXiv**: [2510.07664](https://arxiv.org/abs/2510.07664)  
**Code**: [GitHub](https://github.com/bkjod/FedQS_)  
**Area**: Optimization
**Keywords**: Federated Learning, Semi-Asynchronous, Gradient Aggregation, Model Aggregation, Divide-and-Conquer

## TL;DR

This paper proposes FedQS, the first framework to simultaneously optimize both gradient aggregation and model aggregation strategies in semi-asynchronous federated learning (SAFL). By partitioning clients into four categories and adaptively adjusting training strategies, FedQS achieves comprehensive improvements over baselines in accuracy, convergence speed, and stability.

## Background & Motivation

In federated learning, the semi-asynchronous paradigm (SAFL) strikes a balance between synchronous and asynchronous training, yet faces critical challenges:

**Performance gap between two aggregation strategies**:
   - **Gradient aggregation** (FedSGD): faster convergence and higher accuracy, but severe oscillation
   - **Model aggregation** (FedAvg): more stable, but slower convergence and lower accuracy
   - When stale updates and non-IID data coexist, the accuracy gap between the two surges to 11.52%

**Lack of theoretical understanding**: existing analyses remain empirical

**Limitations of server-side vs. client-side approaches**:
   - Server-side methods: tightly coupled with specific aggregation strategies
   - Client-side methods: lack access to global information

## Method

### Overall Architecture

FedQS comprises three modules:
- **Mod①** (Global Aggregation Estimation): deployed on clients to estimate the global gradient direction
- **Mod②** (Local Training Adaptation): deployed on clients to adjust training strategies according to client type
- **Mod③** (Global Model Aggregation): deployed on the server for adaptive weighted aggregation

### Key Designs

**Mod①: Global Aggregation Estimation**
- Each client stores the two most recent global models and computes a pseudo-global gradient: $L_g(w_g^t) = w_g^t - w_g^{t-1}$
- Computes the local-global gradient similarity $s_i^t$ (e.g., cosine similarity)
- Core innovation: acquires global information from the client perspective, achieving decoupling from the aggregation strategy

**Mod②: Local Training Adaptation (Divide-and-Conquer Strategy)**

Clients are partitioned into four categories based on update frequency $f_i^t$ and gradient similarity $s_i^t$:

| Type | Speed | Similarity | Strategy |
|------|-------|------------|----------|
| FBC (fast but biased) | $f_i^t > \bar{f}^t$ | $s_i^t < \bar{s}^t$ | Maintain learning rate; trigger feedback mechanism to increase aggregation weight |
| FUC (fast and aligned) | $f_i^t > \bar{f}^t$ | $s_i^t > \bar{s}^t$ | Reduce learning rate $\eta_i^t = \eta_i^{t-1} - a\mathcal{F}$; incorporate momentum |
| SUC (slow but aligned) | $f_i^t < \bar{f}^t$ | $s_i^t > \bar{s}^t$ | Increase learning rate $\eta_i^t = \eta_i^{t-1} + a\mathcal{F}$; incorporate momentum |
| SBC (slow and biased) | $f_i^t < \bar{f}^t$ | $s_i^t < \bar{s}^t$ | Increase learning rate; distinguish between straggling and distribution shift via validation set |

**Momentum update formula**:
$$w_{i,e}^t = w_{i,e-1}^t - \eta_i^t \left[\sum_{r=1}^{e}(m_i^t)^r \nabla F_{i,e-r}(w_{i,e-r-1}^t) + \nabla F_{i,e}(w_{i,e-1}^t)\right]$$

**Mod③: Global Model Aggregation**
- Adjusts aggregation weights for clients triggering the feedback mechanism: $p_i = \frac{\exp(\phi - \mathcal{F})}{2\phi - \mathcal{F}} \cdot \frac{(1+\mathcal{G})}{2K}$
- Performs normalized weighted aggregation

### Loss & Training

**Convergence guarantees** (Theorems 4.2 and 4.3):
- Both FedQS-SGD and FedQS-Avg achieve exponential convergence rates
- The convergence bound consists of three terms: $\mathcal{V}^t$ (exponential decay term), $\mathcal{U} = O(\delta^2)$ (data heterogeneity), and $\mathcal{W} = O(G_c^2)$ (gradient variation)
- Assumptions: $L$-smoothness, bounded gradients, bounded heterogeneity

## Key Experimental Results

### Main Results

**Accuracy and convergence speed** (three task types):

| Method | CV (x=0.1) | CV (x=0.5) | CV (x=1) | NLP (R=200) | NLP (R=600) | RWD-Gender | RWD-Ethnicity |
|--------|-----------|-----------|----------|------------|------------|------------|---------------|
| FedAvg | 56.05 | 73.71 | 77.86 | 47.04 | 45.52 | 77.10 | 77.25 |
| M-step | 62.17 | 80.49 | 82.46 | 49.38 | 48.12 | 78.20 | 78.01 |
| **FedQS-Avg** | **63.91** | **80.26** | **82.74** | **50.43** | **50.08** | **78.94** | **78.85** |
| FedSGD | 65.71 | 83.87 | 85.42 | 48.04 | 49.64 | 77.15 | 78.33 |
| WKAFL | 64.66 | 85.14 | 86.02 | 50.49 | 50.09 | 78.96 | 76.97 |
| **FedQS-SGD** | **68.88** | **86.11** | **86.79** | **52.22** | **52.49** | **78.74** | **79.24** |

**Wall-clock time comparison** (seconds):

| Method | CV (x=0.1) | NLP (R=200) | RWD-Gender |
|--------|-----------|------------|------------|
| FedAvg (Sync) | 78,048 | 22,417 | 30,149 |
| FedQS-Avg (SAFL) | 32,827 | 6,023 | 5,701 |
| FedQS-SGD (SAFL) | 32,784 | 5,248 | 5,523 |

Compared to synchronous baselines, FedQS reduces wall-clock time by approximately 70% on average.

### Ablation Study

**Module ablation** (average over CV tasks):

| Module | Configuration | Avg Accuracy | SGD Accuracy | Avg Convergence | SGD Convergence |
|--------|--------------|-------------|-------------|----------------|----------------|
| Mod① | Cosine | 74.14 | 80.59 | 251 | 230 |
| Mod① | Euclidean | 75.69 | 79.55 | 244 | 232 |
| Mod① | Manhattan | 76.56 | 80.28 | 228 | 221 |
| Mod② | w/o momentum | 73.21 | 78.88 | 269 | 242 |
| Mod② | with momentum | 74.14 | 80.59 | 251 | 230 |
| Mod③ | w/o feedback | 68.35 | 78.83 | 284 | 268 |
| Mod③ | with feedback | 74.14 | 80.59 | 251 | 230 |

**Key ablation findings**:
- Removing momentum: average accuracy drops by 4.3%, requiring 6% more epochs to converge
- Removing the feedback mechanism: FedQS-Avg accuracy drops sharply by 7.81%
- Choice of similarity function has a relatively minor impact (cosine/Euclidean/Manhattan differences are small)

**Robustness to system configurations**:

| Scenario | FedAvg | FedQS-Avg | FedSGD | FedQS-SGD |
|----------|--------|-----------|--------|-----------|
| N=50, 1:20 | 70.1 | 79.2 | 77.4 | **80.7** |
| N=200, 1:100 | 49.4 | 64.7 | 74.4 | **80.1** |

Under extreme heterogeneity (200 clients, speed ratio 1:100), FedQS-SGD still achieves 80.1%.

### Key Findings

1. **Root cause of the gradient–model aggregation gap**: stale updates in gradient aggregation affect only the direction and magnitude, whereas in model aggregation they reset the optimization trajectory; non-IID data further amplifies this gap.
2. **Effectiveness of the divide-and-conquer strategy**: the four-category client partition covers all combinations of heterogeneity, each addressed by a targeted optimization strategy.
3. **Complementarity of momentum and feedback mechanism**: momentum accelerates local convergence while the feedback mechanism improves global aggregation.
4. **Hyperparameter sensitivity**: $a$ (learning rate adjustment rate) has the largest impact; $k$ (momentum adjustment speed) has the smallest.

## Highlights & Insights

1. **First unified framework**: simultaneously optimizes both gradient and model aggregation strategies rather than focusing on one alone.
2. **Client-side adaptivity**: does not require the server to have prior knowledge of client characteristics; clients can dynamically adjust strategies in response to changing resources.
3. **Theoretical guarantees**: provides exponential convergence proofs for both aggregation strategies.
4. **Negligible additional overhead**: clients only require one additional similarity computation and two comparisons; communication overhead increases by only a 1-bit signal and a few floating-point values.
5. **Broad experimental coverage**: three task types — CV (CIFAR-10), NLP (Shakespeare), and real-world data (UCI Adult).

## Limitations & Future Work

1. The model aggregation mode introduces a degree of oscillation.
2. Three new hyperparameters ($a, m_0, k$) are introduced, increasing implementation and reproduction complexity.
3. Experiments are limited to medium-scale models (ResNet-18, LSTM, FCN); applicability to large-scale models remains unverified.
4. Automatic hyperparameter tuning mechanisms could be explored in future work.

## Related Work & Insights

- WKAFL employs cosine similarity for weighted aggregation of stale gradients, inspiring Mod① of FedQS.
- FedAT adopts a hierarchical asynchronous framework but requires prior knowledge of client performance distributions.
- The gradient correction idea from SCAFFOLD is applied in FedAC; FedQS addresses the problem from an orthogonal perspective via divide-and-conquer.
- The role of momentum in federated optimization is further validated.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to jointly optimize both aggregation strategies
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three task types + 8 baselines + ablations + hyperparameter analysis + system configuration analysis
- Practicality: ⭐⭐⭐⭐ — Low additional overhead, good scalability
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with rich figures and tables

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Layer-wise Update Aggregation with Recycling for Communication-Efficient Federated Learning](layer-wise_update_aggregation_with_recycling_for_communication-efficient_federat.md)
- [\[ICLR 2026\] Beyond Aggregation: Guiding Clients in Heterogeneous Federated Learning](../../ICLR2026/optimization/beyond_aggregation_guiding_clients_in_heterogeneous_federated_learning.md)
- [\[NeurIPS 2025\] Streaming Federated Learning with Markovian Data](streaming_federated_learning_with_markovian_data.md)
- [\[ICLR 2026\] Byzantine-Robust Federated Learning with Learnable Aggregation Weights](../../ICLR2026/optimization/byzantine-robust_federated_learning_with_learnable_aggregation_weights.md)
- [\[CVPR 2025\] Model Poisoning Attacks to Federated Learning via Multi-Round Consistency](../../CVPR2025/optimization/model_poisoning_attacks_to_federated_learning_via_multi-round_consistency.md)

</div>

<!-- RELATED:END -->

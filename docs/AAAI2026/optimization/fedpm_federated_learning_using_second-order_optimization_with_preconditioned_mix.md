---
title: >-
  [Paper Note] FedPM: Federated Learning Using Second-order Optimization with Preconditioned Mixing of Local Parameters
description: >-
  [AAAI 2026][Optimization][Federated Learning] This paper proposes FedPM (Federated Preconditioned Mixing), a novel federated learning method that replaces the conventional simple parameter averaging on the server side wi…
tags:
  - "AAAI 2026"
  - "Optimization"
  - "Federated Learning"
  - "Second-order Optimization"
  - "Preconditioned Mixing"
  - "Data Heterogeneity"
  - "Convergence Analysis"
date: 2026-05-08
content_hash: f82e616deeb9a883
---

# FedPM: Federated Learning Using Second-order Optimization with Preconditioned Mixing of Local Parameters

**Conference**: AAAI 2026  
**arXiv**: [2511.09100](https://arxiv.org/abs/2511.09100)  
**Code**: [https://github.com/rioyokotalab/fedpm](https://github.com/rioyokotalab/fedpm)  
**Area**: Optimization  
**Keywords**: Federated Learning, Second-order Optimization, Preconditioned Mixing, Data Heterogeneity, Convergence Analysis

## TL;DR

This paper proposes FedPM (Federated Preconditioned Mixing), a novel federated learning method that replaces the conventional simple parameter averaging on the server side with "preconditioned mixing," addressing the local preconditioner drift problem inherent in existing second-order federated optimization methods. FedPM is theoretically shown to achieve superlinear convergence for strongly convex objectives and empirically outperforms existing methods under heterogeneous data settings.

## Background & Motivation

Federated learning (FL) methods can be categorized along two dimensions into four classes:

| Category | Optimization Order | Server Operation | Representative Methods |
|----------|--------------------|------------------|------------------------|
| FOGM | First-order | Gradient mixing | PSGD |
| FOPM | First-order | Parameter mixing | FedAvg, FedProx, SCAFFOLD |
| SOGM | Second-order | Gradient mixing | FedNL, FedNew, FedNS |
| SOPM | Second-order | Parameter mixing | LocalNewton, LTDA, FedSophia |

Existing SOPM methods share a **fundamental limitation**: simple mixing (i.e., plain averaging) of local parameters on the server **does not correspond to a genuine global second-order update**.

Concretely, with a single local update step ($K=1$), the global update of SOGM is:

$$\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} - \eta \underbrace{\left(\frac{1}{N}\sum_{i=1}^N \nabla^2 f_i(\boldsymbol{\theta}^{(t)})\right)^{-1}}_{\text{global preconditioner}} \underbrace{\frac{1}{N}\sum_{i=1}^N \nabla f_i(\boldsymbol{\theta}^{(t)})}_{\text{global gradient}}$$

whereas SOPM under the same $K=1$ setting reduces to:

$$\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} - \frac{\eta}{N}\sum_{i=1}^N \underbrace{(\nabla^2 f_i(\boldsymbol{\theta}^{(t)}))^{-1}}_{\text{local preconditioner}} \underbrace{\nabla f_i(\boldsymbol{\theta}^{(t)})}_{\text{local gradient}}$$

The critical distinction is that SOPM aggregates **locally preconditioned gradients** rather than applying a **global preconditioner to the global gradient**. Under data heterogeneity, the local curvature (each client's Hessian) is a poor approximation of the global curvature, causing **local preconditioner drift** that severely impedes convergence.

## Method

### Overall Architecture

The core idea of FedPM is to decompose the ideal global second-order update (Eq. 6) into client-side local parameter updates and server-side parameter mixing. This decomposition naturally gives rise to "preconditioned mixing"—a curvature-aware parameter aggregation scheme that replaces conventional simple averaging.

### Key Designs

#### 1. **Derivation of Preconditioned Mixing**: Exact Decomposition from the Global Second-order Update

**Core derivation** (Eq. 8): Starting from the ideal global second-order update:

$$\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} - \eta (\boldsymbol{P}^{(t)})^{-1}\boldsymbol{g}^{(t)}$$

By inserting the identity $(\boldsymbol{P}^{(t)})^{-1}\boldsymbol{P}^{(t)}$ and expanding via $\frac{1}{N}\sum_{i=1}^N \boldsymbol{P}_i^{(t)}(\boldsymbol{P}_i^{(t)})^{-1}$, algebraic manipulation yields:

$$\boldsymbol{\theta}^{(t+1)} = \frac{1}{N}\sum_{i=1}^N (\boldsymbol{P}^{(t)})^{-1}\boldsymbol{P}_i^{(t)} \underbrace{(\boldsymbol{\theta}^{(t)} - \eta(\boldsymbol{P}_i^{(t)})^{-1}\boldsymbol{g}_i^{(t)})}_{\text{client-side local second-order update}}$$

where $\boldsymbol{P}^{(t)} = \frac{1}{N}\sum_i \boldsymbol{P}_i^{(t)}$ (global preconditioner) and $\boldsymbol{P}_i^{(t)} = \nabla^2 f_i(\boldsymbol{\theta}^{(t)})$ (local preconditioner).

**FedPM single-step update rule**:

- **Client**: $\boldsymbol{\theta}_i^{(t+1)} = \boldsymbol{\theta}^{(t)} - \eta(\boldsymbol{P}_i^{(t)})^{-1}\nabla f_i(\boldsymbol{\theta}^{(t)})$
- **Server**: $\boldsymbol{P}^{(t)} = \frac{1}{N}\sum_i \boldsymbol{P}_i^{(t)}$

  $\boldsymbol{\theta}^{(t+1)} = \frac{1}{N}\sum_{i=1}^N \underbrace{(\boldsymbol{P}^{(t)})^{-1}\boldsymbol{P}_i^{(t)}\boldsymbol{\theta}_i^{(t+1)}}_{\text{preconditioned mixing}}$

**Distinction from simple mixing**: Simple mixing computes $\frac{1}{N}\sum_i \boldsymbol{\theta}_i$, whereas preconditioned mixing computes $\frac{1}{N}\sum_i (\boldsymbol{P}^{(t)})^{-1}\boldsymbol{P}_i^{(t)}\boldsymbol{\theta}_i$. The latter reweights each client's parameters by the product of the inverse global preconditioner and the local preconditioner, thereby incorporating global curvature information.

**Design Motivation**: This is not a heuristic design but follows directly from an exact mathematical decomposition of the ideal global optimization update. At $K=1$, the global update of FedPM is equivalent to the ideal second-order update.

#### 2. **Extension to Multiple Local Steps**: Natural Generalization to $K>1$ for Communication Efficiency

The single-step decomposition generalizes naturally to multiple local update steps:

- **Client**: Performs $K$ local second-order updates, computing $\boldsymbol{P}_i^{(t,k)} = \nabla^2 f_i(\boldsymbol{\theta}_i^{(t,k)})$ and updating parameters at each step.
- **Server**: $\boldsymbol{P}^{(t)} = \frac{1}{N}\sum_i \boldsymbol{P}_i^{(t,K-1)}$ (using the preconditioner from the last local step)

  $\boldsymbol{\theta}^{(t+1)} = \frac{1}{N}\sum_{i=1}^N (\boldsymbol{P}^{(t)})^{-1}\boldsymbol{P}_i^{(t,K-1)}\boldsymbol{\theta}_i^{(t,K)}$

**Additional communication cost**: Unlike conventional SOPM, FedPM requires transmitting both local parameters and local preconditioners. However, the performance gains under data heterogeneity justify this overhead.

#### 3. **FOOF Preconditioner Approximation**: Efficient Implementation for Large-scale DNNs

Computing the full Hessian is infeasible for DNNs due to $\mathcal{O}(d^2)$ storage. FedPM therefore employs FOOF (Fisher Information Matrix approximation):

- The FIM is approximated as a block-diagonal matrix, with each block corresponding to one layer of the DNN.
- Each block is further approximated by the uncentered input covariance matrix $\boldsymbol{A}_{i,l}^{(t,k)}$ of the corresponding layer.
- Update rule: $\boldsymbol{\theta}_{i,l}^{(t,k+1)} = \boldsymbol{\theta}_{i,l}^{(t,k)} - \eta \text{vec}((\boldsymbol{A}_{i,l}^{(t,k)})^{-1}\text{vec}^{-1}(\boldsymbol{g}_{i,l}^{(t,k)}))$

| Operation | FedPM (Full Hessian) | FedPM + FOOF |
|-----------|----------------------|--------------|
| Construction | $\mathcal{O}(Md^2)$ | $\mathcal{O}(Md)$ |
| Inversion | $\mathcal{O}(d^3)$ | $\mathcal{O}(d\sqrt{d/L})$ |
| Communication | $\mathcal{O}(d^2)$ | $\mathcal{O}(d)$ |

### Theoretical Convergence Analysis

**Theorem 1** (informal): Under assumptions of strong convexity and Hessian smoothness, when the initial parameters are sufficiently close to the optimum, FedPM with single local update steps ($K=1$) achieves **superlinear convergence**.

The analysis follows the strategy of FedNL, with one key distinction: FedPM conditions on the **most recent Hessian**, which corresponds more directly to Newton's method than the stale Hessian used in the original FedNL proof.

## Key Experimental Results

### Test 1: Strongly Convex Models (Theoretical Validation)

L2-regularized logistic regression on the w8a ($d=300$, $N=142$ clients) and a9a ($d=123$, $N=80$ clients) datasets.

Results:
- FedPM and FedNL significantly outperform all other methods.
- The rapid accelerating decrease in parameter distance $\|\boldsymbol{\theta}^{(t)} - \boldsymbol{\theta}^*\|$ is consistent with the superlinear convergence theory.
- FedPM ($K=1$) is equivalent to FedNL in its global update, yielding nearly identical results (minor discrepancies attributable to numerical precision).

### Main Results — Test 2: Non-convex DNN Models

| Method | CIFAR10 α=1.0 | CIFAR10 α=0.1 | CIFAR100 α=1.0 | CIFAR100 α=0.1 |
|--------|--------------|--------------|----------------|----------------|
| FedAvg | 70.1±1.3 | 63.1±2.1 | 59.8±0.7 | 52.4±0.9 |
| FedAvgM | 72.6±1.5 | 64.8±3.0 | 61.2±0.2 | 54.9±0.6 |
| FedProx | 70.1±1.1 | 62.8±1.3 | 56.6±0.7 | 50.7±0.9 |
| SCAFFOLD | 71.4±0.5 | 65.3±2.7 | 64.7±1.1 | 58.2±0.6 |
| FedAdam | 70.4±1.6 | 64.6±2.8 | 58.3±0.3 | 51.9±0.4 |
| LocalNewton | 73.4±1.4 | 62.4±2.3 | 70.2±0.3 | 61.9±0.6 |
| **FedPM** | **74.8±0.5** | **68.6±2.0** | **68.4±2.2** | **63.1±0.8** |

Setup: Simple CNN for CIFAR10 and ResNet18 (BatchNorm replaced with GroupNorm) for CIFAR100; $N=10$ clients, 5 local epochs, 100 communication rounds. $\alpha$ is the Dirichlet distribution parameter; smaller $\alpha$ indicates stronger heterogeneity.

### Ablation Study

**Effect of the number of local epochs** (CIFAR10, $\alpha=0.1$, total local epochs fixed at 500):

| Configuration | Communication Rounds | Local Epochs | FedAvg | SCAFFOLD | LocalNewton | FedPM |
|---------------|----------------------|--------------|--------|----------|-------------|-------|
| Low comm. frequency | 50 | 10 | ~60% | ~62% | ~60% | ~66% |
| Medium comm. frequency | 100 | 5 | ~63% | ~65% | ~62% | ~69% |
| High comm. frequency | 500 | 1 | ~67% | ~68% | ~69% | ~72% |

FedPM consistently outperforms all baselines across all configurations.

### Key Findings

1. **FedPM yields the largest gains under high heterogeneity**: At $\alpha=0.1$, FedPM outperforms LocalNewton by +6.2% on CIFAR10, compared to only +1.4% at $\alpha=1.0$, confirming the robustness of preconditioned mixing to data heterogeneity.
2. **LocalNewton degrades severely under heterogeneous data**: At $\alpha=0.1$, LocalNewton even underperforms some first-order methods, directly illustrating the local preconditioner drift problem.
3. **FedPM converges faster in both communication rounds and wall-clock time**: Owing to the efficient preconditioner approximation and reduced communication round requirements.
4. **The FOOF approximation is effective**: Under FOOF, FedPM still significantly outperforms LocalNewton with full Hessian.
5. **Superlinear convergence is empirically validated under strong convexity**: The accelerating decrease in parameter distance matches theoretical predictions.

## Highlights & Insights

1. **Precise problem identification**: The paper clearly identifies the core flaw of existing SOPM methods—simple mixing yields a sum of locally preconditioned gradients rather than a globally preconditioned global gradient.
2. **Elegant derivation**: Starting from a simple identity transformation, the global second-order update is exactly decomposed into a client–server update rule.
3. **Unified theory and practice**: The superlinear convergence theory is validated experimentally, and the FOOF approximation makes large-scale deployment feasible.
4. **Taxonomic contribution**: FL methods are systematically organized into four categories (FOGM/FOPM/SOGM/SOPM), and FedPM is shown to be the only SOPM method that achieves genuine global second-order optimization.

## Limitations & Future Work

1. **Additional communication overhead**: Transmitting local preconditioners (or FOOF matrices) increases communication cost.
2. **Limited theoretical coverage**: Convergence proofs are restricted to strong convexity with single-step updates; theoretical guarantees for $K>1$ are absent.
3. **Limited model scale in experiments**: Validation is confined to a simple CNN and ResNet18; performance on large models such as Transformers remains untested.
4. **Privacy considerations**: Transmitting preconditioners may leak additional information about local data.
5. **Client sampling**: The main experiments assume full client participation; the behavior under partial participation in practical settings warrants further investigation.

## Related Work & Insights

- Complementary to FedNL/FedNew/FedNS (SOGM-class methods): FedPM achieves equivalent global second-order optimization within the SOPM framework while maintaining communication efficiency.
- The preconditioned mixing paradigm may generalize to other distributed optimization settings, such as decentralized learning.
- The successful use of the FOOF approximation opens a path toward deploying large-scale second-order methods in federated learning.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Precisely identifies the problem and provides an elegant solution; the derivation of preconditioned mixing is concise and insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Strong convex validation, DNN experiments, and ablation studies are thorough, though model scale is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ — Problem motivation, taxonomy, derivations, and experimental organization are presented with exceptional clarity.
- Value: ⭐⭐⭐⭐ — Provides significant theoretical and practical contributions to second-order optimization in federated learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Tackling Resource-Constrained and Data-Heterogeneity in Federated Learning with Double-Weight Sparse Pack](tackling_resource-constrained_and_data-heterogeneity_in_federated_learning_with_.md)
- [\[NeurIPS 2025\] Second-Order Optimization Under Heavy-Tailed Noise: Hessian Clipping and Sample Complexity](../../NeurIPS2025/optimization/second-order_optimization_under_heavy-tailed_noise_hessian_clipping_and_sample_c.md)
- [\[AAAI 2026\] Data Heterogeneity and Forgotten Labels in Split Federated Learning](data_heterogeneity_and_forgotten_labels_in_split_federated_learning.md)
- [\[AAAI 2026\] SMoFi: Step-wise Momentum Fusion for Split Federated Learning on Heterogeneous Data](smofi_step-wise_momentum_fusion_for_split_federated_learning_on_heterogeneous_da.md)
- [\[ICLR 2026\] DeepAFL: Deep Analytic Federated Learning](../../ICLR2026/optimization/deepafl_deep_analytic_federated_learning.md)

</div>

<!-- RELATED:END -->

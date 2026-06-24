---
title: >-
  [Paper Note] FEDTAIL: Federated Long-Tailed Domain Generalization with Sharpness-Guided Gradient Matching
description: >-
  [ICML 2025][federated learning] FedTAIL proposes a federated domain generalization framework that simultaneously addresses the dual challenges of domain shift and long-tailed class imbalance through three modules: gradient coherence regularization, class-wise sharpness-aware minimization, and curvature-aware dynamic weighting, achieving SOTA performance on multiple benchmarks.
tags:
  - "ICML 2025"
  - "federated learning"
  - "Domain Generalization"
  - "Long-Tailed"
  - "Sharpness-Aware Minimization"
  - "Gradient Coherence"
date: 2026-05-08
content_hash: ffb28cdd5b85c90f
---

# FEDTAIL: Federated Long-Tailed Domain Generalization with Sharpness-Guided Gradient Matching

**Conference**: ICML 2025  
**arXiv**: [2506.08518](https://arxiv.org/abs/2506.08518)  
**Code**: [https://github.com/sunnyinAI/FedTail](https://github.com/sunnyinAI/FedTail)  
**Area**: LLM Evaluation  
**Keywords**: federated learning, Domain Generalization, Long-Tailed, Sharpness-Aware Minimization, Gradient Coherence

## TL;DR
FedTAIL proposes a federated domain generalization framework that simultaneously addresses the dual challenges of domain shift and long-tailed class imbalance through three modules: gradient coherence regularization, class-wise sharpness-aware minimization, and curvature-aware dynamic weighting, achieving SOTA performance on multiple benchmarks.

## Background & Motivation
**Background**: Domain Generalization (DG) aims to train models that can generalize to unseen target domains. Sharpness-Aware Minimization (SAM) improves generalization by seeking flat minima.

**Limitations of Prior Work**: Standard SAM operates globally, neglecting curvature differences across classes, which may cause tail classes to converge to saddle points in long-tailed scenarios. Furthermore, gradients of classification loss and adversarial domain alignment loss may conflict.

**Key Challenge**: In federated scenarios, data is naturally non-i.i.d. and exhibits long-tailed distributions, simultaneously suffering from both domain shift and class imbalance.

**Key Insight**: Unifying gradient coordination, class-aware regularization, and conditional distribution alignment into a scalable framework.

**Core Idea**: Compute class-wise SAM perturbations $\epsilon_c$ and dynamically weight them by the inverse of the maximum eigenvalue of the class Hessian.

## Method

### Overall Architecture
Feature extractor $F_\theta$ + classifier $T_\phi$ + domain discriminator $D_\psi$, trained federatedly across multiple clients. The total loss is: $\mathcal{L}_{\text{FedTAIL}} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{adv}} + \mathcal{L}_{\text{sharp-er}} + \sum_c \gamma_c \mathcal{L}_c + \mathcal{L}_{\text{coh}}$

### Key Designs

1. **Gradient Coherence Regularization**:

    - **Function**: Alleviate gradient conflicts between classification and adversarial domain alignment.
    - **Mechanism**: $\mathcal{L}_{\text{coh}} = -\alpha \langle \nabla_\theta \mathcal{L}_{\text{cls}}, \nabla_\theta \mathcal{L}_{\text{adv}} \rangle$, punishing the negative inner product of the two gradient directions.
    - **Design Motivation**: Ensure that domain alignment does not compromise classification performance.

2. **Class-wise Sharpness-Aware Minimization (Class-wise SAM)**:

    - **Function**: Compute SAM perturbations separately for each class.
    - **Mechanism**: $\epsilon_c = \rho \cdot \nabla_\theta \mathcal{L}_c / \|\nabla_\theta \mathcal{L}_c\|_2$, then $\mathcal{L}_{\text{sharp}} = \sum_c \mathbb{E}_{(x,y=c)}[\ell(h_{\theta+\epsilon_c}(x), y)]$.
    - Introducing curvature-aware weights: $\gamma_c = 1/(1 + \sigma_{\max}(\nabla^2 \mathcal{L}_c))$, where larger curvature (high-frequency/tail classes) $\rightarrow$ larger weight.
    - **Design Motivation**: Global SAM fails to capture differences between classes, and tail classes require more attention.

3. **Sharpness-Aware Conditional Distribution Alignment (Sharpness-Aware ER)**:

    - **Function**: Inject SAM perturbations into entropy regularization.
    - **Mechanism**: $\mathcal{L}_{\text{sharp-er}} = \sum_i \text{KL}(P_i(Y|F(X)) \| Q_T(Y|F(X+\epsilon)))$.
    - **Design Motivation**: Traditional entropy regularization amplifies the gradients of easily transferrable samples while neglecting hard samples.

### Loss & Training
Federated Averaging (FedAvg) aggregates client updates, and each client locally computes gradients, class-wise perturbations, and sharpness-aware updates.

## Key Experimental Results

### Main Results

| Dataset | Metric | FedTAIL | Prev. SOTA | Gain |
|--------|------|---------|----------|------|
| PACS | Avg Acc | 88.9% | 87.6% (SAMALTDG) | +1.3% |
| OfficeHome | Avg Acc | 71.4% | 69.8% | +1.6% |
| Digits-DG | Avg Acc | 88.5% | 86.9% | +1.6% |
| mini-DomainNet | Avg Acc | 73.2% | 71.5% | +1.7% |

### Ablation Study

| Configuration | PACS Avg | Description |
|------|----------|------|
| Full FedTAIL | 88.9% | Complete model |
| w/o Gradient Coherence | 87.1% | Without gradient coherence, drops by 1.8% |
| w/o Class-wise SAM | 87.5% | Without class-wise SAM, drops by 1.4% |
| w/o Curvature Weighting | 88.0% | Without curvature weighting, drops by 0.9% |

### Key Findings
- Gradient coherence regularization contributes the most, indicating that the classification-adversarial gradient conflict is the primary bottleneck.
- Class-wise SAM yields more pronounced effects under severe long-tailed imbalance.
- The approach is effective in both federated and centralized settings.

## Highlights & Insights
- Connects the gradient flow analysis of entropy regularization with the long-tailed distribution problem, revealing the mechanism where high-confidence samples dominate gradients.
- The curvature-aware weight $\gamma_c$ utilizes the maximum eigenvalue of the Hessian to automatically identify undertrained tail classes.
- The modules of the framework are decoupled and can be flexibly combined.

## Limitations & Future Work
- The computation of the maximum eigenvalue of the Hessian is costly, which is not discussed in detail regarding efficiency.
- Experiments are mainly conducted on medium and small-scale datasets; large-scale federated scenarios remain to be validated.
- It assumes that all clients share the same model architecture, leaving heterogeneous scenarios unaddressed.

## Rating
- Novelty: ⭐⭐⭐⭐ Innovation lies in the multi-module combination, though individual modules are incremental.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive benchmarks and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear description of the methodology.
- Value: ⭐⭐⭐⭐ High practical value in the crossed scenario of federated + long-tailed + DG.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Gradient-Guided Annealing for Domain Generalization](../../CVPR2025/others/gradient-guided_annealing_for_domain_generalization.md)
- [\[ICML 2025\] Set-Valued Predictions for Robust Domain Generalization](set_valued_predictions_for_robust_domain_generalization.md)
- [\[NeurIPS 2025\] Sharpness-Aware Minimization with Z-Score Gradient Filtering](../../NeurIPS2025/others/sharpness-aware_minimization_with_z-score_gradient_filtering.md)
- [\[ICLR 2026\] Noise-Aware Generalization: Robustness to In-Domain Noise and Out-of-Domain Generalization](../../ICLR2026/others/noise-aware_generalization_robustness_to_in-domain_noise_and_out-of-domain_gener.md)
- [\[ICML 2025\] Learning Distances from Data with Normalizing Flows and Score Matching](learning_distances_from_data_with_normalizing_flows_and_score_matching.md)

</div>

<!-- RELATED:END -->

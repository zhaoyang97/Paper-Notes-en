---
title: >-
  [Paper Note] Towards Source-Free Machine Unlearning
description: >-
  [CVPR 2025][AI Safety][Machine Unlearning] This paper proposes a source-free machine unlearning algorithm. In the absence of original training data, it approximates the Hessian matrix of the remaining data using only the forget data and the trained model, achieving efficient unlearning for linear and mixed linear classifiers with rigorous theoretical upper-bound guarantees.
tags:
  - "CVPR 2025"
  - "AI Safety"
  - "Machine Unlearning"
  - "Data Privacy"
  - "Source-Free Unlearning"
  - "Hessian Estimation"
  - "Theoretical Guarantees"
date: 2026-05-08
content_hash: 27155204ba4653bc
---

# Towards Source-Free Machine Unlearning

**Conference**: CVPR 2025  
**arXiv**: [2508.15127](https://arxiv.org/abs/2508.15127)  
**Code**: [https://github.com/UCR-Vision-and-Learning-Group/mixed-linear-forgetting](https://github.com/UCR-Vision-and-Learning-Group/mixed-linear-forgetting)  
**Area**: AI Safety / Machine Unlearning  
**Keywords**: Machine Unlearning, Data Privacy, Source-Free Unlearning, Hessian Estimation, Theoretical Guarantees

## TL;DR

This paper proposes a source-free machine unlearning algorithm. In the absence of original training data, it approximates the Hessian matrix of the remaining data using only the forget data and the trained model, achieving efficient unlearning for linear and mixed linear classifiers with rigorous theoretical upper-bound guarantees.

## Background & Motivation

**Background**: With the implementation of data protection regulations such as GDPR, removing the influence of specific data from trained models (the "right to be forgotten") has become a core requirement in machine learning. Machine unlearning aims to efficiently modify model parameters to forget designated data while preserving performance on the remaining data. Existing methods are divided into exact unlearning (e.g., SISA sharded training) and approximate unlearning (e.g., influence function-based parameter updates).

**Limitations of Prior Work**: Almost all existing unlearning methods assume access to full or partial original training data. However, in practical scenarios, model owners may no longer retain the original training data due to storage costs and privacy concerns. Existing "zero-shot unlearning" methods have severe limitations: (1) the method by Chundawat et al. can only forget entire classes and cannot target specific instances; (2) the instance-level unlearning method by Cha et al. suffers from sharp performance degradation as the number of forgotten samples increases, showing poor scalability; (3) most crucially, none of the existing zero-shot unlearning methods provide theoretical guarantees for the unlearning effect.

**Key Challenge**: Influence function-based unlearning (Newton update step) $w_{uf} = w^* + H_r^{-1}\nabla_f + \sigma^2\varepsilon$ requires computing the Hessian matrix $H_r$ of the remaining data. However, under the source-free setting, the remaining data is inaccessible, and $H_r$ cannot be directly computed.

**Goal**: To approximate the Hessian $H_r$ of the remaining data given only the forget data $\mathcal{D}_f$ and the trained model parameters $w^*$, thereby achieving efficient machine unlearning with theoretical guarantees.

**Key Insight**: Leveraging the Taylor expansion of the loss function around the optimal point and a key assumption—under tiny perturbations, the difference in loss variations between the remaining and forget data is bounded ($|\delta\mathcal{L}_r(w_i) - \delta\mathcal{L}_f(w_i)| \leq \epsilon$).

**Core Idea**: By generating small perturbations around the optimal point and using the loss difference of the forget data as a proxy for that of the remaining data, the Hessian estimation problem is formulated as a semidefinite programming (SDP) optimization problem.

## Method

### Overall Architecture

Given a trained optimal classifier $w^*$ and forget data $\mathcal{D}_f$: (1) generate $m$ small perturbations $w_i = w^* + (\delta w)_i$ around $w^*$; (2) compute the loss difference for the forget data $\delta\mathcal{L}_f(w_i)$ for each perturbation; (3) solve for the Hessian estimation $\hat{H}_r$ by minimizing the objective function $\tilde{\Psi}(H_r)$; (4) execute the Newton update step using the estimated Hessian to complete unlearning.

### Key Designs

1. **Retain Hessian Estimation**:

    - Function: Estimate the Hessian matrix corresponding to the remaining data without accessing it
    - Mechanism: Perform a Taylor expansion on the convex loss function to obtain $\delta\mathcal{L}_r \approx \nabla_r(w^*)^\top \delta w + \frac{1}{2}(\delta w)^\top H_r(w^*) \delta w$. Since training converges to the optimal point $\nabla(w^*) = 0$, it holds that $\nabla_r(w^*) = -\nabla_f(w^*)$ (which can be computed from the forget data). Then, the objective function $\tilde{\Psi}(H_r) = \frac{1}{m}\sum_{i=1}^m (\tilde{f}_i(H_r))^2$ is constructed, substituting the incomputable $\delta\mathcal{L}_r(w_i)$ with $\delta\mathcal{L}_f(w_i)$. This is ultimately transformed into an SDP optimization problem under a PSD constraint.
    - Design Motivation: The Newton update step is the theoretically optimal unlearning method but requires the Hessian of the remaining data. This method achieves zero-shot estimation by exploiting the proximity of the loss developments between the remaining and forget data under small perturbations (both bounded by $L\|\delta w\|$).

2. **Theoretical Guarantees Framework**:

    - Function: Provide a rigorous upper bound on the gap between the estimated Hessian and the true Hessian
    - Mechanism: Lemma 1 proves that $\|\Delta H_r\|_F \leq \frac{2\epsilon\sqrt{d}}{2+d}$, where $\epsilon$ is the upper bound of the estimation error of loss differences and $d$ is the feature dimension. Notably, this upper bound decreases as the dimension $d$ increases, meaning the Hessian estimation becomes more accurate in high-dimensional settings. Theorem 1 further provides the upper bound of the gradient norm of the unlearned model on the remaining data: $\|\nabla\mathcal{L}(w_{uf}, \mathcal{D}_r)\|_2 \leq \frac{4\gamma C^2 n_f^2(n-n_f)}{[\lambda(n-n_f) - \frac{2\epsilon}{2+d}]^2}$
    - Design Motivation: The lack of theoretical guarantees is the main weakness of existing zero-shot unlearning methods. The theoretical upper bound provides confidence in the reliability of unlearning and privacy compliance.

3. **Mixed Linear Unlearning**:

    - Function: Extend the unlearning method of linear classifiers to deep neural networks
    - Mechanism: Leverage Neural Tangent Kernel (NTK) theory to linearize the last few layers of the network via a first-order Taylor expansion: $\mathcal{L}(w) = \sum_{i=1}^n \|f_{w_c^*}(x_i) + \nabla_w f_{w_c^*}(x_i) \cdot w - y_i\|_2^2 + \frac{\lambda n}{2}\|w\|_2^2$. This linearization makes the loss function convex, allowing direct application of the proposed Hessian estimation and unlearning algorithms.
    - Design Motivation: Handling non-convex losses in deep networks directly is highly challenging, but NTK-based linearization transforms the problem into a convex optimization task, retaining the validity of the theoretical guarantees.

### Loss & Training

- Convex loss functions with $\ell_2$ regularization (quadratic loss is used for experiments)
- Hessian estimation is solved via SDP optimization
- Each element of the perturbation $\delta w$ is sampled from a standard Gaussian distribution $\mathcal{N}(0, 1)$
- Gaussian noise $\sigma^2\varepsilon$ is added after unlearning to prevent information leakage

## Key Experimental Results

### Main Results

CIFAR-10 linear classifier (forgetting 10% of training data):

| Method | Test Accuracy | Retain Accuracy | Forget Accuracy | MIA |
|------|-----------|-------------|-------------|-----|
| Retrained (Target) | 72% | 74% | 72% | 50% |
| NegGrad | 51.9% | 53.2% | 51.2% | 48% |
| Random Labels | 20.6% | 21.6% | 21.4% | 47% |
| JiT | 52.1% | 53.1% | 51.1% | 49.1% |
| Adversarial | 51.5% | 52.7% | 51.0% | 50.0% |
| **Unlearned (-) (Ours)** | **70%** | **71%** | **68%** | **51.4%** |

Ours significantly outperforms all source-free baselines, approaching the ideal retraining performance.

### Ablation Study

Impact of forget data ratio (CIFAR-10):

| Forget Ratio | Test Performance Gap | Retain Gap | MIA Gap |
|---------|-----------|-----------|--------|
| 5% | 0% | 0% | 0% |
| 10% | 2% | 3% | 1.4% |
| 15% | 14% | 15% | 5.8% |

Impact of the number of perturbations:

| Number of Perturbations | Test Performance Gap | MIA Gap |
|---------|-----------|--------|
| 250 | 15% | 6.2% |
| 500 | 2% | 1.4% |
| 1000 | 0% | 1% |

### Key Findings

- At a 5% forget ratio, the unlearned model is identical to the retrained model (0% performance gap).
- Increasing the number of perturbations significantly improves the quality of Hessian estimation, largely eliminating the performance gap at 1000 perturbations.
- Increasing the regularization parameter $\lambda$ tightens the theoretical upper bound, as validated by the experiments.
- The method is equally effective on mixed linear networks (ResNet-18 + CIFAR-100), with a performance gap of only 0.5–3.7% when forgetting 10% of the data.

## Highlights & Insights

- First work to provide theoretical guarantees for source-free machine unlearning, filling a significant gap in the literature.
- Very clever core insight: under tiny perturbations, the loss differences between remaining and forget data are close, enabling Hessian estimation via SDP.
- The theoretical upper bound tightens as the dimension increases—implying that the method becomes more effective in high-dimensional (deep learning) scenarios, which carries substantial practical significance.
- Formulating the problem as an SDP is an elegant mathematical treatment.
- The MIA score close to 50% indicates that the unlearning process indeed erases membership information.

## Limitations & Future Work

- Currently directly applicable only to linear and mixed linear classifiers; fully non-linear deep networks require approximation via NTK linearization.
- Performance drops significantly when the forget ratio exceeds 10%, indicating that scalability is limited by the accuracy of Hessian estimation.
- Solving the SDP has high computational complexity, and the storage and inversion of large-scale Hessians remain bottlenecks.
- Future research could integrate approximation techniques like Hessian diagonalization or Hessian-vector products to enhance practicality.
- Extension to more complex unlearning scenarios, such as concept erasure in generative models, is worth exploring.

## Related Work & Insights

- The Newton update step by Guo et al. (2020) serves as the theoretical foundation for this work, and this paper overcomes its limitation of requiring the Hessian of the remaining data.
- Compared to the zero-shot unlearning of Chundawat et al. (class-level only) and the instance-level unlearning of Cha et al., this work achieves both instance-level unlearning and theoretical guarantees.
- Mixed Linear Unlearning (Golatkar et al., 2021) provides a viable path for the linearization of deep networks.
- This work holds direct practical significance for data privacy compliance (such as GDPR) and model governance.

## Rating

- **Novelty**: 8/10 — The core Hessian estimation method is clever, and the theoretical contribution is solid.
- **Experimental Thoroughness**: 7/10 — Validated via multiple datasets and extensive ablation studies, though lacking large-scale deep learning scenarios.
- **Writing Quality**: 8/10 — The theoretical derivation is clear and complete, and the experimental design is solid.
- **Value**: 8/10 — The theoretical guarantees for source-free unlearning hold great significance for privacy compliance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Efficient Verified Machine Unlearning for Distillation](../../NeurIPS2025/ai_safety/efficient_verified_machine_unlearning_for_distillation.md)
- [\[ICLR 2026\] Remaining-data-free Machine Unlearning by Suppressing Sample Contribution](../../ICLR2026/ai_safety/remaining-data-free_machine_unlearning_by_suppressing_sample_contribution.md)
- [\[NeurIPS 2025\] Position: Bridge the Gaps between Machine Unlearning and AI Regulation](../../NeurIPS2025/ai_safety/position_bridge_the_gaps_between_machine_unlearning_and_ai_regulation.md)
- [\[ICML 2025\] A Certified Unlearning Approach without Access to Source Data](../../ICML2025/ai_safety/a_certified_unlearning_approach_without_access_to_source_data.md)
- [\[NeurIPS 2025\] Rewind-to-Delete: Certified Machine Unlearning for Nonconvex Functions](../../NeurIPS2025/ai_safety/rewind-to-delete_certified_machine_unlearning_for_nonconvex_functions.md)

</div>

<!-- RELATED:END -->

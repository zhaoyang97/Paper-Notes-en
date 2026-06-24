---
title: >-
  [Paper Note] Set-Valued Predictions for Robust Domain Generalization
description: >-
  [ICML2025][Set-valued prediction] A set-valued predictor is proposed to address the robustness issue in domain generalization (DG): it outputs a subset of labels rather than a single label, satisfying predefined coverage requirements on as many unseen domains as possible while minimizing the prediction set size.
tags:
  - "ICML2025"
  - "Set-valued prediction"
  - "Domain generalization"
  - "Robustness"
  - "Conformal prediction"
  - "VC dimension"
  - "Worst-case guarantee"
date: 2026-05-08
content_hash: cbbc74145f7d4384
---

# Set-Valued Predictions for Robust Domain Generalization

**Conference**: ICML2025  
**arXiv**: [2507.03146](https://arxiv.org/abs/2507.03146)  
**Code**: [ront65/set-valued-ood](https://github.com/ront65/set-valued-ood)  
**Area**: LLM Evaluation  
**Keywords**: Set-valued prediction, Domain generalization, Robustness, Conformal prediction, VC dimension, Worst-case guarantee

## TL;DR

A set-valued predictor is proposed to address the robustness issue in domain generalization (DG): it outputs a subset of labels rather than a single label, satisfying predefined coverage requirements on as many unseen domains as possible while minimizing the prediction set size.

## Background & Motivation

- **Core Challenge of Domain Generalization**: Models perform well on training distributions but experience severe performance degradation on unseen test distributions (domains). In high-risk scenarios such as medical imaging, worst-case performance guarantees are required rather than just average performance.
- **Limitations of Prior Work**:
    - Invariant-feature-based methods (e.g., IRM, DRO) pursue stability across domains, but often at the cost of performance degradation—a "stable" predictor with only 60% accuracy across all domains is of little practical value.
    - All existing methods rely on **singleton prediction**, which naturally limits robustness when there is no single optimal prediction across different domains.
- **Key Insight**: Different domains induce different probabilistic relationships between $X$ and $Y$, making it unrealistic to expect a single predictor to be optimal across all domains. Set-valued outputs can capture a range of potential relationships learned from training domains, thereby enhancing robustness.
- **Key Trade-off**: The prediction set must not be too large (collapsing to predicting the entire label space is useless), so the goal is to **minimize the prediction set size while achieving the predefined coverage**.

## Method

### Problem Formulation

Let features be $X \in \mathcal{X}$, labels $Y \in \mathcal{Y}$ (a finite set), and domain $e \in \mathcal{E}$ correspond to distribution $P_e$. During the training stage, $m$ domains are observed, with $n_e$ samples for each domain $e$. At test time, an unseen domain $e_{test}$ is encountered.

A set-valued predictor is defined as $h: \mathcal{X} \to \mathcal{P}(\mathcal{Y})$, which outputs a subset of labels.

### Learning Objective

Define the domain-level loss as the worst-case label recall:

$$\mathcal{L}_{recall}(h, e) = \max_{y \in \mathcal{Y}} P_e[y \notin h(X) | Y = y]$$

Define the performance indicator $\mathbb{1}_{\mathcal{L},\gamma}(h,e) = \mathbb{1}[\mathcal{L}(h,e) \geq \gamma]$, and the optimization objective is:

$$\min_{h \in \mathcal{H}} P_{e \sim D}[\mathcal{L}(h,e) \geq \gamma]$$

That is, to minimize the probability that the predictor fails to achieve a $1-\gamma$ recall on a domain.

### Theoretical Contributions: Hierarchical VC Dimension

- Generalizes the classical VC dimension to a **multi-domain hierarchical framework**, measuring the shattering capability of the hypothesis set $\mathcal{H}$ over the domain space $\mathcal{E}$.
- **Theorem 3.4**: $\mathcal{H}$ has uniform convergence ⟺ $VCdim_{\mathcal{L},\gamma}(\mathcal{H}) < \infty$, with the required number of domains $m = \Theta\!\left(\frac{d + \log(1/\delta)}{\epsilon^2}\right)$.
- **Negative Result (Theorem 3.6)**: When the feature dimension $d > 1$, the VC dimension of the linear hypothesis set on an unconstrained domain space is infinite—domain generalization is inherently more difficult than classical learning.
- **Positive Result (Theorem 3.7)**: If domains satisfy the conditional Gaussian assumption (covariance structures are shared up to a scaling factor), the VC dimension of the linear hypothesis set is finite, with the required number of training domains $m = \Theta\!\left(\frac{|\mathcal{Y}|^2(d + \log(|\mathcal{Y}|/\delta))}{\epsilon^2}\right)$.

### SET-COVER Algorithm

Constrained optimization problem: minimize the prediction set size, subject to the constraint that the recall for each label in each training domain is $\geq 1-\gamma$:

$$\min_\theta \sum_{i \in S} |h(x_i)| \quad \text{s.t.} \quad \frac{1}{|G_{e,y}|}\sum_{i \in G_{e,y}} \mathbb{1}[y \notin h(X_i)] \leq \gamma, \; \forall e, y$$

By decomposing the set-valued predictor into per-label binary classifiers $h_y$ and introducing the parameterization $h_y^\theta$ along with a hinge relaxation, a differentiable optimization problem is obtained.

Solved via Lagrangian multipliers $C_{e,y}$:
- Perform gradient descent on $\theta$ (minimizing prediction set size + satisfying constraints)
- Perform gradient ascent on $C$ (increasing multipliers when constraints are violated), updated every few batches

Final loss function:

$$L_y(\theta, C) = \sum_{i \notin G_y} \max\{0, 1+h_y^\theta(X_i)\} + \sum_{e} \mathbb{1}_{i \in G_{e,y}} C_{e,y} \max\{0, 1-h_y^\theta(X_i)\}$$

### Baseline: Robust Conformal Prediction

Train a classifier on the pooled training data to obtain logits as conformal scores, calibrate the threshold $t_{e,y}$ **per-domain and per-label**, and adopt the most conservative (loosest) criterion over all training domain thresholds during testing (included in the prediction set if any domain's criterion is met).

## Key Experimental Results

### Synthetic Data

| Setting | Method | Min-Recall | Avg Set Size |
|---------|--------|------------|--------------|
| d=50 | ERM | Low (Not meeting target) | 1.0 |
| d=50 | Robust Conformal | Meeting target | Larger |
| d=50 | **SET-COVER** | **Meeting target** | **Smaller** |

### WILDS Real-World Data (Target min-recall ≥ 90%)

| Dataset | Method | Median Min-Recall | Median Avg Size | Proportion of Domains with ≥90% |
|--------|------|-------------------|-----------------|------------|
| Camelyon | ERM | 0.93 | 1.0 | 63% |
| Camelyon | Robust Conformal | **0.98** | 1.79 | 93% |
| Camelyon | **SET-COVER** | 0.96 | **1.05** | 71% |
| FMoW | ERM | 0.76 | 1.0 | 8% |
| FMoW | Robust Conformal | 0.89 | 1.17 | 43% |
| FMoW | **SET-COVER** | **0.91** | **1.10** | **72%** |
| iWildCam | ERM | 0.99 | 1.0 | 71% |
| iWildCam | Robust Conformal | 0.99 | 2.00 | 86% |
| iWildCam | **SET-COVER** | 0.99 | **1.01** | 82% |
| Amazon | ERM | 1.0 | 1.0 | 68% |
| Amazon | Robust Conformal | 1.0 | 4.68 | **99%** |
| Amazon | **SET-COVER** | 1.0 | **2.43** | 96% |

### Key Findings

- Both SET-COVER and Robust Conformal significantly outperform the ERM and Pooling-CDFs baselines.
- While achieving similar coverage, SET-COVER maintains a **much smaller** prediction set size than Robust Conformal (e.g., Amazon: 2.43 vs 4.68; Camelyon: 1.05 vs 1.79).
- On FMoW, the proportion of domains with ≥90% recall for SET-COVER (72%) far exceeds that of Robust Conformal (43%), demonstrating that end-to-end optimization is more effective than post-hoc calibration.
- SET-COVER's performance remains stable even when the covariance matrix varies randomly across domains (experiments in the Appendix).

## Highlights & Insights

1. **Paradigm Shift**: Shifting from "pursuing a single optimal prediction" to "outputting a small and robust prediction set" provides a fresh perspective on domain generalization.
2. **Theoretical Rigor**: For the first time, the VC dimension under a multi-domain hierarchical framework is defined and generalization bounds are proven, revealing that domain generalization is inherently more difficult than classical learning (the VC dimension of the linear hypothesis is infinite when $d>1$).
3. **High Practicality**: Optimized via Lagrangian duality, the SET-COVER algorithm can be seamlessly integrated with standard neural network architectures without requiring additional calibration data.
4. **Significant Set Size Advantage**: Compared to the strong baseline Robust Conformal, SET-COVER consistently maintains smaller prediction sets across all four WILDS datasets.

## Limitations & Future Work

- **Theory-Practice Gap**: Theorem 3.7 strictly holds only for linear predictors on conditional Gaussian domains, whereas the neural networks used in experiments lack theoretical guarantees.
- **Scalability of Label Count**: When $|\mathcal{Y}|$ is very large (e.g., ImageNet scale), training binary classifiers independently for each label incurs a large computational overhead.
- **Requirement for Domain Information**: Domain labels are required during the training phase to group constraints, which is impractical in scenarios where domain boundaries are ambiguous.
- **Lack of Conditional Coverage**: The current objective is domain-level min-recall, but conditional coverage (coverage guarantee given $X$) is not discussed.
- **Limited Experimental Scale**: Experiments on FMoW and iWildCam were only conducted on 2- or 3-class subsets; the full-class scenarios remain unverified.

## Related Work & Insights

- **Conformal Prediction**: The Robust Conformal baseline in this paper is a natural extension of CP to multi-domain scenarios, but CP itself only provides marginal coverage guarantees.
- **Pooling CDFs** (Dunn et al., 2018): A set-valued method for DG in regression tasks, which optimizes average coverage rather than the worst case.
- **QRM** (Eastwood et al., 2022): A singleton prediction method that minimizes the $\alpha$-quantile risk across domains, limited by the robustness bottleneck of singleton outputs.
- **Invariant Learning** (IRM, DRO, etc.): Pursuing the stability of invariant features, but often ignoring valuable domain-dependent features.
- **Insights**: The concept of set-valued prediction can be generalized to other OOD scenarios (e.g., covariate shift, label shift), and the hierarchical generalization of the VC dimension can also be applied to the theoretical analysis of meta-learning.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of set-valued prediction and domain generalization provides a novel perspective, and the hierarchical VC dimension theoretical framework is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Synthetic + 4 WILDS datasets, including ablation studies and comparisons with various baselines, though the number of labels is limited.
- Writing Quality: ⭐⭐⭐⭐ — Clear theoretical derivations, well-articulated motivation, and a logical structure.
- Value: ⭐⭐⭐⭐ — Proposes a practical and theoretically grounded new paradigm for domain generalization, offering direct value for high-risk scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Gradient-Guided Annealing for Domain Generalization](../../CVPR2025/others/gradient-guided_annealing_for_domain_generalization.md)
- [\[ICLR 2026\] Noise-Aware Generalization: Robustness to In-Domain Noise and Out-of-Domain Generalization](../../ICLR2026/others/noise-aware_generalization_robustness_to_in-domain_noise_and_out-of-domain_gener.md)
- [\[ICML 2025\] FEDTAIL: Federated Long-Tailed Domain Generalization with Sharpness-Guided Gradient Matching](fedtail_federated_long-tailed_domain_generalization_with_sharpness-guided_gradie.md)
- [\[ICCV 2025\] Adversarial Data Augmentation for Single Domain Generalization via Lyapunov Exponents](../../ICCV2025/others/adversarial_data_augmentation_for_single_domain_generalization_via_lyapunov_expo.md)
- [\[CVPR 2026\] Bridging Domain Expertise and Generalization for Performance Estimation](../../CVPR2026/others/bridging_domain_expertise_and_generalization_for_performance_estimation.md)

</div>

<!-- RELATED:END -->

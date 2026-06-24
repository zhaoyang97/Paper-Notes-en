---
title: >-
  [Paper Note] Doubly Robust Fusion of Many Treatments for Policy Learning
description: >-
  [ICML2025][AI Safety][Individualized Treatment Rules] A Calibration-Weighted Treatment Fusion method is proposed to reduce the dimensionality of the action space by doubly robustly merging treatment arms with similar effects, enabling existing multi-armed policy learning methods (such as policy trees) to be efficiently applied to individualized recommendation scenarios with a large number of treatment options.
tags:
  - "ICML2025"
  - "AI Safety"
  - "Individualized Treatment Rules"
  - "Treatment Fusion"
  - "Calibration Weighting"
  - "Doubly Robust"
  - "Fused Lasso"
  - "Policy Tree"
date: 2026-05-08
content_hash: b2ac3a1a37b57499
---

# Doubly Robust Fusion of Many Treatments for Policy Learning

**Conference**: ICML2025  
**arXiv**: [2505.08092](https://arxiv.org/abs/2505.08092)  
**Code**: To be confirmed  
**Area**: AI Safety  
**Keywords**: Individualized Treatment Rules, Treatment Fusion, Calibration Weighting, Doubly Robust, Fused Lasso, Policy Tree

## TL;DR

A Calibration-Weighted Treatment Fusion method is proposed to reduce the dimensionality of the action space by doubly robustly merging treatment arms with similar effects, enabling existing multi-armed policy learning methods (such as policy trees) to be efficiently applied to individualized recommendation scenarios with a large number of treatment options.

## Background & Motivation

In precision medicine, individualized treatment rules (ITRs) aim to recommend the optimal treatment plan based on patient characteristics. Existing ITR learning methods (such as Q-learning, A-learning, and policy trees) face two core challenges when the number of treatment options $K$ is large:

**Data Sparsity**: The sample size in each treatment arm is small, making it difficult to accurately estimate treatment effects.

**Covariate Shift**: Significant differences in covariate distributions across different treatment groups render inverse probability weighting (IPW) unstable.

Key Observation: Many treatments (e.g., different drugs targeting the same disease mechanism) exhibit similar or even identical effects, indicating a latent grouping structure. If the $K$ treatments can be correctly fused into $M \ll K$ groups, the complexity of ITR learning can be significantly reduced.

However, treatment fusion itself suffers from both data sparsity and covariate shift—sparse data restricts researchers to using linear working models (which are prone to misspecification), and severe covariate shift makes traditional IPW balancing unreliable. Existing methods (Ma et al., 2022) require the outcome model to be correctly specified, resulting in a sharp decline in fusion quality once the model is misspecified.

## Method

### Overall Architecture

The method consists of two stages:
1. **Treatment Fusion** (Algorithm 1): Discovering latent groupings $\delta: \mathcal{A} \to \mathcal{B}$ using calibration weighting and Fused Lasso.
2. **Policy Learning** (Algorithm 2): Learning the optimal ITR by applying CAIPWL and policy trees on the fused grouping space.

### Calibration Weighting

For each treatment group $a$, weights $\{w_i\}$ are assigned to samples within the group by solving a constrained optimization problem:

$$\min_{w_i} \sum_{i:A_i=a} h_\gamma(w_i), \quad \text{s.t.} \sum_{i:A_i=a} w_i X_i = \bar{X}, \quad \sum_{i:A_i=a} w_i = 1$$

where $h_\gamma$ belongs to the Cressie-Read family of divergences. This weighting aligns the weighted covariate mean of each treatment group with the overall sample mean, thereby mitigating the impact of covariate shift.

### Calibration-Weighted Fused Lasso for Treatment Fusion

Letting the linear working model be $Y = M_0(X) + \sum_a \mathbb{I}(A=a) X^\top \boldsymbol{\zeta}_a + \epsilon$, the weighted Fused Lasso is solved via:

$$\min_{\boldsymbol{\zeta}} \left\{ \frac{1}{2n} \sum_{a \in \mathcal{A}} \sum_{i:A_i=a} \hat{w}_i (\tilde{Y}_i - X_i^\top \boldsymbol{\zeta}_a)^2 + \sum_{1 \leq a < a' \leq K} p_{\lambda_n}(\|\boldsymbol{\zeta}_a - \boldsymbol{\zeta}_{a'}\|_1) \rightfixed}$$

The penalty term encourages the parameter vectors $\boldsymbol{\zeta}_a$ of different treatments to fuse. When $\hat{\boldsymbol{\zeta}}_a = \hat{\boldsymbol{\zeta}}_{a'}$, treatments $a$ and $a'$ are classified into the same group.

### Double Robustness

The core theoretical contribution is that the consistency of treatment fusion requires only one of the following two conditions to hold:

- **Correct Calibration Weighting**: $w_i^* = 1/\pi_{A_i}(X_i)$ (i.e., the propensity score is correctly estimated)
- **Correct Outcome Model**: $\mathbb{E}\{\varepsilon(a) \mid X\} = 0$ (i.e., the linear working model is correctly specified)

This represents a significant improvement over prior work (Ma et al., 2022), which requires the outcome model to be correctly specified.

### Theoretical Guarantees

- **Oracle Estimator Consistency** (Theorem 3.8): Under doubly robust conditions and regularity assumptions, $\|\hat{\boldsymbol{\zeta}}^{\text{or}} - \boldsymbol{\zeta}^*\|_\infty \leq C\sqrt{p \cdot n \cdot \log(n)}/N_{\min}$
- **Oracle Property** (Theorem 3.12): Local minima of the Fused Lasso equal the Oracle estimator with probability approaching 1, meaning the latent grouping can be correctly recovered.
- **Policy Tree Regret Bound** (Proposition 3.18): $R(\hat{d}^{\mathcal{B}}) = O_\mathbb{P}\left(\left\{\sqrt{(2^D-1)\log p + 2^D \log M} + \frac{4}{3} D^{1/4}\sqrt{2^D-1}\right\}\sqrt{V_*/n}\right)$

Key condition: $K$, $M$, and $p$ are allowed to grow with $n$, requiring $M = o(\sqrt{n/\{p\log(n)\}})$.

### Policy Learning Stage

Upon completion of the fusion stage, Cross-Fitted AIPW Learning (CAIPWL) is deployed on the grouped space $\mathcal{B}$:
1. L-fold cross-fitting is used to estimate group-level propensity scores $\pi_b(x)$ and outcome functions $\mu_b(x)$.
2. Construct the AIPW value function estimator.
3. Maximize the value function over the policy tree class.

## Key Experimental Results

### Simulation Experiments (K=16, M=4)

| Method | ARI | Number of Groups | Policy Value |
|------|-----|------|--------|
| Policy tree (Baseline) | / | 16 | 8.77 (0.08) |
| Fusion + policy tree | 0.26 (0.14) | 10.73 (1.93) | 8.78 (0.09) |
| **CW + fusion + policy tree** | **0.96 (0.06)** | **4.34 (0.60)** | **8.89 (0.11)** |
| Ma et al. (2022) | 0.26 (0.14) | 10.73 (1.93) | 8.51 (0.12) |

- Calibration weighting improves the ARI from 0.26 to 0.96, recovering the true 4-group structure almost perfectly.
- Due to outcome model misspecification, the policy value of Ma et al. (2022) is lower than the baseline (8.51 vs 8.77).

### Real-World Data: Treatment Recommendation for CLL/SLL Patients

- Dataset: Flatiron Health Electronic Health Records, 10,346 patients with CLL/SLL, 7 first-line treatments.
- Fusion Results: 7 treatments are clustered into 5 groups (two monotherapies merged, combination therapies remained distinct, chemotherapy alone formed its own group).
- Learned Policy Tree (depth 5) reveals: Chemo is preferred for elder patients or those with short time since diagnosis; combination therapies are favored for younger patients or those with a chronic disease course.

## Highlights & Insights

1. **Doubly Robust Treatment Fusion**: Introduces double robustness to the treatment grouping problem for the first time, significantly enhancing robustness against model misspecification.
2. **Modular Design**: Decouples the fusion stage from the policy learning stage, allowing flexible use of different covariates (using all covariates for fusion, and an actionable subset for the policy tree to exclude sensitive variables like race from decisions, thereby balancing fairness).
3. **Theoretical Completeness**: Provides a comprehensive chain of theoretical guarantees, spanning consistency, the Oracle property, and regret bounds.
4. **High Practicality**: The approach is easy to implement and seamlessly integrates with the existing R package `policytree`.

## Limitations & Future Work

1. **Extremely Sparse Scenarios**: Calibration weighting may become unstable when some treatment arms have extremely few or zero samples.
2. **One-off Fusion**: The current implementation performs fusion only once; iterative, alternating fusion and weight estimation could enhance stability.
3. **Single Data Source**: Does not consider multi-center/multi-source data integration or generalization to target populations.
4. **Lack of Uncertainty Quantification**: Lacks confidence intervals or uncertainty estimation for the recommended policy.
5. **Continuous Treatments**: The method is designed for discrete treatments; continuous or combinatorial treatments require different frameworks.

## Related Work & Insights

- **Ma et al. (2022, 2023)**: Proposed Fused Lasso for treatment fusion but without calibration weighting, which requires a correctly specified outcome model.
- **Zhou et al. (2023)**: CAIPWL + policy tree; Ours adds a pre-processing fusion step on top of this.
- **Lee et al. (2023); Wu & Yang (2023)**: Foundations of calibration weighting methods.
- **Athey & Wager (2021)**: Theoretical framework for policy trees.
- Inspiration: The idea of double robustness can be generalized to other causal inference problems that require dimensionality reduction or grouping.

## Rating

- Novelty: ⭐⭐⭐⭐ — Doubly robust fusion is a novel combination with solid theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Simulations + real medical data, but lacks more large-scale scenarios.
- Writing Quality: ⭐⭐⭐⭐ — Clear theoretical presentation with a complete notation system.
- Value: ⭐⭐⭐⭐ — Addresses practical pain points in precision medicine, offering a method that is both practical and theoretically grounded.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FedAFD: Multimodal Federated Learning via Adversarial Fusion and Distillation](../../CVPR2026/ai_safety/fedafd_multimodal_federated_learning_via_adversarial_fusion_and_distillation.md)
- [\[ICLR 2026\] Doubly-Regressing Approach for Subgroup Fairness](../../ICLR2026/ai_safety/doubly-regressing_approach_for_subgroup_fairness.md)
- [\[ICML 2026\] Position: Machine Learning for Heart Transplant Allocation Policy Optimization Should Account for Incentives](../../ICML2026/ai_safety/position_machine_learning_for_heart_transplant_allocation_policy_optimization_sh.md)
- [\[ACL 2025\] Building a Long Text Privacy Policy Corpus with Multi-Class Labels](../../ACL2025/ai_safety/building_a_long_text_privacy_policy_corpus_with_multi-class_labels.md)
- [\[ECCV 2024\] Fisher Calibration for Backdoor-Robust Heterogeneous Federated Learning](../../ECCV2024/ai_safety/fisher_calibration_for_backdoor-robust_heterogeneous_federated_learning.md)

</div>

<!-- RELATED:END -->

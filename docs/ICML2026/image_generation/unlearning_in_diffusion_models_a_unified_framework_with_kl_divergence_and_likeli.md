---
title: >-
  [Paper Note] 扩散模型中的遗忘：基于 KL 散度和似然约束的统一框架
description: >-
  [ICML 2026][Image Generation][Diffusion Model] This paper proposes a unified constrained optimization framework that formalizes machine unlearning in diffusion models as minimizing the deviation from a pre-trained model subject to explicit separation conditions from the unlearning distribution. Through three constraint forms (Reverse KL, Forward KL, and Likelihood
tags:
  - ICML 2026
  - Image Generation
  - Diffusion Model
date: 2026-05-08
content_hash: 03ca5022898ecba5
---
# Forgetting in Diffusion Models: A Unified Framework via KL Divergence and Likelihood Constraints

**Conference**: ICML 2026  
**arXiv**: [2605.30825](https://arxiv.org/abs/2605.30825)  
**Code**: To be confirmed  
**Area**: Diffusion Models / Image Generation / Machine Unlearning  
**Keywords**: Machine Unlearning, Diffusion Models, KL Divergence Constraint, Likelihood Constraint, Strong Duality

## TL;DR
This paper proposes a unified constrained optimization framework that formalizes machine unlearning in diffusion models as minimizing the deviation from a pre-trained model subject to explicit separation conditions from the unlearning distribution. Through three constraint forms (Reverse KL, Forward KL, and Likelihood constraints), it uniformly handles concept and data unlearning while proving strong duality.

## Background & Motivation

**Background**: Diffusion models are widely applied due to their high-quality image generation capabilities but may generate harmful content, violate copyrights, or contain inappropriate concepts. Machine unlearning has emerged as a critical research direction.

**Limitations of Prior Work**: Existing empirical methods (e.g., concept erasure) use simple weight combinations to balance two conflicting objectives (retaining model capability vs. removing inappropriate content). However, weight settings are inherently heuristic and exhibit poor generalization across different scenarios.

**Key Challenge**: There is a fundamental conflict between preserving the utility of the pre-trained model and preventing the generation of specific harmful data or concepts, requiring a systematic trade-off mechanism.

**Goal**: To establish a principled constrained optimization framework that explicitly characterizes this trade-off.

**Key Insight**: Formalize preserving model capability as minimizing the distance to the pre-trained model and unlearning as a constraint for separation from undesirable distributions, utilizing the Lagrange multiplier method to handle conflicting goals.

**Core Idea**: Unify concept and data unlearning through three constrained optimization problems (RU/FU/LU). Leverage the convexity of non-atomic vector measures to prove strong duality, resulting in explicit optimal solutions.

## Method

### Overall Architecture
The paper reformulates machine unlearning in diffusion models from "manually tuning conflicting goals with weights" into a constrained optimization problem: Preservation = keeping the new model $p$ as close as possible to the pre-trained model $q$; Unlearning = forcing $p$ to maintain explicit separation from $m$ unlearning distributions $q_u^i$. Based on the chosen separation metric, the authors provide three unified forms—Reverse KL Unlearning (RU): minimize $D_{KL}(p \| q)$ subject to $D_{KL}(p \| q_u^i) \geq b_i$; Forward KL Unlearning (FU): minimize $D_{KL}(q \| p)$ subject to $D_{KL}(q_u^i \| p) \geq b_i$; Likelihood Constrained Unlearning (LU): minimize $D_{KL}(p \| q)$ subject to $\mathbb{E}_p[q_u^i] \leq \epsilon_i$. These cover both concept and data unlearning. Strong duality is then used to map these explicit optimal solutions in distribution space back to the score functions of diffusion models.

### Key Designs

**1. Constrained Optimization Framework: Replacing Heuristic Weights with Interpretable Thresholds**

Existing concept erasure methods linearly mix the conflicting "preserve vs. remove" goals using a manually tuned weight, which relies on trial and error and fails when scenarios change. This work instead formulates preservation as the objective function (staying close to $q$) and unlearning as a hard constraint. The intensity of unlearning is directly specified via thresholds: RU/FU use KL divergence constraints $\geq b_i$ to "push away" undesirable distributions, while LU uses a likelihood upper bound $\mathbb{E}_p[q_u^i] \leq \epsilon_i$ to limit the probability of sampling unlearning content. All three forms transform "unlearning strength" into a specifiable and reproducible scalar, naturally handling multiple unlearning targets—something heuristic weights struggle to achieve.

**2. Strong Duality: Bypassing Non-convexity via Lyapunov Convexity**

The KL constraints for RU/FU are non-convex in distribution space, which typically makes it impossible to guarantee optimality in the dual domain. The authors introduce Lyapunov's convexity theorem to prove that the image of a non-atomic vector measure is a convex set. By transferring this convexity from the measure space to the score function space, they overcome the non-convex barrier of KL constraints and establish strong duality. This allows for closed-form optimal solutions: RU follows $p^* \propto q / \prod_i (q_u^i)^{\alpha_i}$, FU follows $p^* \propto q - \sum_i \lambda_i q_u^i$, and LU follows $p^* \propto q \cdot e^{-\sum_i \lambda_i q_u^i}$. These forms intuitively correspond to "scaling down," "direct subtraction," and "exponential decay" of the unlearning components.

**3. Primal-Dual Algorithm: Mapping Closed-form Solutions to Score Functions**

As closed-form solutions exist in distribution space while actual training involves parameterized score functions, a "parameterization gap" exists. The authors utilize primal-dual iteration: the primal step performs gradient descent on model parameters to minimize the Lagrangian, while the dual step performs gradient ascent on the Lagrange multiplier $\lambda$ to tighten unsatisfied constraints. The multiplier $\lambda$ adjusts dynamically based on the degree of constraint violation, effectively assigning larger weights to samples that are harder to forget.

## Key Experimental Results

### Main Results

| Scenario | Base Model | Contrastive Method | Main Metrics | Conclusion |
|------|---------|---------|---------|------|
| Likelihood Unlearning | GMM + Stable Diffusion | Concept Erasure | KL (Preserved modes) | Constraints yield smaller deviation for same unlearning level |
| Forward KL Unlearning | CelebA-HQ DDPM | Unconstrained baseline | KID (Preserve) / max SSCD (Unlearn) | Better KID at the same unlearning intensity |
| Reverse KL Multi-concept | Stable Diffusion | Unconstrained baseline | KL(orig) vs CLIP score | Smaller KL deviation for equivalent CLIP scores |

### Ablation Study

| Constraint Type | Unlearning Intensity | Preservation Capability | Gain over Baseline |
|---------|---------|------------------|-----------------|
| Likelihood (High) | Low likelihood | KL↓ significantly | Significantly superior |
| Forward KL (Strong) | High max SSCD | KID↓ significantly | Constraints learn dynamic weights |
| Reverse KL (Multi) | KL(unlearn)↑ | KL(origin)↓ significantly | Closer to original model for same unlearning |

### Key Findings
- Constraints learn optimal weight allocation—in Forward KL, the method automatically assigns larger weights to hard-to-forget samples and smaller weights to easy ones.
- Likelihood constraints offer finer preservation—LU suppresses undesirable concepts without forcibly pushing away other preserved modes like RU might.
- Parameterization gap impacts—when constraints are too aggressive or distributions are heavily entangled, the parameterization gap increases, leading to constraint violations.

## Highlights & Insights
- **Three Unified Perspectives**: RU, FU, and LU provide a unified view of unlearning through explicit optimal solutions derived via strong duality.
- **Novel Application of Lyapunov's Theorem**: Clever application of Lyapunov's convexity theorem to non-atomic vector measures breaks the non-convexity barrier of KL constraints.
- **High Transferability**: The framework is not limited to image generation and can, in principle, be extended to language, audio, and other generative tasks.

## Limitations & Future Work
- Parameterization gap—while theory provides optimal solutions in distribution space, the gap in actual diffusion model implementations can be significant.
- Computational cost—requires multiple iterative updates for dual variables compared to baselines.
- Scalability verification—only validated on text-to-image models; performance when unlearning multiple highly entangled concepts needs further evaluation.

## Related Work & Insights
- **vs. Concept Erasure (Gandikota et al., 2023)**: Concept erasure uses heuristic weight combinations; this work derives optimal weights via constrained optimization.
- **vs. Data Unlearning (Wu et al., 2025)**: This work unifies the FU case from a dual perspective and provides an explicit optimal solution.
- **vs. Regularization Methods**: The constraint thresholds are more interpretable, allowing for finer control over multi-target unlearning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reformulating unlearning as constrained optimization with strong duality and explicit solutions are novel contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Sufficient comparison across three scenarios; missing extension tasks (text/audio).
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, precise problem definitions, and detailed theoretical derivations.
- Value: ⭐⭐⭐⭐ Provides a principled framework and algorithm for unlearning; significant for safety and controllability in generative models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] GUDA: Counterfactual Group-wise Training Data Attribution for Diffusion Models via Unlearning](guda_counterfactual_group-wise_training_data_attribution_for_diffusion_models_vi.md)
- [\[ICML 2026\] Stage-wise Distortion-Perception Traversal in Zero-shot Inverse Problems with Diffusion Models](stage-wise_distortion-perception_traversal_in_zero-shot_inverse_problems_with_di.md)
- [\[ICML 2026\] SAEmnesia: Erasing Concepts in Diffusion Models with Supervised Sparse Autoencoders](saemnesia_erasing_concepts_in_diffusion_models_with_supervised_sparse_autoencode.md)
- [\[ICML 2026\] Diffusion Models Are Statistically Optimal for Learning Low-Dimensional Multi-Modal Distributions](diffusion_models_are_statistically_optimal_for_learning_low-dimensional_multi-mo.md)
- [\[ICML 2026\] Local Hessian Spectral Filtering for Robust Intrinsic Dimension Estimation](local_hessian_spectral_filtering_for_robust_intrinsic_dimension_estimation.md)

</div>

<!-- RELATED:END -->

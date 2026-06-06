---
title: >-
  [Paper Note] Unlearning in Diffusion Models: A Unified Framework Based on KL Divergence and Likelihood Constraints
description: >-
  [ICML 2026][Image Generation][Machine Unlearning] This paper proposes a unified constrained optimization framework that formalizes the machine unlearning problem in diffusion models as minimizing the deviation from a pre…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Machine Unlearning"
  - "Diffusion Models"
  - "KL Divergence Constraint"
  - "Likelihood Constraint"
  - "Strong Duality"
date: 2026-05-08
content_hash: b01532c21bf88551
---

# Unlearning in Diffusion Models: A Unified Framework Based on KL Divergence and Likelihood Constraints

**Conference**: ICML 2026  
**arXiv**: [2605.30825](https://arxiv.org/abs/2605.30825)  
**Code**: To be confirmed  
**Area**: Diffusion Models / Image Generation / Machine Unlearning  
**Keywords**: Machine Unlearning, Diffusion Models, KL Divergence Constraint, Likelihood Constraint, Strong Duality

## TL;DR
This paper proposes a unified constrained optimization framework that formalizes the machine unlearning problem in diffusion models as minimizing the deviation from a pre-trained model while being subject to explicit separation conditions from unlearning distributions. It uniformly handles concept and data unlearning through three constraint forms (Reverse KL, Forward KL, and Likelihood Constraints) and proves strong duality.

## Background & Motivation

**Background**: Diffusion models are widely applied for their high-quality image generation capabilities but may generate harmful content, violate copyrights, or contain inappropriate concepts. Machine unlearning has emerged as a critical research direction.

**Limitations of Prior Work**: Existing empirical methods (e.g., concept erasure) adopt simple weight combinations to balance two conflicting goals (retaining model capability vs. removing inappropriate content). However, these weight settings are essentially heuristic and generalize poorly across different scenarios.

**Key Challenge**: Maintaining the utility of pre-trained models while preventing the generation of specific harmful data/concepts is a fundamental trade-off that requires a systematic mechanism.

**Goal**: To establish a principled constrained optimization framework that explicitly characterizes this trade-off.

**Key Insight**: Model capability retention is formalized as minimizing the distance to the pre-trained model, while unlearning is formalized as constraints for separation from undesirable distributions, utilizing the method of Lagrange multipliers to handle conflicting objectives.

**Core Idea**: Three constrained optimization problems (RU/FU/LU) are used to unify concept and data unlearning. By leveraging the convexity of non-atomic vector measures, the authors prove strong duality and derive explicit optimal solutions.

## Method

### Overall Architecture
Given a pre-trained model $q$ and $m$ distributions to be forgotten $q_u^i$, the paper proposes three constrained optimization problems: (1) **Reverse KL-constrained Unlearning (RU)**: minimize $D_{KL}(p \| q)$ subject to $D_{KL}(p \| q_u^i) \geq b_i$; (2) **Forward KL-constrained Unlearning (FU)**: minimize $D_{KL}(q \| p)$ subject to $D_{KL}(q_u^i \| p) \geq b_i$; (3) **Likelihood-constrained Unlearning (LU)**: minimize $D_{KL}(p \| q)$ subject to $\mathbb{E}_p[q_u^i] \leq \epsilon_i$.

### Key Designs

1. **Constrained Optimization Framework vs. Weight Heuristics**:
    - Function: Systematically encodes the unlearn-retain trade-off by directly specifying the target separation degree or likelihood upper bound via constraint thresholds $b_i$ or $\epsilon_i$.
    - Mechanism: RU/FU push away undesirable distributions via KL divergence constraints; LU limits high-probability sampling through likelihood constraints. The three correspond to different unlearning semantics.
    - Design Motivation: Compared to the ad-hoc nature of weight-based methods, the constrained approach provides interpretable and reproducible objectives, offering better control over multi-target unlearning.

2. **Strong Duality via Non-atomic Vector Measures**:
    - Function: Proves strong duality for the three problems (including non-convex problems involving KL constraints).
    - Mechanism: Utilizes Lyapunov’s convexity theorem to prove the convexity of the range of non-atomic vector measures. This transfers convexity properties from probability measure space to score function space, overcoming the non-convexity of KL constraints.
    - Design Motivation: Strong duality allows for solving in the dual domain and yields explicit solutions (e.g., distribution ratio $p^* \propto q/\prod_i (q_u^i)^{\alpha_i}$ in RU, distribution difference $p^* \propto q - \sum_i \lambda_i q_u^i$ in FU, and exponential tilting $p^* \propto q \cdot e^{-\sum_i \lambda_i q_u^i}$ in LU).

3. **Primal-Dual Algorithm for Diffusion Models**:
    - Function: Maps optimal solutions in the distribution space to the actual score function parameterization of diffusion models.
    - Mechanism: The primal step uses gradient descent to minimize the Lagrangian; the dual step uses gradient ascent to update the Lagrange multipliers $\lambda$. Corresponding losses are designed for RU, FU, and LU respectively.
    - Design Motivation: While a "parameterization gap" always exists between diffusion model parameters and optimal distributions, the primal-dual method converges to an approximately optimal solution when constraints are satisfied.

## Key Experimental Results

### Main Results

| Scenario | Baseline Model | Comparison Method | Primary Metric | Conclusion |
|------|---------|---------|---------|------|
| Likelihood-constrained | 3-Gaussian Mix + SD | Concept Erasure | KL (Retained Mode) | Constrained method shows smaller deviation at equal unlearning level. |
| Forward KL Unlearning | CelebA-HQ DDPM | Unconstrained baseline | KID (Retain) / max SSCD (Unlearn) | Better KID achieved at same unlearning level. |
| Reverse KL Multi-concept | Stable Diffusion | Unconstrained baseline | KL (Orig. Model) vs CLIP Score | Smaller KL deviation at equivalent CLIP scores. |

### Ablation Study

| Constraint Type | Unlearning Degree | Retention Capability | Gain over Baseline |
|---------|---------|------------------|-----------------|
| Likelihood (High) | Low likelihood | KL↓ significant | Significantly superior |
| Forward KL (Strong) | max SSCD high | KID↓ significant | Constraints learn dynamic weights |
| Reverse KL (Multi-concept) | KL(unlearn)↑ | KL(origin)↓ significant | Closer to original model at same unlearning level |

### Key Findings
- Constraints learn optimal weight allocation: In Forward KL, the constrained method automatically assigns larger weights to hard-to-forget samples and smaller weights to easy-to-forget ones.
- Finer retention in Likelihood Constraints: LU suppresses undesirable concepts without forcibly pushing away other retained modes as RU does.
- Impact of parameterization gap: When constraints are too aggressive or when retained/forgotten distributions are heavily entangled, the parameterization gap increases, leading to constraint violations.

## Highlights & Insights
- **Three Unified Perspectives**: RU/FU/LU yield explicit optimal solutions through strong duality, providing a unified view of unlearning.
- **Novel Application of Lyapunov’s Theorem**: Cleverly applying Lyapunov’s convexity theorem to non-atomic vector measures breaks the non-convexity barrier of KL constraints.
- **Strong Transferability**: The framework is not limited to image generation and can, in principle, be extended to generative tasks in language, audio, etc.

## Limitations & Future Work
- Parameterization gap issue: Theory provides optimal solutions in the distribution space, but the gap in actual diffusion model implementations may be significant.
- Computational cost: Requires multiple iterative updates of dual variables compared to baselines.
- Scalability verification: Primarily validated on text-to-image models; performance when unlearning multiple highly entangled concepts simultaneously requires further evaluation.

## Related Work & Insights
- **vs Concept Erasure (Gandikota et al., 2023)**: Concept erasure uses heuristic weight combinations; this work derives optimal weights via constrained optimization.
- **vs Data Unlearning (Wu et al., 2025)**: This paper unifies the FU case from a dual perspective and provides an explicit optimal solution.
- **vs Regularization Methods**: Constraint thresholds are more interpretable and provide finer control over multi-target unlearning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Re-formalizing unlearning as constrained optimization; strong duality and explicit solutions are novel contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Sufficient across three scenarios; extended tasks (text/audio) are missing.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, precise problem definitions, and detailed theoretical derivations.
- Value: ⭐⭐⭐⭐ Provides a principled framework and algorithm for unlearning; significant for safe and controllable generative models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] A Unified Framework for Diffusion Model Unlearning with f-Divergence](a_unified_framework_for_diffusion_model_unlearning_with_f-divergence.md)
- [\[ICML 2026\] GUDA: Counterfactual Group-wise Training Data Attribution for Diffusion Models via Unlearning](guda_counterfactual_group-wise_training_data_attribution_for_diffusion_models_vi.md)
- [\[ICML 2026\] Unified Masked Diffusion Models with Diverse Generation Orders](unifying_masked_diffusion_models_with_various_generation_orders_and_beyond.md)
- [\[ICML 2026\] Quantifying Error Propagation and Model Collapse in Diffusion Models](quantifying_error_propagation_and_model_collapse_in_diffusion_models.md)
- [\[ICML 2026\] SAEmnesia: Erasing Concepts in Diffusion Models with Supervised Sparse Autoencoders](saemnesia_erasing_concepts_in_diffusion_models_with_supervised_sparse_autoencode.md)

</div>

<!-- RELATED:END -->

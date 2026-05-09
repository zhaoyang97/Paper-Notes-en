---
title: >-
  [Paper Note] Composition and Alignment of Diffusion Models using Constrained Learning
description: >-
  [NeurIPS 2025][Image Generation][Diffusion Models] This paper proposes a unified constrained optimization framework that formalizes reward alignment and multi-model composition of diffusion models as constrained optimization problems. By applying Lagrangian duality, the framework automatically determines optimal weights, eliminating the need for manual hyperparameter search.
tags:
  - NeurIPS 2025
  - Image Generation
  - Diffusion Models
  - Constrained Optimization
  - Model Alignment
  - Model Composition
  - Lagrangian Duality
date: 2026-05-08
content_hash: be063061a307f1ea
---

# Composition and Alignment of Diffusion Models using Constrained Learning

**Conference**: NeurIPS 2025
**arXiv**: [2508.19104](https://arxiv.org/abs/2508.19104)
**Code**: [GitHub](https://github.com/shervinkhalafi/constrained_comp_align)
**Area**: Image Generation
**Keywords**: Diffusion Models, Constrained Optimization, Model Alignment, Model Composition, Lagrangian Duality

## TL;DR

This paper proposes a unified constrained optimization framework that formalizes reward alignment and multi-model composition of diffusion models as constrained optimization problems. By applying Lagrangian duality, the framework automatically determines optimal weights, eliminating the need for manual hyperparameter search.

## Background & Motivation

1. **Background**: Alignment (adapting to rewards/preferences) and composition (combining multiple pretrained models) are two widely used adaptation strategies for diffusion models, yet both face trade-offs arising from multi-objective conflicts.

2. **Limitations of Prior Work**: Alignment via weighted combinations of KL divergence and rewards requires manual weight tuning; improper weights lead to overfitting a single reward or excessive deviation from the pretrained model. Equal-weight composition may favor certain similar models while neglecting others.

3. **Key Challenge**: The search space of weighting methods is unintuitive—weights themselves carry no semantic meaning, whereas constraint-based specifications (e.g., "reward must reach at least $b$") are more natural and interpretable.

4. **Goal**: Provide a unified framework that automatically balances multiple objectives, eliminating the need for manual weight tuning.

5. **Key Insight**: Replace weighted optimization with constrained optimization: the alignment problem is reformulated as "minimize KL divergence from the pretrained model subject to reward constraints"; the composition problem is reformulated as "minimize the maximum KL divergence to all constituent models."

6. **Core Idea**: A constrained optimization framework unifies alignment and composition. Lagrangian duality automatically learns optimal weights, making the constraint threshold the only intuitive hyperparameter to specify.

## Method

### Overall Architecture

Two core problems: (1) **Constrained Alignment (UR-A)**: $\min_p D_{KL}(p\|q)$ s.t. $\mathbb{E}_{x\sim p}[r_i(x)] \geq b_i$; (2) **Constrained Composition (UR-C)**: $\min_{p,u} u$ s.t. $D_{KL}(p\|q^i) \leq u$. Both are converted into trainable primal-dual algorithms via Lagrangian duality.

### Key Designs

**1. Closed-Form Solution for Constrained Alignment (Theorem 1)**

- **Function**: Provides the closed-form optimal distribution for the constrained alignment problem.
- **Mechanism**: The optimal solution is a reward-tilted distribution $q_{rw}^{(\lambda^*)}(\cdot) = \frac{1}{Z}q(\cdot)e^{\lambda^{*\top}r(\cdot)}$, where the optimal dual variable $\lambda^*$ is determined automatically via dual ascent.
- **Design Motivation**: The theorem proves that constrained optimization is equivalent to exponential reward tilting of the pretrained distribution, with optimal weights determined automatically by the constraints.

**2. Tilted Product Distribution for Constrained Composition (Theorem 3)**

- **Function**: Automatically determines optimal weights for multi-model composition.
- **Mechanism**: The optimal solution is a tilted product distribution $q_{AND}^{(\lambda)}(\cdot) \propto \prod_{i=1}^m (q^i(\cdot))^{\lambda_i/\mathbf{1}^\top\lambda}$. KL constraints ensure the composed distribution deviates equidistantly from each pretrained model.
- **Design Motivation**: Equal-weight composition favors the most similar models (e.g., two similar Gaussians dominate a third); the constrained approach automatically balances across all models.

**3. Distinguishing Path-wise and Point-wise KL Divergence**

- **Function**: Selects the appropriate KL measure for each task.
- **Mechanism**: Path-wise KL measures discrepancy over entire diffusion trajectories and is suited for alignment; point-wise KL measures discrepancy in the final distribution and is more appropriate for composition. Lemma 2 introduces a novel method to compute point-wise KL divergence.
- **Design Motivation**: The two KL measures have distinct properties; choosing correctly affects both theoretical guarantees and practical performance.

### Loss & Training

Primal-dual alternating optimization: the primal step minimizes the Lagrangian (SGD), while the dual step updates multipliers via subgradient ascent proportional to constraint violation. LoRA is used to fine-tune Stable Diffusion v1.5.

## Key Experimental Results

### Main Results

Multi-reward constrained alignment (Stable Diffusion + MPS / Saturation / Local Contrast):

| Method | MPS Reward | Saturation Constraint | Contrast Constraint | KL from Pretrained |
|---|---|---|---|---|
| Equal-weight | Decreased | Partially satisfied | Partially satisfied | Larger |
| **Constrained Alignment** | **+50% improvement** | **Satisfied** | **Satisfied** | **Smaller** |

Multi-model composition (each model fine-tuned with a different reward):

| Method | Reward Retention Rate | Worst Reward |
|---|---|---|
| Equal-weight Composition | Some rewards drop by 30% | Notable degradation |
| **Constrained Composition** | All maintained or improved | No notable degradation |

### Ablation Study

| Configuration | Key Metric | Remarks |
|---|---|---|
| Constrained Alignment vs. Weighted Alignment | Constrained achieves smaller KL and satisfies all rewards | Weighted method tends to overfit easily optimized rewards |
| Constrained Composition vs. Equal-weight Composition | Constrained achieves higher minimum CLIP/BLIP scores | Equal-weight method favors similar models |

### Key Findings

- Weighted methods in multi-reward alignment tend to overfit easily optimizable rewards (e.g., saturation) while neglecting harder ones (e.g., HPS).
- The constrained method improves all rewards simultaneously while maintaining a smaller KL divergence from the pretrained model.
- In concept composition, the constrained method achieves higher minimum CLIP and BLIP scores compared to equal-weight composition.

## Highlights & Insights

- **Theoretical Rigor**: A complete theoretical chain from the distribution space to the diffusion model space, establishing strong duality.
- **Practicality**: Constraint thresholds are more intuitive than weights—"reward must improve by at least 50%" is easier to reason about than "set weight to 0.3."
- **Unification**: A single framework addresses alignment and composition, two seemingly distinct problems.

## Limitations & Future Work

- MCMC sampling is costly in high-dimensional settings such as image generation.
- Point-wise KL divergence computation depends on the quality of the score function.
- Validation is currently limited to Stable Diffusion v1.5; scaling to larger models warrants further investigation.
- The framework is extensible to mixture composition (OR mode) and conditional constraints.

## Related Work & Insights

- The AlignProp framework is extended to a multi-reward constrained variant.
- Compared to alignment methods such as DPO, the constrained approach does not require manual tuning of regularization weights.
- Insight: Constrained optimization offers a more interpretable and easier-to-tune alternative to weighted optimization.

## Rating

- Novelty: ⭐⭐⭐⭐ The constrained perspective unifying alignment and composition is original.
- Experimental Thoroughness: ⭐⭐⭐ Moderate scale, but core claims are validated.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear and well-presented.
- Value: ⭐⭐⭐⭐ Provides a principled framework for adapting diffusion models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Constrained Discrete Diffusion](constrained_discrete_diffusion.md)
- [\[NeurIPS 2025\] Training-Free Constrained Generation with Stable Diffusion Models](training-free_constrained_generation_with_stable_diffusion_models.md)
- [\[NeurIPS 2025\] Information Theoretic Learning for Diffusion Models with Warm Start](information_theoretic_learning_for_diffusion_models_with_warm_start.md)
- [\[NeurIPS 2025\] Exploring Semantic-constrained Adversarial Example with Instruction Uncertainty Reduction](exploring_semantic-constrained_adversarial_example_with_instruction_uncertainty_.md)
- [\[NeurIPS 2025\] Physics-Constrained Flow Matching: Sampling Generative Models with Hard Constraints](physics-constrained_flow_matching_sampling_generative_models_with_hard_constrain.md)

</div>

<!-- RELATED:END -->

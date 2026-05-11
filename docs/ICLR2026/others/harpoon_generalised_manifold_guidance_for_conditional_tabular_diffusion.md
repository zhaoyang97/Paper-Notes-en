---
title: >-
  [Paper Note] Harpoon: Generalised Manifold Guidance for Conditional Tabular Diffusion
description: >-
  [ICLR 2026][Tabular data] This paper extends manifold theory from image to tabular diffusion models, proving that the gradient of any differentiable inference-time loss lies in the tangent space of the data manifold (bey…
tags:
  - "ICLR 2026"
  - "Tabular data"
  - "manifold guidance"
  - "conditional generation"
  - "inference-time guidance"
  - "inequality constraints"
date: 2026-05-08
content_hash: da8f22e950625d2e
---

# Harpoon: Generalised Manifold Guidance for Conditional Tabular Diffusion

**Conference**: ICLR 2026
**arXiv**: [2602.07875](https://arxiv.org/abs/2602.07875)
**Code**: [GitHub](https://github.com/adis98/Harpoon)
**Area**: Diffusion Models / Tabular Data
**Keywords**: Tabular data, manifold guidance, conditional generation, inference-time guidance, inequality constraints

## TL;DR
This paper extends manifold theory from image to tabular diffusion models, proving that the gradient of any differentiable inference-time loss lies in the tangent space of the data manifold (beyond the square-error loss restriction). Based on this result, the proposed Harpoon method guides unconditional samples at inference time along the manifold to satisfy diverse tabular constraints.

## Background & Motivation

**Background**: Tabular diffusion models can generate high-quality tabular data, but conditional generation (missing value imputation, inequality constraints, etc.) is a core requirement. Existing conditional approaches fall into two categories: training-time methods (which struggle to generalise to unseen constraints) and inference-time methods (which are limited to imputation tasks).

**Limitations of Prior Work**: (1) Training-time methods (conditional input, classifier guidance, classifier-free guidance) cannot generalise to constraints unseen during training; (2) inference-time methods support only imputation and not inequality constraints; (3) the manifold theory developed for image diffusion assumes continuous features and flat geometry, making it unsuitable for mixed-type tabular data.

**Key Challenge**: A method that trains once and adapts to arbitrary constraints at inference time is needed, yet existing manifold guidance theory provides guarantees only for square-error loss on flat manifolds.

**Key Insight**: The paper establishes two stronger theoretical results: (1) Theorem 3.1: the denoising map $Q_t$ converges to the orthogonal projection onto the manifold as $\bar{\alpha}_t \to 1$ (without assuming flatness); (2) Theorem 3.2: the gradient of any differentiable loss lies in the tangent space (beyond the square-error restriction).

**Core Idea**: By proving that the gradient of any differentiable inference-time objective aligns with the data manifold, the method alternates between unconditional denoising steps and tangential corrections to satisfy diverse constraints.

## Method

### Overall Architecture
A single unconditional diffusion model is trained; at inference time, each step alternates between: (1) an unconditional denoising step and (2) a tangential correction using the gradient of the inference-time loss $\mathcal{L}_{\text{inf}}$. The framework supports imputation, inequality constraints, and other diverse conditions.

### Key Designs

1. **Theorem 3.1 (Orthogonal Projection)**:

   - Statement: An MSE-trained denoiser is equivalent to the orthogonal projection onto the manifold $\mathcal{M}_0$ as $\bar{\alpha}_t \to 1$.
   - Contribution: Generalises the result of Chung et al.—the flat-manifold assumption is no longer required; the result holds for curved manifolds.
   - Practical Implication: The "dirty estimate" $\hat{x}_0 = Q_t(x_t)$ lies on the manifold.

2. **Theorem 3.2 (Tangent-Space Gradient)**:

   - Statement: For any differentiable inference-time loss $\mathcal{L}_{\text{inf}}$, its gradient $\nabla_{x_t}\mathcal{L}_{\text{inf}}(\hat{x}_0, c) \in T_{\hat{x}_0}\mathcal{M}_0$.
   - Contribution: Extends the guarantee from square-error loss only to arbitrary differentiable losses (cross-entropy, L1, ReLU inequality, etc.).
   - Practical Implication: Gradient-based corrections using any reasonable inference-time loss will not push samples off the manifold.

3. **Harpoon Algorithm**:

   - Function: Performs unconditional denoising followed by tangential correction at each step.
   - Mechanism: $x_{t-1} = x_{t-1}' - \eta \cdot \nabla_{x_t}\mathcal{L}_{\text{inf}}(\hat{x}_0, c)$
   - Supported Constraints: Imputation (partial observations), range constraints (Age $\geq$ 10), categorical constraints (Gender = Male), conjunctions/disjunctions.

### Loss & Training
- Training: Standard MSE denoising loss (trained once).
- Inference-time loss options: MAE (default; sparsity-inducing properties suit tabular data), MSE, cross-entropy, ReLU inequality loss.
- Guidance strength $\eta$ controls the degree of constraint satisfaction.

## Key Experimental Results

### Main Results — Imputation (MAR, 50% Missing)

| Method | Adult | Bean | California | Magic | Average |
|--------|-------|------|-----------|-------|---------|
| GAIN | 1.86 | 1.41 | 15.06 | 1.27 | High |
| DiffPuter (Prev. SOTA) | Mid | Mid | Mid | Mid | Mid |
| Harpoon | **Low** | **Low** | **Low** | **Low** | **SOTA** |

### Inequality Constraints

| Constraint Type | Violation Rate↓ | α-score↑ | Utility↑ |
|----------------|----------------|----------|---------|
| Range constraint | **Lowest** | High | High |
| Categorical constraint | **Lowest** | High | High |
| Conjunction (and) | **Lowest** | High | High |
| Disjunction (or) | **Lowest** | High | High |

### Key Findings
- Experiments confirm that inference-time gradients are approximately orthogonal to the dirty estimate (~90°), even at larger timesteps.
- Different inference-time losses (MSE/MAE/CE) behave consistently under the same training objective, empirically validating Theorem 3.2.
- MAE loss performs best for tabular data due to its sparsity-inducing properties, which suit discrete features.
- A single trained model supports multiple inference-time constraints, offering far greater flexibility than training-time conditional methods.

## Highlights & Insights
- **Theory is the Core Contribution**: The two theorems substantially extend the manifold guidance theory developed for image diffusion—covering curved manifolds and arbitrary differentiable losses. These results carry implications beyond tabular data for other modalities.
- **"Train Once, Any Constraint"**: Training an unconditional model and adding arbitrary constraints at inference time represents an ideal paradigm for conditional generation. Harpoon demonstrates that this is feasible for tabular data with theoretical guarantees.
- **MAE over MSE**: The finding that L1 loss better suits tabular data due to the sparsity-inducing properties appropriate for discrete features is a domain-specific insight with practical value.

## Limitations & Future Work
- The orthogonal projection guarantee holds strictly only as $\bar{\alpha}_t \to 1$; deviations may occur at larger timesteps in practice.
- The continuous embedding of tabular data (e.g., one-hot encoding) is approximate, and a discrete–continuous gap remains.
- The guidance strength $\eta$ requires tuning.
- Validation is limited to UCI datasets; scalability to larger tabular data is unknown.

## Related Work & Insights
- **vs. DiffPuter**: DiffPuter is a training-time conditional method, whereas Harpoon operates at inference time—the former is more specialised, the latter more flexible.
- **vs. Chung et al.'s Image Manifold Guidance**: Harpoon extends the theory (curved manifolds, arbitrary losses) and applies it to tabular data for the first time.
- **vs. CTGAN/TabDDPM**: These methods do not support inference-time conditioning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The extension of manifold theory is an important theoretical contribution, and the adaptation to tabular data is natural.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multiple datasets, multiple tasks (imputation + inequality constraints), and theoretical validation.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theoretical derivations are clear and intuitive explanations are well presented.
- Value: ⭐⭐⭐⭐ — Theoretical impact extends beyond the tabular domain and carries general significance for diffusion model guidance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ASAG: Toward the Frontiers of Reliable Diffusion Sampling via Adversarial Sinkhorn Attention Guidance](../../AAAI2026/others/toward_the_frontiers_of_reliable_diffusion_sampling_via_adversarial_sinkhorn_att.md)
- [\[ICLR 2026\] Compositional Diffusion with Guided Search for Long-Horizon Planning](compositional_diffusion_long_horizon_planning.md)
- [\[AAAI 2026\] Tab-PET: Graph-Based Positional Encodings for Tabular Transformers](../../AAAI2026/others/tab-pet_graph-based_positional_encodings_for_tabular_transformers.md)
- [\[AAAI 2026\] Local Guidance for Configuration-Based Multi-Agent Pathfinding](../../AAAI2026/others/local_guidance_for_configuration-based_multi-agent_pathfinding.md)
- [\[NeurIPS 2025\] Kernel Conditional Tests from Learning-Theoretic Bounds](../../NeurIPS2025/others/kernel_conditional_tests_from_learning-theoretic_bounds.md)

</div>

<!-- RELATED:END -->

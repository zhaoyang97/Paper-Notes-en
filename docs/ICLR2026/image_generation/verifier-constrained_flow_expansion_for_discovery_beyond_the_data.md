---
title: >-
  [Paper Note] Verifier-Constrained Flow Expansion for Discovery Beyond the Data
description: >-
  [ICLR 2026][Image Generation][Flow Expansion] This paper proposes Flow Expander (FE), which expands the coverage of pretrained flow models in probability space via verifier-constrained entropy maximization, enabling the generation of design samples beyond the training data distribution while maintaining validity. FE increases diversity in molecular conformation design while preserving chemical validity.
tags:
  - ICLR 2026
  - Image Generation
  - Flow Expansion
  - Verifier Constraint
  - Entropy Maximization
  - Mirror Descent
  - Molecular Design
date: 2026-05-08
content_hash: 866b11c17cb3623e
---

# Verifier-Constrained Flow Expansion for Discovery Beyond the Data

**Conference**: ICLR 2026
**arXiv**: [2602.15984](https://arxiv.org/abs/2602.15984)
**Code**: None
**Area**: Flow Models / Scientific Discovery
**Keywords**: Flow Expansion, Verifier Constraint, Entropy Maximization, Mirror Descent, Molecular Design

## TL;DR
This paper proposes Flow Expander (FE), which expands the coverage of pretrained flow models in probability space via verifier-constrained entropy maximization, enabling the generation of design samples beyond the training data distribution while maintaining validity. FE increases diversity in molecular conformation design while preserving chemical validity.

## Background & Motivation

**State of the Field**: Flow models and diffusion models are trained via divergence minimization and thus cover only a small subset of the design space corresponding to the training data distribution. Scientific discovery tasks (e.g., drug design, materials design) require exploration beyond the data distribution while remaining valid.

**Limitations of Prior Work**: (1) Pretrained flow models concentrate on high-density regions, and low-probability regions may correspond to invalid designs; (2) manifold exploration methods (e.g., density rebalancing) lose validity signals in data-sparse regions; (3) there is no principled approach to leveraging external verifiers (e.g., atomic bond checkers) to guide exploration.

**Root Cause**: Exploring beyond the data distribution requires increased coverage (entropy maximization), yet unconstrained expansion produces invalid designs. A balance between expansion and validity must be achieved.

**Paper Goals**: How can a pretrained flow model be adapted using a given verifier to expand its density beyond high-data-availability regions while preserving sample validity?

**Starting Point**: The paper formalizes the notions of strong and weak verifiers, and proposes separate mathematical frameworks for global and local flow expansion corresponding to each case.

**Core Idea**: Principled expansion of pretrained flow models is achieved through verifier-constrained entropy maximization and Mirror Descent optimization in noise space.

## Method

### Overall Architecture
Strong verifiers ($\Omega_v = \Omega$, fully characterizing the valid space) and weak verifiers ($\Omega_v \supset \Omega$, serving only as filters) are defined. Global expansion applies to strong verifiers (target: uniform distribution over the valid space); local expansion applies to weak verifiers (target: constrained local expansion).

### Key Designs

1. **Global Flow Expansion (Problem 5)**:

    - Under a strong verifier, solve $\pi^* = \arg\max_{\pi} \mathcal{H}(p_1^\pi)$ s.t. $\mathbb{E}_{x \sim p_1}[v(x)] = 1$, $p_0^\pi = p_0^{\text{pre}}$
    - The optimal solution is the uniform distribution over the valid design space $p_1^{\pi^*} = \mathcal{U}(\Omega)$
    - This is independent of the pretrained model, since the strong verifier fully characterizes the valid space

2. **Local Flow Expansion (Problem 7)**:

    - Under a weak verifier, introduce KL regularization: $\max_\pi \mathcal{H}(p_1^\pi) - \alpha D_{\text{KL}}(p_1^\pi \| p_1^{\text{pre}})$ s.t. $\mathbb{E}[v(x)] = 1$
    - The KL term prevents the model from assigning density to invalid regions undetectable by the weak verifier
    - $\alpha$ controls conservatism: large $\alpha$ → distribution close to the pretrained model

3. **Flow Expander Algorithm (ExpandThenProject)**:

    - **Expansion step**: unconstrained expansion via noise-space optimization (Eq. 15) using running cost $f_t = \lambda_t \delta\mathcal{G}_t$
    - **Projection step**: constrained via reward-guided fine-tuning (Eq. 16) using verifier $\log v$
    - Alternated for $K$ iterations

4. **Closed-Form Gradient Expressions**:

    - Global FE: $\nabla_x \delta\mathcal{G}_t = -s_t^\pi$ (negative score function)
    - Local FE: $\nabla_x \delta\mathcal{G}_t = -s_t^\pi - \alpha_t(s_t^\pi - s_t^{\text{pre}})$
    - The score is approximated from the flow velocity field via a linear transform: $s_t^\pi(x) = \frac{1}{\kappa_t(\frac{\dot{\omega}_t}{\omega_t}\kappa_t - \dot{\kappa}_t)}(\pi(x,t) - \frac{\dot{\omega}_t}{\omega_t}x)$

5. **Noise Space Exploration (NSE)**:

    - A byproduct of FE with the projection step removed
    - Leverages score information from the entire flow trajectory (rather than only the terminal $t=1$), resolving the divergence issue of $s_1^\pi$
    - Outperforms existing flow exploration methods in high-dimensional settings

### Theoretical Guarantees
- Proposition 1: ExpandThenProject solves the MD step exactly
- Theorem 5.1 (idealized): finite-time convergence under exact updates $D_{\text{KL}}(\mathbf{Q}^* \| \mathbf{Q}^K) \leq \frac{C}{K}$
- Theorem 5.2 (general setting): asymptotic convergence under approximate updates given mild noise/bias assumptions

## Key Experimental Results

### Synthetic Experiments (Visual Verification)
- FE successfully expands the pretrained distribution to cover the entire valid region
- NSE demonstrates significantly better stability than existing methods in high-dimensional settings

### Molecular Design Experiments
- FE increases conformational diversity while better preserving validity compared to existing flow exploration methods
- The weak verifier (atomic bond checker) effectively filters invalid conformations
- Combining multiple weak verifiers $\Omega_v = \bigcap_i \Omega_{v_i}$ further tightens the valid region

### Key Findings
- Noise-space exploration (using the full trajectory rather than terminal scores) substantially improves stability in high dimensions
- The verifier-constrained projection step is critical — unconstrained expansion produces a large proportion of invalid samples
- The choice of $\alpha$ should reflect verifier quality: strong verifier → small $\alpha$; weak verifier → large $\alpha$

## Highlights & Insights
- **Elegant Problem Formulation**: The strong/weak verifier distinction and the corresponding global/local expansion framework are conceptually clear and practically useful
- **Theoretical Rigor**: The theoretical chain from continuous-time RL to Mirror Descent is complete, with solid convergence guarantees
- **Noise-Space Optimization as a Key Contribution**: This resolves the practical problem of terminal score divergence, and NSE itself is a valuable byproduct
- **General Framework**: Applicable to any scientific discovery task with an available verifier

## Limitations & Future Work
- The approximation quality of the score function affects practical performance, requiring high-quality pretrained models
- Automatic tuning mechanisms for $\alpha_t$ and $\lambda_t$ are lacking
- The molecular design experiments are relatively small-scale; large-scale evaluation remains to be conducted
- Learned verifiers (e.g., GNNs) could be explored as alternatives to hand-crafted rules

## Related Work & Insights
- **vs. De Santi et al. 2025**: That work uses only the terminal score $s_1^\pi$ for exploration, which suffers from divergence; FE stabilizes exploration by leveraging the full trajectory
- **vs. reward-guided fine-tuning**: FE additionally incorporates verifier constraints to prevent expansion into invalid regions
- **Continuous-time RL perspective**: Unifying flow model fine-tuning as an optimal control problem is a notable conceptual contribution

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Verifier-constrained flow expansion is an entirely new problem with a complete theoretical framework
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic and molecular design experiments are provided, though large-scale validation is pending
- Writing Quality: ⭐⭐⭐⭐ Theory-dense but logically clear, with effective illustrations
- Value: ⭐⭐⭐⭐⭐ Significant contribution to the application of generative models in scientific discovery

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Pseudo-Nonlinear Data Augmentation: A Constrained Energy Minimization Viewpoint](pseudo-nonlinear_data_augmentation_a_constrained_energy_minimization_viewpoint.md)
- [\[ICLR 2026\] Beyond Confidence: The Rhythms of Reasoning in Generative Models](beyond_confidence_the_rhythms_of_reasoning_in_generative_models.md)
- [\[ICLR 2026\] VFScale: Intrinsic Reasoning through Verifier-Free Test-time Scalable Diffusion Model](vfscale_intrinsic_reasoning_through_verifier-free_test-time_scalable_diffusion_m.md)
- [\[ICLR 2026\] Infinity and Beyond: Compositional Alignment in VAR and Diffusion T2I Models](infinity_and_beyond_compositional_alignment_in_var_and_diffusion_t2i_models.md)
- [\[CVPR 2026\] Beyond the Golden Data: Resolving the Motion-Vision Quality Dilemma via Timestep Selective Training](../../CVPR2026/image_generation/beyond_the_golden_data_resolving_the_motion-vision_quality_dilemma_via_timestep_.md)

<!-- RELATED:END -->

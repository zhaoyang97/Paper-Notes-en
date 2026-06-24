---
title: >-
  [Paper Note] Neural Stochastic Differential Equations on Compact State Spaces: Theory, Methods and Applications
description: >-
  [ICML 2025][Medical Imaging][Neural SDEs] This paper proposes a Neural SDE parameterization method (WSP) based on stochastic viability theory, ensuring that SDE trajectories are provably constrained within compact polytopic spaces with continuous dynamics and a strong inductive bias, overcoming the limitations of chain-rule methods and reflected SDEs.
tags:
  - "ICML 2025"
  - "Medical Imaging"
  - "Neural SDEs"
  - "Compact State Spaces"
  - "Stochastic Viability Theory"
  - "Inductive Bias"
  - "Polytopes"
date: 2026-05-08
content_hash: 8bc6a6054ff68074
---

# Neural Stochastic Differential Equations on Compact State Spaces: Theory, Methods and Applications

**Conference**: ICML 2025  
**arXiv**: [2508.17090](https://arxiv.org/abs/2508.17090)  
**Code**: None  
**Area**: Dynamical Systems / Neural SDEs  
**Keywords**: Neural SDEs, Compact State Spaces, Stochastic Viability Theory, Inductive Bias, Polytopes

## TL;DR
This paper proposes a Neural SDE parameterization method (WSP) based on stochastic viability theory, ensuring that SDE trajectories are provably constrained within compact polytopic spaces with continuous dynamics and a strong inductive bias, overcoming the limitations of chain-rule methods and reflected SDEs.

## Background & Motivation
**Background**: SDEs are powerful probabilistic modeling tools underpinning continuous-time time series, diffusion models, infinite-depth networks, etc. However, training SDEs under non-linear dynamics remains unstable.

**Limitations of Prior Work**: (a) Simplified dynamics combined with training tricks (e.g., KL annealing) reduce practicality and interpretability; (b) Reflected SDEs (RSDEs) constrain trajectories within compact spaces but exhibit discontinuous dynamics and lack high-order solvers; (c) Chain-rule methods (Ito's Lemma / sigmoid transformations) lead to numerical instability or boundary sticking.

**Key Challenge**: SDEs in compact spaces require special treatment at the boundaries—RSDEs employ discontinuous reflections, whereas sigmoid transformations cause dynamics to vanish at the boundaries.

**Key Insight**: Leveraging theorems from stochastic viability theory (Milian 1995) to derive the necessary and sufficient conditions that drift and diffusion must satisfy on polytopic boundaries.

**Core Idea**: A Weighted Sums Parameterization (WSP) is proposed: utilizing arbitrary neural network dynamics in the interior while smoothly transitioning near the boundary to a simple function that satisfies the constraints.

## Method

### Overall Architecture
Input: Compact polytopic space $K$ + arbitrary unconstrained drift $\tilde{h}$ and diffusion $\tilde{g}$ $\rightarrow$ WSP transforms them into constrained dynamics $h, g$ satisfying the viability conditions $\rightarrow$ Guarantees $\mathbb{P}(z_t \in K) = 1$ $\rightarrow$ Optional: Derive a stationary drift from $g$ such that the SDE has a specified stationary distribution.

### Key Designs

1. **Viability Conditions on Polytopic Spaces (Theorem 3.2)**:

    - Condition (a): The drift at the boundary must point inward $\langle h(t,z_t), v_s \rangle \geq 0$
    - Condition (b): The diffusion at the boundary must be zero $\langle g(t,z_t) \odot e_d, v_s \rangle = 0$
    - Combined with Lipschitz continuity and linear growth conditions
    - Design Motivation: These are the necessary and sufficient conditions for the SDE to remain viable in $K$

2. **Weighted Sums Parameterization (WSP)**:

    - $\text{WSP}(f, c, t, z) = w(z) \cdot f(t,z) + (1-w(z)) \cdot c(z)$
    - $w(z) \in [0,1]$: boundary $\rightarrow$ 0 (using constrained function $c$), interior $\rightarrow$ 1 (using free function $f$)
    - $w(z)$ is constructed based on the distance to each boundary: $w(z) = \tanh(\beta \prod_s \frac{e^{-d(u_s,v_s,z)}}{\sum_{s'} e^{-d(u_{s'},v_{s'},z)}} \cdot \tanh(\alpha \cdot d(u_s,v_s,z)))$
    - Drift constraint: $c_h(z) = \gamma \cdot \frac{z^* - z}{\|z^* - z\| + \epsilon}$ (pushing towards the Chebyshev center)
    - Diffusion constraint: $c_g(z) = 0$ (noise vanishes at the boundary)
    - Design Motivation: The smooth transition avoids the discontinuity of RSDEs and the boundary sticking of sigmoid transformations

3. **Stationary SDE (Theorem 3.3)**:

    - Given diffusion $g$ and target distribution $\tilde{p}$, the closed-form solution of drift is: $h(z_t) = \frac{1}{2} \text{diag}(\nabla_{z_t}[g(z_t)^2]) + \frac{1}{2} g(z_t)^2 \odot \nabla_{z_t} \log \tilde{p}(z_t)$
    - Proves that this drift satisfies all conditions of Theorem 3.2
    - Design Motivation: Automatically deriving dynamics that drive the SDE to converge to a specified distribution

### Loss & Training
- Compatible with standard SDE inference frameworks (variational inference, score matching, etc.)
- WSP only modifies the parameterization of the dynamics without altering the training objectives

## Key Experimental Results

### Main Results (Inductive Bias Comparison, $K=[0,1]$, NN with Random Weights)

| Method | Inductive Bias | Explanation |
|------|---------|------|
| Unconstrained SDE (Eq.1) | Rapidly leaves $K$ | NN drift/diffusion do not satisfy boundary conditions |
| Sigmoid Transformation (Eq.2,3) | Stuck at boundaries | The $(z-z^2)$ factor vanishes at boundaries |
| WSP (Eq.5) | ✓ Successfully remains within $K$ | Continuous dynamics + strong inductive bias |

### Ablation Study

| Configuration | Key Metric | Explanation |
|------|---------|------|
| Ito SDE + WSP | Viability ✓ | Ito-Milstein solver |
| Stratonovich SDE + WSP | Viability ✓| Pathwise expansion + ODE solver |
| Stationary WSP + Various Target Distributions | Matching ✓ | Sin/cubic/normal distributions are all matched |
| Different Polytopes (Rectangular/Triangular) | Viability ✓ | WSP is valid for any compact polytope |

### Key Findings
- WSP is the only method that simultaneously satisfies continuous dynamics, viability in compact spaces, and strong inductive bias.
- The "sticking" of sigmoid transformations at boundaries is fundamental, as the $z(1-z)$ factor vanishes at 0/1.
- Stationary WSP automatically matches arbitrary target marginal distributions over time.
- ODE solvers (via pathwise expansion) achieve consistent performance with SDE solvers.

## Highlights & Insights
- **Theoretical Depth**: Deriving necessary and sufficient conditions from stochastic viability theory, followed by designing a parameterization that satisfies these conditions.
- **Unified Solution to Three Challenges**: Addresses theoretical infeasibility $\rightarrow$ numerical instability $\rightarrow$ poor inductive bias, all resolved at once by WSP.
- **Stationary SDE Derivation**: Closed-form drift = score function + diffusion correction, both elegant and practical.
- **Application Motivation**: Target real-world problems requiring compact state space modeling, such as mental health time series.

## Limitations & Future Work
- Currently only validated for inductive bias (random-weighted NNs), without end-to-end task training.
- It remains uncertain whether the gradient landscape of WSP is optimization-friendly.
- The setting of hyperparameters $\alpha, \beta$ in $w(z)$ may affect training dynamics.
- Only polytopic spaces are considered; more general compact spaces like spheres or manifolds are not yet addressed.

## Related Work & Insights
- Reflected SDEs (Pilipenko 2014) are theoretically sound but possess discontinuous dynamics.
- Constrained diffusion models (Lou & Ermon 2023, Fishman 2023) provide practical application motivation.
- Neural SDEs (Kidger 2021, Li et al. 2020) provide the foundational framework.
- Insight: Integrating the mathematical theory of stochastic differential equations more closely into deep learning can fundamentally improve model properties.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Outstanding theoretical contribution by integrating stochastic viability theory with Neural SDEs.
- Experimental Thoroughness: ⭐⭐⭐ Primarily focused on inductive bias validation, lacking task-level experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous and complete theoretical derivations with detailed proofs in the appendix.
- Value: ⭐⭐⭐⭐ Provides a solid theoretical foundation for SDE modeling on compact spaces.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Generalizable, Real-Time Neural Decoding with Hybrid State-Space Models](../../NeurIPS2025/medical_imaging/generalizable_real-time_neural_decoding_with_hybrid_state-space_models.md)
- [\[NeurIPS 2025\] GeoDynamics: A Geometric State-Space Neural Network for Understanding Brain Dynamics on Riemannian Manifolds](../../NeurIPS2025/medical_imaging/geodynamics_a_geometric_state-space_neural_network_for_understanding_brain_dynam.md)
- [\[ICCV 2025\] SciVid: Cross-Domain Evaluation of Video Models in Scientific Applications](../../ICCV2025/medical_imaging/scivid_cross-domain_evaluation_of_video_models_in_scientific_applications.md)
- [\[NeurIPS 2025\] Mind the (Data) Gap: Evaluating Vision Systems in Small Data Applications](../../NeurIPS2025/medical_imaging/mind_the_data_gap_evaluating_vision_systems_in_small_data_applications.md)
- [\[ICLR 2026\] Stochastic Optimal Control for Continuous-Time fMRI Representation Learning](../../ICLR2026/medical_imaging/stochastic_optimal_control_for_continuous-time_fmri_representation_learning.md)

</div>

<!-- RELATED:END -->

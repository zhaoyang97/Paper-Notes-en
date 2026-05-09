---
title: >-
  [Paper Note] One Operator to Rule Them All? On Boundary-Indexed Operator Families in Neural PDE Solvers
description: >-
  [ICLR 2026][Scientific Computing][Neural Operator] This paper argues that neural PDE solvers, when trained under varying boundary conditions, do not learn a single solution operator but rather a family of operators indexed by boundary conditions. It formalizes the non-identifiability problem induced by boundary distribution shift under ERM from a learning-theoretic perspective.
tags:
  - ICLR 2026
  - Scientific Computing
  - Neural Operator
  - Boundary Conditions
  - Distribution Shift
  - Non-Identifiability
  - Fourier Neural Operator
date: 2026-05-08
content_hash: b6c96f8047332882
---

# One Operator to Rule Them All? On Boundary-Indexed Operator Families in Neural PDE Solvers

**Conference**: ICLR 2026
**arXiv**: [2603.01406](https://arxiv.org/abs/2603.01406)
**Code**: [Available](https://github.com/lennonshikhman/boundary-indexed-neural-pde)
**Area**: Scientific Computing / Neural PDE Solvers
**Keywords**: Neural Operator, Boundary Conditions, Distribution Shift, Non-Identifiability, Fourier Neural Operator

## TL;DR

This paper argues that neural PDE solvers, when trained under varying boundary conditions, do not learn a single solution operator but rather a family of operators indexed by boundary conditions. It formalizes the non-identifiability problem induced by boundary distribution shift under ERM from a learning-theoretic perspective.

## Background & Motivation

Neural PDE solvers (e.g., FNO) are commonly described as "learning solution operators"—mappings from problem inputs to PDE solutions. However, from classical PDE theory, solution operators are defined not only by the differential equation itself; **boundary conditions** are central to well-posedness and uniqueness.

Existing methods typically encode boundary conditions implicitly (e.g., boundary padding, auxiliary channels), which raises a fundamental question: when boundary conditions are not fixed, what exactly are neural solvers approximating? The authors argue that the learned mappings are intrinsically tied to the boundary condition distribution seen during training, leading to unpredictable failures under boundary distribution shift—failures that stem neither from architectural limitations nor optimization issues.

## Method

### Overall Architecture

Rather than proposing a new architecture, this paper provides a **learning-theoretic framework** to reinterpret the learning behavior of neural PDE solvers:

- Operator learning is formalized as **conditional risk minimization**: $\min_\theta \mathbb{E}_{(f,\mathcal{B})\sim\mu}[\ell(\hat{\mathcal{S}}_\theta(f,\mathcal{B}), \mathcal{S}(f,\mathcal{B}))]$
- When boundary conditions $\mathcal{B}$ are fixed, the model learns a single operator $\mathcal{S}_\mathcal{B}: f \mapsto u$
- When $\mathcal{B}$ varies, the model learns a joint mapping $\mathcal{S}: (f, \mathcal{B}) \mapsto u$, but its behavior is only constrained on the support of the training distribution $\mu_\mathcal{B}$

### Key Designs

1. **Formalization of Boundary-Indexed Operator Families**

    Function: Proves that neural PDE solvers learn not a single boundary-agnostic operator, but a family of operators parameterized by boundary conditions.

    Mechanism: ERM only imposes constraints on the training distribution $\mu_\mathcal{B}$; for out-of-distribution boundary conditions, multiple distinct mappings can achieve the same training loss. This is formalized as non-identifiability: the learned mapping is non-unique outside the support of $\mu_\mathcal{B}$.

    Design Motivation: This explains why robustness to forcing functions does **not** imply robustness to boundary conditions—the former is densely sampled in the input space, while the latter may occupy only a low-dimensional sparse subspace.

2. **Conditional Expectation Degeneracy**

    Function: Analyzes how ERM-trained models degrade to the conditional expectation of the boundary condition when boundary information is absent or weakly represented.

    Mechanism: When boundary information is unavailable, the model's optimal prediction becomes $\hat{u}(f) \approx \mathbb{E}_{\mathcal{B}\sim\mu_\mathcal{B}}[\mathcal{S}(f,\mathcal{B}) \mid f]$, which is an average over unobserved boundary variables and does not correspond to a valid solution operator under any fixed boundary condition.

    Design Motivation: This theoretically explains why boundary-ablated models fail across all distributions—they learn an "averaged solution" rather than any specific one.

### Loss & Training

Experiments employ the FNO architecture with Adam optimizer (learning rate $8 \times 10^{-4}$), MSE loss, 2500 gradient update steps, batch size 12, trained on $64 \times 64$ grids. Boundary functions are parameterized via truncated Fourier expansions (bandwidth $K=6$).

## Key Experimental Results

### Main Results (Tables)

**Cross-Distribution Generalization (Poisson Equation)**:

| Model | Test $\mu_{B_0}$ | Test $\mu_{B_1}$ |
|-------|-------------------|-------------------|
| FNO (trained on $\mu_{B_0}$) | 0.078 ± 0.005 | **0.489 ± 0.022** |
| FNO (trained on $\mu_{B_1}$) | **0.601 ± 0.036** | 0.102 ± 0.003 |
| FNO (no boundary channel) | 0.999 ± 0.001 | 1.001 ± 0.001 |

Each model performs well only on its training boundary distribution; error surges 5–6× on the other distribution. Models without boundary channels fail completely across all distributions (error ≈ 1.0).

### Ablation Study (Tables)

**Boundary Extrapolation (Dirichlet Mean Shift)**:

| Shift $\delta$ | -1.0 | -0.5 | 0 | 0.5 | 1.0 |
|----------------|------|------|---|-----|-----|
| Error Trend | High | Medium | Low (in-domain) | Medium | High |

Error grows symmetrically with shift magnitude, exhibiting continuous degradation rather than abrupt failure. Frequency extrapolation (increasing Dirichlet bandwidth from $K=6$ to 12) similarly induces monotonic performance degradation.

### Key Findings

- **Cross-distribution failure is structural**: It is not caused by insufficient model capacity or optimization instability, but by an intrinsic limitation of the ERM objective—confirmed by the fact that in-domain performance remains strong under the same FNO architecture.
- **Conditional expectation behavior validated**: Fixing a forcing function $f^*$, the output of the boundary-ablated model almost exactly matches the Monte Carlo estimate of $\mathbb{E}[u \mid f^*]$.
- **Resolution robustness ≠ boundary robustness**: Models that generalize across grid resolutions still fail severely under boundary distribution shift.

## Highlights & Insights

- An exemplary contribution that advances understanding rather than proposing a new method—precisely characterizing a neglected fundamental problem using conditional risk minimization and non-identifiability theory.
- Offers a sober structural warning to the "foundation model for PDE" trend: simply scaling data and model capacity cannot resolve boundary generalization failures.
- The experimental design is elegant and controlled: by fixing the equation, forcing function, and resolution, it precisely isolates the effect of boundary conditions.

## Limitations & Future Work

- Experiments are limited to the two-dimensional Poisson equation and are not extended to parabolic, hyperbolic, or nonlinear PDEs.
- The interaction between boundary conditions and initial conditions in time-dependent systems is not considered.
- The paper provides a diagnostic framework but proposes no remedies (e.g., boundary-aware architectures or invariant representations).
- Variability across random initialization seeds is not explored.

## Related Work & Insights

- Complements prior work that observed boundary sensitivity and proposed boundary-aware architectures by providing theoretical grounding.
- Suggests future directions: treating boundary conditions as first-class citizens in operator learning design—e.g., conditional operator decomposition, structured boundary encoding, and invariant representation learning.
- Has important implications for the evaluation protocols of PDE foundation models (e.g., Subramanian et al., 2023): explicit testing of boundary distribution shift is necessary.

## Rating

⭐⭐⭐⭐ Theoretically insightful, with concise experiments that precisely validate the core argument. The work carries significant cautionary value for the neural PDE solver community, though it falls short of offering constructive solutions.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] NESTOR: A Nested MOE-based Neural Operator for Large-Scale PDE Pre-Training](../../CVPR2026/scientific_computing/nestor_a_nested_moe-based_neural_operator_for_large-scale_pde_pre-training.md)
- [\[ICLR 2026\] DRIFT-Net: A Spectral--Coupled Neural Operator for PDEs Learning](drift-net_a_spectral--coupled_neural_operator_for_pdes_learning.md)
- [\[ICCV 2025\] JPEG Processing Neural Operator for Backward-Compatible Coding](../../ICCV2025/scientific_computing/jpeg_processing_neural_operator_for_backward-compatible_coding.md)
- [\[NeurIPS 2025\] From Black Hole to Galaxy: Neural Operator Framework for Accretion and Feedback Dynamics](../../NeurIPS2025/scientific_computing/from_black_hole_to_galaxy_neural_operator_framework_for_accretion_and_feedback_d.md)
- [\[NeurIPS 2025\] Hamiltonian Neural PDE Solvers through Functional Approximation](../../NeurIPS2025/scientific_computing/hamiltonian_neural_pde_solvers_through_functional_approximation.md)

<!-- RELATED:END -->

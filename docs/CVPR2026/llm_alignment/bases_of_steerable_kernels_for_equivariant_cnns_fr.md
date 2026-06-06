---
title: >-
  [Paper Note] Bases of Steerable Kernels for Equivariant CNNs: From 2D Rotations to the Lorentz Group
description: >-
  [CVPR 2026][LLM Alignment][Steerable Kernels] This paper proposes a method that bypasses Clebsch-Gordan (CG) coefficient computation and directly constructs explicit steerable kernel bases from group representation matri…
tags:
  - "CVPR 2026"
  - "LLM Alignment"
  - "Steerable Kernels"
  - "Equivariant CNN"
  - "Symmetry Groups"
  - "Lorentz Group"
  - "Clebsch-Gordan Coefficients"
date: 2026-05-08
content_hash: fcbcf9cdbf697f0d
---

# Bases of Steerable Kernels for Equivariant CNNs: From 2D Rotations to the Lorentz Group

**Conference**: CVPR 2026
**arXiv**: [2603.12459](https://arxiv.org/abs/2603.12459)  
**Code**: N/A  
**Area**: Equivariant Neural Networks / Group Theory
**Keywords**: Steerable Kernels, Equivariant CNN, Symmetry Groups, Lorentz Group, Clebsch-Gordan Coefficients

## TL;DR

This paper proposes a method that bypasses Clebsch-Gordan (CG) coefficient computation and directly constructs explicit steerable kernel bases from group representation matrix elements. Through a three-step strategy of "stabilizer constraint + Schur's lemma + steering," it uniformly covers SO(2), O(2), SO(3), O(3), and the non-compact Lorentz group, substantially simplifying the kernel design pipeline for equivariant CNNs.

## Background & Motivation

**Current landscape**: Equivariant CNNs encode symmetry priors (e.g., rotational invariance) into network architectures, significantly improving performance and data efficiency on tasks such as molecular simulation, particle physics, and 3D vision. The core component is the steerable convolution kernel, which must satisfy the constraint $K(g \cdot x) = \rho_{\text{out}}(g) K(x) \rho_{\text{in}}(g)^{-1}$.

**Existing limitations**: The mainstream approach for solving this constraint (Lang et al. 2021) relies on computing Clebsch-Gordan (CG) coefficients, requiring repeated transformations between "coupled bases" and "uncoupled bases." For certain groups (especially non-compact groups such as the Lorentz group), CG coefficient computation is extremely difficult or infeasible. Alternative approaches (Finzi et al. 2021, Zhdanov et al. 2023) use MLP-based implicit parameterization but sacrifice the interpretability of analytic solutions.

**Core tension**: Theoretically, one only needs to find a basis for the kernel space satisfying a linear constraint, yet the introduction of CG coefficients turns a conceptually simple problem into a computationally complex one.

**Objective**: Can one bypass CG coefficients and directly derive ready-to-use steerable kernel bases from group representation matrix elements for arbitrary symmetry groups and arbitrary tensor-type feature maps?

**Approach**: The stabilizer subgroup at a point on the orbit reduces the global constraint to a local invariance condition, which is then solved directly via Schur's lemma.

**Core idea**: Solve the simplified constraint at a stabilizer-fixed point, then extend to the entire orbit through the steering operation; harmonic basis functions naturally emerge as representation matrix elements.

## Method

### Overall Architecture

The entire method consists of three steps: (1) select a reference point $x_0$ on a $G$-orbit and determine its stabilizer subgroup $H = \text{Stab}_{x_0}$; (2) reduce the steerable constraint to an invariance condition at $x_0$: $K(x_0) = \rho_j(h) K(x_0) \rho_l(h)^{-1}, \forall h \in H$, and solve the intertwiner basis directly using Schur's lemma on the irreducible decomposition of $H$; (3) extend the result to any point on the orbit via the steering operation $K(g \cdot x_0) = \rho_j(g) K(x_0) \rho_l(g)^{-1}$. While this idea has been mentioned sporadically in the literature, this paper is the first to systematically develop it across multiple groups and provide complete closed-form formulae.

### Key Designs

1. **Stabilizer Constraint Simplification and Schur's Lemma Solution**:

    - Purpose: Reduce the global steerable constraint to a local invariance condition at the reference point
    - Core idea: After fixing $x_0$, restrict $\rho_j$ and $\rho_l$ to $H$ to obtain reducible representations $\rho_j^H$ and $\rho_l^H$, then decompose them into direct sums of irreducible representations. By Schur's lemma, the block structure of the intertwiner $K(x_0)$ is almost fully determined—blocks between inequivalent irreducible representations are zero, and blocks between equivalent irreducible representations are proportional to the identity map (which may include two generators $\mathbb{I}$ and $J$ in the real case)
    - Design rationale: For SO(2), $H$ is just the identity, so the constraint is automatically satisfied and any matrix is a solution; for SO(3), $H \simeq$ SO(2), and the constraint is solved via diagonal block decomposition with free parameter dimension $2\min(j,l)+1$

2. **Steering Operation for Complete Kernel Bases**:

    - Purpose: Extend the intertwiner basis at $x_0$ to arbitrary points on the orbit
    - Core idea: For each basis element $T_m$, obtain the complete kernel function via $K_m(g \cdot x_0) = \rho_j(g) T_m \rho_l(g)^{-1}$. For complex representations of SO(3), the kernel basis matrix elements take the form $D^j_{m_j m}(g) D^l_{m m_l}(g)^{-1}$, where $D$ denotes Wigner-D matrices
    - Design rationale: Harmonic basis functions do not need to be pre-selected; they naturally emerge as representation matrix elements. Compared to traditional methods, this eliminates all coupled/uncoupled basis transformation steps

3. **Handling the Non-compact Lorentz Group**:

    - Purpose: Extend the method to the physically important Lorentz group $\text{SO}^+(1,3)$
    - Core idea: Two physical scenarios are distinguished. **Massive particles**: the orbit is a timelike hyperboloid with $H \simeq$ SO(3); decomposition uses SU(2) CG coefficients (rather than Lorentz group CG coefficients), and integer-spin intertwiners take the form of projection operators (e.g., $\Delta^\mu{}_\nu = \delta^\mu{}_\nu - u^\mu u_\nu$). **Massless particles**: the orbit is the light cone with $H \simeq$ ISO(2), considering only the SO(2) subgroup; the intertwiner is a transverse projection operator. Half-integer spins are handled through charge conjugation $\mathcal{C}$ and $\gamma$ matrices to construct quaternionic structures
    - Design rationale: Lorentz group CG coefficients are extremely complex (Bogatskiy et al. 2020); this paper completely bypasses this difficulty by directly constructing projection operators in the spacetime tensor representation, yielding results with clear physical interpretation

### Loss Function / Training Strategy

This paper is a purely theoretical contribution with no training procedure. The authors note that their approach allows independent truncation of input representation label $l$ and output representation label $j$, whereas the traditional coupled-basis approach can only truncate the single coupled label $J$. This flexibility may yield stronger network expressiveness but requires experimental validation.

## Key Experimental Results

### Main Experiments

This paper contains no numerical experiments. Correctness is verified through analytic derivation and systematic comparison with known results:

| Symmetry Group | Verification Content | Verification Result |
|---|---|---|
| SO(2) complex | Kernel basis $e^{i(j-l)\phi}$ | Exactly matches Weiler et al. (2019) Table 8 |
| SO(2) real | 4D kernel basis (trigonometric matrices) | Strictly reproduces all cases in Table 8 |
| O(2) real | Simplified basis under reflection constraint | Matches corresponding cases in Table 9 (including $\rho_{\tilde{0}}$ representation) |
| SO(3) real | Basis dimension $2\min(j,l)+1$ | Consistent with known theory |
| SO(3) complex vs real | Real parameter ratio $4\min(j,l)+2$ vs $2\min(j,l)+1$ | Unified explanation via charge conjugation symmetry |
| O(3) | Parity-classified intertwiners | $\min(l,j)+1$ complex intertwiners per pair of irreps |
| Lorentz group (massive) | Spin 0/1/2 projection operators | Physically intuitive forms such as $u^\mu u_\nu$, $\Delta^\mu{}_\nu$ |
| Lorentz group (massless) | Spin 1/2 transverse projection | Contains positive/negative energy projections $P_\pm(u)$ and quaternionic structure |

### Ablation Studies

| Configuration | Key Metric | Note |
|---|---|---|
| Complex vs real (SO(3)) | Real params: $4\min(j,l){+}2$ vs $2\min(j,l){+}1$ | Charge conjugation halves the parameters; both bases span the same solution space |
| Integer vs half-integer spin (Lorentz) | Intertwiner type: $\mathbb{R}$ vs $\mathbb{H}$ | Half-integer spins introduce quaternionic structure $\{𝕀, I, J, K\}$ |
| Independent vs coupled truncation | Flexibility | $j,l$ can be chosen independently, more flexible than single $J$ truncation |

### Key Findings

- In all compact group results, harmonic basis functions emerge automatically from representation matrix elements without pre-selection
- Integer-spin intertwiners for the Lorentz group reduce to projection operator forms with direct physical correspondence (energy-momentum decomposition)
- The parameter dimension difference between complex and real representations is unified through charge conjugation symmetry

## Highlights & Insights

- **Extreme simplicity**: The full derivation for SO(2) requires only a few lines—the stabilizer constraint is automatically satisfied, and steering yields $e^{i(j-l)\phi}$ in one step, in stark contrast to traditional approaches that require basis selection followed by decomposition
- **Theoretical unification**: The same three-step framework (select point → Schur's lemma → steering) seamlessly covers from the simplest 2D rotations to the non-compact Lorentz group, demonstrating remarkable methodological elegance
- **Clear physical intuition**: Results for the Lorentz group naturally take projection operator form ($u^\mu u_\nu$, $\Delta^\mu{}_\nu$), interfacing perfectly with standard tools in quantum field theory
- **Independent truncation flexibility**: Traditional methods truncate on the coupled label $J$; this paper allows independent truncation on input/output representation labels $j,l$, providing greater design freedom for networks

## Limitations & Future Work

- This is a purely theoretical paper with no numerical experiments verifying actual performance gains on vision or physics tasks
- No quantitative comparison of computational efficiency or memory overhead with existing equivariant frameworks (e3nn, escnn)
- Only completely reducible representations are considered; more general irreducible representation cases for non-compact groups are not discussed
- Aliasing handling and discretization implementation details are deferred to future work
- The expressiveness advantage of independent $j,l$ truncation requires experimental validation

## Related Work & Inspiration

- **vs Lang et al. (2021)**: The latter requires CG coefficient computation + coupled basis transformation + inverse transformation in three steps; this paper achieves results in one step directly from representation matrix elements, with clearer concepts and simpler computation
- **vs Weiler et al. (2019, E2CNN)**: Both obtain the same solution vector space but with different basis elements; this paper obtains results more directly through the steering operation
- **vs Bogatskiy et al. (2020, LorentzNet)**: The latter computes full CG coefficients for the Lorentz group (difficult and unintuitive); this paper bypasses this via SO(3) subgroup decomposition + projection operators
- **vs Finzi et al. (2021), Zhdanov et al. (2023/2024)**: The latter implicitly parameterize steerable kernels via MLPs (applicable to arbitrary matrix groups); this paper provides explicit analytic solutions with stronger interpretability
- Provides more practical kernel construction tools for building Lorentz equivariant networks in particle physics simulation

## Rating

- Novelty: ⭐⭐⭐⭐ The core idea has been sporadically mentioned in the literature, but this is the first systematic development to the Lorentz group with complete closed-form formulae
- Experimental rigor: ⭐⭐ A purely theoretical work with no numerical experiments, though cross-validation with known analytic results is thorough
- Writing quality: ⭐⭐⭐⭐ Rigorous mathematical derivation with a progressive exposition that is accessible to non-representation-theory experts
- Impact: ⭐⭐⭐ Provides more direct theoretical tools for equivariant network researchers, but the lack of experimental validation limits current practical impact

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MetaGDPO: Alleviating Catastrophic Forgetting with Metacognitive Knowledge through Group Direct Preference Optimization](../../AAAI2026/llm_alignment/metagdpo_alleviating_catastrophic_forgetting_with_metacognitive_knowledge_throug.md)
- [\[NeurIPS 2025\] GVPO: Group Variance Policy Optimization for Large Language Model Post-Training](../../NeurIPS2025/llm_alignment/gvpo_group_variance_policy_optimization_for_large_language_model_post-training.md)
- [\[ICLR 2026\] Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm: Demystifying Some Myths About GRPO and Its Friends](../../ICLR2026/llm_alignment/group-relative_reinforce_is_secretly_an_off-policy_algorithm_demystifying_some_m.md)
- [\[CVPR 2026\] LocalDPO: Direct Localized Detail Preference Optimization for Video Diffusion Models](mind_the_generative_details_direct_localized_detail_preference_optimization_for_.md)
- [\[CVPR 2026\] Principled Steering via Null-space Projection for Jailbreak Defense in Vision-Language Models](principled_steering_via_null-space_projection_for_jailbreak_defense_in_vision-la.md)

</div>

<!-- RELATED:END -->

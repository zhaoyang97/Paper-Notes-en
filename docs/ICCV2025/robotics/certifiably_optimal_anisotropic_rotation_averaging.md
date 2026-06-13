---
title: >-
  [Paper Note] Certifiably Optimal Anisotropic Rotation Averaging
description: >-
  [ICCV 2025][Robotics][rotation averaging] This paper proposes a novel SDP relaxation that enforces solutions to lie within the convex hull of SO(3), conv(SO(3))…
tags:
  - "ICCV 2025"
  - "Robotics"
  - "rotation averaging"
  - "anisotropic optimization"
  - "semidefinite programming"
  - "convex relaxation"
  - "structure from motion"
date: 2026-05-08
content_hash: a2103ed675279a68
---

# Certifiably Optimal Anisotropic Rotation Averaging

**Conference**: ICCV 2025
**arXiv**: [2503.07353](https://arxiv.org/abs/2503.07353)  
**Code**: None  
**Area**: Robotics
**Keywords**: rotation averaging, anisotropic optimization, semidefinite programming, convex relaxation, structure from motion

## TL;DR

This paper proposes a novel SDP relaxation that enforces solutions to lie within the convex hull of SO(3), conv(SO(3)), achieving for the first time certifiably globally optimal rotation averaging under anisotropic cost functions. It resolves the fundamental failure of the conventional O(3) relaxation in anisotropic settings.

## Background & Motivation

Rotation averaging is a core subproblem in SfM and SLAM: given a set of pairwise relative rotation measurements, estimate the absolute orientations of all cameras. Existing certifiably optimal methods almost exclusively focus on the **isotropic** chordal distance, which assumes uniform uncertainty across all directions for each relative rotation measurement.

In practice, however, the uncertainty of two-view relative pose estimation is typically **highly anisotropic** — in-plane rotations are generally better constrained than out-of-plane rotations. Recent empirical results (e.g., Zhang et al., CVPR 2023) have demonstrated that incorporating anisotropy in local optimization significantly improves reconstruction quality. Nevertheless, handling anisotropic costs within a global optimization framework has remained an open problem.

## Core Problem

1. **Limitations of the isotropic assumption**: The standard chordal distance ignores the directional uncertainty of measurements, resulting in inappropriate weighting along high-uncertainty directions.
2. **Failure of the O(3) relaxation under anisotropy**: The conventional SDP relaxation optimizes over O(3) by dropping the determinant constraint det(R)=1. For isotropic costs this is generally safe, as the optimal solution naturally satisfies det=1. For anisotropic costs, however, the optimal solution over O(3) systematically favors det=-1 "reflection" solutions, causing the relaxation to fail entirely to recover a rank-3 solution. In 1,000 synthetic problem instances, the authors find that the standard relaxation **never** yields a rank-3 solution.
3. **How to incorporate the determinant constraint while preserving convexity?** — The constraint det(R)=1 is non-convex, and adding it directly destroys the SDP's convex structure.

## Method

### Overall Architecture

The method proceeds in three steps: encoding anisotropic uncertainty as a weighting matrix → analyzing why the O(3) relaxation fails → proposing a tighter relaxation based on conv(SO(3)).

**Step 1: Construction of the anisotropic cost**

For each camera pair $(i,j)$, a weighting matrix is constructed from the two-view BA Hessian $H_{ij}$ as $M_{ij} = \frac{\text{tr}(H_{ij})}{2}I - H_{ij}$. The objective function changes from the isotropic $-\langle \tilde{R}_{ij}, R_j R_i^\top \rangle$ to the anisotropic $-\langle M_{ij}\tilde{R}_{ij}, R_j R_i^\top \rangle$. This cost is equivalent to the negative log-likelihood of the Langevin distribution and approximately equivalent to a Gaussian model $\mathcal{N}(0, H_{ij}^{-1})$ in axis-angle space, thus admitting a clear statistical interpretation.

**Step 2: Why the O(3) relaxation fails**

A key theorem (Theorem 1) shows that when $M_{ij}$ is indefinite (i.e., its minimum eigenvalue $\lambda_3 < 0$, which holds in 77%–100% of practical cases), the minimizer of $-\langle M\tilde{R}, R \rangle$ over O(3) is a matrix with det=-1, whose objective value is lower than that of the true solution $\tilde{R}$ by $2|\lambda_3|$. This means the O(3) relaxation is systematically attracted to spurious reflection solutions.

**Step 3: The conv(SO(3)) relaxation**

The core idea is that while directly enforcing $R \in SO(3)$ (non-convex) is infeasible, one can enforce $R \in \text{conv}(SO(3))$ (convex hull, which is convex). Corollary 1 establishes that the minimum of a linear objective over conv(SO(3)) equals its minimum over SO(3), since linear functions attain their extrema at extreme points, and the extreme points of conv(SO(3)) are precisely SO(3).

### Key Designs

In practice, an auxiliary variable $Q_{ij} = R_i R_j^\top$ is introduced, and the constraint $Q_{ij} \in SO(3)$ is converted via Lagrangian duality and biconjugate techniques into the convex constraint $X_{ij} \in \text{conv}(SO(3))$. The resulting SDP program (SDP-cSO(3)) is:

$$\min_{X \succeq 0} -\text{tr}(NX) \quad \text{s.t.} \quad X_{ii} = I, \quad X_{ij} \in \text{conv}(SO(3))$$

where $X_{ij} \in \text{conv}(SO(3))$ is equivalent to a $4\times 4$ positive semidefinite constraint $\mathcal{A}(X_{ij}) + I \succeq 0$ (based on results from Saunderson et al.), so the full program remains a standard SDP.

### Loss & Training

This is a pure optimization method rather than a learning-based approach. The SDP is solved using the cone-splitting solver SCS in Julia via the JuMP interface. Implementation details include:
- Cost matrices are normalized by the mean maximum eigenvalue of the Hessians across the scene.
- Absolute and relative feasibility tolerances are set to $10^{-5}$ and $10^{-6}$, respectively.
- Maximum iterations are set to 500,000, though in practice far fewer are required.

## Key Experimental Results

### Synthetic Experiments

Across settings with 2–100 cameras and 0%–90% missing-data ratios, SDP-cSO(3) with anisotropic cost consistently achieves the lowest rotation error, and notably converges faster than the baseline (as the conv(SO(3)) constraints reduce oscillation in the first-order SDP solver).

### Main Results

| Dataset | Metric ($\sum\|R_i - R_i^*\|^2$) | SDP-O(3)-iso | SDP-cSO(3) (Ours) | Gain |
|---|---|---|---|---|
| LU Sphinx (70 cameras) | Rotation error | 0.0944 | **0.0740** | 21.6% |
| Round Church (92 cameras) | Rotation error | 0.1399 | **0.1267** | 9.4% |
| UWO (114 cameras) | Rotation error | 0.3142 | **0.2274** | 27.6% |
| Tsar Nikolai I (89 cameras) | Rotation error | 0.1170 | **0.0534** | 54.4% |
| King's College (77 cameras) | Rotation error | 0.1656 | **0.0796** | 51.9% |
| Museum Barcelona (133 cameras) | Rotation error | 0.2255 | **0.1310** | 41.9% |
| Temple Singapore (157 cameras) | Rotation error | 0.2646 | **0.1696** | 35.9% |
| Kronan (131 cameras) | Rotation error | 0.2149 | 0.3892 | −81.1% (only regression) |

Note: SDP-O(3)-aniso (anisotropic cost + conventional O(3) relaxation) **completely fails** on all datasets (rank = 6–7, error = 14–30), validating the theoretical analysis of the paper.

### Ablation Study

- **Switching cost vs. constraint independently**: The four combinations (iso/aniso × O(3)/cSO(3)) in synthetic experiments clearly demonstrate that both components — anisotropic cost and conv(SO(3)) constraint — are individually necessary.
- **Proportion of indefinite matrices**: In real-world data, 77%–100% of the $M_{ij}$ matrices are indefinite, corroborating the systematic failure of the O(3) relaxation under anisotropy.
- **Mahalanobis distance evaluation**: Under uncertainty-aware Mahalanobis error, SDP-cSO(3) shows even larger advantages on most scenes (e.g., Mahalanobis distance on Tsar Nikolai I drops from 0.687 to 0.188).

## Highlights & Insights

1. **An elegant theory-to-practice loop**: Theorem 1 precisely explains why the O(3) relaxation fails under anisotropy ($\lambda_3 < 0$ makes the det=−1 solution preferable), and conv(SO(3)) is exactly the right fix — this "diagnose-then-repair" style is exemplary.
2. **Clever use of conv(SO(3))**: Replacing the original non-convex set with its convex hull eliminates spurious reflection solutions while preserving convexity. The key insight is that a linear objective attains the same minimum over the convex hull as over the original set.
3. **Statistical modeling rigor**: Starting from the two-view BA Hessian, via Laplace approximation and first-order uncertainty propagation, the derivation arrives at a Langevin-distribution cost in a coherent chain of reasoning.
4. **Counter-intuitive finding**: A tighter relaxation (with more constraints) actually leads to faster solver convergence, as it reduces oscillation in the solution space.

## Limitations & Future Work

1. **Computational efficiency**: Although SDP runs in polynomial time, the additional positive semidefinite constraints introduced by the conv(SO(3)) formulation substantially increase solve time for large scenes (>150 cameras) — Temple Singapore requires 628 seconds versus 31 seconds for the baseline. The authors acknowledge that designing an efficient dedicated solver is an important future direction.
2. **Lack of theoretical tightness guarantees**: Tightness of the relaxation is verified only empirically (all tested instances yield rank-3 solutions); explicit theoretical conditions under which the relaxation is tight have not been established.
3. **Robustness not addressed**: The paper assumes no outliers, yet erroneous relative rotations are common in practical SfM. Combining anisotropic costs with robust loss functions is a promising direction for future work.
4. **Regression on Kronan**: The anisotropic method performs worse on this dataset, which the paper attributes to randomness in a single noisy instance, though this may hint at deeper limitations of the model.

## Related Work & Insights

- **vs. Zhang et al. (CVPR 2023)**: Proposes anisotropic costs for local optimization but cannot guarantee global optimality. This paper extends the idea to a certifiably optimal global optimization framework.
- **vs. Barfoot et al. / Holmes et al.**: Anisotropic SDP relaxations based on the Cayley map require a large number of redundant constraints to obtain a "reasonably tight" relaxation. This paper argues that their underlying issue is also the enforcement of only O(3) membership.
- **vs. Dellaert et al. (ECCV 2020, Shonan)**: The Riemannian staircase approach is highly efficient under isotropic costs, but its generalization to anisotropic settings is not straightforward, as it relies heavily on properties of O(d) that are insufficient to guarantee SO(3) membership.
- **vs. Eriksson et al. (TPAMI 2021)**: Provides theoretical tightness bounds for the isotropic relaxation (depending on the algebraic connectivity of the graph); the anisotropic case is more complex and analogous bounds remain to be established.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The conv(SO(3)) relaxation that resolves the O(3) failure under anisotropy is theoretically elegant and genuinely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Synthetic experiments plus 11 real-world SfM datasets with comprehensive ablations; direct comparison with local optimization methods is missing.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Mathematical derivations are rigorous and clear; the logical chain from problem motivation through theorems to algorithm is textbook-quality.
- **Value**: ⭐⭐ — Loosely connected to the reviewer's current research focus, but the "uncertainty → cost weighting" principle has transferable value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Optimal and Scalable MAPF via Multi-Marginal Optimal Transport and Schrödinger Bridges](../../ICML2026/robotics/optimal_and_scalable_mapf_via_multi-marginal_optimal_transport_and_schrödinger_b.md)
- [\[ICCV 2025\] DexVLG: Dexterous Vision-Language-Grasp Model at Scale](dexvlg_dexterous_vision-language-grasp_model_at_scale.md)
- [\[ICCV 2025\] TransiT: Transient Transformer for Non-line-of-sight Videography](transit_transient_transformer_for_non-line-of-sight_videography.md)
- [\[ICCV 2025\] Adaptive Articulated Object Manipulation On The Fly with Foundation Model Reasoning and Part Grounding](adaptive_articulated_object_manipulation_on_the_fly_with_foundation_model_reason.md)
- [\[ICCV 2025\] Resolving Token-Space Gradient Conflicts: Token Space Manipulation for Transformer-Based Multi-Task Learning](resolving_token-space_gradient_conflicts_token_space_manipulation_for_transforme.md)

</div>

<!-- RELATED:END -->

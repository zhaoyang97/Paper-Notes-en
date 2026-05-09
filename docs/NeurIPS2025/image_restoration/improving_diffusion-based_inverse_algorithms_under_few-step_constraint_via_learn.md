---
title: >-
  [Paper Note] Improving Diffusion-based Inverse Algorithms under Few-Step Constraint via Learnable Linear Extrapolation
description: >-
  [NeurIPS 2025][Image Restoration][Diffusion inverse problems] This paper proposes Learnable Linear Extrapolation (LLE), which combines current and historical clean data estimates via learnable linear coefficients to enhance any diffusion inverse problem algorithm conforming to the Sampler-Corrector-Noiser paradigm under few-step (3–5 steps) constraints. The method requires only 50 training samples and a few minutes of training, yielding consistent improvements across 9+ algorithms × 5 tasks.
tags:
  - NeurIPS 2025
  - Image Restoration
  - Diffusion inverse problems
  - few-step acceleration
  - linear extrapolation
  - learnable coefficients
  - plug-and-play
date: 2026-05-08
content_hash: 90ae36ac54dd1e96
---

# Improving Diffusion-based Inverse Algorithms under Few-Step Constraint via Learnable Linear Extrapolation

**Conference**: NeurIPS 2025
**arXiv**: [2503.10103](https://arxiv.org/abs/2503.10103)
**Code**: To be confirmed
**Area**: Image Restoration / Diffusion Models
**Keywords**: Diffusion inverse problems, few-step acceleration, linear extrapolation, learnable coefficients, plug-and-play

## TL;DR
This paper proposes Learnable Linear Extrapolation (LLE), which combines current and historical clean data estimates via learnable linear coefficients to enhance any diffusion inverse problem algorithm conforming to the Sampler-Corrector-Noiser paradigm under few-step (3–5 steps) constraints. The method requires only 50 training samples and a few minutes of training, yielding consistent improvements across 9+ algorithms × 5 tasks.

## Background & Motivation
**Background**: Diffusion models achieve strong performance on inverse problems (deblurring, super-resolution, inpainting, compressed sensing, etc.), but require a large number of sampling steps (100–1000) to produce high-quality results. Fast ODE solvers are effective for unconditional generation but perform poorly on inverse problems due to heterogeneous formulations and approximation errors.

**Limitations of Prior Work**: The corrector step (data consistency enforcement) in diffusion inverse problems introduces additional errors, which accumulate under few-step regimes and lead to significant quality degradation. Different algorithms (DPS, DDNM, DDRM, MCG, etc.) employ distinct corrector designs, motivating the need for a unified enhancement strategy.

**Key Challenge**: Practical applications demand few-step inference (3–5 steps), yet the few-step regime exposes the cumulative effect of corrector errors.

**Goal**: Design a lightweight, general-purpose "patch" that can enhance any diffusion inverse algorithm under few-step constraints.

**Key Insight**: All diffusion inverse problem algorithms follow the Sampler → Corrector → Noiser paradigm, enabling unified representation and enhancement via linear extrapolation.

**Core Idea**: Learn a small number of linear combination coefficients to combine historical clean data estimates with the current estimate, compensating for approximation errors introduced by few-step inference.

## Method

### Overall Architecture
Nine or more diffusion inverse problem algorithms are unified into a canonical form: each step comprises a Sampler ($\mathbf{x}_{0,t_i} = \Phi_{t_i}(\mathbf{x}_{t_i})$) → Corrector ($\hat{\mathbf{x}}_{0,t_i} = \mathbf{h}_{t_i}(\mathbf{x}_{0,t_i}, \mathcal{A}, \mathbf{y})$) → Noiser ($\mathbf{x}_{t_{i-1}} = \Psi_{t_i}(\hat{\mathbf{x}}_{0,t_i})$). LLE inserts a learnable linear extrapolation step after the Corrector.

### Key Designs

1. **Unified Canonical Form**:

    - Function: Reformulates 9+ algorithms including DDNM, DPS, DDRM, MCG, PGDM, ReSample, DDPG, DiffPIR, and PSLD under the Sampler-Corrector-Noiser three-step paradigm.
    - Design Motivation: Differences among algorithms lie in the Corrector design (how data consistency is enforced), while the overall framework is consistent, enabling universal enhancement.

2. **Learnable Linear Extrapolation (LLE)**:

    - Function: At step $i$, linearly combines the corrected estimate $\hat{\mathbf{x}}_{0,t_i}$ with estimates from all previous steps.
    - Mechanism: $\tilde{\mathbf{x}}_{0,t_i} = \gamma_{t_i,S-i}^{\perp}\hat{\mathbf{x}}_{0,t_i} + \sum_{j=0}^{S-i-1}\gamma_{t_i,j}^{\perp}\tilde{\mathbf{x}}_{0,t_{S-j}}$
    - Learnable Parameters: Only a few coefficients $\gamma$ per step, resulting in an extremely small total parameter count.
    - Design Motivation: Historical estimates carry information from prior steps; their linear combination can correct errors in the current step.

3. **Decoupled Coefficients (Linear Inverse Problems)**:

    - Function: For linear inverse problems, separates coefficients into range space and null space components.
    - Mechanism: Components in the measurement space can be precisely constrained (data consistency), while the null space must be supplemented by the diffusion prior. These two subspaces require different extrapolation strategies.
    - Effect: Inpainting +0.96 PSNR, SR +0.26 PSNR.

### Loss & Training
$\mathcal{L} = \mathcal{L}_{MSE} + 0.1 \cdot \mathcal{L}_{LPIPS}$. Only 50 training samples are required; training completes in 2–20 minutes on an RTX 3090.

## Key Experimental Results

### Main Results (CelebA-HQ, noisy $\sigma=0.05$)

| Task | Steps | DDNM | DDNM+LLE | DPS | DPS+LLE |
|------|-------|------|----------|-----|---------|
| Deblur | 3 | 27.80/0.758 | **28.08/0.784** | 23.59/0.650 | **24.59/0.675** |
| Inpainting | 5 | 22.76/0.550 | **26.35/0.659** | 25.49/0.647 | **27.51/0.748** |
| Super-Res | 3 | 27.09/0.773 | **27.84/0.770** | 25.49/0.647 | **24.57/0.666** |
| CS 50% | 5 | 18.20/0.474 | **19.41/0.536** | 17.27/0.591 | **18.44/0.605** |

### Ablation Study

| Configuration | Key Finding | Note |
|---------------|------------|------|
| Decoupled vs. coupled coefficients | Decoupling yields +0.96 PSNR on inpainting | Range/null space require separate treatment |
| $\omega=0.1$ (LPIPS weight) | Balances PSNR (32.5) vs. LPIPS (0.19) | Optimal trade-off |
| Inference overhead | DDNM: 2.0→2.0 min, DPS: 5.07→5.13 min | Near-zero additional cost (<2%) |
| Cross-dataset transfer | CelebA→FFHQ performance maintained | Good generalization |
| Cross-task transfer | Moderate performance degradation | Limited by linear search space |

### Key Findings
- LLE consistently improves PSNR/SSIM across 9+ algorithms without any performance degradation.
- Training is extremely lightweight (50 samples + a few minutes); inference incurs negligible additional overhead.
- Inpainting benefits the most (DDNM: 22.76→26.35 PSNR, +3.6 dB).
- Cross-dataset transfer is effective; cross-task transfer is limited.

## Highlights & Insights
- **Universal plug-and-play enhancement**: LLE does not modify the design of any original algorithm; it simply inserts a lightweight linear combination after the Corrector. This "patch" paradigm is generalizable to other iterative algorithms.
- **Few-shot training**: The coefficients can be trained with only 50 samples, indicating that the parameter space of linear extrapolation is compact and easy to optimize—particularly valuable for data-scarce domains such as medical imaging.
- **Unified perspective**: Reformulating 9+ heterogeneous algorithms under the Sampler-Corrector-Noiser paradigm is itself a significant contribution, providing a clear framework for future algorithm design and analysis.

## Limitations & Future Work
- The search space is restricted to linear combinations of historical estimates; nonlinear combinations may yield further improvements.
- Cross-task transfer is limited—optimal coefficients vary substantially across tasks.
- Validation is conducted only on VP diffusion; newer paradigms such as flow matching remain unexplored.
- Theoretical analysis is relatively weak—the mechanism underlying the effectiveness of linear extrapolation is not yet well understood.

## Related Work & Insights
- **vs. DPM-Solver (Lu et al., 2022)**: DPM-Solver employs high-order ODE solvers to accelerate unconditional generation, while LLE targets inverse problem scenarios with corrector steps—the two approaches are complementary.
- **vs. DDNM (Wang et al., 2022)**: DDNM's null-space projection degrades severely under few-step regimes; LLE directly compensates for its errors.
- **vs. DPS (Chung et al., 2023)**: DPS gradient guidance becomes inaccurate under few steps; LLE smooths historical information to improve guidance quality.

## Rating
- Novelty: ⭐⭐⭐⭐ The unified paradigm combined with lightweight linear enhancement is concise and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive coverage across 9+ algorithms × 5 tasks × multiple datasets.
- Writing Quality: ⭐⭐⭐⭐ The unified framework is described clearly.
- Value: ⭐⭐⭐⭐ Directly beneficial for practical applications of diffusion-based inverse problems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Cocoercive Conservative Denoisers via Helmholtz Decomposition for Poisson Inverse Problems](learning_cocoercive_conservative_denoisers_via_helmholtz_decomposition_for_poiss.md)
- [\[NeurIPS 2025\] Rethinking Nighttime Image Deraining via Learnable Color Space Transformation](rethinking_nighttime_image_deraining_via_learnable_color_space_transformation.md)
- [\[CVPR 2026\] FiDeSR: High-Fidelity and Detail-Preserving One-Step Diffusion Super-Resolution](../../CVPR2026/image_restoration/fidesr_high-fidelity_and_detail-preserving_one-step_diffusion_super-resolution.md)
- [\[CVPR 2026\] GSNR: Graph Smooth Null-Space Representation for Inverse Problems](../../CVPR2026/image_restoration/gsnr_graph_smooth_null_space_representation_for_inverse_problems.md)
- [\[NeurIPS 2025\] DynaGuide: Steering Diffusion Policies with Active Dynamic Guidance](dynaguide_steering_diffusion_polices_with_active_dynamic_guidance.md)

</div>

<!-- RELATED:END -->

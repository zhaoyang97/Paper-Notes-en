---
title: >-
  [Paper Note] Advancing Text-to-3D Generation with Linearized Lookahead Variational Score Distillation
description: >-
  [ICCV 2025][3D Vision][Score Distillation] By identifying the optimization order mismatch between the LoRA model and the 3D model in VSD, this paper proposes a linearized lookahead correction term, $L^2$-VSD, which significantly improves text-to-3D generation quality at the cost of only one additional forward pass.
tags:
  - ICCV 2025
  - 3D Vision
  - Score Distillation
  - VSD
  - Text-to-3D
  - LoRA
  - Forward-mode Autodiff
date: 2026-05-08
content_hash: 6c74446326836f81
---

# Advancing Text-to-3D Generation with Linearized Lookahead Variational Score Distillation

**Conference**: ICCV 2025
**arXiv**: [2507.09748](https://arxiv.org/abs/2507.09748)
**Code**: Implemented based on the [threestudio](https://github.com/threestudio-project/threestudio) framework
**Area**: 3D Vision / Text-to-3D Generation
**Keywords**: Score Distillation, VSD, Text-to-3D, LoRA, Forward-mode Autodiff

## TL;DR

By identifying the optimization order mismatch between the LoRA model and the 3D model in VSD, this paper proposes a linearized lookahead correction term, $L^2$-VSD, which significantly improves text-to-3D generation quality at the cost of only one additional forward pass.

## Background & Motivation

Score distillation-based text-to-3D generation is a prominent research direction. While SDS (Score Distillation Sampling) pioneered the use of pretrained 2D diffusion models to guide 3D content generation, it is widely known to produce over-smoothed results. VSD (Variational Score Distillation) introduces an auxiliary LoRA model to estimate the score function of the rendered image distribution, theoretically correcting the gradient direction of SDS.

However, VSD exhibits a critical practical flaw: the LoRA model $\epsilon_{\phi_i}$ is trained on the previous 3D state $\theta_{i-1}$, yet is used to guide the update of the current state $\theta_i$. This **optimization order mismatch** leads to slow convergence or even collapse. Through systematic diagnostic experiments, the authors identify this issue and propose a linearized correction scheme that captures the benefits of lookahead without overfitting.

## Method

### Overall Architecture

$L^2$-VSD builds upon the VSD framework. The core idea is to decompose the lookahead update of the LoRA model via Taylor expansion into a first-order term (carrying semantic information) and higher-order terms (carrying noise), retaining only the first-order linear correction for score distillation.

### Key Designs

1. **Problem Diagnosis: LoRA–3D Mismatch**

    - In the practical implementation of VSD, the LoRA model is first trained with rendered images from $\theta_{i-1}$ to obtain $\phi_i$, which is then used to guide the update of $\theta_i$.
    - Theoretically, the LoRA model should be adapted to the current 3D state $\theta_i$ prior to guiding its update.
    - A toy example using 2D Gaussian distributions confirms that this mismatch indeed exists.

2. **Lookahead VSD (L-VSD): Attempt and Failure**

    - A straightforward fix swaps the optimization order: first update LoRA $\phi_i \to \phi_{i+1}$, then use $\phi_{i+1}$ to update $\theta_i$.
    - Experiments show sharper edges and faster convergence.
    - However, the LoRA model tends to overfit to a single 3D particle, leading to color oversaturation and geometric collapse.

3. **Taylor Expansion Analysis**

    - Expanding $\epsilon_{\phi_{i+1}}$ around $\phi_i$:
    $\epsilon_{\phi_{i+1}}(x_t,t,c,y) = \epsilon_{\phi_i}(x_t,t,c,y) + \underbrace{(-2\eta \Delta_{\phi_i} J_{\phi_i}^T)}_{\Delta\epsilon_{first}} + \underbrace{\mathcal{O}(\Delta_{\phi_i}^2)}_{\Delta\epsilon_{high}}$
    - The first-order term $\Delta\epsilon_{first}$ encodes semantic contour information corresponding to the prompt.
    - The higher-order term $\Delta\epsilon_{high}$ has a norm far larger than the first-order term and is dominated by stochastic high-frequency noise.
    - Visualization confirms that decoding the first-order term yields clear object shapes, while the higher-order term produces meaningless noise.

4. **$L^2$-VSD: Linearized Lookahead Correction**

    - Only the linearized LoRA prediction is retained: $\epsilon_{\phi_{i+1}}^{lin} = \epsilon_{\phi_i} + \Delta\epsilon_{first}$
    - This is equivalent to performing one SGD step on the linear model $\epsilon_{\phi}^{lin}(x_t) = \epsilon_{\phi_i}(x_t) + (\phi - \phi_i)J_{\phi_i}^T(x_t)$.
    - The low complexity of the linear model naturally prevents overfitting.
    - The final gradient consists of two contrasting terms: the original VSD objective plus a linearized lookahead correction (with the Jacobian product serving as a preconditioning matrix).

### Loss & Training

- The correction term is $\Delta\epsilon_{first} = -2\eta \Delta_{\phi_i} J_{\phi_i}^T(x_t,t,c,y)$.
- $\Delta_{\phi_i}$ denotes the LoRA gradient, obtained via one backward pass.
- The vector–Jacobian product $\Delta_{\phi_i} J_{\phi_i}^T$ is computed efficiently via **forward-mode autodiff**.
- The additional cost over VSD is a single forward pass through the LoRA model.
- Optionally, only the Jacobian of the last LoRA layer is used as an approximation to further reduce computational overhead.

## Key Experimental Results

### Main Results

| Method | CLIP Sim (↓) | FID (↓) | Notes |
|--------|-------------|---------|-------|
| SDS | 0.305 | 372.35 | Over-smoothed |
| ESD | 0.316 | 315.15 | Addresses Janus problem |
| VSD | 0.324 | 301.54 | Baseline |
| L-VSD | 0.337 | 496 | Overfitting collapse |
| HiFA | 0.313 | 292.88 | Prev. SOTA |
| **$L^2$-VSD** | **0.285** | **284.06** | Best overall |

Evaluation is conducted on 20 prompts from the DreamFusion gallery at resolution 256.

### Ablation Study

| Configuration | Effect | Notes |
|---------------|--------|-------|
| $\eta$ = 1e-3 ~ 1 | Consistent improvement | Robust to $\eta$, even when first-order norm is ~1e-2 |
| Last-layer approximation | Slightly lower quality but still better than VSD | Computation time significantly reduced |
| $\gamma$=1,2,5,10 (VSD) | No substantial quality gain | Increasing LoRA convergence alone is insufficient |
| $\gamma$=2,5 (L-VSD) | Increased oversaturation | Overfitting problem worsens |
| Reduced LoRA learning rate (L-VSD) | Occasionally effective but unstable | Fails on complex prompts |

### Key Findings

- Even when the norm of the first-order correction per iteration is small (~1e-2), its cumulative effect over long optimization leads to substantial quality improvements.
- $L^2$-VSD produces photorealistic results at low resolution (64) without multi-stage training.
- It integrates seamlessly into VSD variants such as ESD ($L^2$-ESD) and HiFA ($L^2$-HiFA).

## Highlights & Insights

- **Rigorous diagnostic study**: The work systematically diagnoses VSD's failure from three perspectives—convergence, optimization order, and their interaction—with well-designed experiments.
- **Elegant linearization**: Taylor expansion cleanly separates the first-order semantic term from higher-order noise, simultaneously capturing the lookahead benefit while avoiding overfitting.
- **Efficient implementation**: Forward-mode autodiff requires only one additional forward pass.
- **Plug-and-play**: The correction can be integrated into any VSD-based method.

## Limitations & Future Work

- Generation still requires several hours, a common bottleneck of score distillation methods.
- A distributional-level optimization objective corresponding to the first-order correction term has not yet been identified, leaving a gap in theoretical grounding.
- The method still relies on conventional 3D representations such as NeRF/Mesh; 3D Gaussian Splatting has not been explored.
- The connection to SiD warrants further investigation.

## Related Work & Insights

- Relationship to SDS, VSD, ESD, and HiFA: this work is a direct improvement over VSD and is composable with ESD and HiFA.
- The use of forward-mode autodiff in deep learning serves as a reference case, inspiring further method designs involving Jacobian–vector products.
- The in-depth analysis of the theory–practice gap in score distillation provides an important foundation for future work.

## Rating

- Novelty: ⭐⭐⭐⭐ Proposes an elegant linearization scheme grounded in the theory–practice gap of VSD.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Diagnostic, comparative, ablation, and combination experiments are all comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ The diagnose–analyze–solve narrative structure is clear and well-organized.
- Value: ⭐⭐⭐⭐ Offers direct practical utility to the text-to-3D community as a plug-and-play module.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Stable Score Distillation](stable_score_distillation.md)
- [\[ICCV 2025\] SegmentDreamer: Towards High-Fidelity Text-to-3D Synthesis with Segmented Consistency Trajectory Distillation](segmentdreamer_towards_high-fidelity_text-to-3d_synthesis_with_segmented_consist.md)
- [\[ICCV 2025\] Identity Preserving 3D Head Stylization with Multiview Score Distillation](identity_preserving_3d_head_stylization_with_multiview_score_distillation.md)
- [\[ICCV 2025\] Benchmarking and Learning Multi-Dimensional Quality Evaluator for Text-to-3D Generation](benchmarking_and_learning_multidimensional_quality_evaluator.md)
- [\[ICCV 2025\] Text2VDM: Text to Vector Displacement Maps for Expressive and Interactive 3D Sculpting](text2vdm_text_to_vector_displacement_maps_for_expressive_and_interactive_3d_scul.md)

</div>

<!-- RELATED:END -->

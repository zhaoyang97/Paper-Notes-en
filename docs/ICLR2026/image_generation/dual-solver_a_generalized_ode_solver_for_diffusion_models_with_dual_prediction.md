---
title: >-
  [Paper Note] Dual-Solver: A Generalized ODE Solver for Diffusion Models with Dual Prediction
description: >-
  [ICLR 2026][Image Generation][ODE solver] This paper proposes Dual-Solver, which generalizes multi-step samplers for diffusion models via three sets of learnable parameters — prediction type interpolation $\gamma$…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "ODE solver"
  - "learnable sampler"
  - "prediction interpolation"
  - "domain selection"
  - "low-NFE"
date: 2026-05-08
content_hash: 38b734a779267153
---

# Dual-Solver: A Generalized ODE Solver for Diffusion Models with Dual Prediction

**Conference**: ICLR 2026
**arXiv**: [2603.03973](https://arxiv.org/abs/2603.03973)  
**Code**: None  
**Area**: Diffusion Models / Sampling Acceleration
**Keywords**: ODE solver, learnable sampler, prediction interpolation, domain selection, low-NFE

## TL;DR
This paper proposes Dual-Solver, which generalizes multi-step samplers for diffusion models via three sets of learnable parameters — prediction type interpolation $\gamma$, integration domain selection $\tau$, and residual adjustment $\kappa$ — and learns these parameters using the classification loss of frozen pretrained classifiers (MobileNet/CLIP) without requiring teacher trajectories. The method consistently outperforms DPM-Solver++ and related approaches in the low-NFE regime (3–9 NFE).

## Background & Motivation

### State of the Field

**Background**: Inference acceleration is a central challenge in diffusion models. ODE solvers (e.g., DPM-Solver, DEIS) exploit the structure of diffusion dynamics to design efficient samplers. Learnable solvers (e.g., BNS-Solver, DS-Solver) further improve quality by optimizing timesteps and sampling parameters.

**Limitations of Prior Work**: (a) Conventional solvers fix the prediction type (noise/data/velocity) and integration domain (log/linear), yet no single choice is universally optimal across different NFE settings; (b) Learnable solvers typically require large numbers of teacher trajectories or high-NFE samples as training targets, incurring substantial preparation overhead.

**Key Challenge**: The choice of prediction type and integration domain significantly affects sampling quality, but the optimal choice is backbone- and NFE-dependent — an adaptive approach is therefore necessary.

**Goal**: Unify different prediction types and integration domains into a single continuously parameterized framework, and learn the optimal parameters using a classification loss that requires no target samples.

**Key Insight**: The paper observes that noise prediction, velocity prediction, and data prediction are mutually interchangeable via linear combination, and that integration in the $\log$ SNR domain versus the linear $t$ domain can be continuously interpolated — both aspects can thus be parameterized and learned end-to-end.

**Core Idea**: Parameterize prediction type, integration domain, and residual terms jointly, and use the classification accuracy of a frozen pretrained classifier as a target-sample-free training signal.

## Method

### Overall Architecture

Dual-Solver retains the standard predictor-corrector structure but introduces learnable parameters along three dimensions:
1. **$\gamma$ (prediction interpolation)**: Continuously interpolates among noise/velocity/data prediction
2. **$\tau$ (domain selection)**: Continuously interpolates between the log-SNR domain and the linear time domain
3. **$\kappa$ (residual adjustment)**: Adjusts residual terms in multi-step updates while preserving second-order accuracy

### Key Designs

1. **Prediction type interpolation**: $\hat{p}_\gamma = (1-\gamma) \hat{\epsilon}_\theta + \gamma \hat{x}_\theta$, where $\gamma$ continuously selects the optimal combination of prediction types.
2. **Integration domain interpolation**: A continuous transition between $\lambda = \log(\alpha_t/\sigma_t)$ and $t$ controlled by the parameter $\tau$.
3. **Classification-based learning strategy**: A frozen MobileNet/CLIP classifier is applied to generated images, and the classification loss is backpropagated to update the solver parameters. No high-NFE target samples are required.
4. All modifications preserve second-order local accuracy.

### Loss & Training
- Classification loss (cross-entropy with conditional class labels)
- Pretrained classifier is frozen; only $\gamma$, $\tau$, and $\kappa$ are updated
- Applicable to diverse backbones including DiT, GM-DiT, SANA, and PixArt-α

## Key Experimental Results

### Main Results (ImageNet 256, DiT-XL/2, 50k samples, CFG=1.5)

FID comparison of different learning strategies on DiT:

| Learning Strategy | NFE=3 FID↓ | NFE=5 FID↓ | NFE=7 FID↓ | NFE=9 FID↓ |
|---|---|---|---|---|
| Sample regression | 107.13 | 11.71 | 4.60 | 2.99 |
| Trajectory regression | 100.89 | 11.59 | 3.66 | 2.84 |
| Feature regression (AlexNet) | 47.75 | 7.24 | 3.42 | 2.91 |
| Feature regression (VGG) | 41.58 | 5.48 | 3.23 | 2.88 |
| **Classification learning (Hard-label)** | **24.91** | **3.52** | **2.75** | **2.67** |

The classification-based learning strategy significantly outperforms all regression-based baselines across all NFE settings; at NFE=3, FID drops from 41.58 to 24.91 (−40%).

### Ablation Study

Parameter configuration ablation (DiT, p1c2 setting):

| Configuration | NFE=3 FID | NFE=5 FID | NFE=7 FID | NFE=9 FID |
|---|---|---|---|---|
| All learnable | **0.574** | **0.197** | 0.178 | 0.173 |
| $\gamma=0$ fixed | 0.600 | 0.202 | 0.183 | 0.180 |
| $\gamma=1$ fixed | 0.816 | 0.223 | 0.182 | 0.176 |
| $\gamma=-1$ fixed | 7.871 | 7.676 | 0.238 | 0.196 |
| $\kappa=0$ fixed | 0.944 | 0.256 | 0.202 | 0.190 |
| p1 (no corrector) | 0.667 | 0.225 | 0.183 | 0.175 |
| p2 | 1.023 | 0.253 | 0.222 | 0.181 |

Results are validated across multiple backbones: DiT (class-conditional ImageNet generation), GM-DiT (flow matching, ImageNet), PixArt-α (T2I, 512px), and SANA (flow matching T2I, 512px).

### Key Findings
- **Adaptive $\gamma$ is critical**: Fixing $\gamma=-1$ (noise prediction) leads to catastrophic degradation at low NFE (FID 7.87), whereas adaptive $\gamma$ automatically selects the optimal prediction type at each step.
- **A V-shaped curve exists for classifier selection**: Evaluating 20 pretrained classifiers reveals that classifiers of intermediate accuracy perform best — those that are too strong over-constrain the optimization, while those that are too weak provide insufficient signal.
- **Parameters interpolate across NFE**: Learned parameters exhibit similar patterns at adjacent NFE settings; linear interpolation to unseen NFE values still outperforms hand-crafted solvers.
- GM-DiT (flow matching) shows slightly inferior performance for Dual-Solver at NFE 7–9; adding trajectory regression restores the advantage.

## Highlights & Insights
- The **three-dimensional parameterization** unifies a broad range of sampler design choices — DPM-Solver++ is a special case with $\gamma=0, \tau=0$.
- **Classification-based learning** as a replacement for regression-based learning is the key innovation: it eliminates the need for high-NFE target samples and relies solely on a frozen classifier. This paradigm is transferable to the optimization of any differentiable metric.

## Limitations & Future Work
- Parameters are backbone- and NFE-dependent; each configuration requires relearning.
- The classification loss may bias generation toward class-discriminability rather than perceptual quality.
- Validation is limited to the 3–9 NFE range; whether the advantage persists at higher NFE remains unknown.

## Related Work & Insights
- **vs. DPM-Solver++**: Dual-Solver generalizes DPM-Solver++ by adaptively learning the optimal configuration through parameterization.
- **vs. BNS/DS-Solver**: Both are learnable solvers, but Dual-Solver does not require target samples.
- The method is orthogonal to consistency distillation and related approaches.

## Rating
- Novelty: ⭐⭐⭐⭐ — The combination of a unified parameterization framework and classification-based learning is a novel contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive evaluation across multiple backbones (DiT/SANA/PixArt) and NFE settings.
- Writing Quality: ⭐⭐⭐⭐ — Mathematical derivations are clear and well-presented.
- Value: ⭐⭐⭐⭐ — A practical, plug-and-play improvement in the low-NFE regime.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Image Diffusion Preview with Consistency Solver](../../CVPR2026/image_generation/image_diffusion_preview_with_consistency_solver.md)
- [\[ICLR 2026\] LVTINO: LAtent Video consisTency INverse sOlver for High Definition Video Restoration](lvtino_latent_video_consistency_inverse_solver_for_high_definition_video_restora.md)
- [\[ICLR 2026\] Generating Directed Graphs with Dual Attention and Asymmetric Encoding](generating_directed_graphs_with_dual_attention_and_asymmetric_encoding.md)
- [\[CVPR 2026\] Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception](../../CVPR2026/image_generation/enhancing_image_aesthetics_with_dualconditioned_di.md)
- [\[ICLR 2026\] Generalization of Diffusion Models Arises with a Balanced Representation Space](generalization_of_diffusion_models_arises_with_a_balanced_representation_space.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] FlowCast: Advancing Precipitation Nowcasting with Conditional Flow Matching
description: >-
  [ICLR 2026][Image Generation][Conditional Flow Matching] This work is the first to apply Conditional Flow Matching (CFM) as an end-to-end probabilistic generative model for precipitation nowcasting. By learning a direct…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Conditional Flow Matching"
  - "precipitation nowcasting"
  - "probabilistic forecasting"
  - "latent-space generation"
  - "spatiotemporal prediction"
date: 2026-05-08
content_hash: 733160e4b37a7a52
---

# FlowCast: Advancing Precipitation Nowcasting with Conditional Flow Matching

**Conference**: ICLR 2026
**arXiv**: [2511.09731](https://arxiv.org/abs/2511.09731)
**Code**: [GitHub](https://github.com/b-rbmp/FlowCast)
**Area**: Diffusion Models / Weather Forecasting
**Keywords**: Conditional Flow Matching, precipitation nowcasting, probabilistic forecasting, latent-space generation, spatiotemporal prediction

## TL;DR
This work is the first to apply Conditional Flow Matching (CFM) as an end-to-end probabilistic generative model for precipitation nowcasting. By learning a direct noise-to-data mapping in a compressed latent space, the proposed method surpasses diffusion-based models in both predictive accuracy and probabilistic performance with significantly fewer sampling steps.

## Background & Motivation

**Background**: Precipitation nowcasting is critical for flood prevention and operational decision-making. Deep learning approaches have evolved from deterministic RNN/Transformer models to probabilistic diffusion-based methods. Latent-space diffusion models such as PreDiff and LDCast represent the current state of the art, while CasCast—a hybrid deterministic–diffusion framework—achieves the strongest overall performance.

**Limitations of Prior Work**: Deterministic models optimized with MSE produce blurry predictions and fail to quantify uncertainty. Diffusion models require hundreds of iterative denoising steps, incurring substantial computational cost that is incompatible with time-sensitive applications (e.g., flash flood warnings) that demand rapid ensemble forecasting.

**Key Challenge**: There exists a fundamental trade-off between predictive accuracy and computational efficiency—diffusion models are accurate but slow at inference, whereas deterministic models are fast but produce oversmoothed outputs. A probabilistic forecasting method that is simultaneously fast and accurate is needed.

**Goal**: To investigate whether CFM can replace diffusion models in this setting, achieving or exceeding predictive accuracy while drastically reducing the number of sampling steps.

**Key Insight**: The straight-line ODE trajectories of CFM are better suited for spatiotemporal prediction than the curved probability-flow paths of diffusion models. Although radar reflectivity distributions are multimodal, their strong temporal consistency makes linear interpolation a more stable prior.

**Core Idea**: The straight transport paths learned by CFM in latent space naturally align with the continuity of spatiotemporal data, enabling high-quality probabilistic forecasts with fewer steps.

## Method

### Overall Architecture
The framework follows a two-stage pipeline: (1) training a VAE to compress radar frames into a low-dimensional latent space; and (2) training a CFM model conditioned on past observations, built upon a Cuboid Attention U-Net, within that latent space. The model takes 13 historical radar frames (65 minutes) as input and outputs 12 future forecast frames (60 minutes), with $N$ ensemble members sampled to form a probabilistic forecast.

### Key Designs

1. **Frame-wise VAE**:

    - **Function**: Compresses individual radar frames from high-dimensional pixel space into compact latent representations.
    - **Mechanism**: A hierarchical encoder–decoder with residual blocks and self-attention, trained with a combination of L1 reconstruction loss, KL divergence, and PatchGAN adversarial loss.
    - **Design Motivation**: Reduces the computational dimensionality for the generative model, consistent with the design philosophy of latent-space diffusion models.

2. **Independent CFM (I-CFM) Training**:

    - **Function**: Trains a vector field $v_\theta$ in latent space to learn the mapping from Gaussian noise to radar latent representations.
    - **Mechanism**: The probability path is defined as $p_t(x_t|x_0,x_1) = \mathcal{N}((1-t)x_0 + tx_1, \sigma^2 I)$ with target vector field $u_t = x_1 - x_0$, and the training objective is $\mathcal{L} = \|v_\theta(Z_t, t, Z_{\text{past}}) - u_t\|^2$. A key design choice is $\sigma > 0$, which provides regularization.
    - **Design Motivation**: Compared to rectified flows (where $\sigma \to 0$), a non-zero $\sigma$ "thickens" the training trajectories, yielding greater stability for high-dimensional data. The straight ODE trajectories of CFM are also better suited to few-step sampling than the curved paths of diffusion models.

3. **FlowCast U-Net Architecture**:

    - **Function**: A spatiotemporal U-Net built on Cuboid Attention layers from Earthformer, conditioned on the flow time $t$.
    - **Mechanism**: An encoder–decoder structure whose core building block is Cuboid Attention (local self-attention within 3D cuboids); the embedding of time step $t$ is injected into each layer.
    - **Design Motivation**: Cuboid Attention efficiently captures local spatiotemporal dynamics, while the hierarchical U-Net structure facilitates global information sharing.

### Loss & Training
- **VAE**: L1 reconstruction + KL divergence (weight $1 \times 10^{-4}$) + PatchGAN adversarial loss.
- **CFM**: Mean squared error regression on the vector field; AdamW optimizer with learning rate $1 \times 10^{-4}$.
- **Sampling**: Euler solver with variable number of function evaluations (NFE $\in \{5, 10, 20, 50, 100\}$).

## Key Experimental Results

### Main Results
Evaluated on the SEVIR dataset (US radar) with 8-member ensemble forecasts:

| Model | Type | CSI-M↑ | FSS-M↑ | CRPS↓ | NFE |
|-------|------|--------|--------|-------|-----|
| Earthformer | Deterministic | baseline | baseline | high | 1 |
| PreDiff | Diffusion | 2nd | 2nd | 2nd | 250 |
| CasCast | Hybrid | strong | strong | strong | 250 |
| FlowCast (50 steps) | CFM | **best** | **best** | **best** | 50 |
| FlowCast (20 steps) | CFM | near-best | near-best | near-best | 20 |

### Ablation Study: CFM vs. Diffusion Objective (Identical Architecture)

| Configuration | CSI-M↑ | CRPS↓ | Notes |
|---------------|--------|-------|-------|
| CFM, 50 steps | best | best | Full proposed method |
| Diffusion, 50 steps | degraded | degraded | Same architecture, diffusion objective |
| CFM, 20 steps | still strong | still strong | High performance retained with fewer steps |
| Diffusion, 20 steps | sharp drop | sharp drop | Performance degrades rapidly with fewer steps |

### Key Findings
- FlowCast with 50 steps surpasses PreDiff and CasCast, both requiring 250 steps, achieving a 5× improvement in computational efficiency.
- A critical ablation study demonstrates that, under an identical architecture, the CFM objective yields higher accuracy and greater robustness to step count than the diffusion objective.
- Results on the ARSO local dataset corroborate the findings, indicating that the method generalizes beyond a specific dataset.
- CFM maintains high performance at 20 steps, whereas diffusion models degrade sharply under the same reduction.

## Highlights & Insights
- **Direct CFM vs. Diffusion Comparison**: By ablating CFM and diffusion objectives under an identical architecture, this work provides the first rigorous and fair comparison in the spatiotemporal forecasting domain, demonstrating that the advantage of CFM arises from the training objective itself rather than architectural differences.
- **Inductive Bias of Straight Trajectories**: The paper offers a compelling insight into meteorological spatiotemporal data—although radar reflectivity distributions are multimodal, their strong temporal continuity makes CFM's linear interpolation paths a better fit than the curved paths of diffusion models.
- **End-to-End Probabilistic Model**: Unlike CasCast, which requires a deterministic backbone followed by diffusion-based refinement, FlowCast performs complete probabilistic modeling from noise to data in a single, more elegant framework.

## Limitations & Future Work
- Validation is limited to 5-minute / 1 km resolution; higher resolutions and longer forecast horizons remain unexplored.
- The VAE is trained separately and then frozen; joint end-to-end training could raise the performance ceiling.
- The ensemble size is fixed at 8; the optimal ensemble size is not analyzed.
- Only the Euler solver is evaluated; higher-order solvers (e.g., RK45) may further reduce the required number of steps.

## Related Work & Insights
- **vs. PreDiff / LDCast**: All three are latent-space generative models, but FlowCast replaces the diffusion objective with CFM, reducing NFE from 250 to 50.
- **vs. CasCast**: CasCast requires two separate models (deterministic + diffusion); FlowCast achieves end-to-end probabilistic modeling with a single model, offering a simpler design.
- **vs. Feng et al.'s rectified flow approach**: Their work uses rectified flows only for deterministic prediction refinement, whereas FlowCast constitutes a complete probabilistic generative model.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First application of CFM as an end-to-end probabilistic model for precipitation nowcasting, with a rigorous ablation design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Two datasets, multiple metrics, CFM vs. diffusion ablation, and step-count sensitivity analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Motivation is clearly articulated, experimental design is principled, and code is publicly available.
- **Value**: ⭐⭐⭐⭐ — Direct implications for operational weather forecasting; establishes CFM as a strong alternative to diffusion models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FlowCast: Trajectory Forecasting for Scalable Zero-Cost Speculative Flow Matching](flowcast_trajectory_forecasting_for_scalable_zero-cost_speculative_flow_matching.md)
- [\[CVPR 2026\] HazeMatching: Dehazing Light Microscopy Images with Guided Conditional Flow Matching](../../CVPR2026/image_generation/hazematching_dehazing_light_microscopy_images_with_guided_conditional_flow_match.md)
- [\[ICLR 2026\] scDFM: Distributional Flow Matching for Robust Single-Cell Perturbation Prediction](scdfm_distributional_flow_matching_model_for_robust_single-cell_perturbation_pre.md)
- [\[NeurIPS 2025\] LeapFactual: Reliable Visual Counterfactual Explanation Using Conditional Flow Matching](../../NeurIPS2025/image_generation/leapfactual_reliable_visual_counterfactual_explanation_using_conditional_flow_ma.md)
- [\[ICLR 2026\] Multi-agent Coordination via Flow Matching](multi-agent_coordination_via_flow_matching.md)

</div>

<!-- RELATED:END -->

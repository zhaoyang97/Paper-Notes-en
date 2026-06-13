---
title: >-
  [Paper Note] Walking the Schrödinger Bridge: A Direct Trajectory for Text-to-3D Generation
description: >-
  [NeurIPS 2025][3D Vision][Text-to-3D Generation] This paper theoretically establishes SDS as a special case of the Schrödinger Bridge, and builds upon this insight to propose TraCe — a framework that constructs an explic…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "Text-to-3D Generation"
  - "Score Distillation Sampling"
  - "Schrödinger Bridge"
  - "Diffusion Models"
  - "3D Gaussian Splatting"
date: 2026-05-08
content_hash: b376eedbc9503e1a
---

# Walking the Schrödinger Bridge: A Direct Trajectory for Text-to-3D Generation

**Conference**: NeurIPS 2025
**arXiv**: [2511.05609](https://arxiv.org/abs/2511.05609)  
**Code**: [GitHub](https://github.com/emmaleee789/TraCe.git)  
**Area**: 3D Vision
**Keywords**: Text-to-3D Generation, Score Distillation Sampling, Schrödinger Bridge, Diffusion Models, 3D Gaussian Splatting

## TL;DR

This paper theoretically establishes SDS as a special case of the Schrödinger Bridge, and builds upon this insight to propose TraCe — a framework that constructs an explicit diffusion bridge between the current rendering and the text-conditioned target, learns the score dynamics along the bridge trajectory via LoRA fine-tuning, and achieves high-quality text-to-3D generation at low CFG values.

## Background & Motivation

The dominant paradigm for text-to-3D generation leverages pretrained T2I diffusion models to optimize 3D representations via Score Distillation Sampling (SDS). However, SDS suffers from two core issues:

**High CFG values are required** (typically 100) to achieve strong text alignment — yet high CFG causes visual artifacts such as **oversaturation** and **oversmoothing**.

**The gradient signal in SDS is inherently noisy** — score estimates from diffusion models are not guaranteed to provide optimal update directions for 3D optimization.

Methods such as VSD attempt to operate at low CFG (e.g., 7.5), but perform poorly on representations like 3DGS. The root cause is that existing methods all **follow gradient directions predicted by diffusion models** whose score functions are trained for 2D image generation, introducing a domain gap with the 3D generation task.

The key insight of this paper is that SDS can be understood through the lens of the Schrödinger Bridge — SDS effectively employs the reverse process of a special Schrödinger Bridge (with one endpoint fixed as Gaussian noise). Based on this theoretical connection, the authors propose constructing a **direct transport trajectory from the current rendering to the target image** (rather than starting from noise), yielding a more stable and direct optimization path.

## Method

### Overall Architecture

Each optimization step of TraCe: render the current 3D model → estimate the target image $x_0^{pred}$ using a pretrained diffusion model → construct a Schrödinger Bridge between the rendering and the target → sample an intermediate state $x_t$ → predict noise with the LoRA model → compute gradients to update 3D parameters $\theta$.

### Key Designs

1. **SDS as a Special Case of the Schrödinger Bridge (Theoretical Contribution)**

   The Schrödinger Bridge finds the most likely stochastic evolution between two arbitrary distributions $P_A$ and $P_B$. When $P_B \approx \mathcal{N}(0,I)$ (Gaussian noise) and the forward Schrödinger factor $\Psi(x,t) \approx 1$, Nelson's duality $\Psi \cdot \hat{\Psi} = q$ yields $\hat{\Psi}(X_t,t) \approx p(X_t,t)$ — the score function $-\nabla_{X_t}\log\hat{\Psi}$ of the reverse SDE reduces to the score $s_\psi(X_t,t)$ learned by a standard diffusion model. SDS thus exploits the reverse score of this special Schrödinger Bridge.

2. **Constructing an Explicit Diffusion Bridge**

   Key steps:
   - **Target endpoint** $X_0 \leftarrow x_0^{pred}$: estimate the ideal target image via **one-step denoising** of the current rendering with the pretrained model
   - **Source endpoint** $X_1 \leftarrow x_{rndr}$: the rendered image of the current 3D model
   - **Intermediate sampling**: sample from the analytically tractable bridge posterior $q(x_t|x_0^{pred}, x_{rndr}) = \mathcal{N}(x_t; \mu_t, \Sigma_t I)$:
   $\mu_t = \gamma_t x_0^{pred} + (1-\gamma_t) x_{rndr}, \quad \gamma_t = \frac{\bar{\sigma}_t^2}{\sigma_t^2 + \bar{\sigma}_t^2}$

3. **LoRA-Based Adaptive Learning of Bridge Trajectory**

   A LoRA-fine-tuned T2I diffusion model $\epsilon_\phi$ is trained to learn the score dynamics along the bridge trajectory. The final gradient is:
   $\nabla_\theta \mathcal{L}_{TraCe}(\theta) = \mathbb{E}_{\epsilon,t,c}\left[w(t)\left(\epsilon_\phi(x_t,t,y,c) - \frac{x_t - x_{rndr}}{\sigma_t}\right)\frac{\partial x_{rndr}}{\partial\theta}\right]$

   Compared to SDS: the noise target in SDS is $\epsilon_{noise}$ (random Gaussian noise), whereas in TraCe it is $\frac{x_t - x_{rndr}}{\sigma_t}$ (the exact noise along the bridge trajectory), yielding more stable and accurate gradient signals.

4. **Progressive $t$ Sampling Schedule**

   During training, $t$ is annealed from 0.5 to 0.02, progressively shifting the interpolation focus of the Schrödinger Bridge from broadly distributed states toward states close to the target $x_0^{pred}$, facilitating progressive refinement of the rendering.

### Loss & Training

- Stable Diffusion is used as the T2I backbone
- 3D Gaussian Splatting is adopted as the 3D representation
- CFG is set to 20 (far below SDS's value of 100)
- LoRA parameters are continuously updated at each optimization step
- Average training time: 14 minutes; VRAM: 18,741 MiB

## Key Experimental Results

### Main Results: Quantitative Comparison on 83 DreamFusion Prompts

| Method | CLIP-L/14↑ | GPTEval3D↑ | ImageReward↑ | Time | VRAM |
|--------|-----------|------------|-------------|------|------|
| SDS | 68.61 | 1018.09 | -0.43 | 10min | 18147M |
| VSD | 67.27 | 1007.49 | -0.53 | 17min | 26473M |
| CSD | 68.03 | 983.04 | -0.67 | 11min | 19804M |
| ISM | 69.01 | 1012.37 | -0.39 | 20min | 10151M |
| SDI | 63.04 | 971.98 | -0.83 | 10min | 16011M |
| **TraCe** | **69.26** | **1028.03** | **-0.29** | 14min | 18741M |

### Ablation Study: LoRA + Progressive $t$ Sampling

| Configuration | ImageReward↑ |
|---------------|-------------|
| Both disabled | -0.4488 |
| Progressive $t$ only | -0.3389 |
| LoRA only | -0.4020 |
| **Both enabled (full model)** | **-0.2486** |

### Key Findings

- TraCe achieves the highest CLIP Score across **all** ViT backbones
- TraCe already reaches high quality at CFG=15–20, without requiring high CFG values — directly addressing the core limitation of SDS
- LoRA and progressive $t$ sampling exhibit **synergistic effects** — their combined gain substantially exceeds the sum of individual contributions
- VSD and CSD perform poorly on 3DGS (as visualized in Figure 4), while TraCe remains consistently strong

## Highlights & Insights

1. **Solid theoretical contribution**: The SDS ↔ Schrödinger Bridge connection provides a novel optimization perspective that is not merely explanatory but directly guides the design of TraCe.
2. The intermediate state $x_t$ constitutes a **meaningful interpolation** between the current rendering and the target (rather than pure noise), yielding more stable gradient signals.
3. The framework is broadly applicable — any distillation-based text-to-3D method may benefit from the bridge trajectory idea introduced in TraCe.

## Limitations & Future Work

- The quality of $x_0^{pred}$ depends on the pretrained model's one-step denoising capability, which may be inaccurate for complex or rare prompts
- The method still relies on the per-view SDS-style optimization paradigm and lacks explicit 3D priors
- A training time of 14 minutes, while reasonable, remains too slow for real-time applications
- The paper does not discuss multi-view consistency (e.g., whether the Janus problem is alleviated)

## Related Work & Insights

- The application of Schrödinger Bridge in generative models can be further extended to image editing, video generation, and beyond
- The LoRA fine-tuning strategy along bridge trajectories is potentially transferable to other score distillation scenarios

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The theoretical connection between SDS and the Schrödinger Bridge is novel; the resulting method design is natural and elegant
- Experimental Thoroughness: ⭐⭐⭐⭐ 83 prompts, 6 competing methods, ablations, and CFG analysis provide comprehensive evaluation
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are rigorous and clear; the motivation chain is complete and well-structured
- Value: ⭐⭐⭐⭐ Provides a new theoretical perspective and practical improvements for text-to-3D optimization

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SegmentDreamer: Towards High-Fidelity Text-to-3D Synthesis with Segmented Consistency Trajectory Distillation](../../ICCV2025/3d_vision/segmentdreamer_towards_high-fidelity_text-to-3d_synthesis_with_segmented_consist.md)
- [\[NeurIPS 2025\] EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting](ef-3dgs_event-aided_free-trajectory_3d_gaussian_splatting.md)
- [\[ICCV 2025\] Benchmarking and Learning Multi-Dimensional Quality Evaluator for Text-to-3D Generation](../../ICCV2025/3d_vision/benchmarking_and_learning_multi-dimensional_quality_evaluator_for_text-to-3d_gen.md)
- [\[AAAI 2026\] AnchorDS: Anchoring Dynamic Sources for Semantically Consistent Text-to-3D Generation](../../AAAI2026/3d_vision/anchords_anchoring_dynamic_sources_for_semantically_consiste.md)
- [\[ICCV 2025\] FlexGen: Flexible Multi-View Generation from Text and Image Inputs](../../ICCV2025/3d_vision/flexgen_flexible_multi-view_generation_from_text_and_image_inputs.md)

</div>

<!-- RELATED:END -->

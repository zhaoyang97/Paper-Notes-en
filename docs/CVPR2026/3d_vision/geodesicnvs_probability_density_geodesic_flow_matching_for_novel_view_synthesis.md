---
title: >-
  [Paper Note] GeodesicNVS: Probability Density Geodesic Flow Matching for Novel View Synthesis
description: >-
  [CVPR 2026][3D Vision][Novel View Synthesis] This paper proposes a Probability Density Geodesic Flow Matching (PDG-FM) framework that replaces the noise-to-data diffusion process with a deterministic data-to-data flow ma…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Novel View Synthesis"
  - "Flow Matching"
  - "Geodesic"
  - "Probability Density Manifold"
  - "Data-to-Data Mapping"
date: 2026-05-08
content_hash: 12e33fefa24f0b3e
---

# GeodesicNVS: Probability Density Geodesic Flow Matching for Novel View Synthesis

**Conference**: CVPR 2026
**arXiv**: [2603.01010](https://arxiv.org/abs/2603.01010)  
**Code**: N/A  
**Area**: 3D Vision
**Keywords**: Novel View Synthesis, Flow Matching, Geodesic, Probability Density Manifold, Data-to-Data Mapping

## TL;DR

This paper proposes a Probability Density Geodesic Flow Matching (PDG-FM) framework that replaces the noise-to-data diffusion process with a deterministic data-to-data flow matching scheme, and optimizes interpolation paths to traverse high-density regions of the data manifold via probability-density-based geodesics, achieving geometrically consistent novel view synthesis.

## Background & Motivation

Novel view synthesis (NVS) aims to generate unseen viewpoints of a scene from limited observations. Although diffusion models achieve high generation quality, they rely on **stochastic noise-to-data transitions**, which obscure deterministic structure and lead to inconsistent cross-view predictions. Flow Matching offers a deterministic alternative, yet existing conditional Flow Matching (CFM) methods largely employ **simple linear interpolation** between source and target data, failing to faithfully capture the non-linear geometry of the data manifold in latent space and thus yielding suboptimal view transitions.

Core problem: how to explicitly incorporate data-dependent geometric regularization into the generation process so that intermediate interpolations in flow matching adhere to the data manifold, thereby improving view consistency?

## Method

### Overall Architecture

PDG-FM consists of two major components:
1. **Data-to-Data Flow Matching (D2D-FM)**: learns a deterministic flow between paired samples $(x_0, x_1)$ without requiring a noise prior.
2. **Variational Distillation of Geodesics**: trains a GeodesicNet to align interpolation paths with the probability density manifold.

### Key Designs

1. **Data-to-Data Flow Matching**: Unlike conventional noise-to-data Flow Matching, D2D-FM establishes a deterministic flow directly between encoded pairs $(x_0, x_1)$ of different viewpoints of the same scene. The velocity network $v_\theta(x_t, t, q, c)$ adopts a U-Net architecture with intermediate state $x_t$ and time $t$ as inputs; conditioning includes Plücker ray embeddings (target camera pose), CLIP-encoded source-view semantic features (injected via cross-attention), and VAE-encoded source-view spatial features (concatenated with $x_t$ as input). Linear interpolation: $x_t = (1-t)x_0 + tx_1 + \sigma_{\min}\epsilon$; target velocity: $u_t = x_1 - x_0$.

2. **Probability Density Geodesic (PDG)**: A local metric tensor $G(x) = p(x)^{-2}I$ is defined inversely proportional to data density, with path length $S[\gamma] = \int_0^1 \|\dot{\gamma}\|_{G(\gamma)} dt$. The geodesic satisfies the Euler–Lagrange equation: $\ddot{\gamma} + \|\dot{\gamma}\|^2 (I - \hat{\dot{\gamma}}\hat{\dot{\gamma}}^\top)\nabla\log p(\gamma) = 0$. The data density gradient $\nabla\log p$ is approximated via the score function of a pretrained diffusion model, estimated at DDIM timestep $\tau=0.6$ using classifier-free guidance.

3. **GeodesicNet Distillation**: A teacher–student architecture is adopted — the teacher network $\phi_\xi$ optimizes geodesic paths in diffusion latent space by minimizing the Euler–Lagrange residual, while the student network $\phi_\eta$ distills these paths into VAE space via DDIM inversion. The geodesic interpolation is parameterized as $x_t = (1-t)x_0 + tx_1 + \phi_\eta(x_0, x_1, t)$, where $\phi_\eta$ satisfies the boundary constraints $\phi_\eta(x_0,x_1,0) = \phi_\eta(x_0,x_1,1) = 0$. This two-stage design decouples geometric optimization from efficient path generation.

### Loss & Training

Three-stage training:
1. **D2D-FM Training**: $\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}[\|v_\theta(x_t,t,q,c) - (x_1-x_0)\|^2]$; source views are augmented with a cosine schedule.
2. **GeodesicNet Distillation**: Teacher minimizes the functional derivative $\ell^\tau(\xi) = \mathbb{E}_t[\text{StopGrad}(g_t) \cdot z_t]$; student minimizes MSE: $\ell^0(\eta) = \mathbb{E}_t[\|x_t - \text{DDIM-B}(z_t,c,\tau)\|^2]$.
3. **Geodesic FM Training**: Fine-tuned from the pretrained velocity network; the target velocity incorporates the time derivative of $\phi_\eta$: $v_{\text{target}} = x_1 - x_0 + \nabla_t\phi_\eta(x_0,x_1,t)$.

- Optimizer: AdamW; batch size: 256; learning rate: $1 \times 10^{-5}$; resolution: 256×256.
- Training data: Objaverse (772k+ 3D objects, 12 rendered views per object).

## Key Experimental Results

### Main Results

| Dataset | Metric | D2D-FM (Ours) | Naive FM | Free3D | Zero-1-to-3 |
|---------|--------|--------------|----------|--------|-------------|
| Objaverse | FID↓ | **5.43** | 5.51 | 5.54 | 6.00 |
| Objaverse | PSNR↑ | **20.84** | 20.82 | 20.32 | 19.59 |
| Objaverse | SSIM↑ | **0.8634** | 0.8622 | 0.8537 | 0.8446 |
| GSO30 | FID↓ | **15.05** | 15.28 | 12.06 | 12.58 |
| Objaverse (10 NFE) | FID↓ | **5.82** | 5.78 | 22.45 | - |

Geodesic FM vs. Linear FM:

| Dataset | Metric | Geodesic FM | Linear FM |
|---------|--------|-------------|-----------|
| Objaverse | FID↓ | **10.40** | 11.81 |
| Objaverse | SSIM↑ | **0.8768** | 0.8736 |
| Objaverse | LPIPS↓ | **0.0804** | 0.0809 |

### Ablation Study

| Configuration | PPL↓ | AOFM↑ | Description |
|---------------|------|-------|-------------|
| Linear Interpolation | **0.213** | 1.04 | Simple blending, minimal geometric motion |
| DDIM Initialization | 0.571 | 6.48 | Motion present but unoptimized |
| Geodesic Interpolation | 0.502 | **13.70** | Strongest geometrically consistent motion along manifold |

### Key Findings

- D2D-FM consistently outperforms Noise-to-Data FM in fidelity and perceptual quality, particularly on FID and LPIPS.
- The advantage of D2D-FM is more pronounced under few-step inference (10 NFE): the diffusion baseline (Free3D) degrades from FID 5.5 to 22.5, whereas D2D-FM degrades only from 5.4 to 5.8.
- The average optical flow magnitude (AOFM) of geodesic interpolation is 13× that of linear interpolation, indicating genuine viewpoint transformation rather than simple blending.
- Geodesic paths exhibit lower Euler–Lagrange residuals, confirming adherence to meaningful manifold structure.

## Highlights & Insights

- **Theoretical Elegance**: Introducing probability density geodesics into conditional Flow Matching provides a rigorous mathematical framework for geometric regularization in generative models.
- **Two-Stage Decoupled Design**: GeodesicNet distillation separates the score-dependent Riemannian metric from flow model training and deployment, ensuring computational efficiency.
- **Data-to-Data Paradigm**: Eliminating the noise prior and establishing deterministic flows directly between structured data pairs yields significant advantages under few-step inference.
- **AOFM as a Novel Evaluation Metric**: Low PPL may correspond to simple blending; AOFM more faithfully reflects the quality of genuine viewpoint transformation.

## Limitations & Future Work

- Experiments are limited to single-object NVS (Objaverse/GSO); validation on scene-level multi-view data is absent.
- GeodesicNet distillation requires the score function of a pretrained diffusion model, increasing training complexity and dependency.
- Generation resolution is limited to 256×256, falling short of current high-resolution generation requirements.
- Evaluation focuses primarily on perceptual metrics, lacking direct measures of 3D consistency (e.g., multi-view reconstruction quality).
- Geodesic optimization in high-dimensional latent spaces may be susceptible to local optima.

## Related Work & Insights

- **Riemannian Flow Matching** (RFM) performs flow matching on fixed geometry; PDG-FM further introduces data-dependent metrics.
- The **Zero-1-to-3** series represents canonical diffusion-based NVS methods; this work demonstrates the advantages of Flow Matching on the same architectural backbone.
- **Metric Flow Matching** (MFM) directly employs Riemannian metrics during training; the two-stage approach proposed here is more computationally efficient.
- **Insights**: The probability density geodesic idea is extensible to video generation, 4D scene modeling, and other tasks requiring spatiotemporal consistency; using the score function as a proxy for manifold metric is a broadly applicable and valuable idea.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of probability density geodesics and Flow Matching is novel, with solid theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐ Evaluation is confined to single-object NVS; scene-level and high-resolution experiments are missing.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are rigorous, the framework is clearly presented, and illustrations are intuitive.
- Value: ⭐⭐⭐⭐ Opens a new direction for geometric regularization in Flow Matching, offering meaningful inspiration to the generative modeling community.
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PR-IQA: Partial-Reference Image Quality Assessment for Diffusion-Based Novel View Synthesis](pr-iqa_partial-reference_image_quality_assessment_for_diffusion-based_novel_view.md)
- [\[ICML 2026\] Geodesic Flow Matching for Denoising High-Dimensional Structured Representations](../../ICML2026/3d_vision/geodesic_flow_matching_for_denoising_high-dimensional_structured_representations.md)
- [\[CVPR 2026\] Physically Inspired Gaussian Splatting for HDR Novel View Synthesis](physically_inspired_gaussian_splatting_for_hdr_novel_view_synthesis.md)
- [\[CVPR 2026\] Scaling View Synthesis Transformers (SVSM)](scaling_view_synthesis_transformers.md)
- [\[CVPR 2026\] PhysGaia: A Physics-Aware Benchmark with Multi-Body Interactions for Dynamic Novel View Synthesis](physgaia_a_physics-aware_benchmark_with_multi-body_interactions_for_dynamic_nove.md)

</div>

<!-- RELATED:END -->

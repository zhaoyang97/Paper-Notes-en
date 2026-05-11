---
title: >-
  [Paper Note] Learning Latent Proxies for Controllable Single-Image Relighting
description: >-
  [CVPR 2026][Image Generation][Single-image relighting] This paper proposes LightCtrl, a diffusion-based single-image relighting framework that achieves precise and continuous control over lighting direction, intensity…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Single-image relighting"
  - "PBR priors"
  - "latent proxy encoder"
  - "DPO post-training"
  - "lighting-aware mask"
date: 2026-05-08
content_hash: 5caffb78db380209
---

# Learning Latent Proxies for Controllable Single-Image Relighting

**Conference**: CVPR 2026
**arXiv**: [2603.15555](https://arxiv.org/abs/2603.15555)
**Code**: N/A
**Area**: Image Relighting / Diffusion Models
**Keywords**: Single-image relighting, PBR priors, latent proxy encoder, DPO post-training, lighting-aware mask

## TL;DR

This paper proposes LightCtrl, a diffusion-based single-image relighting framework that achieves precise and continuous control over lighting direction, intensity, and color temperature. It introduces a few-shot latent proxy encoder to provide lightweight material–geometry priors, a lighting-aware mask to guide spatially selective denoising, and DPO post-training to enhance physical consistency. The method outperforms existing approaches on both synthetic and real-world benchmarks.

## Background & Motivation

Single-image relighting is a severely under-constrained problem: shadows, highlights, and diffuse reflections depend on unobservable geometry and material properties, and small lighting changes can cause large, nonlinear appearance variations. Existing methods exhibit clear capability boundaries:

**Intrinsic/G-buffer methods** (e.g., Neural LightRig) require dense PBR supervision, making them brittle and expensive.

**Pure latent-space methods** (e.g., LBM) lack physical grounding, resulting in unreliable directional and intensity control.

**End-to-end methods** (e.g., IC-Light) perform well on portraits but lack geometric awareness, limiting generalization to complex scenes.

**Key Insight**: Accurate relighting does not require complete intrinsic decomposition. Sparse but physically meaningful cues—indicating *where* lighting should change and *how* materials respond—are sufficient to guide a diffusion model. This motivates the lightweight proxy + mask design.

## Method

### Overall Architecture

LightCtrl is built on a Stable Diffusion backbone:
- **Input**: source image $x_s^{\ell_s}$ + relative lighting encoding $\Delta\ell$ (differences in direction, intensity, and color temperature)
- **Output**: relit result $\hat{x}_s^{\ell_t} = f_\theta(x_s^{\ell_s}, \Delta\ell)$ under the target lighting
- **Condition injection**: appearance token $t_{\mathrm{img}}$, lighting token $t_{\mathrm{light}}$, and physics proxy token $t_{\mathrm{phys}}$

The diffusion loss is spatially weighted:
$$\mathcal{L}_{\mathrm{diff}} = \|W \odot (\epsilon - \epsilon_\theta(z_t, t \mid t_{\mathrm{img}}, t_{\mathrm{light}}, t_{\mathrm{phys}}))\|_2^2$$

### Key Designs

1. **Few-shot Latent Proxy Conditioning**

   A lightweight encoder-decoder $E_\phi$ predicts a compact latent proxy $\hat{\mathcal{B}} = \{a, n, r, m\} \in \mathbb{R}^{H \times W \times 8}$ (albedo, normal, roughness, metalness) from the source image. It is trained with PBR supervision on a small number of samples:

   $$\mathcal{L}_{\text{proxy}} = \lambda_a\|a-\hat{a}\|_1 + \lambda_n(1-\langle n, \hat{n}\rangle) + \lambda_r\|r-\hat{r}\|_1 + \lambda_m \mathrm{BCE}(m, \hat{m})$$

   The proxy maps are spatially pooled and projected into a conditioning token $t_{\text{proxy}} = f_{\text{proj}}(E_\phi(x_s^{\ell_s})) \in \mathbb{R}^{1 \times 768}$ and injected into the denoiser. **Design Motivation**: Rather than pursuing precise intrinsic reconstruction, the proxy only needs to provide sufficient material–geometry cues to constrain the denoising trajectory.

2. **Lighting-Aware Mask Prediction**

   Lighting changes typically affect only a small fraction of pixels (shadow boundaries, specular regions). A soft ground-truth mask is derived from the log-luminance difference between source and target:

   $$M_{\mathrm{gt}} = \mathcal{N}\left(\alpha|\log Y_t - \log Y_s| + (1-\alpha)D_{\mathrm{robust}}(Y_s, Y_t)\right)$$

   Since the target image is unavailable at inference, a lightweight predictor $M_\theta = m_\theta(x_s^{\ell_s}, \Delta\ell)$ estimates the mask from the source image and the lighting delta (supervised with BCE + Dice loss). The mask is converted into a spatial weight map $W$ that modulates the noise reconstruction loss, directing the denoiser's attention toward lighting-sensitive regions.

3. **DPO Post-training for Latent Encoder**

   To compensate for sparse PBR supervision, the main diffusion backbone is frozen and the PBR encoder $E_\phi$ is fine-tuned via DPO-style post-training. GT PBR maps serve as positive samples $y_{\text{pos}}$ and current encoder outputs as negative samples $y_{\text{neg}}$. A physical reward $\Delta r = r(y_{\text{pos}}) - r(y_{\text{neg}})$ aggregates L1, angular, and BCE metrics. A frozen reference encoder provides stable likelihood estimates. The DPO objective increases the likelihood of high-reward predictions, substantially improving the physical consistency of the proxy.

### Loss & Training

- The backbone is fully fine-tuned on ScaLight to learn generalizable light transport priors.
- The proxy branch is trained with few-shot PBR supervision, followed by DPO post-training for improved stability.
- The final diffusion objective uses lighting-aware spatial weighting.
- The **ScaLight** dataset is constructed with 300K+ controllable 3D objects and 1M+ rendered images with systematic variation in lighting direction, intensity, and color temperature, accompanied by complete camera–light metadata.

## Key Experimental Results

### Main Results

Evaluated on the ScaLight test set across three lighting variation types (color temperature / direction / intensity):

| Method | Condition | Temp RMSE↓/PSNR↑ | Pos RMSE↓/PSNR↑ | Energy RMSE↓/PSNR↑ |
|--------|-----------|-----------------|-----------------|-------------------|
| IC-Light | text | 0.397/8.21 | 0.375/8.65 | 0.380/8.63 |
| LBM | image | 0.064/27.8 | 0.084/23.1 | 0.073/25.3 |
| LumiNet | image | 0.172/15.8 | 0.146/17.8 | 0.164/16.2 |
| **Ours (full)** | **Light Info** | **0.053/30.2** | **0.074/25.6** | **0.083/27.1** |

Scene-level evaluation (MIIW):

| Method | RMSE↓ | SSIM↑ | PSNR↑ |
|--------|-------|-------|-------|
| IC-Light | 0.413 | 0.337 | 7.94 |
| LumiNet | 0.139 | 0.904 | 17.20 |
| **Ours** | 0.167 | 0.655 | **18.30** |

User preference study (N=35): scene-level 55.73%, object-level **81.45%**.

### Ablation Study

| Configuration | Temp RMSE↓ | Pos PSNR↑ | Energy PSNR↑ |
|---------------|-----------|-----------|-------------|
| w/o proxy | 0.062 | 22.4 | 18.0 |
| w/o mask | 0.073 | 20.5 | 23.2 |
| w/o DPO | 0.114 | 19.8 | 17.5 |
| **Full** | **0.053** | **25.6** | **27.1** |

### Key Findings

- DPO post-training yields the largest gains across all lighting variation types (removing it roughly doubles RMSE), making it the most critical component of the system.
- The lighting-aware mask is particularly important for directional changes, where shadow boundary accuracy is essential.
- User preference reaches 81.45% at the object level, far exceeding IC-Light (11.45%) and LumiNet (4.3%).

## Highlights & Insights

- **"Middle-path" philosophy**: The method neither pursues complete intrinsic decomposition nor abandons physical grounding; instead, sparse physical cues are used to constrain the diffusion process.
- **DPO for PBR quality optimization**: Applying the RLHF paradigm to intrinsic estimation is a novel cross-domain contribution.
- **ScaLight large-scale dataset**: With 300K+ objects and systematic lighting parameter variation, it fills a significant gap in controllable object-level relighting data.

## Limitations & Future Work

- Scene-level performance still lags behind object-level performance; complex global light transport (e.g., long-range shadow casting) remains a weak point.
- High-frequency geometry and strong specular regions tend to be over-smoothed due to insufficient high-frequency constraints in the proxy.
- Training is primarily conducted on synthetic data; generalization to real-world scenes relies on fine-tuning.

## Related Work & Insights

- **IC-Light**: An end-to-end diffusion relighting method that performs strongly on portraits but lacks physical modeling; this work complements it by introducing physical priors.
- **Neural LightRig**: A dense G-buffer pipeline; this work replaces it with a few-shot proxy approach.
- **LBM**: Latent-space lighting interpolation with weak physical grounding; this work enhances controllability via proxy conditioning and lighting-aware masking.

## Rating

- **Novelty**: ★★★★☆ — The combination of latent proxy conditioning and DPO post-training is novel.
- **Technical Depth**: ★★★★☆ — The three-module design is coherent and well-motivated, with ablations thoroughly validating each component.
- **Experimental Thoroughness**: ★★★★★ — Comprehensive evaluation spanning synthetic benchmarks, real-world scenes, user studies, and ablations; the ScaLight dataset has lasting value.
- **Practicality**: ★★★★☆ — Continuous lighting control is practically valuable, though complex scene performance still requires improvement.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Learning Latent Transmission and Glare Maps for Lens Veiling Glare Removal](learning_latent_transmission_and_glare_maps_for_lens_veiling_glare_removal.md)
- [\[ICLR 2026\] PI-Light: Physics-Inspired Diffusion for Full-Image Relighting](../../ICLR2026/image_generation/pi-light_physics-inspired_diffusion_for_full-image_relighting.md)
- [\[ICLR 2026\] SSCP: Flow-Based Single-Step Completion for Efficient and Expressive Policy Learning](../../ICLR2026/image_generation/flow-based_single-step_completion_for_efficient_and_expressive_policy_learning.md)
- [\[CVPR 2026\] RenderFlow: Single-Step Neural Rendering via Flow Matching](renderflow_single-step_neural_rendering_via_flow_matching.md)
- [\[CVPR 2026\] Self-Corrected Image Generation with Explainable Latent Rewards](self-corrected_image_generation_with_explainable_latent_rewards.md)

</div>

<!-- RELATED:END -->

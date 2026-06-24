---
title: >-
  [Paper Note] Scale-wise Distillation of Diffusion Models
description: >-
  [ICLR 2026][Image Generation][Diffusion Distillation] SwD proposes a "distillation by scale" framework that transforms any pretrained diffusion model into a few-step generator. It progressively increases the resolution at each sampling step—running initial steps at low resolution and only reaching full resolution at the end. This reduces single-step computation by half without increasing total steps. Additionally, a patch-level distillation loss based on MMD is introduced…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Diffusion Distillation"
  - "Scale-progressive Generation"
  - "few-step sampling"
  - "MMD"
  - "T2I/T2V"
date: 2026-05-08
content_hash: ade1df335ec0b34d
---

# Scale-wise Distillation of Diffusion Models

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=Z06LNjqU1g](https://openreview.net/forum?id=Z06LNjqU1g)  
**Paper**: [Project Page](https://yandex-research.github.io/swd)  
**Code**: https://github.com/yandex-research/swd (available on project page)  
**Area**: Diffusion Models / Model Distillation / Image Generation  
**Keywords**: Diffusion Distillation, Scale-progressive Generation, few-step sampling, MMD, T2I/T2V

## TL;DR
SwD proposes a "distillation by scale" framework that transforms any pretrained diffusion model into a few-step generator. It progressively increases the resolution at each sampling step—running initial steps at low resolution and only reaching full resolution at the end. This reduces single-step computation by half without increasing total steps. Additionally, a patch-level distillation loss based on MMD is introduced, which independently approaches SOTA performance, accelerating text-to-image by ~2× and text-to-video by ~3× without degrading image quality.

## Background & Motivation

**Background**: Distilling diffusion models into 1–4 step few-step generators is one of the most successful directions for accelerating large-scale text-to-image and text-to-video generation. Distribution matching methods, such as DMD2 and ADD, can already approach teacher quality within approximately 4 steps.

**Limitations of Prior Work**: Existing methods almost exclusively focus on the "reducing sampling steps" dimension, while freezing other potential degrees of freedom like model architecture and input resolution. The problem is that further compressing steps (4 steps → 2 steps → 1 step) becomes increasingly difficult and leads to significant quality degradation, suggesting that the gains from step reduction are nearly exhausted. Acceleration must be found along other axes.

**Key Challenge**: The diffusion process is inherently "coarse-to-fine." Recent works (Rissanen, Dieleman) have pointed out that reverse diffusion implicitly performs "spectral autoregression," where high-noise steps only recover low-frequency structures, and high-frequency details appear only at low-noise steps. Since high-noise steps contain no high-frequency information, **computing them at full resolution is wasteful**: those high-frequency components masked by noise simply do not exist in the low-resolution latent, making their computation redundant.

**Goal**: (1) Verify whether this "spectral autoregression" also applies to the latent space (VAE latent) and the temporal dimension of video; (2) Design a distillation framework based on this to allow few-step models to progressively increase resolution during sampling; (3) Propose a simpler and more efficient loss for the distribution matching distillation family.

**Key Insight**: The authors first perform spectral analysis (RAPSD) on the VAE latents of models like SD3.5 and Wan2.1, confirming that the spatial and temporal resolutions of latents indeed "implicitly grow" during the diffusion process. High-noise steps can be safely represented at a lower resolution without signal loss. This provides a principled basis for determining "when to use which resolution."

**Core Idea**: Implement **multi-scale progressive generation** (noise → low resolution → step-by-step increase to full resolution) using a single diffusion process and a single few-step model to eliminate redundant computation in intermediate high-noise steps. Use a feature-space MMD loss to achieve fast and high-quality distillation.

## Method

### Overall Architecture

The goal of SwD (Scale-wise Distillation) is to distill a generic pretrained diffusion model into a few-step (4 or 6 steps) generator. The unique feature of this generator is that each step corresponds to **an increasing resolution**. The authors predefine a few-step schedule $[t_1,\dots,t_N]$ and associate each $t_i$ with a non-decreasing scale $s_i$ ($[s_1,\dots,s_N]$). Sampling begins with Gaussian noise at the lowest resolution $s_1$, and each subsequent step increases the resolution until full resolution is reached at the final step. This differs fundamentally from "cascaded diffusion" (cascaded DM), which runs a complete diffusion process at each stage—SwD uses **one** diffusion process and **one** model throughout.

The pipeline consists of three components: first, the "scale/timestep schedule" driven by spectral analysis; second, the "cross-scale transition" mechanism which avoids destroying noise statistics by predicting the clean sample $\hat{x}_0$, upsampling it, and then re-noising; and third, the "distillation loss" (DMD/ADD combined with a patch-level MMD loss) to align the student with the teacher. During training, the model iterates on adjacent scale pairs $[s_i, s_{i+1}]$ to learn both generation and robust upsampling.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Gaussian Noise<br/>Min Resolution s1"] --> B["Spectral Analysis Driven<br/>Scale/Timestep Schedule<br/>High Noise → Low Resolution"]
    B --> C["Scale-wise Sampling<br/>Predict x̂0 → Upsample → Re-noise"]
    C -->|If not full res, scale up s_i→s_{i+1}| C
    C -->|Reach full resolution| D["Full Res Image/Video"]
    C -.Align with Teacher during Training.-> E["Patch-level MMD Distillation Loss<br/>+ DMD/GAN"]
```

### Key Designs

**1. Spectral Analysis Driven Scale Schedule: Lower Resolution for High-Noise Steps**

This provides the physical basis for the method and dictates the resolution for each step. The authors performed Radial Averaged Power Spectral Density (RAPSD) analysis on SD3.5 (128×128 latent) and Wan2.1 (21×60×104 latent), finding that the latent spectrum follows a power law and the noising process **gradually filters high frequencies**. At $t=800$, noise obscures high-frequency components that only appear at resolutions above 32×32; thus, downsampling the 128×128 latent by 4× loses no signal ("green zone"), whereas downsampling by 8× destroys the signal ("red zone"). The same applies to the temporal dimension of video—at $t=600$, the 21-frame signal can be represented by approximately 11 frames. **Summary: Latent space diffusion can be modeled at lower resolutions in high-noise steps without signal loss in both spatial and temporal dimensions.** Consequently, computing full resolution at high-noise steps spends FLOPS on invisible high frequencies. While spectral analysis provides average guidance, the final schedule is treated as a hyperparameter—practically, standard few-step schedules are used with a slight shift towards higher noise, starting from 2–4× lower resolution and scaling up.

**2. Scale-wise Sampling and Training: Upsampling $\hat{x}_0$ to Maintain Noise Statistics**

To solve the challenge of transitioning between scales smoothly, the authors avoid direct upsampling of noisy latents $x_{t_i}$, which distorts noise variance and introduces local correlations. The key observation is that few-step models already predict a clean sample $\hat{x}_0$ before re-noising it for the next step. This provides the **correct timing for upsampling**: upsampling at the $\hat{x}_0$ stage (clean, without noise statistic constraints) and then re-noising maintains correct statistics at the higher resolution. Table 1 (FID-5K, 64→128) validates this: strategy B (upsampling $x_0^{down}$ then re-noising) matches the reference FID at high noise (e.g., 14.7 vs 13.7 at $t=800$), whereas strategy C (noising then upsampling) produces OOD latents (FID 122+). Implementation uses bicubic interpolation for space and adjacent frame fusion for time. During training, the model iterates on scale pairs $[s_i, s_{i+1}]$, using pixel-space downsampling before VAE encoding. This trains the generator to also act as a robust "upsampler" that removes interpolation artifacts.

**3. Patch-level MMD Distillation Loss: Distribution Matching in Teacher Feature Space**

Standard distribution matching (DMD/GAN) requires training extra fake-score networks or discriminators. The authors propose a simple alternative: direct Maximum Mean Discrepancy (MMD) matching in the **intermediate feature space of the pretrained teacher DM**. MMD is defined as $\text{MMD}^2(P,Q)=\mathbb{E}_{x,x'\sim P}[k(x,x')]+\mathbb{E}_{y,y'\sim Q}[k(y,y')]-2\mathbb{E}_{x\sim P,y\sim Q}[k(x,y)]$. The generator and real samples are noised within a preset range, and features $F\in\mathbb{R}^{N\times L\times C}$ (where $L$ is the number of spatial tokens) are extracted from teacher transformer blocks. Using a linear kernel simplifies the loss to **per-image MSE of spatial-averaged tokens**:

$$\mathcal{L}_{\text{MMD}}=\sum_{n=1}^{N}\left\|\frac{1}{L}\sum_{l=1}^{L}F^{\text{real}}_{n,l,\cdot}-\frac{1}{L}\sum_{l=1}^{L}F^{\text{fake}}_{n,l,\cdot}\right\|^2.$$

A critical detail: feature averaging must be **per-image** rather than over the entire batch to avoid washing out condition-dependent information. This loss can be viewed as an adaptation of "feature matching loss" for diffusion distillation, but utilizing pretrained DM features and multi-noise-level feedback without extra trainable models.

### Loss & Training
- Total Objective: $\mathcal{L}_{\text{SwD}}=\mathcal{L}_{\text{MMD}}+\alpha\cdot\mathcal{L}_{\text{DMD}}+\beta\cdot\mathcal{L}_{\text{GAN}}$. For SDXL and Wan2.1, where DMD fails at low resolutions, only $\mathcal{L}_{\text{MMD}}$ is used.
- Data: Uses **synthetic data generated by the teacher** (isolated distillation), achieving fast convergence (~3K iterations) with much lower data requirements than training a DM from scratch.
- Settings: Distilled to 4 or 6 steps; T2I scales from 256/512 to 1024; T2V scales from 21×160×272 to 81×480×832.

## Key Experimental Results

### Main Results (T2I, Table 3 Excerpt)

| Model | Steps | Latency(s/img) | PS↑(MJHQ) | HPSv3↑ | IR↑ | GenEval↑ |
|------|------|-----------|-----------|--------|-----|----------|
| SD3.5-L (Teacher) | 28 | 8.3 | 21.8 | 10.4 | 1.04 | 0.70 |
| SD3.5-L-Turbo | 4 | 0.63 | 21.7 | 9.9 | 0.9 | 0.70 |
| **SD3.5-L-SwD** | 4 | **0.39** | 21.8 | **11.1** | **1.12** | **0.71** |
| FLUX (Teacher) | 30 | 10.0 | 21.7 | 10.7 | 0.93 | 0.66 |
| FLUX-Schnell | 4 | 1.41 | 21.5 | 10.3 | 0.96 | 0.69 |
| **FLUX-SwD** | 4 | **0.72** | **21.9** | **11.6** | **1.06** | **0.71** |

SwD achieves SOTA in PS/HPSv3/IR/GenEval within each model family, with latency nearly 2× lower than the fastest competitors, often **outperforming the teacher while being 10× faster**.

### Main Results (T2V, Table 2)

| Model | Latency(s/vid) | VisionReward↑ | VideoReward↑ | VBench2 Overall↑ |
|------|-------------|---------------|--------------|------------------|
| Wan2.1 (Teacher) | 137 | 0.038 | 5.43 | 51.6 |
| CausVid (3 steps) | 4.2 | 0.042 | 6.21 | 52.3 |
| Spatial SwD (4 steps) | 2.1 | 0.064 | 6.15 | 52.8 |
| **SwD (4 steps, Spatio-temporal)** | **1.8** | **0.064** | **6.27** | **53.2** |

SwD is **72×** faster than the teacher with higher quality; ~2.3× faster than CausVid with comparable quality.

### Ablation Study (MMD Loss, SD3.5-M SwD, MJHQ30K, Table 6)

| Config | PS↑ | HPSv3↑ | IR↑ | FID↓ | Note |
|------|-----|--------|-----|------|------|
| $\mathcal{L}_{\text{SwD}}$ (Full) | 21.8 | 10.7 | 1.11 | 13.6 | Main Model |
| Only $\mathcal{L}_{\text{MMD}}$ | 21.5 | 10.5 | 1.15 | 13.8 | MMD alone is strong |
| w/o $\mathcal{L}_{\text{MMD}}$ | 21.2 | 9.7 | 0.91 | 19.5 | Significant degradation |
| B: Batch Average | 21.5 | 10.5 | 0.97 | 16.4 | Performance drops |
| C: No Noise | 21.3 | 10.2 | 1.01 | 16.6 | Performance drops |

### Key Findings
- **MMD Loss drives quality**: Removing $\mathcal{L}_{\text{MMD}}$ makes FID worsen from 13.6 to 19.5. Using MMD alone is highly competitive and offers 7× faster iterations since no extra models are trained.
- **Critical design details**: Switching to batch-level averaging or removing noise before feature extraction significantly degrades FID, proving that per-image averaging preserves conditional information and noise-level feedback is essential.
- **Scale-wise vs. Full Resolution**: At equal steps, scale-wise maintains or improves quality. At equal latency (Scale-wise 4-step vs. Full-res 2-step), scale-wise is superior as it avoids the high artifacts of 2-step full-resolution baselines.

## Highlights & Insights
- **From "Reducing Steps" to "Reducing Resolution Per Step"**: The authors identified a new axis for acceleration from spectral analysis—high-noise steps don't need full resolution. This is a transferable insight for any few-step diffusion.
- **Upsampling Timing**: Upsampling at the $\hat{x}_0$ stage vs. noisy latent stage determines whether cross-scale transitions preserve noise statistics. Few-step models naturally facilitate this timing.
- **MMD Loss without Extra Models**: By leveraging the teacher's own feature space, the loss eliminates the need for score/discriminator networks, making it low-cost and highly practical for distillation pipelines.

## Limitations & Future Work
- **Low-resolution Generation Quality**: If the base model cannot generate reasonable images at low resolution (e.g., SDXL), DMD loss fails in scale-wise mode, requiring a fallback to pure MMD.
- **Heuristic Schedules**: While spectral analysis provides a guide, the exact scale/step schedule remains a hyperparameter and is not yet automated.
- **FID Limitations**: FID correlations with human perception are imperfect; conclusions rely heavily on human evaluation and preference metrics like PS/HPSv3.
- **Future**: The goal is to develop the MMD loss into a completely self-contained distillation pipeline.

## Related Work & Insights
- **vs. Cascaded/Progressive Diffusion**: Unlike methods that run multiple diffusion processes or require specific techniques for transition continuity, SwD uses one process and one model, handling scales naturally via $\hat{x}_0$ upsampling.
- **vs. DMD2 / ADD**: SwD is complementary, reusing their sampling algorithms while adding scale-wise efficiency and MMD loss.
- **vs. DMMD**: SwD differs by extracting MMD from **pretrained DM feature spaces**, using multi-noise feedback and per-image averaging to create a stronger distribution matching objective.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Resolving "per-step resolution" + Feature-space MMD are both original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers SDXL/SD3.5/FLUX/Wan2.1, T2I+T2V, and comprehensive metrics.
- Writing Quality: ⭐⭐⭐⭐ Clear logic from spectral analysis to method to experiments.
- Value: ⭐⭐⭐⭐⭐ Plug-and-play, 2-3× speedup without quality loss, highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency](large_scale_diffusion_distillation_via_score-regularized_continuous-time_consist.md)
- [\[ICLR 2026\] Stage-wise Dynamics of Classifier-Free Guidance in Diffusion Models](stage-wise_dynamics_of_classifier-free_guidance_in_diffusion_models.md)
- [\[ICLR 2026\] Joint Distillation for Fast Likelihood Evaluation and Sampling in Flow-based Models](joint_distillation_for_fast_likelihood_evaluation_and_sampling_in_flow-based_mod.md)
- [\[ICLR 2026\] RealUID: Supervising the Distillation of All Matching Models with Real Data (Without GAN)](universal_inverse_distillation_for_matching_models_with_real-data_supervision_no.md)
- [\[ICLR 2026\] ViPO: Visual Preference Optimization at Scale](vipo_visual_preference_optimization_at_scale.md)

</div>

<!-- RELATED:END -->

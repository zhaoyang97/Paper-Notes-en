---
title: >-
  [Paper Note] Fewer Denoising Steps or Cheaper Per-Step Inference: Towards Compute-Optimal Diffusion Model Deployment
description: >-
  [ICCV 2025][Image Generation][diffusion model acceleration] This paper proposes PostDiff — a training-free diffusion model acceleration framework that reduces redundancy at two levels: at the input level via a mixed-resolution denoising strategy (low resolution in early steps → high resolution in later steps), and at the module level via a hybrid caching strategy (DeepCache + cross-attention caching). The work systematically addresses the key question of whether reducing the number of denoising steps or reducing the per-step computation cost is more effective — concluding that the latter is superior across most efficiency regimes.
tags:
  - ICCV 2025
  - Image Generation
  - diffusion model acceleration
  - mixed-resolution denoising
  - module caching
  - training-free compression
  - compute-optimal deployment
date: 2026-05-08
content_hash: fdd562dc80075756
---

# Fewer Denoising Steps or Cheaper Per-Step Inference: Towards Compute-Optimal Diffusion Model Deployment

**Conference**: ICCV 2025
**arXiv**: [2508.06160](https://arxiv.org/abs/2508.06160)
**Code**: [https://github.com/GATECH-EIC/PostDiff](https://github.com/GATECH-EIC/PostDiff)
**Area**: Diffusion Models / Model Compression
**Keywords**: diffusion model acceleration, mixed-resolution denoising, module caching, training-free compression, compute-optimal deployment

## TL;DR

This paper proposes PostDiff — a training-free diffusion model acceleration framework that reduces redundancy at two levels: at the input level via a mixed-resolution denoising strategy (low resolution in early steps → high resolution in later steps), and at the module level via a hybrid caching strategy (DeepCache + cross-attention caching). The work systematically addresses the key question of whether reducing the number of denoising steps or reducing the per-step computation cost is more effective — concluding that the latter is superior across most efficiency regimes.

## Background & Motivation

**State of the Field**: Diffusion models have achieved remarkable success in image and video generation, but their iterative denoising nature and complex model architectures result in substantial computational cost, limiting deployment on resource-constrained platforms.

**Limitations of Prior Work**:
- Reducing the number of denoising steps (e.g., DDIM, DPM-Solver, consistency models) and reducing per-step computation cost (e.g., token merging/pruning, module caching, quantization) represent two major acceleration paradigms.
- However, a systematic study comparing the efficiency–quality trade-offs of these two strategies in the post-training setting has been lacking.
- Reducing steps increases the variance of inter-step feature changes, potentially degrading compression compatibility; retaining more steps preserves inter-step redundancy, making compression more applicable.

**Root Cause**: In post-training deployment scenarios without fine-tuning, it remains unclear whether fewer denoising steps or cheaper per-step inference is preferable — a question critical to both researchers and practitioners, yet without a definitive answer.

**Paper Goals**:
- Propose a unified framework, PostDiff, that simultaneously reduces redundancy at both the input and module levels.
- Systematically compare the two acceleration strategies through controlled experiments.
- Identify and explain the "low-resolution enhancement of low-frequency components → improved final quality" phenomenon in mixed-resolution denoising.

**Starting Point**: Early denoising steps primarily generate low-frequency semantic layouts and do not require high resolution; later steps add high-frequency details and thus benefit from higher resolution. This stage-wise characteristic can be exploited.

**Core Idea**: Apply low-resolution denoising in early steps (enhancing low-frequency components while saving compute), switch to high resolution in later steps to refine details, and combine with module caching — demonstrating that reducing per-step cost is more effective than reducing the number of steps.

## Method

### Overall Architecture

PostDiff consists of two complementary training-free techniques: (1) a **mixed-resolution denoising strategy** at the input level — early denoising steps operate on low-resolution latents, switching to full resolution at a designated step; and (2) a **hybrid module caching strategy** at the module level — combining DeepCache (caching the deep skip branches of U-Net) and cross-attention caching (caching conditional guidance information) for reuse across steps.

### Key Designs

1. **Mixed-Resolution Denoising**

    - **Function**: Dynamically switches input resolution during denoising — low resolution in early steps, high resolution in later steps.
    - **Mechanism**: Initializes a low-resolution latent $x_T^l$ of shape $(\beta w, \beta h)$, where $0 < \beta < 1$ is the scaling factor. At step $t = sT$, the method switches to high resolution: the low-resolution prediction $\hat{x}_0^{l,t}$ is computed via Eq.(2), upsampled via bilinear interpolation as $\hat{x}_0^{f,t} = \text{Upsample}(\hat{x}_0^{l,t})$, and then mapped to a full-resolution latent via the forward diffusion formula $x_t^f = \sqrt{\alpha_t}\hat{x}_0^{f,t} + \sqrt{1-\alpha_t}\epsilon$, after which denoising proceeds normally.
    - **Design Motivation**: Early denoising steps predominantly govern low-frequency semantic layout generation, for which low resolution is both sufficient and beneficial — prior literature and empirical results both indicate that low-resolution early steps enhance low-frequency components, ultimately improving final generation quality (a win-win). This is consistent with the empirical success of cascaded diffusion models, where low-frequency information from the low-resolution stage benefits the high-resolution stage.

2. **Visualization of the "Low-Resolution Enhancement" Phenomenon**

    - **Function**: Visualizes the per-step evolution of CLIP Score throughout the denoising process.
    - **Mechanism**: Compares full-resolution denoising against mixed-resolution denoising under various settings of $s$. Observations include: (a) an appropriate number of low-resolution steps can improve the final CLIP Score; (b) although low-resolution steps yield lower CLIP Scores initially, the score recovers more rapidly after switching to high resolution and ultimately surpasses the full-resolution baseline; (c) the balance between low- and high-resolution steps is critical.
    - **Design Motivation**: This visualization not only explains why the method works (low-resolution steps genuinely enhance low-frequency components) but also provides intuition for selecting the hyperparameter $s$.

3. **Hybrid Module Caching**

    - **Function**: Reuses computation at the module level to reduce redundancy.
    - **Mechanism**: Combines two caching strategies: (a) **DeepCache** — caches the deep skip branches of the U-Net and reuses them for the next $k=2$ steps; (b) **cross-attention caching** — disables CFG after step $m$ and caches the conditional cross-attention, adopting the "Cond" mode as the cache ($CA_{cache} = CA_t^c$), which experiments show to be most effective. Key insight: CFG primarily determines layout in early steps and becomes redundant later; moreover, cross-attention can be precomputed during the low-resolution phase for reuse in subsequent steps.
    - **Design Motivation**: (a) High similarity between inter-step feature maps provides the basis for effective caching; (b) cross-attention primarily conveys text-guided layout information, which stabilizes after early steps and can thus be reused; (c) the two caching strategies are complementary — DeepCache reduces spatial computation while cross-attention caching reduces conditional guidance computation.

### Loss & Training

PostDiff is entirely **training-free** — no fine-tuning or distillation is required, and it can be applied directly to pretrained diffusion models. Hyperparameters are determined efficiently via a small calibration set whose performance is shown to be highly correlated with that of the full evaluation set.

Core hyperparameters:
- $\beta$: low-resolution scaling factor (1/2 for SD V1.5; 3/4 for SDXL/PixArt)
- $s$: low-to-high resolution switching point (typically 1/2 or 1/5)
- $m$: step at which CFG is disabled (typically 5–15)

## Key Experimental Results

### Main Results: Performance Across Multiple SOTA Diffusion Models

| Model | Steps | Mix | Cache | FID ↓ | CLIP Score ↑ | FLOPs (T) ↓ | Latency (s) ↓ |
|------|------|-----|-------|-------|-------------|-------------|----------|
| SD V1.5 | 20 | | | 18.42 | 30.80 | 30.420 | 2.930 |
| SD V1.5 | 8 | | | 20.60 | 30.41 | 12.168 | 1.298 |
| SD V1.5 | 20 | ✓ | | 15.69 | 30.78 | 19.035 | 1.945 |
| **SD V1.5** | **20** | **✓** | **✓** | **16.65** | **30.25** | **11.129** | **1.139** |
| SDXL | 20 | | | 14.10 | 31.95 | 119.641 | 6.521 |
| SDXL | 8 | | | 18.01 | 30.92 | 47.856 | 2.843 |
| **SDXL** | **20** | **✓** | **✓** | **14.18** | **31.11** | **52.682** | **3.119** |
| PixArt-α | 20 | | | 29.16 | 30.41 | 85.640 | 7.093 |
| PixArt-α | 8 | | | 33.09 | 30.21 | 34.256 | 3.031 |
| **PixArt-α** | **20** | **✓** | **✓** | **25.44** | **30.23** | **54.718** | **4.768** |

Notable finding: on SD V1.5, PostDiff achieves a 63.14% reduction in FLOPs while simultaneously improving FID by 1.8.

### Ablation Study: Cross-Attention Caching Strategy Comparison

| Configuration | FLOPs (T) ↓ | FID ↓ | CLIP Score ↑ |
|------|-------------|-------|-------------|
| Original | 30.420 | 18.42 | 30.80 |
| DeepCache (DC) | 17.787 | 17.79 | 30.75 |
| DC + CA (m=5, Ave) | 11.610 | 18.77 | 28.40 |
| DC + CA (m=5, Cond) | 11.610 | 18.82 | 29.20 |
| DC + CA (m=5, CFG) | 11.610 | 103.71 | 18.22 |
| DC + CA (m=10, Cond) | 15.061 | 21.26 | 30.11 |
| DC + CA (m=15, Cond) | 16.360 | 21.67 | 30.37 |

Comparison with other training-free methods (SD V1.5):

| Method | FID ↓ | CLIP Score ↑ | Latency (s) ↓ |
|------|-------|-------------|----------|
| Original | 18.42 | 30.80 | 2.930 |
| DeepCache | 17.79 | 30.75 | 1.737 |
| TGATE | 19.51 | 29.55 | 1.992 |
| ToMe | 17.43 | 30.55 | 2.730 |
| **PostDiff** | **16.65** | **30.25** | **1.139** |

### Key Findings
- **Reducing per-step cost > reducing the number of steps**: When the goal is to maintain high generation quality (FID < 20), retaining more steps while using PostDiff to reduce per-step cost is superior. Only when pursuing extreme efficiency (> 60% FLOPs reduction) does reducing the number of steps become more favorable.
- **Mixed-resolution is a win-win**: An appropriate number of low-resolution early steps not only saves compute but also improves final FID by enhancing low-frequency components (optimal FID on SD V1.5 improves from 18.42 to 15.69).
- **CFG caching should not be applied too early**: Completely disabling CFG at $m=5$ (CFG mode) causes quality collapse (FID > 100); the "Cond" mode remains stable across all settings.
- **Cross-architecture generality**: PostDiff is effective across U-Net and Transformer architectures, large and small models, and both LDM and LCM variants.
- **PostDiff achieves the lowest latency**: 1.139s on SD V1.5 versus 1.737s for the next best method, DeepCache (−34%).

## Highlights & Insights

- **Systematic answer to the "fewer steps vs. cheaper steps" question**: Prior work lacked a fair comparison of these two acceleration paradigms. PostDiff, as a unified framework, enables controlled experiments and yields the actionable conclusion that per-step cost is more important — directly informing deployment decisions.
- **Low-resolution enhancement of low-frequency components**: This is more than an engineering trick; it reflects a deeper insight into the diffusion process — low-resolution denoising in early steps effectively "forces" the model to focus on low-frequency semantic structures, reducing interference from high-frequency noise, consistent with the principles underlying the success of cascaded diffusion models.
- **Fully training-free**: No fine-tuning or distillation is required; PostDiff can be applied plug-and-play to any pretrained diffusion model, making it highly practical.

## Limitations & Future Work

- The paper adopts a simple binary resolution schedule (low → high); more sophisticated progressive resolution schedules may yield further improvements.
- Cross-attention caching relies on the CFG mechanism and its applicability to models that do not use CFG (e.g., flow matching) remains to be validated.
- The combination with training-aware compression methods such as quantization and pruning has not been explored.
- Although efficient, the calibration-set-based hyperparameter selection still incurs some computational overhead.
- Validation is limited to image generation; video diffusion models may exhibit different redundancy patterns.

## Related Work & Insights

- **vs. DeepCache**: DeepCache caches only the U-Net skip branches; PostDiff additionally incorporates cross-attention caching and mixed-resolution denoising, with all three components jointly pushing the efficiency–quality frontier.
- **vs. TGATE**: TGATE completely disables CFG after step $m$, which is overly aggressive and degrades quality; PostDiff's cached cross-attention is more conservative, preserving partial conditional information.
- **vs. ToMe/ToDo**: These methods reduce redundancy at the token level (merging/pruning); PostDiff operates at the resolution level — the two approaches are orthogonal and potentially composable.
- **vs. few-step diffusion models (LCM, consistency models)**: These require additional training or distillation costs; PostDiff is entirely training-free and can also be combined with LCM (validated experimentally).

## Rating

- Novelty: ⭐⭐⭐⭐ The mixed-resolution denoising strategy is elegant and effective; the research angle of systematically comparing two acceleration paradigms is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four SOTA models × multiple configurations; detailed FID–FLOPs trade-off curves; comparison against 6+ methods.
- Writing Quality: ⭐⭐⭐⭐ The research question is well-focused, the experimental design is sound, and the step-wise CLIP Score visualization is compelling.
- Value: ⭐⭐⭐⭐ A practical training-free acceleration solution combined with systematic guidance on diffusion model deployment strategies.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Inference-Time Diffusion Model Distillation](inference-time_diffusion_model_distillation.md)
- [\[NeurIPS 2025\] Two-Steps Diffusion Policy for Robotic Manipulation via Genetic Denoising](../../NeurIPS2025/image_generation/two-steps_diffusion_policy_for_robotic_manipulation_via_genetic_denoising.md)
- [\[NeurIPS 2025\] AccuQuant: Simulating Multiple Denoising Steps for Quantizing Diffusion Models](../../NeurIPS2025/image_generation/accuquant_simulating_multiple_denoising_steps_for_quantizing.md)
- [\[ICCV 2025\] Compression-Aware One-Step Diffusion Model for JPEG Artifact Removal](compression-aware_one-step_diffusion_model_for_jpeg_artifact_removal.md)
- [\[NeurIPS 2025\] Denoising Weak Lensing Mass Maps with Diffusion Model and Generative Adversarial Network](../../NeurIPS2025/image_generation/denoising_weak_lensing_mass_maps_with_diffusion_model_and_generative_adversarial.md)

<!-- RELATED:END -->

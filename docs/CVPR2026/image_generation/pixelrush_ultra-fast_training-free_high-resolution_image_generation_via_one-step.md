---
title: >-
  [Paper Note] PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion
description: >-
  [CVPR 2026][Image Generation][high-resolution image generation] This paper proposes PixelRush, a training-free high-resolution image generation framework that combines four components — partial inversion…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "high-resolution image generation"
  - "training-free"
  - "diffusion models"
  - "few-step diffusion"
  - "patch-based inference"
date: 2026-05-08
content_hash: c7cb6e4960214efc
---

# PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion

**Conference**: CVPR 2026
**arXiv**: [2602.12769](https://arxiv.org/abs/2602.12769)  
**Code**: None  
**Area**: Image Generation
**Keywords**: high-resolution image generation, training-free, diffusion models, few-step diffusion, patch-based inference

## TL;DR

This paper proposes PixelRush, a training-free high-resolution image generation framework that combines four components — partial inversion, few-step diffusion models, Gaussian filter blending, and noise injection — to compress 4K image generation time from several minutes to approximately 20 seconds (10×–35× speedup), while surpassing existing SOTA methods on FID/IS metrics.

## Background & Motivation

Pre-trained diffusion models such as SDXL excel at generating high-quality images, but are constrained by fixed training resolutions (1024×1024 for SDXL). Direct inference at resolutions beyond the training distribution leads to severe structural artifacts and quality degradation. Fine-tuning to target resolutions faces three major obstacles: scarcity of high-resolution data, prohibitive training costs, and model lock-in to specific resolutions.

Existing training-free high-resolution generation methods fall into two categories:

**Direct inference methods** (e.g., ScaleCrafter, FreeScale): These operate on full high-resolution latents, mitigating object repetition through modified convolution dilation rates or frequency-domain interventions. However, frequency-domain operations introduce unnatural textures, and memory consumption scales with latent size, typically limiting applicability to below 8K.

**Patch-based methods** (e.g., DemoFusion, MultiDiffusion): These divide high-resolution latents into overlapping patches matching the model's native resolution, bypassing memory bottlenecks. However, like direct inference methods, they rely on full multi-step reverse diffusion (e.g., 50 steps), causing 4K generation to take several minutes and 8K to exceed one hour.

Existing acceleration attempts have achieved limited gains: CutDiffusion achieves marginal speedup by reducing patch count at the cost of quality; LSNR requires training additional plugin modules and only reduces steps from 50 to 30. **The root cause is that existing methods are incompatible with fast few-step sampling**, which constitutes the primary obstacle to practical deployment.

PixelRush's core insight is: since the reverse diffusion process reconstructs images in a frequency-hierarchical manner — recovering low-frequency global structure first, then high-frequency details — and since a coarsely upsampled image already contains low-frequency information, **perturbing the latent to full Gaussian noise and denoising from scratch is redundant**. It suffices to begin from a shallow intermediate noise level and focus solely on high-frequency detail synthesis.

## Method

### Overall Architecture

PixelRush adopts a classic two-stage pipeline: base generation followed by cascaded upsampling.

**Base generation stage**: Given a text prompt and target resolution, a multi-step diffusion model (e.g., SDXL) generates a base image at native resolution.

**Cascaded upsampling stage**: Resolution is doubled at each stage (4× area increase), following the pipeline: pixel-space upsampling → VAE encoding to obtain a coarse latent → **Refinement Stage** for high-frequency detail synthesis → VAE decoding. Multiple cascades yield target resolutions such as 4K and 8K.

The refinement stage constitutes the core contribution of PixelRush, consisting of four key components:

### Key Design 1: Partial Inversion

**Core problem**: Existing methods perturb coarse latents to full Gaussian noise $\mathbf{z}_T$ ($t=T$) and perform complete 50-step reverse diffusion. However, since reverse diffusion reconstructs frequency content hierarchically — early steps primarily recover low-frequency global structure, while later steps synthesize high-frequency details — the early denoising steps are redundant for latents that already possess global structure.

**Solution**: The coarse latent is mapped via DDIM inversion only to a shallow intermediate noise level $\mathbf{z}_K$ ($K \ll T$), rather than to full Gaussian noise. For example, truncating at $t=259$ (rather than 999) saves approximately 75% of computation. Experiments confirm (Table 3) that replacing 50-step denoising with 15-step partial inversion reduces inference time from 67 seconds to 18 seconds (3.7× speedup) while improving FID from 54.70 to 52.90.

### Key Design 2: Few-Step Model Acceleration

The short truncated reverse trajectory formed by partial inversion is naturally compatible with few-step diffusion models (e.g., SDXL-Turbo), since such models produce large updates per step and can synthesize the required high-frequency details within a very short trajectory.

Concretely, both forward perturbation and reverse refinement are performed in a **single step**. The intermediate timestep $K$ corresponding to the few-step model's schedule (e.g., $K=249$ in SDXL-Turbo's 4-step schedule) is selected, and one-step DDIM inversion followed by one-step reverse denoising is applied. Deterministic DDIM inversion is used rather than stochastic $q$-sampling, in order to preserve the structural information of the base image.

This design achieves approximately 10×–35× speedup, but introduces two new problems: checkerboard artifacts at patch boundaries and over-smoothing.

### Key Design 3: Gaussian Filter Blending

**Root cause**: Conventional patch blending (e.g., average blending in MultiDiffusion) performs adequately under multi-step denoising but fails in few-step or single-step settings, where the reverse process produces drastic, sharp updates within each patch. Simple averaging merely blurs the discrepancies without reconciling them, resulting in visible seams.

**Solution**: Inspired by image feathering, the hard binary overlap mask is convolved with a Gaussian blur kernel to produce a smoothly varying weight mask. During blending, pixels closer to the center of a given patch contribute more of that patch's value, achieving smooth gradual transitions. This completely eliminates boundary artifacts even in the single-step setting.

### Key Design 4: Noise Injection

**Root cause**: The large denoising step size of few-step models may be insufficient to fully recover high-frequency details, resulting in over-smoothed outputs.

**Solution**: During the reverse denoising step, the model-predicted noise $\epsilon_\gamma(\mathbf{x},t)$ is spherically interpolated (slerp) with random noise $\epsilon_{\text{rand}}$ using coefficient $\lambda=0.95$:

$$\epsilon'_\gamma(\mathbf{x},t) = \text{slerp}(\epsilon_\gamma(\mathbf{x},t),\, \epsilon_{\text{rand}},\, \lambda)$$

Injecting stochasticity helps flatten the data distribution and promotes synthesis of high-frequency components. Slerp is preferred over lerp because the operation is performed in latent space.

**Note**: This technique is specifically designed for the over-smoothing problem in few-step patch pipelines. Applying it in multi-step pipelines leads to quality degradation due to error accumulation.

### Loss & Training

PixelRush is a **completely training-free** method, involving no loss functions or training procedures. All components intervene solely in the inference process. The only hyperparameters to be specified are: the partial inversion timestep $K$ (recommended: 249), the noise injection interpolation coefficient $\lambda$ (fixed at 0.95), and the patch overlap ratio.

## Key Experimental Results

### Main Results: Quantitative Comparison with SOTA

| Method | 2K FID↓ | 2K IS↑ | 2K Time (s) | 4K FID↓ | 4K IS↑ | 4K Time (s) |
|--------|---------|--------|-------------|---------|--------|-------------|
| SDXL-DI | 73.34 | 10.93 | 28 | 153.53 | 7.32 | 247 |
| FouriScale | 72.65 | 12.31 | 87 | 98.97 | 8.54 | 680 |
| DemoFusion | 68.46 | 13.15 | 75 | 74.75 | 12.57 | 507 |
| FreeScale | 52.87 | 13.56 | 53 | 58.28 | 13.35 | 323 |
| **PixelRush** | **50.13** | **14.32** | **4** | **54.67** | **13.75** | **20** |

PixelRush outperforms SOTA across all metrics, generating 2K images in just 4 seconds (13× faster than FreeScale) and 4K images in 20 seconds (16× faster than FreeScale, 34× faster than FouriScale).

### Ablation Study: Contribution of Each Component

| Configuration | Denoising Steps | FID↓ | IS↑ | Time (s) |
|---------------|----------------|------|-----|----------|
| Baseline (50-step DDIM) | 50 | 54.70 | 13.92 | 67 |
| + Partial Inversion | 15 | 52.90 | 13.89 | 18 |
| + Few-step Model | 1 | 57.23 | 13.65 | 4 |
| + Gaussian Blend | 1 | 56.16 | 13.77 | 4 |
| + Noise Injection (PixelRush) | 1 | **50.13** | **14.32** | **4** |

- Partial inversion: 3.7× speedup with quality improvement (FID 54.70→52.90)
- Few-step model: further acceleration to 4 seconds, but FID degrades to 57.23
- Gaussian blending: eliminates checkerboard artifacts, FID improves to 56.16
- Noise injection: resolves over-smoothing, FID drops substantially to 50.13, surpassing the baseline comprehensively

### Key Findings

1. **The choice of partial inversion timestep is critical**: $K=249$ (shallowest level) achieves the best FID of 50.13; as $K$ increases (499→749→999), performance degrades monotonically (FID 66.24→72.34→79.45), due to incompatibility between multi-step DDIM inversion and few-step models.
2. **Robustness to model selection**: Consistent performance is observed across different base/refinement model combinations (SDXL+SDXL-Turbo, SDXL+SD-Turbo, SANA+SDXL-Turbo, SDXL+Pixart-δ), demonstrating the generality of the approach.
3. **Qualitative analysis**: SDXL-DI produces severe object repetition and unnatural textures; DemoFusion exhibits structural artifacts (e.g., duplicated dragon heads); FouriScale/FreeScale introduce grid/noise textures; PixelRush maintains sharp, natural details with intact global structure.

## Highlights & Insights

1. **High-value core insight**: The frequency-hierarchical reconstruction property of diffusion models implies that refinement tasks do not require a complete reverse process — this observation, while conceptually simple, was overlooked by all prior methods, and yields an order-of-magnitude speedup.
2. **Targeted component design**: Each component precisely addresses one specific problem (partial inversion → redundant computation; few-step model → further acceleration; Gaussian blending → boundary artifacts; noise injection → over-smoothing), forming a clear and logically progressive solution.
3. **Breaking the speed-quality tradeoff**: PixelRush improves generation quality while dramatically accelerating inference (FID reduced from 52.87 to 50.13), challenging the conventional assumption that acceleration necessarily sacrifices quality.
4. **Practical breakthrough**: For the first time, 8K image generation on a single A100 GPU within 100 seconds is demonstrated, transforming high-resolution generation from an offline task into a practically deployable real-time tool.

## Limitations & Future Work

1. **Dependence on distilled models**: The acceleration capability relies on the availability of few-step distilled models (e.g., SDXL-Turbo); compatibility with newer model architectures lacking distilled variants (e.g., DiT-based models) remains to be verified.
2. **Fixed hyperparameters**: The noise injection coefficient $\lambda=0.95$ is fixed across all experiments; adaptive tuning may be beneficial for different content types and styles.
3. **Limited evaluation metrics**: Only FID and IS are used; evaluation of text alignment (e.g., CLIP score) and human preference is absent.
4. **Global structural consistency**: Patch-based methods are inherently challenged by global consistency; edge cases such as geometric continuity in panoramic images may still be problematic.
5. **Extension to video**: The current framework targets image generation only; extending analogous strategies to high-resolution video generation represents a valuable future direction.

## Related Work & Insights

- **MultiDiffusion / DemoFusion**: Established the patch-based high-resolution generation paradigm, but are bottlenecked by multi-step denoising costs. PixelRush preserves the patch-based framework while achieving qualitative acceleration through partial inversion and few-step models.
- **FreeScale / FouriScale**: Address object repetition via frequency-domain intervention, but introduce texture artifacts. PixelRush uses DDIM-inverted latents as structural priors, naturally avoiding object repetition.
- **SDXL-Turbo / consistency distillation**: The development of few-step diffusion models provides the infrastructure underpinning PixelRush, demonstrating that distilled models can be effectively integrated into high-resolution refinement pipelines beyond their original use case.
- **HiWave**: Proposes replacing random Gaussian noise with DDIM-inverted noise to preserve structural information — a concept inherited and extended by PixelRush to the few-step extreme.

## Rating

| Dimension | Score (1–10) | Notes |
|-----------|-------------|-------|
| Novelty | 8 | The combination of partial inversion and few-step models is novel; the insight is concise and profound |
| Technical Depth | 7 | Each component has clear design objectives and solid analysis, though individual techniques are not highly complex |
| Experimental Thoroughness | 8 | Main experiments, multi-dimensional ablations, model robustness analysis, and qualitative comparisons provide comprehensive coverage |
| Practical Value | 9 | 10×–35× speedup with quality improvement directly addresses real-world deployment bottlenecks |
| Writing Quality | 8 | Motivation is clear, logic is progressive, and figures and tables are informative |
| **Overall** | **8.0** | A highly practical contribution that resolves the speed bottleneck in high-resolution generation in a concise and elegant manner |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Language-Free Generative Editing from One Visual Example](language-free_generative_editing_from_one_visual_example.md)
- [\[CVPR 2026\] Low-Resolution Editing is All You Need for High-Resolution Editing](low-resolution_editing_is_all_you_need_for_high-resolution_editing.md)
- [\[CVPR 2026\] HAM: A Training-Free Style Transfer Approach via Heterogeneous Attention Modulation for Diffusion Models](ham_a_training-free_style_transfer_approach_via_heterogeneous_attention_modulati.md)
- [\[CVPR 2026\] Object-WIPER: Training-Free Object and Associated Effect Removal in Videos](object-wiper_training-free_object_and_associated_effect_removal_in_videos.md)
- [\[CVPR 2026\] DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution](duo-vsr_dual-stream_distillation_for_one-step_video_super-resolution.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] The Silent Assistant: NoiseQuery as Implicit Guidance for Goal-Driven Image Generation
description: >-
  [ICCV 2025][Image Generation][Initial noise optimization] This paper proposes NoiseQuery, a training-free T2I generation enhancement method that pre-constructs a large-scale noise library and retrieves the initial noise best matching the user's goal at inference time, enabling fine-grained control over both high-level semantics and low-level visual attributes, with only 0.002s/prompt additional overhead, improving performance across multiple T2I models and enhancement techniques.
tags:
  - ICCV 2025
  - Image Generation
  - Initial noise optimization
  - Noise Library
  - cross-model consistency
  - low-level visual attribute control
  - T2I enhancement
date: 2026-05-08
content_hash: be77ef274c9ad068
---

# The Silent Assistant: NoiseQuery as Implicit Guidance for Goal-Driven Image Generation

**Conference**: ICCV 2025
**arXiv**: [2412.05101](https://arxiv.org/abs/2412.05101)
**Code**: [https://github.com/wangruoyu02/NoiseQuery](https://github.com/wangruoyu02/NoiseQuery)
**Area**: Diffusion Models / Image Generation
**Keywords**: Initial noise optimization, Noise Library, cross-model consistency, low-level visual attribute control, T2I enhancement

## TL;DR

This paper proposes NoiseQuery, a training-free T2I generation enhancement method that pre-constructs a large-scale noise library and retrieves the initial noise best matching the user's goal at inference time, enabling fine-grained control over both high-level semantics and low-level visual attributes, with only 0.002s/prompt additional overhead, improving performance across multiple T2I models and enhancement techniques.

## Background & Motivation

**State of the Field**: Diffusion models have achieved remarkable success in T2I generation, yet still suffer from misalignment between generated content and text prompts. Existing enhancement methods include fine-tuning the UNet (DPO), enhancing text encoders (LaVi-Bridge), improving inference strategies (CFG++), and optimizing initial noise (ReNO). Noise optimization has recently attracted attention, as images generated from the same initial noise within a single model often exhibit high similarity.

**Limitations of Prior Work**:
   - Existing noise optimization methods (e.g., ReNO) rely on iterative gradient back-propagation, incurring high computational cost (23.56s vs. baseline 0.072s) and numerical instability (gradient explosion)
   - These methods are typically designed for specific models and cannot be reused across models
   - Text prompts are naturally suited for expressing high-level semantics but offer weak control over low-level visual attributes (color, texture, sharpness, etc.), since text encoders such as CLIP primarily learn high-level semantics

**Root Cause**: The initial noise has a substantial impact on generation outcomes—not only within a single model but also across models—yet existing methods either overlook this influence or exploit it at prohibitive cost. The key challenge is how to efficiently and universally leverage the generative tendency encoded in the initial noise.

**Paper Goals**: (a) Reveal the cross-model generative consistency phenomenon of initial noise and provide a theoretical explanation; (b) construct a reusable noise library for efficient noise retrieval; (c) leverage noise to control low-level visual attributes that are difficult to describe via text.

**Starting Point**: Analysis of the forward noising process in diffusion models shows that a finite-step noise scheduler cannot fully eliminate original image information—e.g., the final step of Stable Diffusion still retains $\sqrt{\bar\alpha_T} = 0.068265$ of the original signal. This causes models to "learn" to exploit residual signals in the noise as a shortcut during training. At inference, models continue to rely on this implicit knowledge to "interpret" pure Gaussian noise, making the initial noise a "silent assistant."

**Core Idea**: Unconditional generation (empty prompt) is used to reveal the implicit generative posterior of the initial noise. A large-scale noise library is pre-built and indexed with multi-granularity features; at inference time, the optimal noise is rapidly retrieved via feature matching, enabling dual control over both semantics and low-level attributes.

## Method

### Overall Architecture

NoiseQuery operates in two stages: (1) **Offline stage**—a large set of Gaussian noise samples (e.g., 100K) is drawn, and the diffusion model generates corresponding generative posterior images under an empty prompt; multi-granularity features (CLIP, color, texture, etc.) are extracted to build the noise library. (2) **Online stage**—given the user's goal (text prompt and/or low-level attribute requirements), corresponding features are extracted and used to retrieve the best-matching noise from the library as the generation starting point.

### Key Designs

1. **Theoretical Analysis of Implicit Generative Tendency**:

    - Function: Theoretically explains why the initial noise affects generation outcomes, and why this effect is consistent across models.
    - Mechanism: The forward process of diffusion models is $x_T = \sqrt{\bar\alpha_T} x_0 + \sqrt{1-\bar\alpha_T} \epsilon$. In theory, $\bar\alpha_T \to 0$ requires infinite steps; however, under a practical finite-step scheduler, $\bar\alpha_T > 0$ (e.g., $\sqrt{\bar\alpha_T} = 0.068265$ in SD), causing residual original image information to persist in $x_T$. Models learn to exploit these residuals as shortcuts during training, and continue to "interpret" similar patterns in pure noise at inference.
    - Design Motivation: This explains why images generated from the same noise by different architectures (UNet-based SD vs. Transformer-based PixArt-α) are highly similar in semantics and visual attributes—both architectures learn to exploit the non-asymptotic property of the same noise scheduler.

2. **Generative Posterior as a Noise Proxy**:

    - Function: Images generated with an empty prompt are used to reveal the implicit information in the noise.
    - Mechanism: Without text guidance, the generation process is entirely driven by the noise, forcing the model to decode residual signals preserved in the noise. The generated images (generative posteriors) exhibit tendencies in semantics, spatial layout, and color tone similar to those observed in subsequent prompt-guided generation.
    - Design Motivation: The generative posterior serves as an interpretable proxy for the implicit information in the noise, providing the operand for feature extraction in the noise library.

3. **Multi-Granularity Feature Indexing and Matching**:

    - Function: Multiple features are extracted from the generative posterior of each noise sample to support multi-objective retrieval.
    - Mechanism: Different feature types and matching functions are used according to different generation objectives—semantics (CLIP/BLIP + cosine similarity), style (Gram Matrix + MSE), color (RGB/HSV/LAB + absolute difference), texture (GLCM + Euclidean distance), shape (Hu Moments + Euclidean distance), and sharpness (high-frequency energy + absolute difference). The retrieval formula is: $\epsilon^* = \arg\max_{\epsilon_i \in \mathcal{N}_{set}} S(\mathcal{F}_i, \mathcal{F}_O)$
    - Design Motivation: Text prompts primarily control high-level semantics, whereas initial noise naturally encodes low-level visual attributes (directly related to pixel space). Noise retrieval thus fills the gap in text-based control. A progressive re-ranking strategy is adopted when multiple objectives are combined.

### Loss & Training

NoiseQuery is entirely training-free and does not modify any model parameters. The offline construction of the 100K noise library is a one-time cost, after which the library can be reused across models.

## Key Experimental Results

### Main Results

Evaluated on DrawBench and MSCOCO, covering 6 models and 4 enhancement techniques:

| Model | Enhancement | ImageReward | PickScore | HPS v2 | CLIPScore | Time |
|-------|-------------|-------------|-----------|--------|-----------|------|
| SD 1.5 | Base | 0.04 | 21.11 | 24.57 | 30.90 | 1.334s |
| SD 1.5 | + NoiseQuery | **0.08** | **21.16** | **25.02** | **31.41** | 1.336s |
| SD 1.5 | + DPO | 0.09 | 21.29 | 25.02 | 31.19 | 1.350s |
| SD 1.5 | + DPO + NoiseQuery | **0.17** | **21.33** | **25.25** | **31.41** | 1.352s |
| SD-Turbo | Base | 0.26 | 21.78 | 25.23 | 31.29 | 0.072s |
| SD-Turbo | + NoiseQuery | **0.41** | **21.87** | **25.66** | **31.58** | 0.074s |
| SD-Turbo | + ReNO | 1.67 | 23.40 | 32.48 | 32.55 | 23.56s |
| SD-Turbo | + ReNO + NoiseQuery | **1.71** | **23.52** | **32.92** | **32.78** | 23.56s |
| PixArt-α | Base | 0.70 | 22.08 | 28.27 | 30.83 | 4.327s |
| PixArt-α | + NoiseQuery | **0.82** | **22.11** | **28.45** | **31.27** | 4.328s |

### Ablation Study

**Effect of noise library size on performance**:

| Library Size | 0.5k | 1k | 2k | 5k | 10k | 50k | 100k |
|--------------|------|----|----|----|----|-----|------|
| Matching time (×10⁻⁴s) | 1.39 | 1.39 | 1.39 | 1.39 | 1.40 | 1.40 | 1.51 |
| Argmax selection (×10⁻⁴s) | 0.25 | 0.25 | 0.25 | 0.25 | 0.37 | 6.16 | 13.25 |
| CLIP Score | 31.51 | 31.53 | 31.57 | 31.59 | 31.66 | 31.73 | 31.74 |

### Key Findings

- **Universal enhancement**: NoiseQuery consistently improves performance across all 6 models and can be stacked with 4 different enhancement methods (DPO, CFG++, ReNO, LaVi-Bridge) without conflicting gains.
- **Minimal overhead**: Only 0.002s additional time per prompt (matching + selection), in sharp contrast to ReNO's 23.56s.
- **Cross-model reuse**: The same noise library can be shared between UNet-based (SD series) and Transformer-based (PixArt-α) models, validating cross-architecture consistency.
- **Effective at low CFG scale**: NoiseQuery enables high-quality outputs at low CFG scales, avoiding the over-saturation associated with high scales.
- **Diminishing returns from larger libraries**: Scaling from 50K to 100K yields only a 0.01 CLIPScore improvement, while a 10K library already captures most of the benefit.
- **Diversity preserved**: Results drawn from the top-20 matched noises maintain good diversity (DIV metric comparable to random noise), indicating that NoiseQuery does not restrict generation diversity.

## Highlights & Insights

- **Creative exploitation of a diffusion model "design flaw"**: Signal residuals caused by finite-step noise schedulers are typically regarded as a problem (e.g., *common diffusion noise schedules are flawed*), but this paper turns them into an advantage—residual information in the noise becomes an exploitable generative cue. This "flaw-to-feature" perspective is remarkably ingenious.
- **Low-level visual attribute control as a distinctive contribution**: This is among the few works to systematically leverage initial noise for controlling low-level attributes such as color, texture, and sharpness, establishing a complementary framework in which "text controls high-level semantics, noise controls low-level attributes."
- **Model-agnostic and composable design**: As a "base-layer" enhancement method that can be seamlessly added to any T2I pipeline, this design philosophy is highly valuable for practical deployment.

## Limitations & Future Work

- NoiseQuery is bounded by the generative capacity of the underlying model and cannot surpass its ceiling.
- The method has limited effectiveness for fine structural control (e.g., Canny edges); noise is better suited for "soft" control.
- Offline construction of the noise library requires generating a large number of images (e.g., 100K), entailing a non-trivial initial cost.
- Strategies for updating and maintaining the noise library as new models emerge are not explored.
- While cross-model consistency is claimed, the paper does not provide quantitative metrics to measure the degree of consistency in depth.

## Related Work & Insights

- **vs. ReNO**: ReNO iteratively optimizes noise via gradient back-propagation, with very high computational cost (×300+) and applicability limited to single-step models; NoiseQuery achieves retrieval-based optimization with no gradient computation, applicable to all models.
- **vs. ControlNet**: ControlNet controls spatial layout through additional structural signals (edge maps, depth maps); NoiseQuery controls different aspects of generation through initial noise. The two approaches are complementary (composability is validated experimentally).
- **vs. Diffusion-DPO**: DPO requires retraining model parameters; NoiseQuery is fully training-free and can be directly stacked on top of DPO-enhanced models for additional gains.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The discovery of cross-model noise consistency and the "noise as assistant" perspective are highly inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Full matrix experiments across 6 models and 4 enhancement methods, with validation on both high- and low-level attribute control.
- Writing Quality: ⭐⭐⭐⭐⭐ — The logical chain is clear, flowing seamlessly from theoretical analysis to method design to experimental validation.
- Value: ⭐⭐⭐⭐ — Strong practical utility, though the magnitude of improvement is relatively moderate (marginal gains on strong baselines).

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] FICGen: Frequency-Inspired Contextual Disentanglement for Layout-driven Degraded Image Generation](ficgen_frequency-inspired_contextual_disentanglement_for_layout-driven_degraded_.md)
- [\[ICCV 2025\] Anchor Token Matching: Implicit Structure Locking for Training-free AR Image Editing](anchor_token_matching_implicit_structure_locking_for_training-free_ar_image_edit.md)
- [\[ICCV 2025\] LIFT: Latent Implicit Functions for Task- and Data-Agnostic Encoding](lift_latent_implicit_functions_for_task-_and_data-agnostic_encoding.md)
- [\[ICCV 2025\] AutoPrompt: Automated Red-Teaming of Text-to-Image Models via LLM-Driven Adversarial Prompts](autoprompt_automated_red-teaming_of_text-to-image_models_via_llm-driven_adversar.md)
- [\[ICCV 2025\] FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems](flowdps_flow-driven_posterior_sampling_for_inverse_problems.md)

<!-- RELATED:END -->

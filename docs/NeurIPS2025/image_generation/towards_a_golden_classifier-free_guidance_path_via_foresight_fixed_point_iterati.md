---
title: >-
  [Paper Note] Towards a Golden Classifier-Free Guidance Path via Foresight Fixed Point Iterations
description: >-
  [NeurIPS 2025][Image Generation][Classifier-Free Guidance] This paper unifies conditional guidance under a fixed-point iteration framework, showing that CFG and its variants are all special cases of single-step iterations over short intervals. It theoretically proves their suboptimality and proposes Foresight Guidance (FSG)—performing multi-step iterations over longer intervals in early diffusion stages to achieve better alignment quality with less computation.
tags:
  - NeurIPS 2025
  - Image Generation
  - Classifier-Free Guidance
  - Fixed Point Iteration
  - Conditional Guidance
  - Golden Path
  - Inference-Time Optimization
date: 2026-05-08
content_hash: ce95a6d458bbc973
---

# Towards a Golden Classifier-Free Guidance Path via Foresight Fixed Point Iterations

**Conference**: NeurIPS 2025
**arXiv**: [2510.21512](https://arxiv.org/abs/2510.21512)
**Code**: [GitHub](https://github.com/Ka1b0/Foresight-Guidance)
**Area**: Diffusion Models / Image Generation
**Keywords**: Classifier-Free Guidance, Fixed Point Iteration, Conditional Guidance, Golden Path, Inference-Time Optimization

## TL;DR

This paper unifies conditional guidance under a fixed-point iteration framework, showing that CFG and its variants are all special cases of single-step iterations over short intervals. It theoretically proves their suboptimality and proposes Foresight Guidance (FSG)—performing multi-step iterations over longer intervals in early diffusion stages to achieve better alignment quality with less computation.

## Background & Motivation

CFG (Classifier-Free Guidance) is a core component of text-to-image diffusion models, enhancing prompt alignment by amplifying the difference between conditional and unconditional outputs. However, CFG entails an inherent trade-off: excessively strong guidance degrades image quality and diversity.

**Fragmentation of existing improvements**:
- **CFG++**: Corrects manifold deviation from a posterior sampling perspective
- **Z-sampling**: Improves semantic alignment via reflective sampling
- **Resampling**: Similar to reflective sampling but uses a different forward process

These methods stem from **different theoretical interpretations**, each forming a closed framework whose components cannot be independently modified or cross-adopted. **The design space is locked by their respective theoretical assumptions.**

**Key observation (Golden Path)**: When the unconditional and conditional generation results from a latent $\hat{x}_t$ coincide, i.e., $f_{t \to 0}^u(\hat{x}_t) = f_{t \to 0}^c(\hat{x}_t)$, both image quality and alignment are improved. Such a trajectory is termed the "golden path." The intuition is that if the model does not need to make sharp turns between conditional and unconditional directions, it can achieve both quality and alignment simultaneously.

**Core Problem**: How to systematically explore the design space of conditional guidance?

## Method

### Overall Architecture

Each denoising step $x_t \to x_{t-1}$ is decomposed into two decoupled stages:
1. **Calibration**: The latent $x_t$ is calibrated to $\hat{x}_t$ via fixed-point iteration so that it approaches the golden path.
2. **Denoising**: Standard sampling is performed using the unconditional noise prediction $\epsilon^u(\hat{x}_t)$.

The fixed-point equation is $\hat{x}_t = F(\hat{x}_t)$, where the fixed point of $F$ satisfies consistency between conditional and unconditional generation.

### Key Designs

1. **Unified Fixed-Point Framework — Unifying the CFG Family**:

   All existing methods can be expressed as special cases of fixed-point iteration:
   - **CFG**: Linear operator $F(x_t) = x_t - w\xi_t \Delta\epsilon(x_t)$, interval $[t-1, t]$, single step
   - **CFG++**: Linear operator $F(x_t) = x_t - \lambda\tilde{\xi}_t \Delta\epsilon(x_t)$, interval $[t-1, t]$, with a more stable guidance schedule
   - **Z-sampling**: Forward-backward operator, interval $[t-1, t+1]$, requires DDIM inversion
   - **Resampling**: Forward-backward operator, interval $[t-1, t+1]$, replaces inversion with noise addition

   **Four design dimensions are identified**:
   - Consistency interval (short vs. long)
   - Fixed-point operator type (linear vs. forward-backward)
   - Guidance strength / schedule
   - Number of iterations $K$

2. **Suboptimality of Short-Interval Single-Step Iterations (Theorem 1)**:

   Given a total iteration budget $N$ and timesteps $T$, uniformly divided into $M$ subproblems each with $N/M$ iterations, the upper bound is:
   $$\mathcal{L} \leq B^2 \left(C r^{\frac{2N}{M}} + \frac{2L^2}{M^2}\right)$$

   The optimal $M^*$ is generally **not equal to** $T$, implying that performing fixed-point iterations at every timestep is unnecessary. Key insights:
   - The smoother the noise predictor (smaller $L$), the smaller $M^*$, favoring fewer but longer subproblems
   - With sufficient computation ($N \to \infty$), $M \to T$, recovering the step-wise strategy

3. **Foresight Guidance (FSG)**:

   Core idea: **Apply longer intervals with more iterations in early stages and reduce them in later stages.**

   Parameterized as $\mathcal{S} = \{(t_i, K_i, \Delta t_i)\}_{i=1}^M$:
   - $t_i$: timestep at which fixed-point iteration is applied
   - $K_i$: number of iterations at that timestep
   - $\Delta t_i$: consistency interval length

   Design principles:
   - Iteration allocation ratio across early/middle/late stages is approximately 3:2:1
   - Uses a forward-backward operator: conditional denoising $f_{t \to t-\Delta t}^\gamma$ + unconditional inversion $f_{t-\Delta t \to t}^u$
   - Non-foresight steps use CFG++ to maintain stable guidance
   - Single-step DDIM solver reduces per-iteration cost

### Loss & Training

FSG is a purely inference-time method requiring no training. Key design choices:
- Foresight interval is set in $[0.02T, 0.125T]$
- Larger intervals and more iterations are allocated in early stages
- Compatible with preference alignment models (SPO) and noise optimization (NPNet) in combination

## Key Experimental Results

### Main Results — SDXL, DrawBench & Pick-a-Pic

| Method | NFE | IR↑ (DrawBench) | HPSv2↑ | IR↑ (Pick-a-Pic) | HPSv2↑ |
|--------|-----|-----------------|--------|------------------|--------|
| CFG | 50 | 59.02 | 28.73 | 82.14 | 28.46 |
| CFG++ | 50 | 65.21 | 28.98 | 89.75 | 28.72 |
| Z-Sampling | 50 | 72.75 | 29.08 | 96.77 | 28.68 |
| **FSG** | **50** | **82.81** | **29.42** | **98.59** | **28.89** |
| CFG×3 | 150 | 83.56 | 29.51 | 102.13 | 29.04 |
| CFG++×3 | 150 | 82.58 | 29.45 | 103.32 | 29.05 |
| **FSG** | **150** | **88.18** | **29.44** | **104.86** | **29.04** |

### Geneval (Fine-Grained Instruction Following, SDXL)

| Method | Overall↑ | Single Obj. | Two Obj. | Counting | Color | Position | Color Attr. |
|--------|----------|-------------|----------|----------|-------|----------|-------------|
| CFG | 48.39% | 97.50% | 61.62% | 22.50% | 78.72% | 14% | 16% |
| CFG×3 | 55.94% | 98.75% | 75.76% | 40% | 85.11% | 8% | 28% |
| **FSG** | **57.95%** | **100%** | 79.80% | 43.75% | **86.17%** | **12%** | **28%** |

### ImageNet 256×256 (DiT, Class-Conditional Generation)

| Method | NFE=25 FID↓ | Vendi↑ | NFE=50 FID↓ | Vendi↑ |
|--------|-------------|--------|-------------|--------|
| CFG×2 | 17.81 | 3.44 | 14.69 | 3.79 |
| CFG++×2 | 13.27 | 3.91 | 8.85 | 4.43 |
| Z-sampling | 19.89 | 3.40 | 8.62 | 4.64 |
| **FSG** | **10.56** | **4.73** | **7.91** | **5.79** |

### Ablation Study

| Design Choice | ΔIR | ΔHPSv2 |
|---------------|-----|--------|
| Interval halved | −8.20 | −0.04 |
| Interval doubled | −2.40 | −0.12 |
| Iterations halved | −6.16 | −0.21 |
| Iterations doubled | −2.41 | −0.50 |
| Early foresight only | −4.82 | −0.19 |

### Synergy with Orthogonal Methods

| Method | IR↑ | HPSv2↑ |
|--------|-----|--------|
| CFG | 82.14 | 28.46 |
| FSG | 98.59 | 28.89 |
| SPO (preference fine-tuning) | 111.86 | 29.08 |
| SPO + FSG (100) | **117.93** | **29.20** |

### Key Findings

- **Existing methods directly benefit from increased iterations**: CFG×3 improves IR by 24.5 over CFG on DrawBench, validating the practical utility of the fixed-point framework.
- **FSG at NFE=50 surpasses other methods at NFE=150**: Computational efficiency substantially outperforms naive iteration scaling.
- **Fixed-point iterations do not harm diversity**: On ImageNet, FID decreases while the Vendi score (diversity) simultaneously increases.
- **Weaker models benefit more**: FSG yields an IR gain of +8.19 on SD2.1, far exceeding the +4.16 gain on Hunyuan-DiT.
- **Synergy with SPO/NPNet**: As an inference-time method, FSG complements training-time preference alignment, achieving the highest scores in combination.

## Highlights & Insights

- **Contribution of the unified perspective**: By subsuming CFG, CFG++, Z-sampling, and Resampling under a single fixed-point framework, the paper clearly exposes the design choice differences across methods.
- **Strong alignment between theory and practice**: Theorem 1 formally proves the suboptimality of short-interval single-step iterations, and FSG's design is directly driven by this theoretical guidance.
- **A new dimension for inference-time scaling**: The number of fixed-point iterations serves as a test-time compute knob, representing a different scaling direction from simply increasing the number of denoising steps.
- **Extensive experimental coverage**: 4 datasets × 3 models (SDXL / SD2.1 / Hunyuan-DiT) × 2 samplers (DDIM / DDPM).

## Limitations & Future Work

- The allocation strategy for foresight intervals and iteration counts (3:2:1 ratio) is empirically determined and lacks an adaptive mechanism.
- Single-step DDIM solving over long intervals introduces truncation errors, potentially limiting the effectiveness of very long intervals.
- Different content types may require distinct foresight strategies.
- Theoretical guarantees rely on mild assumptions (smoothness, contractiveness); behavior under violations is not sufficiently discussed.

## Related Work & Insights

- The relationships with CFG++, Z-sampling, and related methods are clearly characterized—they represent different design choices within the same framework.
- This work provides a theoretical foundation for test-time scaling of conditional guidance.
- The foresight concept is generalizable to scenarios requiring long-range consistency, such as video generation.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The unified fixed-point perspective is highly elegant, and FSG's design is naturally motivated by theory.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Coverage across datasets, models, samplers, and combination methods is exceptionally comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The derivation progresses clearly from the unified framework to the concrete algorithm, with intuitive figures and tables.
- **Value**: ⭐⭐⭐⭐⭐ — Provides deep understanding and a systematic improvement for CFG as a core component, with strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Studying Classifier(-Free) Guidance From A Classifier-Centric Perspective](../../AAAI2026/image_generation/studying_classifier-free_guidance_from_a_classifier-centric_perspective.md)
- [\[CVPR 2026\] CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance](../../CVPR2026/image_generation/cfg-ctrl_control-based_classifier-free_diffusion_guidance.md)
- [\[AAAI 2026\] DICE: Distilling Classifier-Free Guidance into Text Embeddings](../../AAAI2026/image_generation/dice_distilling_classifier-free_guidance_into_text_embedding.md)
- [\[ICCV 2025\] TeEFusion: Blending Text Embeddings to Distill Classifier-Free Guidance](../../ICCV2025/image_generation/teefusion_blending_text_embeddings_to_distill_classifier-free_guidance.md)
- [\[NeurIPS 2025\] Entropy Rectifying Guidance for Diffusion and Flow Models](entropy_rectifying_guidance_for_diffusion_and_flow_models.md)

</div>

<!-- RELATED:END -->

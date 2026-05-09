---
title: >-
  [Paper Note] EditInfinity: Image Editing with Binary-Quantized Generative Models
description: >-
  [NeurIPS 2025][Image Generation][Binary-quantized generative models] This paper proposes EditInfinity, the first work to apply the classical "image inversion–image editing" paradigm to binary-quantized autoregressive generative models (Infinity). By leveraging the inherent property of quantized representations that enables exact intermediate supervision, EditInfinity achieves high-fidelity image inversion. Combined with a piecewise linear smoothing kernel for seamless editing, it comprehensively surpasses diffusion model baselines on PIE-Bench.
tags:
  - NeurIPS 2025
  - Image Generation
  - Binary-quantized generative models
  - Infinity
  - image inversion
  - piecewise linear smoothing
  - multi-scale autoregressive editing
date: 2026-05-08
content_hash: 9d72a6fb2f407b26
---

# EditInfinity: Image Editing with Binary-Quantized Generative Models

**Conference**: NeurIPS 2025
**arXiv**: [2510.20217](https://arxiv.org/abs/2510.20217)
**Code**: [Available](https://github.com/yx-chen-ust/EditInfinity)
**Area**: Image Generation / Image Editing / Autoregressive Models
**Keywords**: Binary-quantized generative models, Infinity, image inversion, piecewise linear smoothing, multi-scale autoregressive editing

## TL;DR

This paper proposes EditInfinity, the first work to apply the classical "image inversion–image editing" paradigm to binary-quantized autoregressive generative models (Infinity). By leveraging the inherent property of quantized representations that enables exact intermediate supervision, EditInfinity achieves high-fidelity image inversion. Combined with a piecewise linear smoothing kernel for seamless editing, it comprehensively surpasses diffusion model baselines on PIE-Bench.

## Background & Motivation

The classical paradigm for text-driven image editing consists of two steps: (1) image inversion—reversing the generation trajectory, and (2) editing along the trajectory guided by the target text. The core bottleneck lies in:

- **Imprecise image inversion in diffusion models**: Exact intermediate representations along the source image's generation trajectory are unavailable and can only be approximated.
- Approximation errors propagate to the editing stage, causing a trade-off between background preservation and semantic alignment.

Key insight: **Binary-quantized generative models** (e.g., Infinity) model images in a discrete latent space, and their intrinsic property is that **exact multi-scale quantized representations of any image can be obtained directly**. This enables the use of exact intermediate results as supervision signals for image inversion optimization, fundamentally resolving the approximation error problem inherent in diffusion models.

## Method

### Overall Architecture

EditInfinity is built upon Infinity-2B (a binary-quantized T2I model) and operates in two stages:
1. **Image Inversion**: Optimize learnable text embeddings + LoRA fine-tuning, supervised by exact quantized tokens.
2. **Image Editing**: Multi-scale autoregressive token replacement + piecewise linear smoothing kernel for seamless transitions.

#### Overview of the Infinity Pretrained Model

- Image → Encoder → Feature $F$ → Multi-scale residual quantization $\{R_k\}_{k=1}^K$
- At each scale $k$: residual = $F - F_{k-1}$, downsampled to $(h_k, w_k)$, BSQ binary quantization applied
- Autoregressive modeling: $p(R_{1:K} \mid \Psi(t)) = \prod_k p(R_k \mid R_{<k}, \Psi(t))$
- Infinite-Vocabulary Classifier decomposes predictions into $d$ independent binary classifiers

### Key Designs

#### Image Inversion with Exact Supervision

Core advantage: quantized tokens $R_{1\ldots K}^{\text{sou}}$ serve as exact supervision signals.

**Textual Prompting Rectification:**
- The source text prompt $t_{\text{sou}}$ often fails to precisely match the source image.
- 20 learnable prompt tokens $t_l$ and an instruction prompt $t_{\text{ins}}$ are appended.
- All Infinity parameters are frozen; only $t_l$ is optimized.
- Cross-entropy loss: $\mathcal{L}_{\text{inv}} = -\frac{1}{K} \sum_k \left( R_k^{\text{sou}} \cdot \log p(R_k^{\text{inv}} \mid R_{<k}^{\text{sou}}, \Psi(t)) \right)$
- Exact quantized tokens serve as ground truth rather than approximations.

**Image Style Preservation:**
- Learnable prompt optimization improves semantic alignment but may alter global style.
- LoRA is applied to FFN layers, exploiting the property of low-rank biases to induce smooth global modifications.
- LoRA training is stopped after only 20 steps to prevent overfitting that would cause the model to ignore editing intent.
- The trained $\Delta W$ is retained during the editing stage to maintain source image style consistency.

#### Holistic Smoothing Strategy

**Piecewise Linear Smoothing Kernel $G$:**
- Weights are computed based on Manhattan distance $d^{i,j} = \min_{(x,y) \in M} (|i-x| + |j-y|)$
- Three-segment design:
    - $d \leq \tau_1$: $G = 0$ (edited region, fully uses target content)
    - $\tau_1 < d < \tau_2$: $G$ linearly interpolates (smooth transition zone)
    - $d \geq \tau_2$: $G = 1$ (unedited region, fully preserves source content)
- Default $\tau_1 = 1$, $\tau_2 = 4$; effectively suppresses boundary stitching artifacts.

**Multi-scale Autoregressive Editing:**
- Quantize the source image to obtain $R_{1:K}^{\text{sou}}$
- At each scale $k$:
    - Infinity generates $R_k^{\text{tar}}$ conditioned on $[\Psi(t_{\text{tar}}, t_{\text{ins}}), t_l]$
    - Upsample $R_k^{\text{tar}}$ and $R_k^{\text{sou}}$ to maximum resolution
    - Blend under guidance of $G$: $E_k^{\text{tar}} = R_k^{\text{tar}} \odot (1-G) + R_k^{\text{sou}} \odot G$
    - If $k < K$, downsample $E_k^{\text{tar}}$ to $\hat{R}_k^{\text{tar}}$ for use at the next scale
- Blended semantics and structure propagate across scales
- Final $E_{1:K}^{\text{tar}}$ is decoded into the edited image

### Loss & Training

- Inversion stage: cross-entropy loss optimizes the learnable prompt (exact quantized tokens as GT)
- LoRA applied to FFN layers only; frozen after 20 training steps
- Editing stage requires no training; pure inference
- Hardware: 2× NVIDIA L20 (inversion), 1× NVIDIA L20 (editing)

## Key Experimental Results

### Main Results

**Full Quantitative Comparison on PIE-Bench**

| Method | Backbone | PSNR↑ | LPIPS×10³↓ | SSIM×10²↑ | Whole CLIP↑ | Edited CLIP↑ | IR×10↑ |
|--------|----------|-------|-----------|-----------|------------|-------------|--------|
| NTI (CVPR'23) | U-Net | 27.03 | 60.67 | 84.11 | 24.75 | 21.86 | 2.77 |
| PnP-Inv (ICLR'24) | U-Net | 22.46 | 106.06 | 79.68 | 25.41 | 22.62 | 4.17 |
| RF-Edit (ICML'25) | DiT | 23.22 | 131.18 | 81.44 | 25.22 | 22.40 | 5.18 |
| Gemini 2.0 | Commercial | 23.22 | 105.17 | 81.10 | 25.28 | 22.28 | 5.30 |
| **EditInfinity** | **AR** | **27.95** | **33.08** | **92.12** | **26.41** | **23.47** | **5.88** |

EditInfinity achieves comprehensive superiority in both **background preservation** and **text alignment**. LPIPS is reduced from 60.67 to 33.08 (best U-Net comparison), PSNR reaches 27.95 (highest), and IR reaches 5.88 (highest editing success rate).

**Backbone Fairness Verification (GenEval Benchmark)**

| Backbone | Overall |
|----------|---------|
| SD v1.4 | 0.42 |
| FLUX.1-dev | 0.66 |
| Infinity-2B | 0.66 |

Infinity achieves generation capability on par with FLUX (0.66 vs. 0.66), yet EditInfinity substantially outperforms FLUX-based methods, demonstrating that the gains stem from the proposed method rather than a stronger backbone.

### Ablation Study

**Smoothing Kernel Ablation (PIE-Bench random class)**

| $G$ Configuration | PSNR↑ | LPIPS×10³↓ | IR×10↑ |
|-------------------|-------|-----------|--------|
| No $G$ | 31.12 | 24.47 | 2.85 |
| Gaussian kernel | 28.15 | 32.91 | 4.61 |
| **Linear kernel** | **28.50** | **31.58** | **5.39** |

Without $G$, IR is lowest (poor editing effect). The linear kernel achieves the best balance between editing quality and background preservation.

**Learnable Prompt + LoRA Ablation:**
- Removing both: severe structural inconsistency
- Learnable prompt only: improved text alignment but global style drift
- +LoRA (20 steps): style consistency restored
- +LoRA (excessive steps): overfitting, editing intent ignored

### Key Findings

- **Exact intermediate supervision** is the core of EditInfinity's success—an intrinsic advantage of quantized models.
- Editing speed is extremely fast (3.64 s/edit); inversion cost is front-loaded (107 s) but amortized over multiple edits.
- The piecewise linear kernel outperforms the Gaussian kernel with smoother transitions.
- LoRA training steps must be strictly controlled (20 steps); excessive steps lead to overfitting.
- EditInfinity achieves the highest preference rate of 43.2% in user studies.

**Runtime Comparison**

| Method | Inversion | Per Edit |
|--------|-----------|---------|
| NTI | 95.5 s | 10.3 s |
| RF-Edit | 55.5 s | 54.1 s |
| **EditInfinity** | 107.1 s | **3.6 s** |

Inversion overhead is moderately high, but single editing takes only 3.6 seconds—**approximately 7× faster than average**—making it highly suitable for iterative editing workflows.

## Highlights & Insights

1. **Opens a new track**: The first work to apply autoregressive quantized models to image editing, leveraging the intrinsic advantage of exact quantized representations.
2. **Exact supervision resolves the core bottleneck**: Using quantized tokens as GT for inversion training is fundamentally superior to the approximations employed in diffusion models.
3. **Extremely fast iterative editing**: 3.6 s per edit is highly practical; the front-loaded inversion cost can be amortized across multiple editing operations.
4. **Elegant smoothing kernel design**: Manhattan distance + linear interpolation, requiring no learnable parameters, outperforms the Gaussian kernel.
5. **Rigorous fairness evaluation**: GenEval verification confirms comparable backbone capability, ruling out the possibility that gains originate from a stronger base model.

## Limitations & Future Work

- The 107-second inversion stage is relatively slow (despite fast per-edit speed); faster text optimization strategies warrant exploration.
- Relies on user-provided editing masks (standard setting but limits automation).
- LoRA training steps (20 steps) require manual tuning and may vary across images.
- Currently validated only on Infinity-2B; extensibility to larger models or other quantized architectures remains to be explored.
- Generation diversity of quantized models may be lower than that of diffusion models.

## Related Work & Insights

- Infinity (Han et al., 2024): BSQ binary quantization + multi-scale residual prediction, establishing a new benchmark for AR T2I.
- I2SB inspired the "exact intermediate supervision" concept—though in EditInfinity, this exactness is an inherent property of the quantized model.
- Provides a diffusion-vs.-autoregressive contrast relative to DiT-based methods such as RF-Edit and StableFlow.
- LoRA fine-tuning as a style preservation mechanism; low-rank biases naturally tend toward globally smooth modifications.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ (First application of quantized models to editing; exact supervision concept is original)
- **Technical Depth**: ⭐⭐⭐⭐ (Complete two-stage inversion+editing design with well-motivated components)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (9 editing task categories, 8+ baselines, user study, runtime analysis, ablations)
- **Practicality**: ⭐⭐⭐⭐⭐ (3.6 s per edit is highly practical; code is open-source)
- **Writing Quality**: ⭐⭐⭐⭐⭐ (Clear algorithmic pseudocode, rich illustrations, comprehensive comparisons)

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] BitMark: Watermarking Bitwise Autoregressive Image Generative Models](bitmark_watermarking_bitwise_autoregressive_image_generative_models.md)
- [\[ICLR 2026\] QVGen: Pushing the Limit of Quantized Video Generative Models](../../ICLR2026/image_generation/qvgen_pushing_the_limit_of_quantized_video_generative_models.md)
- [\[NeurIPS 2025\] ThermalGen: Style-Disentangled Flow-Based Generative Models for RGB-to-Thermal Image Translation](thermalgen_style-disentangled_flow-based_generative_models_for_rgb-to-thermal_im.md)
- [\[NeurIPS 2025\] SplitFlow: Flow Decomposition for Inversion-Free Text-to-Image Editing](splitflow_flow_decomposition_for_inversion-free_text-to-image_editing.md)
- [\[NeurIPS 2025\] CAMILA: Context-Aware Masking for Image Editing with Language Alignment](camila_contextaware_masking_for_image_editing_with_language.md)

<!-- RELATED:END -->

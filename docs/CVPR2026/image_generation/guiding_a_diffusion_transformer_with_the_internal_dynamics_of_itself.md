---
title: >-
  [Paper Note] Guiding a Diffusion Transformer with the Internal Dynamics of Itself
description: >-
  [CVPR 2026][Image Generation][internal guidance] This paper proposes Internal Guidance (IG), which adds auxiliary supervision losses to intermediate layers of a Diffusion Transformer to produce weaker generative outputs, then extrapolates the discrepancy between intermediate-layer and final-layer outputs at sampling time to achieve an Autoguidance-like effect — requiring no additional sampling steps or external model training. On ImageNet 256×256, IG pushes LightningDiT-XL/1 to FID 1.34 (without CFG) and 1.19 (+CFG), achieving state-of-the-art results among contemporaneous methods.
tags:
  - CVPR 2026
  - Image Generation
  - internal guidance
  - intermediate-layer supervision
  - diffusion Transformer
  - sampling guidance
  - training acceleration
date: 2026-05-08
content_hash: e8656e9a3a094bd7
---

# Guiding a Diffusion Transformer with the Internal Dynamics of Itself

**Conference**: CVPR 2026
**arXiv**: [2512.24176](https://arxiv.org/abs/2512.24176)
**Code**: [https://github.com/xy-chou/Internal-Guidance](https://github.com/xy-chou/Internal-Guidance) (project page)
**Area**: Diffusion Models / Image Generation
**Keywords**: internal guidance, intermediate-layer supervision, diffusion Transformer, sampling guidance, training acceleration

## TL;DR
This paper proposes Internal Guidance (IG), which adds auxiliary supervision losses to intermediate layers of a Diffusion Transformer to produce weaker generative outputs, then extrapolates the discrepancy between intermediate-layer and final-layer outputs at sampling time to achieve an Autoguidance-like effect — requiring no additional sampling steps or external model training. On ImageNet 256×256, IG pushes LightningDiT-XL/1 to FID 1.34 (without CFG) and 1.19 (+CFG), achieving state-of-the-art results among contemporaneous methods.

## Background & Motivation

1. **State of the Field**: Classifier-Free Guidance (CFG) is the standard approach for improving generation quality in diffusion models by steering samples toward high-probability regions of the conditional distribution. However, excessively large CFG scales lead to oversimplification or distortion and reduce sample diversity. Methods such as Autoguidance address this by using a "degraded version of the model" as the reference, but require separately training a weaker model or incurring additional sampling steps.
2. **Limitations of Prior Work**: (1) CFG at high guidance scales over-emphasizes class conditioning, pushing samples toward "template images" and reducing diversity; (2) Autoguidance requires dedicated training of a weaker model, which is costly and inflexible; (3) Methods such as PAG/SEG require carefully designed degradation strategies and impose additional sampling overhead.
3. **Root Cause**: The desired effect of Autoguidance — improving quality while preserving diversity — cannot be achieved without either training a separate degraded model or increasing sampling steps.
4. **Paper Goals**: To achieve Autoguidance-level improvements in generation quality and diversity at virtually zero additional cost.
5. **Starting Point**: Intermediate-layer outputs of a deep network are inherently a "weaker version" of the final output, having been processed by only a subset of Transformer blocks. If intermediate layers are trained to perform denoising, they naturally provide a "weak-vs-strong" contrastive signal at sampling time.
6. **Core Idea**: Add auxiliary supervision to intermediate layers of a Diffusion Transformer to train a built-in weak model, then use the difference between intermediate-layer and final-layer outputs as a guidance signal during sampling.

## Method

### Overall Architecture
**Training phase**: An additional output head is appended after an intermediate layer (e.g., layer 4 or 8) of a standard diffusion Transformer (e.g., SiT, LightningDiT), applying an auxiliary denoising loss $\mathcal{L}_{\text{inter}}$ on the intermediate-layer output. This is combined with the final-layer loss $\mathcal{L}_{\text{final}}$ in a weighted sum. **Sampling phase**: At each denoising step, both the intermediate-layer output $D_i$ and the final-layer output $D_f$ are obtained, and guidance is realized by extrapolation: $D_w = D_i + w(D_f - D_i)$. When $w > 1$, this pushes the sample away from the lower-quality distribution of the intermediate layer (weak version) and toward the higher-quality distribution of the final layer (strong version). No additional forward passes are required, as the intermediate-layer output is a natural byproduct of the full forward pass.

### Key Designs

1. **Auxiliary Supervision Loss at Intermediate Layers**:

    - **Function**: Equips intermediate layers with denoising capability, serving as a built-in weak model.
    - **Mechanism**: An output head $D_i$ is defined after the $l$-th Transformer block and trained with the same denoising objective as the final layer: $\mathcal{L}_{\text{inter}} = \|D_i(\mathbf{x}_t, t) - \mathbf{x}_0\|^2$. The total loss is $\mathcal{L} = \mathcal{L}_{\text{final}} + \lambda \mathcal{L}_{\text{inter}}$, where $\lambda$ controls the auxiliary loss weight. Experiments show that $\lambda \leq 0.5$ yields stable results.
    - **Design Motivation**: (1) This is the most direct way to create a "weak version" — intermediate layers have processed only half the blocks and are naturally weaker than the full network; (2) The auxiliary supervision also alleviates vanishing gradients in deep networks, yielding accelerated convergence as a bonus — experiments show this effect is competitive with more complex self-supervised representation alignment methods (e.g., REPA, SRA).

2. **Internal Guidance Sampling Strategy**:

    - **Function**: Uses the discrepancy between intermediate-layer and final-layer outputs as a sampling guidance signal.
    - **Mechanism**: During sampling, the guided output is computed as $D_w(\mathbf{x}; \mathbf{c}) = D_i(\mathbf{x}; \mathbf{c}) + w(D_f(\mathbf{x}; \mathbf{c}) - D_i(\mathbf{x}; \mathbf{c}))$. For $w > 1$, this is equivalent to extrapolating along the "weak-to-strong" direction, moving away from the lower-quality distribution of the intermediate layer toward the higher-quality distribution of the final layer. No additional forward passes are needed.
    - **Design Motivation**: The approach is conceptually aligned with Autoguidance — using a weak version to guide a strong version, improving quality while maintaining diversity. Unlike Autoguidance, which requires separately training a weak model, IG uses the network's own intermediate layers as the weak model at zero additional cost.

3. **Complementarity of IG and CFG, and Guidance Interval**:

    - **Function**: Further improves generation quality and controls the temporal application schedule of guidance.
    - **Mechanism**: IG provides class-agnostic guidance (pushing samples toward the interior of the data manifold), while CFG provides class-conditional guidance (pushing samples toward the target class). Their combination works best with a moderate IG scale alongside CFG. Regarding the guidance interval, IG should be applied during high-to-medium noise levels ($\sigma \in (0.3, 1)$) and is unnecessary at low noise levels — naturally complementary to the optimal interval for CFG (medium-to-low noise).
    - **Design Motivation**: 2D toy experiments clearly illustrate the complementary mechanism: IG eliminates outliers at distribution tails (class-agnostic), while CFG suppresses inter-class confusion (class-conditional). The two methods improve generation quality along orthogonal dimensions.

### Loss & Training
- Training is based on SiT and LightningDiT using standard settings; LightningDiT adopts the Muon optimizer (replacing AdamW to address early-stage instability) and an EMA decay of 0.9995 (changed from 0.9999).
- Auxiliary supervision applied to early layers yields the best results (layer 4 for SiT-B/2; layer 8 for larger models); placing supervision in the latter half of the network degrades final-layer output.
- Training on ImageNet-1K 256×256 after VAE encoding.
- Sampling uses SDE Euler–Maruyama with 250 steps (SiT/DiT) or ODE Heun with 125 steps (LightningDiT).

## Key Experimental Results

### Main Results — ImageNet 256×256 (without CFG)

| Method | Training Epochs | FID↓ | IS↑ |
|--------|----------------|------|-----|
| SiT-XL/2 | 1400 | 8.61 | 131.7 |
| REPA | 800 | 5.90 | 157.8 |
| **SiT-XL/2 + IG** | **80** | **5.31** | **147.7** |
| **SiT-XL/2 + IG** | **800** | **1.75** | **228.6** |
| LightningDiT-XL/1 | 800 | 2.17 | 205.6 |
| **LightningDiT-XL/1 + IG** | **60** | **2.42** | **173.7** |
| **LightningDiT-XL/1 + IG** | **680** | **1.34** | **229.3** |

### SOTA Comparison with CFG

| Method | FID↓ | sFID↓ |
|--------|------|-------|
| REPA + CFG (800ep) | 1.42 | 4.70 |
| REPA-E + CFG (800ep) | 1.26 | 4.11 |
| SiT-XL/2 + IG + CFG (800ep) | 1.46 | 4.79 |
| **LightningDiT-XL/1 + IG + CFG (680ep)** | **1.19** | **4.11** |

### Ablation Study

| Ablation | FID↓ | IS↑ | Notes |
|----------|------|-----|-------|
| SiT-B/2 baseline | 33.02 | 43.71 | No auxiliary supervision |
| Aux. supervision (layer 2) | 30.45 | 47.97 | Effective at early layers |
| Aux. supervision (layer 4) | 30.60 | 47.70 | Best or near-best |
| Aux. supervision (layer 8) | 38.05 | 37.97 | Harmful in latter half |
| +IG (layer 4, $w$=1.5) | **19.02** | **65.06** | Large gain from guidance |
| +IG ($w$=1.9) | 17.38 | 69.12 | Best scale without interval |
| +IG ($w$=2.3) + interval $[0.3,1)$ | **16.19** | **72.95** | Best configuration |

### Key Findings
- **Remarkable training efficiency**: SiT-XL/2 + IG achieves FID 5.31 in only 80 epochs, surpassing the original SiT at 1400 epochs (FID 8.61) and REPA at 800 epochs (FID 5.90).
- **Auxiliary supervision layer placement is critical**: It must be within the first few layers (top ~1/3); placing it in the latter half (layers 8 or 10) is harmful.
- **Auxiliary supervision alone accelerates convergence**: Even without IG sampling guidance, adding only the auxiliary loss yields convergence comparable to complex self-supervised representation alignment methods.
- **IG and CFG guidance intervals are complementary**: IG should be applied at high-to-medium noise levels, CFG at medium-to-low noise levels — the two are naturally non-overlapping.
- **IG scales better with model size**: The relative improvement from IG increases as model size grows from B → L → XL.

## Highlights & Insights
- **The "built-in weak model" insight is remarkably elegant**: Intermediate-layer outputs of a deep network are naturally a weakened version of the final output. This observation reduces Autoguidance from "training a separate degraded model" to "adding a single auxiliary loss" — a genuine simplification.
- **A two-birds-one-stone design**: The auxiliary supervision simultaneously provides intermediate outputs for sampling guidance and alleviates vanishing gradients to accelerate convergence, solving two problems with one simple mechanism.
- **A new finding on guidance intervals**: IG is effective at high-to-medium noise levels and unnecessary at low noise levels — the opposite of CFG's optimal interval. This finding offers useful guidance for combining multiple guidance strategies in future work.
- **Extension from guidance to training acceleration**: Section 6 demonstrates that incorporating the IG principle into the training loss as $\mathbf{x}_0 + \omega \cdot \text{sg}(D_f - D_i)$ directly accelerates convergence, further revealing the underlying mechanism of the method.

## Limitations & Future Work
- The placement of the auxiliary supervision layer requires separate tuning for each model architecture (layer 4 for SiT-B, layer 8 for larger models).
- Three hyperparameters — the IG scale $w$, the guidance interval $[\sigma_{\text{low}}, \sigma_{\text{high}}]$, and the auxiliary loss weight $\lambda$ — require joint tuning.
- Validation is limited to class-conditional ImageNet; the method has not been tested on text-conditional generation (e.g., SD, SDXL).
- The intermediate output head introduces a small number of additional parameters (one extra output layer), which, while minimal, may warrant attention in large-scale distributed training.

## Related Work & Insights
- **vs. Autoguidance**: Autoguidance requires separately training a degraded model; IG uses intermediate layers as a natural "degraded version" at zero additional training cost. The two methods exhibit similar behavior in 2D distribution experiments.
- **vs. CFG**: CFG provides class-conditional directional guidance (pushing toward the target class); IG provides class-agnostic manifold guidance (pushing toward high-probability regions of the data distribution). The two are complementary and their combination achieves state-of-the-art results.
- **vs. PAG/SEG/SAG**: These methods construct a weak version at inference time by perturbing attention maps or inputs; IG embeds the weak version during training, requiring no modifications at inference. The approach is cleaner and more efficient.
- **vs. REPA/SRA**: Self-supervised representation alignment regularizes intermediate layers using complex pretrained models; IG's auxiliary supervision is simpler yet achieves comparable convergence acceleration.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The "intermediate layers as weak model" insight is elegant and profound; the two-birds-one-stone design is highly commendable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Multiple model scales, detailed ablations, 2D visualizations, training acceleration extensions, and SOTA results — extremely comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure; the 2D toy experiment visualizations are particularly effective for conveying the core intuition.
- **Value**: ⭐⭐⭐⭐⭐ FID 1.19 SOTA + training acceleration + plug-and-play applicability — strong contributions in both theory and practice.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Guiding a Diffusion Model by Swapping Its Tokens](guiding_a_diffusion_model_by_swapping_its_tokens.md)
- [\[CVPR 2026\] Guiding Diffusion Models with Semantically Degraded Conditions](guiding_diffusion_models_with_semantically_degraded_conditions.md)
- [\[CVPR 2026\] Guiding Diffusion Models with Semantically Degraded Conditions (CDG)](cdg_condition_degradation_guidance_diffusion.md)
- [\[CVPR 2026\] DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression](dit-ic_aligned_diffusion_transformer_for_efficient_image_compression.md)
- [\[CVPR 2026\] MPDiT: Multi-Patch Global-to-Local Transformer Architecture for Efficient Flow Matching](mpdit_multi-patch_global-to-local_transformer_architecture_for_efficient_flow_ma.md)

<!-- RELATED:END -->

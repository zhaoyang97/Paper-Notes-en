---
title: >-
  [Paper Note] FreeMorph: Tuning-Free Generalized Image Morphing with Diffusion Model
description: >-
  [Image Generation] FreeMorph proposes the first tuning-free generalized image morphing method. Through two key designs—guidance-aware spherical interpolation and step-oriented change trend—it generates smooth transition…
tags:
  - "Image Generation"
date: 2026-05-08
content_hash: 85ac82ac426919c8
---

# FreeMorph: Tuning-Free Generalized Image Morphing with Diffusion Model

## Meta Information
- **Conference**: ICCV 2025
- **arXiv**: [2507.01953](https://arxiv.org/abs/2507.01953)
- **Code**: [GitHub](https://yukangcao.github.io/FreeMorph/)
- **Area**: Image Generation / Image Transformation
- **Keywords**: Image Morphing, Diffusion Model, Tuning-Free, Self-Attention, Spherical Interpolation

## TL;DR

FreeMorph proposes the first tuning-free generalized image morphing method. Through two key designs—guidance-aware spherical interpolation and step-oriented change trend—it generates smooth transition sequences between image pairs of arbitrary semantics and layouts within 30 seconds, achieving a speed improvement of 10–50× over existing methods.

## Background & Motivation

Image morphing aims to generate a sequence of intermediate images that smoothly transition between two input images. Existing diffusion-based methods suffer from critical limitations:

**Require fine-tuning**: DiffMorpher requires approximately 5 minutes per sample for LoRA training; IMPUS requires approximately 30 minutes per sample.

**Semantic/layout constraints**: Existing methods struggle to handle image pairs with large semantic or layout differences.

**Fine-tuning restricts generalization**: The constraints imposed by LoRA modules limit the generalization capability of pre-trained models.

A naive approach of applying spherical interpolation in latent space followed by DDIM faces two challenges:
- **Non-directional transitions and identity loss**: The nonlinearity of multi-step denoising leads to inconsistent transitions.
- **Inconsistent transitions**: Diffusion models lack a mechanism to capture a gradual "change trend."

## Method

### Overall Architecture

FreeMorph builds upon pre-trained Stable Diffusion 2.1 and consists of three core components:
1. **Guidance-aware spherical interpolation**: Provides explicit guidance from the input images.
2. **Step-oriented change trend**: Enables controlled and consistent transitions.
3. **Improved forward diffusion and reverse denoising pipeline**: Integrates the above components.

Given two input images $\mathcal{I}_\text{left}$ and $\mathcal{I}_\text{right}$, the method generates $J=5$ intermediate transition images.

### Key Design 1: Guidance-Aware Spherical Interpolation

**Spherical Feature Aggregation**

A core observation is that replacing the K/V features in self-attention substantially enhances the smoothness of transitions. Accordingly, features from the two input images are blended as explicit guidance during denoising:

$$\text{ATT}(Q_{t-j}, K_{t-j}, V_{t-j}) := \frac{1}{2}(\text{ATT}(Q_{t-j}, K_{t\text{-left}}, V_{t\text{-left}}) + \text{ATT}(Q_{t-j}, K_{t\text{-right}}, V_{t\text{-right}}))$$

**Prior-driven Self-attention**

Using spherical feature aggregation alone causes the transition sequence to exhibit insufficient variation. The solution is to apply different attention mechanisms at different stages:
- **Reverse denoising stage**: Spherical feature aggregation (Eq. 5) is used to preserve identity.
- **Forward diffusion stage**: Aggregation of K/V features from all intermediate images (Eq. 6) is used to ensure smooth transitions.

$$\text{ATT}(Q_{t-j}, K_{t-j}, V_{t-j}) := \frac{1}{J}\sum_{k=1}^{J}\text{ATT}(Q_{t-j}, K_{t-k}, V_{t-k})$$

### Key Design 2: Step-Oriented Change Trend

By gradually shifting the influence weights of the two input images within self-attention, the method achieves a consistent transition from the left image to the right image:

$$\text{ATT} := (1 - \alpha_j) \cdot \text{ATT}(Q, K_\text{left}, V_\text{left}) + \alpha_j \cdot \text{ATT}(Q, K_\text{right}, V_\text{right})$$

where $\alpha_j = j/(J+2-1)$, and $J+2$ includes $J$ generated images and 2 input images.

### High-Frequency Gaussian Noise Injection

After forward diffusion, Gaussian noise is injected into the high-frequency domain of the latent vectors via FFT and a high-pass filter:

$$\mathbf{z} := \begin{cases} \text{IFFT}(\text{FFT}(\mathbf{z})), & \text{if } \mathbf{m} = 1 \\ \text{IFFT}(\text{FFT}(\mathbf{g})), & \text{if } \mathbf{m} = 0 \end{cases}$$

This increases generative flexibility and prevents over-constrained outputs.

### Complete Pipeline

**Forward Diffusion** (total $T=50$ steps):
1. $t < \lambda_1 T$ (0.3): Standard self-attention
2. $\lambda_1 T \leq t < \lambda_2 T$ (0.3–0.6): Prior-driven self-attention (Eq. 6)
3. $t \geq \lambda_2 T$ (0.6+): Step-oriented change trend (Eq. 7)

**Reverse Denoising** (total $T=50$ steps):
1. $t < \lambda_3 T$ (0.2): Step-oriented change trend (Eq. 7)
2. $\lambda_3 T \leq t < \lambda_4 T$ (0.2–0.6): Spherical feature aggregation (Eq. 5)
3. $t \geq \lambda_4 T$ (0.6+): Standard self-attention (for high-fidelity output)

## Key Experimental Results

### Main Results: Quantitative Comparison

| Method | MorphBench LPIPS↓ | MorphBench FID↓ | MorphBench PPL↓ | Morph4Data LPIPS↓ | Morph4Data FID↓ | Overall LPIPS↓ |
|------|-------------------|-----------------|-----------------|-------------------|-----------------|----------------|
| IMPUS | 130.52 | 152.43 | 3263.03 | 134.88 | 210.66 | 265.40 |
| DiffMorpher | 90.57 | 157.18 | 2264.20 | 98.56 | 292.54 | 189.13 |
| Slerp (naive) | 119.77 | 169.17 | 2994.35 | 103.74 | 245.22 | 223.52 |
| **FreeMorph** | **84.91** | **141.32** | **2122.80** | **80.30** | **201.09** | **162.99** |

**User Study** (30 volunteers): FreeMorph achieves a preference rate of **60.13%**, substantially outperforming IMPUS (17.16%), DiffMorpher (14.89%), and Slerp (7.82%).

### Ablation Study: Contribution of Each Component

| Method | Overall LPIPS↓ | Overall FID↓ | Overall PPL↓ |
|------|----------------|--------------|--------------|
| w/ only Eq. 6 | 298.13 | 355.24 | 6453.24 |
| w/ only Eq. 5 | 190.49 | 179.20 | 4761.15 |
| w/o step-oriented trend | 211.89 | 177.80 | 5297.17 |
| w/o Eq. 5 | 168.52 | 179.82 | 4212.88 |
| w/o Eq. 6 | 221.30 | 174.19 | 5572.41 |
| w/o noise injection | 188.61 | 176.28 | 4715.19 |
| Ours (Var-A) | 269.31 | 207.04 | 6732.70 |
| Ours (Var-B) | 179.31 | 191.78 | 4482.70 |
| **FreeMorph** | **162.99** | **152.88** | **4192.82** |

### Key Findings

1. **Speed advantage**: Completes within 30 seconds, **50×** faster than IMPUS and **10×** faster than DiffMorpher.
2. **Comprehensive superiority**: Achieves best performance on all three metrics—LPIPS, FID, and PPL.
3. **Strong generalization**: Handles image pairs with diverse semantics and layouts, validated across four categories in Morph4Data.
4. **Complementary components**: Spherical feature aggregation ensures directionality; prior-driven self-attention preserves identity; step-oriented change trend achieves consistent transitions.
5. **Critical pipeline design**: Swapping the application order of steps (Var-B) or removing standard attention (Var-A) both significantly degrade performance.

## Highlights & Insights

1. **Zero fine-tuning paradigm**: Fully leverages the capability of pre-trained diffusion models without modifying any weights.
2. **Elegant use of attention mechanisms**: Guidance information is injected by modifying K/V features rather than altering the model architecture.
3. **New evaluation dataset Morph4Data**: Covers four categories (same/different semantics × same/different layouts), addressing the bias of MorphBench toward visually similar image pairs.
4. **Extension to text-guided editing**: The image morphing framework can be directly extended to text-guided image editing by treating edits as a morphing process between a real image and a generated image.

## Limitations & Future Work

1. The fixed hyperparameters $\lambda_1$–$\lambda_4$ may not generalize across all scenarios, and no adaptive adjustment mechanism is provided.
2. The number of intermediate images is fixed at 5, limiting flexibility.
3. The method relies on LLaVA for generating text descriptions, whose quality affects the results.
4. Transition quality for image pairs with extreme differences (e.g., abstract paintings vs. photographs) remains to be validated.

## Related Work & Insights

- **DiffMorpher**: A diffusion-based morphing method using AdaIN and LoRA, requiring per-sample fine-tuning.
- **IMPUS**: A multi-stage training framework (text embedding optimization + LoRA training) requiring approximately 30 minutes per sample.
- **MasaCtrl / P2P / PnP**: Attention modification strategies from tuning-free image editing methods inspired the design of FreeMorph.

## Rating

⭐⭐⭐⭐ — The method is elegantly designed with a highly significant efficiency improvement and comprehensive, convincing experiments. However, manually set hyperparameters and a fixed number of intermediate frames limit practical flexibility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Timestep-Aware Diffusion Model for Extreme Image Rescaling](timestep-aware_diffusion_model_for_extreme_image_rescaling.md)
- [\[ICCV 2025\] EmotiCrafter: Text-to-Emotional-Image Generation based on Valence-Arousal Model](emoticrafter_text-to-emotional-image_generation_based_on_valence-arousal_model.md)
- [\[ICCV 2025\] MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](motionstreamer_streaming_motion_generation_via_diffusion-based_autoregressive_mo.md)
- [\[ICCV 2025\] MatchDiffusion: Training-free Generation of Match-Cuts](matchdiffusion_training-free_generation_of_match-cuts.md)
- [\[ICCV 2025\] LiT: Delving into a Simple Linear Diffusion Transformer for Image Generation](lit_delving_into_a_simple_linear_diffusion_transformer_for_image_generation.md)

</div>

<!-- RELATED:END -->

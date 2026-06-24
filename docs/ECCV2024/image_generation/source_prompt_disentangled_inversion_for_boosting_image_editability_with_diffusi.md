---
title: >-
  [Paper Note] Source Prompt Disentangled Inversion for Boosting Image Editability with Diffusion Models
description: >-
  [ECCV 2024][Image Generation] Proposes SPDInv—a source prompt disentangled inversion method. By modeling the inversion process as a fixed-point search problem and solving it using a pretrained diffusion model, the inverted noise map is disentangled from the source prompt, significantly boosting text-driven image editing quality.
tags:
  - "ECCV 2024"
  - "Image Generation"
date: 2026-05-08
content_hash: 3e8c3cd8c5f9a95c
---

# Source Prompt Disentangled Inversion for Boosting Image Editability with Diffusion Models

**Conference**: ECCV 2024  
**arXiv**: [2403.11105](https://arxiv.org/abs/2403.11105)  
**Area**: Image Generation

## TL;DR

Proposes SPDInv—a source prompt disentangled inversion method. By modeling the inversion process as a fixed-point search problem and solving it using a pretrained diffusion model, the inverted noise map is disentangled from the source prompt, significantly boosting text-driven image editing quality.

## Background & Motivation

- Core pipeline of text-driven image editing: given a source image $\rightarrow$ obtain a latent noise map via inversion $\rightarrow$ edit using a target prompt.
- DDIM inversion is the most commonly used method but possesses a fundamental drawback: **the inverted noise map is tightly coupled with the source prompt**.
- Existing improvement methods (such as NTI, NPI, and DirectInv) focus heavily on minimizing **reconstruction error** $D_{Rec}$ while neglecting **noise error** $D_{Noi}$.
- The noise map coupled with the source prompt conflicts when edited by the target prompt, leading to editing artifacts and content inconsistency.
- **Key Insight**: An ideal inversion should satisfy the fixed-point constraint $z_t = C_{t,1} \cdot z_{t-1} + C_{t,2} \cdot \epsilon_\theta(z_t, t, c)$, but DDIM inversion substitutes $(z_t, t)$ with $(z_{t-1}, t-1)$ as network inputs, introducing the source prompt prior.

## Method

### Overall Architecture

The core idea of SPDInv is to **minimize the noise error $D_{Noi}$ (instead of the reconstruction error $D_{Rec}$)**, making the inverted noise map as independent of the source prompt as possible.

1. Obtain an initial approximation $z_t$ using standard DDIM inversion.
2. Convert the fixed-point constraint into an optimization problem at each inversion step.
3. Solve it via gradient descent using the pretrained diffusion model.

### Key Designs

**1. Fixed-Point Constraint Analysis**

The ideal inversion equation (Eq.1) requires $(z_t, t)$ as network input, but DDIM inversion practically uses $(z_{t-1}, t-1)$. Since $z_{t-1}$ is obtained by conditional denoising of $z_t$ under the source prompt, this injects the source prompt information into $z_t$.

Reformulating the ideal inversion as a fixed-point problem:
$$x = f_\theta(x), \quad x = z_t, \quad f_\theta(x) = C_{t,2} \cdot \epsilon_\theta(x, t, c) + C_{t,1} \cdot z_{t-1}$$

**2. Gradient Descent-Based Fixed-Point Search**

- Unlike AIDI's fixed-loop iterations (which are unstable and sub-optimal), SPDInv reformulates the fixed-point constraint as a loss function:
$$L = \|f_\theta(z_t) - z_t\|_2$$
- Perform gradient descent on $z_t$ using the pretrained diffusion model with frozen parameters:
$$z_t := z_t - \eta \nabla L$$
- Introduce a threshold $\delta$ for early stopping: early inversion steps require more optimization loops, while late steps ($t > T/2$) converge in just a few iterations.

**3. Extension to Customized Image Generation**

Integrating SPDInv into customized image generation methods like ELITE:
1. Convert the given image to a text embedding space ("S*") using ELITE.
2. Invert the image to obtain the noise map using SPDInv (preserving layout and background).
3. Generate the edited results using a new text prompt (e.g., "a white S*").

### Loss & Training

The optimization objective for each inversion step:
$$\arg\min_{z_t} L = \|f_\theta(z_t) - z_t\|_2$$

Where $f_\theta(z_t) = C_{t,2} \cdot \epsilon_\theta(z_t, t, c) + C_{t,1} \cdot z_{t-1}$. The pretrained network parameters are frozen, and only the latent features $z_t$ are updated.

## Key Experimental Results

### Main Results

Comparison of different inversion methods under three editing engines on PIE-Bench:

| Inversion Method | Editing Engine | DINO↓(×10³) | PSNR↑ | LPIPS↓(×10³) | MSE↓(×10⁴) | SSIM↑(×10²) | CLIP↑ | Time (s) |
|---------|---------|------------|-------|-------------|------------|------------|------|--------|
| DDIM | P2P | 69.43 | 17.87 | 208.80 | 219.88 | 71.14 | 25.01 | 11.55 |
| NTI | P2P | 13.44 | 27.03 | 60.67 | 35.86 | 84.11 | 24.75 | 137.54 |
| DirectINV | P2P | 11.65 | 27.22 | 54.55 | 32.86 | 84.76 | 25.02 | 19.94 |
| AIDI | P2P | 12.16 | 27.01 | 56.39 | 36.90 | 84.27 | 24.92 | 87.21 |
| **SPDInv** | **P2P** | **8.81** | **28.60** | **36.01** | **24.54** | **86.23** | **25.26** | 27.04 |
| DDIM | MasaCtrl | 28.38 | 22.17 | 106.62 | 86.97 | 79.67 | 23.96 | 11.55 |
| DirectINV | MasaCtrl | 24.70 | 22.64 | 87.94 | 81.09 | 81.33 | 24.38 | 19.94 |
| **SPDInv** | **MasaCtrl** | **20.48** | **24.12** | **71.74** | **64.77** | **82.54** | **24.61** | 27.04 |
| DDIM | PNP | 28.22 | 22.28 | 113.33 | 83.51 | 79.00 | 24.95 | 11.55 |
| **SPDInv** | **PNP** | **15.58** | **26.72** | **91.55** | **34.69** | **82.04** | **25.14** | 27.04 |

Compared to the second-best method under the P2P engine, SPDInv: improves DINO by 24%, decreases LPIPS by 21%, and decreases MSE by 13%.

### Ablation Study

Influence of hyperparameters on performance (subset of PIE-Bench):

| Hyperparameter | DINO↓(×10³) | PSNR↑ | LPIPS↓(×10³) | MSE↓(×10⁴) | SSIM↑(×10²) | CLIP↑ |
|-------|------------|-------|-------------|------------|------------|------|
| K=5 | 8.52 | 31.49 | 22.31 | 10.42 | 90.21 | 26.70 |
| **K=25** | **8.43** | **31.61** | **21.70** | **10.12** | **90.28** | — |
| K=50 | Better | Better | Better | Better | Better | Slightly lower |
| δ=5e-4 | Worse | Worse | Worse | Worse | Worse | — |
| **δ=5e-6** | **Best** | **Best** | **Best** | **Best** | **Best** | — |
| η=0.005 | Slightly worse | Slightly worse | Slightly worse | Slightly worse | Slightly worse | Slightly higher |
| **η=0.001** | **Best** | **Best** | **Best** | **Best** | **Best** | — |

Customized image editing results (SPDInv-ELITE vs. original ELITE):

| Method | DINO↓(×10³) | PSNR↑ | LPIPS↓(×10³) | MSE↓(×10⁴) | SSIM↑(×10²) | CLIP↑ |
|------|------------|-------|-------------|------------|------------|------|
| ELITE (Original) | 148.37 | 14.83 | 201.94 | 359.58 | 67.62 | 15.72 |
| BlendDM | 59.21 | 15.51 | 244.07 | 306.75 | 67.45 | 20.21 |
| InstructP2P | 155.49 | 18.19 | 161.20 | 362.01 | 78.05 | 20.06 |
| **SPDInv-ELITE** | **21.23** | **24.14** | **74.36** | **48.73** | **88.90** | **19.18** |

Compared to the original ELITE, SPDInv-ELITE: improves DINO by 85%, improves PSNR by 62%, and decreases MSE by 86%.

### Key Findings

1. The fixed-point search method (SPDInv) significantly outperforms direct iteration (AIDI), validating the effectiveness of the gradient-based search strategy.
2. The early phase of the inversion process ($t < T/2$) requires more optimization loops to satisfy the fixed-point constraint, whereas the later phase converges extremely fast.
3. SPDInv is plug-and-play: requiring only about 10 lines of code modification to integrate into existing editing pipelines.
4. It demonstrates consistent advantages under three different editing engines (P2P, MasaCtrl, PNP), proving the generality of the method.

## Highlights & Insights

- Precise problem definition: The paradigm shift from "reconstruction error" to "noise error" is the most significant insight of this paper.
- Elegant and solid theoretical derivation: A complete logical chain established from ideal inversion to fixed-point constraints to the optimization problem.
- Extremely high engineering practicality: only 10 lines of code modification, support for multiple editing engines, and extensibility to customized generation.
- Inversion time (27s) is vastly lower than NTI (138s) and AIDI (87s) while offering the best performance.
- The extension direction of empowering customized generation methods (such as ELITE) with local editing capabilities is highly valuable.

## Limitations & Future Work

- Based on Stable Diffusion v1.4; compatibility with newer models (e.g., SD-XL, SD3) remains unverified.
- Each inversion step can require up to K=25 forward passes, still imposing computational overhead for real-time editing scenarios.
- The convergence of fixed-point search lacks rigorous theoretical guarantees and relies on empirically-set hyperparameters.
- The effectiveness of source prompt disentanglement is yet to be verified for extremely large-scale edits (e.g., completely changing the scene).

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Redefines the problem from the perspective of noise error; the fixed-point search idea is highly novel.
- **Value**: ⭐⭐⭐⭐⭐ — Plug-and-play, 10 lines of code, multi-engine compatibility, and highly extensible.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three engines × two benchmark datasets, thorough ablation, and extensions to customized editing.
- **Writing Quality**: ⭐⭐⭐⭐ — The motivation is clearly articulated, but there are many mathematical symbols, and some derivations could be more concise.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] ReNoise: Real Image Inversion Through Iterative Noising](renoise_real_image_inversion_through_iterative_noising.md)
- [\[ECCV 2024\] WildVidFit: Video Virtual Try-On in the Wild via Image-Based Controlled Diffusion Models](wildvidfit_video_virtual_try-on_in_the_wild_via_image-based_controlled_diffusion.md)
- [\[ECCV 2024\] Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)
- [\[ECCV 2024\] Lazy Diffusion Transformer for Interactive Image Editing](lazy_diffusion_transformer_for_interactive_image_editing.md)
- [\[ECCV 2024\] Learning Differentially Private Diffusion Models via Stochastic Adversarial Distillation](learning_differentially_private_diffusion_models_via_stochastic_adversarial_dist.md)

</div>

<!-- RELATED:END -->

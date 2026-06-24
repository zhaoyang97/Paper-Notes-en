---
title: >-
  [Paper Note] DiffiT: Diffusion Vision Transformers for Image Generation
description: >-
  [ECCV 2024][Image Generation][Vision Transformer] DiffiT (Diffusion Vision Transformer) is proposed, which introduces a Time-dependent Multi-head Self-Attention (TMSA) mechanism to dynamically adjust self-attention behaviors at different stages of the denoising process, achieving a state-of-the-art (SOTA) FID score of 1.73 on ImageNet-256 with 16-20% fewer parameters than DiT/MDT.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Vision Transformer"
  - "Time-dependent Self-Attention"
  - "Diffusion Models"
  - "Parameter-Efficient"
date: 2026-05-08
content_hash: 1cbbfbb2d1b97481
---

# DiffiT: Diffusion Vision Transformers for Image Generation

**Conference**: ECCV 2024  
**arXiv**: [2312.02139](https://arxiv.org/abs/2312.02139)  
**Code**: [https://github.com/NVlabs/DiffiT](https://github.com/NVlabs/DiffiT)  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Vision Transformer, Time-dependent Self-Attention, Diffusion Models, Image Generation, Parameter-Efficient

## TL;DR

DiffiT (Diffusion Vision Transformer) is proposed, which introduces a Time-dependent Multi-head Self-Attention (TMSA) mechanism to dynamically adjust self-attention behaviors at different stages of the denoising process, achieving a state-of-the-art (SOTA) FID score of 1.73 on ImageNet-256 with 16-20% fewer parameters than DiT/MDT.

## Background & Motivation

**Background**: Diffusion models have become the mainstream framework for image generation, with the denoising network as their core component. Traditionally, CNN-based U-Nets were used as the denoising backbone, while recent works like DiT and MDT have begun exploring the substitution of U-Nets with Vision Transformers.

**Limitations of Prior Work**:

**Coarse Injection of Temporal Conditions**: DiT and MDT use AdaLN (Adaptive LayerNorm) for noise timestep conditioning, modulating inputs via channel-wise scale and shift parameters. This mechanism cannot effectively model the joint relationship of spatial and temporal dependencies during the denoising process.

**Insufficient Capture of Denoising Temporal Dynamics**: At the beginning of denoising, the model primarily predicts low-frequency content (overall structure), while in the later stages, it focuses on high-frequency details. The AdaLN mechanism cannot dynamically adjust the attention mechanism's focus pattern according to the timestep.

**Low Parameter Efficiency**: AdaLN requires learning 6 modulation components per Transformer block (shift, scale, and gate for both self-attention and MLP), leading to a high parameter overhead.

**Key Challenge**: How to design a denoising network architecture that can finely control spatio-temporal interactions while maintaining parameter efficiency?

**Key Insight**: Directly integrate temporal information into the Q/K/V calculations of self-attention rather than using external modulation, enabling the attention itself to possess time-adaptive capabilities.

**Core Idea**: Q, K, and V are formed as linear combinations of spatial and temporal tokens. Consequently, the self-attention mechanism automatically shifts its focus pattern at different denoising stages—gradually transitioning from focusing on global structure to local details.

## Method

### Overall Architecture

DiffiT offers two variants:
- **Pixel-Space Model**: Adopts a symmetric U-Net encoder-decoder architecture, where each resolution level consists of $L$ DiffiT blocks. Convolutional layers are used for up/downsampling, and skip connections connect the encoder and decoder.
- **Latent-Space Model**: Uses a pre-trained VAE to encode images into latent representations. The denoising network is a pure Transformer (without up/downsampling), similar to the DiT architecture but replacing AdaLN with TMSA.

### Key Designs

1. **Time-dependent Multi-head Self-Attention (TMSA)**:

    - **Function**: Directly integrates time embeddings into the Q/K/V computation of self-attention.
    - **Mechanism**: For spatial token $\mathbf{x_s}$ and temporal token $\mathbf{x_t}$, calculate the time-dependent Q/K/V:
    $\mathbf{q_s} = \mathbf{x_s}\mathbf{W}_{qs} + \mathbf{x_t}\mathbf{W}_{qt}$
    $\mathbf{k_s} = \mathbf{x_s}\mathbf{W}_{ks} + \mathbf{x_t}\mathbf{W}_{kt}$
    $\mathbf{v_s} = \mathbf{x_s}\mathbf{W}_{vs} + \mathbf{x_t}\mathbf{W}_{vt}$
      The self-attention calculation is formulated as: $\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}(\frac{\mathbf{QK}^\top}{\sqrt{d}} + \mathbf{B})\mathbf{V}$
    - **Design Motivation**: Q, K, and V are all linear functions of spatial and temporal tokens, allowing attention to adaptively adjust across different timesteps. Compared to AdaLN's 6 modulation components, TMSA requires only 3 temporal linear projections ($\mathbf{W}_{qt}, \mathbf{W}_{kt}, \mathbf{W}_{vt}$) per block, requiring fewer parameters. Visualization shows that the attention maps of the TMSA model exhibit a progressive focusing effect from global to local.

2. **Window-based TMSA (Pixel-Space Model)**:

    - **Function**: Restricts self-attention computation to non-overlapping local windows.
    - **Mechanism**: Shares information across regions via the U-Net bottleneck layer, without directly computing global self-attention.
    - **Design Motivation**: The quadratic complexity of self-attention is expensive for large feature maps; windowing significantly reduces the token page sequence length. Experiments show that a window size of 4 achieves most of the performance gain.

3. **DiffiT ResBlock (Pixel-Space Model)**:

    - **Function**: A hybrid residual block combining convolutional layers and DiffiT Transformer blocks.
    - **Mechanism**: 
    $\mathbf{\hat{x}_s} = \text{Conv}_{3\times 3}(\text{Swish}(\text{GN}(\mathbf{x_s})))$
    $\mathbf{x_s} = \text{DiffiT-Block}(\mathbf{\hat{x}_s}, \mathbf{x_t}) + \mathbf{x_s}$
    - **Design Motivation**: Convolutional layers embed inductive biases of images, complementing Transformer blocks.

4. **Three-Channel Classifier-Free Guidance (Latent-Space Model)**:

    - **Function**: Uses three-channel Classifier-Free Guidance (CFG) in the latent-space model to improve generation quality.
    - **Design Motivation**: Directly improves the fidelity of conditional generation, achieving an optimal FID of 1.73 on ImageNet-256 using a guidance scale of 4.6.

### Loss & Training

- Standard denoising score matching loss: $\mathbb{E}[\lambda(t)\|\epsilon - \epsilon_\theta(\mathbf{z}_0 + \sigma_t\epsilon, t)\|_2^2]$
- Sampling uses stochastic differential equation solvers (SDE/ODE choice); the ODE solver requires fewer steps, while the SDE solver is more robust to inaccurate scores.
- Time embedding uses positional encoding (which outperforms Fourier encoding).

## Key Experimental Results

### Main Results (Latent-Space Model, ImageNet-256)

| Model | Type | Params (M) | FLOPs (G) | FID↓ | IS↑ | Precision↑ | Recall↑ |
|------|------|----------|---------|------|-----|-----------|---------|
| DiT-XL/2-G | Diffusion | 675 | 119 | 2.27 | 278.24 | 0.83 | 0.57 |
| MDT-G | Diffusion | 700 | 121 | 1.79 | 283.01 | 0.81 | 0.61 |
| SiT-XL | Diffusion | 675 | 119 | 2.06 | 270.27 | 0.82 | 0.59 |
| StyleGAN-XL | GAN | - | - | 2.30 | 265.12 | 0.78 | 0.53 |
| **DiffiT** | **Diffusion** | **561** | **114** | **1.73** | **276.49** | **0.80** | **0.62** |

**Pixel-Space Model**:

| Model | CIFAR-10 FID↓ | FFHQ-64 FID↓ |
|------|-------------|-------------|
| EDM (VP) | 1.99 | 2.39 |
| LSGM | 2.01 | - |
| **DiffiT** | **1.95** | **2.22** |

### Ablation Study

**Architecture Design Ablation (CIFAR-10)**:

| Config | Encoder | Decoder | FID↓ | Description |
|-----|--------|--------|------|------|
| A | ViT | SETR-MLA | 5.34 | Isotropic architecture is suboptimal |
| B | + Multi-resolution | SETR-MLA | 4.64 | Multi-scale features are helpful |
| C | Multi-resolution | + Multi-resolution | 3.71 | Symmetric U-Net yields further improvements |
| D | + DiffiT Encoder | Multi-resolution | 2.27 | Significant effectiveness of TMSA |
| E | + DiffiT Encoder | + DiffiT Decoder | 1.95 | Full DiffiT |

**Effectiveness of TMSA (Replacing DDPM++ Self-Attention)**:

| Model | Without TMSA FID↓ | With TMSA FID↓ | Gain |
|------|-----------|-----------|------|
| DDPM++ (VE) | 3.77 | 3.49 | -0.28 |
| DDPM++ (VP) | 3.01 | 2.76 | -0.25 |

**Ablation on Temporal Condition Injection Location**:

| Config | Injection Method | FID↓ | Description |
|-----|---------|------|------|
| F | Relative position bias only | 3.97 | Unable to jointly encode space and time |
| G | MLP layer only | 3.81 | Suboptimal |
| H | TMSA (Q/K/V) | 1.95 | Optimal location |

### Key Findings

- TMSA introduces significant and consistent FID improvements in DiffiT across datasets and model configurations.
- Temporal tokens must be mixed with spatial tokens rather than processed separately (separate processing degrades the FID from 1.95 to 2.28).
- Positional temporal embeddings outperform Fourier temporal embeddings (1.95 vs 2.02).
- Increasing the window size from 2 to 4 improves performance by 23%, but going from 4 to 8 yields only a 1.5% gain, suggesting spatial redundancy.
- DiffiT operates with 16.88% fewer parameters and 4.38% lower FLOPs than DiT-XL, yet achieves superior FID.

## Highlights & Insights

- The design of TMSA is extremely simple—merely incorporating a linear projection of temporal tokens during Q/K/V projections—yet it is highly effective and generalizable (directly replacing self-attention in any Transformer).
- While the observation that "attention should vary with timesteps during the denoising process" is intuitive, prior methods (like AdaLN) did not truly achieve this at the attention level.
- The hybrid design of convolutions and Transformers in the pixel-space model leverages their complementary advantages.

## Limitations & Future Work

- Latent DiffiT performs slightly worse than StyleGAN-XL on ImageNet-512 (FID 2.67 vs 2.41), though GANs may suffer from limited diversity.
- The window-based variant of TMSA lacks a cross-window communication mechanism (relying on the U-Net bottleneck), which might be insufficient for pure Transformer architectures.
- Not validated on text-to-image generation tasks; experiments are limited to class-conditional and unconditional generation.
- Lacks comparisons with subsequent works (such as DiT-3, Flux, etc.).

## Related Work & Insights

- **DiT**: The first work to replace U-Net with a Transformer for latent-space diffusion, utilizing AdaLN conditioning. This work proves that TMSA is a superior alternative.
- **MDT**: Introduces masked modeling on top of DiT to capture contextual information, though with a more complex training pipeline.
- **SiT**: Combines Flow Matching with DiT, which is orthogonal to the design of DiffiT.
- **EDM**: A strong baseline for pixel-space diffusion models, which DiffiT outperforms on CIFAR-10 and FFHQ-64.
- Insight: The location and method of temporal condition injection are critical for diffusion models; directly decoupling into the Q/K/V of the attention mechanism is more effective than external modulation.

## Rating
- Novelty: ⭐⭐⭐⭐ TMSA offers a simple and novel approach, elegantly integrating temporal information into the self-attention mechanism.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Highly detailed ablation studies, covering multiple dimensions including architecture, TMSA design, window size, CFG, time embeddings, and efficiency.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with organized experiments, and the visualized attention maps are highly convincing.
- Value: ⭐⭐⭐⭐⭐ TMSA can directly replace standard self-attention layers in existing methods with strong generalizability. Released by NVIDIA, the source code is open-source.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Realistic Human Motion Generation with Cross-Diffusion Models](realistic_human_motion_generation_with_cross-diffusion_models.md)
- [\[CVPR 2026\] PixelDiT: Pixel Diffusion Transformers for Image Generation](../../CVPR2026/image_generation/pixeldit_pixel_diffusion_transformers_for_image_generation.md)
- [\[ECCV 2024\] M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)
- [\[ECCV 2024\] Lazy Diffusion Transformer for Interactive Image Editing](lazy_diffusion_transformer_for_interactive_image_editing.md)
- [\[ECCV 2024\] OMG: Occlusion-friendly Personalized Multi-concept Generation in Diffusion Models](omg_occlusion-friendly_personalized_multi-concept_generation_in_diffusion_models.md)

</div>

<!-- RELATED:END -->

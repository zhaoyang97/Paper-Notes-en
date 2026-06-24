---
title: >-
  [Paper Note] ε-VAE: Denoising as Visual Decoding
description: >-
  [ICML 2025][Image Restoration][Diffusion Models] This paper proposes ε-VAE, which replaces the single-step deterministic decoder in traditional autoencoders with a diffusion/denoising process to implement "denoising as decoding." Under the same compression rate, the reconstruction quality is improved by 40% and downstream generation quality is enhanced by 22%. Alternatively, it achieves a 2.3× inference acceleration by increasing the compression rate while maintaining the sam…
tags:
  - "ICML 2025"
  - "Image Restoration"
  - "Diffusion Models"
  - "VAE"
  - "Image Tokenizer"
  - "Rectified Flow"
  - "Visual Decoding"
  - "Iterative Reconstruction"
date: 2026-05-08
content_hash: 856608fc8ed73233
---

# ε-VAE: Denoising as Visual Decoding

**Conference**: ICML 2025  
**arXiv**: [2410.04081](https://arxiv.org/abs/2410.04081)  
**Authors**: Long Zhao, Sanghyun Woo, Ziyu Wan, Yandong Li, Han Zhang, Boqing Gong, Hartwig Adam, Xuhui Jia, Ting Liu
**Institutions**: Google DeepMind
**Code**: None  
**Area**: Image Restoration  
**Keywords**: Diffusion Models, VAE, Image Tokenizer, Rectified Flow, Visual Decoding, Iterative Reconstruction

## TL;DR

This paper proposes ε-VAE, which replaces the single-step deterministic decoder in traditional autoencoders with a diffusion/denoising process to implement "denoising as decoding." Under the same compression rate, the reconstruction quality is improved by 40% and downstream generation quality is enhanced by 22%. Alternatively, it achieves a 2.3× inference acceleration by increasing the compression rate while maintaining the same generation quality.

## Background & Motivation

### Importance of Visual Tokenization

Modern visual generative models (autoregressive models and diffusion models) are typically trained in a low-resolution latent space, relying on a tokenizer to compress high-dimensional images into compact latent representations. This process is crucial for generation quality:

- **Autoregressive Models**: Discrete tokens enable step-by-step conditional generation.
- **Diffusion Models**: Continuous latent variables make the learning of the denoising process more efficient.
- Experimental evidence suggests that tokenization significantly boosts generation performance.

### Limitations of Prior Work

Standard visual autoencoders (such as SD-VAE) adopt an "encoder + deterministic decoder" architecture:
- The encoder $\mathcal{E}$ compresses the image $\mathbf{x} \in \mathbb{R}^{H \times W \times 3}$ into a latent space representation.
- The decoder $\mathcal{G}$ directly reconstructs the latent variables back to the pixel space in a **single step**.

**Core Problem**: The expressive capability of single-step reconstruction is limited. Especially under high compression rates, the reconstruction quality degrades sharply, thereby becoming a bottleneck for the quality of downstream generative models.

### Core Motivation

The authors propose a new perspective: **Why must decoding be a single-step process?** Since diffusion models have demonstrated the strong expressive capability of iterative generation, can this capability be introduced into the decoding phase of autoencoders?

Although existing work (DiffAE, Preechakul et al., 2022) and parallel work (Birodkar et al., 2024) have explored the application of diffusion mechanisms in autoencoding, none have surpassed traditional autoencoding paradigms in practical performance. Through careful co-design of the architecture and objective function, ε-VAE achieves this breakthrough for the first time.

## Method

### Overall Architecture

The core idea of ε-VAE is to **replace the traditional decoder with a diffusion process**:

1. **Encoder** (unchanged): A convolutional encoder $\mathcal{E}$ compresses the input image into a latent representation $z = \mathcal{E}(\mathbf{x})$.
2. **Diffusion Decoder** (Core Innovation): Instead of direct reconstruction using a convolutional network, it starts from noise and is conditioned on the encoder output $z$ to **iteratively denoise** and progressively recover the original image via a diffusion model:

$$\hat{\mathbf{x}} = \text{DiffusionDecode}(\mathbf{x}_T, z) \quad \text{where } \mathbf{x}_T \sim \mathcal{N}(0, I)$$

This design transforms the reconstruction process from a one-step mapping to a progressive refinement process, where the diffusion model step-by-step recovers the original data under the guidance of the encoder's latent variables.

### Key Designs

The authors systematically analyze the impact of the following key design factors on performance through controlled experiments:

#### 1. Condition Injection Architecture

The diffusion decoder must be effectively conditioned on the latent representations of the encoder. This paper explores various condition injection mechanisms, ensuring that the diffusion model can leverage the compressed information provided by the encoder to guide the denoising direction. This forms the core difference between ε-VAE and normal diffusion models—it does not generate from scratch but performs conditional reconstruction under the semantic guidance provided by the encoder.

#### 2. Jointly Designed Training Objectives

The training objective of ε-VAE contains not only the standard denoising loss of diffusion models but also integrates reconstruction losses proven effective in traditional autoencoders:

- **Diffusion Denoising Loss**: Standard noise prediction / velocity field regression target.
- **Perceptual Loss (LPIPS)**: Measures the perceptual similarity between reconstructed images and original ones in deep feature space.
- **Adversarial Loss (GAN loss)**: Incorporates a discriminator to improve the visual quality and sharpness of reconstructed images.

This collaborative training of "diffusion + traditional autoencoder losses" is one of the key factors behind the performance breakthrough of ε-VAE.

#### 3. Model Parameterization

The choice of prediction targets for the diffusion decoder ($\epsilon$-prediction vs. $x$-prediction vs. $v$-prediction) has a significant impact on performance. The "$\epsilon$" in the paper's title implies the core status of noise prediction ($\epsilon$-prediction) in this framework. Different parameterization methods affect the model's optimization trajectory and convergence.

#### 4. Noise Scheduling

The noise scheduling strategy determines the distribution of noise levels at each timestep in the diffusion process, which directly affects:
- The smoothness of the optimization trajectory.
- The reconstruction accuracy of information at different frequencies.
- The reconstruction quality during few-step inference.

#### 5. Timestep Distribution

The sampling distribution of timesteps during training and testing significantly impacts performance:
- **Training Phase**: The timestep sampling strategy affects the model's capacity to handle different noise levels.
- **Testing Phase**: The choice of timesteps for few-step sampling directly determines the trade-off between reconstruction quality and efficiency.

### Sampling Efficiency

A prominent feature of ε-VAE is its **extremely high sampling efficiency**—requiring only 1 to 3 denoising steps to achieve high-quality reconstruction, thanks to the strong semantic guidance provided by the encoder. In contrast, unconditional diffusion models usually require 50 to 1000 steps to generate high-quality images.

### Resolution Generalization

ε-VAE demonstrates robust resolution generalization, meaning that it can generalize to other resolutions after training on a single resolution, which is highly practical for deployment.

## Key Experimental Results

### Reconstruction Quality Comparison (rFID)

Under the standard setup (Rombach et al., 2022), ε-VAE achieves substantial improvements in reconstruction quality (rFID) compared to SOTA autoencoding methods:

| Method Type | Decoding Method | Reconstruction Quality Gain | Inference Steps |
|---|---|---|---|
| Traditional VAE (e.g., SD-VAE) | Single-step deterministic decoding | Baseline | 1 step |
| ε-VAE (Ours) | Diffusion iterative decoding | **↑ 40%** (rFID) | 1-3 steps |

### Downstream Generation Quality Comparison (FID)

When ε-VAE is integrated as a tokenizer into downstream diffusion generative models, the generation quality (FID) is significantly improved:

| Dimension | Traditional VAE | ε-VAE | Gain |
|---|---|---|---|
| FID under the same compression rate | Baseline | Lower | **↓ 22%** |
| Inference speed under the same generation quality | Baseline | Faster | **2.3× speedup** |

### Compression Rate vs. Generation Quality Trade-off

The core advantage of ε-VAE is offering a better trade-off between compression and quality:

| Strategy | Compression Rate | Generation Quality | Inference Speed |
|---|---|---|---|
| Strategy A: Maintain compression rate | Unchanged | ↑ 22% | Unchanged |
| Strategy B: Increase compression rate | ↑ Higher | Equal | ↑ 2.3× |

Strategy B demonstrates that ε-VAE allows for more aggressive compression without sacrificing quality, which is of great significance for deploying large-scale image generation: higher compression translates to smaller latent spaces, making the diffusion model run faster in a smaller space.

### Few-step Inference Capability

| Inference Steps | Reconstruction Quality | Remarks |
|---|---|---|
| 1 step | Effective | Computational cost close to traditional VAE |
| 2-3 steps | Optimal | Best balance between performance and efficiency |
| More steps | Diminishing returns | Limited gains beyond 3 steps |

## Highlights & Insights

1. **Novel Perspective**: Redefines "decoding" as "denoising," breaking the implicit assumption in autoencoders that "decoding must be a single-step." This perspective, while conceptually simple, requires meticulous system design to yield practical results.
2. **Co-design**: Combining the diffusion loss with traditional autoencoder losses (LPIPS + GAN), rather than simply replacing them, provides an important engineering insight.
3. **Efficient Sampling**: The capability for high-quality reconstruction in just 1-3 steps prevents the diffusion decoder from becoming a bottleneck in real-world scenarios. This is attributed to the strong conditional guidance provided by the encoder's latent variables.
4. **Flexible Compression-Speed Trade-off**: Users can choose either "improving quality at the same compression rate" or "speeding up at the same quality" depending on practical requirements, providing useful deployment flexibility.
5. **Resolution Generalization**: Training resolutions can generalize to other resolutions, reducing constraints during real-world deployment.

## Limitations & Future Work

1. **Inference Latency**: Even though only 1-3 steps are required, the diffusion decoder is still slower than traditional single-step decoders, potentially introducing bottlenecks for real-time applications.
2. **Increased Training Complexity**: Joint training of the encoder and the diffusion decoder is required, introducing diffusion-related hyperparameters (parameterization, scheduling, timestep distribution, etc.), which raises tuning costs.
3. **Limited Cached Content**: Detailed method segments of the paper (architecture diagrams, specific conditioning mechanisms, quantitative ablation details) are not fully presented in the cache.
4. **Integration with Discrete Tokenizers**: This work focuses on continuous latents; whether the diffusion decoder is compatible with discrete tokens (such as VQ-VAE) remains to be explored.
5. **Large-scale Validation**: Validation at higher resolutions (e.g., 1024+) and on larger-scale datasets will further enhance persuasion.

## Related Work & Insights

- **Stable Diffusion VAE (Rombach et al., 2022)**: Standard convolutional autoencoder, serving as the direct baseline for ε-VAE.
- **DiffAE (Preechakul et al., 2022)**: The first to introduce diffusion processes into the autoencoding framework, though it did not outperform traditional methods in reconstruction quality.
- **Latent Consistency Models (LCM)**: Also focus on few-step inference efficiency and can complement ε-VAE within a distillation framework.
- **SDXL VAE / DC-AE**: Recent high-performance visual autoencoders, where ε-VAE offers an orthogonal direction of improvement.

**Insight**: ε-VAE reveals an important rule—within the autoencoder framework, the expressive capability of the decoder is a critical bottleneck limiting reconstruction quality. Correcting this with a diffusion process essentially trades "time for accuracy." However, owing to the strong conditioning signals provided by the encoder, the required extra overhead is extremely low (1-3 steps), making this method highly cost-effective in practice. This paradigm can be generalized to other scenarios where "single-step mapping lacks sufficient expressiveness."

## Rating

- Novelty: ⭐⭐⭐⭐ — The perspective of "denoising as decoding" is clear and powerful, though the idea of diffusion decoding has been preliminarily explored before.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Systematic controlled experiments cover key design choices, with both reconstruction and generation quality fully evaluated.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, explicit contributions, and standard structure.
- Value: ⭐⭐⭐⭐ — Provides a new paradigm for visual tokenizer design, carrying direct value for the LDM ecosystem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Prior Does Matter: Visual Navigation via Denoising Diffusion Bridge Models](../../CVPR2025/image_restoration/prior_does_matter_visual_navigation_via_denoising_diffusion_bridge_models.md)
- [\[CVPR 2025\] Iterative Predictor-Critic Code Decoding for Real-World Image Dehazing](../../CVPR2025/image_restoration/iterative_predictor-critic_code_decoding_for_real-world_image_dehazing.md)
- [\[CVPR 2025\] Efficient Visual State Space Model for Image Deblurring](../../CVPR2025/image_restoration/efficient_visual_state_space_model_for_image_deblurring.md)
- [\[CVPR 2025\] Visual-Instructed Degradation Diffusion for All-in-One Image Restoration](../../CVPR2025/image_restoration/visual-instructed_degradation_diffusion_for_all-in-one_image_restoration.md)
- [\[ICML 2026\] DAPD: Dependency-Aware Parallel Decoding via Attention for Diffusion LLMs](../../ICML2026/image_restoration/dapd_dependency-aware_parallel_decoding_via_attention_for_diffusion_llms.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression
description: >-
  [CVPR 2026][Image Generation][Image Compression] Ours proposes DiT-IC, which adapts a pre-trained T2I Diffusion Transformer into a one-step image compression reconstruction model via three alignment mechanisms (Variance-Guided Reconstruction Flow, Self-Distillation Alignment, and Latent Conditional Guidance). By performing diffusion in a deep latent space with $32\times$ downsampling, it achieves SOTA perceptual quality with decoding speeds $30\times$ faster than existing diffusion-based codecs.
tags:
  - CVPR 2026
  - Image Generation
  - Image Compression
  - Diffusion Transformer
  - One-step Denoising
  - Alignment Mechanism
  - Efficient Decoding
date: 2026-05-08
content_hash: ef72b822e9c8f33a
---

# DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression

**Conference**: CVPR 2026  
**arXiv**: [2603.13162](https://arxiv.org/abs/2603.13162)  
**Code**: [Project Page](https://njuvision.github.io/DiT-IC/)  
**Area**: Image Compression / Diffusion Models  
**Keywords**: Image Compression, Diffusion Transformer, One-step Denoising, Alignment Mechanism, Efficient Decoding

## TL;DR

Ours proposes DiT-IC, which adapts a pre-trained T2I Diffusion Transformer into a one-step image compression reconstruction model via three alignment mechanisms (Variance-Guided Reconstruction Flow, Self-Distillation Alignment, and Latent Conditional Guidance). By performing diffusion in a deep latent space with $32\times$ downsampling, it achieves SOTA perceptual quality with decoding speeds $30\times$ faster than existing diffusion-based codecs.

## Background & Motivation

Diffusion-based image compression excels in perceptual fidelity, but its practicality is limited by two bottlenecks:
- **Sampling Overhead**: Existing methods require 4-50 iterative denoising steps, leading to decoding latency exceeding 1 second.
- **Shallow Latent Space**: Most diffusion codecs use U-Net architectures ($8\times$ spatial downsampling), while traditional VAE codecs operate in deep latent domains ($16\times$-$64\times$). This gap is a source of computational waste.

**Core Problem**: Can diffusion operate effectively in a deeply compressed latent space ($32\times$ downsampling) without losing reconstruction quality?

**Key Insights**: The hierarchical downsampling of U-Net further reduces spatial resolution, making it unsuitable for deep latent representations. Conversely, DiT maintains a constant spatial resolution throughout the denoising process, making it naturally compatible with deep latent spaces. However, directly migrating a pre-trained DiT leads to severe degradation due to the fundamental mismatch between the generative goal (starting from pure noise) and the compression reconstruction goal (starting from structured quantized latents).

## Method

### Overall Architecture

Based on SANA (pre-trained T2I DiT) + ELIC (auxiliary encoder), diffusion reconstruction is performed in a $32\times$ downsampled latent space. The encoder produces a quantized latent representation $\hat{\mathbf{y}}$, the DiT performs one-step reconstruction, and the decoder recovers the image. Lightweight adaptation is achieved via LoRA (VAE rank 32, DiT rank 64), with two-stage training covering multiple bitrates.

### Key Designs

1. **Variance-Guided Reconstruction Flow (Alignment from Generation to Reconstruction)**: The initial state of image compression is not pure Gaussian noise but structured quantized latents—most information is already near the data manifold. Compression noise is spatially heterogeneous: smooth regions resemble low noise (small timesteps), while textured regions resemble high noise (large timesteps). Thus, pixel-wise pseudo-timesteps are mapped from the variance $\boldsymbol{\sigma}$ predicted by the encoder:
$$t = \mathcal{F}(\text{proj}_\theta(\boldsymbol{\sigma})) \in \mathbb{R}^{H \times W}$$
High variance $\rightarrow$ large $t$ $\rightarrow$ strong denoising; low variance $\rightarrow$ small $t$ $\rightarrow$ weak denoising. One-step reconstruction is defined as $\hat{\mathbf{y}} = \tilde{\mathbf{y}} - \mathbf{v}_\theta(\tilde{\mathbf{y}}, t)$, compressing the multi-step denoising trajectory into a single spatially adaptive transformation.

2. **Self-Distillation Alignment (Alignment from Multi-step to One-step)**: In compression scenarios, no pre-trained multi-step teacher model is available. Ours innovatively uses the latent representation $\mathbf{y}_0$ from the frozen encoder as a self-supervised target—it is already near the data manifold and serves as a natural alignment target for the one-step denoising output $\hat{\mathbf{y}}_0$:
$$\mathcal{L}_{\text{distil}} = \mathbb{E}\left[1 - m - \frac{\langle\hat{\mathbf{y}}, \mathbf{y}_0\rangle}{|\hat{\mathbf{y}}|_2 |\mathbf{y}_0|_2}\right]$$
Using a marginal cosine alignment loss, the encoder is frozen while the DiT and decoder are jointly optimized to preserve the latent space geometry defined by the encoder.

3. **Latent Conditional Guidance (Alignment from Text to Semantic Latents)**: Compression reconstruction does not require text prompts, but pre-trained DiTs rely on text conditions. A lightweight projection module maps the compressed latents to the embedding space of the text encoder: $c_{\text{lat}} = \text{Proj}_\psi(\hat{y})$. Latent embeddings and text embeddings are aligned via a CLIP-style contrastive learning loss:
$$\mathcal{L}_{\text{cond}} = -\mathbb{E}_{(x_i,t_i)}\left[\log\frac{\exp(\langle c_{\text{lat},i}, c_{\text{text},i}\rangle/\tau)}{\sum_j \exp(\langle c_{\text{lat},i}, c_{\text{text},j}\rangle/\tau)}\right]$$
During training, both embeddings are aligned; during inference, only the latent condition is used, removing the heavy text encoder.

### Loss & Training

- Two-stage training: Stage 1 uses a small $\lambda$ to relax bitrate constraints and preserve features (100K iter, $256\times 256$, batch 32); Stage 2 uses a large $\lambda$ to tighten bitrate and adversarial loss to enhance perceptual quality (60K iter, $512\times 512$, batch 16).
- Distortion Loss: $\mathcal{D} = \lambda_1 \text{MSE} + \lambda_2 \text{LPIPS} + \lambda_3 \text{DISTS}$
- Alignment Loss: $\mathcal{L}_{\text{align}} = \lambda_4 \mathcal{L}_{\text{distil}} + \lambda_5 \mathcal{L}_{\text{cond}}$
- Bitrate Loss: $\mathcal{R} = -\log_2 p_{\hat{\mathbf{y}}}(\hat{\mathbf{y}}|\hat{\mathbf{z}}) - \log_2 p_{\hat{\mathbf{z}}}(\hat{\mathbf{z}})$
- Uses InternVL as the vision-language backbone with an EMA decay rate of 0.999.

## Key Experimental Results

### Main Results (BD-rate $\downarrow$, using PerCo as anchor)

| Method | Diffusion Steps | Decoding Latency (s) | LPIPS BD-rate $\downarrow$ | DISTS BD-rate $\downarrow$ |
|------|---------|------------|---------------|---------------|
| ResULIC (ICML'25) | 4 | 0.83 | -62.27 | -65.64 |
| StableCodec (ICCV'25) | 1 | 0.34 | -79.19 | -83.95 |
| OSCAR (NeurIPS'25) | 1 | 0.32 | -19.04 | -58.38 |
| **Ours** | **1** | **0.15** | **-83.65** | **-87.88** |

DiT-IC achieves the best BD-rate on both LPIPS and DISTS with the fastest decoding speed (0.15s FP16).

### Ablation Study

| Configuration | LPIPS BD-rate | DISTS BD-rate |
|------|-------------|-------------|
| DiT-IC Full | 0.00% | 0.00% |
| W/o Adversarial Loss | -2.27% | -1.80% |
| Training from Scratch (No Pre-training) | +22.00% | +32.45% |
| LoRA rank 16/16 | +12.77% | +13.92% |
| LoRA rank 32/32 | +5.31% | +5.56% |
| Full Fine-tuning | +7.95% | +8.05% |

Decoding latency comparison (FP32, $2048\times 2048$): StableCodec 0.8s diffusion vs. DiT-IC 0.12s diffusion ($-85\%$).

### Key Findings

- Pre-trained weights are crucial: Training from scratch leads to a $32.45\%$ degradation in DISTS.
- LoRA rank 32/64 is the optimal balance point; full fine-tuning is slightly worse (disrupting the pre-trained distribution under small batches).
- The advantage is more pronounced at $4096\times 4096$ resolution: StableCodec latency spikes to 10.3s, while DiT-IC is only 0.47s ($-95\%$).
- DiT's constant spatial resolution architecture is key for diffusion in deep latent spaces—U-Net's downsampling leaves no room for operation on $32\times$ latents.
- Capable of reconstructing $2048\times 2048$ images on a 16GB laptop GPU.

## Highlights & Insights

- Three alignment mechanisms precisely solve the target mismatch from generation to reconstruction: variance-guided alignment for noise characteristics, self-distillation for the learning process, and latent conditional alignment for input conditions.
- The spatial-adaptive design of variance $\rightarrow$ pseudo-timestep is elegant—different regions require different levels of "repair."
- Self-distillation requires no external teacher, using the encoder's own output as the target, making the solution simple and effective.
- Removing the text encoder balances efficiency and semantics—distilling text priors into latent conditions via contrastive learning.

## Limitations & Future Work

- At extremely low bitrates ($<0.01$ bpp), latent information may be insufficient, potentially requiring auxiliary text priors.
- Adversarial training may introduce a perception-distortion tradeoff, causing slight drops in traditional metrics like MSE/SSIM.
- Adversarial distillation techniques (e.g., SDXL-Turbo) are not yet integrated, which could further improve perceptual realism.
- Training data is limited to 150K images; larger scales may bring further improvements.
- While the NoPE (No Positional Encoding) strategy supports resolution generalization, its robustness at extreme resolutions remains to be validated.

## Related Work & Insights

- Comparison with StableCodec: The latter is based on SD/U-Net in the $8\times$ domain; DiT-IC achieves higher efficiency in the $32\times$ domain.
- Comparison with OSCAR: The latter uses image-level bitrate-to-timestep mapping; DiT-IC uses more granular pixel-level variance-to-timestep mapping.
- OneDC uses an image tokenizer as a condition, sharing a similar philosophy with latent conditional guidance.
- Demonstrates the strong transferability of pre-trained Diffusion Transformers to compression tasks—"alignment" is the key to unlocking this capability.

## Rating

- Novelty: ⭐⭐⭐⭐ Three alignment mechanisms systematically solve the generation $\rightarrow$ compression domain shift; variance-guided timesteps are a highlight.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets, comprehensive BD-rate + latency analysis, detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ The alignment perspective unifies three designs with a clear structure.
- Value: ⭐⭐⭐⭐ Significant for actual image compression deployment; $30\times$ acceleration makes diffusion compression feasible for the first time.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] MPDiT: Multi-Patch Global-to-Local Transformer Architecture for Efficient Flow Matching](mpdit_multi-patch_global-to-local_transformer_architecture_for_efficient_flow_ma.md)
- [\[CVPR 2026\] Guiding a Diffusion Transformer with the Internal Dynamics of Itself](guiding_a_diffusion_transformer_with_the_internal_dynamics_of_itself.md)
- [\[CVPR 2026\] DA-VAE: Plug-in Latent Compression for Diffusion via Detail Alignment](da-vae_plug-in_latent_compression_for_diffusion_via_detail_alignment.md)
- [\[CVPR 2026\] EdgeDiT: Hardware-Aware Diffusion Transformers for Efficient On-Device Image Generation](edgedit_hardware-aware_diffusion_transformers_for_efficient_on-device_image_gene.md)
- [\[CVPR 2026\] Memory-Efficient Fine-Tuning Diffusion Transformers via Dynamic Patch Sampling and Block Skipping](memory-efficient_fine-tuning_diffusion_transformers_via_dynamic_patch_sampling_a.md)

<!-- RELATED:END -->

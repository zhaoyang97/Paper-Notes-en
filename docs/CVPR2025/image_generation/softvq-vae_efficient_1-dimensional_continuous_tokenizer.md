---
title: >-
  [Paper Note] SoftVQ-VAE: Efficient 1-Dimensional Continuous Tokenizer
description: >-
  [CVPR 2025][Image Generation][Image Tokenizers] SoftVQ-VAE achieves a fully differentiable continuous image tokenizer by replacing the hard categorical posterior of VQ-VAE with a soft categorical posterior (where each latent token adaptively aggregates multiple codewords). It compresses 256×256 and 512×512 images to extremely high compression ratios using only 32-64 1D tokens, enabling SiT-XL to achieve a 1.78 FID on ImageNet with an 18-55x increase in inference throughput.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Image Tokenizers"
  - "Soft Vector Quantization"
  - "High Compression Ratio"
  - "Continuous Latent Space"
  - "Generative Efficiency"
date: 2026-05-08
content_hash: e6f48d5970ba498e
---

# SoftVQ-VAE: Efficient 1-Dimensional Continuous Tokenizer

**Conference**: CVPR 2025  
**arXiv**: [2412.10958](https://arxiv.org/abs/2412.10958)  
**Code**: [https://github.com/Hhhhhhao/continuous_tokenizer](https://github.com/Hhhhhhao/continuous_tokenizer)  
**Area**: Image Generation / Image Tokenizers  
**Keywords**: Image Tokenizers, Soft Vector Quantization, High Compression Ratio, Continuous Latent Space, Generative Efficiency

## TL;DR

SoftVQ-VAE achieves a fully differentiable continuous image tokenizer by replacing the hard categorical posterior of VQ-VAE with a soft categorical posterior (where each latent token adaptively aggregates multiple codewords). It compresses 256×256 and 512×512 images to extremely high compression ratios using only 32-64 1D tokens, enabling SiT-XL to achieve a 1.78 FID on ImageNet with an 18-55x increase in inference throughput.

## Background & Motivation

**Background**: Denoising generative models (DiT, SiT, MAR) rely on image tokenizers to encode raw images into latent tokens. Mainstream tokenizers include KL-VAE (continuous Gaussian posterior) and VQ-VAE (discrete categorical posterior), which typically encode a 256×256 image into at least 256 2D tokens.

**Limitations of Prior Work**: (1) The computational complexity of Transformer-based generative models scales quadratically with the token length, making 256+ tokens a critical bottleneck for training and inference efficiency; (2) To further increase the compression ratio, KL-VAE suffers from posterior collapse, while VQ-VAE experiences severe degradation in reconstruction and latent space quality due to gradient disconnection of the discrete quantization (the straight-through trick); (3) The latent spaces of existing tokenizers lack semantic discriminativeness, making it difficult for downstream generative models to learn.

**Key Challenge**: High compression ratios require each token to carry more information, but the Gaussian constraint of KL-VAE and the one-to-one quantization of VQ-VAE both restrict the representational capacity of a single token.

**Goal**: To design a continuous image tokenizer that can achieve high-quality reconstruction and generation using an extremely small number of 1D tokens (32-64).

**Key Insight**: Allowing each latent token to adaptively aggregate multiple codewords with soft weights, rather than mapping to a single codeword (as in VQ-VAE), can significantly boost representational capacity while preserving the structured advantages of a codebook.

**Core Idea**: Replace the argmin hard assignment in VQ-VAE with a softmax soft assignment: $q_\phi(\mathbf{z}|\mathbf{x}) = \text{Softmax}(-\|\hat{\mathbf{z}} - \mathcal{C}\|_2 / \tau)$. Each token becomes a weighted sum of multiple codewords, rendering the entire process fully differentiable without requiring the straight-through trick.

## Method

### Overall Architecture

An encoder-decoder architecture based on ViT is adopted. The encoder takes image patch tokens and $L$ 1D learnable query tokens as input, aggregating image information into the query tokens via self-attention. The encoder output passes through the SoftVQ module (soft matching with the codebook) to produce the final latent tokens. The decoder receives the latent tokens and $N$ mask tokens to reconstruct the pixel values.

### Key Designs

1. **Soft Vector Quantization (SoftVQ)**:

    - **Function**: Maps the encoder output to a highly expressive, continuous latent space.
    - **Mechanism**: Given the encoder output $\hat{\mathbf{z}}$ and a learnable codebook $\mathcal{C} \in \mathbb{R}^{K \times D}$, the soft posterior is computed as $q_\phi(\mathbf{z}|\mathbf{x}) = \text{Softmax}(-\|\hat{\mathbf{z}} - \mathcal{C}\|_2 / \tau)$, where temperature $\tau = 0.07$. The final latent token is obtained as $\mathbf{z} = q_\phi(\mathbf{z}|\mathbf{x}) \mathcal{C}$, which is a weighted sum of all codewords in the codebook. The KL regularization is defined as $\mathcal{L}_{\text{kl}} = H(q_\phi) - H(\mathbb{E}_{\mathbf{x}} q_\phi)$ (encouraging sharp individual posteriors while ensuring uniform codebook utilization). This process is fully differentiable, eliminating the need for codebook losses or commitment losses.
    - **Design Motivation**: The K-Means assignment in VQ-VAE restricts each token to only map to a single codeword. In contrast, the Soft K-Means in SoftVQ allows each token to leverage the representational power of the entire codebook, maintaining high information density even with extremely few tokens.

2. **1D Learnable Latent Tokens and ViT Architecture**:

    - **Function**: Supports 1D latent token sequences of arbitrary lengths, achieving flexible compression ratios.
    - **Mechanism**: The encoder input is formed by concatenating image patch tokens ($N = HW/P^2$) and $L$ learnable query tokens. Through self-attention, information is aggregated, and only the query token outputs are retained. The decoder uses learnable mask tokens as queries, prepends them to the latent tokens, and reconstructs the image via self-attention. The 1D positional encoding decouples the token count from the image resolution.
    - **Design Motivation**: The token count in traditional 2D grid tokens is rigidly constrained by spatial resolution (e.g., $32 \times 32 = 1024$). 1D query tokens can have freely defined lengths (e.g., 32, 64, 128).

3. **Latent Space Semantic Alignment**:

    - **Function**: Aligns latent tokens with semantic features to improve downstream generation quality.
    - **Mechanism**: Each latent token is replicated $N/L$ times to expand to the same length as the image patches. The projection MLP of these tokens computes a cosine similarity loss $\mathcal{L}_{\text{align}}$ against features from a pre-trained visual encoder (such as DINOv2). Thanks to the full differentiability of SoftVQ, the alignment gradients flow directly to the encoder and the codebook.
    - **Design Motivation**: The Gaussian constraint of KL-VAE and the gradient disconnection of VQ-VAE make it difficult to propagate semantic alignment effectively. The differentiability of SoftVQ fundamentally solves this issue.

### Loss & Training

The total loss is formulated as Total Loss = Reconstruction Loss + Perceptual Loss + Adversarial Loss + $\mathcal{L}_{\text{kl}}$ + $\mathcal{L}_{\text{align}}$. The temperature is set to $\tau = 0.07$, the codebook size to $K = 8192$, and the latent dimension to $D = 32$. ViT-Base/Large serve as the encoder and decoder. Trained on ImageNet for 300 epochs. Downstream generation is trained using DiT/SiT/MAR.

## Key Experimental Results

### Main Results — ImageNet 256×256 Generation

| Tokenizer | Tokens | SiT-XL FID ↓ | SiT-XL Inference Throughput ↑ |
|--------|---------|--------|------------|
| SD-VAE (KL) | 1024 | 2.06 | 1.0× |
| SDXL-VAE (KL) | 1024 | 2.12 | 1.0× |
| TiTok | 128 | 2.77 | 5.3× |
| DC-AE | 256 | 2.32 | 3.2× |
| SoftVQ-VAE | 64 | **1.78** | **18×** |
| SoftVQ-VAE | 32 | 2.33 | **18×** |

### 512×512 Generation

| Tokenizer | Tokens | SiT-XL FID ↓ | Inference Throughput ↑ |
|--------|---------|--------|---------|
| SD-VAE | 4096 | 3.14 | 1.0× |
| SoftVQ-VAE | 64 | **2.21** | **55×** |

### Ablation Study

| Variant | 64 token rFID ↓ | 32 token rFID ↓ |
|------|---------|---------|
| KL-VAE (ViT) | 5.42 | 12.8 |
| VQ-VAE (ViT) | 3.85 | 8.7 |
| SoftVQ-VAE | **1.48** | **2.12** |
| + Semantic Alignment | 1.48 | 2.12 |

### Key Findings

- With only **64 tokens**, SoftVQ-VAE achieves a **1.78 FID** on ImageNet 256×256, surpassing SD-VAE using 1024 tokens (2.06) while boosting inference throughput by 18×.
- The improvement is even more significant on 512×512: 64 tokens reach **2.21 FID** and a **55× throughput increase**, driven by compressing the quadratic complexity of the original 4096 tokens.
- Under extreme compression with 32 tokens (compressing a 256×256 image to 32 scalar tokens), SoftVQ still maintains a 2.33 FID, whereas the FID of KL-VAE skyrockets to 12.8.
- Semantic alignment does not improve reconstruction metrics but significantly enhances generation FID (supporting 2.3× faster training convergence), demonstrating that generation quality depends more heavily on the semantic structure of the latent space than reconstruction accuracy.

## Highlights & Insights

- **The modification from VQ-VAE to SoftVQ-VAE is incredibly simple** (only replacing argmin with softmax), yet the resulting performance leap is massive—representing a classic example of "minimum change for maximum gain."
- **The counter-intuitive finding of "fewer tokens = better generation"** is highly impressive: 64 tokens achieve a lower FID than 1024 tokens, showing that high compression ratios force tokens to learn more compact and semantic representations.
- **Compounding advantages of full differentiability**—there is no need for codebook loss, commitment loss, or the straight-through estimator, while also allowing direct semantic alignment.

## Limitations & Future Work

- 1D tokens are completely decoupled from 2D spatial structures, which may lead to the loss of local spatial relationship information.
- The current verification is limited to ImageNet; complex text-to-image scenarios (such as COCO) remain to be tested.
- The behavior of extreme compression (e.g., 32 tokens) on high-resolution images remains unknown.
- Compatibility with autoregressive generation paradigms (such as LLaMA-based image generation) needs to be explored.

## Related Work & Insights

- **vs TiTok**: TiTok also uses 1D tokens but requires an additional decoder, and has an FID of 2.77 with 128 tokens; SoftVQ-VAE achieves 1.78 with 64 tokens, offering a simpler architecture and better performance.
- **vs DC-AE**: DC-AE achieves an FID of 2.32 with 256 tokens, but further compression leads to a sharp quality drop; the soft quantization of SoftVQ-VAE makes extreme compression feasible.
- **vs REPA**: REPA aligns features in the intermediate layers of the generative model; SoftVQ-VAE performs alignment directly in the tokenizer's latent space, which is equivalent to aligning in the input space of the generative model and therefore more fundamental.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Simple and elegant idea of changing VQ's hard assignment to soft assignment, yielding highly significant results.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive comparison across three generative models (DiT/SiT/MAR), multiple resolutions, and various tokenizers.
- Writing Quality: ⭐⭐⭐⭐ Natural derivation starting from a unified perspective of KL-VAE/VQ-VAE.
- Value: ⭐⭐⭐⭐⭐ Improves image generation efficiency by an order of magnitude, exerting a profound impact on the entire visual generation field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation](tokenflow_unified_image_tokenizer_for_multimodal_understanding_and_generation.md)
- [\[CVPR 2025\] Divot: Diffusion Powers Video Tokenizer for Comprehension and Generation](divot_diffusion_powers_video_tokenizer_for_comprehension_and_generation.md)
- [\[ICCV 2025\] Spectral Image Tokenizer](../../ICCV2025/image_generation/spectral_image_tokenizer.md)
- [\[ICCV 2025\] DC-AR: Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer](../../ICCV2025/image_generation/dc-ar_efficient_masked_autoregressive_image_generation_with_deep_compression_hyb.md)
- [\[CVPR 2025\] EvoTok: A Unified Image Tokenizer via Residual Latent Evolution for Visual Understanding and Generation](evotok_a_unified_image_tokenizer_via_residual_latent_evolution_for_visual_unders.md)

</div>

<!-- RELATED:END -->

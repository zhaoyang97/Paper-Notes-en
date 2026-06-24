---
title: >-
  [Paper Note] DLF: Extreme Image Compression with Dual-generative Latent Fusion
description: >-
  [ICCV 2025][Model Compression][extreme low-bitrate image compression] This paper proposes the Dual-generative Latent Fusion (DLF) framework, which decomposes the image latent space into semantic and detail branches for separate compression, and eliminates inter-branch redundancy via a cross-branch interactive design. At extreme low bitrates (<0.01 bpp), DLF achieves state-of-the-art reconstruction quality with BD-Rate savings of up to 67.82% over MS-ILLM…
tags:
  - "ICCV 2025"
  - "Model Compression"
  - "extreme low-bitrate image compression"
  - "generative codec"
  - "dual-branch encoding"
  - "vector quantization"
  - "semantic-detail decomposition"
date: 2026-05-08
content_hash: f5c171ce2610459f
---

# DLF: Extreme Image Compression with Dual-generative Latent Fusion

**Conference**: ICCV 2025
**arXiv**: [2503.01428](https://arxiv.org/abs/2503.01428)  
**Code**: [dlfcodec.github.io](https://dlfcodec.github.io/)  
**Area**: Model Compression / Image Compression
**Keywords**: extreme low-bitrate image compression, generative codec, dual-branch encoding, vector quantization, semantic-detail decomposition

## TL;DR

This paper proposes the Dual-generative Latent Fusion (DLF) framework, which decomposes the image latent space into semantic and detail branches for separate compression, and eliminates inter-branch redundancy via a cross-branch interactive design. At extreme low bitrates (<0.01 bpp), DLF achieves state-of-the-art reconstruction quality with BD-Rate savings of up to 67.82% over MS-ILLM, while decoding significantly faster than diffusion-based approaches.

## Background & Motivation

### State of the Field

Extreme low-bitrate image compression (e.g., below 0.01 bpp) is a central challenge in the image compression community. Traditional codecs (VVC) and MSE-optimized neural codecs suffer from severe blurring at low bitrates. Recent methods based on generative tokenizers (e.g., VQGAN) achieve extremely high compression ratios by encoding images into a small number of token indices, yet face a fundamental tension.

### Limitations of Prior Work

**Capacity bottleneck of a single codebook**: Generative tokenizers such as VQGAN learn compact codebooks by clustering semantics across an entire dataset, prioritizing dataset-level common content. As codebook size shrinks, instance-specific details (e.g., precise geometric features) are severely distorted.

**Limitations of diffusion-based approaches**: Methods such as PerCo and DiffEIC improve perceptual realism but fall short on fidelity and suffer from extremely slow decoding (over 4 seconds).

**Redundancy in dual-branch designs**: The existing dual-branch method HybridFlow uses independent encoders, resulting in substantial information redundancy between the two bitstreams.

### Root Cause

Can the latent space be flexibly decomposed into a "semantic" component and a "detail" component, each compressed with the most appropriate strategy, while cross-branch interaction eliminates redundancy?

## Method

### Overall Architecture

DLF adopts a dual-branch encoding architecture. An input image $X \in \mathbb{R}^{3 \times H \times W}$ is first projected via patch embedding to $Emb(X) \in \mathbb{R}^{C \times h \times w}$ ($h=H/16, w=W/16$), then fed into a semantic branch and a detail branch respectively. The semantic branch uses a 1-D tokenizer to cluster high-level semantics into compact tokens, while the detail branch uses scalar quantization (SQ) to encode perceptually important details. The two branches interact through multi-layer Interactive Transform (IT) modules. After decoding, semantic and detail features are fused via a latent adaptor and passed to a pretrained VQGAN decoder to generate the reconstructed image.

### Key Designs

#### 1. **Semantic Branch (1-D Tokenizer)**
- **Function**: Compresses high-level image semantics into a minimal number of 1-D tokens.
- **Mechanism**: $Emb(X)$ is partitioned into $16 \times 16$ windows. Within each window, 256 2-D image tokens and 32 additional 1-D tokens are jointly fed into a ViT. Through cascaded attention, the 1-D tokens efficiently aggregate key semantic information from the image tokens, yielding $y_s \in \mathbb{R}^{N \times C \times 32}$, where $N = \frac{h \times w}{16 \times 16}$.
- **Quantization**: Vector quantization (VQ) with fixed-length coding for codebook index transmission.
- **Design Motivation**: Compared to manually predefined mask-based token reduction strategies, learning-based semantic compression is more flexible and adapts better to diverse images.

#### 2. **Detail Branch (SQ + Adaptive Bit Allocation)**
- **Function**: Captures instance-level detail information that the VQ codebook cannot represent.
- **Mechanism**: Shifted window attention combined with ConvNeXT extracts local and global details, downsampled to $y_d \in \mathbb{R}^{C \times h/2 \times w/2}$. SQ with a quadtree-based entropy model is used for arithmetic coding.
- **Quantization formula**: $\hat{y}_s = VQ(y_s), \quad \hat{y}_d = Q(y_d)$
- **Design Motivation**: SQ provides a far larger quantization space than VQ, enabling learnable, spatially adaptive quantization step sizes—allocating more bits to distinctive object contours and fewer bits to common content well-represented by the semantic branch.

#### 3. **Cross-Branch Interaction (Interactive Transform, IT)**
- **Function**: Eliminates information redundancy between the semantic and detail bitstreams.
- **Mechanism**: Detail features $f_d$ are rearranged according to the same windowing strategy as the semantic branch into $\tilde{f_d} \in \mathbb{R}^{N \times C \times 256}$, then jointly processed with semantic features through multi-head self-attention. The processed detail features are subsequently restored to their original shape.
- **Design Motivation**: (1) Self-attention dynamically redistributes semantic and detail information across branches, reducing redundancy while allowing detail information to correct generation errors in the semantic branch. (2) Cross-window perceptual capability is introduced to the semantic branch—detail features contribute global information, expanding the semantic branch's receptive field.

### Loss & Training

A two-stage progressive training scheme is adopted:

1. **Stage 1 (Latent Space Alignment)**: A rate-distortion loss is applied in latent space, supervising the fused feature $\hat{h}$ with reference features $\tilde{h}$ generated by a pretrained VQGAN encoder. Training uses 256×256 patches, batch size 8, fixed $\lambda = 24.0$.
2. **Stage 2 (End-to-End Fine-tuning)**: The full model is fine-tuned with generative losses in pixel space. Training uses 512×512 patches, batch size 4, with $\lambda \in \{5.8, 8.5, 16.0, 28.0\}$ to cover different bitrates.

## Key Experimental Results

### Main Results

| Method | Dataset | LPIPS BD-Rate | DISTS BD-Rate | Note |
|--------|---------|--------------|--------------|------|
| DLF | Kodak | **-43.05%** | **-67.82%** | Ours |
| GLC | Kodak | -17.24% | -33.41% | Tokenizer-based |
| DiffEIC | Kodak | +66.05% | +14.67% | Diffusion-based |
| PerCo | Kodak | +101.74% | -4.02% | Diffusion-based |
| HybridFlow | Kodak | +65.30% | — | Dual-branch |
| MS-ILLM | Kodak | 0.00% (anchor) | 0.00% (anchor) | Anchor |
| DLF | CLIC2020 | **-27.93%** | **-53.55%** | Ours |

### Ablation Study

| Configuration | Kodak LPIPS | Kodak DISTS | CLIC LPIPS | CLIC DISTS | Note |
|---------------|------------|------------|-----------|-----------|------|
| w/ SQ detail (DLF) | 0.0% | 0.0% | 0.0% | 0.0% | Full model (anchor) |
| w/o detail | +17.5% | +20.2% | +47.9% | +47.6% | Remove detail branch |
| w/o interactive | +64.1% | +73.6% | +68.8% | +61.8% | Remove IT modules |
| w/ VQ detail | +18.3% | +40.7% | +27.3% | +58.1% | Replace SQ with VQ in detail branch |

### Complexity Analysis

| Model | Encoding Time | Decoding Time | DISTS BD-Rate |
|-------|--------------|--------------|--------------|
| MS-ILLM | 0.064s | 0.070s | 0.00% |
| PerCo | 0.461s | 2.443s | -4.02% |
| DiffEIC | 0.152s | **4.093s** | +14.67% |
| DLF | 0.178s | **0.252s** | **-67.82%** |

### Key Findings

- Removing the cross-branch interaction (IT modules) causes the most severe performance degradation (>60% BD-Rate loss), demonstrating that independent dual branches suffer from substantial redundancy.
- Using SQ in the detail branch significantly outperforms VQ, confirming the importance of a large quantization space for representing diverse details.
- DLF decodes 10–16× faster than diffusion-based methods while achieving significantly better fidelity.
- On CLIC2020 at 768×768, DLF substantially outperforms PerCo in FID, demonstrating its advantage on high-quality datasets.

## Highlights & Insights

1. **Semantic-detail decomposition**: Decoupling "dataset-level common content" and "instance-level diversity" into two separate bitstreams, each compressed with the most suitable quantization strategy, is an elegant and effective design principle.
2. **Critical role of cross-branch interaction**: Ablation results clearly show that independent dual branches without interaction perform even worse than a single branch; the interactive design is the key to the success of the dual-branch framework.
3. **Deeper insight into SQ vs. VQ**: VQ's limited codebook is naturally suited for encoding clustered common semantics, while SQ's large quantization space accommodates diverse details. A hybrid quantization strategy is therefore optimal.
4. **Strong competition against diffusion-based methods**: DLF achieves comparable perceptual realism while substantially improving fidelity and decoding speed.

## Limitations & Future Work

1. **Not yet real-time**: Encoding at 0.178s and decoding at 0.252s remain insufficient for real-time applications.
2. **High training cost**: Two-stage training is required, with dependency on a pretrained VQGAN tokenizer.
3. **Limited bitrate control granularity**: Bitrate is controlled via $\lambda$ adjustment, offering limited flexibility.
4. **Evaluation limited to perceptual metrics**: Downstream task evaluation (e.g., detection, segmentation) is absent.

## Related Work & Insights

- HybridFlow's independent dual-branch design motivates the necessity of redundancy elimination; DLF achieves 2.6× higher compression ratio through the IT modules.
- The 1-D tokenizer provides a foundation for compact semantic compression, upon which DLF adds detail encoding.
- Comparison with diffusion-based methods demonstrates the advantage of the tokenizer-based approach in terms of speed and fidelity.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The semantic-detail decomposition combined with cross-branch interaction is an elegant design, though the dual-branch concept itself is not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-dataset evaluation, comprehensive ablation, and complexity analysis provide a thorough empirical study.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Motivation is clearly articulated, deriving the proposed method from the limitations of VQ in a logically coherent manner.
- **Value**: ⭐⭐⭐⭐ — Represents a significant advance in extreme low-bitrate compression and offers important reference value for the tokenizer-based compression direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Generative Video Compression with One-Dimensional Latent Representation](../../CVPR2026/model_compression/generative_video_compression_with_one-dimensional_latent_representation.md)
- [\[ICCV 2025\] Learned Image Compression with Hierarchical Progressive Context Modeling](learned_image_compression_with_hierarchical_progressive_context_modeling.md)
- [\[CVPR 2026\] CADC: Content Adaptive Diffusion-Based Generative Image Compression](../../CVPR2026/model_compression/cadc_content_adaptive_diffusion-based_generative_image_compression.md)
- [\[ICML 2025\] Strategic Fusion Optimizes Transformer Compression](../../ICML2025/model_compression/strategic_fusion_optimizes_transformer_compression.md)
- [\[ICCV 2025\] Fuse Before Transfer: Knowledge Fusion for Heterogeneous Distillation](fuse_before_transfer_knowledge_fusion_for_heterogeneous_distillation.md)

</div>

<!-- RELATED:END -->

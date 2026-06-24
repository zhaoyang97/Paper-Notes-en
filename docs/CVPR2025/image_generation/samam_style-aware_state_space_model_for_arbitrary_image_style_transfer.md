---
title: >-
  [Paper Note] SaMam: Style-aware State Space Model for Arbitrary Image Style Transfer
description: >-
  [CVPR 2025][Image Generation][Style Transfer] SaMam is proposed as the first arbitrary image style transfer framework based on the Mamba state space model. By predicting SSM weight parameters from style embeddings via a style-aware S7 block, and combining this with zigzag scanning and local enhancement mechanisms, it achieves the optimal balance between stylization quality and efficiency.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Style Transfer"
  - "State Space Model"
  - "Mamba"
  - "Global Receptive Field"
  - "Efficient Inference"
date: 2026-05-08
content_hash: 9d7861a3a5e941a5
---

# SaMam: Style-aware State Space Model for Arbitrary Image Style Transfer

**Conference**: CVPR 2025  
**arXiv**: [2503.15934](https://arxiv.org/abs/2503.15934)  
**Code**: None  
**Area**: Image Generation / Style Transfer  
**Keywords**: Style Transfer, State Space Model, Mamba, Global Receptive Field, Efficient Inference

## TL;DR

SaMam is proposed as the first arbitrary image style transfer framework based on the Mamba state space model. By predicting SSM weight parameters from style embeddings via a style-aware S7 block, and combining this with zigzag scanning and local enhancement mechanisms, it achieves the optimal balance between stylization quality and efficiency.

## Background & Motivation

- A global effective receptive field is crucial for style transfer: (1) larger receptive fields can capture style patterns better; (2) more pixels participate in the style transformation of anchor pixels.
- CNN-based methods expand the receptive field by stacking convolutional layers, incurring high computational costs, while Transformer-based methods achieve a global receptive field but suffer from quadratic computational complexity.
- Although diffusion models provide high generation quality, they require numerous iteration steps, leaving the efficiency issue essentially unresolved.
- The trade-off between global receptive field and computational efficiency in style transfer has not been fundamentally resolved.
- Mamba state space models model long-range dependencies with linear complexity, offering a potential solution to this trade-off.
- However, existing SSMs suffer from local pixel forgetting (caused by 1D flattening which places spatially adjacent pixels far apart in the sequence), channel redundancy, and spatial discontinuity issues.
- The parameters $\mathbf{A}$ and $\mathbf{D}$ of standard SSMs originate from a fixed embedding space, making them unable to dynamically adjust according to different styles.

## Method

### Overall Architecture

SaMam consists of a style Mamba encoder, a content Mamba encoder, and a style-aware Mamba decoder. The encoders project the content image $\mathbf{I_c}$ and the style image $\mathbf{I_s}$ into content features $\mathbf{E_c}$ and style embeddings $\mathbf{E_s}$, respectively. The style embeddings act as conditional information to adapt the decoder parameters, ultimately generating the stylized image $\mathbf{I_{cs}}$. Both encoder and decoder are built upon VMamba's SS2D blocks, enhanced with local reinforcement and zigzag scanning.

### Key Designs

**1. Style-aware S6 Block**
- **Function**: Inject style information into the state update process of SSM, enabling the model to dynamically adjust its behavior according to different styles.
- **Mechanism**: Unlike the standard S6 block, the S7 block predicts the key parameters $\mathbf{A}$ and $\mathbf{D}$ of the SSM from the style embedding $\mathbf{E_s}$: $\mathbf{A}, \mathbf{D} = \text{Embedder}(\mathbf{E_s})$. After discretization, $\mathbf{A}$ is expanded into a global convolution kernel, while $\mathbf{D}$ serves as a channel-level scaling factor. The style-dependency of both parameters allows the SSM to simultaneously account for both content and style during hidden state updates.
- **Design Motivation**: (1) The standard S6 block updates hidden states solely based on content, neglecting style influence; (2) $\mathbf{A}$ possesses selective capability through discretization, and predicting it from style embeddings enables style-aware selectivity; (3) the style-dependent global convolution kernel achieves style adaptation while maintaining parallel computing efficiency.

**2. Zigzag Scan**
- **Function**: Maintain the spatial and semantic continuity of 2D image token sequences.
- **Mechanism**: Starting from 4 vertices, the image is traversed using a zigzag path instead of standard row-by-row or column-by-column linear scanning. The first clockwise column (or row) serves as the starting scanline. This ensures that tokens of adjacent rows/columns remain close in the sequence.
- **Design Motivation**: Traditional raster scanning causes spatial discontinuity when switching lines, leading to abrupt changes in the SSM decay parameter $\bar{\mathbf{A}}$ between adjacent tokens, which results in semantic discontinuity and unnatural stylized textures. Zigzag scanning eliminates line-switch jumps, maintaining smooth decay transitions.

**3. Style-Aware Modules (SAIN + SConv + SCM)**
- **Function**: Integrate style information into content feature processing at multiple levels.
- **Mechanism**: (1) SAIN (Style-Aware Instance Normalization): predicts mean $\gamma$ and variance $\beta$ from $\mathbf{E_s}$ to perform feature-level normalization, transferring global style attributes; (2) SConv (Style-Aware Convolution): generates a depthwise convolution kernel $K \in \mathbb{R}^{C \times 1 \times k_w \times k_h}$ from $\mathbf{E_s}$ to preserve the local geometric structure of the style image; (3) SCM (Style-Aware Channel Modulation): generates sigmoid modulation coefficients $v \in \mathbb{R}^C$ from $\mathbf{E_s}$ for channel-level feature adaptation. The embedders for SAIN and SCM are initialized to output zero vectors, making SAVSSM initialize as an identity function.
- **Design Motivation**: Style transfer requires style injection at both the level of global attributes (color tone, contrast) and local structures (brush strokes, textures). The three modules cover instance normalization (global), depthwise convolution (local spatial), and channel modulation (feature selection), respectively.

### Loss & Training

Standard training losses for style transfer are employed, including content loss ($\mathcal{L}_c$, to preserve content structure), style loss ($\mathcal{L}_s$, to match Gram matrix statistics), and perceptual loss.

## Key Experimental Results

### Main Results: Quantitative Comparison

| Method | LPIPS↓ | FID↓ | ArtFID↓ | Type |
|------|--------|------|---------|------|
| AesPA | 0.405 | 20.24 | 29.84 | CNN |
| S2WAT | 0.426 | 23.43 | 34.83 | Transformer |
| StyleID | 0.480 | 24.49 | 37.73 | Diffusion |
| **SaMam** | **0.388** | **17.95** | **26.31** | **Mamba** |

*SaMam consistently outperforms all types of methods across all three key metrics.*

### Efficiency Comparison

| Method | Inference Time (ms) | MACs (G) |
|------|-------------|---------|
| StyTr2 (Transformer) | ~150 | ~80 |
| AesPA (CNN) | ~50 | ~40 |
| **SaMam** | **~35** | **~25** |

*SaMam achieves the optimal efficiency in both inference speed and computation.*

### Key Findings

- The Mamba architecture achieves superior style transfer quality compared to CNN and Transformer methods with linear complexity.
- Zigzag scanning effectively reduces unnatural artifacts in stylized textures compared to linear scanning.
- The Local Enhancement (LoE) module compensates for the loss of local information caused by the SSM flattening operation.
- SAIN (Instance Normalization) is more suitable for style transfer tasks than standard Layer Normalization.
- The style-aware parameter prediction in the S7 block is more effective than using fixed parameters combined with late fusion.

## Highlights & Insights

1. **Style-Dependent SSM Parameters**: Transforming $\mathbf{A}$ and $\mathbf{D}$ from fixed parameters to style-conditional parameters, elegantly injecting style information into the core mechanism of state updates.
2. **Systematic Solution to Spatial Continuity**: Zigzag scanning fundamentally addresses the spatial discontinuity problem when SSMs are applied to 2D images.
3. **Optimal Efficiency-Quality Trade-off**: Demonstrating for the first time the linear complexity advantages of the Mamba architecture in style transfer.

## Limitations & Future Work

- Four-directional scanning is still required, resulting in four times the computational cost of single-direction scanning.
- Generalization capability to extreme styles (e.g., highly abstract artworks) remains to be explored.
- The design of the style embedder is relatively simple; more sophisticated style modeling could potentially further improve quality.
- The training stability of Mamba models in vision tasks still warrants attention.

## Related Work & Insights

- Compared to Transformer-based methods like StyTr2, SaMam achieves better quality with linear complexity.
- The idea of style-aware parameter prediction (S7 block) can be generalized to the application of SSMs in other conditional generation tasks.
- The zigzag scanning strategy provides valuable insights for all image-level SSM applications.

## Rating

⭐⭐⭐⭐ — This work systematically applies Mamba to style transfer for the first time. The S7 block is elegantly designed, and zigzag scanning effectively addresses the spatial continuity problem. The experimental results are convincing in terms of both quality and efficiency, leading across all quantitative metrics. However, generalization capabilities on more diverse style datasets still need further validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] HSI: A Holistic Style Injector for Arbitrary Style Transfer](hsi_a_holistic_style_injector_for_arbitrary_style_transfer.md)
- [\[CVPR 2025\] OmniStyle: Filtering High Quality Style Transfer Data at Scale](omnistyle_filtering_high_quality_style_transfer_data_at_scale.md)
- [\[CVPR 2025\] StyleStudio: Text-Driven Style Transfer with Selective Control of Style Elements](stylestudio_text-driven_style_transfer_with_selective_control_of_style_elements.md)
- [\[CVPR 2025\] SCSA: A Plug-and-Play Semantic Continuous-Sparse Attention for Arbitrary Semantic Style Transfer](scsa_a_plug-and-play_semantic_continuous-sparse_attention_for_arbitrary_semantic.md)
- [\[CVPR 2026\] StyleGallery: Training-free and Semantic-aware Personalized Style Transfer from Arbitrary Image References](../../CVPR2026/image_generation/stylegallery_training-free_and_semantic-aware_personalized_style_transfer_from_a.md)

</div>

<!-- RELATED:END -->

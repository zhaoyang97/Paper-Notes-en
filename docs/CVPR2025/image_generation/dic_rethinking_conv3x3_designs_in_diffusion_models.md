---
title: >-
  [Paper Note] DiC: Rethinking Conv3x3 Designs in Diffusion Models
description: >-
  [CVPR 2025][Image Generation][Pure Convolutional Diffusion Models] This paper revisits the potential of 3x3 convolutions in diffusion models. By introducing a series of architectural improvements (hourglass U-Net and sparse skip connections) and conditioning enhancements (stage-specific embeddings, mid-block injection, and condition gating), the authors build a pure 3x3 convolutional diffusion model, DiC. It outperforms DiT of comparable scale on ImageNet generation while ach…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Pure Convolutional Diffusion Models"
  - "3x3 Convolution"
  - "U-Net Architecture"
  - "Sparse Skip Connections"
  - "Condition Injection"
date: 2026-05-08
content_hash: 437c0e2a7536879c
---

# DiC: Rethinking Conv3x3 Designs in Diffusion Models

**Conference**: CVPR 2025  
**arXiv**: [2501.00603](https://arxiv.org/abs/2501.00603)  
**Code**: [GitHub](https://github.com/YuchuanTian/DiC)  
**Area**: Diffusion Models / Model Architecture  
**Keywords**: Pure Convolutional Diffusion Models, 3x3 Convolution, U-Net Architecture, Sparse Skip Connections, Condition Injection

## TL;DR
This paper revisits the potential of 3x3 convolutions in diffusion models. By introducing a series of architectural improvements (hourglass U-Net and sparse skip connections) and conditioning enhancements (stage-specific embeddings, mid-block injection, and condition gating), the authors build a pure 3x3 convolutional diffusion model, DiC. It outperforms DiT of comparable scale on ImageNet generation while achieving significantly faster inference.

## Background & Motivation

**Background**: Diffusion model architectures have evolved from hybrid CNN-attention designs (such as ADM) to pure Transformers (such as DiT, PixArt). While the latter show excellent scalability and performance, the computational overhead of self-attention is extremely high.

**Limitations of Prior Work**: Transformer-based diffusion models suffer from slow inference, making them unsuitable for real-time or resource-constrained scenarios. Existing acceleration schemes (such as downsampling tokens, linear attention, or substituting with SSMs) either still operate within the attention paradigm or yield suboptimal latency.

**Key Challenge**: Although 3x3 convolutions are extremely fast (benefiting from Winograd acceleration and hardware friendliness), their receptive fields are naturally limited. Directly placing pure 3x3 convolutions into existing scalable architectures (like isotropic DiT) yields performance far inferior to Transformers.

**Goal**: To explore whether pure 3x3 convolutions can achieve generative quality competitive with Transformer-based diffusion models through meticulous design, while retaining their speed advantages.

**Key Insight**: The receptive field limitation of 3x3 convolutions can be naturally alleviated via encoder-decoder downsampling—specifically, a 3x3 kernel on downsampled feature maps covers a 6x6 or even 12x12 region on the original image.

**Core Idea**: U-Net hourglass architecture (to expand receptive fields) + sparse skip connections (to reduce redundancy) + stage-specific condition embeddings (to adapt to different feature spaces across resolutions).

## Method

### Overall Architecture
The basic block consists of a two-layer 3x3 convolution with residual connections (eliminating the self-attention layer found in traditional U-Net blocks). An encoder-decoder hourglass structure is adopted to expand the receptive field through downsampling and upsampling. Additionally, sparse skip connections and multiple conditioning enhancements are integrated.

### Key Designs

1. **Hourglass Architecture + Sparse Skip Connections**:

    - **Function**: To provide sufficient receptive fields for pure 3x3 convolutions while maintaining scalability.
    - **Mechanism**: Experimental results compare isotropic (FID 29.31), isotropic+skip (15.07), U-Net hourglass (14.65), and U-Net+sparse skip (11.49). The hourglass architecture naturally expands the receptive field through downsampling. Traditional block-wise dense skip connections introduce excessive overhead when stacking a large number of convolutions, which is addressed here by placing skip connections only every few blocks.
    - **Design Motivation**: In a pure isotropic architecture, each 3x3 convolutional layer only expands the receptive field by 1 pixel, requiring an extremely deep network to achieve a global receptive field. The hierarchical structure of the hourglass bottleneck efficiently resolves this issue.

2. **Stage-Specific Condition Embeddings**:

    - **Function**: To adapt to different feature spaces across various stages of the encoder-decoder.
    - **Mechanism**: In conventional diffusion models, all stages share the same set of conditioning embedding lookup tables. In the hourglass architecture, different stages have varying channel dimensions representing different levels of features. DiC trains separate, stage-specific embedding tables (matching the channel count of each stage), adding only 2% parameters.
    - **Design Motivation**: Since the bottom of the encoder processes high-level semantics and the top processes local details, a single shared embedding is suboptimal across all levels.

3. **Mid-Block Condition Injection + Condition Gating**:

    - **Function**: To optimize the injection position and mechanism of conditioning signals.
    - **Mechanism**: Instead of the LayerNorm at the beginning of the block, conditions are injected into the second convolutional layer (the middle position) of each basic block. A gating vector adapted from AdaLN in DiT is used to scale features channel-wise. All activation functions are converted from SiLU to GELU.
    - **Design Motivation**: Experimental results verify that mid-block injection outperforms injection at the block's beginning, and gating provides finer-grained conditioning control.

### Loss & Training
Standard diffusion denoising loss (using the same hyperparameter settings as DiT to ensure a fair comparison).

## Key Experimental Results

### Main Results (ImageNet 256×256, Class-Conditional Generation)

| Model | Parameters | FLOPs | FID↓ | IS↑ | Throughput |
|------|--------|-------|------|-----|--------|
| DiT-XL | 675M | 119G | 2.27 | 278 | Baseline |
| **DiC-XL** | **708M** | **119G** | **2.10** | **286** | **~2× Faster** |

### Ablation Study (200K iterations)

| Configuration | FID↓ | Description |
|------|------|------|
| Isotropic Conv3x3 | 29.31 | Extremely poor due to insufficient receptive fields |
| U-Net Hourglass | 14.65 | Significant improvement |
| + Sparse Skip | 11.49 | Further improvement |
| + Stage-Spec. Emb. | 10.07 | Conditioning improvement |
| + Mid-Block Injection | 8.80 | Optimized injection position |
| + Gating | 6.54 | Gating yields significant gains |
| + GELU (DiC) | **6.26** | Final model |
| DiT-XL under the same conditions | 12.96 | DiC leads by a large margin |

### Key Findings
- Pure 3x3 convolutional diffusion models can outperform DiT of the same scale through architectural and conditioning improvements.
- The hourglass architecture is crucial for 3x3 convolutions—an isotropic architecture is completely unfeasible.
- Sparse skip connections outperform dense skip connections by reducing the overhead of redundant concatenations.
- Stage-specific embeddings add only 2% parameters but yield significant improvements.
- DiC achieves an approximate 2x speedup in inference throughput, benefiting from Winograd acceleration and high parallelism.

## Highlights & Insights
- It brings 3x3 convolutions—often overlooked in the current "pure Transformer" trend—back to the forefront of diffusion models. The finding that they can outperform Transformers is highly counter-intuitive and inspiring.
- A systematic roadmap clearly highlights the contribution of each modification, presenting a very translucent analysis.
- The logical progression from "receptive field is the core bottleneck of 3x3 convolutions" to "addressing it naturally through hourglass downsampling" is elegant and powerful.

## Limitations & Future Work
- Currently evaluated only on ImageNet 256×256; validation on higher resolutions and text-to-image generation is yet to be established.
- Pure convolutional models lack the global modeling capability of attention for long-range dependencies, which might limit performance on tasks that require global coherence.
- The stride of sparse skip connections is a hyperparameter and may require tuning for different model scales.
- Hybridizing the architecture with a small number of attention layers could be explored to strengthen global modeling while retaining the speed advantages.

## Related Work & Insights
- **vs DiT**: DiT achieves scalability with full attention, whereas DiC demonstrates that 3x3 convolutions combined with proper architectural design can be equally scalable and much faster.
- **vs U-ViT**: U-ViT adds skip connections on an isotropic architecture while still relying on attention; DiC utilizes an hourglass structure with sparse skip connections to completely eliminate attention.
- **vs ConvNeXt**: ConvNeXt proved that CNNs can match ViTs in classification; DiC extends a similar design philosophy to generative models.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Moving against the mainstream trend, it successfully proves the feasibility of pure convolutional diffusion models.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Features a comprehensive ablation roadmap and cross-scale comparisons.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Highly logical, with clear design motivations for every step.
- **Value**: ⭐⭐⭐⭐ Offers a compelling alternative for diffusion model architecture design with strong practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Rethinking Direct Preference Optimization in Diffusion Models](../../NeurIPS2025/image_generation/rethinking_direct_preference_optimization_in_diffusion_models.md)
- [\[AAAI 2026\] Rethinking Flow and Diffusion Bridge Models for Speech Enhancement](../../AAAI2026/image_generation/rethinking_flow_and_diffusion_bridge_models_for_speech_enhancement.md)
- [\[ICCV 2025\] Rethinking Cross-Modal Interaction in Multimodal Diffusion Transformers](../../ICCV2025/image_generation/rethinking_cross-modal_interaction_in_multimodal_diffusion_transformers.md)
- [\[CVPR 2025\] Decentralized Diffusion Models](decentralized_diffusion_models.md)
- [\[CVPR 2026\] Transition Models: Rethinking the Generative Learning Objective](../../CVPR2026/image_generation/transition_models_rethinking_the_generative_learning_objective.md)

</div>

<!-- RELATED:END -->

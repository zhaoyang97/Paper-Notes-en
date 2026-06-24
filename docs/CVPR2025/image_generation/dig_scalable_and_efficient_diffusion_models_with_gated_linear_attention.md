---
title: >-
  [Paper Note] DiG: Scalable and Efficient Diffusion Models with Gated Linear Attention
description: >-
  [CVPR 2025][Image Generation][Diffusion Models] DiG introduces Gated Linear Attention (GLA) into the backbone of diffusion models, addressing the unidirectional modeling and lack of local awareness in GLA through a Spatial Redirection and Enhancement Module (SREM). This achieves performance surpassing DiT on the ImageNet 256×256 generation task, while offering a 2.5x speedup and a 75.7% reduction in GPU memory at a resolution of 1792.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Diffusion Models"
  - "Gated Linear Attention"
  - "Sub-quadratic Complexity"
  - "Efficient Generation"
  - "ImageNet"
date: 2026-05-08
content_hash: 24fc0b597c800e95
---

# DiG: Scalable and Efficient Diffusion Models with Gated Linear Attention

**Conference**: CVPR 2025  
**arXiv**: [2405.18428](https://arxiv.org/abs/2405.18428)  
**Code**: [https://github.com/hustvl/DiG](https://github.com/hustvl/DiG)  
**Area**: Diffusion Models  
**Keywords**: Diffusion Models, Gated Linear Attention, Sub-quadratic Complexity, Efficient Generation, ImageNet

## TL;DR

DiG introduces Gated Linear Attention (GLA) into the backbone of diffusion models, addressing the unidirectional modeling and lack of local awareness in GLA through a Spatial Redirection and Enhancement Module (SREM). This achieves performance surpassing DiT on the ImageNet 256×256 generation task, while offering a 2.5x speedup and a 75.7% reduction in GPU memory at a resolution of 1792.

## Background & Motivation

**Background**: Diffusion models have become the mainstream paradigm for visual generation, with their backbone evolving from U-Net to Vision Transformer (ViT/DiT). DiT has been adopted by leading works such as Sora and Stable Diffusion 3 due to the scalability of Transformers. However, the self-attention mechanism in DiT has a computational complexity of $O(T^2)$ (where $T$ is the sequence length), posing severe efficiency bottlenecks when processing high-resolution images (long sequences).

**Limitations of Prior Work**: Existing alternatives with sub-quadratic time complexity are mainly based on Mamba (SSM), such as DiS and DiffuSSM. However, Mamba-like methods offer limited efficiency gains as the model size increases—their complex block designs and inability to efficiently utilize GPU tensor cores lead to suboptimal actual throughput for large-scale models. The speed of DiS-XL/2 at 1024 resolution is only about 60% of DiT-XL/2.

**Key Challenge**: Diffusion models need to process increasingly longer token sequences (high-resolution image and video generation), requiring sub-quadratic complexity backbones to break through efficiency bottlenecks. However, existing alternatives either underperform compared to Transformers or fail to deliver satisfactory practical efficiency.

**Goal**: To introduce Gated Linear Attention (GLA) Transformer into diffusion models, achieving truly efficient inference with sub-quadratic complexity while maintaining the scalability and generation quality of DiT.

**Key Insight**: GLA is an efficient linear attention Transformer variant that has demonstrated excellent performance in NLP. However, directly applying GLA to 2D image generation poses two challenges: (1) GLA performs unidirectional (causal) sequence modeling, whereas images require bidirectional context; (2) GLA lacks local spatial awareness.

**Core Idea**: To design a lightweight Spatial Redirection and Enhancement Module (SREM) to achieve global context modeling by alternating scanning directions layer by layer, assisted by depthwise separable convolutions initialized with identity weights to provide local perception, seamlessly adapting GLA to 2D diffusion backbones.

## Method

### Overall Architecture

DiG adopts the pipeline of Latent Diffusion: the input image is encoded by a VAE encoder into a $32 \times 32 \times 4$ latent representation, converted into a token sequence through a patchify layer (with a sequence length of 256 for a patch size of 2), and processed through $N$ layers of DiG Blocks after adding positional and conditional embeddings (timestep + class label). Finally, a linear projection head outputs the predicted noise and covariance. The overall architecture remains faithful to the design philosophy of DiT, simply replacing self-attention with GLA.

### Key Designs

1. **Spatial Redirection and Enhancement Module (SREM)**:

    - **Function**: To overcome the limitation of unidirectional modeling in GLA, achieving global context awareness and local spatial information capture for 2D images.
    - **Mechanism**: SREM consists of two components: **layer-wise scanning direction control** and **depthwise separable convolution (DWConv2d)**. Regarding the scanning direction, each DiG Block only processes one direction of GLA scanning. At the end of the block, efficient matrix operations (transposing the 2D token matrix + flipping the sequence) change the scanning direction for the next block. Four basic scanning patterns (left-to-right, right-to-left, top-to-bottom, bottom-to-top) are used alternately to form a crisscross coverage. For local awareness, a $3 \times 3$ DWConv2d layer is inserted in SREM, using **identity initialization** (only the center weight is 1, and the surrounding weights are 0), adding almost no parameters.
    - **Design Motivation**: Simple bidirectional scanning (FID 69.28) is underperforming compared to crisscross four-directional scanning (FID 62.06). Meanwhile, the identity initialization of DWConv2d addresses the slow convergence caused by standard random initialization, as the model can start with an identity map at the early stage and progressively learn local information.

2. **DiG Block Design**:

    - **Function**: Serving as the fundamental computational unit of the DiG network, organically combining GLA, FFN, and SREM.
    - **Mechanism**: The forward process of each DiG Block is: (1) adding the timestep embedding $\mathbf{t}$ and class embedding $\mathbf{y}$ before regressing the scale/shift parameters $\alpha, \beta, \gamma$ of adaLN through an MLP; (2) applying adaLN normalization to the input and feeding it to GLA to compute global attention; (3) applying adaLN normalization again and feeding it through the FFN; (4) reshaping the sequence to 2D and passing it through DWConv2d to capture local information; (5) finally transforming the scanning direction. The entire block follows the adaLN-Zero conditioning of DiT.
    - **Design Motivation**: To maintain an architectural design highly consistent with DiT, enabling the direct utilization of DiT's training recipes and hyperparameters to minimize transition costs.

3. **Hardware-Friendly Efficiency Design**:

    - **Function**: To implement highly efficient visual linear attention computations on GPUs.
    - **Mechanism**: The training complexity of GLA is $O(TMD + TD^2)$ (where $M$ is the chunk size), which outperforms the $O(T^2D)$ complexity of standard attention when the sequence length $T > D$. DiG adopts a chunk-wise parallel implementation of GLA, splitting the sequence into several chunks to complete computations in SRAM, which avoids HBM bandwidth bottlenecks. Matrix transposition and flipping operations in SREM are highly efficient $O(T)$ operations. The overall GFLOPs are only 77-79% of those of a DiT of the same size.
    - **Design Motivation**: GLA natively supports hardware-friendly chunk computations (fully utilizing tensor cores), which is the key reason for its superior efficiency over SSM methods like Mamba on large-scale models.

### Loss & Training

The training strategy strictly follows DiT: the noise prediction network $\epsilon_\theta$ is trained with $\mathcal{L}_{simple}$ (MSE), and the covariance prediction $\Sigma_\theta$ is trained with the full $D_{KL}$ loss. The AdamW optimizer is used with a constant learning rate of $1e-4$ and an EMA decay rate of 0.9999. The EMA model is used for all image generation.

## Key Experimental Results

### Main Results (ImageNet 256×256, class-conditional)

| Model | FID↓ | sFID↓ | IS↑ | Precision↑ | Recall↑ |
|------|------|-------|-----|-----------|---------|
| DiT-S/2-400K | 68.40 | — | — | — | — |
| **DiG-S/2-400K** | **62.06** | 11.77 | 22.81 | 0.39 | 0.56 |
| DiT-B/2-400K | 43.47 | — | — | — | — |
| **DiG-B/2-400K** | **39.50** | 8.50 | 37.21 | 0.51 | 0.63 |
| DiT-XL/2-400K | 19.47 | — | — | — | — |
| **DiG-XL/2-400K** | **18.53** | 6.06 | 68.53 | 0.63 | 0.64 |
| DiG-XL/2-1200K (cfg=1.5) | **2.84** | **5.47** | **250.36** | 0.82 | 0.56 |
| ADM-G,ADM-U | 3.94 | 6.14 | 215.84 | 0.83 | 0.53 |
| LDM-4-G (cfg=1.50) | 3.60 | — | 247.67 | 0.87 | 0.48 |

### Ablation Study (DiG-S/2, 400K iterations)

| Configuration | Flops (G) | FID-50K↓ |
|------|----------|----------|
| DiT-S/2 (Baseline) | 6.06 | 68.4 |
| DiG-S/2 (Causal only) | 4.29 | 175.84 |
| + Bidirectional Scanning | 4.29 | 69.28 |
| + DWConv2d (Random Initialization) | 4.30 | 96.83 |
| + DWConv2d (Identity Initialization) | 4.30 | 63.84 |
| + Crisscross Four-Directional Scanning (Full SREM) | 4.30 | **62.06** |

### Key Findings

- **DiG systematically outperforms DiT across all model scales**: From S to XL, DiG achieves superior FID compared to counterpart DiTs, with GFLOPs of only 77-79% of DiT.
- **Efficiency gains are more prominent at high resolutions**: DiG-S/2 achieves a 2.5x speedup and saves 75.7% GPU memory compared to DiT-S/2 at 1792 resolution; DiG-XL/2 is 1.8x faster than Flash-DiT-XL/2 at 2048 resolution.
- **Every component of SREM is crucial**: Unidirectional GLA yields a catastrophically poor FID of 175.84; adding bidirectional scanning reduces it to 69.28; adding DWConv reduces it to 63.84; and implementing the full four-directional crisscross scanning reaches 62.06.
- **Identity initialization is vital for DWConv**: Randomly initialized DWConv deteriorates FID to 96.83, whereas identity initialization improves it to 63.84—an essential engineering detail.
- **Favorable scalability**: As model size and sequence length increase, DiG's FID decreases consistently, displaying scaling behavior identical to DiT.
- **DiG-XL/2 is 4.2x faster than the Mamba-based DiS-XL/2 (at 1024 resolution)**: The chunk-wise parallel computing of GLA is significantly more efficient than Mamba's sequential scan on large models.

## Highlights & Insights

- **Identity initialization of DWConv is a critical trick**: This deceptively simple initialization strategy yields a massive performance discrepancy (FID 96.83 vs 63.84). The core mechanism is to allow the model to begin training with an identity mapping, preventing random weight initialization from destroying global features captured by GLA. This technique is transferable to other scenarios where convolutions are introduced into sequence models.
- **The first successful exploration of linear attention in diffusion models**: DiG is the first diffusion backbone built on linear attention Transformers, proving that linear attention can substitute for quadratic attention while improving efficiency, charting a new path for large-scale visual generation.
- **Effectiveness of the crisscross scanning strategy**: Alternating the four scanning directions layer-by-layer allows each patch to eventually aggregate global information from all four directions, overcoming the causal limits of linear attention at an extremely low overhead.

## Limitations & Future Work

- **Validated primarily on ImageNet 256×256**: The generation quality at higher resolutions (e.g., 512, 1024) has not been presented, although efficiency analyses covered these higher resolutions.
- **Lacking exploration in complex conditional generation (e.g., text-to-image)**: DiG has only been verified for class-conditional generation; its ability to maintain advantages in more complex scenarios like CLIP-conditioned or text-conditioned generation remains unvalidated.
- **Incomplete comparisons with recently proposed methods**: The paper does not thoroughly compare DiG with other sub-quadratic methods like Diffusion-RWKV or ZigMa under identical configurations.
- **Self-acknowledged limitation by the authors**: The exploration of building large-scale foundation models (such as Sora) on DiG remains untouched.
- **Directions for improvement**: Extending DiG to text-to-image and video generation tasks; exploring deeper integration with FlashLinearAttention; conducting full generation quality evaluations at 512 and 1024 resolutions.

## Related Work & Insights

- **vs DiT**: DiT is the direct baseline for DiG. DiG retains DiT's architectural design philosophy (patchify, adaLN-Zero), merely substituting self-attention with GLA+SREM, making it easy to reuse DiT's hyperparameters.
- **vs DiS (Mamba-based)**: DiS builds diffusion backbones on Mamba, but Mamba's sequential scan and complex block designs are inefficient for large-scale models. DiG-XL/2 is 4.2x faster than DiS-XL/2 at 1024 resolution.
- **vs Flash-DiT**: Even when DiT is optimized with state-of-the-art FlashAttention-2, it is still 1.8x slower than DiG-XL/2 at 2048 resolution, demonstrating that the architectural advantage of linear attention cannot be fully compensated for by attention optimizations.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to introduce linear attention into diffusion models, featuring a simple yet effective SREM.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Complete ablation studies and multi-scale scaling analysis, though lacking high-resolution generation quality evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear methodological descriptions and detailed efficiency analyses.
- **Value**: ⭐⭐⭐⭐⭐ Offers a compelling candidate for next-generation diffusion backbones, carrying significant importance for video and high-resolution generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] EDiT: Efficient Diffusion Transformers with Linear Compressed Attention](../../ICCV2025/image_generation/edit_efficient_diffusion_transformers_with_linear_compressed_attention.md)
- [\[NeurIPS 2025\] DiCo: Revitalizing ConvNets for Scalable and Efficient Diffusion Modeling](../../NeurIPS2025/image_generation/dico_revitalizing_convnets_for_scalable_and_efficient_diffus.md)
- [\[CVPR 2026\] Gated Condition Injection without Multimodal Attention: Towards Controllable Linear-Attention Transformers](../../CVPR2026/image_generation/gated_condition_injection_without_multimodal_attention_towards_controllable_line.md)
- [\[CVPR 2025\] MCA-Ctrl: Multi-party Collaborative Attention Control for Image Customization](mca_ctrl_attention_control_customization.md)
- [\[CVPR 2025\] InPO: Inversion Preference Optimization with Reparametrized DDIM for Efficient Diffusion Model Alignment](inpo_inversion_preference_optimization_diffusion_alignment.md)

</div>

<!-- RELATED:END -->

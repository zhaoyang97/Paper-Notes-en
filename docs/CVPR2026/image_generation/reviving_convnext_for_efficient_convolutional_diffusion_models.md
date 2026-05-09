---
title: >-
  [Paper Note] Reviving ConvNeXt for Efficient Convolutional Diffusion Models
description: >-
  [CVPR 2026][Image Generation][Diffusion models] This paper proposes FCDM (Fully Convolutional Diffusion Model), which adapts the ConvNeXt architecture as a backbone for conditional diffusion models. Using only 50% of DiT-XL's FLOPs, FCDM achieves a competitive FID of 2.03 on ImageNet and can train an XL-scale model on 4× RTX 4090 GPUs, demonstrating the severely underestimated efficiency of fully convolutional architectures in generative modeling.
tags:
  - CVPR 2026
  - Image Generation
  - Diffusion models
  - ConvNeXt
  - fully convolutional
  - efficient generation
date: 2026-05-08
content_hash: f2c5cb752098f988
---

# Reviving ConvNeXt for Efficient Convolutional Diffusion Models

**Conference**: CVPR 2026
**arXiv**: [2603.09408](https://arxiv.org/abs/2603.09408)
**Code**: Available (official implementation released)
**Area**: Image Generation
**Keywords**: Diffusion models, ConvNeXt, fully convolutional, efficient generation, image generation
**Institution**: KAIST, ETH Zürich, ISTI-CNR, University of Pisa

## TL;DR
This paper proposes FCDM (Fully Convolutional Diffusion Model), which adapts the ConvNeXt architecture as a backbone for conditional diffusion models. Using only 50% of DiT-XL's FLOPs, FCDM achieves a competitive FID of 2.03 on ImageNet and can train an XL-scale model on 4× RTX 4090 GPUs, demonstrating the severely underestimated efficiency of fully convolutional architectures in generative modeling.

## Background & Motivation

**Background**: Diffusion model backbones have evolved from convolutional–attention hybrid architectures (DDPM, ADM, LDM) to fully Transformer-based architectures (DiT, SiT, FLUX). The scalability of Transformers has driven the success of large-scale models such as FLUX and SD3, but has also introduced a strong dependency on GPU cluster resources.

**Limitations of Prior Work**: DiT-XL/2 requires 7M training steps to reach optimal FID, with a training throughput of only 80.5 it/s. The $O(n^2)$ computational complexity of Transformers is particularly severe at high resolutions—doubling the resolution reduces DiT throughput by approximately 4×. This makes the training and inference cost of diffusion models a major bottleneck.

**Key Challenge**: The prevailing assumption that "scaling Transformers = better generation quality" has left the locality bias, parameter efficiency, and hardware friendliness of ConvNets largely unexplored in modern generative modeling. While ConvNeXt has demonstrated ViT-matching performance on classification tasks, it has been entirely absent from the generative domain.

**Key Insight**: The paper adapts ConvNeXt into a backbone for conditional diffusion models, preserving its core design (depthwise convolution, inverted bottleneck, GRN) while adding only conditional injection (AdaLN) and a U-shaped layout, to verify whether a fully convolutional architecture can simultaneously achieve high generation quality and computational efficiency.

## Method

### Overall Architecture

FCDM operates in latent space (consistent with DiT). Input RGB images of $256 \times 256 \times 3$ are encoded by a VAE into a $32 \times 32 \times 4$ latent tensor, processed through multiple FCDM blocks, and decoded back to pixel space by a VAE decoder. The FCDM blocks are organized in a simplified U-shaped architecture, with encoder and decoder connected via skip connections.

**Core Design Philosophy**: Unlike DiT, which requires four hyperparameters (number of layers $L$, channels $C$, attention heads, patch size), FCDM requires only **two hyperparameters**—the number of blocks $L$ and the hidden channel dimension $C$. At each 2× downsampling step, both $L$ and $C$ are doubled. This "Easy Scaling Law" substantially simplifies the architecture search space.

### FCDM Block Design

The FCDM block is a minimal adaptation of the ConvNeXt block, preserving the original ConvNeXt's core structure while adding conditional injection capability:

**Original ConvNeXt Block**:
$\text{Input} \to 7\times7 \text{ DWConv} \to \text{LayerNorm} \to 1\times1 \text{ Conv}(\uparrow r) \to \text{GRN} \to 1\times1 \text{ Conv}(\downarrow r) \to \text{Output}$

**FCDM Block Modifications**:
- **Conditional Injection**: LayerNorm is replaced with Adaptive LayerNorm (AdaLN). A lightweight MLP maps the concatenated class embedding and timestep embedding to three parameter groups $(\gamma, \beta, \alpha)$: $\gamma$ and $\beta$ are used for affine transformation of the normalized features, while $\alpha$ serves as a scaling factor applied to the final output.
- **Zero Initialization**: Following DiT, the final modulation scale $\alpha$ is zero-initialized, so that each FCDM block behaves as an identity mapping at the start of training, promoting optimization stability in deep networks.
- **7×7 Depthwise Convolution**: The large-kernel depthwise convolution from ConvNeXt is retained to provide a sufficiently large receptive field for capturing spatial context. Ablation studies confirm that 7×7 significantly outperforms 5×5 and 3×3 kernels.

### Inverted Bottleneck — The Core Source of Efficiency

The most critical structural difference between FCDM and DiCo lies in the treatment of the channel dimension, which is the primary reason FCDM's FLOPs are only 75% of DiCo's.

**DiCo's Approach**: The channel dimension is kept constant throughout the convolutional module; channel expansion is performed in a separate feedforward module consisting of two $1\times1$ convolutions.

**FCDM's Approach (Inverted Bottleneck)**:
1. Apply $7\times7$ depthwise convolution (with $C$ channels, compute $\propto C$).
2. Use a $1\times1$ pointwise convolution to expand channels to $rC$ (expansion ratio $r=3$).
3. After GRN, use another $1\times1$ pointwise convolution to compress channels back to $C$.

**Key Trick**: The depthwise convolution is placed **before** channel expansion rather than after. Since the computational cost of depthwise convolution is proportional only to the number of input channels (with no cross-channel interaction), this reordering ensures that the depthwise convolution always operates over a low-dimensional channel space, and its cost does not scale with the expansion ratio. The high-dimensional expanded features are handled exclusively by the more lightweight pointwise convolutions, achieving the dual benefit of lower compute and enhanced representational capacity.

**Numerical Verification**: Under parameter-matched conditions, the FLOPs of each FCDM scale are as follows:

| Model | Params | Blocks $L$ | Channel $C$ | FLOPs (G) | vs. DiT | vs. DiCo |
|-------|--------|------------|-------------|-----------|---------|----------|
| FCDM-S | 32.7M | 2 | 128 | 3.1 | 50.8% | 72.9% |
| FCDM-B | 127.7M | 2 | 256 | 12.2 | 53.0% | 72.3% |
| FCDM-L | 504.5M | 2 | 512 | 48.3 | 59.9% | 80.2% |
| FCDM-XL | 698.8M | 3 | 512 | 64.6 | 54.5% | 74.0% |

### GRN vs. CCA — Lightweight Channel Diversity

DiCo introduces Compact Channel Attention (CCA) to mitigate channel redundancy, which essentially learns channel-wise attention weights via an additional $1\times1$ pointwise convolution.

FCDM instead uses **Global Response Normalization (GRN)** from ConvNeXt V2. GRN consists primarily of parameter-free operations: it computes the global L2 norm for each channel and applies response normalization. Both mechanisms share the same goal—promoting diversity in channel activations and reducing channel redundancy—but GRN introduces virtually no additional learnable parameters.

Feature visualizations (Figure 7 of the paper) illustrate the effect of GRN: the 64-channel feature maps after GRN exhibit clear diversity, whereas the pre-GRN features contain a large number of redundant channels.

### No Additional Feedforward Module

In addition to its convolutional module, DiCo includes a feedforward module (two $1\times1$ convolutions for channel expansion). FCDM does not require this module, as the inverted bottleneck structure already performs channel expansion and compression within the block. Ablation experiments show that adding an extra feedforward module to FCDM causes FID to increase sharply from 19.97 to 28.52—performing channel expansion twice is detrimental.

### Simplified U-Shaped Architecture

Traditional U-Nets require careful design of the number of blocks and channel dimensions at each resolution level. FCDM simplifies this entirely:
- **Downsampling Rule**: At each 2× downsampling step, both the number of blocks $L$ and the channel dimension $C$ are doubled.
- **Skip Connections**: Features from each encoder level are passed directly to the corresponding decoder level.
- **No Resolution-Specific Design**: All levels use the same FCDM block structure without any resolution-dependent special treatment.

This means scaling from FCDM-S to FCDM-XL requires adjusting only two numbers ($L$: 2→3, $C$: 128→512), greatly reducing the cost of hyperparameter tuning.

### Loss & Training

The training setup follows DiT/ADM exactly, without introducing any additional tricks:
- **Diffusion Process**: $t_{\max}=1000$ steps, linear noise schedule ($\beta$ from $1\times10^{-4}$ to $2\times10^{-2}$), iDDPM covariance parameterization.
- **Optimizer**: AdamW, lr = $1\times10^{-4}$ (constant schedule), no weight decay.
- **Training Precision**: fp32 (no mixed precision).
- **EMA**: Decay factor = 0.9999.
- **Evaluation**: 250-step DDPM sampling, FID computed on 50K samples, classifier-free guidance used when applicable.

## Key Experimental Results

### Main Results: ImageNet 256×256 Multi-Scale Comparison (400K Steps)

| Model | Architecture | FLOPs (G)↓ | Throughput (it/s)↑ | FID↓ | IS↑ |
|-------|-------------|------------|-------------------|------|-----|
| DiT-XL/2 | Transformer | 118.6 | 80.5 | 19.47 | — |
| DiG-XL/2 | Hybrid | 89.4 | 71.7 | 18.53 | 68.53 |
| DiCo-XL | Conv | 87.3 | 174.2 | 11.67 | 100.4 |
| DiC-H | Conv | 204.4 | 144.5 | 11.36 | 106.5 |
| **FCDM-XL** | **Conv** | **64.6** | **272.7** | **10.72** | **108.0** |

FCDM-XL achieves the lowest FID (10.72), fewest FLOPs (64.6G), and highest throughput (272.7 it/s) at 400K steps. After 1M steps, FID further decreases to 7.91, whereas DiT-XL/2 requires 7M steps to reach 9.62.

### Benchmark Results (Long Training + Classifier-Free Guidance)

| Model | Training Epochs | FLOPs (G)↓ | Throughput↑ | FID↓ | IS↑ |
|-------|----------------|------------|-------------|------|-----|
| DiT-XL/2 | 1400 | 118.6 | 80.5 | 2.27 | 278.2 |
| SiT-XL/2 | 1400 | 118.6 | 80.5 | 2.06 | 277.5 |
| DiCo-XL | 750 | 87.3 | 174.2 | 2.05 | 282.2 |
| **FCDM-XL** | **400** | **64.6** | **272.7** | **2.03** | **285.7** |

FCDM-XL achieves FID 2.03 (state of the art) in 400 epochs—3.5× fewer epochs than DiT and 1.9× fewer than DiCo.

### 512×512 Resolution

FCDM-XL achieves FID 7.46 at 512×512 with 1M steps, outperforming DiT-XL/2's FID of 12.03 at 3M steps (7.5× fewer training steps). Notably, when resolution is doubled, DiT throughput drops by approximately 4× (due to the $O(n^2)$ effect of global attention), while FCDM throughput drops by only approximately 2× (owing to the linear complexity of convolution), making FCDM's advantage even more pronounced at higher resolutions.

### Ablation Study (FCDM-L, 200K Steps)

| Configuration | FLOPs (G) | FID↓ | IS↑ | Conclusion |
|---------------|-----------|------|-----|------------|
| Default (7×7 DWConv + GRN) | 48.3 | 19.97 | 69.19 | Baseline |
| → 5×5 DWConv | 48.2 | 20.48 | 66.69 | Smaller receptive field slightly degrades |
| → 3×3 DWConv | 48.1 | 21.28 | 64.11 | Large kernel is important (FID +1.3) |
| → CCA replaces GRN | 48.3 | 23.85 | 61.60 | GRN far superior to CCA (FID +3.9) |
| → Add Feedforward | 48.2 | 28.52 | 47.16 | Extra FFN is harmful (FID +8.5) |
| → Remove Inverted Bottleneck | 48.3 | 28.76 | 52.20 | IB structure is critical |
| → Replace with ResNet block | 48.4 | 31.14 | 49.10 | Modern ConvNeXt design far surpasses classic ResNet |

### Compute Requirements

FCDM-XL can complete ImageNet 256×256 training on 4× RTX 4090 (consumer-grade) GPUs with batch size 256 at approximately 0.9 steps/s. The same batch size also fits on a single A100 40GB. By comparison, DiT at the same scale typically requires 8× A100/H100 GPUs.

## Highlights & Insights

- **Two-Parameter Scaling Law**: The entire network is defined by two hyperparameters, $L$ and $C$, substantially reducing the architecture search space.
- **Inverted Bottleneck Reordering**: Placing the depthwise convolution before channel expansion is the key to achieving a 25% FLOPs reduction over DiCo, and this trick generalizes to other convolutional generative architectures.
- **Unexpected Effectiveness of GRN**: A module originally designed for classification in ConvNeXt V2 proves equally effective in the generative setting, substantially outperforming CCA, which was specifically designed for diffusion models.
- **Resolution Friendliness**: The linear computational complexity of convolution results in far smaller throughput degradation as resolution increases compared to Transformers, making FCDM's advantage even more pronounced at 512×512.
- **XL-Scale Training on 4× RTX 4090**: This has significant practical value for academic and resource-constrained settings.

## Limitations & Future Work

- FCDM has not yet surpassed state-of-the-art methods such as EDM-2 and Simpler Diffusion, which employ more advanced training frameworks (e.g., improved noise schedules, preconditioning).
- Evaluation is limited to ImageNet class-conditional generation; performance on more complex conditioning scenarios such as text-to-image and video generation remains to be verified.
- Fully convolutional architectures are theoretically weaker than Transformers at modeling very long-range dependencies, which may limit global semantic consistency.
- Training is performed in fp32 only; the efficiency gains and stability of mixed-precision training are unexplored.

## Related Work & Insights

- **vs. DiT/SiT**: FCDM replaces attention entirely with convolution, halving FLOPs at matched parameter counts and reaching equivalent FID with 7× fewer training steps. DiT's advantages lie in its theoretical capacity for global attention and its natural compatibility with text conditioning.
- **vs. DiCo**: The structurally closest competitor. FCDM achieves a 25% FLOPs reduction through inverted bottleneck reordering and replacing CCA + feedforward with GRN, while also slightly improving generation quality.
- **vs. DiC**: DiC uses standard 3×3 convolutions and achieves better hardware utilization (higher throughput) at S/B scales, but FCDM comprehensively outperforms it at L/XL scales.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Applying ConvNeXt to diffusion models is not entirely novel (DiC/DiCo preceded it), but the analysis of inverted bottleneck reordering and the systematic comparison with DiCo offer genuine conceptual value.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four scales × multiple training steps, dual resolutions (256/512), detailed ablations and feature visualizations, and comprehensive evaluation of FLOPs/throughput/FID.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure; Figure 4's DiCo vs. FCDM comparison is highly intuitive, and the three-point differential analysis against DiCo is thorough.
- **Value**: ⭐⭐⭐⭐ — Training an XL-scale model on 4× RTX 4090 GPUs is highly attractive for resource-constrained settings; the two-parameter scaling law has practical engineering value.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] GLASS Flows: Efficient Inference for Reward Alignment of Flow and Diffusion Models](../../ICLR2026/image_generation/glass_flows_reward_alignment_diffusion.md)
- [\[AAAI 2026\] DogFit: Domain-guided Fine-tuning for Efficient Transfer Learning of Diffusion Models](../../AAAI2026/image_generation/dogfit_domain-guided_fine-tuning_for_efficient_transfer_learning_of_diffusion_mo.md)
- [\[CVPR 2026\] DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression](dit-ic_aligned_diffusion_transformer_for_efficient_image_compression.md)
- [\[ICLR 2026\] Test-Time Iterative Error Correction for Efficient Diffusion Models](../../ICLR2026/image_generation/test-time_iterative_error_correction_for_efficient_diffusion_models.md)
- [\[ICLR 2026\] SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models](../../ICLR2026/image_generation/speed_scalable_precise_and_efficient_concept_erasure_for_diffusion_models.md)

<!-- RELATED:END -->

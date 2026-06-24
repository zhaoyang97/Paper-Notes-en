---
title: >-
  [Paper Note] Efficient Visual State Space Model for Image Deblurring
description: >-
  [CVPR 2025][Image Restoration][Image Deblurring] This paper proposes EVSSM, which efficiently captures non-local information by applying alternating geometric transformations (transpose/flip) before unidirectional SSM scanning, and designs an efficient discriminative frequency-domain FFN (EDFFN) to enhance local details. It outperforms existing SSM methods and achieves SOTA on image deblurring tasks with only 1/4 of the computational cost.
tags:
  - "CVPR 2025"
  - "Image Restoration"
  - "Image Deblurring"
  - "State Space Model"
  - "Mamba"
  - "Frequency-domain FFN"
  - "Geometric Transformation Scan"
date: 2026-05-08
content_hash: 206d15116215d788
---

# Efficient Visual State Space Model for Image Deblurring

**Conference**: CVPR 2025  
**arXiv**: [2405.14343](https://arxiv.org/abs/2405.14343)  
**Code**: [https://github.com/kkkls/EVSSM](https://github.com/kkkls/EVSSM)  
**Area**: Image Restoration  
**Keywords**: Image Deblurring, State Space Model, Mamba, Frequency-domain FFN, Geometric Transformation Scan

## TL;DR

This paper proposes EVSSM, which efficiently captures non-local information by applying alternating geometric transformations (transpose/flip) before unidirectional SSM scanning, and designs an efficient discriminative frequency-domain FFN (EDFFN) to enhance local details. It outperforms existing SSM methods and achieves SOTA on image deblurring tasks with only 1/4 of the computational cost.

## Background & Motivation

**Background**: Image deblurring aims to restore sharp images from blurry ones, with mainstream methods divided into two categories: CNNs and Transformers. CNN-based methods are limited by the spatial invariance and local receptive fields of convolutional operations, making it difficult to capture spatially-varying characteristics and non-local information. Transformers model global dependencies through self-attention mechanisms, achieving better results, but their computational complexity is quadratic with respect to the number of tokens, making the cost unacceptable when processing high-resolution images.

**Limitations of Prior Work**: To reduce the computational overhead of Transformers, existing methods employ strategies such as local window attention, transposed attention, and frequency-domain approximations. However, while reducing computational cost, these methods sacrifice the ability to model non-local or spatial information, which limits restoration quality. Recently, State Space Models (SSMs/Mamba) have demonstrated potential for modeling long-range dependencies with linear complexity. However, existing visual SSM methods (such as VMamba) adopt multi-directional scanning mechanisms, which incur a computational cost four times that of unidirectional scanning, largely offsetting the efficiency gains.

**Key Challenge**: How can SSMs effectively explore non-local information in 2D images while maintaining linear computational complexity? Multi-directional scanning provides comprehensive coverage but incurs a high computational cost, whereas unidirectional scanning fails to fully exploit spatial structures.

**Goal**: (1) Design an efficient visual scanning strategy to capture multi-directional non-local information without significantly increasing computational cost; (2) Resolve the issue where SSM parameters $B$, $C$, and $\Delta$ are derived from the same linear transformation, leading to uniform spatial information; (3) Reduce the computational overhead of the frequency-domain FFN while preserving the capacity for local detail enhancement.

**Key Insight**: The authors observe that instead of repeatedly scanning in multiple directions, applying simple geometric transformations (transpose or flip) to the input features before each scan allows a unidirectional scan to automatically cover information in different directions. Since convolution possesses translation invariance, geometric transformations do not affect the convolution itself but only alter the behavior of the selective scan.

**Core Idea**: Replace multi-directional scanning with alternating geometric transformations and unidirectional scanning to achieve multi-directional non-local information exploration at nearly zero cost.

## Method

### Overall Architecture

EVSSM adopts a classic three-level symmetric encoder-decoder architecture. The input blurry image $I_{blur} \in \mathbb{R}^{H \times W \times 3}$ is first processed by a 3×3 convolution to extract shallow features $F_s \in \mathbb{R}^{H \times W \times C}$ ($C=48$), which are then fed into the three-level encoder-decoder. Each level of the encoder/decoder is stacked with several EVSS blocks (the number of blocks per level is [6, 6, 12]). Downsampling and upsampling between levels are achieved via bilinear interpolation and 1×1 convolutions, complemented by skip connections. Finally, a residual image $R$ is output through a 3×3 convolution, and added to the input to obtain the deblurred result $I_{deblur} = R + I_{blur}$.

### Key Designs

1. **Efficient Visual Scan Block (EVS Block)**:

    - Function: Explores multi-directional non-local information with minimal computational cost
    - Mechanism: For the $i$-th EVSS block, geometric transformations are applied alternately based on the block index before scanning: feature transposition is performed when $i \% 2 = 0$, and horizontal + vertical flipping is performed when $i \% 2 = 1$. Consequently, every 4 EVSS blocks automatically restore the original spatial structure. After the transformation, the features are split into two branches $X_1, X_2$ via a linear layer. $X_1$ is fed into S6 selective scanning after a 3×3 depthwise convolution and SiLU activation, while $X_2$ serves as a gating signal multiplied with the scanning output
    - Design Motivation: Geometric transformations incur almost zero overhead (involving only memory reshaping), yet they enable unidirectional scanning to "see" different spatial arrangements of information across different blocks, effectively achieving the results of multi-directional scanning

2. **1D Depthwise Convolution for Enhancing SSM Parameter Diversity**:

    - Function: Enables the SSM parameters $B$, $C$, and $\Delta$ to encode distinct spatial information
    - Mechanism: After deriving $B$, $C$, and $\Delta$ through linear projection, a 1D depthwise convolution with a kernel size of 7 is applied to each parameter individually. Due to the preceding geometric transformation, the 1D convolution actually aggregates multi-directional information on the original 2D input, providing each parameter with a differentiated spatial representation
    - Design Motivation: In the original Mamba, $B$, $C$, and $\Delta$ are all obtained from linear projections of the same input, which means they encode the same spatial information and limit the model's capacity to capture diverse spatial patterns. Adding 1D depthwise convolutions endows each parameter with independent local attention capabilities

3. **Efficient Discriminative Frequency-domain FFN (EDFFN)**:

    - Function: Enhance local detail information that is not fully covered by the SSM
    - Mechanism: Perform FFT at the end (rather than the middle) of the FFN and learn a quantization matrix $W$ to adaptively filter frequency components that need to be retained. Since the number of feature channels at the end of the FFN is much smaller than that of the middle layer (the original DFFN performs FFT on the middle layer with 3x channel expansion), the computational overhead is significantly reduced
    - Design Motivation: The DFFN in FFTformer performs FFT in the middle of the FFN, where the channel dimension is expanded by 3 times, leading to huge FFT computational costs. Shifting the frequency-domain filtering to the end of the FFN significantly reduces computation time without sacrificing performance

### Loss & Training

The training loss consists of a pixel-domain L1 loss and a frequency-domain L1 loss: $\mathcal{L} = \|I_{deblur} - I_{gt}\|_1 + 0.1 \|\mathcal{F}(I_{deblur}) - \mathcal{F}(I_{gt})\|_1$. Progressive training is employed: first training with 128×128 patches and a batch size of 64 for 300K iterations, then switching to 256×256 patches and a batch size of 16 for another 300K iterations. The AdamW optimizer and cosine annealing strategy are used throughout.

## Key Experimental Results

### Main Results

| Dataset | Method | PSNR (dB) | SSIM |
|--------|------|-----------|------|
| GoPro | FFTformer | 34.21 | 0.9692 |
| GoPro | GRL | 33.93 | 0.9680 |
| GoPro | **EVSSM** | **34.51** | **0.9713** |
| HIDE | FFTformer | 31.62 | 0.9455 |
| HIDE | GRL | 31.65 | 0.9470 |
| HIDE | **EVSSM** | **31.99** | **0.9503** |
| RealBlur-R | FFTformer | 40.11 | 0.9753 |
| RealBlur-R | **EVSSM** | **41.27** | **0.9776** |
| RealBlur-J | FFTformer | 32.62 | 0.9326 |
| RealBlur-J | **EVSSM** | **34.34** | **0.9456** |

### Ablation Study

| Configuration | Computational Characteristics | Description |
|------|---------|------|
| VMamba four-directional scan | 4× computational cost | Common practice in existing visual SSM methods |
| EVSSM geometric transform + unidirectional | 1× computational cost | Only minor overhead added from geometric transformation |
| CU-Mamba (another SSM method) | GoPro PSNR 33.53 | Significantly lower than EVSSM's 34.51 |

Furthermore, the generalization of EVSSM is validated on image deraining (PSNR 49.00 vs. Restormer 47.98) and image dehazing (PSNR 32.05 vs. DehazeFormer 31.45) tasks.

### Key Findings
- The geometric transformation strategy is the core contribution: it achieves information coverage equivalent to multi-directional scanning at nearly zero computational cost.
- EDFFN shifts the frequency-domain filtering to the end of the FFN, significantly reducing runtime while maintaining identical performance.
- The enhancement of $B$, $C$, and $\Delta$ parameters by 1D depthwise convolution also makes an independent contribution, providing spatial disparity to the parameters during scans.
- The improvement is particularly pronounced on the real-world blur dataset RealBlur (+1.16 dB / +1.72 dB), demonstrating robust generalization to real-world degradation.

## Highlights & Insights
- **Replacing multi-directional scanning with geometric transformations** is the most ingenious design of this paper—utilizing zero-cost operations like transposing/flipping to alter the relative positions of information in the sequence, allowing unidirectional scanning to equivalently scan different spatial directions across different layers. This represents an elegant engineering solution.
- **Optimizing the position of the frequency-domain FFN** (shifting from the middle to the end) is a simple yet highly effective improvement with broad transferability—any network that performs expensive transformations on intermediate feature expansion layers can consider moving those transformations to positions with smaller channel dimensions.
- The successful application of EVSSM to deraining and dehazing tasks beyond deblurring indicates that the architecture possesses general image restoration capabilities.

## Limitations & Future Work
- Although the geometric transformation strategy is simple, the transformation patterns are fixed (alternating transpose and flips), and the possibility of adaptively choosing transformation types remains unexplored.
- The paper does not fully analyze the variance in impact of geometric transformations across different feature scales, nor does it compare it against learnable scanning directions.
- While shifting the frequency-domain filtering to the end in EDFFN reduces computational cost, it might compromise some representation capacity over high-channel features, which lacks validation via ablation studies.

## Related Work & Insights
- **vs. FFTformer**: FFTformer employs a frequency-domain Transformer, whereas EVSSM replaces the attention mechanism with an SSM to achieve linear complexity while improving FFTformer's DFFN component. EVSSM outperforms FFTformer on all baselines.
- **vs. VMamba/CU-Mamba**: These methods employ multi-directional scanning, whereas EVSSM achieves equivalent performance via geometric transformations while requiring only 1/4 of the computational cost.
- **vs. NAFNet/Restormer**: CNN/Transformer-based methods struggle to efficiently model non-local information. The linear complexity benefit of EVSSM becomes exceptionally prominent in high-resolution scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of replacing multi-directional scanning with geometric transformations is novel and elegant, though the overall framework still follows the standard encoder-decoder structure.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across multiple deblurring benchmarks and other restoration tasks, although the ablation studies could be more comprehensive.
- Writing Quality: ⭐⭐⭐⭐ The logic is sound, the motivation derivation is reasonable, and the figures/tables are clearly illustrated.
- Value: ⭐⭐⭐⭐ It provides a simple and practical solution for the efficiency of visual SSMs, offering valuable references for related fields.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MambaIRv2: Attentive State Space Restoration](mambairv2_attentive_state_space_restoration.md)
- [\[ICCV 2025\] EAMamba: Efficient All-Around Vision State Space Model for Image Restoration](../../ICCV2025/image_restoration/eamamba_efficient_all-around_vision_state_space_model_for_image_restoration.md)
- [\[CVPR 2025\] QMambaBSR: Burst Image Super-Resolution with Query State Space Model](qmambabsr_burst_image_super-resolution_with_query_state_space_model.md)
- [\[CVPR 2025\] URWKV: Unified RWKV Model with Multi-State Perspective for Low-Light Image Restoration](urwkv_unified_rwkv_model_with_multi-state_perspective_for_low-light_image_restor.md)
- [\[ICCV 2025\] PRE-Mamba: A 4D State Space Model for Ultra-High-Frequent Event Camera Deraining](../../ICCV2025/image_restoration/pre-mamba_a_4d_state_space_model_for_ultra-high-frequent_event_camera_deraining.md)

</div>

<!-- RELATED:END -->

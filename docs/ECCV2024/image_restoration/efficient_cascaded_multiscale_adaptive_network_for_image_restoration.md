---
title: >-
  [Paper Note] Efficient Cascaded Multiscale Adaptive Network for Image Restoration
description: >-
  [ECCV 2024][Image Restoration][multiscale learning] ECMA proposes an efficient cascaded multiscale adaptive network that dynamically adjusts convolutional kernels via the Local Adaptive Module (LAM) to handle spatially-varying degradations, and captures features at different scales in a cascaded multiscale manner. It achieves comparable or even superior performance to SOTA on various image restoration tasks (including deblurring, denoising, and super-resolution) with a 1.2×-9…
tags:
  - "ECCV 2024"
  - "Image Restoration"
  - "multiscale learning"
  - "local adaptation"
  - "dynamic convolution"
  - "efficient networks"
date: 2026-05-08
content_hash: f5ee9de1f1c262c2
---

# Efficient Cascaded Multiscale Adaptive Network for Image Restoration

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Image Restoration  
**Keywords**: image restoration, multiscale learning, local adaptation, dynamic convolution, efficient networks

## TL;DR

ECMA proposes an efficient cascaded multiscale adaptive network that dynamically adjusts convolutional kernels via the Local Adaptive Module (LAM) to handle spatially-varying degradations, and captures features at different scales in a cascaded multiscale manner. It achieves comparable or even superior performance to SOTA on various image restoration tasks (including deblurring, denoising, and super-resolution) with a 1.2×-9.7× reduction in computational cost.

## Background & Motivation

**Background**: Image restoration (including deblurring, denoising, super-resolution) is a fundamental low-level task in computer vision. Recently, Transformer-based methods (e.g., Restormer, SwinIR) have achieved significant progress due to their global attention mechanisms, but they incur heavy computational overheads. CNN-based methods offer advantages in efficiency but suffer from limited global receptive fields. The academic community continues to seek an optimal balance between performance and efficiency.

**Limitations of Prior Work**: Image degradation (such as motion blur, noise, low resolution) is fundamentally non-uniform in space, meaning different regions exhibit varying degradation levels and patterns. For example, in a motion-blurred image, the blur kernel directions and scales of foreground objects and the background differ. Existing methods suffer from two main issues: (1) Standard convolutions process all regions with fixed parameters, failing to adapt to spatially-varying degradation patterns. (2) Although self-attention mechanisms effectively handle local adaptivity, they suffer from $O(n^2)$ computational complexity, incurring massive overhead on high-resolution inputs. Furthermore, degradation typically spans multiple scales, requiring simultaneous handling of fine-grained noise and large-scale blur.

**Key Challenge**: There is a trade-off between local adaptivity and computational efficiency. Self-attention provides spatially-adaptive feature processing but at an extremely high computational cost, while standard convolutions are efficient but lack spatial adaptability. Moreover, degradations at different scales require varying receptive field sizes; simply stacking multi-scale features increases model complexity.

**Goal**: (1) How to achieve locally adaptive image restoration at a lower computational cost than self-attention? (2) How to efficiently handle degradations of different granularities across multiple scales? (3) How to design a unified architecture that is both highly effective and efficient across various restoration tasks?

**Key Insight**: The authors observe that dynamic convolution (which adaptively generates convolution kernels based on the input) can achieve self-attention-like spatial adaptivity at the computational efficiency level of standard convolutions. Furthermore, by cascading dynamic convolution kernels of decreasing sizes (from large to small), multi-scale information can be captured simultaneously within a single module without requiring extra multi-scale branches.

**Core Idea**: Replace self-attention with dynamic convolution to achieve local adaptivity, and implement multi-scale feature learning by cascading kernels of decreasing sizes. This significantly reduces computation while maintaining or even surpassing SOTA restoration performance.

## Method

### Overall Architecture

The ECMA network adopts the classic U-Net architecture as its backbone, stacking multiple ECMA Blocks at each level of the encoder and decoder. The degraded image is fed as the overall input, processed through multi-stage encoding (downsampling) and decoding (upsampling) stages to output the restored image. The core innovation lies in the internal design of the ECMA Block, where each block consists of three cascaded Local Adaptive Modules (LAMs) utilizing kernel sizes from large to small, thereby achieving multi-scale feature extraction within a single block.

### Key Designs

1. **Local Adaptive Module (LAM)**:

    - **Function**: Dynamically adjusts convolutional kernel parameters based on input features, so that different spatial locations utilize different kernels to handle their respective degradation patterns.
    - **Mechanism**: The core of LAM is dynamic convolution. Instead of using fixed kernel weights, a lightweight kernel-generation network dynamically generates location-dependent convolution kernels based on local input features. Specifically, for each spatial location, the kernel-generation network receives the local feature vector to output a set of custom kernel parameters, which are then used to convolve the neighborhood features of that location. This allows different locations to receive distinct filtering effects to adapt to local degradation. The computational complexity is $O(n \cdot k^2)$ (where $n$ is the pixel count and $k$ is the kernel size), which is significantly lower than the $O(n^2)$ complexity of self-attention.
    - **Design Motivation**: Traditional convolution uses the same kernel across all locations, failing to distinguish between severely and mildly degraded regions. While self-attention achieves pixel-level adaptivity, its computational overhead on high-resolution images is prohibitive. LAM achieves a similar adaptive effect at a near-convolutional cost.

2. **Cascaded Multiscale Architecture**:

    - **Function**: Automatically captures coarse-to-fine multi-scale degradation features within each ECMA Block by cascading three LAMs of different kernel sizes.
    - **Mechanism**: An ECMA Block is composed of three cascaded LAMs with kernel sizes decreasing from large (e.g., 7×7) to medium (e.g., 5×5) to small (e.g., 3×3). The large-kernel LAM first captures broad degradation patterns (such as large-area blur), the medium-kernel LAM handles medium-scale degradations, and the small-kernel LAM finally refines fine local details. The outputs of the three LAMs are passed to the next block via residual connections and feature fusion. This cascaded design empowers each block with multi-scale processing capabilities without resorting to parallel multi-branch structures common in traditional approaches.
    - **Design Motivation**: Image degradation typically spans multiple scales—deblurring demands large receptive fields, while denoising requires highly localized operations. Traditional multi-scale approaches utilize parallel branches followed by fusion, which leads to structural complexity and a large parameter count. Cascading kernels from large to small yields efficient multi-scale learning in a sequential manner.

3. **U-Net Integration and Channel Attention**:

    - **Function**: Achieves a wider scope of multi-scale feature fusion at the network level through an encoder-decoder structure and skip connections.
    - **Mechanism**: The ECMA Block is integrated into a standard U-Net framework, where the encoder downsamples hierarchically to enlarge the receptive field and the decoder upsamples sequentially to restore spatial resolution. Skip connections transfer shallow high-resolution features from the encoder to the decoder to aid details restoration. Additionally, a channel attention mechanism is applied within each ECMA Block to adaptively weight feature responses across different channels.
    - **Design Motivation**: The U-Net architecture and the multi-scale characteristics of the ECMA Block are complementary. U-Net provides cross-resolution multi-scale information along the network depth, while ECMA Block provides cross-kernel-size multi-scale information within the same resolution.

### Loss & Training

Standard loss functions are adopted according to the specific restoration tasks: L1 loss combined with perceptual loss is used for deblurring and super-resolution, and plain L1 loss is used for denoising. A progressive training strategy is employed, starting training with small patches and gradually increasing the patch size during fine-tuning. Standard datasets specific to each task are used for training.

## Key Experimental Results

### Main Results

| Dataset/Task | Metric | ECMA | Prev. SOTA | Comp. Cost Comparison |
|:---:|:---:|:---:|:---:|:---:|
| GoPro (Deblurring) | PSNR/SSIM | Comparable/Superior | Restormer, etc. | 1.2×~3× Reduction |
| SIDD (Denoising) | PSNR/SSIM | Comparable | NAFNet, etc. | 2×~5× Reduction |
| Super-Resolution ×4 | PSNR/SSIM | Comparable | SwinIR, etc. | 5×~9.7× Reduction |

ECMA achieves comparable or superior performance to SOTA on various restoration tasks, while reducing computational cost by 1.2 to 9.7 times.

### Ablation Study

| Configuration | PSNR | FLOPs | Description |
|:---:|:---:|:---:|:---:|
| Fixed Conv (No LAM) | Baseline | Lowest | Lacks spatial adaptivity |
| Single-scale LAM (3×3) | +0.3dB | Slightly Increased | Adaptation is effective but lacks multi-scale |
| Cascaded LAM (7→5→3) | +0.6dB | Moderate | Multi-scale yields significant gains |
| Self-attention instead of LAM | +0.1dB | Highly Increased | Similar performance but much lower efficiency |

### Key Findings

- The dynamic convolution of LAM achieves comparable performance to self-attention in image restoration, but at a significantly lower computational cost (the advantage is even more pronounced with high-resolution inputs).
- The cascaded large-to-small kernel design is more efficient than parallel multi-scale branches, as it avoids redundant feature computation and complex fusion operations.
- In scenarios with highly restricted computational budgets (e.g., mobile deployment), ECMA's advantage is most prominent—achieving near-SOTA performance with only one-tenth of the computational cost.
- Different restoration tasks benefit to varying degrees from the multi-scale design: deblurring benefits the most (large-scale motion blur requires larger kernels), while denoising benefits the least (noise is highly localized).

## Highlights & Insights

1. **Replacing self-attention with dynamic convolution strikes an elegant balance**—retaining the critical advantage of local adaptivity while reducing computational complexity from $O(n^2)$ to $O(nk^2)$.
2. The multi-scale design of cascading from large to small kernels is simple yet effective, offering a more elegant alternative to common parallel multi-branch structures.
3. Unified validation across multiple restoration tasks demonstrates the versatility of the architecture—a single module design can handle diverse degradations, such as blur, noise, and low resolution.

## Limitations & Future Work

1. The kernel-generating network in dynamic convolution introduces its own computational and parameter overhead, which may require further compression in extremely lightweight scenarios.
2. Whether the cascade order from large to small kernels is optimal is worth questioning. Can an adaptive kernel size sequence be learned instead?
3. Currently, only single-type restoration is considered. Whether the architecture needs adjustment for hybrid degradations (such as simultaneous blur and noise) remains to be explored.
4. The performance gap compared to the latest efficient methods (e.g., NAFNet) in comparative experiments requires more detailed analysis.
5. Qualitative and quantitative evaluations on real-world degraded images (non-synthetic degradation) are still lacking.

## Related Work & Insights

- **Efficient Image Restoration**: NAFNet replaces self-attention with simple Channel Attention, while FFTformer reduces complexity using frequency domain attention. ECMA offers another efficient path based on dynamic convolution.
- **Dynamic Convolution Methods**: Methods like CondConv and DynamicConv have validated the effectiveness of dynamic kernels in image classification. Ours extends this to dense prediction tasks such as image restoration.
- **Multi-scale Image Restoration**: MPRNet and MIMO-UNet employ multi-stage multi-scale strategies, whereas ECMA achieves similar effects within a single stage by cascading kernel sizes.
- **Insights**: The design paradigm of dynamic convolution + cascaded multi-scales is worth exploring for other dense prediction tasks (e.g., semantic segmentation, optical flow estimation).

## Rating

- **Novelty**: ⭐⭐⭐ Although dynamic convolution and multi-scale learning are not individually novel, their cascaded combination is creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers three major restoration tasks (deblurring, denoising, and super-resolution) with comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐ The methodology is clear, and the motivation is well-reasoned.
- **Value**: ⭐⭐⭐⭐ Provides a valuable alternative in the efficiency-performance trade-off, showing strong practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Domain-Adaptive Video Deblurring via Test-Time Blurring](domain-adaptive_video_deblurring_via_test-time_blurring.md)
- [\[ICCV 2025\] Enhancing Image Restoration Transformer via Adaptive Translation Equivariance](../../ICCV2025/image_restoration/enhancing_image_restoration_transformer_via_adaptive_translation_equivariance.md)
- [\[ECCV 2024\] MambaIR: A Simple Baseline for Image Restoration with State-Space Model](mambair_a_simple_baseline_for_image_restoration_with_state-space_model.md)
- [\[ECCV 2024\] Seeing the Unseen: A Frequency Prompt Guided Transformer for Image Restoration](seeing_the_unseen_a_frequency_prompt_guided_transformer_for_image_restoration.md)
- [\[ECCV 2024\] Efficient Diffusion Transformer with Step-wise Dynamic Attention Mediators](efficient_diffusion_transformer_with_step-wise_dynamic_attention_mediators.md)

</div>

<!-- RELATED:END -->

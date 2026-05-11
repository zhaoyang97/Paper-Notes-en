---
title: >-
  [Paper Note] IM-LUT: Interpolation Mixing Look-Up Tables for Image Super-Resolution
description: >-
  [ICCV 2025][Image Restoration][Arbitrary-scale super-resolution] This paper proposes IM-LUT, which achieves arbitrary-scale image super-resolution by learning to mix multiple interpolation functions…
tags:
  - "ICCV 2025"
  - "Image Restoration"
  - "Arbitrary-scale super-resolution"
  - "look-up table"
  - "interpolation function mixing"
  - "lightweight inference"
  - "CPU-friendly"
date: 2026-05-08
content_hash: d1a648289a24fd49
---

# IM-LUT: Interpolation Mixing Look-Up Tables for Image Super-Resolution

**Conference**: ICCV 2025
**arXiv**: [2507.09923](https://arxiv.org/abs/2507.09923)
**Code**: N/A
**Area**: Image Restoration / Super-Resolution
**Keywords**: Arbitrary-scale super-resolution, look-up table, interpolation function mixing, lightweight inference, CPU-friendly

## TL;DR

This paper proposes IM-LUT, which achieves arbitrary-scale image super-resolution by learning to mix multiple interpolation functions, and converts the prediction network into a look-up table form to enable lightweight, fast CPU inference while maintaining reconstruction quality.

## Background & Motivation

**Background**: Image super-resolution (SR) is a fundamental vision task for enhancing image resolution. Mainstream methods include CNN-based approaches such as EDSR and RCAN, and Transformer-based methods such as SwinIR, which achieve strong results at fixed upscaling factors but typically incur substantial computational overhead. In recent years, look-up table (LUT) methods have emerged—precomputing and storing the input-output mappings of trained networks as LUTs, so that inference requires only table lookups rather than network computation, achieving extremely fast inference speeds.

**Limitations of Prior Work**: LUT methods suffer from two key limitations. First, existing LUT-SR methods (e.g., SR-LUT, MuLUT) support only fixed upscaling factors (e.g., ×2, ×4), requiring a separate LUT for each scale and being incapable of handling arbitrary factors (e.g., ×2.5, ×3.7). Second, arbitrary-scale image super-resolution (ASISR) methods (e.g., LIIF, LTE, CiaoSR) are based on implicit neural representations (INR), which are flexible but computationally and memory intensive, making them unsuitable for resource-constrained devices.

**Key Challenge**: LUT methods are fast but inflexible (fixed scale), while INR methods are flexible but slow. The challenge lies in supporting arbitrary-scale super-resolution while retaining the inference efficiency of LUTs.

**Goal**: To design a framework capable of handling arbitrary scale factors in super-resolution while achieving lightweight, fast CPU inference similar to LUT-based methods.

**Key Insight**: The key insight of this work is that standard interpolation functions (bilinear, bicubic, etc.) are inherently "arbitrary-scale," but any single interpolation function has limited expressive capacity. If the model can learn to adaptively mix multiple interpolation functions based on local image patterns and the target scale factor, high-quality arbitrary-scale super-resolution can be achieved without resorting to INR. Crucially, the mixing weight prediction network can be converted into a LUT.

**Core Idea**: Train an IM-Net to predict mixing weights over multiple interpolation functions, then convert IM-Net into IM-LUT (look-up table form) to enable lightweight arbitrary-scale super-resolution.

## Method

### Overall Architecture

IM-LUT operates in two stages: training and inference. In the training stage: (1) an IM-Net is constructed, taking as input the pixel values of a local image patch and the target scale factor, and outputting mixing weights over multiple interpolation functions; (2) the input image is upsampled separately using multiple predefined interpolation functions, yielding multiple candidate values; (3) IM-Net's predicted weights are used to compute a weighted sum of these candidates, producing the final SR result; (4) the entire pipeline is trained end-to-end. In the inference stage, the trained IM-Net is converted into IM-LUT, replacing network computation with quantized index-based table lookups for pure lookup-based inference.

### Key Designs

1. **IM-Net: Interpolation Mixing Weight Prediction Network**:

    - **Function**: Predicts optimal interpolation function mixing weights based on local image patterns and the target scale factor.
    - **Mechanism**: IM-Net takes as input the pixel values of an $R \times R$ local window centered on the target pixel (quantized to discrete values) and the target scale factor $s$. The network is a lightweight MLP that outputs $K$ weight values $\{w_1, ..., w_K\}$ (normalized via softmax), corresponding to $K$ predefined interpolation functions (e.g., nearest-neighbor, bilinear, bicubic, Lanczos). The final SR pixel value is computed as $y = \sum_{k=1}^{K} w_k \cdot f_k(x, s)$, where $f_k$ denotes the interpolation result of the $k$-th function applied to input $x$ at scale $s$. Although the network contains only a few thousand parameters, it substantially outperforms any single interpolation function by learning "when to use which interpolation."
    - **Design Motivation**: Rather than learning an implicit neural representation from scratch for arbitrary-scale mapping, this approach builds upon classical interpolation functions—which already handle arbitrary scales but with limited quality. By learning their optimal combination, significant quality improvements are achieved with minimal parameter overhead.

2. **IM-Net to IM-LUT Conversion**:

    - **Function**: Converts network inference into table lookup operations, enabling extremely fast CPU inference.
    - **Mechanism**: Since IM-Net's inputs are quantized local pixel values (each pixel quantized to $L$ levels) and a quantized scale factor (discretized into $S$ levels), the input space is finite and discrete. In an offline phase, all possible input combinations are enumerated and the corresponding IM-Net outputs are stored as a multi-dimensional look-up table, IM-LUT. During inference, given a local window and a target scale, mixing weights are retrieved directly from the table and then used to compute a weighted sum of precomputed interpolation candidates. Table lookup is $O(1)$ and extremely efficient. To prevent storage explosion, a grouped LUT strategy is adopted—the input window is split into multiple subgroups, each queried independently, and the results are subsequently combined.
    - **Design Motivation**: LUT conversion is central to the practical utility of the method. Although the network itself is small, it still requires matrix operations, whereas LUT access is a pure memory operation that is highly CPU-friendly. The grouping strategy is a classic solution to the exponential growth in LUT size caused by high-dimensional inputs.

3. **Scale Factor Encoding and Continuity**:

    - **Function**: Enables IM-LUT to handle continuous arbitrary scale factors rather than being restricted to discrete grid points.
    - **Mechanism**: The target scale factor $s$ is discretized into $S$ representative values, and a LUT incorporating scale information is constructed. For scale factors not seen during training, linear interpolation is performed between the two nearest discrete scale entries in the LUT to obtain the corresponding mixing weights. Thus, although IM-LUT is stored discretely, it supports arbitrary continuous scales through interpolation along the scale dimension. The scale factor encoding is incorporated into IM-Net's input, enabling the network to learn scale-aware interpolation mixing strategies.
    - **Design Motivation**: The core challenge of arbitrary-scale SR lies in the word "arbitrary"—storing a separate LUT for every possible scale is infeasible. Continuous interpolation along the scale dimension provides an elegant compromise, covering a continuous scale space with a small number of discrete-scale LUTs.

### Loss & Training

Training employs the standard L1 pixel loss $\mathcal{L} = |y_{pred} - y_{gt}|_1$, with joint training across multiple randomly sampled scale factors. Standard SR training data settings are followed, using the DIV2K training set with random patch cropping and random scale factor sampling (e.g., uniform sampling in the range ×1.5 to ×4). After training IM-Net, IM-LUT is constructed offline.

## Key Experimental Results

### Main Results

PSNR (dB) comparison for arbitrary-scale super-resolution on standard benchmarks:

| Method | Type | Set5 ×2 | Set5 ×3 | Set5 ×4 | Set14 ×2 | B100 ×4 | Inference Time (ms) | Params |
|--------|------|---------|---------|---------|----------|---------|-------------------|--------|
| Bicubic | Interpolation | 33.66 | 30.39 | 28.42 | 30.24 | 25.96 | <1 | 0 |
| LIIF | INR | 37.99 | 34.68 | 32.19 | 33.69 | 32.15 | ~120 | 1.2M |
| LTE | INR | 38.13 | 34.78 | 32.30 | 33.79 | 32.22 | ~130 | 1.3M |
| CiaoSR | INR | 38.22 | 34.86 | 32.39 | 33.85 | 32.28 | ~150 | 1.5M |
| SR-LUT (×2 only) | LUT | 36.42 | - | - | 32.56 | - | ~5 | - |
| **IM-LUT** | **LUT** | **37.15** | **34.05** | **31.62** | **33.12** | **31.68** | **~8** | **~10K** |

### Ablation Study

| Configuration | Set5 ×2 PSNR | Set5 ×4 PSNR | Inference Time | Notes |
|--------------|-------------|-------------|---------------|-------|
| IM-LUT (Full) | 37.15 | 31.62 | ~8ms | Full method |
| Bilinear only | 33.66 | 28.42 | <1ms | Most basic baseline |
| Bicubic only | 34.89 | 29.56 | <1ms | Single interpolation ceiling |
| Mix 2 functions | 36.28 | 30.85 | ~6ms | 2 functions already significant |
| Mix 4 functions | 37.15 | 31.62 | ~8ms | Optimal with 4 functions |
| Mix 6 functions | 37.18 | 31.65 | ~12ms | Diminishing returns |
| w/o scale factor input | 36.52 | 30.91 | ~8ms | Scale-unaware; performance drops |
| IM-Net (no LUT conversion) | 37.21 | 31.68 | ~25ms | Marginally better quality, 3× slower |

### Key Findings

- Increasing the number of interpolation functions from 2 to 4 yields significant gains (+0.87 dB), while increasing from 4 to 6 provides negligible improvement (+0.03 dB), indicating that 4 interpolation functions represent the optimal efficiency–quality trade-off.
- The PSNR gap between IM-LUT and IM-Net is minimal (<0.06 dB), demonstrating that the accuracy loss from LUT quantization is negligible.
- Scale factor encoding is critical for cross-scale generalization—without scale information as input, the model cannot adaptively adjust its interpolation strategy across different scales, resulting in a substantial performance drop at ×4.
- Compared to INR-based methods, IM-LUT achieves 15–20× faster inference with two orders of magnitude fewer parameters, at a PSNR cost of approximately 0.5–0.8 dB, representing a highly attractive efficiency–quality trade-off.
- The advantage is even more pronounced for CPU inference—INR methods are extremely slow on CPU (>1s), whereas IM-LUT requires only ~8ms.

## Highlights & Insights

- **The "standing on the shoulders of interpolation functions" approach is elegant**: Rather than learning a mapping from scratch, the model learns how to optimally combine existing interpolation functions. This philosophy of "composing existing tools rather than reinventing them" offers valuable inspiration for algorithm design, particularly in resource-constrained scenarios.
- **LUT conversion makes the method highly deployment-friendly**: Pure table lookup operations require neither a GPU nor matrix computation libraries, enabling deployment on embedded devices, mobile phones, and other resource-constrained platforms. The paradigm of "using a network at training time, using a LUT at inference time" is transferable to other tasks demanding extreme inference efficiency.
- **Continuous interpolation along the scale dimension is an elegant solution to "arbitrary" scale**: Using a finite set of discrete LUTs to cover an infinite continuous scale space avoids the need to train or store separate resources for each scale factor.

## Limitations & Future Work

- A PSNR gap of ~0.5–0.8 dB relative to state-of-the-art INR methods remains, which may be insufficient for applications requiring the highest possible quality.
- LUT storage requirements grow exponentially with the size of the input window, limiting the use of larger receptive fields—which generally lead to better quality.
- Only single-image super-resolution is addressed; temporal information in video super-resolution has not been explored.
- The set of interpolation functions is manually predefined—it remains an open question whether learning custom basis functions could further improve quality.
- Future work could explore combining IM-LUT with perceptual losses (LPIPS) or GAN losses to improve visual quality beyond PSNR, or extending the IM-LUT paradigm to other image processing tasks (denoising, deblurring, etc.).

## Related Work & Insights

- **vs. SR-LUT / MuLUT**: These are fixed-scale LUT-SR methods. By introducing the scale factor as an additional dimension of the LUT, IM-LUT is the first to bring arbitrary-scale super-resolution capability to LUT-based methods.
- **vs. LIIF / LTE**: INR-based ASISR methods that achieve higher quality but are two orders of magnitude slower in inference. IM-LUT is the preferable choice under extreme efficiency constraints, particularly for CPU-only deployment scenarios.
- **vs. MetaSR**: MetaSR employs meta-learning to generate scale-specific upsampling modules. IM-LUT is more lightweight—it does not require dynamic generation of network weights, only table lookups to obtain mixing weights.
- The "classical methods + learned combination" paradigm introduced in this paper suggests a new direction: for tasks with well-established classical algorithms that fall short in quality, learning the optimal combination of those classical algorithms may be more efficient than training a network from scratch.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of interpolation mixing and LUT conversion is novel, though each individual component (LUT conversion, interpolation function mixing) has been explored separately in prior work.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Coverage of multiple standard benchmarks with thorough ablation studies; perceptual quality metrics (e.g., LPIPS) are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ The method is described clearly with comprehensive experimental comparisons; figures aid understanding.
- **Value**: ⭐⭐⭐⭐ Provides a practical solution for arbitrary-scale super-resolution on resource-constrained devices; CPU inference speed is impressive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ShiftLUT: Spatial Shift Enhanced Look-Up Tables for Efficient Image Restoration](../../CVPR2026/image_restoration/shiftlut_spatial_shift_enhanced_look-up_tables_for_efficient_image_restoration.md)
- [\[ICCV 2025\] Outlier-Aware Post-Training Quantization for Image Super-Resolution](outlier-aware_post-training_quantization_for_image_super-resolution.md)
- [\[ICCV 2025\] Emulating Self-Attention with Convolution for Efficient Image Super-Resolution](emulating_self-attention_with_convolution_for_efficient_image_super-resolution.md)
- [\[ICCV 2025\] Lightweight and Fast Real-time Image Enhancement via Decomposition of the Spatial-aware Lookup Tables](lightweight_and_fast_real-time_image_enhancement_via_decomposition_of_the_spatia.md)
- [\[ICCV 2025\] Benchmarking Burst Super-Resolution for Polarization Images: Noise Dataset and Analysis](benchmarking_burst_super-resolution_for_polarization_images_noise_dataset_and_an.md)

</div>

<!-- RELATED:END -->

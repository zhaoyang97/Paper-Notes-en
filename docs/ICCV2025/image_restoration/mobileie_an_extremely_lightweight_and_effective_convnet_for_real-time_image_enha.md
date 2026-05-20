---
title: >-
  [Paper Note] MobileIE: An Extremely Lightweight and Effective ConvNet for Real-Time Image Enhancement on Mobile Devices
description: >-
  [ICCV 2025][Image Restoration][Mobile image enhancement] This paper proposes MobileIE, an extremely lightweight CNN framework with approximately 4K parameters…
tags:
  - "ICCV 2025"
  - "Image Restoration"
  - "Mobile image enhancement"
  - "re-parameterization"
  - "lightweight CNN"
  - "real-time inference"
  - "attention mechanism"
date: 2026-05-08
content_hash: f9f39e40269d02dc
---

# MobileIE: An Extremely Lightweight and Effective ConvNet for Real-Time Image Enhancement on Mobile Devices

**Conference**: ICCV 2025
**arXiv**: [2507.01838](https://arxiv.org/abs/2507.01838)  
**Code**: [https://github.com/AVC2-UESTC/MobileIE.git](https://github.com/AVC2-UESTC/MobileIE.git)  
**Area**: Image Enhancement / Image Restoration
**Keywords**: Mobile image enhancement, re-parameterization, lightweight CNN, real-time inference, attention mechanism

## TL;DR

This paper proposes MobileIE, an extremely lightweight CNN framework with approximately 4K parameters, which achieves real-time image enhancement at over 1100 FPS on mobile devices for the first time. This is accomplished through multi-branch re-parameterizable convolution (MBRConv), a feature self-transformation (FST) module, hierarchical dual-path attention (HDPA), and an incremental weight optimization (IWO) strategy. MobileIE achieves state-of-the-art speed–performance trade-offs across three tasks: low-light enhancement, underwater enhancement, and ISP.

## Background & Motivation

Deep learning-based image enhancement (IE) models have achieved remarkable quality improvements, yet **deploying them on resource-constrained mobile devices** remains a significant challenge:

- **Transformer/Diffusion methods**: The computational overhead of self-attention and iterative diffusion is prohibitive for mobile deployment.
- **Existing lightweight models**: While reducing FLOPs, they often sacrifice enhancement quality and are typically designed for specific degradation types.
- **High-resolution demands**: Growing user demand for high-resolution images further intensifies the computational burden on mobile hardware.

The core philosophy is that **mobile IE should strike a balance between speed and performance**, employing general-purpose architectures and hardware-friendly operators. Training and inference should be decoupled — complex multi-branch structures are used during training for feature learning, and re-parameterized into a single convolution at inference for maximum efficiency.

## Method

### Overall Architecture

The MobileIE architecture is highly compact:
Shallow feature extraction (MBRConv5×5 + PReLU) → Deep feature extraction (2× MBRConv3×3 + FST) → Attention (HDPA) → Output refinement (MBRConv3×3)

At inference, all MBRConv branches are re-parameterized into standard convolutions, yielding a model of approximately 4K parameters.

### Key Designs

1. **Multi-Branch Re-parameterizable Convolution (MBRConv)**:

    - During training, multiple parallel branches with different kernel sizes capture multi-scale features.
    - Each branch incorporates **parallel Batch Normalization** — while BN offers limited benefit for IE tasks in isolation, it enhances nonlinearity and can be merged into the convolution at inference.
    - Branch outputs are concatenated and projected to the target dimension via a Conv 1×1.
    - At inference, all branches are re-parameterized into a single standard convolution with zero additional overhead.
    - **Key innovation**: Unlike methods such as RepVGG, the parallel BN in MBRConv simultaneously preserves both smoothed and original features, improving robustness across data distributions.

2. **Incremental Weight Optimization (IWO)**:

    - Addresses **performance stagnation** in compact networks during late-stage training.
    - $W_{final} = \text{Frozen}(W_{pre}) + W_{learn}$
    - $W_{pre}$: Optimal weights from the earlier training phase (frozen), providing a stable initial feature representation.
    - $W_{learn}$: Dynamically updated weights that refine task-specific details.
    - Effect: Enhances skeleton features (center rows/columns) of convolutional kernels, reduces inter-channel redundancy (verified by significantly increased KL divergence), and breaks through training convergence bottlenecks.

3. **Feature Self-Transformation Module (FST)**:

    - $\text{FST}(x) = Scale \cdot (x * x) + bias$
    - Captures higher-order nonlinear feature relationships through quadratic interaction, compensating for the limited expressiveness of linear convolutions.
    - Learnable Scale and bias adaptively adjust the dynamic range of features.
    - In the frequency domain, the squaring operation is more sensitive to high-frequency information than ReLU, better preserving edges and fine details.
    - Computationally inexpensive, making it well-suited for lightweight models.

4. **Hierarchical Dual-Path Attention (HDPA)**:

    - Global path: Adaptive AvgPool → MBRConv1×1 → Sigmoid → channel attention weights $A_g$
    - Local path: Globally weighted features → MaxPool → MBRConv1×1 → Sigmoid → local attention weights $A_l$
    - Final output: $\hat{F} = (A_g * A_l) * F$
    - The dual-path design enables mutual optimization during backpropagation, capturing global context and local details in a hierarchical manner.

### Loss & Training

**Local Variance Weighted (LVW) Loss**:
- Computes the per-pixel prediction error $\Delta_{m,n} = \|O_{m,n} - L_{m,n}\|_1$
- Computes local mean $\mu$ and variance $\sigma^2$ over the spatial dimension
- Weight $W_\Delta = \text{Tanh}(\frac{|\Delta_{m,n} - \mu_{m,n}|}{\sigma_{m,n} + \epsilon})$
- Final loss $\mathcal{L}_{LVW} = \frac{1}{HW}\sum(W_\Delta \cdot \Delta_{m,n})$
- Advantage: Avoids the over-sensitivity of L2 to extreme pixels, while handling outliers more effectively than L1, and dynamically adjusting the contribution of each pixel.

Training setup: Adam optimizer, cosine annealing learning rate (initial 0.001), reset every 50 epochs, 2000 total epochs, 10-epoch warm-up.

## Key Experimental Results

### Main Results

**Low-Light Enhancement (LOLv1 + LOLv2-Real)**

| Method | Params | GPU Latency (ms) | SoC Latency (ms) | LOLv1 PSNR | LOLv2 PSNR |
|--------|--------|------------------|------------------|------------|------------|
| IAT | 86.9K | 6.204 | 202.33 | 23.38 | 25.46 |
| SYELLE | 5.3K | 0.944 | 7.73 | 21.03 | 21.26 |
| Zero-DCE++ | 10.6K | 1.974 | 57.91 | 14.68 | 17.23 |
| **MobileIE** | **4.0K** | **0.895** | **6.72** | **23.62** | **25.08** |

**Underwater Image Enhancement (UIEB)**

| Method | Params | SoC Latency (ms) | PSNR | SSIM |
|--------|--------|------------------|------|------|
| FiveA+ | 9.0K | 423.43 | 22.51 | 0.902 |
| Boths | 6.4K | 58.04 | 22.23 | 0.904 |
| **MobileIE** | **4.0K** | **8.94** | **22.81** | **0.906** |

**ISP (ZRR)**

| Method | Params | SoC Latency (ms) | PSNR | SSIM |
|--------|--------|------------------|------|------|
| SYEISP | 5.6K | 16.47 | 20.84 | 0.728 |
| NAFNet | 7.8K | 78.62 | 21.12 | 0.736 |
| **MobileIE** | **4.1K** | **14.40** | **21.43** | **0.731** |

### Ablation Study

**Re-parameterization and Loss Function Ablation (UIEB)**

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ |
|---------------|-------|-------|--------|
| Inference network only (no re-param.) | 21.48 | 0.887 | 0.192 |
| L1 loss | 22.20 | 0.902 | 0.168 |
| L2 loss | 21.74 | 0.894 | 0.175 |
| Charbonnier loss | 22.31 | 0.905 | 0.162 |
| **LVW loss** | **22.57** | **0.906** | **0.160** |
| RepVGG | 22.69 | 0.821 | 0.202 |
| ECBSR | 23.96 | 0.816 | 0.204 |
| MBRConv (w/o BN) | 23.27 | 0.821 | 0.232 |
| MBRConv (w/o IWO) | 24.02 | 0.823 | 0.199 |
| **MBRConv + IWO (full)** | **24.35** | **0.829** | **0.189** |

**Attention Mechanism Comparison (LOLv1)**

| Attention Method | Params | PSNR | SSIM |
|-----------------|--------|------|------|
| SE-Net | 4.2K | 22.27 | 0.804 |
| CBAM | 4.3K | 22.38 | 0.796 |
| ECA-Net | 3.9K | 21.79 | 0.799 |
| **HDPA (Ours)** | **4.0K** | **23.62** | **0.812** |

### Key Findings

- MobileIE is the first IE model to achieve **over 1100 FPS** (at 600×400 resolution) on a mobile device.
- With only 4K parameters, it achieves enhancement quality comparable to or better than models with 20× more parameters.
- The IWO strategy effectively breaks late-stage training stagnation — training loss continues to decrease after IWO is applied.
- IWO strengthens the skeleton features (center rows/columns) of convolutional kernels and significantly reduces inter-channel redundancy (verified via KL divergence visualization).
- The quadratic transformation in FST exhibits stronger high-frequency response than ReLU in the frequency domain, better preserving image details.
- LVW loss outperforms all other tested loss functions (L1, L2, Smooth L1, Charbonnier, Robust Loss).

## Highlights & Insights

- **Extreme minimalism**: The entire model architecture requires only a few convolutions and lightweight modules, demonstrating the effectiveness of "less is more" for mobile IE.
- **Elegant training–inference decoupling**: The multi-branch structure during training provides rich feature learning capacity, while re-parameterization at inference ensures maximum efficiency.
- **The IWO strategy** is generalizable to other compact model training scenarios, and is essentially a form of self-knowledge distillation (distilling from the model's own earlier state).
- **Three distinct IE tasks** (low-light, underwater, ISP) are addressed with a single architecture, validating the generality of the proposed method.

## Limitations & Future Work

- Although PSNR surpasses other lightweight methods, a gap remains compared to large-scale models (e.g., DDNet).
- The method has not been validated on other low-level vision tasks such as denoising and dehazing.
- The extreme compression to 4K parameters may be insufficient for more complex degradation scenarios (e.g., mixed degradations).
- The optimal freezing point in IWO is currently fixed at 1000 epochs; an adaptive scheduling strategy may yield better results.
- SoC latency has only been tested on the Snapdragon 8 Gen 3; validation on additional mobile platforms is needed.

## Related Work & Insights

- **RepVGG / ACNet / DBB**: Pioneering re-parameterization works; MBRConv builds upon these by adding parallel BN and IWO.
- **StarNet**: The star operation inspired the quadratic interaction design in FST.
- **VanillaNet**: Its minimalist design philosophy is closely aligned with MobileIE.
- **SYELLE** (ICCV'23): A lightweight IE method in the same track and the most direct point of comparison for MobileIE.
- **NTIRE Challenge**: Multiple efficient IE submissions have employed re-parameterization structures.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The IWO strategy and FST module are novel and effective; achieving 1100 FPS with 4K parameters is impressive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across three IE tasks; ablations cover every module, loss function, and attention mechanism, supplemented by visualization and frequency-domain analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with intuitive explanations and theoretical analysis for each module, supported by rich visualizations.
- **Value**: ⭐⭐⭐⭐⭐ — Offers direct practical value for mobile AI deployment; the extreme efficiency of 4K parameters provides a new option for mobile developers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Lightweight and Fast Real-time Image Enhancement via Decomposition of the Spatial-aware Lookup Tables](lightweight_and_fast_real-time_image_enhancement_via_decomposition_of_the_spatia.md)
- [\[ICCV 2025\] Learning Pixel-adaptive Multi-layer Perceptrons for Real-time Image Enhancement](learning_pixel-adaptive_multi-layer_perceptrons_for_real-time_image_enhancement.md)
- [\[ICCV 2025\] CWNet: Causal Wavelet Network for Low-Light Image Enhancement](cwnet_causal_wavelet_network_for_low-light_image_enhancement.md)
- [\[ICCV 2025\] Low-Light Image Enhancement using Event-Based Illumination Estimation (RetinEV)](low-light_image_enhancement_using_event-based_illumination_estimation.md)
- [\[ICCV 2025\] Self-Calibrated Variance-Stabilizing Transformations for Real-World Image Denoising](self-calibrated_variance-stabilizing_transformations_for_real-world_image_denois.md)

</div>

<!-- RELATED:END -->

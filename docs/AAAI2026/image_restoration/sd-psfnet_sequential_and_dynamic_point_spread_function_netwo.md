---
title: >-
  [Paper Note] SD-PSFNet: Sequential and Dynamic Point Spread Function Network for Image Deraining
description: >-
  [AAAI 2026][Image Restoration][image deraining] SD-PSFNet is a cascaded CNN-based deraining network driven by a dynamic PSF mechanism. It models the optical effects of raindrops via a multi-scale learnable PSF dictionary, combined with a sequential restoration architecture featuring adaptive gated fusion. The method achieves SOTA performance of 33.12 dB on Rain100H and 42.28 dB on RealRain-1k-L, yielding a cumulative gain of 5.04 dB (13.5%) over the baseline MPRNet.
tags:
  - "AAAI 2026"
  - "Image Restoration"
  - "image deraining"
  - "point spread function"
  - "physics-aware"
  - "multi-stage restoration"
  - "dynamic filtering"
date: 2026-05-08
content_hash: 2ea712b5fd7f0aeb
---

# SD-PSFNet: Sequential and Dynamic Point Spread Function Network for Image Deraining

**Conference**: AAAI 2026
**arXiv**: [2511.17993](https://arxiv.org/abs/2511.17993)  
**Code**: [https://github.com/Aster-1024/SD-PSFNet](https://github.com/Aster-1024/SD-PSFNet)  
**Area**: Image Deraining / Physics-Aware Image Restoration
**Keywords**: image deraining, point spread function, physics-aware, multi-stage restoration, dynamic filtering

## TL;DR

SD-PSFNet is a cascaded CNN-based deraining network driven by a dynamic PSF mechanism. It models the optical effects of raindrops via a multi-scale learnable PSF dictionary, combined with a sequential restoration architecture featuring adaptive gated fusion. The method achieves SOTA performance of 33.12 dB on Rain100H and 42.28 dB on RealRain-1k-L, yielding a cumulative gain of 5.04 dB (13.5%) over the baseline MPRNet.

## Background & Motivation

Image deraining is a fundamental low-level vision task critical to downstream applications such as object detection, autonomous driving, and surveillance systems. Existing methods face three core challenges:

**Lack of Physical Modeling**: Most deep learning methods learn input-output mappings purely from data, neglecting the optical physical properties of raindrops (orientation, density, refraction, etc.), which prevents dynamic adaptation to varying rain types, results in incomplete deraining, and lacks interpretability.

**Limitations of Conventional PSF**: Explicit physical modeling relies on fixed prior assumptions (e.g., static PSF templates), which cannot capture the multi-scale distribution and optical variability of rain degradation.

**Efficiency Bottleneck**: Although Transformer/GAN/Diffusion-based methods deliver strong performance, they impose large parameter counts (Restormer 26.10M, HINet 88.67M), making real-time deployment prohibitive.

Core insight: Transform PSF from a static preset into a **data-driven learnable form**, approximating spatially varying degradation functions with a finite-dimensional dictionary of degradation modes.

## Method

### Overall Architecture

SD-PSFNet adopts a three-stage sequential restoration architecture (Stage In → $\tau$ Stage Mid → ORStage), inspired by the sequential modeling concept of LSTM and the multi-stage design of MPRNet. Unlike MPRNet's spatial splitting strategy, each stage in SD-PSFNet processes the full image and dynamically predicts PSF features within a UNet-style multi-scale downsampling structure. Across stages, physical priors and feature information are propagated via adaptive gated fusion.

### Key Designs

1. **Dynamic PSF Mechanism**: The spatially varying degradation function is modeled as a linear combination of a learnable degradation mode dictionary: $K(x,y) \approx \sum_{j=1}^{K_c} w_j(x,y) \cdot k_j$, where $k_j$ are learnable 2D convolutional kernels representing basic degradation modes, and $w_j$ are adaptively mapped from image features by a neural network.
2. **Multi-Scale PSF Head**: Three prediction heads with kernels of size 3×3, 5×5, and 7×7 capture high-frequency detail degradation, balanced local-global information, and macro low-frequency patterns, respectively. After adaptive fusion via a Channel Attention Block (CAB), a 1×1 convolution projects the output into a $K_c=40$-channel PSF representation, with spatial normalization applied per channel to ensure energy conservation.
3. **PSF-Aware Attention**: A dual-path mechanism — (a) Channel Modulation: a PSF encoder extracts compact features to generate $\gamma, \beta$ parameters, with $x_{mod} = x \odot \gamma + \beta$; (b) Spatial Attention: a single-channel PSF is upsampled to feature resolution and combined with the modulated features to produce spatial weights that highlight regions of varying degradation severity.
4. **Gated Cross-Stage Fusion**: $F^{(t)} = G_\theta \cdot F_{current} + (1-G_\theta) \cdot F_{prev}$, applied at three locations — adaptive fusion of shallow features at stage input, hierarchical encoder feature updates, and an enhanced CSFF dual-gate unit for precise control of cross-stage information flow.

### Loss & Training

$$\mathcal{L}_{total} = \sum_{s=1}^{\tau+1} (\mathcal{L}_{char}(I_s, T) + 0.05 \cdot \mathcal{L}_{edge}(I_s, T) + 0.01 \cdot \mathcal{L}_{freq}(I_s, T))$$

The Charbonnier loss provides pixel-level supervision, the edge-aware loss preserves high-frequency details, and the frequency-domain loss aligns PSF degradation characteristics. Training uses the AdamW optimizer with lr=1e-4, 3-epoch linear warmup followed by cosine annealing to 1e-6, FP16 mixed precision, and gradient clipping at 2.0. Patch size is 128×128, training runs for 2000 epochs on a single RTX 4090.

## Key Experimental Results

### Main Results: Comparison with SOTA Methods

| Method | Type | Params (M) | Rain100L PSNR | Rain100H PSNR | RealRain-1k-L PSNR | RealRain-1k-H PSNR |
|--------|------|------------|---------------|---------------|---------------------|---------------------|
| MPRNet | CNN | 3.64 | 36.40 | 30.41 | 36.29 | 34.74 |
| HINet | CNN | 88.67 | 37.28 | 30.65 | **41.98** | **40.82** |
| M3SNet | CNN | 16.70 | 40.04 | 30.64 | 41.55 | 40.01 |
| Restormer | Transformer | 26.10 | 38.99 | 31.46 | 40.90 | 39.57 |
| DRSFormer | Transformer | 33.66 | 41.32 | 32.07 | 41.52 | 40.21 |
| NeRD-Rain-S | Transformer | 10.53 | **42.00** | **32.86** | 38.64 | 36.69 |
| **SD-PSFNet** | **CNN** | **9.63** | 41.47 | **33.12** | **42.28** | **41.08** |

### Ablation Study: Incremental Component Contributions (RealRain-1k-L)

| Model Configuration | PSNR (dB) | SSIM | Gain |
|--------------------|-----------|------|------|
| MPRNet (Baseline) | 37.24 | 0.9754 | — |
| + Gate Mechanism | 38.65 | 0.9773 | +1.41 |
| + Hierarchical Update | 40.41 | 0.9792 | +1.76 |
| + Enhanced CSFF | 41.41 | 0.9850 | +1.00 |
| + 1-channel PSF | 41.63 | 0.9855 | +0.22 |
| + 40-channel PSF (Ours) | 42.28 | 0.9872 | +0.65 |
| **Cumulative Gain** | | | **+5.04** |

### Key Findings

- **Effect of Stage Count $\tau$**: Performance increases monotonically from $\tau=0$ (40.81 dB, 3.64M) to $\tau=3$ (41.54 dB, 9.63M); the CNN architecture manages parameter growth efficiently.
- **Cross-Domain Generalization**: Training on Rain100H and testing on RealRain-1k-L yields 26.98 dB, surpassing Restormer (26.59 dB) and NeRD-Rain-S (26.67 dB) trained on Rain13K.
- **Synthetic-to-Real Domain Gap**: Models trained on synthetic data exhibit severe performance degradation on real-world test sets (e.g., training on Rain100L → only 27.54 dB on RealRain-1k-L), reflecting that data-driven methods tend to learn dataset-specific features rather than generalizable deraining principles.

## Highlights & Insights

- **CNN + Physical Priors Can Match Transformers**: A 9.63M-parameter CNN surpasses HINet (88.67M) and Restormer (26.10M) on real-world datasets, demonstrating the value of physics-informed modeling.
- **Elegant Design of the Dynamic PSF Dictionary**: The combination of a finite-dimensional dictionary with data-driven weight assignment achieves both physical interpretability (each basis kernel corresponds to a specific degradation mode) and data adaptability.
- **Information Flow Control in Sequential Restoration**: The dual-gate CSFF mechanism precisely governs the flow of shallow-level details and deep-level semantics during cross-stage feature propagation.

## Limitations & Future Work

- The synthetic-to-real generalization gap remains notable; the linear degradation assumption of PSF may be insufficient for non-linear degradations (e.g., combined rain and haze).
- Temporal consistency in video deraining scenarios has not been validated.
- At $\tau=3$, MACs reach 244.83G, which may not offer an efficiency advantage over lightweight Transformers such as DRSFormer.

## Related Work & Insights

| Method Category | Representative Methods | Characteristics | Limitations |
|----------------|----------------------|-----------------|-------------|
| Early CNN | DerainNet, DDN | Local feature extraction | Limited receptive field |
| Multi-Scale CNN | RESCAN, DID-MDN | Multi-resolution feature fusion | Lacks physical modeling |
| Transformer | Restormer, DRSFormer | Long-range dependency modeling | Large parameter count, difficult to deploy |
| GAN/Diffusion | SSCGAN, DCDGAN | Generates perceptually realistic results | Mode collapse, unstable training |
| **SD-PSFNet** | **Ours** | **Physical PSF + Sequential Restoration** | **The only CNN achieving SOTA on both synthetic and real datasets simultaneously** |

## Rating

- Novelty: ⭐⭐⭐⭐ The dynamic PSF mechanism organically integrates physical modeling with data-driven learning.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive ablations, cross-domain evaluation, and feature visualization.
- Writing Quality: ⭐⭐⭐⭐ The correspondence between the physical model and network design is clearly articulated.
- Value: ⭐⭐⭐⭐ Demonstrates that CNN + physical priors can be competitive with Transformers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MFmamba: A Multi-function Network for Panchromatic Image Resolution Restoration Based on State-Space Model](mfmamba_a_multi-function_network_for_panchromatic_image_resolution_restoration_b.md)
- [\[AAAI 2026\] SpatioTemporal Difference Network for Video Depth Super-Resolution](spatiotemporal_difference_network_for_video_depth_super-resolution.md)
- [\[CVPR 2026\] Dynamic Exposure Burst Image Restoration](../../CVPR2026/image_restoration/dynamic_exposure_burst_image_restoration.md)
- [\[CVPR 2026\] Unpaired Image Deraining Using Reward-Guided Self-Reinforcement Strategy](../../CVPR2026/image_restoration/unpaired_image_deraining_using_reward-guided_self-reinforcement_strategy.md)
- [\[CVPR 2026\] DVAR: Dynamic Visual Autoregressive Modeling for Image Super-Resolution](../../CVPR2026/image_restoration/dvar_dynamic_visual_autoregressive_modeling_for_image_super-resolution.md)

</div>

<!-- RELATED:END -->

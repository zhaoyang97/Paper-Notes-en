---
title: >-
  [Paper Note] TinyViM: Frequency Decoupling for Tiny Hybrid Vision Mamba
description: >-
  [ICCV 2025][Segmentation][Lightweight visual backbone] This paper proposes TinyViM, a lightweight convolution-Mamba hybrid visual backbone based on frequency decoupling. A Laplace Mixer routes low-frequency components to…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "Lightweight visual backbone"
  - "Mamba"
  - "frequency decoupling"
  - "Laplacian pyramid"
  - "high-low frequency separation"
date: 2026-05-08
content_hash: ceb9840896f92795
---

# TinyViM: Frequency Decoupling for Tiny Hybrid Vision Mamba

**Conference**: ICCV 2025
**arXiv**: [2411.17473](https://arxiv.org/abs/2411.17473)
**Code**: [GitHub](https://github.com/xwmaxwma/TinyViM)
**Area**: Image Segmentation
**Keywords**: Lightweight visual backbone, Mamba, frequency decoupling, Laplacian pyramid, high-low frequency separation

## TL;DR

This paper proposes TinyViM, a lightweight convolution-Mamba hybrid visual backbone based on frequency decoupling. A Laplace Mixer routes low-frequency components to Mamba for global context modeling and enhances high-frequency components via depthwise convolution. A frequency ramp Inception structure progressively adjusts frequency allocation across stages. TinyViM achieves 2–3× higher throughput than existing Mamba models on classification, detection, and segmentation tasks.

## Background & Motivation

Mamba has attracted considerable attention in vision due to its linear-complexity global modeling capability, with methods such as ViM and VMamba demonstrating competitive performance on image classification. However, existing lightweight Mamba backbones (e.g., EfficientVMamba) **fail to compete with lightweight CNN- or Transformer-based counterparts** in terms of both performance and efficiency.

Through spectral analysis, the authors identify a key phenomenon: **in convolution-Mamba hybrid architectures, Mamba primarily models low-frequency information while suppressing high-frequency information**. Specifically:
- Low-frequency components at the spectral center are amplified after Mamba processing.
- High-frequency details such as edges and textures are attenuated.

This observation leads to two inferences:
1. Feeding all frequency components into the Mamba block is **inefficient** — high-frequency information constitutes unnecessary overhead for Mamba.
2. Uniformly processing low-frequency information across all stages **degrades high-frequency components**, harming fine-grained recognition.

Core motivation: since Mamba naturally favors low frequencies, it is more principled to **explicitly decouple high and low frequencies** — letting Mamba handle only the low-frequency components (at lower resolution, thus lower cost) while efficiently enhancing high-frequency components with convolutions.

## Method

### Overall Architecture

TinyViM follows a four-stage multi-scale design. Each stage contains Local Blocks (re-parameterized 3×3 convolution + FFN) and TinyViM Blocks (Laplace Mixer + FFN). Patch Embedding performs downsampling and channel expansion between stages.

### Key Design 1: Quantitative Validation of Frequency Decoupling

A baseline combining convolution and standard Mamba (SS2D) is constructed and compared against several input variants:

| Input Variant | GMACs | Throughput | Top-1 (%) |
|---|:---:|:---:|:---:|
| Baseline (all frequencies) | 0.96 | 1673 | 79.1 |
| Low frequency only | 0.93 | 2574 | 79.0 |
| High frequency only | 0.96 | 1377 | 78.6 |
| High + Low (parallel) | 0.97 | 1509 | 79.1 |

Using only low-frequency input incurs negligible accuracy loss (79.0 vs. 79.1) while achieving 1.5× throughput improvement, validating the frequency decoupling strategy.

### Key Design 2: Laplace Mixer

Given input features $X \in \mathbb{R}^{H \times W \times D}$, the channel dimension is split by ratio $\alpha$ into a low-frequency input $X_l$ and a high-frequency input $X_h$.

**Low-frequency branch**: Laplacian pyramid decomposition is applied:
$$X_{ll} = \text{Pool}(X_l), \quad X_{lh} = X_l - \text{Upsample}(X_{ll})$$

The low-frequency component $X_{ll}$, at $\frac{1}{2}$ the original resolution, is fed into SS2D (VMamba's 2D selective scan) for global context modeling:
$$\hat{X}_{ll} = \text{SS2D}(X_{ll})$$

**High-frequency branch**: The high-frequency residual $X_{lh}$ from the low-frequency branch and $X_h$ are concatenated and enhanced via a re-parameterized 3×3 depthwise convolution:
$$\hat{X}_{hh} = \text{Rep}_3(X_{hh})$$

Corresponding high- and low-frequency features are summed element-wise and fused through a 1×1 convolution.

### Key Design 3: Frequency Ramp Inception

Motivated by two observations: (1) deep-layer features exhibit redundancy; and (2) shallow layers require more high-frequency detail while deep layers benefit more from global information, the paper proposes progressively adjusting the split ratio $\alpha$ per stage — allocating more channels to the high-frequency branch (smaller $\alpha$) in shallow stages and more to the low-frequency branch (larger $\alpha$) in deep stages:

| Stage | $\alpha_1$ | $\alpha_2$ | $\alpha_3$ | $\alpha_4$ | Top-1 |
|---|:---:|:---:|:---:|:---:|:---:|
| Uniform allocation | 0.5 | 0.5 | 0.5 | 0.5 | 79.0 |
| **Ramp allocation** | **0.25** | **0.5** | **0.5** | **0.75** | **79.2** |

### Loss & Training

Standard classification loss (cross-entropy) is used. Downstream tasks adopt the default losses of their respective frameworks.

## Key Experimental Results

### ImageNet-1K Classification

| Model | Type | Param | GMACs | Throughput (im/s) | Top-1 (%) |
|---|---|---|---|:---:|:---:|
| SwiftFormer-S | CNN+ViT | 6.1M | 1.0 | 2626 | 78.5 |
| EfficientVMamba-T | Mamba | 6.1M | 1.0 | 1396 | 76.5 |
| **TinyViM-S** | **CNN+Mamba** | **5.6M** | **0.9** | **2563** | **79.2** |
| MobileOne-S4 | CNN | 14.8M | 3.0 | 1223 | 79.4 |
| EfficientVMamba-S | Mamba | 11M | 1.3 | 674 | 78.7 |
| **TinyViM-B** | **CNN+Mamba** | **11M** | **1.5** | **1851** | **81.2** |
| VMamba-T | Mamba | 30M | 4.9 | 383 | 82.6 |
| EfficientVMamba-B | Mamba | 33M | 4.0 | 580 | 81.8 |
| **TinyViM-L** | **CNN+Mamba** | **31.7M** | **4.7** | **843** | **83.3** |

TinyViM-S achieves 79.2% Top-1 accuracy with only 5.6M parameters, surpassing EfficientVMamba-T by 2.7% with 1.8× higher throughput.

### COCO Detection and Instance Segmentation (Mask R-CNN)

| Backbone | Throughput | AP^box | AP^mask |
|---|:---:|:---:|:---:|
| EfficientVMamba-S | 104 | 39.3 | 36.7 |
| SwiftFormer-L1 | 174 | 41.2 | 38.1 |
| **TinyViM-B** | **180** | **42.3** | **38.7** |
| FastViT-SA24 | 93 | 42.0 | 38.0 |
| EfficientVMamba-B | 104 | 43.4 | 39.5 |
| **TinyViM-L** | **119** | **44.5** | **40.3** |

TinyViM-B outperforms SwiftFormer-L1 by +1.1 AP^box while achieving higher throughput.

### ADE20K Semantic Segmentation (Semantic FPN)

| Backbone | mIoU |
|---|:---:|
| EfficientFormer-L1 | 38.9 |
| TinyViM-S | 38.9 |
| SwiftFormer-L1 | 41.1 |
| **TinyViM-B** | **41.9** |
| PoolFormer-S36 | 42.0 |
| **TinyViM-L** | **44.1** |

### Ablation Study

**Laplacian kernel size**:

| Kernel | Top-1 | Throughput |
|---|:---:|:---:|
| 3 | 79.0 | 2510 |
| 5 (axial) | 79.0 | 2598 |
| **7 (axial)** | **79.2** | **2563** |
| 9 (axial) | 79.2 | 2479 |

The 7×7 axial convolution achieves the best accuracy-efficiency trade-off.

### Key Findings

1. Frequency decoupling is essential for lightweight Mamba — restricting hidden-state propagation to low-frequency components preserves accuracy while substantially improving throughput.
2. The progressive frequency allocation of Frequency Ramp Inception outperforms uniform allocation, consistent with the principle of "high frequencies in shallow layers, low frequencies in deep layers."
3. Re-parameterization provides no gain for small models (TinyViM-S ±0) but benefits larger models (TinyViM-L +0.2).
4. ERF visualizations show that TinyViM has substantially larger effective receptive fields than MobileOne and SwiftFormer.

## Highlights & Insights

- **Novel frequency-domain perspective**: This is the first work to analyze Mamba's behavioral preference from a frequency-domain viewpoint and to design a dedicated architecture accordingly.
- **Extreme efficiency**: TinyViM-S achieves 79.2% Top-1 with only 5.6M parameters and 0.9 GMACs, making it highly competitive in the ultra-lightweight regime.
- **Outstanding throughput advantage**: TinyViM delivers 2–3× the throughput of comparable Mamba models, making it genuinely suitable for real-time deployment.

## Limitations & Future Work

- The Laplacian pyramid decomposition increases implementation complexity, and non-standard operators may not be fully optimized on all hardware platforms.
- The assumption that Mamba consistently favors low frequencies requires further validation under different pre-training configurations.
- Throughput is evaluated only on V100 GPUs; actual latency on mobile or edge devices is not reported.

## Related Work & Insights

- Efficient visual backbones: MobileNet series, EfficientFormer, SwiftFormer, FastViT, etc.
- Vision Mamba: ViM, VMamba, EfficientVMamba, QuadMamba, etc.
- Frequency analysis: Prior work has analyzed the frequency preferences of CNNs and Transformers; this paper is the first to conduct such analysis for Mamba.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of frequency decoupling and Mamba is novel, with analysis-driven design.
- **Technical Depth**: ⭐⭐⭐⭐ — The logical chain from spectral analysis to quantitative validation to architecture design is complete and rigorous.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Full coverage of classification, detection, and segmentation with detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is well-argued and figures are clear.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] VSSD: Vision Mamba with Non-Causal State Space Duality](vssd_vision_mamba_with_non-causal_state_space_duality.md)
- [\[ICCV 2025\] Inter2Former: Dynamic Hybrid Attention for Efficient High-Precision Interactive Segmentation](inter2former_dynamic_hybrid_attention_for_efficient_high-precision_interactive_s.md)
- [\[ICCV 2025\] DeRIS: Decoupling Perception and Cognition for Enhanced Referring Image Segmentation through Loopback Synergy](deris_decoupling_perception_and_cognition_for_enhanced_referring_image_segmentat.md)
- [\[ICCV 2025\] Hybrid-TTA: Continual Test-time Adaptation via Dynamic Domain Shift Detection](hybrid-tta_continual_test-time_adaptation_via_dynamic_domain_shift_detection.md)
- [\[NeurIPS 2025\] RoMA: Scaling up Mamba-based Foundation Models for Remote Sensing](../../NeurIPS2025/segmentation/roma_scaling_up_mamba-based_foundation_models_for_remote_sensing.md)

</div>

<!-- RELATED:END -->

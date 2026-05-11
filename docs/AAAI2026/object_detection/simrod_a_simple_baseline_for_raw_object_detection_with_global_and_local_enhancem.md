---
title: >-
  [Paper Note] SimROD: A Simple Baseline for Raw Object Detection with Global and Local Enhancements
description: >-
  [AAAI 2026][Object Detection][RAW image] This paper proposes SimROD, an extremely lightweight (only 0.003M parameters) RAW image object detection method that surpasses complex state-of-the-art approaches on multiple RAW…
tags:
  - "AAAI 2026"
  - "Object Detection"
  - "RAW image"
  - "Gamma enhancement"
  - "green channel"
  - "lightweight"
date: 2026-05-08
content_hash: 4ed078cf7d5932d9
---

# SimROD: A Simple Baseline for Raw Object Detection with Global and Local Enhancements

**Conference**: AAAI 2026
**arXiv**: [2503.07101](https://arxiv.org/abs/2503.07101)
**Code**: [https://ocean146.github.io/SimROD2025/](https://ocean146.github.io/SimROD2025/)
**Area**: Object Detection
**Keywords**: RAW image, object detection, Gamma enhancement, green channel, lightweight

## TL;DR

This paper proposes SimROD, an extremely lightweight (only 0.003M parameters) RAW image object detection method that surpasses complex state-of-the-art approaches on multiple RAW detection benchmarks through global Gamma enhancement (4 learnable parameters) and green channel-guided local enhancement.

## Background & Motivation

### Why RAW Object Detection?

Conventional vision models are designed for sRGB images; however, sRGB images processed by an Image Signal Processor (ISP) lose substantial amounts of raw sensor information. RAW data directly preserves unprocessed sensor signals and offers the following advantages:

**Richer dynamic range**: RAW data retains more detail, particularly under extreme lighting conditions and adverse weather.

**Simplified system architecture**: Direct use of RAW data bypasses the ISP module, reducing system complexity, latency, and cost.

**Suitable for lightweight real-time applications**: This is especially critical for scenarios such as autonomous driving.

### Limitations of Prior Work

Existing RAW object detection methods rely on complex end-to-end ISP optimization frameworks that integrate learnable ISP stages with detection models, leading to the following issues:

- **High computational overhead**: RAW-Adapter introduces 0.46M additional parameters; DIAP introduces 0.26M.
- **High design complexity**: Unnecessary design complexity is introduced.
- **Neglect of the green channel advantage**: Most methods treat RGB channels equally and overlook the unique advantage of the green channel in RAW data.

### Core Observation: Superiority of the Green Channel

The authors reveal the importance of the green channel through two key analyses:

**Channel sensitivity analysis**: On the LOD dataset, per-channel detection performance is evaluated using DIAP. The green channel achieves approximately 10 AP higher than the red channel and approximately 20 AP higher than the blue channel.

**Signal-to-Noise Ratio (SNR) analysis**: The SNR of the green channel consistently exceeds that of the red and blue channels, indicating stronger noise robustness.

The biological basis for these findings is that the human eye is highly sensitive to green wavelengths under both bright and dark conditions. Consequently, the Bayer filter design prioritizes the green channel (green accounts for 50% in the RGGB pattern).

## Method

### Overall Architecture

The SimROD pipeline is straightforward:

1. Input RAW image $X_{RAW} \in \mathbb{R}^{2H \times 2W}$
2. Repacked into a four-channel image $X_{packed} \in \mathbb{R}^{H \times W \times 4}$ (RGGB pattern)
3. Global Gamma enhancement via the **GGE module** → $X_\gamma$
4. Green channel-guided local enhancement via the **GGLE module** → $\hat{X} \in \mathbb{R}^{H \times W \times 3}$
5. Fed into a downstream detection model

### Key Designs

#### 1. **Global Gamma Enhancement (GGE)**: Dynamic Range Adjustment with Only 4 Parameters

**Core problem**: Pixel values in RAW data tend to concentrate in the low range, making it difficult for deep networks to learn and extract features effectively.

**Design motivation**: A separate learnable Gamma parameter is assigned to each of the four RGGB channels, performing channel-wise Gamma transformation:

$$X_\gamma^i = \Gamma(X_{packed}^i; \gamma_i) = 255 \cdot (X_{packed}^i)^{\gamma_i}, \quad i \in \{R, G_1, G_2, B\}$$

**Parameterization**: A learnable parameter $\alpha_i \in \mathbb{R}$ is defined, constrained to $(-1, 1)$ via tanh, and then linearly scaled to the range $(\gamma_{min}, \gamma_{max})$, where $\gamma_{max} = 1/7.0$ and $\gamma_{min} = 1/10.5$.

**Design motivation**:
- Global dynamic range adjustment requires only 4 parameters, whereas DIAP requires 0.26M.
- The authors observe that the Gamma parameters predicted by DIAP's image-level adjustment module remain nearly unchanged even when the input is pure random noise, suggesting that complexity in image-level adjustment is unnecessary.
- Gamma values increase slightly during training, corresponding to a slight decrease in pixel values, which is physically reasonable.

#### 2. **Green Channel-Guided Local Enhancement (GGLE)**: Exploiting High-Frequency Details of the Green Channel

**Core idea**: The richer high-frequency detail information contained in the green channel of the Bayer pattern is leveraged to enhance local features.

**Dual-branch design**:
- **RGGB branch**: A convolutional network $\mathcal{F}_l$ processes the full RGGB data $X_\gamma$ to extract full-channel spatial features $\mathcal{F}_l(X_\gamma)$.
- **Guidance branch**: The two green channels $X_\gamma^{G_1}$ and $X_\gamma^{G_2}$ are concatenated and processed by another convolutional network $\mathcal{F}_l^G$ to extract green channel features $\mathcal{F}_l^G(X_\gamma^G)$.

**Multi-level fusion**:

$$\hat{X} = \text{Conv}(\text{Concat}[\mathcal{F}_l(X_\gamma) + \mathcal{F}_l^G(X_\gamma^G), \mathcal{F}_l(X_\gamma)])$$

Green-guided features and full-channel features are fused via addition, then concatenated with the original full-channel features and convolved to produce a three-channel enhanced representation.

**Architecture details**: Both branches adopt a simple Conv + BN + LeakyReLU structure.

#### 3. **Parameter Efficiency**: GGE + GGLE Totaling Only 0.003M Parameters

This is approximately 150× fewer parameters than RAW-Adapter (0.46M), yet achieves superior performance.

### Loss & Training

SimROD is an end-to-end framework trained directly with the standard loss function of the downstream detector, without any additional loss for the enhancement modules:

$$\mathcal{L}_{total} = \mathcal{L}_{cls} + \lambda \mathcal{L}_{reg}$$

Using YoloX as an example, $\lambda = 3$, encompassing classification and regression losses. The GGE and GGLE modules are jointly optimized with the detector.

Training details:
- YoloX uses the SGD optimizer, 300 epochs, with cosine learning rate scheduling.
- COCO pretrained weights are used for initialization (improving DIAP's performance on ROD from 24.0% mAP to 30.7% mAP).
- RetinaNet uses the MMDetection framework; the SimROD learning rate is set to 3e-3.

## Key Experimental Results

### Main Results

**YoloX-Tiny detector (Table 1)**:

| Method | LOD AP/AP50 | Pascal-Raw AP/AP50 | ROD AP/AP50 | Extra Params (M) |
|--------|------------|-------------------|------------|-----------------|
| DIAP | 25.9/43.4 | 68.7/94.2 | 30.7/53.4 | 0.260 |
| RAW-Adapter | 26.4/45.1 | 67.5/93.7 | N/A | 0.460 |
| **SimROD** | **26.7/46.3** | **69.7/95.1** | **33.1/57.6** | **0.003** |

**RetinaNet-R50 detector (Table 2)**:

| Method | LOD AP/AP50 | Pascal-Raw AP/AP50 | Extra Params (M) |
|--------|------------|-------------------|-----------------|
| RAW-Adapter | 62.1/62.1 | 89.7/89.7 | 0.46 |
| DIAP | 59.1/59.1 | 89.5/89.5 | 0.260 |
| **SimROD** | **63.4/63.4** | **90.1/90.1** | **0.003** |

SimROD achieves +2.4 AP over DIAP on ROD, +0.8 AP on LOD, and +1.0 AP on Pascal-Raw.

### Ablation Study

**Impact of each component (Table 4)**:

| Pretrain | GGE | GGLE (RGGB) | GGLE (Guidance) | LOD AP50 | Pascal-Raw AP50 |
|----------|-----|------------|----------------|---------|----------------|
| ✗ | ✗ | ✗ | ✗ | 27.7 | 85.0 |
| ✓ | ✗ | ✗ | ✗ | 44.6 | 93.0 |
| ✓ | ✓ | ✗ | ✗ | 45.0 | 94.3 |
| ✓ | ✓ | ✓ | ✗ | 45.1 | 94.7 |
| ✓ | ✓ | ✓ | GG | **46.3** | **95.1** |
| ✓ | ✓ | ✓ | RGGB | 46.2 | 94.5 |
| ✓ | ✓ | ✓ | R | 44.9 | 94.5 |
| ✓ | ✓ | ✓ | B | 44.2 | 94.2 |

**GGE vs. DIAP (Table 5)**: GGE with only 4 parameters achieves AP comparable to DIAP's 0.26M-parameter scheme, with zero GFLOPs overhead.

### Key Findings

1. **The green channel is indeed most informative**: Using only GG guidance outperforms all other configurations including RGGB, R, and B.
2. **Noise from the red and blue channels is detrimental**: The RB combination performs even worse than R alone, and RGGB as a whole underperforms GG.
3. **Green channel sampling frequency has a significant impact**: Reducing the green channel sampling frequency from 1× to 0.5× drops AP50 from 42.0 to 40.3 on LOD.
4. **Pretrained weights are critical**: COCO pretraining improves the DIAP baseline on ROD from 24.0% AP to 30.7% AP.

## Highlights & Insights

1. **Extreme parameter efficiency**: Only 0.003M parameters (approximately 3,000 parameters) surpass complex methods with hundreds of thousands of parameters, demonstrating that simplicity can be effective.
2. **Novel and well-supported green channel insight**: Starting from the biology of human color vision and Bayer filter design, the superiority of the green channel is rigorously validated through channel analysis and ablation experiments.
3. **End-to-end optimization without auxiliary losses**: Complex multi-task learning is avoided; the enhancement modules are driven solely by the detector loss.
4. **Cross-task generalization**: The method also demonstrates strong performance on semantic segmentation (ADE20K-Raw).

## Limitations & Future Work

1. **ROD dataset issue**: Due to the inability to obtain the complete ROD dataset, the authors evaluate on a subset only, which may affect the comparability of results.
2. **Limited scene diversity**: Experiments focus primarily on driving scenarios; generalization to other RAW detection settings (e.g., surveillance, industrial inspection) remains unverified.
3. **Adaptability to different camera sensors**: Noise characteristics and the degree of green channel advantage may vary across sensors.
4. **Fixed Gamma parameter range**: $\gamma_{min}$ and $\gamma_{max}$ are manually set hyperparameters that may require scene-specific adjustment.

## Related Work & Insights

- **DIAP** (Xu et al. 2023): An end-to-end RAW detection method that introduces an image-level adjustment module.
- **RAW-Adapter** (Cui and Harada 2024): An adapter-based RAW processing approach with a relatively large parameter count.
- **Bayer filter design**: The 50% green channel ratio in the RGGB pattern motivates the green channel-guided strategy in this work.

**Insight**: Simple domain priors (e.g., the green channel is more informative) combined with minimal-parameter modules can sometimes outperform complex methods. Understanding the intrinsic properties of the data is often more important than increasing model complexity.

## Rating

- Novelty: ⭐⭐⭐⭐ (Green channel guidance is an interesting insight, though Gamma transformation itself is not novel)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Multiple datasets, detectors, tasks, and thorough ablations)
- Writing Quality: ⭐⭐⭐⭐ (Clear and accessible; motivation and method are well described)
- Value: ⭐⭐⭐⭐ (Practically meaningful for the RAW detection field; extremely parameter-efficient)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DA-Mamba: Learning Domain-Aware State Space Model for Global-Local Alignment in Domain Adaptive Object Detection](../../CVPR2026/object_detection/da-mamba_learning_domain-aware_state_space_model_for_global-local_alignment_in_d.md)
- [\[CVPR 2026\] SpiralDiff: Spiral Diffusion with LoRA for RGB-to-RAW Conversion Across Cameras](../../CVPR2026/object_detection/spiraldiff_spiral_diffusion_with_lora_for_rgb-to-raw_conversion_across_cameras.md)
- [\[AAAI 2026\] MonoCLUE: Object-Aware Clustering Enhances Monocular 3D Object Detection](monoclue_object-aware_clustering_enhances_monocular_3d_object_detection.md)
- [\[AAAI 2026\] Temporal Object-Aware Vision Transformer for Few-Shot Video Object Detection](temporal_object-aware_vision_transformer_for_few-shot_video_object_detection.md)
- [\[AAAI 2026\] YOLO-IOD: Towards Real Time Incremental Object Detection](yolo-iod_towards_real_time_incremental_object_detection.md)

</div>

<!-- RELATED:END -->

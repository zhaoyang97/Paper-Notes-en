---
title: >-
  [Paper Note] RS-vHeat: Heat Conduction Guided Efficient Remote Sensing Foundation Model
description: >-
  [ICCV2025][Remote Sensing][Remote sensing foundation model] This work is the first to introduce the physical heat conduction process into a remote sensing foundation model. RS-vHeat replaces the attention mechanism with a Heat Conduction Operator (HCO) to model local region correlations in remote sensing images, achieving strong performance across 4 tasks and 10 datasets while reducing GPU memory by 84%, FLOPs by 24%, and improving throughput by 2.7× compared to the attention…
tags:
  - "ICCV2025"
  - "Remote Sensing"
  - "Remote sensing foundation model"
  - "heat conduction"
  - "self-supervised learning"
  - "frequency domain masking"
  - "multimodal"
date: 2026-05-08
content_hash: 856a6806722e30c4
---

# RS-vHeat: Heat Conduction Guided Efficient Remote Sensing Foundation Model

**Conference**: ICCV2025
**arXiv**: [2411.17984](https://arxiv.org/abs/2411.17984)  
**Code**: [iecashhy/RS-vHeat](https://github.com/iecashhy/RS-vHeat)  
**Area**: Remote Sensing
**Keywords**: Remote sensing foundation model, heat conduction, self-supervised learning, frequency domain masking, multimodal

## TL;DR

This work is the first to introduce the physical heat conduction process into a remote sensing foundation model. RS-vHeat replaces the attention mechanism with a Heat Conduction Operator (HCO) to model local region correlations in remote sensing images, achieving strong performance across 4 tasks and 10 datasets while reducing GPU memory by 84%, FLOPs by 24%, and improving throughput by 2.7× compared to the attention-based baseline.

## Background & Motivation

Remote Sensing Foundation Models (RSFMs) have emerged from the traditional task-specific modeling paradigm, offering cross-task scalability. However, existing methods face two major challenges:

**Tension between computational efficiency and receptive field**: Remote sensing images contain large-scale objects, requiring the model to respond over sufficiently large regions. CNNs are constrained by fixed convolutional kernels and lack a global receptive field; attention-based models (ViT/Swin) enable global modeling but incur quadratic complexity, resulting in substantial computational overhead. For high-resolution remote sensing images (e.g., 1024×1024), existing RSFMs struggle to achieve both fast inference and high accuracy simultaneously.

**Insufficient physical interpretability**: Remote sensing objects are typically irregular polygons. While existing models can extract features, they cannot explain how features propagate through physical principles. The lack of interpretability makes it difficult for researchers to effectively adjust learning strategies.

The authors draw inspiration from the physical process of **heat conduction**—the natural diffusion of heat from high-temperature to low-temperature regions—which can be analogized to the feature extraction process in neural networks. Regions containing complex remote sensing objects correspond to high-temperature zones (heat accumulation), while sparse regions correspond to low-temperature zones (heat dissipation). This physical analogy provides a natural interpretability framework for remote sensing image processing.

## Method

RS-vHeat consists of three core components: a frequency domain hierarchical masking strategy, a heat-conduction-based visual encoder, and a multi-domain reconstruction decoder.

### 1. Frequency Domain Hierarchical Masking

Traditional spatial domain masking directly occludes image patches, which can easily destroy small-target information in remote sensing images. RS-vHeat employs a frequency domain hierarchical masking strategy:

- For multimodal inputs (optical image $I_o \in \mathbb{R}^{H \times W \times 3}$, SAR image $I_s \in \mathbb{R}^{H \times W \times 1}$), the images are first transformed from the spatial domain to the frequency domain via Discrete Cosine Transform (DCT).
- A randomly generated sector mask $\tilde{M}$ separates the frequency domain image into a low-frequency component $\tilde{I}_{low}$ (carrying global structural information) and a high-frequency component $\tilde{I}_{high}$ (carrying detail information).
- The masking ratio is set to 20%–30% for flexible image processing.
- The high- and low-frequency signals are converted back to the spatial domain via Inverse DCT (IDCT) and concatenated to restore the original dimensions.

The key formulation is:

$$\tilde{I}_{low}(u,v), \tilde{I}_{high}(u,v) = \tilde{M} \odot \tilde{I}(u,v)$$

The advantage of frequency domain masking is that even after masking, the spatial structure of objects is not completely lost, thereby preserving small-target information.

### 2. Heat-Conduction-Based Visual Encoder

The encoder comprises 4 stages, each containing $L$ blocks (configured as 2, 2, 18, 2, consistent with Swin-B). Each block includes two key modules:

#### (a) HCO Block — Frequency Domain Heat Diffusion

The general solution to the heat conduction equation is:

$$u(x,y,t) = \mathcal{F}^{-1}\left(\tilde{f}(\omega_x, \omega_y) \cdot e^{-k(\omega_x^2 + \omega_y^2)t}\right)$$

Discretized for visual feature processing:

$$U_t^m = \text{IDCT}_{2D}\left(\text{DCT}_{2D}(U_0^m) \cdot e^{-k(\omega_x^2 + \omega_y^2)t}\right)$$

where $e^{-k(\omega_x^2 + \omega_y^2)t}$ acts as a frequency-domain adaptive filter performing the heat conduction computation. The thermal diffusivity $k$ is predicted via learnable **Frequency Value Embeddings (FVEs)**, $W_{FVEs} \in \mathbb{R}^{M \times N \times C}$, which are aligned with the image dimensions and can adaptively process multimodal images in different frequency domains based on scene-specific information.

The complexity of HCO is $O(N^{1.5})$, lower than the $O(N^2)$ of the attention mechanism, while maintaining a global receptive field.

#### (b) Correction Learner — Spatial Domain Information Adjustment

By predicting **Spatial Correction Embeddings (SCEs)** $W_{SCEs} \in \mathbb{R}^{M \times N \times C}$, which interact with and activate the current temperature field, SCEs adaptively adjust object boundaries to enhance or suppress edge features:

$$Z'^{p}_m = CL(Z^p_m, W_{SCEs})$$

SCEs help capture local details and broader contextual regions, assisting in simulating the rate of heat diffusion.

### 3. Multi-Domain Reconstruction Decoders

The pre-training stage employs a combination of three loss functions:

- **Frequency domain reconstruction loss** $\mathcal{L}_{Fre}$: computes the L1 difference between the reconstructed and original images in the frequency domain.
- **Spatial domain reconstruction loss** $\mathcal{L}_{Spa}$: fuses outputs from the third and fourth encoder stages, upsamples to the original resolution, and computes L1 loss in the spatial domain.
- **Contrastive loss** $\mathcal{L}_{Con}$: uses cosine similarity to constrain semantic consistency between high- and low-frequency features of different modalities in the thermal space.

Total loss: $\mathcal{L}_{total} = \mathcal{L}_{Con} + \mathcal{L}_{Spa} + \mathcal{L}_{Fre}$

### 4. Pre-training and Fine-tuning

- **Pre-training data**: 3 million multimodal remote sensing images (including 450K paired optical-SAR images) from six continents worldwide, with spatial resolutions ranging from 0.3m to 30m, uniformly cropped to 448×448.
- **Pre-training setup**: 8× A100 (80G) GPUs, 200 epochs, 224-size images, base learning rate 2e-4, cosine annealing.
- **Downstream fine-tuning**: The embedding layer and heat-conduction visual encoder structure and weights are directly transferred; FVEs and SCEs are adjusted via interpolation for downstream tasks with different image sizes.

## Key Experimental Results

### Semantic Segmentation (4 datasets)

| Dataset | Metric | RS-vHeat | Best Competing RSFM |
|---------|--------|----------|----------------------|
| Potsdam | mF1 | **92.82** | SkySense 93.99 (>702M params) |
| iSAID | mIoU | **68.72** | SkySense 70.91 (>702M params) |
| AIR-PolSAR-Seg | mIoU | **57.46** | DANet 51.93 |
| WHU-OPT-SAR (multimodal) | OA | **83.9** | MCANet 82.9 |

RS-vHeat achieves accuracy close to SkySense (>702M params, >2708G FLOPs) with only 148M parameters and 921G FLOPs.

### Object Detection (3 datasets)

| Dataset | Metric | RS-vHeat | Comparison |
|---------|--------|----------|------------|
| DIOR | mAP50 | **82.30** | SkySense 78.73, Scale-MAE 73.81 |
| FAIR1M-2.0 | mAP | 48.29 | SkySense 54.57 |
| SAR-AIRcraft-1.0 | mAP50 | **87.1** | SA-Net 77.7 |

Outperforms SkySense by 3.57% on DIOR, with only 266G FLOPs (SkySense >1679G).

### Image Classification (2 datasets)

| Dataset | Training Ratio | RS-vHeat | Scale-MAE (ViT-L) |
|---------|----------------|----------|---------------------|
| AID | 20% | **96.81** | 96.44 |
| AID | 50% | 97.58 | 97.58 |
| NWPU-RESISC45 | 10% | 92.01 | 92.63 |
| NWPU-RESISC45 | 20% | **95.66** | 95.04 |

RS-vHeat uses only 150M parameters and 340G FLOPs, compared to Scale-MAE's 310M/2070G — a reduction of over 6× in FLOPs.

### Change Detection

| Dataset | Metric | RS-vHeat | SkySense |
|---------|--------|----------|----------|
| LEVIR-CD | F1 | **93.48** | 92.58 |

### Computational Efficiency (1024×1024 image, single A100)

| Metric | RS-vHeat vs. Swin-B baseline |
|--------|-------------------------------|
| Throughput | **2.7×** |
| GPU memory | **84% reduction** |
| FLOPs | **24% reduction** |

### Ablation Study

Ablation on loss function combinations (DIOR mAP50):
- Spatial domain reconstruction only: 78.2
- + Frequency domain reconstruction: 79.5 (+1.3)
- + Contrastive loss: 82.3 (+2.8) → all three components are indispensable

## Highlights & Insights

1. **Physics-inspired architectural design**: The heat conduction equation is directly mapped to visual feature processing. HCO performs heat diffusion computation in the frequency domain, achieving both a global receptive field and sub-quadratic $O(N^{1.5})$ complexity — an elegant design that balances efficiency and modeling capacity.

2. **Frequency domain masking preserves small targets**: Unlike spatial domain masking, which may completely occlude small objects, frequency domain masking retains the spatial structure of targets, which is particularly important for small objects in remote sensing scenes (e.g., aircraft, ships).

3. **Unified thermal space for multimodal fusion**: Both optical and SAR images are projected into the same thermal space, with contrastive loss constraining deep semantic consistency between the two modalities, enabling natural multimodal fusion.

4. **Efficiency gains amplified at higher resolutions**: RS-vHeat's efficiency advantage becomes more pronounced when processing large-scale remote sensing images — precisely where the remote sensing domain needs it most.

5. **Interpretability**: The heat conduction process provides an intuitive explanation — regions containing complex objects accumulate heat and become high-temperature zones, while sparse regions dissipate heat and become low-temperature zones, which aligns closely with the semantic logic of remote sensing object detection.

## Limitations & Future Work

1. **Relatively weaker performance on FAIR1M**: On FAIR1M-2.0 fine-grained object detection (mAP 48.29 vs. SkySense 54.57), RS-vHeat does not surpass the strongest baseline, indicating room for improvement in complex multi-class, multi-scale, multi-orientation scenarios.

2. **Inconsistent superiority on classification tasks**: On NWPU-RESISC45 (10% labels) and AID (50% labels), RS-vHeat still lags behind Swin-based baselines (e.g., RingMo), suggesting that scene-level classification may not be the most suitable use case for the heat conduction paradigm.

3. **Large-scale pre-training data not fully public**: The 3 million images include non-public data such as imagery from the Gaofen-2 satellite, which may affect reproducibility.

4. **Interpolation bottleneck for fixed-size FVE/SCE**: Downstream tasks with varying image sizes require interpolation of FVEs and SCEs, which may introduce information loss, especially when resolution differences are large.

5. **Only a Base-scale model is designed**: The absence of experiments across different model scales (Tiny/Small/Large) makes it impossible to assess the scaling behavior of the model.

## Related Work & Insights

- **vHeat** (Wang et al., 2024): The foundational architecture of this work, which first introduced the heat conduction equation into vision models. RS-vHeat extends this by designing frequency domain masked pre-training and the SCE module tailored for remote sensing scenarios.
- **RingMo** (Sun et al., TGRS 2022): A representative RSFM using Swin-B with an incomplete masking strategy.
- **SkySense** (Guo et al., CVPR 2024): The current state-of-the-art RSFM (2.6B parameters); RS-vHeat approaches or surpasses its accuracy on multiple tasks with far fewer parameters.
- **Scale-MAE** (Reed et al., ICCV 2023): Reconstructs high- and low-frequency images based on spatial domain masking; RS-vHeat's frequency domain masking strategy is a further development of this idea.
- **VMamba** (Liu et al., NeurIPS 2024): An exploration of state space models in the visual domain, representing an alternative to attention mechanisms alongside RS-vHeat; however, the heat conduction paradigm offers a distinct advantage in physical interpretability.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SkySense V2: A Unified Foundation Model for Multi-Modal Remote Sensing](skysense_v2_a_unified_foundation_model_for_multi-modal_remote_sensing.md)
- [\[NeurIPS 2025\] GeoLink: Empowering Remote Sensing Foundation Model with OpenStreetMap Data](../../NeurIPS2025/remote_sensing/geolink_empowering_remote_sensing_foundation_model_with_openstreetmap_data.md)
- [\[ICCV 2025\] Towards a Unified Copernicus Foundation Model for Earth Vision](towards_a_unified_copernicus_foundation_model_for_earth_vision.md)
- [\[CVPR 2026\] CrossEarth-Gate: Fisher-Guided Adaptive Tuning Engine for Efficient Adaptation of Cross-Domain Remote Sensing Semantic Segmentation](../../CVPR2026/remote_sensing/crossearth-gate_fisher-guided_adaptive_tuning_engine_for_efficient_adaptation_of.md)
- [\[CVPR 2026\] ORSATR-X: A Foundation Model based on Differential-and-Excitation Networks for Optical Remote Sensing Object Recognition](../../CVPR2026/remote_sensing/orsatr-x_a_foundation_model_based_on_differential-and-excitation_networks_for_op.md)

</div>

<!-- RELATED:END -->

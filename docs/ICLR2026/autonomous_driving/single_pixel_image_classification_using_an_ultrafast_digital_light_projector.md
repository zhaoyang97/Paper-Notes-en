---
title: >-
  [Paper Note] Single Pixel Image Classification using an Ultrafast Digital Light Projector
description: >-
  [ICLR 2026][Autonomous Driving][Single-pixel imaging] This paper presents an experimental single-pixel imaging (SPI) system based on a microLED-on-CMOS ultrafast digital light projector…
tags:
  - "ICLR 2026"
  - "Autonomous Driving"
  - "Single-pixel imaging"
  - "image classification"
  - "microLED"
  - "Hadamard patterns"
  - "extreme learning machine"
date: 2026-05-08
content_hash: cadbc4ca7e523863
---

# Single Pixel Image Classification using an Ultrafast Digital Light Projector

**Conference**: ICLR 2026
**arXiv**: [2603.12036](https://arxiv.org/abs/2603.12036)
**Code**: None
**Area**: Autonomous Driving
**Keywords**: Single-pixel imaging, image classification, microLED, Hadamard patterns, extreme learning machine

## TL;DR
This paper presents an experimental single-pixel imaging (SPI) system based on a microLED-on-CMOS ultrafast digital light projector, combined with low-complexity machine learning models (ELM and DNN) to achieve sub-millisecond image encoding and kHz-rate image classification. The system attains >90% accuracy on the MNIST dataset and >99% AUC in binary classification scenarios.

## Background & Motivation
1. **Background**: Machine vision is a mature technology embedded in autonomous agents such as self-driving vehicles; however, the operational bandwidth of conventional digital cameras is becoming a bottleneck. Event cameras reduce data volume in dynamic scenes but are constrained to the visible and near-infrared spectrum.
2. **Limitations of Prior Work**:
    - Conventional SPI systems employ DMDs (digital micromirror devices) for pattern generation, which are limited by mechanical switching speeds (~$10^4$ fps), keeping overall imaging rates comparable to standard CMOS cameras ($\lesssim 10^2$ Hz).
    - Most existing single-pixel image classification (SPIC) work relies on simulation or low-speed experiments, lacking true ultrafast optical experimental validation.
    - The image reconstruction step introduces additional latency and computational overhead.
3. **Key Challenge**: SPI requires projecting long sequences of patterns to acquire sufficient information, and projection speed constitutes the bandwidth bottleneck; compressed sensing can reduce the number of patterns but at the cost of classification accuracy.
4. **Goal**: To experimentally validate a single-pixel image classification system based on ultrafast microLED projection that performs direct classification on photodetector time series without image reconstruction.
5. **Key Insight**: Leveraging the ~100× faster switching speed of microLED arrays compared to DMDs to bypass image reconstruction entirely and classify directly on spatiotemporally transformed data.
6. **Core Idea**: Reformulating image classification from the spatial domain to the spatiotemporal domain—each image is encoded as a light-intensity time series and classified directly by a low-complexity ML model.

## Method

### Overall Architecture
Hadamard pattern sequence → high-speed microLED projector → target image displayed on DMD → single-pixel detector captures superimposed light-intensity signal → real-time oscilloscope records time series → ML model performs direct classification (no reconstruction required).

### Key Designs
1. **Ultrafast Single-Pixel Imaging System**:
    - Core hardware: 128×128 microLED-on-CMOS array, 30×30 μm² pixels, 50 μm pitch, supporting MHz-rate frame refresh.
    - Projects a 12×12 Hadamard pattern set (Had12) at 330,000 fps in global shutter mode.
    - Image reconstruction formula: $I_{(x,y),M} = \frac{1}{M}\sum_{m=1}^{M} S_m P_{(x,y),m}$, where $S_m$ is the differential signal between each pair of complementary Hadamard patterns.
    - Binarized MNIST images are displayed on a DMD (1024×768 resolution).
    - An Onsemi SiPM single-pixel detector captures light intensity, recorded by a 1 GHz bandwidth oscilloscope.
    - **Design Motivation**: The MHz-level switching speed of microLEDs overcomes the mechanical limitations of DMDs, enabling true kHz-rate SPI.

2. **Extreme Learning Machine (ELM) Classifier**:
    - Single hidden-layer neural network with randomly initialized and fixed input weights.
    - Hidden layer output: $H = f(XW_{\text{in}} + b)$, using ReLU activation.
    - Output weights solved in closed form via Ridge regression: $\beta = (H^\top H + \alpha I)^{-1} H^\top T$.
    - Multi-class prediction uses $\hat{y} = \max(Y)$; binary classification uses a threshold of 0.5.
    - Regularization parameter $\alpha = 1.0$.
    - **Design Motivation**: ELM training is extremely fast (no iterative optimization); inference requires only 31 μs/image, making it well-suited for ultrafast scenarios.

3. **Deep Neural Network (DNN) Classifier**:
    - Feedforward DNN: input layer (286-dimensional) → three hidden layers with decreasing width + ReLU → softmax output.
    - Adam optimizer, sparse categorical cross-entropy loss, trained for 300 epochs.
    - Inference time: 73 μs/image.
    - **Design Motivation**: Serves as a higher-complexity baseline for comparison with ELM, exploring the accuracy–speed trade-off.

4. **Hadamard Pattern Subset Optimization**:
    - Low-index (low spatial frequency) Hadamard patterns are found to carry more classification-relevant information.
    - Using only the first 1/4 of patterns preserves ≃78% classification accuracy.
    - Cat1 (first 44 patterns) varies along a single spatial axis and captures coarse features; Cat2 (patterns 45–288) varies along two spatial directions and captures finer features.
    - **Design Motivation**: Reducing the number of projected patterns proportionally increases the effective imaging bandwidth.

### Loss & Training
- ELM: closed-form Ridge regression solution, $\alpha = 1.0$, no iterative training.
- DNN: Adam optimizer, sparse categorical cross-entropy loss, 300 epochs.
- Data: MNIST dataset (60K training, 10K testing), binarized and rescaled to fill the full DMD surface.

## Key Experimental Results

### Main Results

| Method / Configuration | Accuracy (%) | Inference Speed | Notes |
|------------------------|-------------|-----------------|-------|
| DNN + full Had12 (experimental) | >90 | 73 μs/image | 1.2 kfps frame rate |
| ELM + full Had12 (experimental) | 87.37 | 31 μs/image | 2× faster than DNN |
| DNN + binarized MNIST (simulation) | 97.50 | — | Theoretical upper bound |
| ELM + binarized MNIST (simulation) | 93.32 | — | ELM upper bound |
| ELM binary classification (one-vs-all) | AUC >99% | — | Anomaly detection |

### Ablation Study

| Configuration | Accuracy (%) | Notes |
|---------------|-------------|-------|
| Full Had12 (DNN) | >90 | All 144 patterns |
| First 1/2 Had12 | ~86 | Minor accuracy drop |
| First 1/4 Had12 | ~78 | Acceptable accuracy, ×4 bandwidth |
| First 1/8 Had12 | ~68 | Significant accuracy drop |
| Last 1/2 Had12 | ~75 | High-frequency patterns less informative |
| Random 1/2 Had12 | ~82 | Between first/last halves |
| Gaussian noise σ=0.1 | >95 | Minimal noise impact |
| Gaussian noise σ=0.5 | >95 | Convergence maintained |
| Gaussian noise σ=1.0 | ~85 | Notable drop and variance |

### Key Findings
- The primary cause of accuracy degradation is not reduced effective SNR but spatial information loss due to compressed sensing.
- Low spatial frequency Hadamard patterns contribute most to classification; high-frequency patterns are less informative.
- Although ELM achieves lower accuracy than DNN, its inference speed is 2× faster, making it suitable for extreme real-time scenarios.
- When the number of patterns is reduced, the DNN exhibits longer vanishing gradient phases, consistent with the nature of compressed inputs.
- In binary classification, AUC approaches 1.0, indicating suitability for anomaly detection in rapidly changing scenes.

## Highlights & Insights
- This work provides the first experimental validation of single-pixel image classification at kHz frame rates, surpassing the speed limitations of conventional imaging systems.
- Bypassing image reconstruction entirely greatly simplifies the system pipeline and reduces latency.
- The minimalist design of the ELM model aligns well with the demands of ultrafast scenarios, offering fast training, fast inference, and low overhead.
- The frequency-domain analysis of Hadamard pattern subsets offers practical guidance for compression strategy design.
- The comparative experiments on noise versus compressed sensing provide valuable theoretical insights.

## Limitations & Future Work
- Validation is limited to the MNIST dataset, which is relatively simple and far from real-world machine vision scenarios.
- The 12×12 Hadamard pattern resolution is low, limiting the system's ability to resolve complex images.
- The storage depth of the current FPGA board constrains the size of the pattern set.
- The SiPM detector and oscilloscope on the sensing side are difficult to miniaturize and integrate.
- More complex ML models (e.g., CNNs) and larger-scale datasets have not been explored.
- Substantial work remains to transfer the approach from MNIST to practical autonomous driving scenarios.

## Related Work & Insights
- Compressed sensing theory provides the mathematical foundation for reducing the number of projected patterns.
- The application of microLED arrays in analog optical computing underscores their central role in next-generation optical computing architectures.
- Reconstruction-free SPIC methods have advanced rapidly in recent years; this paper represents the fastest experimentally validated instance among them.
- The combination of low-complexity models such as ELM and reservoir computing with optical hardware is a promising research direction.
- Single-pixel imaging technology holds unique advantages in non-visible spectral bands (terahertz, ultraviolet).

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SiMO: Single-Modality-Operable Multimodal Collaborative Perception](simo_single-modality-operable_multimodal_collaborative_perceptio.md)
- [\[ICLR 2026\] SPACeR: Self-Play Anchoring with Centralized Reference Models](spacer_self-play_anchoring_with_centralized_reference_models.md)
- [\[ICLR 2026\] EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video](egodex_learning_dexterous_manipulation_from_large-scale_egocentric_video.md)
- [\[ICLR 2026\] x²-Fusion: Cross-Modality and Cross-Dimension Flow Estimation in Event Edge Space](x2-fusion_cross-modality_and_cross-dimension_flow_estimation_in_event_edge_space.md)
- [\[ICLR 2026\] NeMo-map: Neural Implicit Flow Fields for Spatio-Temporal Motion Mapping](nemo-map_neural_implicit_flow_fields_for_spatio-temporal_motion_mapping.md)

</div>

<!-- RELATED:END -->

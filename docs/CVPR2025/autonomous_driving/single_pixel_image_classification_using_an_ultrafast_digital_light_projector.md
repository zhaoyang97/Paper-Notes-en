---
title: >-
  [Paper Note] Single Pixel Image Classification using an Ultrafast Digital Light Projector
description: >-
  [CVPR2025][Autonomous Driving][single pixel imaging] Achievement of single-pixel imaging (SPI)-based MNIST image classification utilizing an ultrafast microLED-on-CMOS digital light projector, reaching $>90\%$ classification accuracy at a frame rate of 1.2 kfps. This completely bypasses image reconstruction to classify directly from temporal optical signals.
tags:
  - "CVPR2025"
  - "Autonomous Driving"
  - "single pixel imaging"
  - "image classification"
  - "microLED"
  - "Hadamard patterns"
  - "extreme learning machine"
  - "compressed sensing"
date: 2026-05-08
content_hash: 2cfe62b9dbcdaf36
---

# Single Pixel Image Classification using an Ultrafast Digital Light Projector

**Conference**: CVPR2025  
**arXiv**: [2603.12036](https://arxiv.org/abs/2603.12036)  
**Code**: Data publicly available (see paper for link)  
**Area**: Autonomous Driving  
**Keywords**: single pixel imaging, image classification, microLED, Hadamard patterns, extreme learning machine, compressed sensing

## TL;DR
Achievement of single-pixel imaging (SPI)-based MNIST image classification utilizing an ultrafast microLED-on-CMOS digital light projector, reaching $>90\%$ classification accuracy at a frame rate of 1.2 kfps. This completely bypasses image reconstruction to classify directly from temporal optical signals.

## Background & Motivation

### Key Challenge

**Key Challenge**: **Background**: 1. High-speed image classification is increasingly demanded in machine vision, yet the operating bandwidth of conventional digital cameras remains a bottleneck.
2. Single-pixel imaging (SPI) requires only a single-point detector and structured illumination sequences to image, offering unique advantages in high-speed and non-conventional wavebands (outside the range of silicon-based detectors).
3. DMDs (Digital Micromirror Devices) are limited by mechanical switching, with pattern switching rates of only around $10^4$ fps; microLED arrays can achieve pattern generation approximately 100 times faster than DMDs.
4. Existing SPIC (Single-Pixel Image Classification) works are mostly based on simulation or low-speed experiments, lacking validation through real, ultra-high-speed, free-space optical experiments.
5. Classifying directly from spatio-temporally encoded optical signals (bypassing image reconstruction) significantly reduces hardware complexity at the detection end.
6. One-vs-all binary classification scenarios resemble anomaly detection and hold potential for industrial applications.

## Method

### Overall Architecture
The SPI classification framework comprises three stages: (1) a microLED projector projects a sequence of Hadamard patterns onto the target object; (2) a single-pixel photodetector collects the time-series of integrated light intensities; (3) a low-complexity ML model classifies directly from the temporal sequence without reconstructing the image.

### Key Designs

**1. Optical System**
- microLED-on-CMOS projector: 128×128 pixel array, 30×30 μm² pixels, 50 μm pitch, with MHz-level global shutter frame switching.
- 12×12 Hadamard pattern set (Had12), totaling 144 patterns (288 binary frames, including positive/negative complementary pairs).
- Projection frame rate of 330,000 fps $\to$ single image encoding $<1$ ms $\to$ effective classification frame rate of 1.2 kfps.
- A DMD displays binarized MNIST digits, which are then acquired by a SiPM single-pixel detector.

**2. Extreme Learning Machine (ELM)**
- A single-hidden-layer neural network where the input weights are randomly initialized and kept fixed.
- Output weights are solved analytically in a single step via ridge regression: $$\beta = (H^\top H + \alpha I)^{-1} H^\top T$$
- The hidden layer employs ReLU activation, with a regularization parameter $\alpha = 1.0$.
- Inference time of 31 μs/digit (twice as fast as DNN).

**3. Deep Neural Network (DNN)**
- A feedforward fully connected network: Input layer (286) $\to$ three decreasing hidden layers $\to$ softmax output.
- Adam optimizer + sparse categorical cross-entropy loss.
- Implemented with TensorFlow/Keras, trained for 300 epochs.
- Inference time of 73 μs/digit.

**4. Hadamard Pattern Subset Analysis**
- The Had12 set is divided by spatial frequency into Cat1 (low frequency, varying along a single axis, first 44 patterns) and Cat2 (high frequency, varying along both axes).
- Using only the first 1/4 low-frequency patterns still maintains $\sim 78\%$ classification accuracy, effectively improving bandwidth.
- Lower-indexed patterns (low spatial frequency) contain more useful information for classification.

### Loss & Training
- ELM: ridge regression (L2-regularized least squares)
- DNN: sparse categorical cross-entropy + Adam optimization

## Key Experimental Results

### Main Results

| Model | Accuracy | Frame Rate | Inference Time per Digit |
|------|------|------|----------------|
| ELM (1000 neurons) | 87.37% | 1.2 kfps | 31 μs |
| DNN (full Had12) | >90% | 1.2 kfps | 73 μs |
| Numerical Simulation DNN (binarized) | 97.50% | - | - |
| Numerical Simulation ELM (binarized) | 93.32% | - | - |

### Hadamard Subset Compression Analysis

| Had12 Ratio | Strategy | DNN Accuracy | Equivalent Bandwidth |
|-----------|------|---------|---------|
| 1 (All) | - | >90% | 1.2 kHz |
| 1/2 first | First Half | ~87% | 2.4 kHz |
| 1/4 first | First 1/4 | ~78% | 4.8 kHz |
| 1/2 last | Last Half | ~75% | 2.4 kHz |

### Binary Classification (One-vs-All)
- ELM binary classification accuracy is $>99\%$, with AUC for all categories close to 1.0.
- Serves as a practical baseline for scenarios similar to anomaly detection.

### Noise Robustness
- Accuracy of over 95% is still maintained under Gaussian noise of $\sigma = 0.1$ and $0.5$.
- Accuracy degrades significantly and fluctuates widely when $\sigma = 1.0$.
- The primary cause of performance degradation is the loss of spatial information rather than a reduction in the equivalent SNR.

## Highlights & Insights
1. **Unprecedented Speed**: Execution of SPI-based classification at 1.2 kfps in a free-space optical experiment for the first time, which is two orders of magnitude faster than DMD-based systems.
2. **Reconstruction Decoupling**: Complete bypass of image reconstruction by classifying directly from optoelectronic time series, dramatically reducing computational and hardware costs.
3. **Frequency Selection Strategy**: Revelation of the ordered hierarchical structure of Hadamard patterns, showing that low-frequency patterns contribute most to classification, which can guide compressed sensing strategies.
4. **ELM Simplicity & Efficiency**: The single-hidden-layer ELM utilizing ridge regression achieves an inference time of only 31 μs, making it suitable for resource-constrained real-time systems.
5. **Noise vs. Information Loss**: Experimental results demonstrate that the accuracy loss under compressed sensing is primarily driven by the loss of structured information rather than noise.

## Limitations & Future Work
1. Validation is limited to binarized MNIST digits, which is far from the complexity of natural image classification.
2. The 12×12 Hadamard resolution is extremely low, constrained by the memory depth of the FPGA board.
3. The use of a DMD as the target object display (rather than a real-world scene) leaves a gap compared to actual machine vision deployments.
4. Applicability in practical scenarios such as multi-object or moving-target conditions has not been discussed.
5. The classification task is simplistic (10 classes); ELM performance may falter on more complex tasks.

## Related Work & Insights
- Complementarity with event cameras: SPI can operate in non-visible wavebands where event cameras are not supported.
- The role of microLED array technology is increasingly crucial in optical and analog computing.
- The frequency selection philosophy of the compressed-sensing strategy can be extended to other structured illumination systems.
- The minimalist inference of ELM provides a concept for edge-side anomaly detection.

## Rating
- Novelty: ⭐⭐⭐⭐ (First ultra-high-speed experimental SPI classification)
- Experimental Thoroughness: ⭐⭐⭐ (Limited to MNIST, lacks complex scenes)
- Writing Quality: ⭐⭐⭐⭐ (Detailed experimental descriptions and in-depth analysis)
- Value: ⭐⭐⭐ (Novel direction but limited application scenarios)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Neural Inverse Rendering from Propagating Light](neural_inverse_rendering_from_propagating_light.md)
- [\[CVPR 2025\] LightLoc: Learning Outdoor LiDAR Localization at Light Speed](lightloc_learning_outdoor_lidar_localization_at_light_speed.md)
- [\[NeurIPS 2025\] Semantic Glitch: Agency and Artistry in an Autonomous Pixel Cloud](../../NeurIPS2025/autonomous_driving/semantic_glitch_agency_and_artistry_in_an_autonomous_pixel_cloud.md)
- [\[ICCV 2025\] GM-MoE: Low-Light Enhancement with Gated-Mechanism Mixture-of-Experts](../../ICCV2025/autonomous_driving/gm-moe_low-light_enhancement_with_gated-mechanism_mixture-of-experts.md)
- [\[CVPR 2025\] Towards Satellite Image Road Graph Extraction: A Global-Scale Dataset and A Novel Method](towards_satellite_image_road_graph_extraction_a_global-scale_dataset_and_a_novel.md)

</div>

<!-- RELATED:END -->

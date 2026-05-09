---
title: >-
  [Paper Note] Single Pixel Image Classification using an Ultrafast Digital Light Projector
description: >-
  [CVPR 2026][Autonomous Driving][Single-pixel imaging] This paper employs a microLED-on-CMOS digital light projector to realize ultrafast single-pixel imaging (SPI), and combines low-complexity machine learning models (ELM and DNN) to achieve >90% classification accuracy on MNIST handwritten digits at a frame rate of 1.2 kHz, entirely bypassing image reconstruction.
tags:
  - CVPR 2026
  - Autonomous Driving
  - Single-pixel imaging
  - image classification
  - microLED
  - extreme learning machine
  - compressive sensing
date: 2026-05-08
content_hash: 364dcc5773ee675d
---

# Single Pixel Image Classification using an Ultrafast Digital Light Projector

**Conference**: CVPR 2026
**arXiv**: [2603.12036](https://arxiv.org/abs/2603.12036)
**Code**: [Dataset available](https://doi.org/10.15129/e6ba11c5-c931-45c1-bd55-1522fce3e7cc)
**Area**: Autonomous Driving
**Keywords**: Single-pixel imaging, image classification, microLED, extreme learning machine, compressive sensing

## TL;DR

This paper employs a microLED-on-CMOS digital light projector to realize ultrafast single-pixel imaging (SPI), and combines low-complexity machine learning models (ELM and DNN) to achieve >90% classification accuracy on MNIST handwritten digits at a frame rate of 1.2 kHz, entirely bypassing image reconstruction.

## Background & Motivation

**Background**: Machine vision is a core technology in autonomous driving and related domains; however, the bandwidth of conventional digital cameras becomes a bottleneck in high-speed scenarios. Single-pixel imaging (SPI) can substantially reduce hardware complexity by using a single-point detector with structured illumination patterns, yet it is constrained by the mechanical refresh rate of DMDs (~10⁴ fps).

**Limitations of Prior Work**: The limited switching speed of DMDs restricts SPI image generation rates to ~10² Hz, comparable to ordinary CMOS cameras. Although compressive sensing (CS) can reduce the number of required patterns, it sacrifices image quality. Moreover, most existing SPI classification work remains at the simulation stage.

**Key Challenge**: High-speed real-time image classification demands rapid encoding, yet conventional spatial light modulators (DMDs) offer insufficient switching speeds. Simultaneously, adequate classification accuracy must be maintained under extremely limited sampling.

**Goal**: To realize kHz-level single-pixel image classification in a genuine optical experiment, bypassing image reconstruction and performing classification directly from temporal measurement sequences.

**Key Insight**: A microLED-on-CMOS digital light projector is adopted in place of a DMD, increasing pattern switching speed by approximately 100×, and combined with minimal ML models to enable real-time high-speed classification.

**Core Idea**: A microLED array projects Hadamard patterns onto target objects at 330,000 fps; a single-pixel detector captures the resulting time series of light intensities, which are then classified directly in the temporal domain without image reconstruction.

## Method

### Overall Architecture

The system comprises three stages: (1) the microLED projector projects sequences of 12×12 Hadamard patterns onto the target object at an ultra-high frame rate; (2) a single-pixel photodetector (SiPM) acquires the light intensity signal resulting from the superposition of each pattern with the target, forming a temporal measurement sequence; (3) a low-complexity ML model classifies the temporal sequence directly, entirely bypassing the image reconstruction step.

### Key Designs

1. **microLED-on-CMOS Projector**: A 128×128 pixel array with pixel pitch 30×30 μm², supporting MHz-level global-shutter pattern switching. In experiments, 12×12 Hadamard patterns (Had12; 144 positive–negative pattern pairs, 288 pattern frames in total) are projected at 330,000 fps, with a per-image encoding time of <1 ms.
2. **Hadamard Structured Illumination**: Hadamard orthogonal bases are used as illumination patterns. Since LEDs cannot represent negative values, each Hadamard pattern is split into positive and negative frames, and their difference is taken as the measurement value. Patterns are ordered by spatial frequency: low-frequency patterns (Cat1, the first 44) capture coarse structure, while high-frequency patterns (Cat2, the remaining 244) capture fine detail.
3. **ELM (Extreme Learning Machine)**: A single-hidden-layer neural network in which input weights are randomly fixed and not trained; output weights are solved in one step via ridge regression: $\beta = (H^\top H + \alpha I)^{-1} H^\top T$. Inference time is 31 μs/digit. The model supports both multi-class and one-vs-all binary classification (for anomaly detection).
4. **DNN**: A three-hidden-layer feedforward network with ReLU activations, Adam optimizer, and softmax output; input dimensionality is 286. Inference time is 73 μs/digit, achieving higher accuracy at the cost of speed.

### Loss & Training

- **ELM**: Closed-form ridge regression solution with regularization parameter $\alpha = 1.0$; no iterative optimization required.
- **DNN**: Sparse categorical cross-entropy loss, trained with the Adam optimizer for 300 epochs.
- **Data preprocessing**: MNIST images are binarized and then mapped to the full DMD surface.

## Key Experimental Results

### Main Results: Classification Accuracy vs. Bandwidth

| Method | Pattern Set | Effective Bandwidth | Classification Accuracy | Inference Time/Image |
|:---|:---|:---|:---|:---|
| ELM (1000 neurons) | Had12 full set | 1.2 kHz | 87.37% | 31 μs |
| DNN | Had12 full set | 1.2 kHz | >90% | 73 μs |
| DNN | Had12 first 1/2 | 2.4 kHz | ~86% | 73 μs |
| DNN | Had12 first 1/4 | 4.8 kHz | ~78% | 73 μs |
| Numerical simulation DNN | Binarized MNIST | — | 97.50% | — |
| Numerical simulation ELM | Binarized MNIST | — | 93.32% | — |

### Ablation Study: Effect of Pattern Subset Selection Strategy on Accuracy

| Subset Strategy | 1/2 Patterns | 1/4 Patterns | 1/8 Patterns | 1/16 Patterns |
|:---|:---|:---|:---|:---|
| First $n$ (low-frequency priority) | ~86% | ~78% | ~67% | ~55% |
| Last $n$ (high-frequency priority) | ~78% | ~65% | ~52% | ~42% |
| Random selection | ~82% | ~73% | ~61% | ~50% |

### Key Findings

- Low-frequency Hadamard patterns (low spatial frequency, fewer sign inversions) carry more information relevant to classification; prioritizing low-frequency patterns preserves higher accuracy while reducing the number of patterns.
- Under binary one-vs-all classification, the ELM achieves AUC values approaching 1 across all classes, demonstrating its suitability for anomaly detection.
- Gaussian noise causes a uniform degradation in accuracy, whereas compressive sensing (reducing the number of patterns) induces gradient vanishing and stagnation at local minima during training, indicating that the root cause of performance degradation is loss of spatial information rather than reduced signal-to-noise ratio.

## Highlights & Insights

- **Direct classification without reconstruction**: SPI is transformed from an imaging tool into a direct classification tool; the computational overhead of image reconstruction is avoided through a spatiotemporal transformation.
- **Hardware speed breakthrough**: The microLED projector achieves pattern switching speeds approximately 100× faster than DMDs, representing the fastest SPI experimental system reported to date.
- **ELM closed-form solution — minimal and efficient**: ELM training requires no iterative optimization, and inference is 2× faster than DNN, making it well-suited for real-time scenarios.
- **In-depth analysis of noise vs. information loss**: The paper distinguishes between the different mechanistic impacts of additive Gaussian noise and compressive-sensing-induced information loss on the learning process.

## Limitations & Future Work

- Validation is conducted solely on the MNIST dataset, with no experiments on natural images or autonomous driving scenarios.
- The Hadamard basis size is limited by FPGA board memory, restricting the implementation to 12×12 (144 pattern groups in total), which yields very low spatial resolution.
- A significant accuracy gap exists between binarized and original MNIST (93.32% vs. >99% achievable on the original), indicating non-trivial information loss due to binarization.
- Adaptive pattern selection or learning-driven pattern optimization is not explored.

## Related Work & Insights

- Unlike simulation-based studies such as Jiao (2018) and Cao (2021), this work is the first to realize kHz-level single-pixel image classification (SPIC) in a genuine free-space optical system.
- microLED technology is expanding from optical communications into optical computing (Kalinin 2025, Müller 2025); this paper demonstrates its potential for machine vision applications.
- The closed-form training strategy of ELM can inspire low-power AI deployment at the edge.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Significant contribution at the hardware system level (microLED + SPI classification), though the algorithmic components (ELM/DNN) are relatively conventional.
- **Experimental Thoroughness**: ⭐⭐⭐ Ablation studies and analyses are comprehensive, but evaluation is limited to MNIST; natural-scene validation is absent.
- **Writing Quality**: ⭐⭐⭐⭐ Structure is clear, the optical system is described in detail, and experimental analysis follows rigorous logic.
- **Value**: ⭐⭐⭐ Demonstrates the feasibility of ultra-high-speed single-pixel classification, offering a new paradigm for machine vision in extreme-wavelength or high-speed scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SearchAD: Large-Scale Rare Image Retrieval Dataset for Autonomous Driving](searchad_large-scale_rare_image_retrieval_dataset_for_autonomous_driving.md)
- [\[CVPR 2026\] OneOcc: Semantic Occupancy Prediction for Legged Robots with a Single Panoramic Camera](oneocc_semantic_occupancy_prediction_for_legged_robots_with_a_single_panoramic_c.md)
- [\[CVPR 2026\] A Prediction-as-Perception Framework for 3D Object Detection](a_predictionasperception_framework_for_3d_object_d.md)
- [\[CVPR 2026\] Learning Mutual View Information Graph for Adaptive Adversarial Collaborative Perception](learning_mutual_view_information_graph_for_adaptive_adversarial_collaborative_pe.md)
- [\[CVPR 2026\] Learning Vision-Language-Action World Models for Autonomous Driving](vla_world_learning_vision_language_action_world_models_for_autonomous_driving.md)

</div>

<!-- RELATED:END -->

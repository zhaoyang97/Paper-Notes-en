---
title: >-
  [Paper Note] I2E: Real-Time Image-to-Event Conversion for High-Performance Spiking Neural Networks
description: >-
  [AAAI 2026][Spiking Neural Networks] I2E proposes an ultra-efficient image-to-event stream conversion framework that simulates microsaccadic eye movements and implements the conversion via highly parallelized convolution…
tags:
  - "AAAI 2026"
  - "Spiking Neural Networks"
  - "Event Stream Generation"
  - "Image-to-Event Conversion"
  - "Data Augmentation"
  - "Sim-to-Real"
date: 2026-05-08
content_hash: 13eaee403367f1ff
---

# I2E: Real-Time Image-to-Event Conversion for High-Performance Spiking Neural Networks

**Conference**: AAAI 2026
**arXiv**: [2511.08065](https://arxiv.org/abs/2511.08065)  
**Code**: [GitHub](https://github.com/Ruichen0424/I2E)  
**Area**: Neuromorphic Computing / Spiking Neural Networks
**Keywords**: Spiking Neural Networks, Event Stream Generation, Image-to-Event Conversion, Data Augmentation, Sim-to-Real

## TL;DR

I2E proposes an ultra-efficient image-to-event stream conversion framework that simulates microsaccadic eye movements and implements the conversion via highly parallelized convolutions, achieving over 300× speedup compared to prior methods. It enables online data augmentation during SNN training for the first time, achieves a state-of-the-art 60.50% event-based classification accuracy on I2E-ImageNet, and sets a new record of 92.5% on CIFAR10-DVS through a sim-to-real paradigm of synthetic pretraining followed by real-data fine-tuning.

## Background & Motivation

Spiking Neural Networks (SNNs) are a brain-inspired computing paradigm driven by sparse, asynchronous events, offering orders-of-magnitude energy efficiency advantages on dedicated neuromorphic chips such as Loihi and TrueNorth. The natural input modality for SNNs is asynchronous event streams, typically captured by Dynamic Vision Sensors (DVS), which report per-pixel brightness changes rather than full frames. However, reliance on specialized hardware creates a fundamental data bottleneck: acquiring large-scale event datasets is both costly and time-consuming, and existing benchmarks are limited in scale and inconsistent in quality (e.g., monitor-flicker artifacts).

This gives rise to a persistent performance gap: the best event-based ImageNet classification accuracy falls far below the 70%+ achieved by ANN counterparts, casting doubt on the practical utility of SNNs for complex tasks. A common workaround is to repeat the same static image at every timestep, but this introduces dense redundant computation and fundamentally undermines the energy efficiency of the event-driven paradigm. Prior algorithmic conversion methods (e.g., the ODG algorithm used in ES-ImageNet) circumvent hardware acquisition constraints but suffer from severe computational bottlenecks—processing the full ImageNet requires over 10 hours—making them incompatible with online data augmentation.

The core insight of I2E is that simulating microsaccadic eye movements allows image differencing to be equivalently expressed as extremely sparse $3 \times 3$ convolutions, boosting conversion speed to a level suitable for online execution during training. This simultaneously addresses the data scarcity problem at scale and the augmentation problem in training methodology.

## Method

### Overall Architecture

I2E converts static RGB images into 8-timestep binary event streams in three stages: intensity map generation → spatiotemporal convolutional event generation → adaptive event firing. The entire pipeline is designed as a sequence of highly parallelized tensor operations, naturally suited for GPU acceleration.

### Key Designs

1. **Intensity Map Generation (Stage 1)**:

    - Function: Converts an RGB image into a single-channel intensity map.
    - Mechanism: Extracts the V (Value) channel from the HSV color space, i.e., $V(x,y) = \max(I_R(x,y), I_G(x,y), I_B(x,y))$, producing a photoreceptor-like intensity representation at minimal computational cost.
    - Design Motivation: DVS responds to logarithmic luminance changes, and the V channel is the simplest effective approximation. Ablation experiments confirm that the V channel introduces negligible information loss compared to standard grayscale.

2. **Spatiotemporal Convolutional Event Generation (Stage 2)**:

    - Function: Simulates luminance changes induced by microsaccadic motion from a static intensity map.
    - Mechanism: The image is shifted by 1 pixel in each of 8 directions and differenced. The key innovation is that each directional shift-and-difference is equivalent to an extremely sparse $3 \times 3$ convolution kernel $K_t$—with only two nonzero entries (+1 and −1). All 8 directional differences are computed in parallel via a single grouped convolution: $\Delta V_t = V * K_t$.
    - Random Augmentation Strategy: Each direction has a set of equivalent shift vectors; one is randomly selected during training to introduce diversity, while a fixed selection is used at inference.
    - Design Motivation: A naïve implementation requires 8 sequential image shifts and differences, which is memory-intensive and serial. The convolution equivalence makes the operation highly efficient on GPUs, achieving a 300× speedup over ODG.

3. **Adaptive Event Firing (Stage 3)**:

    - Function: Converts continuous luminance change maps into binary spike events.
    - Mechanism: Pixel $(x,y)$ fires an ON event at timestep $t$ when $\Delta V > S_{th}$, and an OFF event when $-\Delta V > S_{th}$. The threshold uses a dynamic adaptive mechanism: $S_{th} = S_{th_0} \cdot (\max(V) - \min(V))$, where $S_{th_0}$ is the sole global hyperparameter.
    - Design Motivation: A fixed global threshold yields inconsistent event rates across images with different luminance levels. The dynamic threshold adapts to each image's luminance dynamic range, ensuring consistent event sparsity across datasets. On ImageNet, $S_{th_0} = 0.12$ corresponds to approximately 5% event rate.

### Efficiency and Information-Theoretic Analysis

- **Speed**: Processing a single image on GPU takes approximately 0.1 ms, which is 30,000× faster than hardware acquisition and 300× faster than the ODG algorithm.
- **Energy**: The standard ANN first-layer convolution consumes approximately 543 μJ, the I2E encoding itself only 0.36 μJ, and the I2E-SNN first layer only 28.68 μJ, yielding an overall 18.9× energy reduction.
- **Storage**: I2E-ImageNet stored as boolean arrays occupies only 47 GB, a 67.8% reduction compared to JPEG-compressed original ImageNet (146 GB).
- **Information Retention**: The original grayscale image has an average Shannon entropy of 7.14, while the I2E event stream retains only 1.53 (less than 22% of the entropy), yet the performance degradation is limited, indicating that the discarded content is primarily redundant information (e.g., uniform textures and backgrounds).

### Loss & Training

- Architecture: MS-ResNet with LIF neurons, implemented in the SpikingJelly framework.
- Cross-entropy loss with label smoothing ($\epsilon = 0.1$) + SGD optimizer.
- Online augmentation (Baseline-II): Standard augmentations (e.g., random crop) are applied to the source image before I2E conversion, yielding substantially higher performance than Baseline-I (random flip only).

## Key Experimental Results

### Main Results (I2E-ImageNet Event Classification)

| Dataset | Architecture | Method | Accuracy (%) |
|--------|------|------|----------|
| ES-ImageNet | ResNet18+LIF | baseline | 39.89 |
| ES-ImageNet | ResNet18+LIAF | pre-train | 52.25 |
| N-ImageNet | ResNet34 | EST | 48.93 |
| **I2E-ImageNet** | **ResNet18+LIF** | **Baseline-II** | **57.97** |
| **I2E-ImageNet** | **ResNet34+LIF** | **Baseline-II** | **60.50** |
| I2E-ImageNet | ResNet18+LIF | pre-train | 59.28 |

ResNet34 on I2E-ImageNet achieves 60.50%, surpassing the previous best result on event-based ImageNet datasets (48.93%) by more than 11 percentage points.

### CIFAR Datasets + Sim-to-Real Transfer

| Dataset | Architecture | Method | Accuracy (%) |
|--------|------|------|----------|
| CIFAR10-DVS | ResNet18 | transfer-I (from I2E-ImageNet) | 83.1 |
| **CIFAR10-DVS** | **ResNet18** | **transfer-II (from I2E-CIFAR10)** | **92.5** |
| CIFAR10-DVS | SpikingResformer | transfer | 84.8 |
| I2E-CIFAR10 | ResNet18 | Baseline-II | 89.23 |
| I2E-CIFAR10 | ResNet18 | transfer-I | 90.86 |
| I2E-CIFAR100 | ResNet18 | Baseline-II | 60.68 |
| I2E-CIFAR100 | ResNet18 | transfer-I | 64.53 |

On the real DVS dataset CIFAR10-DVS, pretraining on I2E synthetic data followed by fine-tuning achieves 92.5%, outperforming the previous state of the art (84.8%) by 7.7%, validating the effectiveness of the sim-to-real paradigm.

### Ablation Study

| Configuration | Accuracy (%) | Notes |
|------|----------|------|
| Fixed threshold + no augmentation | 47.22 | Most basic conversion |
| + Dynamic threshold | 48.30 | Stabilizes event rate |
| + Random vector selection | 49.01 | Introduces data diversity |
| + Standard image augmentation (random crop, etc.) | 57.97 | Unlocked by real-time speed; largest gain |

| Timestep Order | CIFAR10 | CIFAR100 | Notes |
|-----------|---------|----------|------|
| $\gamma\alpha\beta$ (high event rate first) | **89.23** | **60.68** | Best sequence |
| $\alpha\beta\gamma$ | 87.96 | 56.10 | Worst sequence |
| $\gamma\beta\alpha$ | 88.60 | 60.12 | Second best |

### Key Findings

- The large jump from Baseline-I to Baseline-II (48.30% → 57.97%) demonstrates that online data augmentation is the most important ancillary benefit of I2E.
- Timestep ordering has a significant effect: presenting high-event-rate frames (corresponding to larger motion vectors) first yields better performance.
- The RGB→V channel conversion incurs approximately 3.5% accuracy loss (65.68% → 62.21%), and the subsequent event encoding incurs another ~3% (62.21% → 59.28%), with the total conversion loss remaining manageable.
- A tunable trade-off exists between the number of timesteps and accuracy/compression ratio: even with only 2 timesteps, the model still achieves 51.97% (compression ratio 91.95%).

## Highlights & Insights

- The elegant engineering insight of expressing image differencing as sparse convolutions is the central technical contribution of the paper, bringing conversion speed into the regime of online usability.
- The sim-to-real transfer result (92.5%) is a highly compelling experiment, demonstrating that synthetic event data can serve as a high-fidelity proxy for real sensor data.
- The work essentially bridges the rich body of static image datasets to the event-driven domain, opening the door to vast data resources for SNN training.
- The information-theoretic analysis provides valuable insight: although the event stream retains less than 22% of the original entropy, what is retained consists precisely of the salient features needed for classification.

## Limitations & Future Work

- Validation is currently limited to classification tasks; extension to more complex visual tasks such as detection and segmentation remains unexplored.
- The event streams generated by microsaccade simulation still differ from the physical characteristics of real DVS sensors (e.g., noise patterns, pixel response latency).
- $S_{th_0}$ is a manually set global hyperparameter; the possibility of adaptively learning the threshold has not been explored.
- The fixed design of 8 timesteps may limit the model's capacity to represent more complex dynamic scenes.

## Related Work & Insights

- ES-ImageNet (ODG algorithm) is the closest prior work; I2E comprehensively outperforms it in both speed and quality.
- N-ImageNet is collected by recording a monitor with a DVS camera, which is slow to acquire and subject to artifacts; I2E's purely algorithmic approach entirely avoids these issues.
- The sim-to-real paradigm introduced in this work can be adapted to domains reliant on event cameras, such as autonomous driving.
- The convolution equivalence insight can be generalized to simulate more complex motion patterns, such as rotation and scaling.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] TDSNNs: Competitive Topographic Deep Spiking Neural Networks for Visual Cortex Modeling](tdsnns_competitive_topographic_deep_spiking_neural_networks_for_visual_cortex_mo.md)
- [\[AAAI 2026\] DS-ATGO: Dual-Stage Synergistic Learning via Forward Adaptive Threshold and Backward Gradient Optimization for Spiking Neural Networks](ds-atgo_dual-stage_synergistic_learning_via_forward_adaptive_threshold_and_backw.md)
- [\[AAAI 2026\] ParaRevSNN: A Parallel Reversible Spiking Neural Network for Efficient Training and Inference](pararevsnn_a_parallel_reversible_spiking_neural_network_for_efficient_training_a.md)
- [\[ICML 2026\] Bullet Trains: Parallelizing Training of Temporally Precise Spiking Neural Networks](../../ICML2026/others/bullet_trains_parallelizing_training_of_temporally_precise_spiking_neural_networ.md)
- [\[AAAI 2026\] Semi-Supervised High Dynamic Range Image Reconstructing via Bi-Level Uncertain Area Masking](semi-supervised_high_dynamic_range_image_reconstructing_via_bi-level_uncertain_a.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Faster and Stronger: When ANN-SNN Conversion Meets Parallel Spiking Calculation
description: >-
  [ICML2025][ANN-SNN Conversion] Integrates parallel spiking calculation with ANN-SNN conversion for the first time, establishing a mathematically equivalent mapping. This achieves 72.90% Top-1 accuracy on ImageNet within an ultra-low latency of only 4 steps, accelerating inference by 19x to 38x.
tags:
  - "ICML2025"
  - "ANN-SNN Conversion"
  - "Parallel Spiking Calculation"
  - "Low-latency Inference"
  - "Quantized Activation Function"
  - "Training-Free Conversion"
date: 2026-05-08
content_hash: 60d858ad7534c08a
---

# Faster and Stronger: When ANN-SNN Conversion Meets Parallel Spiking Calculation

**Conference**: ICML2025  
**arXiv**: [2412.13610](https://arxiv.org/abs/2412.13610)  
**Code**: [GitHub](https://github.com/hzc1208/Parallel_Conversion)  
**Area**: SNN (Spiking Neural Network)  
**Keywords**: ANN-SNN Conversion, Parallel Spiking Calculation, Low-latency Inference, Quantized Activation Function, Training-Free Conversion

## TL;DR

Integrates parallel spiking calculation with ANN-SNN conversion for the first time, establishing a mathematically equivalent mapping. This achieves 72.90% Top-1 accuracy on ImageNet within an ultra-low latency of only 4 steps, accelerating inference by 19x to 38x.

## Background & Motivation

The two mainstream training paradigms of SNNs each have their own limitations:

- **STBP (Spatio-Temporal Backpropagation)**: Can obtain SNNs with extremely low latency ($\leq4\sim6$ steps), but suffers from huge training overhead (slow speed and large memory footprint), making it difficult to scale to large-scale networks.
- **ANN-SNN Conversion**: Lower training burden and higher performance upper bound, but the converted SNNs require extremely high inference latency to approach ANN accuracy. Moreover, sequential calculation based on IF neurons further amplifies the latency issue.
- **Parallel Spiking Neurons**: Existing work (Fang et al., NeurIPS 2023) has proposed parallel computing schemes, but they are limited to STBP training scenarios and neglect the influence of prior spike trains on the current step (resulting in significant bias when $\lambda^l=1$).

Core insight of this paper: Parallel calculation is more suitable for integration with high-latency conversion methods rather than being confined to STBP training.

## Method

### 1. Construction of the Parallel Conversion Matrix

Core idea: During $T$-step parallel inference, the $x$-th step determines whether the total number of fired spikes is $\geq T-x+1$.

**Prior Control Matrix** $\Lambda_{\text{pre}}^l = \frac{1}{T} \cdot \mathbf{1}$: Projects non-uniform input currents into a uniform distribution.

**Posterior Conversion Matrix**: The scaling factor for each row is $c^{l,x} = \frac{T}{x(T-x+1)}$.

Integrating the two (via reparameterization) yields the final **Parallel Conversion Matrix**:

$$\Lambda_{\text{pc}}^l = \begin{bmatrix} \frac{1}{T} & \frac{1}{T} & \cdots & \frac{1}{T} \\ \frac{1}{T-1} & \frac{1}{T-1} & \cdots & \frac{1}{T-1} \\ \vdots & & \ddots & \vdots \\ 1 & 1 & \cdots & 1 \end{bmatrix}$$

### 2. Optimal Offset and Lossless Proof (Theorem 4.1)

Corresponding to the shift term $\psi^l$ in the QCFS function, the step-wise optimal offset is derived:

$$\mathbf{b}^l = \left[\frac{\psi^l}{T}, \cdots, \frac{\psi^l}{T-x+1}, \cdots, \psi^l\right]^\top$$

- When $T = \tilde{T}$ (simulation steps = physical steps): **Lossless Conversion**, $\mathbf{r}^{l,T} = \mathbf{r}_{\text{QCFS}}^{l,\tilde{T}}$
- When $T \neq \tilde{T}$ and $\psi^l = \theta^l/2$: **Expected Lossless**, $\mathbb{E}(\mathbf{r}^{l,T} - \mathbf{r}_{\text{QCFS}}^{l,\tilde{T}}) = \mathbf{0}$

### 3. Distribution-Aware Error Calibration (DA-QCFS)

To address non-uniform training data distributions and significant inter-channel distribution discrepancies, channel-wise learnable parameters $\psi_{\text{DA}}^l, \phi_{\text{DA}}^l \in \mathbb{R}^C$ are introduced:

$$\mathbf{r}_{\text{DA}}^{l,\tilde{T}} = \frac{\theta^l + \phi_{\text{DA}}^l}{\tilde{T}} \text{Clip}\left(\left\lfloor \frac{(\mathbf{W}^l \mathbf{r}^{(l-1),\tilde{T}} + \psi_{\text{DA}}^l)\tilde{T} + \psi^l}{\theta^l} \right\rfloor, 0, \tilde{T}\right)$$

A layer-wise greedy calibration scheme is adopted: channel mean errors $\mathbf{e}_{\text{pre}}^l$ and $\mathbf{e}_{\text{post}}^l$ are first calculated, and the parameters are updated with a momentum $\alpha$.

### 4. Three-Stage Training-Free Conversion

1. **ReLU → ClipReLU**: Record the historical maximum activation of each channel per layer as $\theta^l$.
2. **ClipReLU → DA-QCFS**: Perform layer-wise error calibration using a calibration dataset.
3. **DA-QCFS → Parallel Spiking Neurons**: Merge the offset terms into biasing terms and set dual pre/post thresholds to achieve equivalent mapping.

### 5. Sorting Property and Binary Search Acceleration

Since the spike trains in parallel inference possess a **sorting property** (if a spike fires at step $x$, spikes will inevitably fire from step $x+1$ to $T$), binary search can be used to locate the first spike timing $t_{\text{fir}}$ in $O(\log T)$ time. Combined with Hadamard product optimization, the complexity of the accumulation phase drops from $O(T^2)$ to $O(T)$.

## Key Experimental Results

### Comparison with SOTA (QCFS Pre-trained ANN)

| Dataset | Method | Network | Time-step T | SNN Accuracy |
|---|---|---|---|---|
| CIFAR-10 | QCFS | VGG-16 | 4 | 93.96% |
| CIFAR-10 | **Ours** | VGG-16 | **4** | **95.50%** |
| CIFAR-100 | QCFS | ResNet-20 | 8 | 55.37% |
| CIFAR-100 | **Ours** | ResNet-20 | **8** | **69.62%** |
| ImageNet | QCFS | VGG-16 | 16 | 50.97% |
| ImageNet | **Ours** | VGG-16 | **8** | **73.92%** |
| ImageNet | COS | ResNet-34 | 10 | 72.66% |
| ImageNet | **Ours†** | ResNet-34 | **4** | **72.90%** |

### Training-Free Conversion (ImageNet)

| Method | Network | T=16 | T=32 | T=64 |
|---|---|---|---|---|
| TBC | ResNet-34 | — | 59.03% | 70.47% |
| **Ours** | ResNet-34 | **68.04%** | **72.46%** | **73.03%** |
| **Ours** | ResNet-101 | 73.86% | 76.42% | 77.01% |

### Inference Speed

Parallel inference achieves a **19x to 38x speedup** compared to sequential IF neurons ($T \geq 32$).

## Highlights & Insights

1. **Pioneering Integration**: Integrates parallel spiking calculation into ANN-SNN conversion for the first time, paving a "third path" for SNN supervised learning.
2. **Theoretical Rigor**: Proves the lossless conversion property, sorting property, and optimal offset, distinguishing it from purely empirical methods.
3. **Unified Framework**: Three scenarios, QCFS ($\tilde{T}=T$ or $\tilde{T}\neq T$) and ReLU, share the same framework, differing only in whether threshold recording and error calibration are required.
4. **Practical Acceleration**: Binary search + Hadamard product optimization reduces the inference overhead from $O(T^2)$ to $O(T)$ accumulation + $O(\log T)$ firing.
5. **Outperforming STBP in 4 Steps**: Reaches 72.90% accuracy on ImageNet ResNet-34 with only 4 steps, surpassing the 6-step STBP method (Dspike 68.19%).

## Limitations & Future Work

1. **Limited to Classification Tasks**: All experiments are restricted to image classification (CIFAR/ImageNet) and have not been validated on downstream tasks such as object detection and segmentation.
2. **Network Architecture Limitations**: Only VGG and ResNet are evaluated, without involving modern architectures like Transformers or MobileNet.
3. **Hardware Adaptation of Parallel Computing**: The calculation paradigm of the parallel conversion matrix does not perfectly align with the sequential architectures of existing neuromorphic chips (e.g., Loihi). The actual energy efficiency advantage in deployment remains to be verified.
4. **Dependency on Calibration Data**: Both DA-QCFS and Training-Free conversion require a calibration dataset; completely zero-shot scenarios are not covered.
5. **Limitations of the Sorting Property**: The binary search optimization depends on the sorting property, which may not hold when extending to more general neuron models like LIF ($\lambda < 1$).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Integrates parallel spiking calculation with conversion for the first time, offering high theoretical novelty)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Covers multiple datasets, networks, and scenarios, but lacks downstream tasks and modern architectures)
- Writing Quality: ⭐⭐⭐⭐ (Clear theoretical derivations and a comprehensive mathematical notation system)
- Value: ⭐⭐⭐⭐⭐ (Provides a brand-new paradigm for the efficient deployment of SNNs, holding significant practical value)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ParaRevSNN: A Parallel Reversible Spiking Neural Network for Efficient Training and Inference](../../AAAI2026/others/pararevsnn_a_parallel_reversible_spiking_neural_network_for_efficient_training_a.md)
- [\[AAAI 2026\] I2E: Real-Time Image-to-Event Conversion for High-Performance Spiking Neural Networks](../../AAAI2026/others/i2e_real-time_image-to-event_conversion_for_high-performance_spiking_neural_netw.md)
- [\[CVPR 2026\] When AVSR Meets Video Conferencing: Dataset, Degradation, and the Hidden Mechanism Behind Performance Collapse](../../CVPR2026/others/when_avsr_meets_video_conferencing_dataset_degradation_and_the_hidden_mechanism_.md)
- [\[CVPR 2025\] Towards Million-Scale Adversarial Robustness Evaluation With Stronger Individual Attacks](../../CVPR2025/others/towards_million-scale_adversarial_robustness_evaluation_with_stronger_individual.md)
- [\[ACL 2025\] Attention Entropy is a Key Factor for Parallel Context Encoding](../../ACL2025/others/attention_entropy_parallel_encoding.md)

</div>

<!-- RELATED:END -->

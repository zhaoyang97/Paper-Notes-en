---
title: >-
  [Paper Note] DS-ATGO: Dual-Stage Synergistic Learning via Forward Adaptive Threshold and Backward Gradient Optimization for Spiking Neural Networks
description: >-
  [AAAI 2026][Spiking Neural Networks] To address spike firing imbalance and gradient vanishing caused by membrane potential distribution shifts during SNN training, this paper proposes DS-ATGO — a dual-stage synergistic learning algorithm combining forward adaptive thresholding (AT) and backward threshold-driven gradient optimization (TGO) — achieving state-of-the-art performance on CIFAR-10/100 and ImageNet with low time-step latency.
tags:
  - AAAI 2026
  - Spiking Neural Networks
  - Adaptive Threshold
  - Surrogate Gradient Optimization
  - Membrane Potential Dynamics
  - Low-Latency Inference
date: 2026-05-08
content_hash: 44d10d1699b9d5a3
---

# DS-ATGO: Dual-Stage Synergistic Learning via Forward Adaptive Threshold and Backward Gradient Optimization for Spiking Neural Networks

**Conference**: AAAI 2026
**arXiv**: [2511.13050](https://arxiv.org/abs/2511.13050)
**Code**: [github.com/jqjiang1999/DS-ATGO](https://github.com/jqjiang1999/DS-ATGO)
**Area**: Spiking Neural Networks / Neuromorphic Computing
**Keywords**: Spiking Neural Networks, Adaptive Threshold, Surrogate Gradient Optimization, Membrane Potential Dynamics, Low-Latency Inference

## TL;DR

To address spike firing imbalance and gradient vanishing caused by membrane potential distribution shifts during SNN training, this paper proposes DS-ATGO — a dual-stage synergistic learning algorithm combining forward adaptive thresholding (AT) and backward threshold-driven gradient optimization (TGO) — achieving state-of-the-art performance on CIFAR-10/100 and ImageNet with low time-step latency.

## Background & Motivation

Spiking Neural Networks (SNNs), as a biologically inspired computing paradigm, perform asynchronous computation using discrete spike signals, offering inherent spatiotemporal processing capabilities and high energy efficiency. However, direct training of SNNs faces a fundamental challenge: the spike firing function (Heaviside function) is non-differentiable, impeding gradient backpropagation.

**Surrogate gradient (SG) learning** is currently the dominant SNN training approach, approximating gradients by replacing the Heaviside function with a continuous smooth function. Nevertheless, existing SG methods suffer from two critical issues:

### Problem 1: Fixed Thresholds Lead to Imbalanced Spike Firing

A neuron fires a spike only when its membrane potential exceeds the threshold $V_{th}$. As spikes propagate across layers, the membrane potential distribution shifts:
- When membrane potential is far below threshold: excessively sparse firing → "spike vanishing problem"
- When membrane potential is far above threshold: excessive firing → loss of discriminability for input patterns

Neuroscience research has shown that the thresholds of biological neurons are not fixed; rather, they exhibit *threshold plasticity*, dynamically adjusting based on the neuron's activity history.

### Problem 2: Fixed Surrogate Gradients Lead to Gradient Signal Attenuation

Surrogate gradient functions (e.g., rectangular functions) provide non-zero gradients only within a limited interval around the threshold. When the membrane potential distribution shifts:
- Small variance → large proportion of membrane potentials fall within the gradient interval → accumulated gradient approximation error
- Large variance → most membrane potentials fall outside the gradient interval → gradient vanishing

**Core Insight**: There exists an intrinsic coupling among membrane potential, threshold, and surrogate gradient; however, existing methods adjust either the threshold or the SG in isolation, neglecting this synergistic relationship.

## Method

### Overall Architecture

DS-ATGO adopts a dual-stage synergistic learning framework (as illustrated in Fig. 3):
- **Forward pass (green)**: The adaptive threshold (AT) mechanism dynamically adjusts the threshold according to the membrane potential distribution at each time step.
- **Backward pass (yellow)**: The threshold-driven gradient optimization (TGO) method dynamically scales the surrogate gradient width based on changes in the adaptive threshold.

### Key Designs

#### 1. Adaptive Threshold Mechanism (AT)

**Core Theorem (Theorem 1)**: When the membrane potential $U(t) \sim N(\mu, \sigma^2)$, setting the threshold to $V_{th} = \mu + \sigma$ ensures that the probability of the membrane potential exceeding the threshold is constantly $P = 1 - \Phi(1) \approx 15.87\%$, **independent of the specific values of $\mu$ and $\sigma$**.

Therefore, by tracking the threshold to follow the mean plus standard deviation of the membrane potential distribution, a stable firing rate can be guaranteed. The specific formula is:

$$\Delta V_{th}^l(t)_n = f_c \cdot (\mathbb{E}(U^l(t)_n) + \sqrt{\mathbb{VAR}(U^l(t)_n)})$$

where $f_c$ is a factor controlling the trade-off between energy efficiency and performance.

**Inference Threshold Stabilization**: Inspired by Batch Normalization, a moving average is used to stabilize the threshold during inference:
$$\Delta V_{th}^l(t) = m \cdot \Delta V_{th}^l(t)_n + (1-m) \cdot \Delta V_{th}^l(t)$$
with momentum coefficient $m=0.1$.

**Design Motivation**: By establishing a positive correlation between the threshold and the membrane potential, this mechanism preserves the adaptive properties of biological neurons, maintaining a moderately active firing rate (approximately $15\% \pm 1.62\%$) per layer per time step.

#### 2. Threshold-Driven Gradient Optimization (TGO)

Although the adaptive threshold shifts the center position of the SG, the **width** of the SG remains fixed, leading to imprecise gradient matching. TGO dynamically adjusts the SG width based on the discrepancy between the adaptive threshold and the initial threshold:

$$k = \begin{cases} (1 - \tanh(V_{th} - \Delta V_{th})) \cdot k, & \Delta V_{th} < V_{th} \\ (1 + \tanh(\Delta V_{th} - V_{th})) \cdot k, & \Delta V_{th} \geq V_{th} \end{cases}$$

**Intuition for the two cases**:
- $\Delta V_{th} < V_{th}$ (concentrated membrane potential distribution): Narrow the SG width to reduce the proportion of neurons falling within the gradient interval, suppressing the accumulation of gradient approximation errors.
- $\Delta V_{th} \geq V_{th}$ (dispersed membrane potential distribution): Widen the SG width to increase the proportion of neurons receiving gradients, mitigating gradient information loss.

**Key Innovation**: By leveraging the adaptive threshold as a bridge between the SG and the membrane potential distribution, TGO achieves spatiotemporally aligned gradient optimization.

### Loss & Training

- The output layer uses accumulated membrane potential (leaky integrate without firing), averaged over $T$ time steps.
- Standard cross-entropy loss is used for training.
- Optimizer: SGD with cosine annealing.
- The time constant $\tau$ is set as a learnable parameter per layer.
- No additional inference overhead is introduced (AT and TGO involve only threshold and gradient scaling).

## Key Experimental Results

### Main Results

| Dataset | Method | Architecture | Time Steps | Accuracy (%) |
|--------|------|------|--------|---------|
| CIFAR-10 | **DS-ATGO** | ResNet-19 | **2** | **96.91±0.12** |
| CIFAR-10 | MPD-AGL | ResNet-19 | 4/2 | 96.35/96.18 |
| CIFAR-10 | DeepTAGE | ResNet-18 | 4 | 95.86 |
| CIFAR-10 | LT-SNN | Spikformer-4-256 | 4 | 95.19 |
| CIFAR-100 | **DS-ATGO** | ResNet-19 | **2** | **80.59±0.17** |
| CIFAR-100 | SNN-ViT | VGG-16 | 4 | 80.01 |
| CIFAR-100 | MPD-AGL | ResNet-19 | 4/2 | 79.72/78.84 |
| CIFAR10-DVS | **DS-ATGO** | VGGSNN | 10 | **83.70±0.41** |
| CIFAR10-DVS | MPD-AGL | VGGSNN | 10 | 82.50 |
| ImageNet | **DS-ATGO** | ResNet-18 | 4 | **68.86±0.25** |
| ImageNet | DeepTAGE | ResNet-18 | 4 | 68.52 |

DS-ATGO achieves state-of-the-art performance across all datasets, surpassing competing methods that use 4–6 time steps with only **2 time steps** on CIFAR-10/100.

### Ablation Study

| Configuration | CIFAR-10 Gain | CIFAR10-DVS Gain | Notes |
|------|-------------|-----------------|------|
| Vanilla-SNN | Baseline | Baseline | Fixed threshold + fixed SG |
| w/ AT only | +0.71% | +0.90% | Adaptive threshold improves information encoding |
| w/ TGO only | +1.05% | +1.40% | Gradient optimization more effective than threshold adjustment alone |
| **w/ AT+TGO** | **+1.71%** | **+2.30%** | Dual-stage synergy yields significant improvement |
| Estimated vs. true distribution | True superior | True superior | Approximation error exists in theorem derivation |
| w/ vs. w/o moving average | w/ superior | w/ superior (esp. DVS) | Stabilizes inference threshold |

### Key Findings

1. **Firing Rate Stability**: DS-ATGO maintains per-layer firing rates within a narrow range of $15\% \pm 1.62\%$, whereas Vanilla-SNN exhibits large inter-layer fluctuations (mean 10.66% with substantial variance); DIET-SNN and LTMD increase firing rates but with even greater variability.
2. **Gradient Availability**: DS-ATGO raises the gradient availability rate of each layer in ResNet-19 to above 38.53%, with deep layers still maintaining 36.54%, while Vanilla-SNN's deep layers fall to below 15.13%.
3. **Loss Landscape**: DS-ATGO produces a flatter and sparser loss landscape, indicating superior generalization. Vanilla-SNN exhibits two local minima and pronounced protrusions in its contour lines.
4. **TGO > AT**: TGO alone outperforms AT alone, as TGO synchronously adjusts the SG across the temporal dimension to accurately capture membrane potential deviations, enhancing the efficiency and correctness of parameter updates.

## Highlights & Insights

- **Biologically Inspired Theoretical Guarantee**: Theorem 1 provides a mathematical basis for adaptive threshold setting (the $\mu + \sigma$ principle), guaranteeing a constant firing probability.
- **Dual-Stage Synergistic Design**: The forward (threshold) and backward (gradient) stages are not independent; information is transmitted through threshold variations, enabling truly synergistic optimization.
- **Low-Latency Advantage**: State-of-the-art results are achieved with only 2 time steps, which is critical for SNN deployment (fewer time steps = shorter latency + lower energy consumption).
- **Zero Inference Overhead**: The moving average in AT and the gradient scaling in TGO introduce no additional inference computation.

## Limitations & Future Work

- Validation is limited to classification tasks; extension to dense prediction tasks such as detection and segmentation remains unexplored.
- Theorem 1 assumes a Gaussian distribution for membrane potentials, which may not always hold in highly nonlinear deep networks.
- The factor $f_c$ requires manual tuning and may need task-specific adjustment.
- The ImageNet accuracy (68.86%) still lags significantly behind ANNs; the scalability of SNNs remains an open problem.
- Specific energy efficiency metrics — one of the core selling points of SNNs — are not discussed.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The perspective of threshold–gradient synergistic optimization is novel, though individual components build on prior work.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive ablation covering firing rate, gradient availability, and loss landscape; four datasets spanning static and neuromorphic benchmarks.
- **Writing Quality**: ⭐⭐⭐⭐ — Method description is clear and figures are intuitive, though some notation is slightly redundant.
- **Value**: ⭐⭐⭐⭐ — Highly valuable to the SNN community; low-latency SOTA results carry practical significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] TDSNNs: Competitive Topographic Deep Spiking Neural Networks for Visual Cortex Modeling](tdsnns_competitive_topographic_deep_spiking_neural_networks_for_visual_cortex_mo.md)
- [\[AAAI 2026\] I2E: Real-Time Image-to-Event Conversion for High-Performance Spiking Neural Networks](i2e_real-time_image-to-event_conversion_for_high-performance_spiking_neural_netw.md)
- [\[AAAI 2026\] ParaRevSNN: A Parallel Reversible Spiking Neural Network for Efficient Training and Inference](pararevsnn_a_parallel_reversible_spiking_neural_network_for_efficient_training_a.md)
- [\[ICLR 2026\] Training Deep Normalization-Free Spiking Neural Networks with Lateral Inhibition](../../ICLR2026/others/training_deep_normalization-free_spiking_neural_networks_with_lateral_inhibition.md)
- [\[AAAI 2026\] Autonomous Concept Drift Threshold Determination](autonomous_concept_drift_threshold_determination.md)

</div>

<!-- RELATED:END -->

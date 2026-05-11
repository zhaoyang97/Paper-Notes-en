---
title: >-
  [Paper Note] Training Deep Normalization-Free Spiking Neural Networks with Lateral Inhibition
description: >-
  This paper proposes DeepEISNN, a normalization-free learning framework based on cortical excitatory-inhibitory (E-I) circuits. Through two techniques—E-I Init and E-I Prop—it achieves stable end-to-end training of deep…
tags:

date: 2026-05-08
content_hash: 5346122ddbdd4409
---

# Training Deep Normalization-Free Spiking Neural Networks with Lateral Inhibition

## Paper Information
- **Conference**: ICLR 2026
- **arXiv**: [2509.23253](https://arxiv.org/abs/2509.23253)
- **Code**: [https://github.com/vwOvOwv/DeepEISNN](https://github.com/vwOvOwv/DeepEISNN)
- **Area**: Spiking Neural Networks / Neuromorphic Computing / Bio-inspired Computing
- **Keywords**: SNN, lateral inhibition, excitatory-inhibitory circuit, normalization-free training, biological plausibility

## TL;DR
This paper proposes DeepEISNN, a normalization-free learning framework based on cortical excitatory-inhibitory (E-I) circuits. Through two techniques—E-I Init and E-I Prop—it achieves stable end-to-end training of deep SNNs while balancing performance and biological plausibility.

## Background & Motivation

### Root Cause
SNN training faces a fundamental **performance vs. biological plausibility** trade-off:
- **High-performance methods** (backpropagation + batch normalization): treat SNNs as standard deep learning components at the cost of basic biological properties.
- **High biological plausibility methods** (STDP, etc.): training is unstable and applicable only to shallow networks.

### Why Eliminate Normalization?
Normalization schemes such as BatchNorm collect statistics from entire batches of inputs, which **have no known analogue in biological systems**. This makes SNNs that rely on normalization unsuitable as computational platforms for large-scale cortical computation modeling.

### Importance of E-I Circuits
Approximately 80% of cortical neurons are excitatory and 20% are inhibitory. E-I interactions play critical roles in gain control, neural oscillations, and selective attention, yet most existing deep SNNs neglect this fundamental principle.

## Method

### E-I Circuit Design

Each layer contains $n_E^{[l]}$ excitatory and $n_I^{[l]}$ inhibitory neurons in a 4:1 ratio.

**Excitatory neurons (LIF model)**:
$$\mathbf{u}_E^{[l]}[t+1] = \left(1-\frac{1}{\tau_E}\right)\left(\mathbf{u}_E^{[l]}[t] - \theta_E \cdot \mathbf{s}_E^{[l]}[t]\right) + \mathbf{I}_E^{[l]}[t]$$

**Inhibitory neurons (fast-spiking approximation)**:
$$\mathbf{s}_I^{[l]}[t] \approx \max(0, \mathbf{I}_I^{[l]}[t])$$

Since $\tau_I \ll \tau_E$, inhibitory neurons are approximated as instantaneous steady-state units, producing ReLU-like outputs.

**Decomposition of lateral inhibition**:
- **Subtractive inhibition** (E-I balance): $\mathbf{I}_{EI,\text{sub}}^{[l]}[t] = \boldsymbol{W}_{EI}^{[l]} \mathbf{s}_I^{[l]}[t]$
- **Divisive inhibition** (gain control): $\mathbf{I}_{EI,\text{div}}^{[l]}[t] = \boldsymbol{W}_{EI}^{[l]}(\mathbf{g}_I^{[l]} \odot \mathbf{s}_I^{[l]}[t])$

**Input integration**:
$$\mathbf{I}_{\text{int}}^{[l]}[t] = \mathbf{g}_E^{[l]} \odot \frac{\mathbf{I}_{EE}^{[l]}[t] - \mathbf{I}_{EI,\text{sub}}^{[l]}[t]}{\mathbf{I}_{EI,\text{div}}^{[l]}[t]} + \mathbf{b}_E^{[l]}$$

### E-I Init: Dynamic Parameter Initialization

Standard Xavier/Kaiming initialization is incompatible with E-I constraints (weights must be sign-constrained).

**Objective 1: E-I balance (subtractive inhibition)**
$$\mathbb{E}[\mathbf{I}_{EE,i}^{[l]}] \approx \mathbb{E}[\mathbf{I}_{EI,\text{sub},i}^{[l]}]$$

Excitatory weights are initialized using an exponential distribution; inhibitory weights are set to $1/n_I^{[l]}$.

**Objective 2: Gain control (divisive inhibition)**
$$\mathbb{E}[\mathbf{I}_{EI,\text{div},i}^{[l]}] = \text{std}(\mathbf{I}_{EE,i}^{[l]}]$$

Each element of $\mathbf{g}_I^{[l]}$ is set to $\sqrt{\frac{2-p}{dp}}$, achieving a normalization-like effect.

**Dynamic estimation of mean firing probability $p$**: computed from the first mini-batch of the training set.

### E-I Prop: Stable End-to-End Training

**Adaptive stabilization**: instead of a fixed $\epsilon$, zero-valued denominators are dynamically replaced with the smallest positive value within each sample.

**Straight-through estimator (STE)**: the adaptive stabilization is applied in the forward pass, while the replacement operation is treated as the identity function during backpropagation.

**Gradient scaling**: gradients of $\boldsymbol{W}_{EI}^{[l]}$ are scaled by $1/d$ to balance update magnitudes between the feedforward and lateral pathways.

## Experiments

### Classification Performance

| Dataset | Method | Architecture | E-I | BN-free | Accuracy (%) |
|--------|------|------|-----|---------|----------|
| CIFAR-10 | Vanilla BN | ResNet-18 | ✗ | ✗ | 95.37 |
| CIFAR-10 | TEBN | ResNet-19 | ✗ | ✗ | 94.70 |
| CIFAR-10 | **DeepEISNN** | **ResNet-18** | **✓** | **✓** | **92.05** |
| CIFAR-10 | DANN (ANN) | VGG-16 | ✓ | ✓ | 88.54 |
| CIFAR-10 | BackEISNN | 5-layer CNN | ✓ | ✓ | 90.93 |
| DVS-Gesture | DeepEISNN | VGG-8 | ✓ | ✓ | **94.86** |
| CIFAR10-DVS | DeepEISNN | VGG-8 | ✓ | ✓ | **77.66** |

### Key Findings

1. DeepEISNN (ResNet-18) achieves 92.05% on CIFAR-10, **surpassing all normalization-free baselines**.
2. On neuromorphic datasets (DVS-Gesture, CIFAR10-DVS), it **outperforms several BN-based methods**.
3. It achieves 50.29% on TinyImageNet, demonstrating scalability to larger datasets.
4. Both E-I Init and E-I Prop are individually necessary—removing either component causes training collapse.

### Ablation Study

- Without E-I Init → training failure (firing rate collapse).
- Without adaptive stabilization → numerical explosion.
- Without STE → incorrect gradient direction.
- Without gradient scaling → excessively large gradients in the inhibitory pathway.

## Highlights & Insights

1. **First normalization-free training framework for deep SNNs** that maintains competitive performance.
2. **Balance between biological plausibility and engineering performance**: the E-I circuit serves not merely as a regularization trick but as a biologically grounded modeling component.
3. **Rigorous theoretical analysis**: derivations span from exponential distribution initialization to gain control conditions.
4. **Provides a platform for large-scale cortical computation simulation**.

## Limitations & Future Work

1. An accuracy gap of ~3% remains compared to BN-based SNNs.
2. Whether the fixed 4:1 E-I ratio is optimal has not been explored.
3. The fast-spiking approximation reduces inhibitory neurons to ReLU units, which may oversimplify their dynamics.
4. Validation is limited to classification tasks; generative and sequence modeling tasks remain untested.

## Related Work & Insights
- **SNN normalization**: BNTT, tdBN, TEBN, TAB — BN variants designed for SNNs.
- **E-I networks**: Cornford et al. (2021) — E-I networks in ANNs.
- **SNN training**: STBP, TEBN — surrogate gradient methods and normalization techniques.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The idea of replacing normalization with E-I circuits is novel and biologically grounded.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Validated across multiple datasets and architectures.
- **Writing Quality**: ⭐⭐⭐⭐ — The derivation from biological principles to engineering implementation is clearly presented.
- **Value**: ⭐⭐⭐ — A performance gap remains, but the work provides an important foundation for NeuroAI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] TDSNNs: Competitive Topographic Deep Spiking Neural Networks for Visual Cortex Modeling](../../AAAI2026/others/tdsnns_competitive_topographic_deep_spiking_neural_networks_for_visual_cortex_mo.md)
- [\[CVPR 2026\] Stronger Normalization-Free Transformers](../../CVPR2026/others/stronger_normalization-free_transformers.md)
- [\[AAAI 2026\] ParaRevSNN: A Parallel Reversible Spiking Neural Network for Efficient Training and Inference](../../AAAI2026/others/pararevsnn_a_parallel_reversible_spiking_neural_network_for_efficient_training_a.md)
- [\[AAAI 2026\] I2E: Real-Time Image-to-Event Conversion for High-Performance Spiking Neural Networks](../../AAAI2026/others/i2e_real-time_image-to-event_conversion_for_high-performance_spiking_neural_netw.md)
- [\[ICLR 2026\] Entropic Confinement and Mode Connectivity in Overparameterized Neural Networks](entropic_confinement_and_mode_connectivity_in_overparameterized_neural_networks.md)

</div>

<!-- RELATED:END -->

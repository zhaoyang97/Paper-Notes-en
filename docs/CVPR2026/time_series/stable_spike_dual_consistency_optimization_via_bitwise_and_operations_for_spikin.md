---
title: >-
  [Paper Note] Stable Spike: Dual Consistency Optimization via Bitwise AND Operations for Spiking Neural Networks
description: >-
  [CVPR 2026][Time Series][spiking neural networks] This paper proposes Stable Spike, a dual consistency optimization framework that employs the hardware-friendly bitwise AND operation to decouple a stable spike skeleton $…
tags:
  - "CVPR 2026"
  - "Time Series"
  - "spiking neural networks"
  - "temporal step consistency"
  - "bitwise AND"
  - "stable spike skeleton"
  - "amplitude-aware noise"
  - "neuromorphic recognition"
  - "low-latency inference"
date: 2026-05-08
content_hash: c276d4c2e39251ca
---

# Stable Spike: Dual Consistency Optimization via Bitwise AND Operations for Spiking Neural Networks

**Conference**: CVPR 2026
**arXiv**: [2603.11676](https://arxiv.org/abs/2603.11676)  
**Code**: To be confirmed  
**Area**: Time Series
**Keywords**: spiking neural networks, temporal step consistency, bitwise AND, stable spike skeleton, amplitude-aware noise, neuromorphic recognition, low-latency inference

## TL;DR

This paper proposes Stable Spike, a dual consistency optimization framework that employs the hardware-friendly bitwise AND operation to decouple a stable spike skeleton $\tilde{S}$ from multi-timestep spike maps, and injects amplitude-aware spike noise to enhance generalization. The method achieves up to 8.33% accuracy improvement on neuromorphic object recognition tasks under ultra-low latency ($T=2$).

## Background & Motivation

**Low-power advantage of SNNs**: Spiking neural networks transmit information via sparse binary spikes, requiring only addition operations on neuromorphic chips, resulting in far lower power consumption than conventional ANNs and representing a key paradigm for energy-efficient AI.

**Temporal inconsistency problem**: Differences in neuron states and input currents across timesteps cause spike maps to vary excessively between steps, severely degrading overall representation quality and prediction stability.

**Early timesteps are particularly disordered**: Because membrane potentials are typically initialized to zero, outputs at early timesteps are considerably more chaotic than those at later steps—a critical issue in low-latency inference scenarios.

**Limitations of Prior Work**: Methods such as MPS promote consistency by modifying neuron dynamics, but this requires altering the neuron model, making universal deployment on neuromorphic chips difficult since neuron models are typically fixed on-chip.

**Special requirements for SNN noise**: Unlike ANNs, where Gaussian noise can be applied directly, the binary discrete nature of SNNs demands discrete noise; otherwise, train–inference accuracy mismatches arise. Moreover, spike firing rates are more sensitive to noise amplitude.

**Practical demand for ultra-low latency**: Neuromorphic object recognition targets low-latency ($T \leq 4$) inference, yet existing methods typically require $10+$ timesteps to achieve competitive performance, highlighting an urgent need for performance improvements at low latency.

## Method

### Overall Architecture

Stable Spike comprises two core modules forming **dual consistency optimization**:

- **Spike Map Consistency**: The bitwise AND operation decouples a stable spike skeleton $\tilde{S}$ from spike maps at adjacent timesteps, which serves as an anchor to guide the original spike maps toward convergence.
- **Perturbation Consistency**: Amplitude-aware spike noise is injected into the stable spike firing rate, encouraging the SNN to produce consistent predictions under perturbation and thereby enhancing generalization.

The overall loss is: $\mathcal{L}_{total} = \mathcal{L}_{CE} + \beta \mathcal{L}_{spike} + \gamma \mathcal{L}_{noise}$

### Key Design 1: Decoupling Stable Spikes via AND Operation

The bitwise AND operation is applied to spike maps at adjacent timesteps $t$ and $t+1$, retaining only positions where both steps fire:

$$\tilde{S}_{i,t} = S_{i,t} \mathbin{\&} S_{i,t+1}$$

- $T-1$ stable spike maps are extracted from $T$ spike maps.
- The stable spike firing rate $\tilde{\Phi} = \frac{1}{T-1}\sum_{t=0}^{T-2}\tilde{S}_t$ is computed as the feature skeleton.
- The AND operation naturally filters out unstable noise spikes and retains semantically consistent features.
- Compared to OR/XOR, AND exclusively retrieves $(1,1)$ pairs, yielding the highest semantic purity.

### Key Design 2: Amplitude-Aware Spike Noise

Noise probability is proportional to the stable spike firing rate, enabling adaptive perturbation:

$$\varepsilon_{c,i,j} = \text{Bernoulli}(\tilde{\Phi}_{c,i,j})$$

- **High firing-rate elements**: Perturbed with high probability, sufficiently promoting generalization.
- **Low firing-rate elements**: Perturbed with low probability, preserving critical semantic information.
- Noise is discrete and binary, maintaining the same data format as SNN spikes and avoiding train–inference mismatches.

The perturbed firing rate $\Phi_{noise} = \tilde{\Phi} + \varepsilon$ is forwarded to obtain the noise prediction $O_{noise}$.

### Loss & Training

| Loss | Formula | Role |
|------|---------|------|
| Spike consistency loss | $\mathcal{L}_{spike} = \text{MSE}(\tilde{\Phi}, \Phi)$ | Guides original spike maps to converge toward the stable skeleton |
| Perturbation consistency loss | $\mathcal{L}_{noise} = \alpha^2 \text{KL}(O \| O_{noise})$ | Encourages consistent predictions under noise perturbation |
| Classification loss | $\mathcal{L}_{CE}$ | Standard cross-entropy |

The temperature parameter is $\alpha=2$ and the balancing coefficients are $\beta=\gamma=1.0$. Stable spikes are computed only on backbone features; the only additional overhead is a single forward pass through the classifier.

## Key Experimental Results

### Main Results

**Neuromorphic datasets (low latency $T=4$)**:

| Method | Architecture | T | CIFAR10-DVS | DVS-Gesture | N-Caltech101 |
|--------|-------------|---|-------------|-------------|--------------|
| TAB (ICLR'24) | VGG-9 | 4 | - | 87.50 | - |
| SLT (AAAI'24) | VGG-9 | 4 | - | 88.19 | - |
| CLIF (ICML'24) | VGG-9 | 4 | - | 89.58 | - |
| **Ours** | **VGG-9** | **4** | **77.1** | **94.44** | **83.92** |
| QKFormer (NeurIPS'24) | QKFormer | 4 | 81.2 | 93.75 | - |
| **Ours** | **QKFormer** | **4** | **82.9** | **95.49** | - |

**ImageNet ($T=4$, ResNet-34)**: Achieves 70.59%, surpassing all baselines including MPS (69.03%) and STAA-SNN (70.40%).

### Ablation Study

**Effect of dual-loss combination (VGG-9, $T=4$)**:

| Configuration | CIFAR10-DVS | DVS-Gesture |
|--------------|-------------|-------------|
| Baseline | 72.9 | 87.15 |
| +$\mathcal{L}_{spike}$ | 75.2 (+2.4) | 91.32 (+4.17) |
| +$\mathcal{L}_{noise}$ | 75.4 (+2.6) | 94.09 (+6.94) |
| +Both | **77.1 (+4.2)** | **94.44 (+7.29)** |

**Bitwise operation selection**: AND outperforms OR (DVS-Gesture: 94.44 vs. 88.54) and XOR (89.58); OR causes severe degradation by simultaneously retrieving both consistent and inconsistent spikes.

**Noise design ablation**: Fixed-probability spike noise ($p=0.5$: 88.89% on DVS-Gesture) and continuous Gaussian noise (std=$0.5$: 91.67%) both fall significantly short of amplitude-aware spike noise (94.44%).

### Key Findings

- **Pronounced advantage at ultra-low latency**: At $T=2$, DVS-Gesture improves by 8.33% (83.68→92.01); gains are more significant at lower latency.
- **Reduced power consumption**: Spike firing rates are lower across all layers except the first, reducing overall power from 189.83 to 181.02 (×$10^6$ pJ).
- **Smoother loss landscape**: Sharp local minima are eliminated, yielding a single global optimum trend and more stable optimization.
- **Compatibility with other methods**: Can be combined with Knowledge-Transfer, achieving 94.25% on N-Caltech101.

## Highlights & Insights

- The idea of decoupling stable spikes via bitwise AND is concise and effective—hardware-friendly and plug-and-play without modifying neurons or architecture.
- Amplitude-aware spike noise elegantly addresses the dual constraints of SNN discreteness and noise sensitivity.
- Performance gains at ultra-low latency ($T=2$) are substantial, directly advancing the practicality of SNNs.
- Broad validation across architectures (VGG/ResNet/Transformer) and data types (neuromorphic/static).

## Limitations & Future Work

- At least $T \geq 2$ timesteps are required to compute AND; the method is inapplicable in the $T=1$ setting.
- The balancing coefficients $\beta, \gamma$ affect performance (92.01%–95.14%) and require dataset-specific tuning.
- Validation is limited to classification tasks; extension to downstream tasks such as detection and segmentation remains unexplored.
- Improvements on static datasets are less pronounced than on neuromorphic datasets.

## Related Work & Insights

- **MPS (ICLR'25)**: Indirectly promotes consistency via membrane potential smoothing and logit distillation between adjacent timesteps, but requires modification of neuron dynamics.
- **Knowledge-Transfer (AAAI'24)**: Transfers knowledge from static to neuromorphic data; complementary to the proposed method.
- **QKFormer (NeurIPS'24)**: A Transformer-style SNN architecture; the proposed method can further improve upon it.
- **STAA-SNN (CVPR'25)**: Focuses on spatiotemporal attention enhancement in SNNs; achieves performance comparable to the proposed method on ImageNet.
- **EnOF-SNN / BKDSNN**: Promote spatial consistency via knowledge distillation and contrastive learning, but lack a stable anchor in the temporal dimension.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The perspective of decoupling stable spikes via bitwise AND is novel, and the amplitude-aware noise design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three architectures × multiple datasets; ablations cover bitwise operations, noise, hyperparameters, timesteps, power consumption, and loss landscape.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated, method derivation is complete, and figures are rich.
- **Value**: ⭐⭐⭐⭐ — A plug-and-play SNN enhancement scheme with significant gains at ultra-low latency, meaningfully advancing the practicality of neuromorphic computing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] WARP: Weight-Space Linear Recurrent Neural Networks](../../ICLR2026/time_series/weight-space_linear_recurrent_neural_networks.md)
- [\[AAAI 2026\] Urban Incident Prediction with Graph Neural Networks: Integrating Government Ratings and Crowdsourced Reports](../../AAAI2026/time_series/urban_incident_prediction_with_graph_neural_networks_integrating_government_rati.md)
- [\[AAAI 2026\] Transparent Networks for Multivariate Time Series](../../AAAI2026/time_series/transparent_networks_for_multivariate_time_series.md)
- [\[AAAI 2026\] SELDON: Supernova Explosions Learned by Deep ODE Networks](../../AAAI2026/time_series/seldon_supernova_explosions_learned_by_deep_ode_networks.md)
- [\[ICML 2026\] DAG: A Dual Correlation Network for Time Series Forecasting with Exogenous Variables](../../ICML2026/time_series/dag_a_dual_correlation_network_for_time_series_forecasting_with_exogenous_variab.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Robust Spiking Neural Networks Against Adversarial Attacks
description: >-
  [ICLR 2026][AI Safety][Spiking Neural Networks] This paper theoretically demonstrates that threshold-proximal spiking neurons are the key robustness bottleneck in directly trained SNNs — they simultaneously set the theoretical upper bound on adversarial attack strength and are most susceptible to state flipping. The proposed Threshold Guarding Optimization (TGO) method addresses this through a dual strategy of membrane potential constraint and noisy LIF neurons, achieving state-of-the-art robustness across multiple adversarial attack scenarios with zero additional inference overhead.
tags:
  - ICLR 2026
  - AI Safety
  - Spiking Neural Networks
  - Adversarial Robustness
  - Membrane Potential Optimization
  - Threshold-Proximal Neurons
  - Noisy LIF Model
date: 2026-05-08
content_hash: 2a8d65bac668c133
---

# Robust Spiking Neural Networks Against Adversarial Attacks

**Conference**: ICLR 2026
**arXiv**: [2602.20548](https://arxiv.org/abs/2602.20548)
**Code**: To be confirmed
**Area**: AI Safety
**Keywords**: Spiking Neural Networks, Adversarial Robustness, Membrane Potential Optimization, Threshold-Proximal Neurons, Noisy LIF Model

## TL;DR
This paper theoretically demonstrates that threshold-proximal spiking neurons are the key robustness bottleneck in directly trained SNNs — they simultaneously set the theoretical upper bound on adversarial attack strength and are most susceptible to state flipping. The proposed Threshold Guarding Optimization (TGO) method addresses this through a dual strategy of membrane potential constraint and noisy LIF neurons, achieving state-of-the-art robustness across multiple adversarial attack scenarios with zero additional inference overhead.

## Background & Motivation

**Background**: Spiking Neural Networks (SNNs), leveraging event-driven mechanisms and biologically plausible spike-based communication, have emerged as a prominent paradigm for energy-efficient neuromorphic computing. Direct training methods based on surrogate gradients (e.g., STBP/BPTT) have enabled SNNs to approach ANN-level performance on classification tasks.

**Limitations of Prior Work**: Directly trained SNNs inherit the adversarial vulnerability of ANNs — carefully crafted small perturbations can cause misclassification. Existing defenses such as adversarial training (AT) and regularized adversarial training (RAT) incur additional training overhead and offer limited transferability.

**Key Challenge**: Prior robustness improvements for SNNs (e.g., gradient sparsity regularization SR, evolutionary leakage factor FEEL-SNN) only yield significant gains when combined with AT/RAT, and a unified theoretical analysis of robustness bottlenecks in SNNs remains absent.

**Goal**: Identify the fundamental cause of adversarial vulnerability in directly trained SNNs and design a defense method that requires no additional inference overhead.

**Key Insight**: By analyzing membrane potential dynamics in spiking neurons, this work identifies that threshold-proximal neurons simultaneously amplify both the upper bound of gradient attack paths and the probability of state flipping.

**Core Idea**: Push membrane potentials away from the threshold + introduce noisy spiking mechanisms → reduce the theoretical upper bound of adversarial attacks + decrease state-flipping probability.

## Method

### Theoretical Analysis: Dual Vulnerability of Threshold-Proximal Neurons

**Vulnerability 1 — Maximum Potential Attack Path Upper Bound**: The maximum potential adversarial attack strength $\mathcal{R}_{\text{adv}}(f,x,\epsilon)$ is positively correlated with the $\ell_2$ norm of the model's Jacobian matrix. Since the surrogate gradient peaks near the threshold, a greater proportion of threshold-proximal neurons leads to a larger $\|J_f(x)\|_2^2$, thereby raising the theoretical upper bound on adversarial perturbation strength.

**Vulnerability 2 — State-Flipping Probability**: Theorem 1 proves that when Gaussian noise $\eta[t] \sim \mathcal{N}(0,\sigma^2)$ is applied to the membrane potential, the neuron state-flipping probability $P_{\text{flip}}$ increases monotonically as the membrane potential approaches the threshold. Theorem 2 further proves that a greater number of threshold-proximal neurons increases the number of reachable activation regions $K$ within the perturbation ball $B_\epsilon(x)$, loosening the upper bound on adversarial robustness.

### TGO: Two Core Components

**Component 1 — Membrane potential Constraint (MC)**: A penalty term is added to the loss function at each spiking neuron layer, penalizing neurons whose membrane potentials fall within a $\delta$-neighborhood of the threshold $V_{\text{th}}$:

$$\mathcal{C}(V(t)_l) = \frac{1}{TN}\sum_{i=1}^{n}\max(0, \delta - |V(t)_i - V_{\text{th}}|)$$

The total loss adopts a Lagrangian form $\mathcal{L}(\mathbf{x},\lambda) = \mathcal{L}_{\text{oss}}(\mathbf{x}) + \lambda \sum_l \mathcal{C}(V(t)_l)$, where $\lambda$ is dynamically adjusted via cosine annealing — small values early in training allow exploration, while larger values later enforce stronger constraints — avoiding convergence difficulties associated with a fixed $\lambda$.

**Component 2 — Noisy LIF Neuron (NLIF)**: Gaussian white noise $\xi[t]$ is injected into the membrane potential, converting the deterministic firing mechanism into a probabilistic one. Theoretical derivation shows that when the membrane potential is near the threshold ($z^2 < 1$), the flipping probability decreases monotonically with noise standard deviation $\sigma$, i.e., appropriately increasing noise reduces the state-flipping sensitivity of threshold-proximal neurons.

**Synergistic Mechanism**: MC pushes the majority of neurons' membrane potentials away from the threshold; for neurons that must remain near the threshold during training, NLIF further reduces their flipping probability. The two components are complementary rather than independent.

### Key Hyperparameters
- $\lambda_{\max}$: Upper bound on membrane potential constraint strength; set to 0.4 for WRN-16 and 0.6 for VGG-11. Larger $\lambda_{\max}$ improves robustness but reduces clean accuracy.
- $\delta$: Threshold neighborhood width, controlling the triggering range of the penalty.
- Noise $\sigma$: Standard deviation of NLIF noise, requiring a balance between robustness improvement and training stability.
- Training timesteps $T=4$: All SNN models uniformly use 4-timestep simulation.

## Key Experimental Results

### Main Results: CIFAR-10 WRN-16 Multi-Attack Comparison

| Training Strategy | Method | Clean | FGSM | RFGSM | PGD10 | PGD20 | PGD40 |
|---------|------|-------|------|-------|-------|-------|-------|
| BPTT | Vanilla | 93.32 | 14.05 | 31.21 | 0.00 | 0.00 | 0.00 |
| BPTT | **TGO** | 88.79 | **51.40** | **71.38** | **6.14** | 1.52 | 0.45 |
| AT | AT | 91.32 | 39.14 | 74.31 | 17.45 | 14.41 | 12.93 |
| AT | **TGO** | 88.16 | **63.03** | **79.69** | **35.01** | **24.76** | **20.11** |
| RAT | RAT | 91.44 | 42.02 | 75.89 | 19.81 | 16.24 | 14.18 |
| RAT | **TGO** | 87.33 | **69.16** | **79.28** | **47.69** | **38.07** | **33.13** |

### Ablation Study: Component Contributions on CIFAR-100 VGG-11

| MC | NLIF | Clean (BPTT) | FGSM (BPTT) | Clean (RAT) | FGSM (RAT) | PGD40 (RAT) |
|----|------|-------------|-------------|-------------|-------------|-------------|
| ✗ | ✗ | 71.4 | 5.9 | 67.8 | 20.9 | 6.9 |
| ✓ | ✗ | 64.3 | 17.1 (+11.2) | 61.4 | 26.2 (+5.3) | 6.2 |
| ✗ | ✓ | 70.6 | 8.1 (+2.1) | 68.1 | 25.2 (+4.3) | 9.1 (+2.2) |
| ✓ | ✓ | 66.9 | **21.5 (+15.5)** | 63.3 | **33.8 (+13.0)** | **9.3 (+2.4)** |

### Advanced Attacks: MTPGD & APGD (CIFAR-100 WRN-16)

| Method | MTPGD-7 | MTPGD-40 | APGD-7 | APGD-40 |
|------|---------|----------|--------|---------|
| AT | 10.01 | 3.92 | 9.34 | 3.62 |
| SR+AT | 16.88 | 7.33 | 14.48 | 7.20 |
| **TGO+AT(EoT)** | **21.23** | **7.40** | **18.93** | **7.53** |

- TGO reduces the proportion of threshold-proximal neurons by approximately 40%, validating the theoretical hypothesis.
- Loss landscape analysis shows that TGO-optimized SNNs exhibit smoother gradient trajectories, effectively avoiding local optima.

## Highlights & Insights
- **Theory-Driven Design**: Rather than blindly adapting ANN defense methods, this work identifies robustness bottlenecks from the perspective of SNN spiking mechanisms and designs targeted defense components accordingly.
- **Zero Inference Overhead**: MC only affects the training loss; NLIF noise can be removed at inference (probabilistic training has already produced more robust weight distributions), leaving the inference stage identical to a standard SNN.
- **High Compatibility**: TGO can be combined arbitrarily with BPTT/AT/RAT, consistently yielding significant improvements across all combinations.
- **40% Reduction in Threshold-Proximal Neurons**: Visualization directly validates the correctness of the theoretical analysis.

## Limitations & Future Work
- **Clean Accuracy Drop of 3–5%**: The constraint pushing membrane potentials away from the threshold inevitably sacrifices some standard classification performance, reflecting a robustness–accuracy trade-off.
- **Evaluation Limited to Image Classification**: Generalizability to downstream tasks such as object detection and semantic segmentation has not been verified.
- **Selection of Noise Standard Deviation $\sigma$**: The paper does not adequately discuss how to automatically determine the optimal $\sigma$ for different architectures and datasets.
- **Limited Adaptive Attack Evaluation**: Although APGD and EoT are tested, a more comprehensive adaptive attack suite such as AutoAttack is not employed.
- **Future Directions**: Layer-wise adaptive $\delta$ and $\sigma$ could be explored, and knowledge distillation could be incorporated to mitigate the clean accuracy loss.

## Related Work & Insights
- **vs. SR (Gradient Sparsity Regularization)**: SR directly constrains gradient sparsity, whereas TGO targets the root cause (membrane potential distribution) to indirectly achieve stronger gradient sparsity effects. TGO outperforms SR across all attack scenarios in the experiments.
- **vs. FEEL-SNN (Evolutionary Leakage Factor)**: FEEL-SNN enhances robustness through stochastic membrane potential decay but is only effective when combined with AT; TGO substantially improves FGSM robustness even under the BPTT strategy (+37%).
- **vs. ANN Adversarial Training**: AT/RAT are transferred from ANNs without accounting for the spiking characteristics of SNNs; TGO exploits the unique properties of the spiking mechanism to design defenses that complement AT/RAT.

## Rating
- Novelty: ⭐⭐⭐⭐ Establishes a theoretical framework for SNN robustness bottlenecks from the perspective of threshold-proximal neurons; highly original viewpoint.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple architectures × multiple attacks × multiple training strategies with complete ablation; AutoAttack evaluation is missing.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are rigorous, method motivation is clear, and figures are intuitive.
- Value: ⭐⭐⭐⭐ Provides both a theoretical foundation and a practical tool for secure SNN deployment; zero inference overhead is a significant advantage.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Time Is All It Takes: Spike-Retiming Attacks on Event-Driven Spiking Neural Networks](time_is_all_it_takes_spike-retiming_attacks_on_event-driven_spiking_neural_netwo.md)
- [\[AAAI 2026\] MPD-SGR: Robust Spiking Neural Networks with Membrane Potential Distribution-Driven Surrogate Gradient Regularization](../../AAAI2026/ai_safety/mpd-sgr_robust_spiking_neural_networks_with_membrane_potential_distribution-driv.md)
- [\[AAAI 2026\] Yours or Mine? Overwriting Attacks Against Neural Audio Watermarking](../../AAAI2026/ai_safety/yours_or_mine_overwriting_attacks_against_neural_audio_watermarking.md)
- [\[ICLR 2026\] ATEX-CF: Attack-Informed Counterfactual Explanations for Graph Neural Networks](atex-cf_attack-informed_counterfactual_explanations_for_graph_neural_networks.md)
- [\[ICCV 2025\] Backdoor Attacks on Neural Networks via One-Bit Flip](../../ICCV2025/ai_safety/backdoor_attacks_on_neural_networks_via_one_bit_flip.md)

<!-- RELATED:END -->

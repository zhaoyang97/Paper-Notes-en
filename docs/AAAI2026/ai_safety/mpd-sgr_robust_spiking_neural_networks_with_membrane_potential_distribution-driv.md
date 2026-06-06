---
title: >-
  [Paper Note] MPD-SGR: Robust Spiking Neural Networks with Membrane Potential Distribution-Driven Surrogate Gradient Regularization
description: >-
  [AAAI 2026][AI Safety][Spiking Neural Networks] This work theoretically establishes a connection between SNN robustness error and surrogate gradient (SG) magnitude…
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Spiking Neural Networks"
  - "Adversarial Robustness"
  - "Surrogate Gradient"
  - "Membrane Potential Distribution"
  - "Regularization"
date: 2026-05-08
content_hash: 2b13eaf89ad54d3b
---

# MPD-SGR: Robust Spiking Neural Networks with Membrane Potential Distribution-Driven Surrogate Gradient Regularization

**Conference**: AAAI 2026
**arXiv**: [2511.12199](https://arxiv.org/abs/2511.12199)  
**Code**: None  
**Area**: AI Safety
**Keywords**: Spiking Neural Networks, Adversarial Robustness, Surrogate Gradient, Membrane Potential Distribution, Regularization

## TL;DR

This work theoretically establishes a connection between SNN robustness error and surrogate gradient (SG) magnitude, demonstrating that reducing the overlap between the membrane potential distribution (MPD) and the effective region of the SG function can effectively decrease sensitivity to adversarial perturbations. Based on this insight, the paper proposes the MPD-SGR regularization method, which substantially outperforms existing SNN defense methods under both vanilla training and adversarial training settings.

## Background & Motivation

**Spiking Neural Networks (SNNs)** encode information using binary spikes in a manner inspired by the brain. Compared to ANNs, SNNs are believed to possess inherent robustness advantages, attributed to noise-filtering properties and the stochasticity of spike-based encoding. However, as surrogate gradient (SG) methods have enabled the training of deep SNNs, these networks have become increasingly exposed to gradient-based adversarial attacks.

**Three existing lines of SNN robustness research**:

**Structural parameters**: Leakage factor $\tau$, threshold $v_{th}$, etc. (e.g., FEEL evolves the leakage factor) → exploits noise-filtering effects via membrane potential leakage

**Neural coding**: Poisson encoding is more robust than direct encoding due to its stochasticity (e.g., NDL, StoG) → exploits noise attenuation during information transmission

**Borrowing from ANNs**: Adversarial training (AT), Lipschitz regularization (RAT) → without sufficiently accounting for the unique properties of SNNs

**A neglected key factor**: Gradient magnitude reflects a model's sensitivity to input perturbations. In SNNs, gradient magnitude is primarily governed by the **interaction between the MPD and the SG function**. Prior works (InfLoR-SNN, RecDis-SNN, LSG, etc.) study MPD–SG alignment to improve training performance, but overlook its implications for robustness.

**Core motivation**: Reducing the overlap ratio between the MPD and the effective interval of the SG function → decreasing SG magnitude → reducing sensitivity to perturbations → improving robustness. However, this trade-off must be carefully balanced, as excessive reduction in overlap impedes gradient propagation and degrades training.

## Method

### Overall Architecture

MPD-SGR applies regularization during training to the membrane potential distribution at each layer, channel, and time step of the SNN, constraining the overlap area $\Omega$ between the MPD and the effective interval of the SG function. This simultaneously preserves training effectiveness and enhances robustness.

### Key Designs

1. **Theoretical Analysis of Robustness Error**

   Upper bound on error induced by adversarial perturbations:
   $|\mathcal{L}(x+\delta) - \mathcal{L}(x)| \leq |\delta \odot \nabla_x \mathcal{L}(x)|_1 + g(\delta, x)$

   Using LIF dynamics and BPTT, the input gradient minimization is reformulated as an optimization over internal network gradients:
   $\min \sum_t \left|\frac{1}{L} \sum_{l=1}^{L} (P_1 \cdot P_2 \cdot P_3) \frac{\partial \mathcal{L}}{O_l^T}\right|_1$

   Three key terms:
   - $P_1$: Perturbation term (related to the leakage factor → basis of the FEEL method)
   - $P_2$: Weight term (→ basis of Lipschitz regularization)
   - $P_3 = \prod_v \frac{\partial O_v^t}{\partial U_v^t}$: **SG term (the focus of this paper, previously neglected)**

   **Design Motivation**: Reducing the magnitude of the SG term $P_3$ directly tightens the robustness error upper bound. The SG magnitude is determined by the proportion of membrane potentials falling within the effective interval of the SG function.

2. **Theoretical Modeling of Membrane Potential Distribution**

   **Theorem 1**: In an iterative LIF model with tdBN, the membrane potential follows a Gaussian distribution:
   $\overline{U}_c^l(t) \sim \mathcal{N}(\beta_c D(\tau, t) - S(t), (\lambda_c \alpha V_{th})^2 D(\tau^2, t))$

   where $D(\tau, t) = \sum_{i=1}^{t} \tau^{t-i}$ is the cumulative decay function.

   The mean $\mu$ and standard deviation $\sigma$ of the MPD are jointly determined by tdBN parameters ($\beta_c$, $\lambda_c$), LIF parameters ($\tau$, $V_{th}$), and the time step $t$ → the MPD can be optimized by learning network parameters.

3. **Derivation of MPD–SG Overlap Area and Regularization**

   Assuming the SG function (triangular) has an effective interval $[-\gamma, \gamma]$ and the MPD is $\mathcal{N}(\mu, \sigma^2)$, the overlap area is:
   $\Omega = \Phi\left(\frac{\mu + \gamma}{\sigma}\right) - \Phi\left(\frac{\mu - \gamma}{\sigma}\right)$

   where $\Phi$ denotes the standard normal CDF. The final MPD-SGR regularization loss is:
   $\mathcal{L}_{MPD-SGR}^b = \frac{1}{LCT} \sum_{l,c,t} \left[\Phi\left(\frac{\mu_c^l(t) + \gamma}{\sigma_c^l(t)}\right) - \Phi\left(\frac{\mu_c^l(t) - \gamma}{\sigma_c^l(t)}\right)\right]$

   This sums the overlap area over each layer $l$, channel $c$, and time step $t$ (excluding the final linear output layer).

   **Design Motivation**: Smaller $\Omega$ → smaller SG magnitude → lower model sensitivity to perturbations. However, excessively small $\Omega$ disrupts gradient propagation; the coefficient $\eta$ balances robustness and training performance.

### Loss & Training

$$\mathcal{L}^b = \mathcal{L}_{task}^b + \eta \mathcal{L}_{MPD-SGR}^b$$

- $\mathcal{L}_{task}$: Standard cross-entropy classification loss
- $\eta$ controls regularization strength
- For adversarial training (AT), PGD adversarial examples are used ($k=2$, $\varepsilon=2/255$)
- Attack settings: $\varepsilon=8/255$, PGD/BIM with $k=7$ iterations, step size $\alpha=0.01$

## Key Experimental Results

### Main Results: Comparison with SOTA Methods (VGG11, T=8)

**Vanilla Training**:

| Method | Clean | FGSM | PGD | BIM |
|--------|:---:|:---:|:---:|:---:|
| REG | 92.49 | 25.18 | 0.88 | 0.60 |
| StoG | 91.64 | 16.22 | 0.28 | 0.12 |
| DLIF | 92.01 | 11.52 | 0.08 | 0.06 |
| FEEL | 90.08 | 29.17 | 6.67 | 5.99 |
| SR | 91.04 | 31.72 | 8.55 | 7.28 |
| **MPD-SGR** | **91.63** | **47.59** | **20.55** | **16.85** |
| **Gain** | -0.86 | +15.87 | +12.00 | +9.57 |

**Adversarial Training**:

| Method | Clean | FGSM | PGD | BIM |
|--------|:---:|:---:|:---:|:---:|
| RAT | 91.41 | 45.00 | 22.95 | 20.80 |
| FEEL | 89.00 | 45.62 | 29.52 | 28.39 |
| SR | 88.26 | 44.28 | 28.63 | 27.03 |
| **MPD-SGR** | **90.69** | **59.27** | **33.38** | **32.61** |
| **Gain** | -0.72 | +13.52 | +3.86 | +4.22 |

Substantial improvements are also observed on CIFAR-100 (Vanilla: FGSM +18.35%; AT: FGSM +16.35%).

### Ablation Study: Different SG Functions (CIFAR-10, VGG11)

| Model+SG | Method | Clean | FGSM | PGD | BIM |
|----------|--------|:---:|:---:|:---:|:---:|
| VGG11+Rectangular | REG | 91.85 | 24.00 | 3.13 | 2.33 |
| VGG11+Rectangular | **Ours** | 91.23 | **43.28** | **15.82** | **14.20** |
| VGG11+Sigmoid | REG | 92.15 | 19.42 | 0.24 | 0.15 |
| VGG11+Sigmoid | **Ours** | 89.38 | **37.25** | **9.26** | **7.23** |
| VGG11+Superspike | REG | 86.82 | 21.39 | 0.82 | 0.50 |
| VGG11+Superspike | **Ours** | 84.45 | **43.42** | **6.32** | **4.50** |

→ MPD-SGR consistently improves robustness across all three SG functions, validating the generalizability of the method.

### Different Encoding Schemes (Tiny-ImageNet, VGG16)

| Encoding | Method | Clean | FGSM | PGD |
|----------|--------|:---:|:---:|:---:|
| Direct (DIR) | Baseline | 57.90 | 2.04 | 0.01 |
| Direct (DIR) | +Ours | 54.78 | **14.33** | **5.72** |
| Poisson (POS) | Baseline | 48.14 | 6.79 | 2.68 |
| Poisson (POS) | +Ours | 47.83 | **20.42** | **8.21** |
| RSC | Baseline | 47.47 | 22.63 | 13.75 |
| RSC | +Ours | 46.98 | **35.06** | **17.60** |

→ MPD-SGR is compatible with different spike encoding schemes and can be applied in combination.

### Key Findings

1. **Most pronounced under Vanilla Training**: Without AT, baseline SNNs achieve near 0% accuracy under PGD; MPD-SGR raises this to ~20%.
2. **Minimal clean accuracy loss**: -0.86% (CIFAR-10) → excellent robustness–accuracy trade-off.
3. **SR also improves robustness but incurs a severe clean accuracy drop** (66.76% on CIFAR-100), whereas MPD-SGR maintains 70.42% → superior practical utility.
4. **Effective under black-box attacks** → robustness stems from the method's intrinsic properties rather than gradient obfuscation.
5. **Effective against non-gradient attacks (random noise)**: CIFAR-100 accuracy under Gaussian noise is 53.01% vs. FEEL's 32.63%.
6. **Generalizes across architectures**: Consistent improvements on both VGG11 and WRN16.

## Highlights & Insights

1. **Solid theoretical contributions**:
    - Establishes a formal relationship between SG magnitude and robustness error.
    - Proves that the MPD follows a Gaussian distribution under LIF + tdBN (Theorem 1).
    - Derives an analytic expression for the MPD–SG overlap area $\Omega$.
2. **Elegant regularization design**: The CDF-based overlap area formula is directly differentiable and requires no additional approximations for backpropagation.
3. **Strong generalizability**: Effective across SG functions, encoding schemes, architectures, and attack types.
4. **Orthogonal to existing methods**: Can be combined with adversarial training and encoding-based approaches.
5. **Bridges SG optimization and robustness research**: Prior work on SG–MPD alignment focused solely on improving training; this paper is the first to leverage it for enhancing robustness.

## Limitations & Future Work

- The $\eta$ parameter requires tuning (appendix analysis is provided but no adaptive mechanism is proposed).
- The theoretical analysis is grounded in the triangular SG function; while experiments validate effectiveness on other SG functions, extending the theory to arbitrary SG functions remains an open problem.
- Validation is limited to image classification; applicability to event-driven tasks (e.g., neuromorphic vision) has not been explored.
- The time step $T$ is fixed ($T=4$ or $T=8$); performance and efficiency on longer temporal sequences remain to be investigated.
- Theorem 1 assumes the use of tdBN; SNNs without tdBN would require re-derivation of the MPD.

## Related Work & Insights

- **Distinction from InfLoR-SNN / RecDis-SNN**: These methods constrain the MPD to improve training (ensuring an appropriate proportion of membrane potentials receive gradients), whereas this paper constrains the MPD to enhance robustness (reducing the proportion that receives gradients) → opposite objectives that are mutually complementary.
- **Distinction from FEEL**: FEEL suppresses noise across different frequency ranges through frequency encoding and attention mechanisms, acting at the input layer; MPD-SGR operates on gradient propagation across all internal layers.
- **Significance to the SNN community**: Reveals that the SG function affects not only training performance but also robustness → introduces a new design dimension for SG function selection.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (A genuinely novel perspective on SG–MPD interaction for robustness, with rigorous theoretical derivations)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (3 datasets × 2 architectures × 4 attacks × 3 SG functions × 3 encodings × AT/non-AT)
- Writing Quality: ⭐⭐⭐⭐ (Clear logical chain from theory to method to experiments)
- Value: ⭐⭐⭐⭐⭐ (Provides a theoretically grounded and general defense strategy for SNN robustness)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Robust Spiking Neural Networks Against Adversarial Attacks](../../ICLR2026/ai_safety/robust_spiking_neural_networks_against_adversarial_attacks.md)
- [\[ICLR 2026\] Time Is All It Takes: Spike-Retiming Attacks on Event-Driven Spiking Neural Networks](../../ICLR2026/ai_safety/time_is_all_it_takes_spike-retiming_attacks_on_event-driven_spiking_neural_netwo.md)
- [\[ICML 2026\] Frequency Matching in Spiking Neural Networks for mmWave Sensing](../../ICML2026/ai_safety/frequency_matching_in_spiking_neural_networks_for_mmwave_sensing.md)
- [\[AAAI 2026\] Angular Gradient Sign Method: Uncovering Vulnerabilities in Hyperbolic Networks](angular_gradient_sign_method_uncovering_vulnerabilities_in_h.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](robust_watermarking_on_gradient_boosting_decision_trees.md)

</div>

<!-- RELATED:END -->

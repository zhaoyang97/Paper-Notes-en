---
title: >-
  [Paper Note] MPD-SGR: Robust Spiking Neural Networks with Membrane Potential Distribution-Driven Surrogate Gradient Regularization
description: >-
  [AAAI 2026][AI Safety][Spiking Neural Networks] A theoretical connection is established between SNN robustness error and surrogate gradient (SG) magnitude. This reveals that reducing the overlap ratio between the membrane potential distribution (MPD) and the effective gradient interval of the SG can effectively decrease sensitivity to adversarial perturbations. Based on this, the MPD-SGR regularization method is proposed, which significantly outperforms existing SNN defense m…
tags:
  - "AAAI 2026"
  - "AI Safety"
  - "Spiking Neural Networks"
  - "Adversarial Robustness"
  - "Surrogate Gradient"
  - "Membrane Potential Distribution"
  - "Regularization"
date: 2026-05-08
content_hash: fd7e11572bf56be2
---

# MPD-SGR: Robust Spiking Neural Networks with Membrane Potential Distribution-Driven Surrogate Gradient Regularization

**Conference**: AAAI 2026  
**arXiv**: [2511.12199](https://arxiv.org/abs/2511.12199)  
**Code**: None  
**Area**: AI Safety  
**Keywords**: Spiking Neural Networks, Adversarial Robustness, Surrogate Gradient, Membrane Potential Distribution, Regularization

## TL;DR

A theoretical connection is established between SNN robustness error and surrogate gradient (SG) magnitude. This reveals that reducing the overlap ratio between the membrane potential distribution (MPD) and the effective gradient interval of the SG can effectively decrease sensitivity to adversarial perturbations. Based on this, the MPD-SGR regularization method is proposed, which significantly outperforms existing SNN defense methods under both vanilla and adversarial training settings.

## Background & Motivation

**Spiking Neural Networks (SNNs)** simulate the brain's mechanism of encoding information via binary spikes, which offers intrinsic robustness advantages over ANNs—noise filtering properties and the randomness of spike coding are widely considered key factors. However, as surrogate gradient (SG) methods make training deep SNNs feasible, SNNs have also started to be exposed to the threat of gradient-based adversarial attacks.

**Three directions of existing SNN robustness research**:

**Structural parameters**: leakage factor $\tau$, threshold $v_{th}$, etc. (e.g., evolution of leakage factors in FEEL) $\rightarrow$ exploiting the noise-filtering effect of membrane potential leakage properties.

**Neural coding**: the randomness of Poisson coding is more robust than direct coding (e.g., NDL, StoG) $\rightarrow$ utilizing noise attenuation during information transmission.

**Adopting from ANNs**: Adversarial Training (AT), Lipschitz regularization (RAT) $\rightarrow$ but without fully considering the uniqueness of SNNs.

**The overlooked key factor**: Gradient magnitude reflects the model's sensitivity to input perturbations, and in SNNs, the gradient magnitude is primarily determined by **the interaction between the membrane potential distribution (MPD) and SG functions**. Existing works (such as InfLoR-SNN, RecDis-SNN, LSG, etc.) study the alignment of MPD and SG to improve training performance, but neglect its impact on robustness.

**Core Motivation**: Reducing the overlap ratio between MPD and the effective gradient interval of SG $\rightarrow$ decreasing SG magnitude $\rightarrow$ mitigating sensitivity to perturbations $\rightarrow$ enhancing robustness. However, a careful balance is required—excessive reduction of this overlap can hinder gradient propagation and impair training.

## Method

### Overall Architecture

During training, MPD-SGR applies regularization to the membrane potential distribution at every layer, channel, and timestep of the SNN. This constrains its overlap area $\Omega$ with the effective gradient interval of the SG function, thereby enhancing robustness while preserving training effectiveness.

### Key Designs

1. **Theoretical Analysis of Robustness Error**

   Upper bound of the error induced by adversarial perturbation:
   $|\mathcal{L}(x+\delta) - \mathcal{L}(x)| \leq |\delta \odot \nabla_x \mathcal{L}(x)|_1 + g(\delta, x)$

   Rewriting input gradient optimization as internal network gradient optimization using LIF dynamics and BPTT:
   $\min \sum_t \left|\frac{1}{L} \sum_{l=1}^{L} (P_1 \cdot P_2 \cdot P_3) \frac{\partial \mathcal{L}}{O_l^T}\right|_1$

   Where three key terms are:
   - $P_1$: Perturbation term (related to the leakage factor $\rightarrow$ the basis for the FEEL method)
   - $P_2$: Weight term ($\rightarrow$ the basis for Lipschitz regularization)
   - $P_3 = \prod_v \frac{\partial O_v^t}{\partial U_v^t}$: **SG term (the factor focused on in this work that was previously overlooked)**

   **Design Motivation**: Lowering the magnitude of the SG term $P_3$ can directly reduce the upper bound of the robustness error. The SG magnitude is determined by the proportion of the membrane potential that falls within the effective gradient interval of the SG.

2. **Theoretical Modeling of Membrane Potential Distribution**

   **Theorem 1**: In iterative LIF models with tdBN, the membrane potential follows a Gaussian distribution:
   $\overline{U}_c^l(t) \sim \mathcal{N}(\beta_c D(\tau, t) - S(t), (\lambda_c \alpha V_{th})^2 D(\tau^2, t))$

   Where $D(\tau, t) = \sum_{i=1}^{t} \tau^{t-i}$ is the cumulative decay function.

   That is, the mean $\mu$ and standard deviation $\sigma$ of the MPD are jointly determined by the tdBN parameters ($\beta_c$, $\lambda_c$), LIF parameters ($\tau$, $V_{th}$), and timestep $t$ $\rightarrow$ the MPD can be optimized by learning network parameters.

3. **Derivation and Regularization of MPD-SG Overlap Area**

   Assuming the effective gradient interval of the SG function (triangular) is $[-\gamma, \gamma]$ and the MPD is $\mathcal{N}(\mu, \sigma^2)$, the overlap area is:
   $$\Omega = \Phi\left(\frac{\mu + \gamma}{\sigma}\right) - \Phi\left(\frac{\mu - \gamma}{\sigma}\right)$$

   Where $\Phi$ is the cumulative distribution function (CDF) of the standard normal distribution. The final MPD-SGR regularization loss is:
   $$\mathcal{L}_{MPD-SGR}^b = \frac{1}{LCT} \sum_{l,c,t} \left[\Phi\left(\frac{\mu_c^l(t) + \gamma}{\sigma_c^l(t)}\right) - \Phi\left(\frac{\mu_c^l(t) - \gamma}{\sigma_c^l(t)}\right)\right]$$

   Summing the overlap area for every layer $l$, channel $c$, and timestep $t$ (excluding the final linear output layer).

   **Design Motivation**: A smaller $\Omega$ $\rightarrow$ lower SG magnitude $\rightarrow$ decreased sensitivity of the model to perturbations. However, an excessively small $\Omega$ would block gradient propagation; the coefficient $\eta$ balances robustness and training performance.

### Loss & Training

$$\mathcal{L}^b = \mathcal{L}_{task}^b + \eta \mathcal{L}_{MPD-SGR}^b$$

- $\mathcal{L}_{task}$: Standard classification cross-entropy loss
- $\eta$ controls the regularization strength
- PGD adversarial examples ($k=2, \varepsilon=2/255$) are used during Adversarial Training (AT)
- Attack settings: $\varepsilon=8/255$, PGD/BIM iteration steps $k=7$, step size $\alpha=0.01$

## Key Experimental Results

### Main Results: Comparison with SOTA Methods (VGG11, T=8)

**Vanilla Training**:

| Method | Clean | FGSM | PGD | BIM |
|------|:---:|:---:|:---:|:---:|
| REG | 92.49 | 25.18 | 0.88 | 0.60 |
| StoG | 91.64 | 16.22 | 0.28 | 0.12 |
| DLIF | 92.01 | 11.52 | 0.08 | 0.06 |
| FEEL | 90.08 | 29.17 | 6.67 | 5.99 |
| SR | 91.04 | 31.72 | 8.55 | 7.28 |
| **MPD-SGR** | **91.63** | **47.59** | **20.55** | **16.85** |
| **Gain** | -0.86 | +15.87 | +12.00 | +9.57 |

**Adversarial Training**:

| Method | Clean | FGSM | PGD | BIM |
|------|:---:|:---:|:---:|:---:|
| RAT | 91.41 | 45.00 | 22.95 | 20.80 |
| FEEL | 89.00 | 45.62 | 29.52 | 28.39 |
| SR | 88.26 | 44.28 | 28.63 | 27.03 |
| **MPD-SGR** | **90.69** | **59.27** | **33.38** | **32.61** |
| **Gain** | -0.72 | +13.52 | +3.86 | +4.22 |

Similar substantial leads are achieved on CIFAR-100 (Vanilla: FGSM +18.35%; AT: FGSM +16.35%).

### Ablation Study: Different SG Functions (CIFAR-10, VGG11)

| Model + SG Function | Method | Clean | FGSM | PGD | BIM |
|------------|------|:---:|:---:|:---:|:---:|
| VGG11+Rectangular | REG | 91.85 | 24.00 | 3.13 | 2.33 |
| VGG11+Rectangular | **Ours** | 91.23 | **43.28** | **15.82** | **14.20** |
| VGG11+Sigmoid | REG | 92.15 | 19.42 | 0.24 | 0.15 |
| VGG11+Sigmoid | **Ours** | 89.38 | **37.25** | **9.26** | **7.23** |
| VGG11+Superspike | REG | 86.82 | 21.39 | 0.82 | 0.50 |
| VGG11+Superspike | **Ours** | 84.45 | **43.42** | **6.32** | **4.50** |

$\rightarrow$ MPD-SGR consistently improves robustness across all three SG functions, demonstrating the generalization capability of the method.

### Different Coding Schemes (Tiny-ImageNet, VGG16)

| Coding Scheme | Method | Clean | FGSM | PGD |
|---------|------|:---:|:---:|:---:|
| Direct (DIR) | Baseline | 57.90 | 2.04 | 0.01 |
| Direct (DIR) | +Ours | 54.78 | **14.33** | **5.72** |
| Poisson (POS) | Baseline | 48.14 | 6.79 | 2.68 |
| Poisson (POS) | +Ours | 47.83 | **20.42** | **8.21** |
| RSC | Baseline | 47.47 | 22.63 | 13.75 |
| RSC | +Ours | 46.98 | **35.06** | **17.60** |

$\rightarrow$ MPD-SGR is compatible with different spike coding schemes and can be used in combination.

### Key Findings

1. **Most significant performance under Vanilla Training**: Without AT, the baseline SNN yields almost 0% accuracy under PGD attacks, while MPD-SGR improves it to ~20%.
2. **Minimal clean accuracy degradation**: -0.86% (CIFAR-10) $\rightarrow$ excellent robustness-accuracy trade-off.
3. **Although the SR method also improves robustness, its clean accuracy plummets** (66.76% on CIFAR-100), whereas MPD-SGR maintains 70.42% $\rightarrow$ demonstrating stronger practicality.
4. **Equally effective under black-box attacks** $\rightarrow$ showing that the robustness stems from intrinsic properties rather than gradient obfuscation.
5. **Applicable under non-gradient attacks (random noise)**: Under Gaussian noise, the accuracy on CIFAR-100 is 53.01% vs 32.63% for FEEL.
6. **Effective across architectures**: Consistent improvements are observed on VGG11 and WRN16.

## Highlights & Insights

1. **Solid theoretical contributions**:
    - Formulated the connection between SG magnitude and robustness error.
    - Proved the Gaussian distribution form of MPD under LIF with tdBN (Theorem 1).
    - Derived the analytical expression for the MPD-SG overlap area $\Omega$.
2. **Elegant regularization design**: The analytical formulation of the overlap area based on CDF can be directly backpropagated without additional approximations.
3. **Strong generalization capability**: Effective across different SG functions, coding schemes, architectures, and attack types.
4. **Orthogonal to existing methods**: Can be integrated alongside adversarial training and coding schemes.
5. **Bridging SG optimization and robustness**: Previously, SG-MPD alignment was addressed only to improve training performance; this work is the first to leverage it for robustness enhancement.

## Limitations & Future Work

- The parameter $\eta$ requires tuning (although analyzed in the appendix, an adaptive mechanism is lacking).
- The theoretical analysis is based on a triangular SG function. While experiments validate its efficacy with other SG functions, the theoretical expansion to arbitrary SG functions remains to be completed.
- Validated only on image classification tasks; applicability to event-driven tasks (such as neuromorphic vision) remains unexplored.
- The timestep $T$ is fixed ($T=4$ or $T=8$); performance and efficiency on longer temporal sequences need further investigation.
- The derivation of Theorem 1 assumes the use of tdBN $\rightarrow$ SNNs without tdBN will require a re-derivation of the MPD.

## Related Work & Insights

- **Difference from InfLoR-SNN / RecDis-SNN**: Those methods constrain the MPD to improve training (by ensuring a proper proportion of membrane potentials have gradients), whereas ours constrains the MPD to enhance robustness (by reducing the proportion of membrane potentials with gradients) $\rightarrow$ the objectives are opposite yet complementary.
- **Difference from FEEL**: FEEL suppresses noise within different frequency ranges via frequency coding and attention mechanisms $\rightarrow$ acting on the input layer, whereas MPD-SGR acts on gradient propagation across all internal layers of the network.
- **Significance to the SNN community**: Reveals that the SG affects not only training performance but also robustness $\rightarrow$ providing a new dimension of consideration for SG design.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (A brand-new perspective of employing SG-MPD interaction for robustness, supported by rigorous theoretical derivation.)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (3 datasets $\times$ 2 architectures $\times$ 4 attacks $\times$ 3 SG functions $\times$ 3 encodings $\times$ AT/non-AT)
- Writing Quality: ⭐⭐⭐⭐ (A clear logical progression from theory to methodology, and then to experiments.)
- Value: ⭐⭐⭐⭐⭐ (Provides a theoretically grounded, generic defense strategy for SNN robustness.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Robust Spiking Neural Networks Against Adversarial Attacks](../../ICLR2026/ai_safety/robust_spiking_neural_networks_against_adversarial_attacks.md)
- [\[CVPR 2026\] Towards Reliable Evaluation of Adversarial Robustness for Spiking Neural Networks](../../CVPR2026/ai_safety/towards_reliable_evaluation_of_adversarial_robustness_for_spiking_neural_network.md)
- [\[ICLR 2026\] Time Is All It Takes: Spike-Retiming Attacks on Event-Driven Spiking Neural Networks](../../ICLR2026/ai_safety/time_is_all_it_takes_spike-retiming_attacks_on_event-driven_spiking_neural_netwo.md)
- [\[ICLR 2026\] Robustify Spiking Neural Networks via Dominant Singular Deflation under Heterogeneous Training Vulnerability](../../ICLR2026/ai_safety/robustify_spiking_neural_networks_via_dominant_singular_deflation_under_heteroge.md)
- [\[ICLR 2026\] A Unified Total Variation Framework for Membrane Potential Perturbation Dynamic](../../ICLR2026/ai_safety/a_unified_total_variation_framework_for_membrane_potential_perturbation_dynamic.md)

</div>

<!-- RELATED:END -->

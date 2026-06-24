---
title: >-
  [Paper Note] Resilience of Entropy Model in Distributed Neural Networks
description: >-
  [ECCV2024][AI Safety][distributed DNN] This paper presents the first systematic study on the robustness of entropy coding models in distributed DNNs under both intentional interference (adversarial attacks) and unintentional interference (weather changes, motion blur, etc.). It reveals that the compression features learned by the entropy model are distinct from classification features, and proposes an object-aware total variation denoising defense method. This approach reduce…
tags:
  - "ECCV2024"
  - "AI Safety"
  - "distributed DNN"
  - "entropy coding"
  - "adversarial attack"
  - "communication efficiency"
  - "total variation denoising"
date: 2026-05-08
content_hash: 59043297a7cf04f9
---

# Resilience of Entropy Model in Distributed Neural Networks

**Conference**: ECCV2024  
**arXiv**: [2403.00942](https://arxiv.org/abs/2403.00942)  
**Code**: [EntropyR](https://github.com/Restuccia-Group/EntropyR)  
**Area**: AI Security  
**Keywords**: distributed DNN, entropy coding, adversarial attack, communication efficiency, total variation denoising

## TL;DR

This paper presents the first systematic study on the robustness of entropy coding models in distributed DNNs under both intentional interference (adversarial attacks) and unintentional interference (weather changes, motion blur, etc.). It reveals that the compression features learned by the entropy model are distinct from classification features, and proposes an object-aware total variation denoising defense method. This approach reduces post-attack transmission overhead to below clean data levels, with an accuracy drop of only around 2%.

## Background & Motivation

Distributed Deep Neural Networks (Distributed DNNs) split a large model into a head network deployed on mobile devices and a tail network on a server, reducing communication overhead by transmitting only the intermediate compressed representations. Recently, entropy coding has been introduced to further compress these representations. The core idea is to jointly train the DNN with an auxiliary entropy model, and during inference, use the prior distribution output by the entropy model as side information to adaptively encode the quantized latent representations into variable-length bitstreams.

However, existing work has never considered the robustness of the entropy model itself. While DNN vulnerability to distribution shifts and adversarial perturbations has been widely studied, entropy models are typically trained on clean data. Small perturbations in the input space can cause entropy estimations to surge dramatically, potentially pushing the encoded bit rate past the transmission bandwidth limit and, in the worst-case, increasing the transmitted data volume by up to 2 times. This not only impairs the end-to-end latency of individual users but can also saturate shared bandwidth, threatening other users in the system.

## Core Problem

1. **How robust is the entropy model to unintentional interference (common corruption)?** How do different types of image corruptions (noise, blur, weather, digital degradation) affect the encoded data volume?
2. **How robust is the entropy model to intentional interference (adversarial attacks)?** What are the fundamental differences between adversarial attacks targeted at entropy (i.e., bit rate, PGD-E) and those targeted at accuracy (PGD-Acc)?
3. **How to design defense mechanisms to protect the entropy model?** Can post-attack transmission overhead be significantly reduced without compromising classification accuracy?

## Method

### Threat Model

The paper models interference as an additive perturbation $\delta$ to the input, formalizing two attack objectives:

- **PGD-Acc**: The classic PGD attack using cross-entropy as the loss function, aiming to degrade classification accuracy.
- **PGD-E**: A rate-targeted attack using the entropy coding rate loss $-\log_2 P_Z(z)$ as the objective, aiming to maximize the entropy of the latent representation to increase the encoded data volume.

Both adopt an $l_\infty$ constraint and are solved via projected gradient descent (PGD).

### Key Findings: Decoupling of Compression and Classification Features

**Frequency Domain Analysis**: By comparing the total variation map of images and the bit rate map of the entropy model, the authors find that the two are highly correlated—the entropy model is highly sensitive to high-frequency features. Introducing high-frequency noise (e.g., shot noise) significantly increases data volume (+65%), while removing high-frequency information (e.g., defocus blur) conversely reduces data volume (-53%). This occurs because the head network of a distributed DNN is split at shallow layers, which primarily capture low-level features (e.g., high-frequency details like edges and textures).

**Spatial Domain Analysis**: PGD-E primarily increases the bit rate in background regions, whereas PGD-Acc largely targets foreground object regions. This indicates that the two types of attacks target different feature sets in the input space and have minimal mutual influence.

### Defense Method: Object-Aware Total Variation Denoising

Based on these findings, the paper proposes an object-aware total variation denoising method:

1. **Total Variation Denoising**: Solves the optimization problem $\min_x \frac{1}{2}\|x - x'\|_2^2 + \lambda \cdot TV(x)$ to remove high-frequency noise while preserving the main image content, solved iteratively using subgradient descent.
2. **Object-Aware Mask**: Since direct global denoising can impair classification information of foreground objects, the output of the entropy model $P_Z(z)$ is leveraged as a soft mask. Object regions with higher bit rates correspond to smaller $P_Z(z)$ values, thereby naturally avoiding over-smoothing of those regions.
3. **Final Iteration Formula**: $x^{i+1} = x^i - \alpha \cdot m \cdot ((x^i - x') + \lambda \cdot g(x^i))$, where $m$ is the mask and $g(x^i)$ is the image gradient.

This method requires no model retraining and can be combined with other methods like adversarial training as an independent pre-processing module.

## Key Experimental Results

**Experimental Setup**: Evaluated on ImageNet / ImageNet-C using 3 DNN architectures (ResNet-50, ResNet-101, RegNetY-6.4GF), 2 entropy models (Factorized Prior / FP, Mean Scale Hyper Prior / MSHP), and 4 rate-distortion trade-offs $\beta$.

### Unintentional Interference

| Corruption Type | Data Volume Change | Explanation |
|---|---|---|
| Shot noise | +65.31% | Introduces high-frequency noise |
| Snow | -4.42% | No specific pattern in the frequency domain |
| Defocus blur | -53.31% | Removes high-frequency information |
| Contrast | -67.45% | Removes high-frequency information |

### Intentional Interference ($\epsilon=8/255$, MSHP)

| Attack Method | Data Volume Increase | Accuracy Drop |
|---|---|---|
| PGD-Acc | +10.19% | -57.10% |
| PGD-E | +46.82% | -6.62% |

### Defense Effectiveness (MSHP, Clean Data 9.62 KB)

| Perturbation Budget | Post-Attack Data Volume | Post-Defense Data Volume | Post-Defense Accuracy Loss |
|---|---|---|---|
| $\epsilon=2/255$ | 12.35 KB | 8.76 KB (< Clean) | -2.44% |
| $\epsilon=4/255$ | 14.11 KB | 9.55 KB (≈ Clean) | -1.18% |
| $\epsilon=8/255$ | 16.15 KB | 11.02 KB | Accuracy increased by +1.60% |
| $\epsilon=16/255$ | 18.79 KB (+95%) | 13.44 KB | Accuracy increased by +7.74% |

### Adaptive Attacks

The defense remains effective against two adaptive strategies, namely low-frequency attacks and regional attacks: reducing data volume by ~67% with an accuracy drop of only ~1.3%.

## Highlights & Insights

1. **Revealing the Security Blind Spot of Entropy Models for the First Time**: The adversarial robustness of the entropy coding module in distributed DNNs has not been scrutinized before; this paper fills this critical gap.
2. **Profound Insights into Feature Decoupling**: The paper demonstrates that compression features and classification features are decoupled across both frequency and spatial domains, offering valuable theoretical insights.
3. **Simple Yet Effective Defense Design**: The proposed scheme combining total variation denoising and entropy soft mask requires no model retraining, incurs low computational overhead, and achieves remarkable results—even lowering the data volume below clean levels under small perturbations.
4. **Robustness to Adaptive Attacks**: The defense remains robust even under white-box settings against specifically designed adaptive attacks.

## Limitations & Future Work

1. **Evaluation Limited to Classification**: The performance has not been validated on other downstream tasks such as object detection or semantic segmentation.
2. **Mask Dependency on the Entropy Model**: The defense relies on $P_Z(z)$ as a soft mask, which might be exploited by more sophisticated attackers.
3. **Elevated Data Volume Under Large Perturbations**: At $\epsilon=16/255$, the post-defense data volume is still 13.44 KB, which is roughly 40% higher than the 9.62 KB of clean data.
4. **Absence of End-to-End Jointly Optimized Defense**: The current method serves as a pre-processing step and is not jointly optimized with the model training process.
5. **Head Network Fixed at Shallow Layers**: Different splitting points correspond to different feature hierarchies, which might affect the frequency characteristics of the compression features.

## Related Work & Insights

| Research Direction | Representative Work | Our Difference |
|---|---|---|
| Dynamic DNN Efficiency Attacks | Hong et al., Haque et al. | Prior works target computational efficiency, whereas ours targets communication efficiency for the first time. |
| Robustness of Density Estimation | Arvinte et al. | They focus on maximizing $P_Z(z)$, while ours focuses on minimizing $P_Z(z)$ (increasing the bit rate). |
| Distributed DNN Compression | Entropic Student | Proposed an entropy coding integration scheme but did not consider its robustness. |
| Traditional Adversarial Defense | Adversarial training, input purification | Our method can be combined with these methods as an independent module. |

## Insights & Connections

- **Security-Perspective Insights**: In edge-computing scenarios, attackers do not necessarily have to degrade accuracy—simply increasing transmission overhead can lead to a bandwidth-saturating DoS effect, which is an overlooked yet highly threatening attack surface.
- **Decoupling of Compression and Recognition**: Shallow features primarily encode high-frequency information (textures, edges), whereas deep features encode semantic information. This observation can guide partition point selection and robustness design in distributed systems.
- **Extensibility to Other Compression Scenarios**: Any neural compression system using a learned entropy model (image compression, video compression) may face similar potential attack risks.

## Rating

- Novelty: ⭐⭐⭐⭐ — Presents the first systematic study on entropy model robustness with a novel problem definition.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive ablation coverage across multiple architectures, entropy models, and parameters, with adaptive attacks thoroughly considered.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with intuitive and powerful visualization analyses (bit rate maps, TV maps, and spatial contrast maps).
- Value: ⭐⭐⭐⭐ — Uncovers an overlooked security issue while delivering a practical and combinable defense scheme.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Singular Bayesian Neural Networks](../../ICML2026/ai_safety/singular_bayesian_neural_networks.md)
- [\[ECCV 2024\] Unveiling Privacy Risks in Stochastic Neural Networks Training: Effective Image Reconstruction from Gradients](unveiling_privacy_risks_in_stochastic_neural_networks_training_effective_image_r.md)
- [\[ICLR 2026\] Robust Spiking Neural Networks Against Adversarial Attacks](../../ICLR2026/ai_safety/robust_spiking_neural_networks_against_adversarial_attacks.md)
- [\[ICML 2026\] Frequency Matching in Spiking Neural Networks for mmWave Sensing](../../ICML2026/ai_safety/frequency_matching_in_spiking_neural_networks_for_mmwave_sensing.md)
- [\[CVPR 2026\] Towards Reliable Evaluation of Adversarial Robustness for Spiking Neural Networks](../../CVPR2026/ai_safety/towards_reliable_evaluation_of_adversarial_robustness_for_spiking_neural_network.md)

</div>

<!-- RELATED:END -->

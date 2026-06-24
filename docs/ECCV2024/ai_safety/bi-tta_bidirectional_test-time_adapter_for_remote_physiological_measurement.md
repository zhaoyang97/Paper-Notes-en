---
title: >-
  [Paper Note] Bi-TTA: Bidirectional Test-Time Adapter for Remote Physiological Measurement
description: >-
  [ECCV2024][AI Safety][rPPG] This paper proposes the Bi-TTA framework, which introduces Test-Time Adaptation to remote photoplethysmography (rPPG) tasks for the first time. By leveraging a spatiotemporal consistency self-supervised prior and a prospective-retrospective bidirectional adaptation strategy, the proposed method achieves model domain adaptation at test-time using only unlabeled single-instance data during inference.
tags:
  - "ECCV2024"
  - "AI Safety"
  - "rPPG"
  - "Test-Time Adaptation"
  - "Remote Physiological Measurement"
  - "Self-Supervised Prior"
  - "Domain Adaptation"
date: 2026-05-08
content_hash: 7dc7acaec3deac0f
---

# Bi-TTA: Bidirectional Test-Time Adapter for Remote Physiological Measurement

**Conference**: ECCV2024  
**arXiv**: [2409.17316](https://arxiv.org/abs/2409.17316)  
**Code**: [bi-tta.github.io](https://bi-tta.github.io)  
**Area**: Model Compression  
**Keywords**: rPPG, Test-Time Adaptation, Remote Physiological Measurement, Self-Supervised Prior, Domain Adaptation

## TL;DR

This paper proposes the Bi-TTA framework, which introduces Test-Time Adaptation to remote photoplethysmography (rPPG) tasks for the first time. By leveraging a spatiotemporal consistency self-supervised prior and a prospective-retrospective bidirectional adaptation strategy, the proposed method achieves model domain adaptation at test-time using only unlabeled single-instance data during inference.

## Background & Motivation

Remote photoplethysmography (rPPG) extracts blood volume pulse (BVP) signals from facial videos using regular cameras, enabling non-contact measurement of physiological indicators such as heart rate (HR), heart rate variability (HRV), and respiratory rate (RF). Compared to traditional electrocardiography (ECG) devices and fingertip sensors, rPPG requires no wearable equipment, incurring lower costs and offering greater convenience.

However, rPPG signals are extremely weak—the skin color variations caused by heartbeats are nearly imperceptible in videos, making them highly susceptible to environmental interference such as changes in lighting, head motion, and variations in camera parameters. Although existing deep learning methods perform excellently under controlled laboratory conditions, their performance drops significantly when deployed in unseen domains. To adapt to new domains, Domain Adaptation requires labeled target domain data, whereas Domain Generalization does not optimize for a specific target domain, both having limitations. Under privacy constraints (i.e., no access to source data or target labels), Test-Time Adaptation (TTA) emerges as the most appropriate paradigm, which adaptively adjusts the model during inference using unlabeled target data.

## Core Problem

Directly applying TTA to rPPG faces two critical challenges:

1. **Absence of Supervision Signals**: Existing TTA methods mostly target classification tasks, relying on entropy minimization or pseudo-labeling. They are inapplicable to regression tasks like rPPG, lacking effective self-supervised signals.
2. **Instability of Single-Instance Learning**: In practical deployment, the model processes target domain videos frame-by-frame, fine-tuning on only a single sample at a time. The bias and noise introduced by single-instance learning make it difficult for the model to distinguish between domain-dependent and domain-independent features, rendering it vulnerable to forgetting existing knowledge (catastrophic forgetting) or overfitting to noisy features.

## Method

### Overall Architecture

Bi-TTA comprises designs across two orthogonal dimensions: (1) expert-knowledge-based self-supervised priors that provide adaptation gradients; (2) a prospective-retrospective bidirectional adaptation strategy that ensures the effectiveness and stability of the adaptation process.

### Spatiotemporal Map (STMap) Construction

The input is a sequence of facial video frames. First, facial alignment and cropping are performed to extract local color signals from different facial sub-regions, which are concatenated into a 2D STMap $\boldsymbol{x} \in \mathbb{R}^{W \times H}$, where $W$ is the sliding window temporal length (256 frames) and $H$ is the spatial dimension (25 facial regions). The STMap is resized to $256 \times 64 \times 3$ and fed into a ResNet-18 network.

### Temporal Consistency Loss (TCL)

Since BVP signals change smoothly and continuously over short periods, a random temporal shift $\delta_T$ (uniformly sampled from $(0, 59]$) is applied to the original sample. The heart rates of the original and shifted samples are predicted separately, with their difference constrained by an L1 regularization to not exceed a tolerance threshold $\xi_T$:

$$L_t = \sum_i^W \max(0, \|\text{HR}(\boldsymbol{x}_t) - \text{HR}(\boldsymbol{x}_{t-\delta_T})\|_1 - \xi_T)$$

### Spatial Consistency Loss (SCL)

Skin color variations caused by heartbeats exhibit spatial consistency across different facial regions. SCL calculates the L1 difference between adjacent spatial positions on multi-scale latent feature maps from the four residual blocks of ResNet-18:

$$L_s = \sum_i^4 \sum_j^{W_i - \delta_S} \|F_{i,j} - F_{i,j+\delta_S}\|_1$$

The multi-scale strategy ensures the supervision signals provide comprehensive coverage from shallow textures to deep semantics. The total self-supervised loss is defined as $L_p = \lambda_s L_s + \lambda_t L_t$.

### Prospective Adaptation Module (PA)

Noise samples in single-instance learning can easily disrupt the model. Drawing inspiration from Sharpness-Aware Minimization (SAM), PA does not directly minimize the loss at the current parameter coordinates. Instead, it seeks a flat localized area where the worst-case (maximum) loss in the neighborhood is minimized:

$$L_p'(\boldsymbol{w}) = \max_{\|\boldsymbol{\epsilon}\|_2 \leq \rho} L_p(\boldsymbol{w} + \boldsymbol{\epsilon})$$

The optimal perturbation direction $\hat{\epsilon}$ is approximated via first-order Taylor expansion, and the gradient $\boldsymbol{g}_t^{PA}$ is computed at the perturbed parameter point. This makes the model more robust to single-instance noise and filters out domain-irrelevant information.

### Retrospective Stabilization Module (RS)

Although PA can withstand single-step noise, the model may still capture harmful noise features during long-term adaptation, leading to performance degradation. RS introduces a "trend gradient" $\boldsymbol{g}^*$, which accumulates past gradients weighted by self-supervised loss. At each step, the projection of the current gradient onto the trend gradient is computed:

- If the projection direction is **opposite** to the trend gradient (detecting oscillation), suggesting the current update might harm generalization performance, RS activates the retrospective mechanism, scaling the projected gradient by a retrospective coefficient $k$;
- If the direction is **consistent**, the current gradient is mixed with the trend gradient according to an annealing coefficient $\lambda_t^{RS}$.

The annealing coefficient $\lambda_t^{RS}$ approaches 1 from 0 as the number of samples increases, ensuring that the trend gradient dominates the optimization direction after accumulating sufficient samples ($\Omega = 4000$).

## Key Experimental Results

A large-scale TTA evaluation benchmark is established across five datasets (VIPL, V4V, PURE, UBFC, and BUAA), evaluated with MAE↓, RMSE↓, and Pearson r↑:

| Method | VIPL MAE↓ | PURE MAE↓ | UBFC MAE↓ | BUAA MAE↓ |
|------|-----------|-----------|-----------|-----------|
| NEST (DG) | 7.86 | 6.71 | 4.67 | 2.88 |
| Tent (TTA) | 8.09 | 6.86 | 4.57 | 2.37 |
| EATA (TTA) | 7.69 | 6.13 | 4.25 | 1.89 |
| SHOT (TTA) | 7.75 | 5.81 | 4.05 | 1.87 |
| ConPhys | 7.43 | 6.09 | 3.92 | 1.75 |
| **Bi-TTA (Prior Only)** | 7.31 | 5.56 | 3.64 | 1.68 |
| **Bi-TTA (Full)** | **7.09** | **5.02** | **3.53** | **1.49** |

Key findings from the ablation study:

- Using only the prior (without the bidirectional strategies) already outperforms all baselines, verifying the effectiveness of TCL+SCL.
- Removing PA: PURE MAE increases from 5.02 to 5.39; removing RS: increases from 5.02 to 5.24.
- The synergy of PA and RS yields better results than using either individually, balancing both convergence speed and long-term stability.
- After approximately 5000-6000 samples, the method using only the prior exhibits obvious performance degradation, whereas Bi-TTA maintains stability.

## Highlights & Insights

1. **First to introduce TTA to rPPG**: Establishes a complete TTA evaluation protocol and a large-scale benchmark, filling the gap in this interdisciplinary field.
2. **Ingenious self-supervised design**: TCL and SCL originate from the inherent spatiotemporal consistency of physiological signals, providing effective supervision without labels, with a convergence speed that is even faster than fully supervised methods.
3. **Highly complementary bidirectional adaptation strategies**: PA ensures single-step robustness (filtering out noise), while RS ensures long-term stability (preventing forgetting/overfitting), making them orthogonally complementary.
4. **No additional requirements for pre-trained models**: Does not demand specific network architectures or auxiliary tasks during the pre-training phase, offering strong adaptability.

## Limitations & Future Work

1. **Only ResNet-18 backbone is validated**: Adaptation effectiveness with more modern architectures, such as Transformers, remains unexplored.
2. **Relatively large number of hyperparameters**: A total of six hyperparameters ($\lambda_s, \lambda_t, \xi_T, \rho, k, \Omega$) need to be tuned. While ablation studies were conducted, cross-dataset generalization has not been fully discussed.
3. **Trend gradient requires an accumulation phase**: RS has limited effectiveness during the first $\Omega=4000$ samples, presenting a cold-start problem.
4. **Computational overhead**: SAM requires two forward and backward passes, which increases inference latency and may act as a bottleneck for real-time applications.
5. **Evaluation limited to heart rate estimation**: The adaptation performance on other physiological metrics, such as HRV and respiratory rate, has not been validated.

## Related Work & Insights

- **vs. DG Methods (NEST/Coral/VREx)**: DG does not target a specific target domain and requires access to all source data. In contrast, Bi-TTA requires no source data and performs targeted adaptation, comprehensively outperforming DG methods.
- **vs. General TTA (Tent/SAR/EATA/SHOT)**: These methods rely on entropy or pseudo-labels of classification tasks and yield limited performance when directly applied to regression tasks. Bi-TTA's domain priors provide more suitable supervision.
- **vs. ConPhys**: While ConPhys also utilizes a spatiotemporal consistency prior, Bi-TTA calculates SCL on multi-scale latent feature maps (rather than only the output layer) and incorporates a bidirectional adaptation strategy, reducing the MAE on VIPL from 7.43 to 7.09.
- **vs. AdaODM**: AdaODM performs slightly better on V4V (tied with an MAE of 9.1 vs. 9.1), but is outperformed by Bi-TTA on all other datasets.

## Inspirations & Connections

- **Design Paradigm for Self-Supervised Priors**: Constructing loss functions based on the intrinsic properties of a task (such as temporal smoothness and spatial consistency) is a approach that can be extended to other physiological signals or time-series regression TTA scenarios.
- **Applying SAM to TTA**: Migrating Sharpness-Aware Minimization from the training phase to test-time adaptation is an inspiring combination.
- **Trend Gradient and Oscillation Detection**: The gradient direction consistency detection mechanism in the RS module is also valuable for other online learning or continual learning scenarios.
- This framework can be combined with parameter-efficient fine-tuning methods like LoRA to reduce the computational overhead of TTA.

## Rating
- Novelty: ⭐⭐⭐⭐ (First to introduce TTA to rPPG, with a highly novel bidirectional strategy design)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive evaluation across five datasets + detailed ablation studies)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure and rich visualizations)
- Value: ⭐⭐⭐⭐ (Establishes a new benchmark, and the method is highly generalizable)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] FRET: Feature Redundancy Elimination for Test Time Adaptation](../../ICCV2025/ai_safety/fret_feature_redundancy_elimination_for_test_time_adaptation.md)
- [\[CVPR 2025\] OODD: Test-time Out-of-Distribution Detection with Dynamic Dictionary](../../CVPR2025/ai_safety/oodd_test-time_out-of-distribution_detection_with_dynamic_dictionary.md)
- [\[ECCV 2024\] Preventing Catastrophic Overfitting in Fast Adversarial Training: A Bi-level Optimization Perspective](preventing_catastrophic_overfitting_in_fast_adversarial_training_a_bi-level_opti.md)
- [\[CVPR 2026\] TTP: Test-Time Padding for Adversarial Detection and Robust Adaptation on Vision-Language Models](../../CVPR2026/ai_safety/ttp_test-time_padding_for_adversarial_detection_and_robust_adaptation_on_vision-.md)
- [\[NeurIPS 2025\] Redundancy-Aware Test-Time Graph Out-of-Distribution Detection](../../NeurIPS2025/ai_safety/redundancy-aware_test-time_graph_out-of-distribution_detection.md)

</div>

<!-- RELATED:END -->

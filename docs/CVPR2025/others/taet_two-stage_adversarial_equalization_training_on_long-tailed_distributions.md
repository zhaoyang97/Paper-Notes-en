---
title: >-
  [Paper Note] TAET: Two-Stage Adversarial Equalization Training on Long-Tailed Distributions
description: >-
  [CVPR 2025][Long-Tailed Distributions] This paper proposes TAET, a two-stage adversarial equalization training framework: it first stabilizes early training using cross-entropy loss, and then balances the performance across all classes using Hierarchical Adversarial Robust Learning (HARL) combined with three losses (BCL/HDL/RCEL). It also introduces a Balanced Robustness evaluation metric to address the insufficient robustness of tail classes in adversarial training under lon…
tags:
  - "CVPR 2025"
  - "Long-Tailed Distributions"
  - "Adversarial Training"
  - "Equalization Loss"
  - "Robust Overfitting"
  - "Balanced Robustness Metrics"
date: 2026-05-08
content_hash: 85739c0866ded3e0
---

# TAET: Two-Stage Adversarial Equalization Training on Long-Tailed Distributions

**Conference**: CVPR 2025  
**arXiv**: [2503.01924](https://arxiv.org/abs/2503.01924)  
**Code**: [GitHub](https://github.com/BuhuiOK/TAET-Two-Stage-Adversarial-Equalization-Training-on-Long-Tailed-Distributions)  
**Area**: Other  
**Keywords**: Long-Tailed Distributions, Adversarial Training, Equalization Loss, Robust Overfitting, Balanced Robustness Metrics

## TL;DR

This paper proposes TAET, a two-stage adversarial equalization training framework: it first stabilizes early training using cross-entropy loss, and then balances the performance across all classes using Hierarchical Adversarial Robust Learning (HARL) combined with three losses (BCL/HDL/RCEL). It also introduces a Balanced Robustness evaluation metric to address the insufficient robustness of tail classes in adversarial training under long-tailed distributions.

## Background & Motivation

Adversarial training performs well on balanced datasets (CIFAR-10, ImageNet), but real-world data often exhibits a long-tailed distribution—where a few majority classes have abundant samples while numerous minority classes are sample-scarce. Existing long-tailed adversarial training methods (such as AT-BSL) exhibit two key issues:

1. **Insufficient improvement of BSL on weak classes**: Balanced Softmax Loss only adjusts logits based on sample size, failing to identify truly weak classes (e.g., in CIFAR-10-LT, class 3 has many samples but performs poorly). The adversarial robustness of tail classes is far lower than that of head classes.

2. **Severe robust overfitting**: BSL causes the model to reach its peak robustness around epoch ~25, after which it gradually declines. While natural accuracy continues to rise, adversarial robustness decreases indeed, showing instability especially during the learning rate adjustment phase.

Moreover, research on long-tailed robustness has ignored **balanced accuracy**, a crucial metric in long-tailed recognition, leading to incomplete evaluations.

## Method

### Overall Architecture

Two-stage training: Stage 1 (Initial Stabilization) — Uses standard cross-entropy loss and PGD adversarial training to achieves fast convergence and stable accuracy; Stage 2 (HARL Equalization) — Switches to the hierarchical equalization loss $\mathcal{L}_{\text{HEL}} = \alpha \cdot \mathcal{L}_{\text{BCL}} + \beta \cdot \mathcal{L}_{\text{HDL}} + \gamma \cdot \mathcal{L}_{\text{RCEL}}$ to dynamically identify and enhance the performance of weak classes.

### Key Design 1: Three-Component Loss of Hierarchical Adversarial Robust Learning (HARL)

- **Function**: Comprehensively balances the performance of all classes in adversarial training.
- **Mechanism**: Three complementary losses: (a) **BCL (Balanced Cross-Class Loss)** — takes the average loss of all classes $\mathcal{L}_{\text{BCL}} = \frac{1}{S_c}\sum_{c=1}^C \mathcal{L}_c$ to prevent head classes from dominating the total loss; (b) **HDL (Hierarchical Deviation Loss)** — penalizes the deviation of each class's loss from the mean loss $\mathcal{L}_{\text{HDL}} = \frac{1}{S_c}\sum_{c=1}^C (\mathcal{L}_c - \bar{\mathcal{L}})^2$ to reduce the performance gap between classes; (c) **RCEL (Rare Class Emphasis Loss)** — assigns higher weights to rare (high-loss) classes $\mathcal{L}_{\text{RCEL}} = \sum_{c=1}^C (\frac{\mathcal{L}_c}{\sum_j \mathcal{L}_j})^2$.
- **Design Motivation**: Unlike BSL, which identifies tail classes solely based on sample counts, HARL dynamically identifies weak classes based on average loss during training—which is more accurate (e.g., addressing cases like class 3 where samples are abundant but performance is poor). The three losses promote equalization from different perspectives.

### Key Design 2: Two-Stage Training Strategy

- **Function**: Mitigates robust overfitting while optimizing both accuracy and robustness.
- **Mechanism**: Stage 1 uses cross-entropy loss to train on adversarial samples for a certain number of epochs, allowing the model to quickly converge to a stable accuracy baseline; Stage 2 switches to the HARL loss for equalized adversarial training.
- **Design Motivation**: Experiments reveal that cross-entropy loss provides stable gradient signals in early training and is less prone to robust overfitting than directly using BSL. The early CE $\to$ late HARL transition achieves both stability and equalization.

### Key Design 3: Balanced Robustness Evaluation Metric

- **Function**: Fairly evaluates the adversarial robustness of each class in long-tailed scenarios.
- **Mechanism**: Extends the concept of balanced accuracy to adversarial examples: $\text{BR} = \frac{1}{S_C}\sum_{i=1}^C \frac{TP_i^{x'}}{TP_i^{x'} + FN_i^{x'}}$, which computes the average class recall rate on adversarial examples.
- **Design Motivation**: Traditional overall robustness is dominated by head classes, masking the vulnerability of tail classes. Balanced Robustness ensures equal contribution from all classes, which is particularly important for safety-critical areas like healthcare.

### Loss & Training

Stage 1: $\mathcal{L}_{\text{CE}}$ (Standard Cross-Entropy)  
Stage 2: $\mathcal{L}_{\text{HEL}} = \alpha \cdot \mathcal{L}_{\text{BCL}} + \beta \cdot \mathcal{L}_{\text{HDL}} + \gamma \cdot \mathcal{L}_{\text{RCEL}}$

## Key Experimental Results

### Main Results: CIFAR-10-LT (IR=10) ResNet-18 Balanced Accuracy/Robustness

| Method | Clean BA↑ | PGD-20 BR↑ | PGD-100 BR↑ | AA BR↑ |
|------|----------|-----------|------------|-------|
| AT | 69.00 | 25.69 | 24.55 | 24.28 |
| AT-BSL | 72.74 | 26.86 | 25.62 | 25.26 |
| RoBal | 73.18 | 27.12 | 26.98 | 24.13 |
| REAT | 74.56 | 24.02 | 22.52 | 22.69 |
| **TAET** | **76.22** | **30.12** | **28.45** | **27.31** |

### Ablation Study: Contribution of Each Component

| Configuration | Effect |
|------|------|
| Only BCL | Baseline equalization improvement |
| BCL + HDL | Significant reduction in class performance gap |
| BCL + HDL + RCEL | Optimal equalization effect |
| Two-Stage vs. Direct HARL | Two-stage training significantly reduces robust overfitting |

### Key Findings

- TAET simultaneously outperforms all baselines in both balanced accuracy and balanced robustness.
- t-SNE visualization shows that TAET achieves significantly better feature separation for tail classes than AT, TRADES, and AT-BSL.
- Loss-based weak class identification is more accurate than sample-count-based identification—it can capture classes with large sample counts but poor performance.
- The cross-entropy pre-training stage effectively alleviates the robust overfitting phenomenon of BSL methods.
- Memory and computational efficiency are also superior to other two-stage methods like RoBal.

## Highlights & Insights

1. **Introduction of Balanced Robustness metric**: Fills the gap in long-tailed robustness-related evaluations, making the vulnerability of tail classes visible.
2. **Loss-based weak class identification**: More flexible and accurate than sample-count-based identification, enabling the discovery of "hidden weak classes".
3. **Simplicity of the two-stage strategy**: The transition from CE pre-training to HARL equalization is simple and effective, requiring no complex curriculum learning designs.

## Limitations & Future Work

- Currently evaluated mainly on CIFAR-10-LT; results on large-scale datasets (ImageNet-LT) need to be supplemented.
- The three hyperparameters $\alpha, \beta, \gamma$ of HARL require tuning.
- The possibility of combining with other adversarial training methods (such as TRADES) has not yet been explored.

## Related Work & Insights

- The idea of introducing balanced accuracy from long-tailed recognition into adversarial robustness evaluation can be extended to other safety-critical imbalanced scenarios.
- The two-stage training strategy serves as a valuable reference for other scenarios requiring goal transitions during training.

## Rating

⭐⭐⭐⭐ — The problem is highly practical and important (real-world data is indeed long-tailed), and the Balanced Robustness metric is a valuable contribution. The method is simple yet effective, with a clear design motivation for each component. The ablation studies are thorough.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Confusion-Aware Spectral Regularizer for Long-Tailed Recognition](../../CVPR2026/others/confusion-aware_spectral_regularizer_for_long-tailed_recognition.md)
- [\[ICML 2025\] FEDTAIL: Federated Long-Tailed Domain Generalization with Sharpness-Guided Gradient Matching](../../ICML2025/others/fedtail_federated_long-tailed_domain_generalization_with_sharpness-guided_gradie.md)
- [\[CVPR 2025\] Towards Million-Scale Adversarial Robustness Evaluation With Stronger Individual Attacks](towards_million-scale_adversarial_robustness_evaluation_with_stronger_individual.md)
- [\[CVPR 2025\] Effortless Active Labeling for Long-Term Test-Time Adaptation](effortless_active_labeling_for_long-term_test-time_adaptation.md)
- [\[CVPR 2025\] EVOS: Efficient Implicit Neural Training via EVOlutionary Selector](evos_efficient_implicit_neural_training_via_evolutionary_selector.md)

</div>

<!-- RELATED:END -->

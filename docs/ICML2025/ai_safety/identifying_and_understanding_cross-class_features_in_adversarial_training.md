---
title: >-
  [Paper Note] Identifying and Understanding Cross-Class Features in Adversarial Training
description: >-
  [ICML2025][AI Safety][Adversarial Training] From the perspective of class-level feature attribution, this work reveals how "cross-class features" in adversarial training (AT) are first learned and then forgotten, offering a unified explanation for both robust overfitting and the advantages of soft-label training.
tags:
  - "ICML2025"
  - "AI Safety"
  - "Adversarial Training"
  - "Cross-Class Features"
  - "Robust Overfitting"
  - "Knowledge Distillation"
  - "Feature Attribution"
date: 2026-05-08
content_hash: c80f51aa7d46f928
---

# Identifying and Understanding Cross-Class Features in Adversarial Training

**Conference**: ICML2025  
**arXiv**: [2506.05032](https://arxiv.org/abs/2506.05032)  
**Code**: [PKU-ML/Cross-Class-Features-AT](https://github.com/PKU-ML/Cross-Class-Features-AT)  
**Area**: Adversarial Training / AI Safety  
**Keywords**: Adversarial Training, Cross-Class Features, Robust Overfitting, Knowledge Distillation, Feature Attribution

## TL;DR

From the perspective of class-level feature attribution, this work reveals how "cross-class features" in adversarial training (AT) are first learned and then forgotten, offering a unified explanation for both robust overfitting and the advantages of soft-label training.

## Background & Motivation

Adversarial training (AT) is one of the most effective methods to make deep networks robust against adversarial attacks, with its core formulated as a min-max optimization:

$$\min_{\boldsymbol{\theta}} \frac{1}{N}\sum_{i=1}^{N} \max_{\|\delta_i\|_p \leq \epsilon} \ell(f(\boldsymbol{\theta}, x_i + \delta_i), y_i)$$

However, two poorly understood phenomena exist in AT:

**Robust Overfitting**: The model reaches its optimal test robust accuracy in the middle of training, after which the test robust accuracy gradually declines while the training robust error continues to decrease, creating a huge generalization gap.

**Advantages of Soft Labels**: Replacing one-hot labels with soft labels such as knowledge distillation can significantly improve AT performance (e.g., from 41% to 48% on CIFAR-10), yet the underlying reason remains unclear.

Existing explanations approach this from data-level loss, label noise, etc., but lack a unified perspective. This work is the first to propose a unified hypothesis from the perspective of **class-level feature attribution**.

## Method

### Core Concepts: Cross-Class Features vs. Class-Specific Features

- **Cross-class Features**: Features shared by multiple classes, such as the "wheel" feature shared by cars and trucks in CIFAR-10.
- **Class-specific Features**: Features belonging exclusively to a single class, such as "frog eyes" for frogs.

### Feature Attribution Metric

Let the classifier be $f(\cdot) = Wg(\cdot)$, where $g$ is the feature extractor and $W \in \mathbb{R}^{K \times n}$ is the linear layer. For sample $x$ on the $i$-th class, the attribution vector is defined as:

$$A_i(x) = (g(x)_1 W[i,1], \cdots, g(x)_n W[i,n])$$

Each component $g(x)_j W[i,j]$ represents the contribution of the $j$-th feature to the logit of the $i$-th class.

### Cross-Class Feature Correlation Matrix

A cross-class feature attribution correlation matrix is constructed, measured by cosine similarity:

$$C[i,j] = \frac{A_i \cdot A_j}{\|A_i\|_2 \cdot \|A_j\|_2}$$

High $C[i,j]$ values indicate that class $i$ and class $j$ share more features.

### Numerical Metric CAS (Class Attribution Similarity)

$$\text{CAS}(C) = \sum_{i \neq j} \max(C[i,j], 0)$$

CAS quantitatively reflects the degree to which the model template utilizes cross-class features, considering only positively correlated terms.

### Main Hypothesis: Two-Stage Dynamics of AT

1. **Initial Phase**: The model simultaneously learns class-specific and cross-class features, both of which jointly reduce the robust loss.
2. **Later Phase**: When the robust loss decreases to a certain extent, cross-class features hinder further decrease in loss because they produce positive logits on non-target classes. The model starts to abandon cross-class features and shifts to depending solely on class-specific features $\rightarrow$ leading to robust overfitting.

### Theoretical Analysis: Synthetic Data Model

Consider a three-class classification task where each class has an exclusive feature $x_{E,i}$ and a cross-class feature $x_{C,j}$. The data distribution is:

$$x_{E,j} \sim \begin{cases} \mathcal{N}(\mu, \sigma^2), & j=i \\ 0, & j \neq i \end{cases}, \quad x_{C,j} \sim \begin{cases} \mathcal{N}(\mu, \sigma^2), & j \neq i \\ 0, & j=i \end{cases}$$

**Theorem 1** (Cross-class features are more sensitive to robust loss): There exists a threshold $\epsilon_0 \in (0, \mu/2)$ such that when $\epsilon > \epsilon_0$, the optimal weights in AT satisfy $w_2 = 0$ (abandoning cross-class features), whereas for any $\epsilon \in (0, \mu/2)$, $w_1 > 0$ always holds (retaining class-specific features).

**Theorem 2** (Cross-class features help robust classification): Within the range $w_2 \in [0, w_1]$, increasing $w_2$ monotonically increases the model's correct classification probability under adversarial attacks.

**Theorem 3** (Soft labels retain cross-class features): The threshold for AT with label smoothing satisfies $\epsilon_1 > \epsilon_0$, and for $\epsilon \in (0, \epsilon_1)$, $w_2^{\text{LS}}(\epsilon) > w_2^*(\epsilon)$, indicating that soft labels retain more cross-class features.

## Key Experimental Results

### CAS and Robust Accuracy at Different AT Stages on CIFAR-10 (PreActResNet-18)

| Stage | Epoch | Robust Accuracy (RA) | CAS |
|-------|-------|-------------|-----|
| Underfitting | 70 | 42.6% | 18.2 |
| Best | 108 | 47.8% | 25.6 |
| Overfitting | 200 | 42.5% | 9.0 |

$\rightarrow$ CAS at the best checkpoint is much higher than at the overfitted checkpoint, validating the positive correlation between cross-class features and robust generalization.

### $\Delta$CAS (Best - Last) under Different Perturbation Strengths $\epsilon$

| $\epsilon$ | $\Delta$CAS | Overfitting Degree |
|---|------|----------|
| 2/255 | 4.1 | Slight |
| 4/255 | 8.9 | Moderate |
| 6/255 | 13.8 | Severe |
| 8/255 | 16.6 | Very Severe |

$\rightarrow$ Larger $\epsilon$ leads to more forgotten cross-class features, which corresponds to more severe robust overfitting.

### CAS Changes under Extremely Large $\epsilon$

| $\epsilon$ | Epoch 10 CAS/RA | Best CAS/RA | Last CAS/RA |
|---|----------------|-------------|-------------|
| 8/255 | 16.7/36.9% | 25.6/47.8% | 9.0/42.5% |
| 12/255 | 15.6/29.8% | 18.9/38.7% | 8.7/34.1% |
| 16/255 | 14.4/23.8% | 17.5/31.3% | 8.4/28.1% |

$\rightarrow$ Under extremely large $\epsilon$, very few cross-class features are learned even in the initial stage, thus the forgetting effect is weakened, and robust overfitting is paradoxically alleviated.

### Comparison with Knowledge Distillation AT

| Method | Stage | RA | CAS |
|------|------|-----|-----|
| AT+KD | Best | 48.1% | 25.7 |
| AT+KD | Last | 46.2% | 24.1 |

$\rightarrow$ KD maintains high CAS throughout the training process, with the CAS gap dropping from 16.6 to 1.6, and robust overfitting is significantly mitigated.

### Cross-Dataset and Cross-Architecture Verification

- **CIFAR-100**: Best CAS=569, Last CAS=352
- **TinyImageNet**: Best CAS=1548, Last CAS=998
- **$\ell_2$-AT**: Best CAS=22.1, Last CAS=10.7
- **DeiT-Ti (Transformer)**: Best CAS=25.4, Last CAS=16.6

$\rightarrow$ The conclusion is consistent across all setups: the utilization of cross-class features is strongly positively correlated with robust generalization.

## Highlights & Insights

1. **Novel Perspective**: Provides the first unified explanation for both robust overfitting and the benefits of soft labels in AT from the perspective of "cross-class features".
2. **Intuitive Quantitative Metric (CAS)**: Concisely and effectively quantifies the model's reliance on cross-class features based on feature attribution of the final linear layer.
3. **Theoretical-Experimental Closed-Loop**: Three theorems on synthetic data precisely characterize the sensitivity and utility of cross-class features, which are perfectly validated by experiments.
4. **Saliency Map Visualization**: Visually demonstrates via Grad-CAM that the best checkpoint focuses on global features (wheels + car body), whereas the overfitted checkpoint focuses only on local exclusive features (curved roof), providing strong explanatory power.
5. **Broad Coverage**: Synthetically validates hypotheses across $\ell_\infty$/$\ell_2$ norms, CNN/Transformer architectures, multiple datasets, and Fast AT (FAT) scenarios.

## Limitations & Future Work

1. **CAS Dependency on Linear Layer Assumption**: The attribution vector $A_i(x) = g(x) \odot W[i]$ is only applicable to architectures where the final layer is linear, and cannot be directly generalized to more complex classification heads.
2. **Theory Limited to Synthetic Models**: The theoretical analysis on the three-class linear model is relatively simplified and still has a gap with the actual training dynamics of deep networks.
3. **No New Defense Method Proposed**: The paper focuses primarily on analysis and understanding, without designing new adversarial training algorithms based on the cross-class feature hypothesis.
4. **Uncertain Causality**: While the correlation between CAS and robust accuracy is confirmed, whether it represents a direct causal relationship still requires more rigorous validation.
5. **Insufficient Large-Scale Dataset Validation**: Experiments are mainly conducted on CIFAR-10/100 and TinyImageNet, lacking validation on the scale of ImageNet.

## Related Work & Insights

- **Ilyas et al., 2019**: Proposed the framework of robust vs. non-robust features. Based on this, this paper further distinguishes between cross-class and class-specific robust features.
- **Rice et al., 2020**: The first to systematically study robust overfitting. This work provides a new feature-level explanation for it.
- **Chen et al., 2021 (ARD)**: A representative work that improves AT using knowledge distillation. This paper explains its success from the perspective of cross-class features.
- **Insight**: The forgetting mechanism of cross-class features suggests that robust overfitting could be mitigated by explicitly encouraging the learning of cross-class features (e.g., through feature regularization or cross-class contrastive learning).

## Rating

- Novelty: ⭐⭐⭐⭐ — Novel perspective of cross-class features, providing a unified explanation for two major phenomena.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive validation across multiple datasets, multiple architectures, and multiple norms.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, combining theory and experiments tightly.
- Value: ⭐⭐⭐⭐ — Provides deep understanding of AT but does not directly yield new methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Understanding Model Ensemble in Transferable Adversarial Attack](understanding_model_ensemble_in_transferable_adversarial_attack.md)
- [\[NeurIPS 2025\] Understanding and Improving Adversarial Robustness of Neural Probabilistic Circuits](../../NeurIPS2025/ai_safety/understanding_and_improving_adversarial_robustness_of_neural_probabilistic_circu.md)
- [\[NeurIPS 2025\] Distributional Adversarial Attacks and Training in Deep Hedging](../../NeurIPS2025/ai_safety/distributional_adversarial_attacks_and_training_in_deep_hedging.md)
- [\[ACL 2025\] Efficiently Identifying Watermarked Segments in Mixed-Source Texts](../../ACL2025/ai_safety/watermark_segment_detection.md)
- [\[CVPR 2026\] V-Attack: Targeting Disentangled Value Features for Controllable Adversarial Attacks on LVLMs](../../CVPR2026/ai_safety/v-attack_targeting_disentangled_value_features_for_controllable_adversarial_atta.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] On the Clean Generalization and Robust Overfitting in Adversarial Training from Two Theoretical Views: Representation Complexity and Training Dynamics
description: >-
  [ICML2025][LLM Pretraining][Adversarial Training] This paper theoretically explains the phenomenon of "coexistence of clean generalization and robust overfitting" (CGRO) in adversarial training from two perspectives: **representation complexity** and **training dynamics**. It shows that while a CGRO classifier can be realized via robust memorization with only an additional $\tilde{O}(ND)$ parameters, true robust generalization requires exponential model capacity in the worst…
tags:
  - "ICML2025"
  - "LLM Pretraining"
  - "Adversarial Training"
  - "Robust Overfitting"
  - "Clean Generalization"
  - "Representation Complexity"
  - "Training Dynamics"
  - "Feature Learning"
date: 2026-05-08
content_hash: 77fd5089399714d8
---

# On the Clean Generalization and Robust Overfitting in Adversarial Training from Two Theoretical Views: Representation Complexity and Training Dynamics

**Conference**: ICML2025  
**arXiv**: [2306.01271](https://arxiv.org/abs/2306.01271)  
**Code**: None  
**Area**: LLM Pre-training  
**Keywords**: Adversarial Training, Robust Overfitting, Clean Generalization, Representation Complexity, Training Dynamics, Feature Learning

## TL;DR

This paper theoretically explains the phenomenon of "coexistence of clean generalization and robust overfitting" (CGRO) in adversarial training from two perspectives: **representation complexity** and **training dynamics**. It shows that while a CGRO classifier can be realized via robust memorization with only an additional $\tilde{O}(ND)$ parameters, true robust generalization requires exponential model capacity in the worst case. On structured data, a three-phase transition process during adversarial training causes the network to partially learn true features while completely memorizing noise, thereby provably converging to the CGRO state.

## Background & Motivation

Adversarial training is the dominant method for improving the adversarial robustness of models, but a contradictory phenomenon is observed in practice:
- **Clean Generalization**: Models trained with adversarial training maintain high accuracy on clean test data (e.g., >80% on CIFAR10).
- **Robust Overfitting**: Robust training error can decrease to near 0, but robust test error remains high (e.g., only ~50% on CIFAR10), showing a significant robust generalization gap.

This "Clean Generalization and Robust Overfitting" (CGRO) phenomenon differs from standard benign overfitting (where generalization also occurs on clean tests) and cannot be explained by a simple "robustness-accuracy trade-off". Although existing works provide partial explanations from perspectives such as sample complexity and Lipschitz stability, two key gaps remain:

**Unclear Mechanism**: They fail to elucidate what mechanism in adversarial training directly causes robust overfitting.

**Ignoring Clean Generalization**: Most theories only focus on the degradation of robustness, ignoring the fact that clean test accuracy remains high.

Core Problem: **What underlying mechanism leads to the coexistence of clean generalization and robust overfitting during adversarial training?**

## Method

### Problem Formulation

Consider a binary classification setting $(X, y) \sim \mathcal{D}$, where $y \in \{-1, 1\}$ and the classifier is $f: \mathcal{X} \to \mathbb{R}$. Three key metrics are defined:
- **Clean Test Error**: $\mathcal{L}_\mathcal{D}(f) = \mathbb{P}[\text{sgn}(f(X)) \neq y]$
- **Robust Test Error**: $\mathcal{L}_\mathcal{D}^{p,\delta}(f) = \mathbb{E}[\max_{\|X'-X\|_p \le \delta} \mathbb{I}\{\text{sgn}(f(X')) \neq y\}]$
- **Robust Training Error**: robust error on the training set $\mathcal{S}$

**Definition of CGRO Classifier**: A classifier that satisfies $\mathcal{L}_\mathcal{D}(f) = o(1)$ (good clean generalization), $\mathcal{L}_\mathcal{S}^{p,\delta}(f) = o(1)$ (good robust training), but $\mathcal{L}_\mathcal{D}^{p,\delta}(f) = \Omega(1)$ (poor robust generalization).

### Perspective 1: Representation Complexity Analysis

Under three reasonable assumptions (bounded data, separation between classes $> 2\delta$, and existence of a polynomial-sized clean classifier), the authors construct a key CGRO classifier:

$$f_\mathcal{S}(X) = \underbrace{f_{\text{clean}}(X)(1 - \mathbb{I}\{X \in \cup_i \mathbb{B}_p(X_i, \delta)\})}_{\text{对未见数据用干净分类}} + \underbrace{\sum_{i=1}^N y_i \mathbb{I}\{X \in \mathbb{B}_p(X_i, \delta)\}}_{\text{对训练数据做鲁棒记忆}}$$

**Core Idea**: Directly memorize the correct label within the $\delta$-neighborhood of the training data points (robust memorization); predict using the clean classifier outside the neighborhood.

**Theorem 4.4 (CGRO Polynomial Upper Bound)**: This CGRO classifier can be represented by a ReLU network with $\text{poly}(D) + \tilde{O}(ND)$ parameters. The key is to approximate the distance function $\|X - X_i\|_p$ and the indicator function using a ReLU network.

**Theorem 4.7 (Robust Classifier Exponential Lower Bound)**: There exists a distribution $\mathcal{D}$ satisfying the above assumptions such that no ReLU network with parameters $\le \Omega(\exp(D))$ can achieve robust generalization.

**Key Inequality**:
$$\underbrace{\text{Clean Classifier}}_{\text{poly}(D)} \lesssim \underbrace{\text{CGRO Classifier}}_{\text{poly}(D) + \tilde{O}(ND)} \ll \underbrace{\text{Robust Classifier}}_{\Omega(\exp(D))}$$

This reveals that CGRO is a "capacity trap" — model capacity is sufficient for CGRO but far below what is required for robust generalization.

### Perspective 2: Training Dynamics Analysis

Analyzing the learning process of adversarial training on structured data (Patch Data):
- **Data Structure**: $X = (X[1], \ldots, X[P])$, containing one signal patch $X[\text{signal}] = \alpha y w^*$ and $P-1$ noise patches $X[j] \sim \mathcal{N}(0, (I_d - w^* w^{*\top})\sigma_p^2)$
- **Model**: Two-layer convolutional network, width $m = \Theta(N)$, using $\text{ReLU}^q$ activation
- **Training**: Gradient descent minimizing the adversarial training loss (logistic loss + robust regularization)

**Theorem 5.9 (Three-Phase Phase Transition)**: After $T = \Omega(\text{poly}(d))$ iterations of adversarial training, the network:
1. **Partially learns true features**: $\mathcal{U}^{(T)} = \Theta(\alpha^{-q})$
2. **Completely memorizes noise features**: $\forall i, \mathcal{V}_i^{(T)} = \Theta(1)$

Three-phase process:
- **Phase I** (Signal Growth Phase): The signal component $u^{(t)}$ grows theoretically, reaching the scale of $\tilde{\Omega}(\alpha^{-1})$, and the model learns a portion of true features.
- **Phase II** (Signal Stagnation Phase): Signal growth is dominated by noise components, and growth tends to stop.
- **Phase III** (Noise Memorization Phase): Noise components $v_{i,j}^{(t)}$ grow quadratically to $\Omega(1)$, and the model achieves robust training accuracy by memorizing sample-level noise.

**Physical Intuition**: Adversarial training forces the model to classify correctly within the $\delta$-neighborhood of training points. Since adversarial perturbations in the direction of the true feature can almost completely flip the signal ($\delta \approx \alpha$), the model cannot resist perturbations solely based on the true feature, and instead turns to memorizing the noise pattern of each training sample. This works on the training data but fails to generalize — because the noise in the test data is completely new.

## Key Experimental Results

### Main Results: Adversarial Training Performance with Different Model Sizes

| Dataset | Model Size | Clean Test Acc | Robust Test Acc | Robust Train Acc |
|--------|---------|------------|------------|------------|
| MNIST  | ×1      | 11.35      | 11.35      | 11.70      |
| MNIST  | ×8      | 11.35      | 11.35      | 11.70      |
| MNIST  | ×12     | 95.06      | 77.96      | 99.30      |
| MNIST  | ×16     | 94.85      | 83.43      | 99.50      |
| CIFAR10| ×1 (WRN)| 82.56      | 43.39      | 64.19      |
| CIFAR10| ×5      | 85.83      | 46.25      | 97.37      |
| CIFAR10| ×10     | 86.05      | 50.08      | 99.57      |

**Key Findings**: As the model size increases, robust training accuracy first increases; the robust generalization gap first widens and then slowly shrinks; for small models (MNIST ×1/×8), adversarial training degenerates into trivial solutions, validating the representation complexity theory.

### Synthetic Data Experiments

| Metric | Train | Test |
|------|------|------|
| Clean Acc | 100.0 | 98.5 |
| Robust Acc | 100.0 | 17.5 |

The training dynamics plot (Figure 2c) clearly illustrates the three-phase phase transition: the signal component first grows rapidly and then stagnates, while noise memorization gradually climbs to $\Theta(1)$, which is in complete agreement with theoretical predictions.

## Highlights & Insights

1. **Formalization of the CGRO Concept**: First to explicitly define the coexistence phenomenon of "clean generalization + robust overfitting", distinguishing it from the simple robust-accuracy trade-off.
2. **Robust Memorization Mechanism**: Reveals the core mechanism of CGRO—models achieve robust training accuracy by memorizing noise within the neighborhood of training points, but this memorization fails to generalize.
3. **Clear Complexity Hierarchy**: Establishes a representation complexity hierarchy of Clean ≲ CGRO ≪ Robust, explaining why adversarial training naturally converges to CGRO rather than robust generalization.
4. **Three-Phase Analysis Technique**: Decouples the complex adversarial training dynamics into three analyzable phases, marking the first application of feature learning theory to the field of adversarial robustness.
5. **Theory-Experiment Consistency**: Synthetic data experiments precisely replicate the theoretically predicted three-phase phase transition.

## Limitations & Future Work

1. **Strong Data Assumptions**: There is a large gap between structured patch data and real image data, and the single signal patch assumption is overly simplified.
2. **Activation Function Restrictions**: The theoretical analysis uses $\text{ReLU}^q$ ($q \ge 2$) rather than standard ReLU, which is not commonly used in practice.
3. **Perturbation Range Constraints**: Adversarial perturbations are restricted to the signal direction $\text{span}(w^*)$, simplifying optimization analysis but deviating from realistic all-directional perturbations.
4. **Analysis Limited to Two-Layer Networks**: Modern adversarial training uses deep networks (e.g., WideResNet); whether the analysis of two-layer CNNs generalizes remains questionable.
5. **The Setup of $\delta \approx \alpha$**: The robust radius is almost equal to the signal magnitude, which is an extreme case, and the impact of the ratio $\delta/\alpha$ on CGRO in practice is not discussed.
6. **Lack of Mitigation Strategies**: The paper only explains the CGRO phenomenon without proposing how to utilize this theory to mitigate robust overfitting.

## Related Work & Insights

- **Empirics of Robust Overfitting**: Rice et al. (2020) first systematically observed robust overfitting in adversarial training.
- **Sample Complexity**: Schmidt et al. (2018) proved that robust generalization requires more data.
- **Representation Complexity**: Li et al. (2022) proved that robust classification requires exponential-sized models; this paper generalizes it to non-linearly separable settings.
- **Feature Learning Theory**: The patch data paradigm of Allen-Zhu & Li (2020, 2022) and Jelassi & Li (2022); this paper introduces it to adversarial training analysis for the first time.
- **Memorization Effect**: Dong et al. (2021) and Xu et al. (2021) explored memorization effects in adversarial training; this paper further distinguishes the preservation of clean generalization.

**Insights**: This work suggests that mitigating robust overfitting may require (1) increasing model capacity to approach the exponential demand, (2) introducing regularization to suppress noise memorization, or (3) designing data augmentations so that models do not rely on sample-level noise.

## Rating

- Novelty: ⭐⭐⭐⭐ — The formal definition of CGRO and the dual-perspective analysis framework are highly original.
- Experimental Thoroughness: ⭐⭐⭐ — Validation on synthetic data is thorough, but the experiments on real data are relatively simple.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured, with a clear presentation of proof ideas.
- Value: ⭐⭐⭐⭐ — Makes key theoretical contributions to understanding the nature of adversarial training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Disaggregation Reveals Hidden Training Dynamics: The Case of Agreement Attraction](../../NeurIPS2025/llm_pretraining/disaggregation_reveals_hidden_training_dynamics_the_case_of_agreement_attraction.md)
- [\[ICCV 2025\] ConstStyle: Robust Domain Generalization with Unified Style Transformation](../../ICCV2025/llm_pretraining/conststyle_robust_domain_generalization_with_unified_style_transformation.md)
- [\[ICLR 2026\] Intrinsic Training Dynamics of Deep Neural Networks](../../ICLR2026/llm_pretraining/intrinsic_training_dynamics_of_deep_neural_networks.md)
- [\[ACL 2025\] Training Dynamics Underlying Language Model Scaling Laws: Loss Deceleration and Zero-Sum Learning](../../ACL2025/llm_pretraining/training_dynamics_underlying_language_model_scaling_laws_loss_deceleration_and_z.md)
- [\[ACL 2025\] Adversarial Tokenization](../../ACL2025/llm_pretraining/adversarial_tokenization.md)

</div>

<!-- RELATED:END -->

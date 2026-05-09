---
title: >-
  [Paper Note] Learning Fair Representations with Kolmogorov-Arnold Networks
description: >-
  [AAAI 2026][Kolmogorov-Arnold Networks] This paper proposes integrating Kolmogorov-Arnold Networks (KAN) into an adversarial debiasing framework, leveraging KAN's spline-based architecture to provide theoretical guarantees of Lipschitz continuity and smoothness. An adaptive $\lambda$ update mechanism is introduced to dynamically balance fairness and accuracy. The approach achieves significant improvements on fairness metrics on the UCI college admissions dataset.
tags:
  - AAAI 2026
  - Kolmogorov-Arnold Networks
  - adversarial debiasing
  - fairness
  - adaptive penalty
  - college admissions
date: 2026-05-08
content_hash: 9c5ef6b3f69a69e4
---

# Learning Fair Representations with Kolmogorov-Arnold Networks

**Conference**: AAAI 2026
**arXiv**: [2511.11767](https://arxiv.org/abs/2511.11767)
**Code**: N/A
**Area**: Fairness in Machine Learning / Explainable AI
**Keywords**: Kolmogorov-Arnold Networks, adversarial debiasing, fairness, adaptive penalty, college admissions

## TL;DR

This paper proposes integrating Kolmogorov-Arnold Networks (KAN) into an adversarial debiasing framework, leveraging KAN's spline-based architecture to provide theoretical guarantees of Lipschitz continuity and smoothness. An adaptive $\lambda$ update mechanism is introduced to dynamically balance fairness and accuracy. The approach achieves significant improvements on fairness metrics on the UCI college admissions dataset.

## Background & Motivation

**Background**: Machine learning models are widely deployed in high-stakes decision-making scenarios such as college admissions and healthcare, yet historical biases embedded in training data frequently lead to discriminatory predictions against marginalized groups. In college admissions specifically, students from low-income families and first-generation college students systematically face resource inequalities.

**Limitations of Prior Work**:

**Pre-processing methods**: Modify the input data distribution to remove bias, but alter data integrity and are ill-suited for admissions scenarios requiring comprehensive evaluation.

**Post-processing methods**: Adjust model outputs after prediction, with limited influence on internal representations.

**In-processing methods (adversarial debiasing)**: Incorporate fairness constraints directly during training, making them the most appropriate choice. However, existing MLP-based adversarial debiasing frameworks suffer from two core problems:
   - **Instability in the fairness–accuracy trade-off**: Zhao et al. proved that the error lower bound of a classifier satisfying exact Demographic Parity (DP) is half the difference in base rates between the two groups, rendering the fairness–accuracy competition during training unstable.
   - **Black-box opacity**: Standard neural networks cannot explicitly express feature contributions, undermining trustworthiness in sensitive contexts such as admissions.

**Key Challenge**: In adversarial debiasing frameworks, the selection of the fairness penalty coefficient $\lambda$ is highly dependent on manual tuning, and a fixed $\lambda$ throughout training cannot adapt to the dynamically shifting fairness–accuracy trade-off.

**Key Insight**: Replace MLP with KAN as both the classifier and adversary in the adversarial debiasing framework. KAN, grounded in the Kolmogorov-Arnold representation theorem, substitutes fixed activation functions with learnable B-spline functions, inherently providing Lipschitz continuity and $\beta$-smoothness to stabilize adversarial training. An adaptive $\lambda$ update strategy is further introduced to eliminate manual tuning.

## Method

### Overall Architecture

The framework adopts the classical adversarial debiasing structure (min-max game), replacing both the classifier $f$ and the adversary $g$ with KAN networks:

- **Classifier $f$**: Takes input features $x$ and predicts label $y$ (admitted/not admitted).
- **Adversary $g$**: Takes the classifier output $f(x)$ and attempts to infer the sensitive attribute $z$ (low-income/non-low-income, first-generation/non-first-generation).
- **Optimization objective**: $\min_{\theta_f} \max_{\theta_g} \mathcal{L}_{\mathcal{Y}}(f(x), y) - \lambda \cdot \mathcal{L}_{\mathcal{Z}}(g(f(x)), z)$

The classifier minimizes prediction loss while maximizing the adversary's loss (preventing the adversary from inferring the sensitive attribute from predictions); the adversary attempts to minimize its own loss. $\lambda$ controls the strength of the fairness constraint.

### Key Designs

1. **KAN Architecture Replacing MLP**:

    - **Function**: Replace conventional fully connected networks with B-spline-based KAN as the classifier and adversary.
    - **Mechanism**: KAN represents any multivariate function as a finite superposition and composition of univariate functions: $f(x_1, ..., x_n) = \sum_{q=1}^{2n+1} \Phi_q(\sum_{r=1}^n \phi_{q,r}(x_r))$, where each $\phi$ is parameterized by B-splines.
    - **Design Motivation**: (1) Each spline function $h(x)$ is a piecewise polynomial that is twice continuously differentiable, guaranteeing Lipschitz continuity (Lemma 1) and robustness to input perturbations. (2) The $\beta$-smoothness of spline functions (Lemma 2) bounds gradient variation, ensuring stable convergence in adversarial training. (3) The visualization and decomposition of spline functions provide model interpretability.

2. **Adaptive $\lambda$ Update Mechanism**:

    - **Function**: Dynamically adjust the penalty coefficient $\lambda$ after each training epoch based on the current fairness status.
    - **Mechanism**: Compute the fairness gap $\delta = (\tau - p\%\text{-Rule}) / \tau$ (where $\tau$ is the target fairness threshold), then update $\lambda \leftarrow \text{clip}(\lambda + \eta \cdot \delta, 0.1, 1.0)$.
    - **Design Motivation**: $\lambda$ is increased to strengthen the constraint when fairness is insufficient, and decreased to recover accuracy headroom when the target is already satisfied. The clip operation keeps $\lambda$ within a safe range, preventing extreme values from destabilizing training.

3. **Multi-Optimizer Exploration**:

    - **Function**: Compare Adam, Optimistic Adam (OAdam), and ADOPT optimizers within the KAN framework.
    - **Mechanism**: OAdam uses gradient extrapolation from past steps to accelerate convergence in the min-max game; ADOPT adaptively adjusts the learning rate for non-stationary environments.
    - **Design Motivation**: KAN's high complexity makes training sensitive to the choice of optimizer, and different optimizers exhibit distinct behaviors in the fairness–accuracy trade-off, necessitating empirical exploration.

### Loss & Training

Training alternates between two phases:
1. Freeze the classifier and train the adversary to minimize $\mathcal{L}_{\mathcal{Z}}(g(f(x)), z)$.
2. Train the classifier to minimize $\mathcal{L}_{\mathcal{Y}}(f(x), y) - \lambda \cdot \mathcal{L}_{\mathcal{Z}}(g(f(x)), z)$.

KAN uses B-spline order $k=3$ (cubic splines) and supports grid refinement to progressively increase model capacity.

## Key Experimental Results

### Main Results (Fairness and Accuracy Comparison)

| Model | Optimizer | Acc | AUROC | p%-Rule (Low-income) | p%-Rule (First-gen) | DP Gap (Low-income) | DP Gap (First-gen) |
|-------|-----------|-----|-------|----------------------|---------------------|---------------------|--------------------|
| KAN(D₁) | Adam | 74.92 | 80.27 | 85.79 | 89.65 | 0.028 | 0.035 |
| KAN(D₁) | ADOPT | 74.58 | 81.45 | 89.21 | 90.92 | 0.047 | 0.055 |
| KAN(D₂) | OAdam | 81.69 | 86.26 | **99.25** | **99.55** | 0.026 | 0.030 |
| KAN(D₂) | ADOPT | **82.51** | **86.68** | **99.25** | **99.70** | 0.032 | 0.037 |
| MLP-B₁(D₁) | Adam | 72.55 | 72.53 | 98.32 | 99.61 | 0.009 | 0.011 |
| MLP-B₁'(D₂) | Adam | 74.87 | 71.77 | 93.97 | 97.79 | 0.010 | 0.004 |
| ExpGrad-B₂(D₁) | Adam | 72.44 | 76.12 | 95.22 | 97.59 | 0.024 | 0.012 |
| ROAD-B₃(D₁) | Adam | 74.92 | 80.95 | 83.10 | 83.91 | 0.085 | 0.080 |

On the larger dataset D₂, the KAN+ADOPT configuration simultaneously achieves the highest accuracy (82.51%) and near-perfect fairness (p%-Rule 99.25%/99.70%).

### Ablation Study (Effect of Spline Order)

| Configuration | Key Performance | Notes |
|---------------|-----------------|-------|
| KAN $k=3$ | Most stable convergence | Optimal balance between expressiveness and smoothness |
| KAN $k=4$ | Slight improvement in p%-Rule | Higher-order splines better model the fairness boundary |
| KAN $k=5$ | Convergence fluctuations | Smoothness–dimensionality trade-off; training loss spikes observed |
| MLP baseline | High fairness but low accuracy | Fixed $\lambda$ strategy excessively sacrifices accuracy for fairness |

### Key Findings

- KAN's advantage is more pronounced on the larger dataset D₂: accuracy exceeds the MLP baseline by approximately 7.6% (82.51 vs. 74.87), while simultaneously achieving higher p%-Rule (99.25 vs. 93.97).
- The ADOPT optimizer is best suited to KAN's spline architecture, achieving the optimal balance between accuracy and fairness.
- The adaptive $\lambda$ strategy is one of the key factors differentiating KAN from MLP baselines; the fixed $\lambda$ used in MLP baselines tends to over-sacrifice accuracy in pursuit of fairness.
- Training loss spikes observed in KAN are an intrinsic characteristic of KAN's universal approximation property (smoothness–dimensionality trade-off) and do not affect final convergence.

## Highlights & Insights

- This work is the first to introduce KAN into an adversarial debiasing framework. Beyond an architectural substitution, it provides theoretical analysis (Lipschitz continuity + $\beta$-smoothness) explaining why KAN is well-suited for adversarial training.
- The adaptive $\lambda$ design is concise and practical: linear adjustment based on the gap between the p%-Rule and the target threshold, with clip-based stabilization, avoids expensive hyperparameter search.
- Validation on real admissions data rather than synthetic datasets enhances practical relevance.

## Limitations & Future Work

- Only binary classification (admitted/not admitted) and binary sensitive attributes are evaluated; multi-class and intersectional sensitive attribute settings remain unexplored.
- The fairness definition is limited to Demographic Parity; conditional fairness criteria such as Equalized Odds are not considered.
- KAN's training complexity exceeds that of MLP due to B-spline coefficient learning, raising scalability concerns on large-scale data.
- Comparisons with more recent fairness methods (e.g., FairMixup, LAFTR) are absent.
- Experiments are conducted solely on the UCI admissions dataset; validation across additional domains is needed.

## Related Work & Insights

- The Lipschitz continuity and smoothness analysis of KAN can be generalized to other scenarios requiring stable adversarial training, such as GAN training and robust classification.
- The adaptive penalty coefficient approach parallels the augmented Lagrangian method in constrained optimization, though the proposed mechanism is simpler and more direct.
- Combining KAN's interpretability (spline function visualization per edge) for fairness auditing is a promising research direction.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] ParaMETA: Towards Learning Disentangled Paralinguistic Speaking Styles Representations](parameta_towards_learning_disentangled_paralinguistic_speaking_styles_representa.md)
- [\[ICLR 2026\] Addressing Divergent Representations from Causal Interventions on Neural Networks](../../ICLR2026/others/addressing_divergent_representations_causal.md)
- [\[AAAI 2026\] Decomposition and Preprocessing of Ternary Constraint Networks](decomposition_and_preprocessing_of_ternary_constraint_networks.md)
- [\[AAAI 2026\] DS-ATGO: Dual-Stage Synergistic Learning via Forward Adaptive Threshold and Backward Gradient Optimization for Spiking Neural Networks](ds-atgo_dual-stage_synergistic_learning_via_forward_adaptive_threshold_and_backw.md)
- [\[AAAI 2026\] Formal Abductive Latent Explanations for Prototype-Based Networks](formal_abductive_latent_explanations_for_prototype-based_networks.md)

<!-- RELATED:END -->

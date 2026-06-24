---
title: >-
  [Paper Note] Model Immunization from a Condition Number Perspective
description: >-
  [ICML 2025][Image Generation][Model Immunization] This paper defines and analyzes the model immunization problem from the perspective of the Hessian matrix condition number, proposing a regularizer that maximizes/minimizes the condition number to render pre-trained models difficult to fine-tune for harmful tasks without affecting their performance on benign tasks.
tags:
  - "ICML 2025"
  - "Image Generation"
  - "Model Immunization"
  - "Condition Number"
  - "Hessian Matrix"
  - "Regularization"
  - "Transfer Learning"
date: 2026-05-08
content_hash: a570f8b732c00f1a
---

# Model Immunization from a Condition Number Perspective

**Conference**: ICML 2025  
**arXiv**: [2505.23760](https://arxiv.org/abs/2505.23760)  
**Code**: [amberyzheng/model-immunization-cond-num](https://github.com/amberyzheng/model-immunization-cond-num)  
**Area**: Image Generation  
**Keywords**: Model Immunization, Condition Number, Hessian Matrix, Regularization, Transfer Learning

## TL;DR

This paper defines and analyzes the model immunization problem from the perspective of the Hessian matrix condition number, proposing a regularizer that maximizes/minimizes the condition number to render pre-trained models difficult to fine-tune for harmful tasks without affecting their performance on benign tasks.

## Background & Motivation

**Model Immunization**, proposed by Zheng & Yeh (2024), aims to pre-train a model such that it is difficult to fine-tune for harmful content generation while maintaining performance on benign tasks. This is of great significance for preventing the abuse of open-source models.

Prior work (IMMA) formulates immunization as a bi-level optimization and demonstrates empirical effectiveness on text-to-image models. However, critical issues remain:

**Lack of a precise definition for immunized models**

**Unclear conditions for when immunization is feasible**

**Lack of theoretical understanding**

This paper connects the problem to the **condition number** in classical optimization theory:
- The condition number $\kappa(S) = \sigma_{\max}/\sigma_{\min}$ measures how "well-behaved" a matrix is.
- The convergence rate of gradient descent is governed by $(1 - \sigma_{\min}/\sigma_{\max})^t$.
- **Larger condition number $\rightarrow$ slower convergence $\rightarrow$ harder fine-tuning**

## Method

### Overall Architecture

Consider a transfer learning setup with a linear feature extractor $f_\theta(x) = x^\top\theta$ ($\theta \in \mathbb{R}^{D_{in} \times D_{in}}$) and linear probing.

**Definition 3.1 (Three Conditions for an Immunized Model)**:
- **(a) Harmful task becomes harder**: $\kappa(\nabla_w^2 \mathcal{L}(\mathcal{D}_H, w, \theta^I)) \gg \kappa(\nabla_w^2 \mathcal{L}(\mathcal{D}_H, w, I))$
- **(b) Benign task does not become harder**: $\kappa(\nabla_\omega^2 \mathcal{L}(\mathcal{D}_P, \omega, \theta^I)) \leq \kappa(\nabla_\omega^2 \mathcal{L}(\mathcal{D}_P, \omega, I))$
- **(c) Pre-training performance is maintained**: $\min_{\omega,\theta} \mathcal{L}(\mathcal{D}_P, \omega, \theta) \approx \min_\omega \mathcal{L}(\mathcal{D}_P, \omega, \theta^I)$

### Key Designs

#### 1. Hessian Analysis (Proposition 3.2)

The Hessian matrix of linear probing is $H_H(\theta) = \theta^\top K_H \theta$ ($K_H = X_H^\top X_H$), with singular values:
$$\sigma_i = \sum_{j=1}^{D_{in}} (\sigma_{\theta,i} (u_{\theta,i}^\top q_j) \sqrt{\gamma_j})^2$$

**Key Insight**: The condition number of the Hessian depends on the relative angle between the singular vectors of the feature extractor $\theta$ and those of the data covariance matrix $K$. When the singular vectors of $K_P$ and $K_H$ are perfectly aligned, immunization is **impossible**.

#### 2. Condition Number Maximization Regularizer (Theorem 4.1)

A novel regularizer is proposed:
$$\mathcal{R}_{\text{ill}}(S) = \frac{1}{\frac{1}{2k}\|S\|_F^2 - \frac{1}{2}(\sigma_S^{\min})^2}$$

Four key properties:
- **Non-negativity**: $\mathcal{R}_{\text{ill}}(S) \geq 0$, and is 0 if and only if $\kappa(S)=\infty$.
- **Upper bound**: $1/\log(\kappa(S)) \leq (\sigma_{\max})^2 \mathcal{R}_{\text{ill}}(S)$.
- **Differentiability**: Differentiable when $\sigma_{\min}$ is unique, and the gradient has a closed-form solution.
- **Monotonic increase guarantee**: Under an appropriate step size, $\kappa(S') > \kappa(S)$ after a gradient descent update.

It is used in conjunction with the existing condition number minimization regularizer $\mathcal{R}_{\text{well}}$ (Nenov et al., 2024).

#### 3. Immunization Algorithm (Algorithm 1)

Optimization objective:
$$\min_{\omega,\theta} \mathcal{R}_{\text{ill}}(H_H(\theta)) + \mathcal{R}_{\text{well}}(H_P(\theta)) + \mathcal{L}(\mathcal{D}_P, \omega, \theta)$$

Key technique: Multiplying the gradient update by $K^{-1}$ to guarantee the monotonicity of the condition number change (Theorem 4.3). Practically, this is integrated into PyTorch automatic differentiation via a "dummy layer" trick.

### Loss & Training

Joint optimization of three terms:
1. $\mathcal{R}_{\text{ill}}(H_H(\theta))$: Maximizes the condition number of the harmful task.
2. $\mathcal{R}_{\text{well}}(H_P(\theta))$: Minimizes the condition number of the benign task.
3. $\mathcal{L}(\mathcal{D}_P, \omega, \theta)$: Maintains performance on the pre-training task.

## Key Experimental Results

### Main Results

**Evaluation Metric**: Relative Immunization Ratio (RIR) = $\frac{\kappa(H_H(\theta^I))/\kappa(H_H(I))}{\kappa(H_P(\theta^I))/\kappa(H_P(I))}$, higher is better.

**House Price regression task** (Table 1):

| Method | Eq.15(i)↑ | Eq.15(ii)↓ | RIR↑ |
|------|-----------|------------|------|
| $\mathcal{R}_{\text{ill}}$ Only | **90.02** | 72.42 | 1.24 |
| IMMA | 7.05 | 3.55 | 2.00 |
| Opt $\kappa$ | 1.52 | **0.016** | 92.58 |
| **Ours** | 18.92 | 0.053 | **356.20** |

**MNIST classification task** (Table 2, average across 90 task pairs):

| Method | RIR↑ |
|------|------|
| $\mathcal{R}_{\text{ill}}$ Only | 1.93 |
| IMMA | 1.77 |
| Opt $\kappa$ | 69.73 |
| **Ours** | **70.04** |

### Ablation Study

- **Convergence Visualization** (Figure 1): Using gradient descent with exact line search, Ours accelerates convergence on $\mathcal{D}_P$ and significantly slows convergence on $\mathcal{D}_H$.
- While $\mathcal{R}_{\text{ill}}$ Only and IMMA make the harmful task harder, they also make the benign task harder (increasing both condition numbers).

### Key Findings

- Simply maximizing the condition number of the harmful task is insufficient; the condition number of the benign task must be simultaneously controlled.
- The feasibility of immunization depends on the angular difference between the singular vectors of $K_P$ and $K_H$.
- Effectiveness is also demonstrated on non-linear models (ResNet, ViT), despite the theory being formulated for linear models.

## Highlights & Insights

1. **Theoretical Contribution from a Condition Number Perspective**: Elegantly connects model immunization with classical optimization theory, providing the first precise mathematical definition of immunized models.
2. **Novel $\kappa$-Maximization Regularizer**: Dual to the existing $\kappa$-minimization regularizer, guaranteeing monotonic increase under gradient descent.
3. **Translation of Theoretical Guarantees to Practice**: Propagates matrix-level monotonicity guarantees to the parameter-level $\theta$ through $K^{-1}$ preconditioning.
4. **Intuitive Feasibility Conditions**: The immunization strength depends on the "angular difference" between the singular vectors of $K_P$ and $K_H$.
5. **Design of the RIR Metric**: Provides a single unified metric to evaluate immunization quality.

## Limitations & Future Work

1. **Linear Model Assumption**: Theoretical analysis is restricted to linear feature extractors and linear probing, exhibiting a gap with practical deep networks.
2. **Practical Validity of Monotonicity Guarantees**: Monotonicity guarantees cannot be linearly combined when the three gradients are jointly updated.
3. **Requirements for Harmful Data Access**: The immunization process requires knowing the data distribution of the harmful task.
4. **Linear Probing Limitation**: The immunization effect under full fine-tuning scenarios is not analyzed.
5. **Hyperparameter Sensitivity**: The selection of $\lambda_P$ and $\lambda_H$ requires balancing the norms of the three gradients.
6. **Lack of Theory on Non-linear Models**: Although experiments on ResNet/ViT yield good results, they lack theoretical guarantees.

## Related Work & Insights

- **Zheng & Yeh (2024) IMMA**: Formulates immunization as a bi-level optimization, whereas this paper provides a clearer theoretical framework.
- **Nenov et al. (2024)**: Proposes the $\mathcal{R}_{\text{well}}$ regularizer to minimize the condition number; this paper designs the dual counterpart.
- **Condition Number and Optimization**: The condition number determines the convergence rate in classical optimization theory (Boyd & Vandenberghe).
- **Model Safety**: Brundage et al. (2018) and Marchal et al. (2024) discuss the abuse risks of open-source models.
- **Insight**: The idea of manipulating condition numbers could be extended to other safety scenarios requiring "selective fine-tuning resistance".

## Rating

- Novelty: ⭐⭐⭐⭐ — The condition number perspective is novel, and the $\kappa$-maximization regularizer is a meaningful theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Experiments cover both linear models and deep networks, compared against multiple baselines.
- Writing Quality: ⭐⭐⭐⭐ — Clear mathematical derivations with rigorous definitions.
- Value: ⭐⭐⭐⭐ — Provides the first theoretical framework for model immunization, though the application scenarios need further expansion.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Broadband Ground Motion Synthesis by Diffusion Model with Minimal Condition](broadband_ground_motion_synthesis_by_diffusion_model_with_minimal_condition.md)
- [\[NeurIPS 2025\] A Closer Look at Model Collapse: From a Generalization-to-Memorization Perspective](../../NeurIPS2025/image_generation/a_closer_look_at_model_collapse_from_a_generalization-to-memorization_perspectiv.md)
- [\[NeurIPS 2025\] Pairwise Optimal Transports for Training All-to-All Flow-Based Condition Transfer Model](../../NeurIPS2025/image_generation/pairwise_optimal_transports_for_training_all-to-all_flow-based_condition_transfe.md)
- [\[ICML 2025\] Tree-Sliced Wasserstein Distance: A Geometric Perspective](tree-sliced_wasserstein_distance_a_geometric_perspective.md)
- [\[CVPR 2025\] Make It Count: Text-to-Image Generation with an Accurate Number of Objects](../../CVPR2025/image_generation/make_it_count_text-to-image_generation_with_an_accurate_number_of_objects.md)

</div>

<!-- RELATED:END -->

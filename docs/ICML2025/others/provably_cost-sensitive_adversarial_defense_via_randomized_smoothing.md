---
title: >-
  [Paper Note] Provably Cost-Sensitive Adversarial Defense via Randomized Smoothing
description: >-
  [ICML2025][randomized smoothing] A "cost-sensitive certified radius" is proposed based on the randomized smoothing framework, achieving the first scalable cost-sensitive adversarial robustness certification and training for large models and high-dimensional data. This significantly improves robustness against high-cost misclassifications while maintaining overall accuracy.
tags:
  - "ICML2025"
  - "randomized smoothing"
  - "cost-sensitive robustness"
  - "certified defense"
  - "adversarial examples"
  - "cost matrix"
date: 2026-05-08
content_hash: 217eb3fda36defef
---

# Provably Cost-Sensitive Adversarial Defense via Randomized Smoothing

**Conference**: ICML2025  
**arXiv**: [2310.08732](https://arxiv.org/abs/2310.08732)  
**Code**: [AppleXY/Cost-Sensitive-RS](https://github.com/AppleXY/Cost-Sensitive-RS)  
**Area**: LLM Evaluation  
**Keywords**: randomized smoothing, cost-sensitive robustness, certified defense, adversarial examples, cost matrix

## TL;DR
A "cost-sensitive certified radius" is proposed based on the randomized smoothing framework, achieving the first scalable cost-sensitive adversarial robustness certification and training for large models and high-dimensional data. This significantly improves robustness against high-cost misclassifications while maintaining overall accuracy.

## Background & Motivation

Existing adversarial defense methods (empirical defenses such as adversarial training, and certified defenses such as randomized smoothing) assume that all misclassifications incur the same cost. However, in real-world scenarios, the costs of different misclassifications often vary dramatically:
- **Medical diagnosis**: Misclassifying a malignant tumor as benign is far more dangerous than the reverse.
- **Autonomous driving**: Misclassifying a pedestrian as background has far more severe consequences than misclassifying background as a pedestrian.

The only prior cost-sensitive certification method (Zhang & Evans, 2019) is based on convex relaxation and cannot scale to deep networks and large perturbation scenarios. Goal: Provide cost-sensitive robustness certification and training algorithms under the scalable framework of randomized smoothing.

## Method

### 1. Cost Matrix and Cost-Sensitive Setup

Define the cost matrix $\mathbf{C} \in \mathbb{R}_{\geq 0}^{m \times m}$, where $C_{jk} > 0$ represents that misclassifying class $j$ as class $k$ incurs a non-negligible cost. For a seed class $y$, its set of sensitive target classes is $\Omega_y = \{k \in [m] : C_{yk} > 0\}$.

### 2. Cost-Sensitive Certified Radius (Core Contribution)

Based on the certified radius of classical randomized smoothing, two new certified radii are proposed:

**Groupwise Cost-Sensitive Certified Radius**:

$$r_{\text{cs-group}}(\mathbf{x}; \Omega_y) = \frac{\sigma}{2}\left[\Phi^{-1}\left(\max_{k \in [m]} [h_\theta(\mathbf{x})]_k\right) - \Phi^{-1}\left(\max_{k \in \Omega_y} [h_\theta(\mathbf{x})]_k\right)\right]$$

**Pairwise Cost-Sensitive Certified Radius**:

$$r_{\text{cs-pair}}(\mathbf{x}; j) = \frac{\sigma}{2}\left[\Phi^{-1}\left(\max_{k \in [m]} [h_\theta(\mathbf{x})]_k\right) - \Phi^{-1}\left([h_\theta(\mathbf{x})]_j\right)\right]$$

where $h_\theta(\mathbf{x})$ is the prediction probability of the smoothed classifier for each class, and $\Phi^{-1}$ is the inverse CDF of the standard normal distribution.

**Key Theorem (Theorem 4.2)**: Under reasonable assumptions, $r_{\text{cs-pair}} \geq r_{\text{cs-group}} \geq r_{\text{standard}}$, which means the cost-sensitive certified radius is strictly no less than the standard certified radius. The advantage is more pronounced when $|\Omega_y|$ is smaller.

### 3. Monte Carlo-Based Certification Algorithm

Algorithm 1 (Certify_Group) and Algorithm 2 (Certify_Pair) are proposed:
- Estimate prediction probabilities of each class through Gaussian sampling.
- Compute the $(1-\alpha/2)$ lower confidence bound for $p_A$ (the primary class probability).
- Compute the $(1-\alpha/(2|\Omega_y|))$ upper confidence bound for $p_B$ (the sensitive target class probability) using the union bound.
- Finally return $\max(\hat{r}_{\text{std}}, \hat{r}_{\text{cs}})$ to ensure the tightest certificate.

### 4. Margin-CS Training Method

A training objective is designed to directly optimize the cost-sensitive certified radius:

$$\min_\theta \left\{ \mathbb{E}_{(\mathbf{x},y)} \mathcal{L}_{\text{CE}}(f_\theta(\mathbf{x}+\boldsymbol{\delta}), y) + \lambda_1 \mathbb{E}_{\mathcal{D}_n} \mathcal{L}_M(r_{\text{cs-group}}; 0, \gamma_1) + \lambda_2 \mathbb{E}_{\mathcal{D}_s} \sum_{j \in \Omega_y} C_{yj} \mathcal{L}_M(r_{\text{cs-pair}}; 0, \gamma_2) \right\}$$

- First term: Standard cross-entropy + Gaussian noise augmentation to maintain overall accuracy.
- Second term: Optimize the groupwise certified radius for non-sensitive samples.
- Third term: Optimize the pairwise certified radius weighted by cost for sensitive samples.
- Use hinge loss $\mathcal{L}_M$ to replace the non-differentiable certified radius, ensuring numerical stability.

## Key Experimental Results

### CIFAR-10 (ResNet-56, $\epsilon=0.5$, $\sigma=0.5$)

| Method | Acc(%) ↑ | Rob_cs(%) ↑ | Rob_cost ↓ |
|------|----------|-------------|------------|
| Gaussian | 65.4 | 22.3 | 4.99 |
| SmoothAdv | 66.9 | 27.1 | 4.94 |
| MACER | 65.9 | 27.3 | 5.27 |
| SmoothAdv-CS | 66.1 | 53.5 | 3.12 |
| **Margin-CS** | **67.5** | **54.8** | **3.04** |

Under the S-Pair setting, the Rob_cs of Margin-CS reaches **92.4%**, far exceeding the 50.4% of Gaussian.

### Imagenette (S-Seed Setup)

| Method | Acc(%) | Rob_cs(%) | Rob_cost |
|------|--------|-----------|----------|
| Gaussian | 80.3 | 64.6 | 3.67 |
| SmoothAdv-CS | 76.1 | 68.9 | 2.24 |
| **Margin-CS** | **79.6** | **81.1** | **1.35** |

### HAM10k Medical Dataset (Malignant/Benign Classification, Cost Ratio 10:1)

| Method | Acc(%) | Rob_cs(%) | Rob_cost | Precision(%) | Recall(%) |
|------|--------|-----------|----------|--------------|-----------|
| Gaussian | 82.9 | 11.8 | 1.56 | 51.0 | 15.0 |
| MACER | 82.7 | 21.1 | 1.41 | 50.0 | 25.0 |
| **Margin-CS** | **83.2** | **34.4** | **1.17** | **52.0** | **41.3** |

### Comparison with Zhang & Evans (2019) (CIFAR-10, $\sigma=0.25$)

| Method | Acc(%) | Rob_cs(%) |
|------|--------|-----------|
| Zhang & Evans | 61.2 | 92.4 |
| **Margin-CS** | **80.9** | **93.5** |

Margin-CS is nearly 20 percentage points higher in accuracy and also has superior robustness.

## Highlights & Insights

1. **First scalable cost-sensitive certification**: The only prior method (Zhang & Evans 2019) cannot handle large datasets and deep networks. This work overcomes this limitation based on randomized smoothing.
2. **Strict certified radius improvement**: It is theoretically proven that the cost-sensitive radius is $\geq$ the standard radius, and the improvement is more significant when $|\Omega_y|$ is smaller.
3. **Exquisite confidence bound construction**: The upper confidence bounds for sensitive classes are estimated individually via the union bound, rather than simply using $1-p_A$ as the upper bound for $p_B$, obtaining a tighter certificate.
4. **Validation in practical medical scenarios**: On HAM10k, the Recall is improved from the baseline of 15% to 41.3%, significantly reducing the missed diagnosis rate of malignant tumors.
5. **Flexible training framework**: Margin-CS finely controls the optimization of different data subgroups through group thresholds, achieving a better trade-off between accuracy and robustness.

## Limitations & Future Work

1. **Limited to $\ell_2$ norm**: Current certification only supports $\ell_2$ perturbations; extensions to other norms such as $\ell_\infty$ remain under-researched.
2. **Cost matrix requires prior knowledge**: In practical applications, the cost matrix needs to be specified by domain experts. How to automatically learn the cost matrix remains an open problem.
3. **Certified radius remains conservative**: Statistical errors introduced by Monte Carlo sampling make the empirical certified radius weaker than the theoretical value.
4. **Failure under large perturbations**: When $\epsilon > 1.5$, the certified rates of all methods drop sharply.
5. **Computational overhead**: Inference requires multiple Gaussian samplings (on the order of $n=10^5$), making it unsuitable for real-time deployment.

## Related Work & Insights

- **Randomized Smoothing** (Cohen et al., 2019): The foundation of the certification framework used in this work.
- **SmoothAdv** (Salman et al., 2019): Adversarial training + smoothing.
- **MACER** (Zhai et al., 2020): Directly optimizing the certified radius.
- **Zhang & Evans (2019)**: The only prior cost-sensitive certification method (convex relaxation, non-scalable).
- Insight: The cost-sensitive concept can be extended to other certification frameworks (such as IBP, CROWN), as well as multi-objective adversarial robustness.

## Rating
- Novelty: ⭐⭐⭐⭐ (Novel integration of cost-sensitivity and randomized smoothing, with solid theory)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Four datasets: CIFAR-10/Imagenette/ImageNet/HAM10k, comparison with multiple baselines, complete ablation studies)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, rigorous theoretical derivation, intuitive figures)
- Value: ⭐⭐⭐⭐ (Fills the gap in scalable cost-sensitive certification, of practical significance for high-risk scenarios like medical diagnosis)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] STRAP-ViT: Segregated Tokens with Randomized Transformations for Defense against Adversarial Patches in ViTs](../../CVPR2025/others/strap-vit_segregated_tokens_with_randomized_--_transformations_for_defense_again.md)
- [\[ICML 2025\] Randomized Dimensionality Reduction for Euclidean Maximization and Diversity Measures](randomized_dimensionality_reduction_for_euclidean_maximization_and_diversity_mea.md)
- [\[ICML 2025\] Efficient Optimization with Orthogonality Constraint: a Randomized Riemannian Submanifold Method](efficient_optimization_with_orthogonality_constraint_a_randomized_riemannian_sub.md)
- [\[ICML 2025\] Provably Improving Generalization of Few-Shot Models with Synthetic Data](provably_improving_generalization_of_few-shot_models_with_synthetic_data.md)
- [\[ICML 2025\] Practical Principles for AI Cost and Compute Accounting](practical_principles_for_ai_cost_and_compute_accounting.md)

</div>

<!-- RELATED:END -->

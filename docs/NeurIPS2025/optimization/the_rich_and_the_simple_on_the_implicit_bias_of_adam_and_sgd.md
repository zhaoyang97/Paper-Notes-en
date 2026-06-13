---
title: >-
  [Paper Note] The Rich and the Simple: On the Implicit Bias of Adam and SGD
description: >-
  [NeurIPS 2025][Optimization][Implicit bias] This paper provides theoretical and empirical evidence that neural networks trained with SGD tend to learn simple linear features (simplicity bias)…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "Implicit bias"
  - "Adam"
  - "SGD"
  - "simplicity bias"
  - "feature learning"
date: 2026-05-08
content_hash: cfa4365504912054
---

# The Rich and the Simple: On the Implicit Bias of Adam and SGD

**Conference**: NeurIPS 2025
**arXiv**: [2505.24022](https://arxiv.org/abs/2505.24022)  
**Code**: Unavailable  
**Area**: Optimization
**Keywords**: Implicit bias, Adam, SGD, simplicity bias, feature learning

## TL;DR

This paper provides theoretical and empirical evidence that neural networks trained with SGD tend to learn simple linear features (simplicity bias), whereas Adam produces richer nonlinear features, yielding predictors closer to the Bayes-optimal classifier and better generalization under distribution shift.

## Background & Motivation

Modern neural networks are heavily overparameterized, and the training process admits a large number of global optima. Different optimization algorithms are biased toward different optima—a phenomenon known as *implicit bias*. While substantial work has studied the implicit bias of gradient descent (GD), the implicit bias of Adam—the most widely used optimizer in deep learning—remains poorly understood.

In practice, neural networks trained with SGD are known to exhibit **simplicity bias**: a tendency to find simple solutions, such as relying on low-dimensional projections of the data for prediction. When spurious features are present, simplicity bias causes models to over-rely on these simple but non-causal features, leading to poor generalization under distribution shift.

The core question is: **Is Adam more resistant to simplicity bias? If so, what is the theoretical basis for this difference?**

The authors' starting point is to analyze the population gradients of a two-layer ReLU network trained with GD and Adam (including signGD) on a carefully constructed Gaussian mixture dataset, derive the asymptotic convergence direction of each neuron, and thereby precisely characterize the difference in the shape of the decision boundaries learned by each optimizer.

## Method

### Overall Architecture

The study employs a two-layer homogeneous ReLU network $f(\mathbf{W};\mathbf{x}) = \mathbf{a}\sigma(\mathbf{W}\mathbf{x})$, where the last layer is fixed and only the first-layer weights $\mathbf{W}$ are trained. Population gradient updates under GD and Adam are analyzed on a specially designed Gaussian mixture dataset.

### Key Designs

1. **Synthetic Dataset Construction**: Data are drawn from two classes. The first feature $x_1$ is discriminative for both classes (linearly separable), while the second feature $x_2$ is only discriminative between the two subclusters of the positive class (requiring a nonlinear boundary). The Bayes-optimal predictor is piecewise linear (nonlinear), meaning that using only $x_1$ for linear prediction is suboptimal. The parameter $\omega$ controls the degree of nonlinearity in the optimal boundary, and $\kappa$ controls anisotropy. The data satisfies Assumption 1 (achievability), ensuring that the optimal boundary passes through the origin.

2. **Closed-Form Population Gradient Derivation (Proposition 2)**: Under correlation loss, the paper derives a closed-form expression for the gradient of each neuron, involving the normal PDF $\phi$ and CDF $\Phi$. The gradient is a weighted combination of three directions $\bar{\mu}_+, \bar{\mu}_-, \bar{\mu}_0$, corresponding to the normalized means of different class clusters.

3. **Simplicity Bias of GD (Theorem 1)**: Under gradient flow (continuous-time GD), all neuron directions are shown to asymptotically converge to $a_k[1,0]$, meaning each neuron attends only to the first dimension and the learned decision boundary is linear. The key proof step constructs the angle $\theta_{k,t}$ (between the neuron and direction $[1,0]$) and shows that $a_k \frac{d\cos\theta_{k,t}}{dt} > C\frac{(\sin\theta_{k,t})^2}{\|\mathbf{w}_{k,t}\|}$, implying $\sin\theta_{k,t} \to 0$.

4. **Rich Feature Learning of Adam/signGD (Theorem 2)**: For Adam with $\beta_1=\beta_2=0$ (i.e., signGD), positive-class neurons ($a_k>0$) converge to $\frac{1}{\sqrt{2}}[1,1]$ or $\frac{1}{\sqrt{2}}[1,-1]$ depending on initialization angle, while negative-class neurons converge to $[-1,0]$. The key distinction is that signGD updates are the sign of the gradient, $[\text{sign}(a_k), \text{sign}(a_k\sin\theta_{k,t})]$, which are nonzero in both dimensions at every step, preventing loss of second-dimension information.

5. **Complete Analysis on Toy Data (Theorem 4)**: In the simplified setting $\sigma \to 0$, the paper fully characterizes the distribution of learned neuron directions for GD, signGD, and Adam with momentum. Adam with momentum ($\beta_1=\beta_2 \approx 1$) learns six directions (including an additional $\frac{1}{\sqrt{s^2+1}}[s,\pm 1]$ direction), producing a more nonlinear boundary.

### Test Error Advantage (Theorem 3)

Under the condition $\omega=\Theta(1)$, the test error of the piecewise linear predictor learned by Adam is strictly lower than that of the linear predictor learned by GD, theoretically establishing that Adam achieves better generalization both in-distribution and under certain distribution shifts.

## Key Experimental Results

### Main Results: Subpopulation Robustness Benchmarks (Fig. 2 & Table 13)

| Dataset | Metric (Worst-Group Acc.) | Adam | SGD | Gain |
|---|---|---|---|---|
| Waterbirds | Worst-Group Acc. | ~93% | ~85% | +8% |
| CelebA | Worst-Group Acc. | ~85% | ~47% | +38% |
| MultiNLI | Worst-Group Acc. | ~73% | ~68% | +5% |
| CivilComments | Worst-Group Acc. | ~70% | ~60% | +10% |

Across four standard spurious-correlation benchmark datasets, Adam consistently outperforms SGD on worst-group accuracy, with a particularly large margin on CelebA.

### Dominoes Dataset (Table 3)

| Configuration | Original Acc. | Core-Only Acc. | Decoded Acc. |
|---|---|---|---|
| SGD | 0.81±0.38 | 1.66±1.79 | 71.04±0.63 |
| Adam | **14.17±3.15** | **20.63±5.75** | **84.66±0.18** |

On the MNIST-CIFAR dataset (95% spurious correlation), Adam's worst-group accuracy is 17× higher than SGD's, and its decoded accuracy is 13.6 points higher, confirming that Adam learns substantially more core features.

### Boolean Features Dataset (Table 4)

| Configuration | Test Acc. | Decoded Core Corr. | Decoded Spurious Corr. |
|---|---|---|---|
| SGD | 89.58±1.92 | 0.51±0.08 | 0.78±0.08 |
| Adam | **97.87±0.69** | **0.87±0.03** | **0.36±0.06** |

Adam substantially increases correlation with core features (+0.36) and reduces correlation with spurious features (−0.42).

### Key Findings

- Under GD, all neurons align to a single direction $[\pm 1, 0]$, resulting in a linear decision boundary that uses only the first feature dimension.
- Under Adam, neurons converge to multiple directions, producing a piecewise linear decision boundary that exploits both signal dimensions.
- These theoretical findings hold under more realistic settings, including finite samples, logistic loss, and Adam with momentum.
- The margin distribution of Adam is broadly larger than that of SGD (Fig. 5).

## Highlights & Insights

- A simple yet insightful Gaussian mixture setup is proposed that precisely exposes the fundamental difference in the implicit biases of GD and Adam.
- The paper challenges the prevailing assumption that "SGD generally outperforms Adam on image data"—in the presence of spurious correlations, Adam generalizes better by learning richer features.
- The "equal-scale" update mechanism of signGD (taking the sign) is the key to avoiding simplicity bias: it does not neglect any dimension merely because its gradient is small.

## Limitations & Future Work

- The theoretical analysis is restricted to two-layer ReLU networks, fixed outer-layer weights, and simplified settings such as correlation loss.
- Simplicity bias is not universally harmful—simple features may generalize better in-distribution.
- The implicit bias of AdamW (Adam with weight decay) is not analyzed.
- Analysis for deeper networks and more complex architectures (e.g., Transformers) remains to be explored.

## Related Work & Insights

- The paper directly contrasts with Kalimeris et al. (2019), who showed "SGD learns linear features before nonlinear ones," and Shah et al. (2020) on simplicity bias.
- Kunstner et al. (2024) explain Adam's advantages through class imbalance; this paper provides a complementary explanation from the perspective of feature learning.
- The results offer theoretical guidance for optimizer selection in practice: when spurious correlations may be present in data, Adam should be preferred.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic theoretical analysis of the implicit bias difference between Adam and GD in nonlinear models
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated comprehensively on synthetic and 6 real-world datasets
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, tight integration of theory and experiments
- Value: ⭐⭐⭐⭐⭐ Provides deep insight into Adam's advantages with direct practical implications

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Unified Stability Analysis of SAM vs SGD: Role of Data Coherence and Emergence of Simplicity Bias](a_unified_stability_analysis_of_sam_vs_sgd_role_of_data_cohe.md)
- [\[ICML 2026\] The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks](../../ICML2026/optimization/the_implicit_bias_of_adam_and_muon_on_smooth_homogeneous_neural_networks.md)
- [\[NeurIPS 2025\] Implicit Bias of Spectral Descent and Muon on Multiclass Separable Data](implicit_bias_of_spectral_descent_and_muon_on_multiclass_separable_data.md)
- [\[NeurIPS 2025\] The Implicit Bias of Structured State Space Models Can Be Poisoned With Clean Labels](the_implicit_bias_of_structured_state_space_models_can_be_poisoned_with_clean_la.md)
- [\[NeurIPS 2025\] In Search of Adam's Secret Sauce](in_search_of_adams_secret_sauce.md)

</div>

<!-- RELATED:END -->

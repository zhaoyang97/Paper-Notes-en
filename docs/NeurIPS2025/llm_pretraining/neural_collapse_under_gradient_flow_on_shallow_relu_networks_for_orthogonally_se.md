---
title: >-
  [Paper Note] Neural Collapse under Gradient Flow on Shallow ReLU Networks for Orthogonally Separable Data
description: >-
  [NeurIPS 2025][LLM Pretraining][Neural Collapse] This paper provides the first provable convergence guarantee that gradient flow (GF) on two-layer ReLU networks with small initialization converges to a Neural Collapse (NC) solution on orthogonally separable data, revealing the critical role of GF's implicit bias—early neuron alignment followed by asymptotic maximum-margin bias—in driving the emergence of NC.
tags:
  - NeurIPS 2025
  - LLM Pretraining
  - Neural Collapse
  - Gradient Flow
  - ReLU Networks
  - Implicit Bias
  - Orthogonally Separable Data
  - Maximum Margin
date: 2026-05-08
content_hash: c71b5b79c217045a
---

# Neural Collapse under Gradient Flow on Shallow ReLU Networks for Orthogonally Separable Data

**Conference**: NeurIPS 2025
**arXiv**: [2510.21078](https://arxiv.org/abs/2510.21078)
**Authors**: Hancheng Min (Shanghai Jiao Tong University), Zhihui Zhu (Ohio State University), René Vidal (University of Pennsylvania)
**Code**: Not released
**Area**: LLM Pretraining
**Keywords**: Neural Collapse, Gradient Flow, ReLU Networks, Implicit Bias, Orthogonally Separable Data, Maximum Margin

## TL;DR

This paper provides the first provable convergence guarantee that gradient flow (GF) on two-layer ReLU networks with small initialization converges to a Neural Collapse (NC) solution on orthogonally separable data, revealing the critical role of GF's implicit bias—early neuron alignment followed by asymptotic maximum-margin bias—in driving the emergence of NC.

## Background & Motivation

### State of the Field
Neural Collapse (NC) is a striking phenomenon observed in the terminal phase of deep network training: last-layer features exhibit a simple geometric structure—(1) within-class features collapse to a single point; (2) class means are maximally separated (forming a simplex ETF); (3) classifier weights align with class means (self-duality). Theoretical understanding of NC is critical for explaining the success of deep learning.

### Limitations of Prior Work
- Existing theory primarily relies on the **Unconstrained Feature Model (UFM)**, which treats last-layer features as free optimization variables and proves that global optima exhibit NC. UFM, however, ignores the structure of input data and the effect of nonlinear activations.
- The few works addressing convergence require additional conditions: initialization close to a global optimum, explicit weight decay regularization, or extremely large network width.
- **Key gap**: The dynamics of *how* neural network training leads to NC remain largely unexplored, and the relationship between NC and the implicit bias of training algorithms is unclear.
- In practice, NC is observed even without explicit regularization, suggesting that implicit bias plays a central role.

### Root Cause
By analyzing the complete gradient flow dynamics of a two-layer ReLU network on orthogonally separable data, this work establishes an explicit connection between NC and GF's implicit bias, thereby: (1) relaxing the unconstrained assumption of UFM; (2) revealing the influence of data structure and ReLU nonlinearity on NC geometry; and (3) clarifying how implicit bias promotes the emergence of NC.

## Method

### Problem Setup
- **Data assumption (orthogonal separability)**: Within-class data are positively correlated (inner product $\geq \mu_s$); between-class data are negatively correlated (inner product $\leq -\mu_d$).
- **Network**: A two-layer ReLU network of width $h$, $f(\mathbf{x};\boldsymbol{\theta}) = \mathbf{V}\sigma(\mathbf{W}^\top \mathbf{x})$, without bias terms.
- **Training**: Gradient flow (GF), i.e., gradient descent with infinitesimally small step size; $\epsilon$-small and balanced initialization.
- **Loss**: Exponential/logistic loss for binary classification; cross-entropy for multiclass classification.

### Main Theorem (Theorem 1)
Under orthogonal separability and small initialization, GF provably converges to the following NC solution:

**1. Within-class directional collapse**: Last-layer features of same-class data collapse to a one-dimensional subspace (rather than a single point as in UFM): $\phi_{\bar{\theta}}(\mathbf{x}_i) = \langle s_k \mathbf{u}_k, \mathbf{x}_i \rangle \cdot \bar{\phi}_k$, where $\mathbf{u}_k$ is the maximum-margin direction for class $k$ and $s_k$ is determined by the margin ratio across classes. Feature directions are consistent but magnitudes may vary.

**2. Orthogonal class means**: Class-mean features are mutually orthogonal and non-negative: $\langle \bar{\phi}_k, \bar{\phi}_{k'} \rangle = 0$ ($k \neq k'$). This orthogonal structure arises from ReLU non-negativity (rather than a simplex ETF as in UFM), but reduces to a simplex ETF after subtracting the global mean.

**3. Projected self-duality**: Classifier weights align with projected class means. In the binary case, $\bar{\mathbf{V}} = s_+ \bar{\phi}_+^\top - s_- \bar{\phi}_-^\top$; in the multiclass case, $\bar{\mathbf{V}}$ forms a (scaled) simplex ETF.

### Proof Framework: Two-Phase Implicit Bias

The core contribution lies in decomposing the emergence of NC into two phases driven by GF's implicit bias:

**Phase 1: Alignment Phase (Early)—Between-Class Separation**

- Under small initialization, weight norms remain small for $\Theta(\log 1/\epsilon)$ time.
- In this phase, the directional dynamics of neuron weights approximately decouple; each neuron is attracted toward the mean direction of its corresponding class.
- Neurons positively correlated with a class align with that class's data while repelling others, ultimately achieving **orthogonal between-class feature separation**.
- Key result: There exists a time $T^*$ after which the inner product between features of different classes is identically zero.

**Phase 2: Asymptotic Phase (Late)—Within-Class Collapse**

- After between-class separation, the loss decouples into independent subproblems for each class.
- Each subproblem is equivalent to training a two-layer linear network on positively correlated data.
- Applying the maximum-margin bias result of Ji & Telgarsky (2021), weight directions are shown to converge to a rank-1 matrix.
- The rank-1 structure implies that last-layer features collapse to a one-dimensional subspace, realizing **within-class directional collapse and self-duality**.

### Technical Challenges in Multiclass Extension
- In binary classification, output weights $v_j$ are scalars (with fixed sign); in multiclass, $\mathbf{v}_j$ are $K$-dimensional vectors with nontrivial directional dynamics.
- The joint dynamics of input and output weight directions on $\mathbb{S}^{D-1} \times \mathbb{S}^{K-1}$ form a highly nonlinear Riemannian flow.
- This work introduces a "semi-local initialization" condition (Assumption 4) to construct an invariant subset of the basin of attraction.
- The maximum-margin result is extended from binary classification to multiclass cross-entropy loss—a significant technical contribution.

## Key Experimental Results

### Experiment 1: MNIST Digit Classification Validating Theorem 1

A two-layer ReLU network is trained on three MNIST digit classes {0, 1, 2} to verify the theoretical predictions of NC.

| Metric | Result | Notes |
|--------|--------|-------|
| Orthogonal separability | Approximately satisfied | Centered normalized correlation matrix shows positive within-class and negative between-class correlations |
| Output neuron weights | Align with pseudo-label directions | Verified by visualization |
| Input neuron weights | Closely resemble the corresponding class-average digit | Confirmed by grayscale image comparison |
| Raw data PCA (top 3 components) | Relative approximation error ~61% | Raw data distribution is dispersed |
| **Last-layer feature PCA (top 3 components)** | **Relative approximation error ~0.2%** | Features collapse nearly perfectly to a low-dimensional subspace, validating NC |
| Feature and classifier geometry | Orthogonal class means + projected self-duality | Consistent with Theorem 1 predictions |

### Experiment 2: Effect of Normalization Layers on NC (Modified ResNet18)

A modified ResNet18 (final linear classifier replaced by a two-layer ReLU classifier) is trained on MNIST and CIFAR10 under different normalization schemes. Each setting is repeated 5 times; mean and standard deviation are reported. Training runs for 50 epochs.

| Dataset | Normalization | NC1 (within-class collapse, lower is better) | NC2 (class-mean separation, lower is better) | NC3 (self-duality, lower is better) |
|---------|--------------|----------------------------------------------|----------------------------------------------|--------------------------------------|
| MNIST | None/Identity | Moderate | Moderate | Moderate |
| MNIST | LayerNorm | Slightly better than None | Slightly better than None | Slightly better than None |
| MNIST | **RMSNorm** | **Significantly best** | Near best | Near best |
| CIFAR10 | None/Identity | Moderate | Moderate | Moderate |
| CIFAR10 | LayerNorm | Slightly better than None | Slightly better than None | Slightly better than None |
| CIFAR10 | **RMSNorm** | **Significantly best** | Near best | Near best |

Key finding: RMSNorm significantly improves within-class directional collapse (NC1 metric), validating the corollary in Theorem 1 that normalization promotes stronger NC—since RMSNorm normalizes features to unit norm, directional collapse directly translates to single-point collapse.

## Highlights & Insights

- **First dynamic proof of NC**: Unlike prior static analyses of the optimization landscape, this work analyzes the full GF dynamics to prove convergence to NC, establishing an explicit bridge between NC and implicit bias.
- **Clear and elegant two-phase mechanism**: Early alignment phase (small initialization → neuron direction alignment → between-class separation) followed by late maximum-margin phase (weight direction convergence → rank-1 → within-class collapse), with an intuitive physical picture.
- **"Rectified NC" induced by ReLU**: The paper reveals that ReLU nonlinearity produces orthogonal class means (rather than a simplex ETF) and directional collapse (rather than single-point collapse), correcting the oversimplification of UFM and better reflecting practice.
- **Theoretical support for RMSNorm**: Theory predicts and experiments confirm that RMSNorm promotes stronger NC, providing theoretical guidance for normalization layer design.
- **Extension of implicit bias to multiclass**: The analysis of GF maximum-margin bias is extended from binary to multiclass cross-entropy loss, filling a gap in implicit bias theory.

## Limitations & Future Work

- **Strong orthogonal separability assumption**: Requiring positive within-class and negative between-class correlations is only approximately satisfied by real data.
- **Additional conditions needed for multiclass**: Proposition 1 requires extra data conditions (involving norm ratios and separation margins), a limitation of the analytical tools.
- **Semi-local initialization assumption (Assumption 4)**: Requires each neuron weight to be well-aligned with a (class-mean, pseudo-label) pair, which is more restrictive than random initialization.
- **Restricted to shallow ReLU networks**: Two-layer networks have limited expressive power; the NC dynamics of deep networks remain an open problem.
- **Gradient flow vs. gradient descent**: GF is a continuous-time limit; convergence guarantees for finite step-size GD are not addressed.
- **Bias-free networks**: Although extensions to networks with bias terms are discussed in the appendix, the main theorem does not directly cover this case.
- **Practical implications of NC metric**: The impact of directional collapse (as opposed to single-point collapse) on generalization in downstream applications remains unclear.

## Related Work & Insights

- **UFM series (Mixon et al. 2022, Zhu et al. 2021, Zhou et al. 2022, etc.)**: Analyze global optima of the unconstrained feature model; this work advances to a constrained realistic network setting, revealing the influence of data and ReLU on NC geometry.
- **Tirer & Bruna (2022)**: Introduces ReLU constraints into UFM and derives orthogonal class means consistent with this work, but remains a static landscape analysis.
- **Jacot et al. (2025)**: Proves convergence to NC for wide networks with weight decay; this work requires neither explicit regularization nor overparameterization by width.
- **Phuong & Lampert (2021), Min et al. (2024)**: Study neuron alignment dynamics in shallow ReLU networks without addressing NC; this work builds on their results to connect alignment with NC.
- **Ji & Telgarsky (2019, 2020)**: Study maximum-margin bias in deep linear networks; this work applies their results to the per-class subproblems after between-class separation.
- **Hong & Ling (2024)**: Study NC in shallow networks with general data, but do not analyze dynamic convergence.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First proof of NC from a training dynamics perspective, establishing an explicit connection between NC and implicit bias.
- Experimental Thoroughness: ⭐⭐⭐ — MNIST validation is clear and ResNet experiments are informative, but coverage of scales and settings is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ — The progressive development from binary to multiclass is exceptionally clear, balancing rigorous theory with intuition.
- Value: ⭐⭐⭐⭐ — Theoretical contributions are outstanding and significantly advance understanding of NC mechanisms, though strong assumptions limit direct applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Alternating Gradient Flows: A Theory of Feature Learning in Two-layer Neural Networks](alternating_gradient_flows_a_theory_of_feature_learning_in_two-layer_neural_netw.md)
- [\[NeurIPS 2025\] Flatness is Necessary, Neural Collapse is Not: Rethinking Generalization via Grokking](flatness_is_necessary_neural_collapse_is_not_rethinking_generalization_via_grokk.md)
- [\[NeurIPS 2025\] Generalization Bounds for Rank-sparse Neural Networks](generalization_bounds_for_rank-sparse_neural_networks.md)
- [\[NeurIPS 2025\] Learning to Flow from Generative Pretext Tasks for Neural Architecture Encoding](learning_to_flow_from_generative_pretext_tasks_for_neural_architecture_encoding.md)
- [\[ICLR 2026\] Intrinsic Training Dynamics of Deep Neural Networks](../../ICLR2026/llm_pretraining/intrinsic_training_dynamics_of_deep_neural_networks.md)

</div>

<!-- RELATED:END -->

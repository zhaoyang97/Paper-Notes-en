---
title: >-
  [Paper Note] Contextures: Representations from Contexts
description: >-
  [ICML 2025][Self-Supervised Learning][Representation Learning Theory] Establishes the contexture theory to unify and prove that various representation learning paradigms, including supervised learning, self-supervised learning, and manifold learning, can be understood as learning the top-$d$ singular functions of the expectation operator induced by context variables, while revealing the law of diminishing marginal returns in model scaling and proposing context quality evaluat…
tags:
  - "ICML 2025"
  - "Self-Supervised Learning"
  - "Representation Learning Theory"
  - "contexture"
  - "singular functions"
  - "expectation operator"
  - "context variable"
  - "scaling laws"
date: 2026-05-08
content_hash: b2c53d780edad31d
---

# Contextures: Representations from Contexts

**Conference**: ICML 2025

**arXiv**: [2505.01557](https://arxiv.org/abs/2505.01557)

**Authors**: Runtian Zhai, Kai Yang, Che-Ping Tsai, Burak Varici, Zico Kolter, Pradeep Ravikumar (CMU)

**Area**: Self-Supervised Learning

**Keywords**: Representation Learning Theory, contexture, singular functions, expectation operator, context variable, scaling laws

## TL;DR

Establishes the contexture theory to unify and prove that various representation learning paradigms, including supervised learning, self-supervised learning, and manifold learning, can be understood as learning the top-$d$ singular functions of the expectation operator induced by context variables, while revealing the law of diminishing marginal returns in model scaling and proposing context quality evaluation metrics.

## Background & Motivation

Foundation models have achieved great success in practice, but a systematic understanding of the **representations** learned by these models remains elusive. Core questions include:

**What representations are learned?** What is the common essence of the representations learned under different training paradigms (supervised, self-supervised, manifold learning)?

**Why are they useful?** Why do learned representations transfer to various downstream tasks?

**Limits of scaling?** Does increasing model scale always improve performance? When do diminishing marginal returns set in?

Existing theories are typically confined to analyzing specific paradigms (e.g., spectral theory for contrastive learning) and lack a unified perspective across paradigms. Prior works (e.g., HaoChen et al., 2021; Zhai et al., 2024) linked self-supervised learning with the spectrum of augmentation graphs but did not extend to supervised and manifold learning.

**Goal**: Establish a sufficiently general theoretical framework to unify and explain the essence of various representation learning methods, and derive practical insights from it.

## Method

### Core Concept: Contexture

**Context Variable**: Given an input $X$, a context variable $A$ is defined to capture information relevant to $X$:
- In supervised learning: $A = Y$ (labels)
- In self-supervised learning: $A = X'$ (augmented views)
- In manifold learning: $A$ is the $K$-nearest neighbors of $X$

**Expectation Operator**: The operator $T_{P^+}$ induced by the joint distribution $P(X, A)$ of $(X, A)$:

$$T_{P^+} f(a) = \mathbb{E}[f(X) | A = a]$$

**Contexture Definition**: A representation $\Phi: \mathcal{X} \to \mathbb{R}^d$ is a contexture if it approximates the top-$d$ singular functions $\{\mu_1, \mu_2, \ldots, \mu_d\}$ of the expectation operator $T_{P^+}$:

$$T_{P^+}^* T_{P^+} \mu_i = s_i^2 \mu_i$$

where $s_1 \geq s_2 \geq \cdots$ are the singular values, capturing the strength of association between the input and the context.

### Proof of Unification

The paper proves that the following methods are all learning contexture:

| Learning Paradigm | Context Variable $A$ | Corresponding Method |
|---------|-------------|---------|
| Supervised Learning | Class label $Y$ | Classification loss $\to$ top singular functions |
| Self-Supervised (Contrastive) | Augmented view $X'$ | SimCLR, spectral contrastive loss |
| Self-Supervised (Non-contrastive) | Augmented view $X'$ | VICReg, Barlow Twins |
| Self-Supervised (Masked) | Visible patch | MAE |
| Manifold Learning | $K$-nearest neighbors | Laplacian Eigenmaps, PCA |

Key Unification Theorem: The objective functions of these methods, at their optimal solutions, are equivalent to extracting the top-$d$ singular functions of $T_{P^+}$.

### Optimality Theorem

**Theorem (Downstream Optimality)**: If the downstream task $(X, Y^*)$ is compatible with the context $(X, A)$, then the $d$-dimensional representation $\Phi$ that learns the contexture is optimal for that task.

Compatibility condition: The task label $Y^*$ can be expressed by the information in the context $A$, meaning $\mathbb{E}[Y^*|X]$ lies in the span of the top-$d$ singular functions.

### Theoretical Explanation of Scaling Laws

**Core Corollary**: Once the model capacity is sufficient to approximate the top-$d$ singular functions, the benefits of further scaling the model dimensions exhibit **diminishing marginal returns**.

$$\text{Model Scaling} \to \Phi \text{ Approximates top-}d\text{ singular functions} \to \text{Performance Saturation}$$

Therefore, **further improvements require better context** (such as better data augmentation strategies) instead of larger models.

### Context Quality Evaluation Metrics

A **task-agnostic metric** $\tau_d$ is proposed to predict downstream performance solely based on the singular value spectrum of the expectation operator:

$$\tau_d = \sum_{i > d} s_i^2$$

Intuition: $\tau_d$ measures the amount of information "missed" by the $d$-dimensional representation. A smaller $\tau_d$ indicates that the context is more beneficial for learning the $d$-dimensional encoder.

**Spectral Estimation Method**: Efficiently estimate the singular value spectrum using methods like Kernel PCA or NeuralEF/NeuralSVD without actually training the encoder.

## Key Experimental Results

### Context-Task Compatibility Verification (OpenML Datasets)

| Dataset | # Samples | Features | Correlation between KNN context $\tau$ and true error |
|--------|-------|------|--------------------------------|
| cpu_act | 8,192 | 21 | High correlation |
| pol | 15,000 | 26 | High correlation |
| elevators | 16,599 | 16 | High correlation |
| wine_quality | 6,497 | 11 | High correlation (failure case)|
| yprop_4_1 | 8,885 | 42 | Moderate correlation |

### Predictive Power of $\tau_d$ Metric (Table 1, summary of 11 datasets)

For most datasets (9 out of 11), the $\tau_d$ metric shows a strong positive correlation with the downstream linear probing error of the encoder, indicating that the metric can effectively predict context quality.

Failure analysis:
- **Case 1**: When context correlation is too strong or too weak, $\tau$ might misjudge
- **Case 2**: The metric cannot be compared across different context types (e.g., augmentation vs. nearest neighbors)

### Large-Scale Data Validation (Rebuttal Addition)

| Dataset | Encoder | Context | Results |
|--------|-------|--------|------|
| MNIST | LeNet | Random Crop (different ratios) | $\tau_d$ is strongly correlated with error |
| CIFAR-10 | ResNet-18 | SimCLR augmentation | $\tau_d$ accurately tracks actual error |

### Scaling Experiments (with varying model widths/depths)

Using $d = 128$-dimensional representations, the CCA alignment is generally $>0.8$, reaching up to $\approx 0.9$, validating that larger models indeed get closer to the top singular functions. However, when the model is excessively large, optimization becomes harder, which in turn reduces the alignment degree.

## Highlights & Insights

1. **Elegant Unification**: Unifies supervised, self-supervised, and manifold learning using a single mathematical object (the singular functions of the expectation operator), which is theoretically beautiful.
2. **Theoretical Limits of Scaling**: Provides a theoretical basis for the "model is large enough" regime, suggesting that future investments should be directed toward better context design (e.g., data augmentation, labeling strategies) rather than blindly scaling model sizes.
3. **Task-Agnostic Metric**: $\tau_d$ depends solely on spectral information, allowing context quality assessment without downstream labels, which offers practical guidance for hyperparameter selection (such as crop ratio or mask ratio).
4. **Explaining the Platonic Representation Hypothesis**: Explains why SSL and CLIP learn highly aligned representations—because their contexts essentially share the same top singular functions.

## Limitations & Future Work

1. **Small Experimental Scale**: The main experiments are validated only on small OpenML datasets (maximum ~28K samples), lacking standard large-scale validation on datasets like ImageNet.
2. **Limited to Linear Downstream**: Theoretical analysis assumes downstream tasks use linear probing, leaving non-linear fine-tuning unaddressed.
3. **Reliance on Optimization Assumptions**: The theoretical analysis focuses on the local minima of the objective functions, whereas in practice, models oscillate at the "edge of stability," leaving a gap between theory and practice.
4. **No Analysis of Architectural Inductive Biases**: The influence of inductive biases, such as the translation invariance of CNNs, is not incorporated into the theoretical framework.
5. **Distribution Shift**: Assumes that upstream and downstream data are in-distribution; real-world distribution shift effects are not considered.

## Related Work & Insights

- **HaoChen et al. (2021)**: Links contrastive learning with the spectrum of augmentation graphs; ours extends this to all SSL methods and broader learning paradigms.
- **Zhai et al. (2024)**: Extends spectral theory from contrastive learning to all SSL; ours further extends it to all representation learning.
- **Balestriero & LeCun (2022)**: Proves that contrastive and non-contrastive SSL recover the top eigenfunctions of spectral embedding methods.
- **Huh et al. (2024) Platonic Representation Hypothesis**: Discovers that different models learn highly aligned representations; the contexture theory provides an explanation.
- This framework may inspire: (a) automatic search of augmentation strategies based on spectral analysis; (b) predicting optimal model size prior to training; (c) unified evaluation standards for different pre-training methods.

## Rating

| Dimension | Score (1-5) | Explanation |
|------|-----------|------|
| Novelty | 5 | Unified theoretical framework for multiple paradigms with elegant concepts |
| Theoretical Depth | 5 | Rigorous mathematical derivations with broad coverage |
| Experimental Thoroughness | 2 | Datasets are too small, lacking validation on standard benchmarks |
| Writing Quality | 4 | Clear theoretical exposition, but the coverage is excessively broad |
| Value | 3 | The $\tau_d$ metric has potential but its utility requires large-scale validation |
| **Overall** | **3.8** | A theoretically beautiful paper but lacks sufficient experimental support |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Contrastive Representations for Temporal Reasoning](../../NeurIPS2025/self_supervised/contrastive_representations_for_temporal_reasoning.md)
- [\[ICLR 2026\] Mechanistic Independence: A Principle for Identifiable Disentangled Representations](../../ICLR2026/self_supervised/mechanistic_independence_a_principle_for_identifiable_disentangled_representatio.md)
- [\[ICLR 2026\] OrthoRF: Exploring Orthogonality in Object-Centric Representations](../../ICLR2026/self_supervised/orthorf_exploring_orthogonality_in_object-centric_representations.md)
- [\[ICLR 2026\] A Bayesian Nonparametric Framework for Learning Disentangled Representations](../../ICLR2026/self_supervised/a_bayesian_nonparametric_framework_for_learning_disentangled_representations.md)
- [\[ICLR 2026\] Self-Predictive Representations for Combinatorial Generalization in Behavioral Cloning](../../ICLR2026/self_supervised/self-predictive_representations_for_combinatorial_generalization_in_behavioral_c.md)

</div>

<!-- RELATED:END -->

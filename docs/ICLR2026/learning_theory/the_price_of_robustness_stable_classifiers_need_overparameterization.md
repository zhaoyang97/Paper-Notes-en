---
title: >-
  [Paper Note] The Price of Robustness: Stable Classifiers Need Overparameterization
description: >-
  [ICLR 2026][overparameterization] This paper establishes stability-generalization bounds for discontinuous classifiers and proves a "law of robustness" for classification: any interpolating classifier with $p \approx n$…
tags:
  - "ICLR 2026"
  - "overparameterization"
  - "robustness"
  - "stability"
  - "classifier"
  - "generalization bound"
  - "margin"
date: 2026-05-08
content_hash: 8cc20f0054822b5a
---

# The Price of Robustness: Stable Classifiers Need Overparameterization

**Conference**: ICLR 2026
**arXiv**: [2603.02806](https://arxiv.org/abs/2603.02806)  
**Code**: None  
**Area**: Learning Theory / Generalization Theory
**Keywords**: overparameterization, robustness, stability, classifier, generalization bound, margin

## TL;DR

This paper establishes stability-generalization bounds for discontinuous classifiers and proves a "law of robustness" for classification: any interpolating classifier with $p \approx n$ parameters is necessarily unstable, and achieving high stability requires overparameterization on the order of $p \approx nd$.

## Background & Motivation

- **The overparameterization paradox**: Classical learning theory holds that more parameters lead to worse overfitting, yet modern overparameterized neural networks generalize well—a phenomenon known as double descent.
- **Bubeck & Sellke 2021 robustness law**: This work proved a smoothness–overparameterization trade-off for Lipschitz-continuous functions in the regression setting. However, the result relies on the Lipschitz assumption and does not apply to classifiers, whose discrete outputs are inherently discontinuous.
- **Empirical findings**: A large-scale study (Jiang et al. 2019) found that among 40+ complexity measures, **margin** (distance to the decision boundary) correlates most consistently with generalization, rather than norm-based measures.
- **Core Problem**: How can one build a stability-driven generalization theory for discontinuous classifiers?

## Method

### 1. Class-Stability Definition

**Definition 1** (Margin and Class-Stability): For a classifier $f: \mathcal{X} \to \{-1, 1\}$:

Unsigned margin:
$$h_f(x) := |d_f(x)| = \inf\{\|x - z\|_2 : f(z) \neq f(x), z \in \mathcal{X}\}$$

Class-stability (expected margin):
$$S(f) := \mathbb{E}[h_f]$$

This measures the average robustness of the classifier's predictions to input perturbations under the data distribution.

### 2. Isoperimetric Assumption

The data distribution $\mu$ is assumed to satisfy $c$-isoperimetry: for any bounded $L$-Lipschitz function $f$ and $t \geq 0$:

$$\mathbb{P}(|f(x) - \mathbb{E}[f]| \geq t) \leq 2 e^{-\frac{dt^2}{2cL^2}}$$

Gaussian distributions and the uniform distribution on the sphere satisfy this condition. Under the manifold hypothesis, $d$ can be interpreted as the intrinsic manifold dimension.

### 3. Rademacher Bounds for Finite Function Classes

**Theorem 4**: Assume $\min_{f \in \mathcal{F}} S(f) > S > 0$ and $\log|\mathcal{F}| \geq n$:

$$\mathcal{R}_{n,\mu}(\mathcal{F}) \leq K_1 \max\left\{\frac{1}{\sqrt{n}}, \frac{\sqrt{c}}{S} \cdot \frac{\log|\mathcal{F}|}{n\sqrt{d}}\right\}$$

Under regularity conditions, this can be improved to:

$$\mathcal{R}_{n,\mu}(\mathcal{F}) \leq K_2 \max\left\{\frac{1}{\sqrt{n}}, \frac{\sqrt{c}}{S}\sqrt{\frac{\log|\mathcal{F}|}{nd}}, 2\exp\left(-\frac{dS^2}{8c}\right)\right\}$$

**Key Insight**: The factor $1/S$ appears in front of $\sqrt{\log|\mathcal{F}|}$—stability reduces the effective complexity of the model class.

### 4. Law of Robustness for Classification

**Corollary 6**: Let $p := \log|\mathcal{F}| \geq n$. Under appropriate conditions, with high probability:

$$\hat{R}_{\text{0-1}}(f) \leq R^* - \varepsilon \implies S(f) < \max\left\{\frac{3K}{\varepsilon}\sqrt{\frac{c\log|\mathcal{F}|}{nd}}, \sqrt{\frac{8c}{d}\log\frac{6K}{\varepsilon}}\right\}$$

**Implications**:
- When $p \approx n$, any interpolating classifier is necessarily unstable.
- Simultaneously achieving low training error and high stability requires overparameterization of order $p \approx nd$.

### 5. Extension to Infinite Function Classes

The paper introduces **normalized co-stability**: the output margin of the score function $g_w$ in a classifier $f = \text{sgn} \circ g_w$ is normalized accordingly. Combined with Lipschitz continuity of $g_w$ with respect to both parameters and inputs, corresponding generalization bounds (Theorem 13) and a law of robustness (Corollary 15) are derived.

## Key Experimental Results

### MNIST Experiments

| Network Width | Test Accuracy | Class-Stability $S(f)$ | Spectral Norm |
|--------------|--------------|------------------------|---------------|
| Small | Low | Low | Irregular |
| Medium | Moderate | Moderate | Irregular |
| Large | High | **High** | Irregular |

### CIFAR-10 Experiments

| Network Width | Test Accuracy | Normalized Co-Stability | Spectral Norm |
|--------------|--------------|------------------------|---------------|
| Narrow | ~70% | Low | Inconsistent |
| Wide | ~85% | **High** | Inconsistent |
| Wider | ~90% | **Higher** | Inconsistent |

### Key Findings

- Both stability and normalized co-stability **monotonically increase** with network width.
- Stability is **positively correlated** with test performance.
- Traditional norm-based measures (e.g., spectral norm) show **no systematic association** with generalization.
- Results validate the theoretical prediction: overparameterization → high stability → good generalization.

## Highlights & Insights

1. **Extension to discontinuous functions**: This is the first work to extend the law of robustness from Lipschitz regression to discontinuous classifiers.
2. **Direct analysis of the 0-1 loss**: No Lipschitz loss assumption is required.
3. **Explaining overparameterization**: Overparameterization is not the source of overfitting but rather a necessary condition for achieving robustness.
4. **Broad applicability**: The framework covers inherently discontinuous models such as quantized neural networks and spiking neural networks.
5. **Special relevance to Transformers**: Self-attention is generally not Lipschitz-continuous, making this framework more applicable than Lipschitz-based ones.

## Limitations & Future Work

- The isoperimetric assumption may not hold for certain data distributions.
- Extending from finite to infinite function classes requires additional Lipschitz assumptions on the parameters.
- Generalization bounds may still be vacuous in practice, consistent with the critique of Nagarajan & Kolter 2021.
- The practical value of the theoretical dimension $d$ is difficult to estimate precisely (extrinsic vs. intrinsic dimension).
- No direct connection to optimization dynamics (e.g., implicit bias) is established.

## Related Work & Insights

- **Law of robustness**: Bubeck & Sellke 2021 (Lipschitz regression)
- **Margin-based generalization bounds**: Bartlett et al. 2017 (spectrally-normalized margin)
- **Algorithmic stability**: Bousquet & Elisseeff 2002
- **Double descent**: Belkin et al. 2019

## Rating

- **Novelty**: ⭐⭐⭐⭐ — A natural and important extension of the law of robustness to classification.
- **Technical Depth**: ⭐⭐⭐⭐⭐ — Proof techniques are elegant, leveraging Lipschitz surrogates and signed distance representations.
- **Experimental Thoroughness**: ⭐⭐⭐ — Validated on MNIST and CIFAR-10; primarily qualitative.
- **Value**: ⭐⭐⭐⭐ — Provides theoretical grounding for understanding generalization in overparameterized models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Evaluating GFlowNet from Partial Episodes for Stable and Flexible Policy-Based Training](evaluating_gflownet_from_partial_episodes_for_stable_and_flexible_policy-based_t.md)
- [\[ICLR 2026\] Fast and Stable Riemannian Metrics on SPD Manifolds via Cholesky Product Geometry](fast_and_stable_riemannian_metrics_on_spd_manifolds_via_cholesky_product_geometr.md)
- [\[ICML 2026\] Towards Optimal Robustness in Learning-Augmented Paging](../../ICML2026/others/towards_optimal_robustness_in_learning-augmented_paging.md)
- [\[ICLR 2026\] LipNeXt: Scaling up Lipschitz-based Certified Robustness to Billion-parameter Models](lipnext_scaling_up_lipschitz-based_certified_robustness_to_billion-parameter_mod.md)
- [\[ICLR 2026\] Key and Value Weights Are Probably All You Need: On the Necessity of the Query, Key, and Value Weight Triplet in Self-Attention](key_and_value_weights_are_probably_all_you_need_on_the_necessity_of_the_query_ke.md)

</div>

<!-- RELATED:END -->

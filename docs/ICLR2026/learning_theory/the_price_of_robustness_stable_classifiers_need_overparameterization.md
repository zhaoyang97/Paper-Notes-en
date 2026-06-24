---
title: >-
  [Paper Note] The Price of Robustness: Stable Classifiers Need Overparameterization
description: >-
  [ICLR 2026][Learning Theory][Overparameterization] The authors establish stability-generalization bounds for discontinuous classifiers, proving a "law of robustness cost" in classification tasks: any interpolating classifier with parameter count $p \approx n$ must be unstable; achieving high stability requires overparameterization of magnitude $p \approx nd$.
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "Generalization Theory"
  - "Overparameterization"
  - "robustness"
  - "stability"
  - "classifier"
  - "generalization bounds"
  - "margin"
date: 2026-05-08
content_hash: 186be89ab32b516f
---

# The Price of Robustness: Stable Classifiers Need Overparameterization

**Conference**: ICLR 2026  
**arXiv**: [2603.02806](https://arxiv.org/abs/2603.02806)  
**Code**: None  
**Area**: Learning Theory / Generalization Theory  
**Keywords**: Overparameterization, robustness, stability, classifier, generalization bounds, margin

## TL;DR

The authors establish stability-generalization bounds for discontinuous classifiers, proving a "law of robustness cost" in classification tasks: any interpolating classifier with parameter count $p \approx n$ must be unstable; achieving high stability requires overparameterization of magnitude $p \approx nd$.

## Background & Motivation

- **Paradox of overparameterization**: While classical learning theory suggests that more parameters lead to more severe overfitting, modern neural networks generalize better when overparameterized (the double descent phenomenon).
- **Robustness law by Bubeck & Sellke 2021**: This work proved a smoothness-overparameterization trade-off for Lipschitz continuous functions in regression scenarios. However, the result relies on Lipschitz assumptions and does not apply to classifiers (which have discrete outputs and are naturally discontinuous).
- **Empirical findings**: Large-scale studies (Jiang et al. 2019) show that among 40+ complexity measures, the one most consistently correlated with generalization is **margin** (distance to the decision boundary), rather than norm-based measures.
- **Core Problem**: How can a stability-driven generalization theory be established for discontinuous classifiers?

## Method

### Overall Architecture

This is a pure theory paper centered around a logical derivation chain: first, the robustness of a classifier is quantified into an analyzable scalar—class stability $S(f)$—using "expected margin"; then, leveraging the isoperimetry hypothesis of the data distribution, this quantity is integrated into an upper bound for Rademacher complexity. This proves that stability reduces the effective complexity of the model class, leading to the "law of robustness cost": to allow a classifier that interpolates the training set to maintain high stability, the parameter count must increase from $p \approx n$ to the order of $p \approx nd$. Finally, these conclusions for finite function classes are extended to infinite function classes, such as neural networks, by utilizing the Lipschitz continuity of the score function.

### Key Designs

**1. Class Stability: Converting "discontinuity" into an analyzable scalar using expected margin**

Since classifiers output discrete labels and are inherently discontinuous, the Lipschitz smoothness framework used in regression cannot be directly applied. Instead, the authors characterize robustness using the distance from samples to the decision boundary. For a classifier $f: \mathcal{X} \to \{-1, 1\}$, the unsigned margin is defined as the Euclidean distance to the nearest point of a different class $h_f(x) := |d_f(x)| = \inf\{\|x - z\|_2 : f(z) \neq f(x), z \in \mathcal{X}\}$ (Definition 1), measuring the minimum perturbation needed to push $x$ across the decision boundary. Taking its expectation under the data distribution yields the class stability $S(f) := \mathbb{E}[h_f]$. $S(f)$ corresponds to the "average robustness of the classifier against input perturbations." Unlike the minimum robustness radius used by Sokolic et al., this uses the **expectation** (average distance), which is a continuous scalar that integrates cleanly into subsequent complexity analyses.

**2. Isoperimetry Hypothesis: Injecting high-dimensional concentration into generalization bounds**

For the margin to effectively constrain generalization, the geometry of the data in high-dimensional space must be characterized; otherwise, stable functions could still fit random labels arbitrarily. The authors assume the distribution $\mu$ satisfies $c$-isoperimetry: for any bounded $L$-Lipschitz function $f$ and $t \geq 0$, $\mathbb{P}(|f(x) - \mathbb{E}[f]| \geq t) \leq 2 e^{-\frac{dt^2}{2cL^2}}$ (Definition 3). Gaussian distributions and uniform measures on manifolds with positive curvature (e.g., spheres) satisfy this condition. It expresses high-dimensional measure concentration—function values concentrate around the mean at an exponential rate, controlled by the dimension $d$. Under the manifold hypothesis, $d$ represents the intrinsic manifold dimension of the data, which is the source of $d$ in the $nd$ magnitude.

**3. Stability reducing complexity to derive the law of robustness cost**

This is the technical core of the theory, proceeding in two steps. The first step proves that stability directly reduces complexity: under the conditions $\min_{f \in \mathcal{F}} S(f) > S > 0$ and $\log|\mathcal{F}| \geq n$, Theorem 4 provides $\mathcal{R}_{n,\mu}(\mathcal{F}) \leq K_1 \max\left\{\frac{1}{\sqrt{n}}, \frac{\sqrt{c}}{S} \cdot \frac{\log|\mathcal{F}|}{n\sqrt{d}}\right\}$, which is further refined under regular conditions to $\mathcal{R}_{n,\mu}(\mathcal{F}) \leq K_2 \max\left\{\frac{1}{\sqrt{n}}, \frac{\sqrt{c}}{S}\sqrt{\frac{\log|\mathcal{F}|}{nd}}, 2\exp\left(-\frac{dS^2}{8c}\right)\right\}$. Crucially, $1/S$ appears before $\sqrt{\log|\mathcal{F}|}$—higher stability results in a lower complexity term. Thus, a model class that appears large can have its effective complexity significantly reduced if all functions within it are stable (note that under discretization, $\log|\mathcal{F}| \in \mathcal{O}(p)$, so $\log|\mathcal{F}|$ represents the parameter count $p$). The second step uses this bound **in reverse** to force a lower bound on parameters: setting $p := \log|\mathcal{F}| \geq n$, Corollary 6 gives $\hat{R}_{\text{0-1}}(f) \leq R^* - \varepsilon \implies S(f) < \max\left\{\frac{3K}{\varepsilon}\sqrt{\frac{c\log|\mathcal{F}|}{nd}}, \sqrt{\frac{8c}{d}\log\frac{6K}{\varepsilon}}\right\}$. This implication explicitly states the cost: when $p \approx n$, the stability upper bound on the right is forced to be very small; any classifier interpolating the training set is destined to be unstable. The only way to achieve both low training error and high stability is to increase the parameter count to $p \approx nd$.

**4. Generalization to infinite function classes: Bridging neural networks via normalized co-stability**

The above conclusions apply to finite function classes ($\log|\mathcal{F}|$ is finite), necessitating a bridge to neural networks, which are continuously parameterized infinite classes. Furthermore, class stability alone is insufficient—Example 9 shows that a high $S(f)$ does not prevent the score function itself from being discontinuous. Consequently, the authors express the classifier as $f = \text{sgn} \circ g_w$ and focus on the **margin in the output space (codomain)**. By normalizing the output margin of the score function $g_w$ and taking its expectation, they define **normalized co-stability** (Definition 10), a stronger robustness measure than class stability. Combined with the Lipschitz continuity of $g_w$ with respect to parameters $w$ and input $x$, they control the complexity of the infinite class to derive corresponding generalization bounds (Theorem 13) and the robustness law (Corollary 15). Notably, this only requires the score function $g_w$ to be continuous, while the final classifier $\text{sgn} \circ g_w$ can remain discontinuous, allowing the framework to cover quantized networks, spiking networks, and even self-attention mechanisms that do not satisfy global Lipschitz continuity.

## Key Experimental Results

### MNIST Results

| Network Width | Test Accuracy | Class Stability $S(f)$ | Spectral Norm |
| :--- | :--- | :--- | :--- |
| Small | Lower | Low | Irregular |
| Medium | Medium | Medium | Irregular |
| Large | High | **High** | Irregular |

### CIFAR-10 Results

| Network Width | Test Accuracy | Normalized co-stability | Spectral Norm |
| :--- | :--- | :--- | :--- |
| Narrow | ~70% | Low | Inconsistent |
| Wide | ~85% | **High** | Inconsistent |
| Wider | ~90% | **Higher** | Inconsistent |

### Key Findings

- Both stability and normalized co-stability **increase monotonically** with network width.
- Stability is **positively correlated** with test performance.
- Traditional norm measures (such as spectral norm) show **no systematic correlation** with generalization.
- The experiments validate the theoretical prediction: overparameterization $\to$ high stability $\to$ good generalization.

## Highlights & Insights

1.  **Extension to discontinuous functions**: For the first time, the law of robustness is extended from Lipschitz regression to discontinuous classifiers.
2.  **Direct analysis of 0-1 loss**: No Lipschitz loss assumption is required.
3.  **Explaining overparameterization**: Overparameterization is not the source of overfitting but a necessary condition for achieving robustness.
4.  **Broad applicability**: Covers naturally discontinuous models such as quantized neural networks and spiking neural networks.
5.  **Special significance for Transformers**: Since self-attention is generally not Lipschitz continuous, this framework is more applicable than Lipschitz-based frameworks.

## Limitations & Future Work

- The isoperimetry hypothesis may not hold for certain data distributions.
- Extending from finite to infinite function classes requires additional parameter Lipschitz assumptions.
- Generalization bounds may still be vacuous (consistent with critiques by Nagarajan & Kolter 2021).
- The actual value of the theoretical dimension $d$ is difficult to estimate precisely (ambient vs. intrinsic dimension).
- No direct link has been established with optimization dynamics (e.g., implicit bias).

## Related Work

- **Robustness Laws**: Bubeck & Sellke 2021 (Lipschitz regression)
- **Margin Generalization Bounds**: Bartlett et al. 2017 (Spectrally-normalized margin)
- **Algorithmic Stability**: Bousquet & Elisseeff 2002
- **Double Descent**: Belkin et al. 2019

## Rating

- **Novelty**: ⭐⭐⭐⭐ — A natural and important extension of the law of robustness to classification.
- **Technical Depth**: ⭐⭐⭐⭐⭐ — Sophisticated proof techniques, ranging from Lipschitz proxies to signed distance representations.
- **Experimental Thoroughness**: ⭐⭐⭐ — MNIST and CIFAR-10 validation, primarily qualitative.
- **Value**: ⭐⭐⭐⭐ — Provides theoretical support for understanding generalization in overparameterized regimes.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Stable Coresets: Unleashing the Power of Uniform Sampling](stable_coresets_unleashing_the_power_of_uniform_sampling.md)
- [\[ICLR 2026\] Why We Need New Benchmarks for Local Intrinsic Dimension Estimation](why_we_need_new_benchmarks_for_local_intrinsic_dimension_estimation.md)
- [\[ICLR 2026\] Random Spiking Neural Networks are Stable and Spectrally Simple](random_spiking_neural_networks_are_stable_and_spectrally_simple.md)
- [\[ICLR 2026\] Price of Quality: Sufficient Conditions for Sparse Recovery using Mixed-Quality Data](price_of_quality_sufficient_conditions_for_sparse_recovery_using_mixed-quality_d.md)
- [\[ICLR 2026\] Revenue Maximization under Sequential Price Competition via the Estimation of s-Concave Demand Functions](revenue_maximization_under_sequential_price_competition_via_the_estimation_of_s-.md)

</div>

<!-- RELATED:END -->

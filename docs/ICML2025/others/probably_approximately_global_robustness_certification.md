---
title: >-
  [Paper Note] Probably Approximately Global Robustness Certification
description: >-
  [ICML2025][Global robustness] A probabilistic approximately global robustness (PAG) certification framework based on ε-net sampling is proposed. The required sample complexity is independent of the input dimension, number of classes, and model architecture, enabling the efficient certification of global robustness for large-scale neural networks.
tags:
  - "ICML2025"
  - "Global robustness"
  - "ε-net"
  - "probabilistic certification"
  - "adversarial robustness"
  - "VC dimension"
  - "sample-based certification"
date: 2026-05-08
content_hash: be3947072e691842
---

# Probably Approximately Global Robustness Certification

**Conference**: ICML2025  
**arXiv**: [2511.06495](https://arxiv.org/abs/2511.06495)  
**Code**: To be confirmed  
**Area**: Robustness Certification  
**Keywords**: Global robustness, ε-net, probabilistic certification, adversarial robustness, VC dimension, sample-based certification

## TL;DR

A probabilistic approximately global robustness (PAG) certification framework based on ε-net sampling is proposed. The required sample complexity is independent of the input dimension, number of classes, and model architecture, enabling the efficient certification of global robustness for large-scale neural networks.

## Background & Motivation

**Core Problem**: How to efficiently provide global robustness guarantees for large-scale neural networks?

Existing methods face a dilemma:

- **Formal verification methods** (Marabou, α-CROWN, etc.): Provide provable local robustness guarantees, but involve extremely high computational overhead, making them applicable only to small networks with hundreds of parameters. Recent work extending these methods to global robustness certification remains limited to extremely small models.
- **Adversarial attack methods** (FGSM, PGD, C&W, etc.): Scalable to large networks, but can only evaluate robustness empirically and fail to provide theoretical guarantees.

**Key Insight**: Can **probabilistic guarantees** for global robustness be obtained via sampling? That is, guaranteeing with high probability that the classifier is robust on most inputs. By introducing the ε-net tool from learning theory, a global robustness certification independent of the input dimension can be obtained merely by i.i.d. sampling from the data distribution combined with a local robustness oracle.

## Method

### 1. Problem Formulation

Given a classifier $f: \mathcal{X} \to \mathbb{R}^n$, define:

- **Local ρ-robustness**: If a robustness oracle $\mathbf{rob}_f(\mathbf{x}) \geq \rho$, then $f$ is locally ρ-robust around $\mathbf{x}$.
- **Global ρ-κ-robustness**: For all points with prediction confidence $\mathbf{conf}_f(\mathbf{x}) \geq \kappa$, $f$ is locally ρ-robust.

$$\forall \mathbf{x} \in \mathcal{X}: \mathbf{conf}_f(\mathbf{x}) \geq \kappa \implies \mathbf{rob}_f(\mathbf{x}) \geq \rho$$

Since strict global robustness certification is computationally intractable, this paper proposes a **probabilistic relaxation**.

### 2. PAG Robustness Definition

**Approximate Global Robustness (Definition 4.1)**: Given a data distribution $\mathcal{D}$, the PAG robustness of a classifier $f$ requires:

$$\Pr\big(\mathbf{rob}_f(X) < \rho \mid \mathbf{conf}_f(X) \geq \kappa\big) < \epsilon$$

That is, among high-confidence predictions, the probability of non-robust points is less than ε.

### 3. Quality Space and ε-net

**Quality Space Mapping**: Map each input $\mathbf{x}$ to a two-dimensional "quality space":

$$q(\mathbf{x}) \mapsto (\mathbf{rob}_f(\mathbf{x}), \mathbf{conf}_f(\mathbf{x}))$$

In the quality space, the counterexample region is defined as:

$$R(\rho, \kappa) = \{(\rho', \kappa') \in \mathbb{R}^2 : \rho' < \rho \wedge \kappa' \geq \kappa\}$$

This is the intersection of two axis-aligned half-spaces, corresponding to a range space with a **VC dimension of $d=2$**.

**ε-net Sampling Theorem**: Obtain $s$ i.i.d. samples from the data distribution $\mathcal{D}$. When the sample size satisfies:

$$s \geq \frac{2}{\ln(2)\epsilon}\left(\ln\frac{1}{\delta} + d\ln(2s) - \ln(1 - e^{-s\epsilon/8})\right)$$

they form an ε-net with a probability of at least $1-\delta$. **Since $d=2$, this sample complexity is entirely independent of the input dimension, number of classes, and model architecture.**

### 4. PAG Robustness Certification (Theorem 4.3)

*Core Theorem*: Take $|N| \geq s(\epsilon, \delta/2, 2)$ i.i.d. samples. If no counterexamples exist in the samples (i.e., $q(N) \cap R(\rho, \kappa) = \emptyset$), then with probability at least $1-\delta$:

$$\Pr\big(\mathbf{rob}_f(X) < \rho \mid \mathbf{conf}_f(X) \geq \kappa\big) < \frac{\epsilon}{p_{\min}}$$

*Proof sketch*: (1) Use ε-net to bound the joint probability of counterexamples; (2) Use Chernoff bound to constrain the lower bound of the confidence quantile; (3) Combine both guarantees using a union bound.

### 5. Lower Bound Mapping M(κ)

Construct a mapping $M(\kappa)$ from sample $N$: for each confidence level $\kappa$, return the minimum robustness value of all sampled points with confidence $\geq \kappa$:

$$M(\kappa) = \min_{\mathbf{x} \in N} \mathbf{rob}_f(\mathbf{x}) \quad \text{s.t.} \quad \mathbf{conf}_f(\mathbf{x}) \geq \kappa$$

This mapping can be constructed in $O(|N|\log|N|)$ time, providing a lower bound of robustness for new data points.

### 6. Guarantees Under Distribution Shift

When sampling is only possible from an approximate distribution $\mathcal{D}'$, if the total variation distance between the two distributions is $\Lambda = \mathrm{TV}(\mathcal{D}, \mathcal{D}')$, the guarantee degrades to:

$$\Pr_{\mathcal{D}}\big(\mathbf{rob}_f(X) < \rho \mid \mathbf{conf}_f(X) \geq \kappa\big) < \frac{\epsilon + \Lambda}{p_{\min} - \Lambda}$$

## Key Experimental Results

### Experimental Setup

| Dataset | Architecture | Parameters | Robustness Oracle |
|--------|------|--------|-------------|
| MNIST | FeedForward | 39K | PGD, LiRPA |
| MNIST | ConvBig | 1,663K | LiRPA |
| CIFAR-10 | ResNet20 | 272K | PGD |
| CIFAR-10 | VGG11_BN | 9,491K | PGD |

Each architecture is trained with 5 standard training instances + 5 TRADES adversarial training instances, verified by cross-validation.

### Parameter Settings

- PGD oracle: $\epsilon=10^{-4}$, $p_{\min}=0.01$, $\delta=0.01$, sample size $s = 989,534$
- LiRPA oracle: $\epsilon=2.5 \times 10^{-3}$, $p_{\min}=0.05$, $\delta=0.01$, sample size $s = 31,635$

### Main Results

| Experiment | Network | $\hat{p} \times 10^3$ | $\epsilon/p_{\min} \times 10^3$ | $n_c$ | Pass Rate | Runtime (min) |
|------|------|----------|------------|-------|--------|------------|
| MNIST_ST,L | FeedForward | **0.00** | 50.00 | **0** | 25/25 | 8.3 |
| MNIST_ST,L | ConvBig | **0.00** | 50.00 | **0** | 15/15 | 245 |
| MNIST_AT,L | FeedForward | **1.30** | 50.00 | **7** | 25/25 | 8.3 |
| MNIST_AT,L | ConvBig | **3.39** | 50.00 | 44 | 13/15 | 246 |
| MNIST_ST,P | FeedForward | **0.18** | 10.00 | **2** | 25/25 | 0.2 |
| MNIST_AT,P | FeedForward | 16.50 | 10.00 | **4** | 19/25 | 0.3 |
| CIFAR_ST,P | ResNet20 | **1.35** | 10.00 | 8 | 17/25 | 4.5 |
| CIFAR_ST,P | VGG11_BN | **0.00** | 10.00 | **0** | 25/25 | 38.8 |
| CIFAR_AT,P | ResNet20 | **1.51** | 10.00 | **4** | 25/25 | 133.4 |
| CIFAR_AT,P | VGG11_BN | 17.00 | 10.00 | **9** | 22/25 | 308 |

**Key Findings**: In 230 experiments, the guarantee holds on unseen test data in 211 cases (91.7%). Most violated cases deviate only slightly, which can be attributed to the minor discrepancy between the Gaussian noise sampling and the true distribution.

## Highlights & Insights

1. **Dimension Independence**: The sample complexity depends only on ε, δ, and the VC dimension ($d=2$), remaining completely independent of the input dimension, the number of classes, and the model scale. This allows the method to be directly applied to networks with tens of millions of parameters.
2. **Oracle Agnosticism**: The framework is entirely transparent to the local robustness oracle and can seamlessly integrate any method, such as PGD, LiRPA, or Marabou.
3. **Multiple Guarantees from a Single Sample**: Robustness guarantees for all $(\rho, \kappa)$ combinations can be obtained simultaneously from a single sampling step without needing to re-sample for each parameter.
4. **Robustness to Distribution Shift**: The degradation of the guarantee when sampling from an approximate distribution is theoretically characterized, enhancing practical applicability.
5. **Theoretical Elegance**: By elegantly projecting the high-dimensional robustness certification problem onto a two-dimensional quality space, the method unifies computational and statistical efficiency using classical ε-net theory.

## Limitations & Future Work

1. **Dependence on the i.i.d. Sampling Assumption**: In practice, sampling strictly i.i.d. from the data distribution is difficult; the approximation using Gaussian noise in the paper may introduce bias.
2. **Limited Strength of Guarantees**: The ratio of $\epsilon/p_{\min}$ can be relatively large (e.g., 1% in the PGD experiments), which might not be sufficiently stringent for safety-critical scenarios.
3. **Confidence Reliance on Softmax**: When the model is poorly calibrated, softmax confidence may fail to reflect the true reliability of predictions.
4. **Upper Bound from Oracle Quality**: When using heuristic oracles like PGD, the robustness estimation is only an upper bound, which may lead to overly optimistic guarantees.
5. **Large Sample Size**: The PGD setup requires nearly a million samples, posing runtime challenges for models with high inference costs.
6. **Neglect of Prediction Class Information**: The current method treats all classes uniformly, failing to leverage the prior knowledge that different classes may exhibit different robustness characteristics.

## Related Work & Insights

- **Formal Verification**: Marabou, α-CROWN, etc. provide exact local robustness verification but scale only to small networks.
- **Adversarial Attacks**: FGSM, PGD, C&W are used for empirical robustness evaluation.
- **Certified Training**: Methods like TRADES train certifiably robust networks.
- **Global Robustness**: Athavale et al. 2024 first extended formal verification to global robustness but remained limited to networks with hundreds of parameters.
- **Statistical Robustness**: Randomized smoothing (Cohen et al. 2019) provides probabilistic robustness guarantees, but only for single points.

The core idea of this paper—projecting high-dimensional problems into a lower-dimensional quality space and leveraging ε-net theory—could inspire other verification tasks that require high-dimensional probabilistic guarantees.

## Rating

- Novelty: ⭐⭐⭐⭐ — The integration of quality space and ε-net is novel, and the dimension-independence conclusion is highly impressive.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 230 experiments covering 4 architectures $\times$ 2 training methods $\times$ 2 oracles provide solid statistical thoroughness.
- Writing Quality: ⭐⭐⭐⭐⭐ — The theoretical derivation is clear and rigorous, with definitions, propositions, and theorems progressing logically.
- Value: ⭐⭐⭐⭐ — Successfully addresses the gap in global robustness certification for large-scale networks, offering high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning (Approximately) Equivariant Networks via Constrained Optimization](../../NeurIPS2025/others/learning_approximately_equivariant_networks_via_constrained_optimization.md)
- [\[ICML 2025\] IBDR: Promoting Ensemble Diversity with Interactive Bayesian Distributional Robustness](promoting_ensemble_diversity_with_interactive_bayesian_distributional_robustness.md)
- [\[NeurIPS 2025\] The Cost of Robustness: Tighter Bounds on Parameter Complexity for Robust Memorization in ReLU Nets](../../NeurIPS2025/others/the_cost_of_robustness_tighter_bounds_on_parameter_complexity_for_robust_memoriz.md)
- [\[ACL 2025\] Neuron Empirical Gradient: Discovering and Quantifying Neurons' Global Linear Controllability](../../ACL2025/others/neuron_empirical_gradient_discovering_and_quantifying_neurons_global_linear_cont.md)
- [\[CVPR 2026\] Global Information Thresholding for Sufficient and Necessary Circuits](../../CVPR2026/others/global_information_thresholding_for_sufficient_and_necessary_circuits.md)

</div>

<!-- RELATED:END -->

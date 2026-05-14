---
title: >-
  [Paper Note] The Cost of Robustness: Tighter Bounds on Parameter Complexity for Robust Memorization in ReLU Nets
description: >-
  [NeurIPS 2025][robust memorization] This paper studies the parameter complexity of robust memorization in ReLU networks — i.e., the number of parameters required to interpolate an arbitrary dataset while maintaining cons…
tags:
  - "NeurIPS 2025"
  - "robust memorization"
  - "ReLU networks"
  - "parameter complexity"
  - "upper and lower bounds"
  - "adversarial robustness"
date: 2026-05-08
content_hash: e7e902e004637bc0
---

# The Cost of Robustness: Tighter Bounds on Parameter Complexity for Robust Memorization in ReLU Nets

**Conference**: NeurIPS 2025
**arXiv**: [2510.24643](https://arxiv.org/abs/2510.24643)
**Code**: None
**Area**: Learning Theory / Robustness
**Keywords**: robust memorization, ReLU networks, parameter complexity, upper and lower bounds, adversarial robustness

## TL;DR

This paper studies the parameter complexity of robust memorization in ReLU networks — i.e., the number of parameters required to interpolate an arbitrary dataset while maintaining consistent predictions within a $\mu$-neighborhood of each training sample — and establishes tighter upper and lower bounds across the full range $(0,1)$ of the robustness ratio $\rho = \mu/\epsilon$.

## Background & Motivation

Understanding the expressive capacity of neural networks is a foundational problem in deep learning theory. **Memorization** serves as a basic measure of expressive power: given $N$ data points, how many parameters suffice for a network to perfectly interpolate them?

Classical results (e.g., Baum, 1988; Huang & Babri, 1998) show that $O(N)$ parameters are sufficient to memorize $N$ points. Under adversarial robustness requirements, however, the problem becomes considerably more complex: the network must not only produce correct predictions at training points, but also maintain consistent predictions throughout the $\mu$-neighborhood of each training point.

**Problem formulation**: Given a dataset $\{(x_i, y_i)\}_{i=1}^N$ where points of different labels are separated by an $\epsilon$-margin, the goal is to find a ReLU network $f$ such that:
- $f(x_i) = y_i$ for all $i$
- $f(x) = y_i$ for all $\|x - x_i\| \leq \mu$ with $\mu < \epsilon$

The **robustness ratio** $\rho = \mu/\epsilon \in (0, 1)$ quantifies the strength of the robustness requirement. As $\rho \to 0$, the problem reduces to standard memorization; as $\rho \to 1$, the required robustness approaches the inter-point separation.

Prior work provided bounds only for specific values or limited ranges of $\rho$, leaving substantial gaps between upper and lower bounds.

## Method

### Overall Architecture

The paper advances existing results along two directions:

1. **Tighter upper bounds**: constructing more parameter-efficient ReLU networks for robust memorization
2. **Tighter lower bounds**: proving lower bounds on the number of parameters required by any ReLU network for robust memorization
3. **Full-range analysis**: providing fine-grained analysis across the entire range $\rho \in (0, 1)$

### Key Designs

**Upper bound construction — improved network architecture**:

The constructed ReLU network consists of two modules:
1. **Region partition module**: partitions the input space into $N$ regions, one per training point
2. **Value assignment module**: outputs the corresponding label within each region

The key improvement lies in the design of the region partition module. Prior approaches use "spherical regions" ($\mu$-balls around each training point), requiring $O(N \cdot d)$ parameters (where $d$ is the dimension). This paper instead employs a more geometrically efficient construction:

$$W(N, d, \rho) = \tilde{O}\left(N \cdot d^{1/2} \cdot \sqrt{\log(1/(1-\rho))}\right)$$

**Lower bound proof — information-theoretic argument**:

The core argument proceeds as follows:
1. A network capable of robust memorization must encode the full information of the dataset
2. As $\rho$ increases, the robust region around each point occupies more space, reducing the "gaps" between regions
3. The network requires increasingly complex decision boundaries to distinguish neighboring regions, necessitating more parameters

$$W(N, d, \rho) = \Omega\left(N \cdot \sqrt{d} \cdot \rho\right) \quad \text{for} \quad \rho \in (0, 1/2)$$

$$W(N, d, \rho) = \Omega\left(N \cdot d \cdot \rho^2\right) \quad \text{for} \quad \rho \in (1/2, 1)$$

### Loss & Training

This is a purely theoretical work and involves no training. The paper addresses an **existence** question — whether ReLU networks satisfying the stated conditions exist, and what the asymptotic bounds on their parameter counts are.

## Key Experimental Results

### Main Results

**Theorem comparison: upper bounds (parameter count)**

| $\rho$ range | Prev. SOTA upper bound | **Ours** | Gain |
|---|---|---|---|
| $\rho \in (0, 0.1)$ | $O(N \cdot d)$ | $O(N \cdot \sqrt{d})$ | $\sqrt{d}$ |
| $\rho \in (0.1, 0.5)$ | $O(N \cdot d \cdot \log(1/\rho))$ | $O(N \cdot \sqrt{d \log(1/(1-\rho))})$ | $\sim \sqrt{d}$ |
| $\rho \in (0.5, 0.9)$ | $O(N \cdot d^2 / (1-\rho))$ | $O(N \cdot d \cdot \sqrt{\log(1/(1-\rho))})$ | $d / \sqrt{\log}$ |
| $\rho \to 1$ | $O(N \cdot d^2 / (1-\rho)^2)$ | $O(N \cdot d / (1-\rho))$ | $d(1-\rho)$ |

**Theorem comparison: lower bounds (parameter count)**

| $\rho$ range | Prev. SOTA lower bound | **Ours** | Gain |
|---|---|---|---|
| $\rho \in (0, 0.1)$ | $\Omega(N)$ | $\Omega(N \cdot \sqrt{d} \cdot \rho)$ | $\sqrt{d} \cdot \rho$ |
| $\rho \in (0.1, 0.5)$ | $\Omega(N \cdot \rho)$ | $\Omega(N \cdot \sqrt{d} \cdot \rho)$ | $\sqrt{d}$ |
| $\rho \in (0.5, 0.9)$ | $\Omega(N \cdot d^{1/3} \cdot \rho)$ | $\Omega(N \cdot d \cdot \rho^2)$ | $d^{2/3} \cdot \rho$ |
| $\rho \to 1$ | $\Omega(N \cdot d^{1/2})$ | $\Omega(N \cdot d \cdot \rho^2)$ | $d^{1/2} \cdot \rho^2$ |

### Ablation Study

**Phase transition behavior in parameter complexity** — theoretical bounds verified numerically:

| $\rho$ | Non-robust baseline $O(N)$ | Ours (upper) | Ours (lower) | Upper/lower ratio |
|---|---|---|---|---|
| 0.01 | $N$ | $1.02N\sqrt{d}$ | $0.01N\sqrt{d}$ | 102 |
| 0.1 | $N$ | $1.15N\sqrt{d}$ | $0.1N\sqrt{d}$ | 11.5 |
| 0.3 | $N$ | $1.41N\sqrt{d}$ | $0.3N\sqrt{d}$ | 4.7 |
| 0.5 | $N$ | $1.73N\sqrt{d}$ | $0.5N\sqrt{d}$ | 3.5 |
| 0.7 | $N$ | $2.8Nd$ | $0.49Nd$ | 5.7 |
| 0.9 | $N$ | $5.2Nd$ | $0.81Nd$ | 6.4 |

A phase transition occurs near $\rho \approx 0.5$, where the dimensional dependence shifts from $\sqrt{d}$ to $d$, indicating a qualitative change in the rate of parameter growth once the robustness requirement exceeds half the inter-point margin.

**Effect of dimension $d$ (fixed $N=100$, $\rho=0.5$)**:

| Dimension $d$ | Non-robust params | Ours (upper) | Ours (lower) | Cost of robustness |
|---|---|---|---|---|
| 10 | 100 | 547 | 158 | 5.5× |
| 50 | 100 | 1,225 | 354 | 12.3× |
| 100 | 100 | 1,732 | 500 | 17.3× |
| 500 | 100 | 3,873 | 1,118 | 38.7× |
| 1000 | 100 | 5,477 | 1,581 | 54.8× |

The "cost" of robustness grows substantially with dimension, revealing the intrinsic difficulty of robust learning in high-dimensional spaces.

### Key Findings

1. **Robustness has a cost**: robust memorization requires significantly more parameters than standard memorization, with the cost increasing in $\rho$
2. **Phase transition**: at $\rho \approx 0.5$, parameter complexity transitions from $\sqrt{d}$ to $d$ dependence
3. **Recovery of non-robust bounds at small $\rho$**: when robustness requirements are weak, parameter complexity matches the non-robust setting
4. **Amplified curse of dimensionality**: robustness requirements magnify the effect of dimension on parameter complexity
5. **Tighter bounds throughout**: the paper provides strictly tighter bounds than prior work across the full range of $\rho$

## Highlights & Insights

- **Fine-grained full-range analysis**: the first work to provide tight upper and lower bounds across the entire range $\rho \in (0,1)$
- **Discovery of a phase transition**: the phase transition in parameter complexity near $\rho = 0.5$ is a theoretically meaningful finding
- **Development of new theoretical tools**: novel geometric and information-theoretic techniques are introduced for both the upper bound construction and the lower bound proof
- **72-page rigorous analysis**: a thorough and complete theoretical contribution

## Limitations & Future Work

1. **Remaining gaps between bounds**: particularly in the extreme regimes where $\rho$ approaches 0 or 1
2. **Restriction to ReLU networks**: results may differ for other activation functions such as GELU or Swish
3. **Memorization vs. generalization**: the paper studies memorization capacity; its connection to generalization in practical training is not direct
4. **Binary classification focus**: the main results target binary classification, and extensions to multiclass settings are non-trivial
5. **Connection to practical adversarial training**: the relationship between the theoretical bounds and phenomena observed in practical adversarial training remains to be established

## Related Work & Insights

- **Standard memorization**: Baum (1988), classical network capacity results
- **Robust memorization**: Bubeck et al. (2021), Vardi et al. (2022) — the direct predecessors improved upon by this paper
- **Theory of adversarial robustness**: Gilmer et al. (2018), Madry et al. (2018) — theoretical and practical foundations of adversarial robustness
- **Expressive power of ReLU networks**: Telgarsky (2016), analysis of depth vs. width in expressive capacity

## Rating

- **Novelty**: 4/5 — full-range fine-grained analysis and discovery of the phase transition phenomenon
- **Technical Quality**: 5/5 — rigorous mathematical proofs with 72 pages of thorough analysis
- **Writing Quality**: 4/5 — theoretically clear, though the length is considerable
- **Value**: 2/5 — purely theoretical contribution with limited direct practical applicability
- **Overall**: 4/5

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Computational Complexity of Counting Linear Regions in ReLU Neural Networks](the_computational_complexity_of_counting_linear_regions_in_relu_neural_networks.md)
- [\[NeurIPS 2025\] Impact of Layer Norm on Memorization and Generalization in Transformers](impact_of_layer_norm_on_memorization_and_generalization_in_transformers.md)
- [\[NeurIPS 2025\] Kernel Conditional Tests from Learning-Theoretic Bounds](kernel_conditional_tests_from_learning-theoretic_bounds.md)
- [\[NeurIPS 2025\] Depth-Bounds for Neural Networks via the Braid Arrangement](depth-bounds_for_neural_networks_via_the_braid_arrangement.md)
- [\[NeurIPS 2025\] Obliviator Reveals the Cost of Nonlinear Guardedness in Concept Erasure](obliviator_reveals_the_cost_of_nonlinear_guardedness_in_concept_erasure.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Counterfactual Identifiability via Dynamic Optimal Transport
description: >-
  [NeurIPS 2025][Image Generation][counterfactual identification] This paper leverages dynamic optimal transport (dynamic OT) theory to resolve—for the first time—the counterfactual identifiability problem in high-dimensio…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "counterfactual identification"
  - "optimal transport"
  - "flow matching"
  - "structural causal model"
  - "monotone transport map"
date: 2026-05-08
content_hash: 8eac62108fe3dea9
---

# Counterfactual Identifiability via Dynamic Optimal Transport

**Conference**: NeurIPS 2025
**arXiv**: [2510.08294](https://arxiv.org/abs/2510.08294)  
**Code**: To be confirmed  
**Area**: Causal Inference / Generative Models / Optimal Transport
**Keywords**: counterfactual identification, optimal transport, flow matching, structural causal model, monotone transport map

## TL;DR
This paper leverages dynamic optimal transport (dynamic OT) theory to resolve—for the first time—the counterfactual identifiability problem in high-dimensional multivariate Markovian SCMs. It proves that the OT flow mechanism yields a unique monotone order-preserving counterfactual transport map, and extends the results to non-Markovian settings (IV/BC/FC criteria).

## Background & Motivation

**Background**: Deep generative models (VAEs, diffusion models, flows) are increasingly used to parameterize structural causal models (SCMs) for counterfactual inference, but these approaches lack identifiability guarantees—multiple observationally equivalent models can yield different counterfactual answers from the same observed data.

**Limitations of Prior Work**: (a) Pearl emphasizes that counterfactual queries must satisfy identifiability requirements to support reliable causal claims; (b) classical symbolic identification methods (Tian & Pearl; Shpitser & Pearl) have not been extended to high-dimensional variables; (c) Nasr-Esfahany et al. (2023) established counterfactual identifiability for bijective mechanisms, but how to generalize the monotonicity condition to multivariate variables ($d>1$) in the Markovian setting remained an open problem.

**Key Challenge**: In the multivariate setting, bijectivity alone is insufficient to guarantee counterfactual identifiability—due to rotational symmetry, infinitely many bijective mechanisms can generate the same observational distribution. A correct multivariate generalization of monotonicity is needed to break this symmetry.

**Key Insight**: The paper exploits Brenier's theorem—under standard regularity conditions, the optimal transport map $T = \nabla \phi$ is unique and monotone (given as the gradient of a convex function). This is connected to causal mechanisms in SCMs to show that OT flow mechanisms naturally satisfy the monotonicity required for multivariate counterfactual identifiability.

## Method

### Problem Setup

Consider an SCM $\mathfrak{C} = (\mathbf{U}, \mathbf{X}, \mathcal{F})$, focusing on the causal mechanism $f$ of a multivariate ($d>1$) variable $X$: $X = f(\mathbf{PA}, U)$, where $\dim(X) = \dim(U) = d$.

Counterfactual query: "Given observation $X=x$ (with parent $\mathbf{PA}=\mathbf{pa}$), what would $X$ be if the parent were set to $\mathbf{pa}^*$?"

Counterfactual transport map: $T^*(\mathbf{pa}^*, \mathbf{pa}, x) = f(\mathbf{pa}^*, f^{-1}(\mathbf{pa}, x))$

### Core Theory

**Definition 4.3 (Monotone Operator)**: A map $f$ is monotone in $u$ if:
$$\langle f(\mathbf{pa}, u_1) - f(\mathbf{pa}, u_2), u_1 - u_2 \rangle \geq 0, \quad \forall u_1, u_2$$

**Proposition 4.4**: If mechanism $f$ is monotone in $u$, then the counterfactual transport map $T^*$ is monotone in $x$—ensuring that the ordering of factual outcomes is preserved under counterfactual intervention (rank preservation).

**Lemma 4.6 (Unique and Monotone Dynamic OT Mechanism)**: In the Markovian setting ($U \perp\!\!\!\perp \mathbf{PA}$), let $T$ be the time-1 map of the dynamic OT flow pushing $P_U$ forward to $P_{X|\mathbf{PA}}$. Under standard regularity conditions, there exists a convex function $\phi$ such that $T(u; \mathbf{pa}) = \nabla_u \phi(u; \mathbf{pa})$; moreover, $T$ is monotone, almost-everywhere bijective, and uniquely determined by $(P_U, P_{X|\mathbf{PA}})$.

**Theorem 4.12 (Counterfactual Identifiability in Markovian SCMs)**: Let $P_U$ be the uniform distribution on $[0,1]^d$ and $T$ the OT map from Lemma 4.6. Then the counterfactual transport map $T^*$ is **strictly monotone** in $x$:

$$\langle T^*(\mathbf{pa}^*, \mathbf{pa}, x_1) - T^*(\mathbf{pa}^*, \mathbf{pa}, x_2), x_1 - x_2 \rangle > 0, \quad \forall x_1 \neq x_2$$

This guarantees $\mathcal{L}_3$-equivalence identifiability—counterfactuals recovered from observational data are unique.

### Non-Markovian Extensions

The theory is extended to non-Markovian settings under three standard causal criteria:
- **Instrumental Variable (IV)**: The monotonicity of Lemma 4.6 is used to generalize the $d=1$ IV result to $d>1$.
- **Back-Door Criterion (BC)**: Bijectivity plus sufficient variability suffice (inheriting results from Nasr-Esfahany et al.).
- **Front-Door Criterion (FC)**: Identifiability is proved under conditions analogous to BC (a new result).

### Practical Inference: Flow Matching

Causal mechanisms are parameterized by continuous-time flow models trained via flow matching:

$$\min_{\theta} \int_0^1 \mathbb{E}_{X_1 \sim p_{\text{data}}} \left[\|v_t(X_t; \theta) - v_t^*(X_t | X_1)\|^2\right] dt$$

Counterfactual inference follows the abduction–action–prediction pipeline:
1. **Abduction**: Recover exogenous noise by backward ODE integration: $u = x - \int_0^1 v_t(x_t; \mathbf{pa}, \theta)\, dt$
2. **Action**: Set the counterfactual parent $\mathbf{pa}^*$
3. **Prediction**: Obtain $x^*$ by forward ODE integration: $x^* = u + \int_0^1 v_t(x_t; \mathbf{pa}^*, \theta)\, dt$

**Markovian Batch-OT Coupling**: The paper corrects an issue in standard Batch-OT flow matching whereby the independence assumption $U \perp\!\!\!\perp \mathbf{PA}$ is implicitly violated—the OT coupling is solved independently for each fixed value of $\mathbf{pa}$.

## Key Experimental Results

### Experiment 1: Counterfactual Ellipse Generation (Synthetic, with Ground Truth)

| Method | NFE | μ_APE (%) ↓ (Markov) | μ_APE (%) ↓ (Front-door) |
|--------|-----|----------------------|--------------------------|
| Baseline (Nasr-Esfahany) | - | 607 | - |
| EBM | 50 | 2.32 | 1.79 |
| Flow | 50 | 2.30 | 1.67 |
| **OT-EBM** | 2 | 1.21 | 1.64 |
| **OT-Flow** | 2 | **1.06** | **1.60** |
| Naive Batch-OT | 2 | Violates Markov assumption; incorrect counterfactuals | - |

- OT variants achieve ~1% error with only 2 function evaluations (NFE), while baselines require 50 NFE.
- As theoretically predicted, OT maps significantly outperform standard flows in the Markovian setting; in the front-door setting, bijectivity alone suffices.

### Experiment 2: Chest X-Ray Counterfactual Generation (MIMIC-CXR, 192×192)

| Intervention | Metric | Baseline (Ribeiro 2023) | Flow (Ours) |
|--------------|--------|------------------------|-------------|
| do(Sex=s) | \|Δ_AUC\| ↓ | 0.370% | **0.173%** |
| do(Race=r) | \|Δ_AUC\| ↓ | 8.640% | **0.050%** |
| do(Age=a) | Δ_MAE ↓ | 0.288 yr | 0.333 yr |
| do(Disease=d) | \|Δ_AUC\| ↓ | 2.490% | **0.023%** |

- Substantial improvement on the race intervention (8.64% → 0.05%), attributable to more consistent counterfactuals provided by OT.
- Markovian OT coupling significantly outperforms the naive OT flow baseline.

## Highlights & Insights

- **Theoretical contribution is the core value**: This paper resolves an important open problem in multivariate Markovian counterfactual identifiability; the connection between Brenier's theorem and SCMs is elegant.
- **Translating the mathematical uniqueness guarantee of OT into a causal identifiability guarantee** establishes a profound link between two seemingly unrelated fields.
- **The Markovian Batch-OT coupling correction** is an important technical contribution—it exposes an implicit flaw in standard practice.
- **Multivariate generalization of rank preservation**: Monotonicity ensures counterfactuals do not produce rank inversions, which is critical for fairness applications.

## Limitations & Future Work

- **Strong regularity assumptions**: Strictly positive, bounded densities defined on bounded convex domains are required, excluding many practical distributions.
- **Scalability of OT in high dimensions**: Batch-OT requires large batch sizes in high dimensions, leading to high computational cost.
- **Limitations of counterfactual validity metrics**: Composition/effectiveness/reversibility do not equate to identifiability—as the authors themselves acknowledge.
- **The Markovian assumption is difficult to verify on real data**: The causal graph for MIMIC-CXR is assumed and may be subject to unobserved confounding.
- **Choice of prior $P_U$**: A uniform or standard Gaussian distribution is assumed, but the true exogenous distribution is unknown.

## Related Work & Insights

- **vs. Nasr-Esfahany et al. (2023)**: Their spline flow fails in the Markovian $d>1$ setting (μAPE = 607%); the proposed OT flow resolves this problem.
- **vs. Pawlowski et al. (2020) / Ribeiro et al. (2023)**: These works parameterize SCMs with VAEs/diffusion models but lack identifiability guarantees; this paper provides the theoretical foundation.
- **vs. Classical symbolic methods (Tian & Pearl)**: Classical methods do not apply to high-dimensional variables; this paper fills that gap.
- **vs. Brenier's theorem**: Applying the purely mathematical OT uniqueness result to causal inference is a novel interdisciplinary contribution.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Resolves an important open problem in causal inference; the OT–causality theoretical connection is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic experiments provide ground-truth validation of the theory; real-data experiments demonstrate practical utility.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are rigorous, though the prerequisite knowledge required is substantial, slightly limiting accessibility.
- Value: ⭐⭐⭐⭐⭐ Provides much-needed theoretical foundations for deep causal inference with broad impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] On the Relation between Rectified Flows and Optimal Transport](on_the_relation_between_rectified_flows_and_optimal_transport.md)
- [\[ICCV 2025\] The Curse of Conditions: Analyzing and Improving Optimal Transport for Conditional Flow-Based Generation](../../ICCV2025/image_generation/the_curse_of_conditions_analyzing_and_improving_optimal_transport_for_conditiona.md)
- [\[CVPR 2026\] COT-FM: Cluster-wise Optimal Transport Flow Matching](../../CVPR2026/image_generation/cot-fm_cluster-wise_optimal_transport_flow_matching.md)
- [\[NeurIPS 2025\] Pairwise Optimal Transports for Training All-to-All Flow-Based Condition Transfer Model](pairwise_optimal_transports_for_training_all-to-all_flow-based_condition_transfe.md)
- [\[ICML 2026\] Pareto-Guided Optimal Transport for Multi-Reward Alignment](../../ICML2026/image_generation/pareto-guided_optimal_transport_for_multi-reward_alignment.md)

</div>

<!-- RELATED:END -->

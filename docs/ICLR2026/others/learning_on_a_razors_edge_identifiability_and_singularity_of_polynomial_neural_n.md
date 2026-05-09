---
title: >-
  [Paper Note] Learning on a Razor's Edge: Identifiability and Singularity of Polynomial Neural Networks
description: >-
  [ICLR 2026][Identifiability] Using tools from algebraic geometry, this paper systematically analyzes MLPs and CNNs with polynomial activations: it proves finite identifiability for MLPs and unique identifiability for CNNs, reveals that sparse subnetworks correspond to singular points of the neuromanifold, and provides a geometric explanation of the sparsity bias in MLPs via the notion of "critical exposure"—a property that CNNs do not possess.
tags:
  - ICLR 2026
  - Identifiability
  - Neuromanifold Singularities
  - Polynomial Neural Networks
  - Sparsity Bias
  - Algebraic Geometry
date: 2026-05-08
content_hash: 5e9b86a37f5673d2
---

# Learning on a Razor's Edge: Identifiability and Singularity of Polynomial Neural Networks

**Conference**: ICLR 2026
**arXiv**: [2505.11846](https://arxiv.org/abs/2505.11846)
**Code**: None
**Area**: Deep Learning Theory / Algebraic Geometry
**Keywords**: Identifiability, Neuromanifold Singularities, Polynomial Neural Networks, Sparsity Bias, Algebraic Geometry

## TL;DR

Using tools from algebraic geometry, this paper systematically analyzes MLPs and CNNs with polynomial activations: it proves finite identifiability for MLPs and unique identifiability for CNNs, reveals that sparse subnetworks correspond to singular points of the neuromanifold, and provides a geometric explanation of the sparsity bias in MLPs via the notion of "critical exposure"—a property that CNNs do not possess.

## Background & Motivation

- **Background**: Neural networks parameterize a function space known as the neuromanifold. The geometric properties of this manifold—its dimension, identifiability, and singular points—directly affect the model's expressive power, training dynamics, and generalization. Existing identifiability analyses are limited to specific activation functions such as Tanh, Sigmoid, and ReLU.
- **Limitations of Prior Work**: (1) A systematic proof of identifiability (i.e., the redundancy of distinct parameters mapping to the same function) for general activation functions is lacking; (2) a complete characterization of neuromanifold singularities exists only for linear networks and monomial CNNs; (3) the sparsity bias observed during training—where networks tend to drop neurons and converge to sparse subnetworks—lacks a geometric theoretical explanation.
- **Key Challenge**: Although it is empirically believed that MLPs possess discrete parameter symmetries arising from neuron permutations, formal proofs exist only for specific activations such as Tanh and Sigmoid. While progress has been made on identifiability for monomial activations, infinite fibers arising from per-neuron scaling prevent direct generalization.
- **Goal**: For "sufficiently generic" polynomial activation functions, this paper aims to (i) provide a unified proof of identifiability and dimension formulas for MLPs and CNNs, (ii) characterize the relationship between singular points and subnetworks, and (iii) explain the geometric origin of the sparsity bias.
- **Key Insight**: Polynomial activations → the neuromanifold is a semi-algebraic variety → tools from algebraic geometry (Zariski topology, fiber dimension theorem, toric geometry) become applicable → results hold for "generic" polynomials → generalization to arbitrary activations via polynomial approximation.
- **Core Idea**: Sparse subnetworks (networks with some neurons set to zero) precisely constitute the singular points of the neuromanifold. For MLPs, these singular points are also critical points of the loss function with positive probability, causing SGD to be attracted to them—providing a geometric explanation of the Lottery Ticket Hypothesis and sparsity bias.

## Method

### Overall Architecture

This is a purely theoretical work with no empirical experiments. The overall approach proceeds as follows:

1. Define the neuromanifold $\mathcal{M}_{\mathbf{d},\sigma}$ as the image of the parameterization map $\varphi: \mathcal{W} \to \mathcal{V}$;
2. Study the fiber structure of $\varphi$ to address identifiability;
3. Analyze points where the tangent space dimension is anomalous to characterize singularities;
4. Introduce the notion of "critical exposure" to connect singularities with optimization dynamics.

### Key Designs

1. **Finite Identifiability of MLPs (Theorem 4.1)**:

    - **Function**: Prove that for a generic polynomial activation of sufficiently large degree $r$, the generic fiber of the MLP parameterization map is finite.
    - **Mechanism**: Construct a special sparse polynomial activation $\sigma(x) = \sum_{i=1}^{L} x^{\beta_i}$ so that the MLP output decomposes into a sum of monomial MLPs (Lemma B.1). Using the known fiber structure of monomial MLPs (permutations and diagonal scaling), the problem reduces to solving a polynomial system $\lambda_{L,1}^{\beta_{L-1}^{L-2}} \cdots \lambda_{L,L-1} = 1$. The Smith normal form and toric geometry are then applied to show that this system has only finitely many solutions. Conclusion: $\dim(\mathcal{M}_{\mathbf{d},\sigma}) = \sum_{i=1}^{L} d_i d_{i-1}$, i.e., the neuromanifold dimension equals the number of parameters.
    - **Design Motivation**: This resolves the dimension conjecture posed by Kileel et al. (2019). Compared to monomial activations (which have infinite fibers due to scaling symmetry), the multiple nonzero coefficients of a polynomial activation provide additional constraint equations that eliminate the scaling degrees of freedom.

2. **Unique Identifiability of CNNs (Theorem 4.4)**:

    - **Function**: Prove that the CNN parameterization is regular (full-rank Jacobian) on $\mathcal{W} \setminus \varphi^{-1}(0)$ and generically injective.
    - **Mechanism**: Exploiting the weight-sharing structure of CNNs, the Jacobian analysis of convolutional operations is more streamlined. A logarithmic derivative technique is employed (Lemma C.1): an auxiliary function $P(x) = x\sigma'(x)/\sigma(x)$ is constructed, and asymptotic expansion arguments show that any scaling factors $\lambda_i$ making two parameterizations equal must all equal 1.
    - **Design Motivation**: Weight sharing in CNNs eliminates the neuron permutation symmetry present in MLPs, yielding a stronger identifiability result. This architectural difference has far-reaching consequences for singularities and sparsity bias.

3. **Subnetworks and Singular Points (Theorems 4.2, 4.6)**:

    - **Function**: Prove that sparse subnetworks (with some neurons set to zero) define singular points of the neuromanifold.
    - **Mechanism (MLP)**: At a subnetwork parameter $\mathbf{W}$, the rows corresponding to zeroed neurons can vary freely without affecting $f_\mathbf{W}$. The partial derivatives of $\varphi$ with respect to these "dead" parameters produce tangent vectors $\frac{\partial \varphi}{\partial W_{i+2}[k,j]}$, which span a space of dimension exceeding $\dim(\mathcal{M})$ as the free parameters vary—precisely the definition of a singular point.
    - **Mechanism (CNN)**: Theorem 4.6 gives a complete characterization: a point is singular if and only if the parameter corresponds to a "proper" subnetwork—i.e., filters padded with zeros on the left or right—and the recurrence quantities $\tilde{t}_i = t_i + \tilde{t}_{i-1}/s_{i-1}$ satisfy an integrality condition. All singularities of CNNs are of "node type" (self-intersections), rather than the "cusp type" found in MLPs.
    - **Design Motivation**: This establishes a precise correspondence between sparse structure and geometric singularity, laying the foundation for subsequent optimization analysis.

4. **Critical Exposure (Theorem 4.3, Proposition 4.5)**:

    - **Function**: Introduce the notion of "critical exposure" and prove that MLP subnetworks are critically exposed, whereas CNN subnetworks are not.
    - **Mechanism**: A set $S$ is defined as critically exposed if $U_S = \{u \in \mathcal{V} \mid \exists \mathbf{W} \in S,\, \nabla(\mathcal{L}_u \circ \varphi)(\mathbf{W}) = 0\}$ has nonempty interior in $\mathcal{V}$. Geometrically, $U_S = \bigcup_{\mathbf{W} \in S} f_\mathbf{W} + \mathrm{im}(J_\mathbf{W}\varphi)^\perp$—a union of affine subspaces. For MLPs, the Jacobian at a strict subnetwork has zero columns (corresponding to dead parameters), causing the union of normal spaces to span the full dimension $\dim(\mathcal{V})$; for CNNs, since the parameterization is regular (no zero columns), the image of this algebraic set cannot achieve full dimension.
    - **Design Motivation**: Critical exposure implies that for a randomly sampled data target $u$, there exists a critical point of the loss within the subnetwork with positive probability. In other words, SGD training dynamics are attracted to sparse subnetworks with positive probability—providing a purely geometric explanation for the sparsity bias of MLPs.

### Loss & Training

The paper considers the standard mean squared error loss $\mathcal{L}_\mathcal{D}(f) = \sum_{(x,y) \in \mathcal{D}} \|f(x) - y\|^2$. The key observation is that this loss can be rewritten as $\mathcal{L}_u(f) = Q(f - u)$, where $Q$ is a quadratic form on $\mathcal{V}$ and $u \in \mathcal{V}$ depends on the dataset. By the chain rule, $\mathbf{W}$ is a critical point of $\mathcal{L}_u \circ \varphi$ if and only if $f_\mathbf{W} - u$ is orthogonal to the image of the Jacobian of the parameterization map—a geometric equivalence that underlies all optimization-related proofs.

## Key Experimental Results

### Computational Examples (Appendix D)

This is a purely theoretical work with no large-scale experiments. The appendix provides small-scale symbolic computations for verification:

| Architecture | Activation | Parameter Space | Neuromanifold Defining Equation | Singular Points |
|---|---|---|---|---|
| MLP (2,2,1) | $\sigma(x) = x^3 + x^2$ | $\mathbb{R}^6$ | Hypersurface $F = 0 \subset \mathbb{R}^7$ | $\nabla F = 0$ corresponds exactly to subnetworks |
| CNN stride=2, $k_1=3, k_2=2$ | $\sigma(y) = y^2 + y$ | $\mathbb{R}^5$ | — | Regular + almost everywhere injective |

### Comparison of Main Theoretical Results

| Property | MLP | CNN |
|---|---|---|
| Generic identifiability | Finite fiber (finitely many parameters per function) | Unique (injective map) |
| Neuromanifold dimension | $= \sum d_i d_{i-1}$ (number of parameters) | $= \sum k_i$ (sum of filter sizes) |
| Subnetworks are singular points? | Yes (under bottleneck conditions) | Yes (complete characterization, if and only if) |
| Type of singularity | Cusp type (Jacobian rank drop) | Node type (self-intersection) |
| Subnetworks critically exposed? | Yes | No |
| Sparsity bias? | Yes (geometric explanation) | No (consistent with empirical findings) |

### Ablation: Degree $r$ Requirements

| Condition | Requirement |
|---|---|
| MLP identifiability | $r > (6m)^{2(L-1)^{L-1}}$, $m = 2\max\{d_1, \ldots, d_{L-1}\}$ |
| MLP singular points | Nonzero coefficients of $\sigma$ $> \dim(\mathcal{M}) + 1$ |
| CNN identifiability | $r \gg 0$ (depending on $L$), $\sigma(0) = 0$ |

### Key Findings

- The dimension of the MLP neuromanifold equals exactly the number of parameters, resolving the dimension conjecture of Kileel et al. (2019).
- Sparse subnetworks of MLPs are singular points of the neuromanifold and are critical points of the MSE loss with positive probability—attracting SGD trajectories.
- Although sparse subnetworks of CNNs are also singular points, they are not critically exposed; hence CNNs do not exhibit the same sparsity bias—consistent with the empirical findings of Blumenfeld et al. (2020).
- The types of singularities in the two architectures are fundamentally different: MLPs exhibit cusp-type singularities (Jacobian rank drop), while CNNs exhibit node-type singularities (self-intersections with the parameterization remaining regular).

## Highlights & Insights

- **From Empirical Observation to Theorem**: The Lottery Ticket Hypothesis and sparsity bias have long been empirical observations. This paper establishes a rigorous causal chain: sparse subnetworks → neuromanifold singular points → critical points of the loss (critical exposure) → SGD attraction (dynamic stability). This provides an entirely new geometric perspective for understanding implicit regularization in neural networks.
- **Deep Architectural Differences Between MLPs and CNNs**: Weight sharing in CNNs eliminates neuron permutation symmetry, transforming the parameterization from many-to-one to one-to-one, changing singularity types from cusps to nodes, and thereby eliminating critical exposure. This demonstrates that subtle differences in architectural design lead to fundamental differences in learning behavior—not only in terms of expressive power, but also in optimization dynamics.
- **The Power of Algebraic Geometry**: The irreducibility of Zariski topology (every nonempty open set is dense) enables arguments of the form "verifying a single example suffices to establish the generic case." Toric geometry and Smith normal forms are precisely applied to eliminate scaling symmetries. This highlights the immense potential of the field of neuroalgebraic geometry.

## Limitations & Future Work

- **Complete Characterization of MLP Singularities**: Theorem 4.2 only proves that subnetworks produce singular points but does not rule out the existence of other singularities. A complete characterization beyond the small examples in the appendix remains an open problem.
- **Critical Point Types Not Distinguished**: The paper only analyzes whether a point is critical, without distinguishing local minima, maxima, and saddle points. Since only local minima are true attractors of gradient flow, this distinction is essential for understanding the strength of the sparsity bias.
- **Large Degree Requirements**: Theorem 4.1 requires polynomial degree $r > (6m)^{2(L-1)^{L-1}}$, which is an extremely loose bound. Although the scope of the results can be argued via polynomial approximation, a rigorous functional-analytic argument remains to be completed (Remark 4.1 provides only a sketch).
- **Single-Channel CNN Limitation**: The paper only considers single-channel CNNs and does not address multi-channel convolutions, multi-head attention, or other more complex architectures.

## Related Work & Insights

- **vs. Identifiability for Monomial Activations (Finkel et al., 2024; Usevich et al., 2025)**: For monomial activations $\sigma(x) = x^r$, MLPs have infinite fibers due to per-neuron scaling, and only the dimension-equals-parameter-count result can be established. This paper uses the multiple nonzero coefficients of polynomial activations to provide additional constraint equations, "cutting" the infinite fiber down to a finite set.
- **vs. Singular Learning Theory (Watanabe, 2009)**: Singular points in SLT are parameters where the Fisher information matrix degenerates, whereas the singular points in this paper are algebraic-geometric singularities of the neuromanifold itself. The two notions are distinct and neither implies the other, yet both point to the core insight that singular structures influence learning.
- **vs. Sparsity Bias Research (Chen et al., 2023; Woodworth et al., 2020)**: Prior work analyzes implicit sparse regularization in simple models (e.g., deep diagonal linear networks) using dynamical systems theory. This paper adopts a geometric rather than dynamical perspective and applies to more general deep polynomial networks.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Simultaneously resolves the dimension conjecture and provides a geometric explanation for sparsity bias; introduces the novel concept of "critical exposure"
- **Experimental Thoroughness**: ⭐⭐ Purely theoretical work; validation limited to small-scale symbolic computation in the appendix
- **Writing Quality**: ⭐⭐⭐⭐⭐ Mathematically rigorous, definitions are clear, theorems build logically on one another, and proof sketches effectively guide readers to the core ideas
- **Value**: ⭐⭐⭐⭐ Provides deep theoretical foundations for understanding the learning process of neural networks; the geometric explanation of MLP vs. CNN differences offers meaningful guidance for architectural design

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Entropic Confinement and Mode Connectivity in Overparameterized Neural Networks](entropic_confinement_and_mode_connectivity_in_overparameterized_neural_networks.md)
- [\[ICLR 2026\] On the Lipschitz Continuity of Set Aggregation Functions and Neural Networks for Sets](on_the_lipschitz_continuity_of_set_aggregation_functions_and_neural_networks_for.md)
- [\[ICLR 2026\] Addressing Divergent Representations from Causal Interventions on Neural Networks](addressing_divergent_representations_causal.md)
- [\[ICLR 2026\] Directional Sheaf Hypergraph Networks: Unifying Learning on Directed and Undirected Hypergraphs](directional_sheaf_hypergraph_networks_unifying_learning_on_directed_and_undirect.md)
- [\[ICLR 2026\] Training Deep Normalization-Free Spiking Neural Networks with Lateral Inhibition](training_deep_normalization-free_spiking_neural_networks_with_lateral_inhibition.md)

</div>

<!-- RELATED:END -->

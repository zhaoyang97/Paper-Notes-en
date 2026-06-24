---
title: >-
  [Paper Note] Algebra Unveils Deep Learning -- An Invitation to Neuroalgebraic Geometry
description: >-
  [ICML 2025][LLM Pretraining][Neuroalgebraic geometry] This paper proposes **neuroalgebraic geometry** as a new research direction, systematically utilizing tools from algebraic geometry (dimension, degree, singularities, fibers, critical point theory, etc.) to analyze the function spaces parameterized by deep learning models (neuromanifolds). It establishes a dictionary mapping algebraic geometric invariants to core machine learning problems (sample complexity, expressivity…
tags:
  - "ICML 2025"
  - "LLM Pretraining"
  - "Neuroalgebraic geometry"
  - "Neuromanifold"
  - "Algebraic geometry"
  - "Deep learning theory"
  - "Semi-algebraic variety"
date: 2026-05-08
content_hash: 23e68e909c693bd0
---

# Algebra Unveils Deep Learning -- An Invitation to Neuroalgebraic Geometry

**Conference**: ICML 2025  
**arXiv**: [2501.18915](https://arxiv.org/abs/2501.18915)  
**Code**: None (theoretical position paper)  
**Area**: LLM Pre-training  
**Keywords**: Neuroalgebraic geometry, Neuromanifold, Algebraic geometry, Deep learning theory, Semi-algebraic variety

## TL;DR

This paper proposes **neuroalgebraic geometry** as a new research direction, systematically utilizing tools from algebraic geometry (dimension, degree, singularities, fibers, critical point theory, etc.) to analyze the function spaces parameterized by deep learning models (neuromanifolds). It establishes a dictionary mapping algebraic geometric invariants to core machine learning problems (sample complexity, expressivity, training dynamics, and implicit bias).

## Background & Motivation

Parameterized machine learning models (such as neural networks) define a function space as parameters vary, referred to as the **neuromanifold**. The geometric properties of this manifold are closely related to core ML questions:

1. **Dimension** governs the statistical properties (sample complexity) and computational properties (expressivity) of the model.
2. The training process can be interpreted as minimizing function distances on the neuromanifold; thus, the **nearest point problem** directly governs training dynamics.
3. Existing methods (information geometry, NTK, geometric deep learning, etc.) have their own limitations: information geometry cannot handle singular spaces, and the NTK relies on the linearization of the infinite-width limit.

The **unique advantages** of algebraic geometry compared to differential geometry lie in:
- Providing the **degree**, an invariant that does not exist in differential geometry.
- Naturally handling **singular spaces** (neuromanifolds are often non-smooth).
- **Metric algebraic geometry** provides precise tools for distance optimization problems.

The authors argue that although polynomial activation functions are rarely used in practice, algebraic models can arbitrarily approximate continuous models via the Weierstrass approximation theorem, rendering the algebraic framework universally significant.

## Method

### Overall Architecture

The core contribution of this work is establishing an **"Algebraic Geometry $\leftrightarrow$ Machine Learning" dictionary** (Table 1), systematically mapping key concepts in ML to the language of algebraic geometry:

| Machine Learning Concept | Algebraic Geometry Counterpart |
|---|---|
| Sample Complexity & Expressivity | Dimension, degree, covering number |
| Subnetworks & Implicit Bias | Singularities |
| Identifiability & Invariance | Fibers of parameterized mappings |
| Optimization & Gradient Descent | Critical point theory, discriminants, dynamical invariants |

**Core Object — Definition of the Neuromanifold**: Given a parameterized model $f: \mathcal{W} \times \mathcal{X} \to \mathcal{Y}$, the neuromanifold is defined as:

$$\mathcal{M} := \{f_w : \mathcal{X} \to \mathcal{Y} \mid w \in \mathcal{W}\}$$

which is the image of the parameterization map $\varphi: \mathcal{W} \to \mathcal{M}$, $w \mapsto f_w$. When the model is polynomial with respect to both parameters and inputs (an **algebraic model**), $\mathcal{M}$ is a **semi-algebraic variety** by the Tarski-Seidenberg theorem.

**Typical Examples of Algebraic Models**:

- **MLP with polynomial activation**: When $\sigma(z) = z^k$, $f_w = W_L \circ \sigma \circ W_{L-1} \circ \sigma \cdots \circ W_1$ is polynomial with respect to both $(x, w)$.
- **Shallow networks** ($L=2$): The neuromanifold corresponds to the space of symmetric tensors with bounded Waring rank.
- **Linear networks** ($k=1$): The neuromanifold is the space of bounded-rank matrices (determinantal varieties).
- **CNNs**: The neuromanifold is linearly birationally equivalent to the Segre-Veronese variety.
- **Linear attention mechanisms**: As cubic layers, their neuromanifold behavior is similar to CNNs.

### Key Designs

This work unfolds algebraic geometric tools across four main themes:

#### 1. Dimension, Degree, and Covering Number

**Dimension** is the most natural numerical invariant, reflecting the intrinsic degrees of freedom of a model and serving as a more accurate measure of expressivity than parameter counting. In the algebraic setting, it can be computed precisely using symbolic methods.

**Degree** is an invariant unique to algebraic geometry, measuring how much a variety "bends" in its ambient space. Formally, $d$ equals the number of (complex) intersection points between the variety and a codimension-matching affine subspace.

Together, they constrain the **covering number**:

$$\log \mathcal{N}_\varepsilon(\mathcal{M}) = \mathcal{O}\left(m \log \frac{d}{\varepsilon} + C\right)$$

where $m$ and $d$ are the dimension and degree of $\mathcal{M}$, respectively. The covering number plays two critical roles in ML:

- **Approximation Expressivity**: The volume of the tubular neighborhood $\mathcal{M}_\varepsilon$ satisfies $\text{Vol}(\mathcal{M}_\varepsilon) \leq \mathcal{N}_\varepsilon(\mathcal{M}) \cdot \omega_{2\varepsilon}$, measuring the size of the function set that the model can approximate within an error of $\varepsilon$.
- **Sample Complexity**: According to classic results in statistical learning theory, the number of samples $|D|$ required for generalization is logarithmically proportional to $\mathcal{N}_\varepsilon(\mathcal{M})$, and is thus directly related to the dimension and degree.

#### 2. Singularities and Implicit Bias

Algebraic varieties (unlike smooth manifolds) can have **singularities** — locations where the dimension of the tangent space is higher than at generic points. Singularities introduce **implicit bias** during training via the following mechanisms:

- The **Voronoi cells** at singularities can exceed the codimension of $\mathcal{M}$, meaning that a larger proportion of target goals have a singularity as their closest point in function space.
- Singularities can act as attractors or slowing-down regions for the optimization flow.
- Key observation: the **singularities of a neuromanifold typically correspond to smaller subnetworks** (lower-rank structures), implying an "automatic model selection" effect during training that favors simpler models.
- This aligns with empirical observations of the Lottery Ticket Hypothesis (where pruning/distillation can recover performance).

Typical example: In a linear two-layer MLP, the singularities of the determinantal variety $\{M \in \mathbb{R}^{N_2 \times N_0} : \text{rank}(M) \leq N_1\}$ are exactly the matrices of rank $< N_1$, corresponding to subnetworks with narrower bottlenecks.

#### 3. Parameterization and Fibers

The **fibers** $\varphi^{-1}(f) = \{w \in \mathcal{W} : f_w = f\}$ of the parameterization map $\varphi$ formalize the problem of **identifiability** in ML.

- **Fiber-Dimension Theorem**: $\dim(\mathcal{M}) = \dim(\mathcal{W}) - \dim(\text{generic fiber})$
- **Fibers and Invariances**: For models satisfying the equivariance property $f_w(Tx) = f_{T \cdot w}(x)$, $f_w$ is invariant to input transformation $T$ if and only if $w$ and $T \cdot w$ reside in the same fiber.
- **Fiber Structure of MLPs**: This includes permutation symmetries of hidden neurons and scaling symmetries under homogeneous activations. For high-degree polynomial activations ($r \gg 0$), it has been shown that these symmetries completely describe the generic fiber.

#### 4. Critical Points and Gradient Descent

The analysis of critical points of the loss function $L_\mathcal{D} = \mathcal{L}_\mathcal{D} \circ \varphi$ connects algebraic geometry with training dynamics:

**Spurious Critical Points**: Parameters $w \in \text{Crit}(\varphi)$ might be critical points in the parameter space, but the corresponding function $f_w$ is not a critical point in the function space. Different architectures behave differently:
- MLP (monomial activation): Can exist as local minima with positive probability.
- CNN: Only corresponds to the zero function $0 \in \mathcal{M}$.

**Euclidean Distance Degree (EDD)**: For an algebraic neuromanifold, the number of complex critical points of $\|{\cdot} - f^*\|^2$ on $\mathcal{M}$ is constant, defining the EDD invariant which quantifies the complexity of the distance minimization problem.

**Data Discriminants**: The number and types of real critical points change abruptly when $f^*$ crosses algebraic hypersurfaces (discriminants) in $\mathcal{V}$, describing the topological transitions of the loss landscape.

**Dynamical Invariants**: Parameter relations conserved by the gradient flow, which can be used to design good initialization strategies (such as balanced initialization to achieve stable learning dynamics).

### Loss & Training

From the perspective of neuromanifolds, ERM is equivalent to constrained optimization in the function space:

$$\min_{f \in \mathcal{M}} \mathcal{L}_\mathcal{D}(f)$$

- For **quadratic loss**: ERM degenerates to the minimization of a (degenerate) quadratic form on $\mathcal{M}$, with the generalization error being equal to $\|f - f^*\|_{L^2}^2$.
- For algebraic losses (such as Wasserstein distance or cross-entropy), the algebraic structure of critical points can be analyzed using tools like the **maximum likelihood degree**.
- Training is essentially the nearest-point problem in **metric algebraic geometry**.

### Beyond Algebra: Extending to Non-Algebraic Models

- **Polynomial Approximation**: By the Weierstrass approximation theorem, the neuromanifold of an MLP with arbitrary continuous activation can be arbitrarily approximated under the Hausdorff distance by an algebraic neuromanifold. This has been applied to generalize covering number bounds of algebraic models to ReLU networks.
- **Tropical Geometry**: The neuromanifolds of piecewise-linear activations like ReLU can be directly analyzed using tropical geometry (min/max algebra). Tropical geometry is closely linked to combinatorics and polyhedral geometry, providing discrete mathematical tools.

## Key Experimental Results

As a position paper, this work lacks traditional numerical experiments, but provides rigorous theoretical results and detailed analysis of toy examples.

### Main Results: Complete Analysis of Linear Two-Layer MLPs

The paper provides closed-form formulas for all algebraic geometric quantities of a linear two-layer MLP $f_w(x) = W_1 W_0 x$ in the appendix:

| Invariant | Formula | Machine Learning Meaning |
|---|---|---|
| Dimension | $\dim(\mathcal{M}) = N_1(N_0 + N_2 - N_1)$ | Model's effective degrees of freedom |
| Degree | $\deg(\mathcal{M}) = \prod_{0 \leq i < \min\{N_0,N_2\}-N_1} \frac{\binom{\max\{N_0,N_2\}+i}{N_1}}{\binom{N_1+i}{N_1}}$ | Complexity upper bound |
| Singularities | Matrices of rank $< N_1$ | Subnetworks with narrower bottlenecks |
| Generic Fiber | $\cong GL(\mathbb{R}, N_1)$ | Ambiguity of invertible transformations between layers |
| EDD | $\binom{\min\{N_0,N_2\}}{N_1}$ | Upper bound on critical points of distance optimization |
| Optimal Solution | Eckart-Young-Schmidt Theorem | Projection onto the top $N_1$ singular values |

### Ablation Study: Comparison of Neuromanifolds across Different Architectures

| Architecture | Neuromanifold Type | Spurious Critical Points | Singularity = Subnetwork? |
|---|---|---|---|
| Linear MLP | Determinantal variety | Yes, can be local minima | Yes (low-rank matrices) |
| Polynomial MLP | Bounded Waring rank tensor space | Local minima with positive probability | Yes |
| Polynomial CNN | $\approx$ Segre-Veronese variety | Only the zero function | Yes |
| Linear Attention | Cubic layer variety | Open problem | Similar to CNNs |
| ReLU MLP | Piecewise linear (Tropical geometry) | N/A (not fully algebraic) | Partially applicable |

### Key Findings

1. **Sample Complexity Bound**: The upper bound of samples required for generalization is $\mathcal{O}\left(\frac{D^2}{\varepsilon^2}\left(m \log \frac{d}{\varepsilon} + C + \log\frac{1}{\delta}\right)\right)$, determined jointly by the dimension $m$ and the degree $d$.
2. **Singular Bias**: Singularities of the neuromanifold correspond to subnetworks; there is an implicit bias during training that automatically selects simpler models.
3. **Architecture Dependency of Spurious Critical Points**: MLPs can produce spurious local minima with positive probability, whereas CNNs do not (except for the zero function).
4. **Dynamical Invariants** can be used to design parameter initialization strategies, preventing subsequent training from getting trapped in suboptimal regions.

## Highlights & Insights

1. **Unification of Conceptual Framework**: Under a concise "dictionary," scattered works in deep learning theory (NTK, singular learning theory, geometric deep learning, etc.) are integrated into a unified algebraic geometric framework.
2. **Discovery of the Degree**: The degree is an invariant unique to algebraic geometry (non-existent in differential geometry) that plays a decisive role in sample complexity — a nuance that information geometry cannot reveal.
3. **The "Singularities = Subnetworks" Insight**: This elegant insight elevates empirical phenomena like the Lottery Ticket Hypothesis to the rigorous language of algebraic geometry.
4. **Introduction of Data Discriminants**: This provides a refreshing perspective for understanding the topological transitions of the loss landscape.
5. **Universality of the Algebraic Framework**: Demonstrated through Weierstrass approximation and tropical geometry, showing its applicability beyond polynomial activations.

## Limitations & Future Work

1. **Theory-Heavy**: This work lacks numerical validation on large-scale practical networks, as all explicit computations are limited to toy examples (e.g., linear two-layer MLPs).
2. **Limited Architectural Coverage**: Mainstream architectures such as skip connections, GNNs, SSMs, and Transformers (with softmax) are not yet integrated into the analysis.
3. **Computational Complexity**: The feasibility of symbolic methods for computing dimension and degree in high-dimensional settings remains undiscussed.
4. **Gap with Practice**: Polynomial activations are rarely used in actual models; whether the error bounds of approximation arguments are tight enough warrants further study.
5. **Abundance of Open Questions**: Crucial problems, such as characterizing spurious critical points in attention mechanisms and the smoothing effect of skip connections on singularities, remain unsolved.

## Related Work & Insights

- **Information Geometry** (Amari, 2016): Also studies the geometry of neuromanifolds but based on a Riemannian framework; it cannot handle degrees and singularities.
- **Singular Learning Theory** (Watanabe, 2009): Studies singularities starting from algebraic geometry and statistical machine learning; neuroalgebraic geometry serves as a natural extension.
- **NTK** (Jacot et al., 2018): Linearizes under the infinite-width limit, which is complementary to the finite-dimensional, non-linear approach of this work.
- **Geometric Deep Learning** (Bronstein et al., 2021): Focuses on symmetry, where fiber analysis provides an algebraic-geometric characterization of invariants.
- **Metric Algebraic Geometry** (Breiding et al., 2024): The source of the core mathematical tools.
- **Algebraic Statistics** (Pistone et al., 2000): This work can be seen as the counterpart of algebraic statistics in the ML domain.

**Insights**: This position paper opens up a highly structured mathematical framework for deep learning theory. For AutoML and neural architecture search, one might consider using algebraic-geometric invariants (dimension, degree) as architecture evaluation metrics; for training stability research, data discriminants could serve as a powerful tool for predicting hard-to-train regions.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First to systematically propose a complete dictionary mapping deep learning to algebraic geometry, opening up a brand-new research direction.
- Experimental Thoroughness: ⭐⭐⭐ — Reasonable for a position paper, but only contains closed-form solutions for toy examples, lacking numerical experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear and elegant, presenting profound algebraic-geometric concepts in a manner accessible to ML researchers.
- Value: ⭐⭐⭐⭐⭐ — Establishes a foundational framework that is poised to catalyze a wealth of subsequent theoretical work, carrying profound significance for the interpretability of deep learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] The Double-Ellipsoid Geometry of CLIP](the_double-ellipsoid_geometry_of_clip.md)
- [\[NeurIPS 2025\] Deep Compositional Phase Diffusion for Long Motion Sequence Generation](../../NeurIPS2025/llm_pretraining/deep_compositional_phase_diffusion_for_long_motion_sequence_generation.md)
- [\[ICLR 2026\] Intrinsic Training Dynamics of Deep Neural Networks](../../ICLR2026/llm_pretraining/intrinsic_training_dynamics_of_deep_neural_networks.md)
- [\[NeurIPS 2025\] Predict Training Data Quality via Its Geometry in Metric Space](../../NeurIPS2025/llm_pretraining/predict_training_data_quality_via_its_geometry_in_metric_space.md)
- [\[ICML 2025\] On the Role of Label Noise in the Feature Learning Process](on_the_role_of_label_noise_in_the_feature_learning_process.md)

</div>

<!-- RELATED:END -->

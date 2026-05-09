---
title: >-
  [Paper Note] RNNs Perform Task Computations by Dynamically Warping Neural Representations
description: >-
  [NeurIPS 2025][RNN] This paper proposes a Riemannian geometric framework that pulls back the metric from the RNN state space onto the input manifold, demonstrating that RNNs perform computation by dynamically warping their representations of task variables—compressing task-irrelevant inputs and stretching space near decision boundaries. Crucially, this warping is not a byproduct of computation but constitutes computation itself.
tags:
  - NeurIPS 2025
  - RNN
  - Riemannian geometry
  - representational geometry
  - dynamical systems
  - manifold warping
date: 2026-05-08
content_hash: 734f3ffdbe9689c9
---

# RNNs Perform Task Computations by Dynamically Warping Neural Representations

**Conference**: NeurIPS 2025
**arXiv**: [2512.04310](https://arxiv.org/abs/2512.04310)
**Code**: None
**Area**: Computational Neuroscience / Dynamical Systems
**Keywords**: RNN, Riemannian geometry, representational geometry, dynamical systems, manifold warping

## TL;DR

This paper proposes a Riemannian geometric framework that pulls back the metric from the RNN state space onto the input manifold, demonstrating that RNNs perform computation by dynamically warping their representations of task variables—compressing task-irrelevant inputs and stretching space near decision boundaries. Crucially, this warping is not a byproduct of computation but constitutes computation itself.

## Background & Motivation

**State of the Field**: Understanding how neural networks represent data features through internal activations—i.e., the geometric structure of "neural representations"—is a central problem in machine learning and computational neuroscience. A large body of work focuses on static representational geometry (e.g., manifold analysis across layers of deep networks), while another line investigates how dynamical systems perform computation through time-varying dynamics (computation-through-dynamics).

**Limitations of Prior Work**: The connection between these two directions remains weak. Existing tools for representational geometry analysis are designed primarily for feedforward networks with static inputs and cannot handle dynamical systems receiving time-varying inputs, such as RNNs. Existing methods for analyzing RNN computation largely rely on linearization near fixed points, which can only characterize steady-state behavior and discard critical computational information encoded in transient dynamics.

**Root Cause**: RNN computation unfolds across the entire time axis—including transients far from attractors—yet existing mathematical tools can only characterize local behavior near steady states. A theoretical framework is needed that can characterize the complete, time-varying geometry of a dynamical system's representational manifold as it receives time-varying inputs.

**Paper Goals**: How can the topology and geometry of a dynamical system's state manifold be derived from the manifold of input functions? How does this geometry evolve over time? What is the relationship between this geometric evolution and computation?

**Starting Point**: The authors hypothesize that RNNs accomplish computation by dynamically warping their representations of task variables, and quantify this warping by introducing a "representational metric"—the pullback metric of the RNN state space metric onto the input manifold.

**Core Idea**: Define a time-varying pullback metric for RNNs to quantify how the intrinsic geometry of the representational manifold is dynamically deformed throughout the course of computation.

## Method

### Overall Architecture

The framework operates at three levels: (1) proving that if time-varying input functions lie on an $m$-dimensional manifold, the system's states are constrained to a manifold of at most $m+1$ dimensions (a topological theorem); (2) defining the "representational metric" on this state manifold as the pullback metric from state space to the input manifold, computed by solving an adjoint differential equation; and (3) applying this framework to RNNs trained on three classes of tasks to reveal the universality of dynamic warping.

### Key Designs

1. **Topological Constraint from Input Manifold to State Manifold (Theorem 3.1)**:

    - Function: Establishes a rigorous relationship between the dimensionality of the input function manifold and that of the RNN state manifold.
    - Mechanism: If the time-varying input function $u(t)$ lies on an $m$-dimensional manifold $\mathcal{M}$ (where each point corresponds to a distinct time-varying function), then the RNN state trajectories over a finite time horizon are constrained to a manifold of at most $m+1$ dimensions. The key insight is that dimensionality is defined over the space of input functions, not over the instantaneous input state space at any given moment.
    - Design Motivation: Conventional analysis focuses on the instantaneous dimensionality of inputs in state space, whereas this paper considers the dimensionality of the input function manifold—a non-trivial distinction, since in a controllable system a single input function can drive the system to any state.

2. **Representational Metric and Pullback Construction (Theorem 3.3/3.4)**:

    - Function: Quantitatively characterizes the time-varying intrinsic geometry of the RNN state manifold.
    - Mechanism: The metric tensor is defined as $G_{ij} = \partial_{u_i} x \cdot \partial_{u_j} x$, where $\partial_{u_i} x$ is the partial derivative of the system state with respect to the $i$-th input parameter. The diagonal entries $G_{ii}$ characterize the degree of stretching or compression along that direction, while off-diagonal entries capture correlations between different input encodings. This metric and its temporal evolution can be computed efficiently by solving an adjoint ODE.
    - Design Motivation: The metric is naturally induced by the Euclidean structure of state space rather than being chosen arbitrarily, and thus captures the true geometry of the representational manifold embedded in high-dimensional state space.

3. **Causal Validation—Necessity and Sufficiency of Warping**:

    - Function: Demonstrates that warping is not a byproduct of computation but constitutes computation itself.
    - Mechanism: In a context-dependent decision task, (a) constraining the RNN to prevent warping (by enforcing a ratio of unity between diagonal metric entries) causes the model to fail to converge; (b) training the RNN solely to perform warping (without a task loss) achieves performance approaching that of the fully trained model. This establishes warping as both necessary and sufficient for computation.
    - Design Motivation: Responds to concerns about whether the observed geometric changes are merely correlational.

### Loss & Training

RNNs are trained with standard MSE loss on specific tasks (context-dependent decision-making, working memory, BCI decoding, etc.), after which the proposed framework is applied as a post-hoc analysis tool. In the causal experiments, a metric ratio constraint term is added to the training loss.

## Key Experimental Results

### Main Results

| Task | Model | Input Manifold | Key Findings |
|------|-------|---------------|--------------|
| Context-dependent decision | vanilla RNN | 2D (two stimulus angles) | Irrelevant input dimensions are compressed; space near decision boundaries is stretched |
| Working memory | vanilla RNN | 2D torus (two memory items) | Torus dynamically warps during encoding; geometry stabilizes during delay |
| Memory subtraction | vanilla RNN | 2D→1D | Manifold dynamically collapses from 2D to near-1D to encode the difference |
| BCI decoding | SSM (POSSM) | $S^1 \times \mathbb{R}$ (direction × speed) | Faster cursor movements correspond to accelerated neural activity along trajectories |

### Ablation Study

| Configuration | Test MSE | Notes |
|--------------|---------|-------|
| Full model (baseline) | 0.001 | Normally trained RNN |
| Warping constrained ($c=1$) | Fails to converge | Cannot perform task; warping is necessary |
| Warping-only training ($c=c^*$) | ~0.002 | Approaches baseline; warping is sufficient |

### Key Findings

- In the context-dependent decision task, RNNs not only compress representations of irrelevant stimuli (a previously known phenomenon) but also stretch the space of relevant stimuli near decision boundaries—a previously unreported finding.
- The non-zero curvature of the torus in the working memory task indicates that the classical notion of "orthogonal coding" is insufficiently precise.
- Warping patterns are highly consistent across different nonlinear activation functions (Tanh, ReLU, Softplus, GeLU), with geodesic distance differences of less than 0.023.
- The framework generalizes to other dynamical system architectures such as SSMs.

## Highlights & Insights

- **The causal validation from correlation to causation** is the paper's most significant contribution. By constraining the metric to test the necessity and sufficiency of warping, the work elevates "observing warping" to "warping as computation"—a methodological innovation in representational geometry analysis in its own right.
- **Defining dimensionality in function space rather than state space** is non-trivial. A one-dimensional input function can drive a controllable system to any state, yet from the function-space perspective it constrains only a two-dimensional manifold—this distinction is critical.
- The framework transfers to any dynamical system (SSMs, Neural ODEs, or even autoregressive Transformers) for which a meaningful input function manifold can be defined.

## Limitations & Future Work

- Experiments are primarily conducted on small-scale tasks from computational neuroscience; the framework has not been demonstrated on large-scale ML tasks such as language modeling.
- A meaningful input manifold of interest (e.g., "different directions and speeds") must be specified in advance, which may not be straightforward for more complex tasks.
- Computing the metric requires solving an adjoint ODE, which may pose computational bottlenecks for large-scale models.
- Causal validation is performed on only one task; generalizability remains to be verified.

## Related Work & Insights

- **vs. Fixed-point/attractor analysis (Mante et al. 2013)**: Traditional methods linearize dynamics near fixed points and can only characterize steady-state behavior. This paper analyzes the time-varying geometry of complete nonlinear dynamics and finds that warping begins early in transient dynamics—well before convergence to any fixed point.
- **vs. Static pullback analysis (Hauser & Ray 2017)**: Prior work applied pullback metrics to feedforward networks with static inputs; this paper extends the approach to dynamical systems receiving time-varying inputs, requiring the solution of an adjoint ODE.
- **vs. Low-rank RNNs (Valente et al. 2022)**: Low-rank RNNs constrain the embedding dimensionality, whereas this paper constrains the intrinsic dimensionality; the two are complementary—the former provides a linear embedding space, while the latter characterizes the geometry of the nonlinear manifold.

## Rating

- Novelty: ⭐⭐⭐⭐ Extending Riemannian geometry to time-varying dynamical systems constitutes a meaningful theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐ Multiple tasks but relatively small scale; the BCI experiment strengthens the case.
- Writing Quality: ⭐⭐⭐⭐⭐ The paper is clearly and elegantly written, with excellent figures and well-calibrated intuitive explanations.
- Value: ⭐⭐⭐⭐ Provides new mathematical tools for understanding RNN computation, though the target audience is relatively narrow.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Learning Dynamics of RNNs in Closed-Loop Environments](learning_dynamics_of_rnns_in_closed-loop_environments.md)
- [\[ICLR 2026\] Addressing Divergent Representations from Causal Interventions on Neural Networks](../../ICLR2026/others/addressing_divergent_representations_causal.md)
- [\[NeurIPS 2025\] An Empirical Investigation of Neural ODEs and Symbolic Regression for Dynamical Systems](an_empirical_investigation_of_neural_odes_and_symbolic_regression_for_dynamical_.md)
- [\[NeurIPS 2025\] Depth-Bounds for Neural Networks via the Braid Arrangement](depth-bounds_for_neural_networks_via_the_braid_arrangement.md)
- [\[NeurIPS 2025\] Learning to Condition: A Neural Heuristic for Scalable MPE Inference](learning_to_condition_a_neural_heuristic_for_scalable_mpe_inference.md)

<!-- RELATED:END -->

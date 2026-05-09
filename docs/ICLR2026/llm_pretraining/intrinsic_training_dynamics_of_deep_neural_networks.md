---
title: >-
  [Paper Note] Intrinsic Training Dynamics of Deep Neural Networks
description: >-
  [ICLR 2026][LLM Pretraining][intrinsic dynamics] This paper investigates when the trajectory in parameter space under gradient flow training of deep neural networks can be "lifted" to a low-dimensional intrinsic space and expressed as an intrinsic Riemannian gradient flow. It proposes an intrinsic recoverability criterion based on conservation laws and extends the results to ReLU networks and linear networks of arbitrary depth.
tags:
  - ICLR 2026
  - LLM Pretraining
  - intrinsic dynamics
  - gradient flow
  - conservation laws
  - implicit bias
  - Riemannian metric
date: 2026-05-08
content_hash: 8912533a9ffbb042
---

# Intrinsic Training Dynamics of Deep Neural Networks

**Conference**: ICLR 2026
**arXiv**: [2508.07370](https://arxiv.org/abs/2508.07370)
**Code**: None
**Area**: Deep Learning Theory / Optimization Dynamics
**Keywords**: intrinsic dynamics, gradient flow, conservation laws, implicit bias, Riemannian metric

## TL;DR

This paper investigates when the trajectory in parameter space under gradient flow training of deep neural networks can be "lifted" to a low-dimensional intrinsic space and expressed as an intrinsic Riemannian gradient flow. It proposes an intrinsic recoverability criterion based on conservation laws and extends the results to ReLU networks and linear networks of arbitrary depth.

## Background & Motivation

**Core problem of implicit bias**: Understanding whether gradient-based training drives parameters toward certain low-dimensional structures (sparsity, low rank, etc.)—the so-called implicit bias—is one of the central challenges in deep learning theory.

**Lifted variable framework**: Many analyses "lift" the parameter $\theta$ via an architecture-dependent map $\phi$ to $z = \phi(\theta)$, e.g., $\phi(\theta) = U_L \cdots U_1$ for linear networks and $\phi(\theta) = (u_j v_j^\top)_j$ for ReLU networks.

**Importance of intrinsic dynamics**: If $z(t)$ can be shown to follow an intrinsic Riemannian gradient flow, tools from convex optimization (e.g., mirror flow) become applicable for analyzing implicit regularization.

**Limitations of Prior Work**: The commuting condition of Li et al. (2022) is rarely satisfied in practice; the involutive condition of Marcotte et al. (2023) applies only to two-layer ReLU networks.

**Restriction of balanced initialization**: Prior results on intrinsic dynamics for linear networks rely on the strict balanced initialization condition $U_{i+1}^\top U_{i+1} = U_i U_i^\top$.

**Lack of unified theory**: A unified framework for intrinsic dynamics analysis is missing for deep ReLU networks with general DAG architectures, linear networks with unbalanced initialization, and infinitely deep linear networks.

## Method

### Overall Architecture

The paper establishes three progressively stronger notions of intrinsic properties and their implications:

$$\text{Intrinsic Recoverability} \Rightarrow \text{Intrinsic Metric} \Rightarrow \text{Intrinsic Dynamic}$$

**Core Problem**: For the gradient flow $\dot{\theta}(t) = -\nabla \ell(\theta(t))$, the dynamics of the lifted variable $z(t) = \phi(\theta(t))$ take the form $\dot{z}(t) = -M(\theta(t)) \nabla f(z(t))$, where $M(\theta) = \partial\phi(\theta) \partial\phi(\theta)^\top$. When can $M(\theta(t))$ be expressed solely in terms of $z(t)$ and the initialization $\theta_0$?

### Key Designs

1. **Intrinsic Dynamic Property (Definition 2.6)**

    - **Function**: Defines when $\theta_0$ possesses the intrinsic dynamic property with respect to $\phi$.
    - **Mechanism**: There exists a function $K_{\theta_0}$ such that $M(\theta(t)) = K_{\theta_0}(\phi(\theta(t)))$ holds for all $f$.
    - **Design Motivation**: Decouples the metric matrix from the data/task; $K_{\theta_0}$ depends only on network architecture and initialization.

2. **Intrinsic Metric Property (Definition 2.10)**

    - **Function**: Requires intrinsicness of the metric on the manifold $\mathcal{M}_{\theta_0}$ constrained by conservation laws.
    - **Mechanism**: There exist conservation laws $\mathbf{h}$ and a function $K_{\theta_0}$ such that $M(\theta) = K_{\theta_0}(\phi(\theta))$ holds for all $\theta \in \mathcal{M}_{\theta_0}$.
    - **Design Motivation**: Employs conservation laws to confine trajectories to a low-dimensional manifold.

3. **Intrinsic Recoverability (Definition 2.15) and Equivalence Criterion (Theorem 2.17)**

    - **Function**: Requires that $\theta$ can be fully recovered from $\phi(\theta)$ and $\mathbf{h}(\theta)$.
    - **Mechanism**: Equivalent to the kernel intersection condition $\ker\partial\phi(\theta) \cap \ker\partial\mathbf{h}(\theta) = \{0\}$.
    - **Design Motivation**: The strongest condition, equivalent to a verifiable linear-algebraic criterion.

4. **Frobenius Property for ReLU Networks (Theorem 3.1 & Corollary 3.3)**

    - **Function**: Proves that the path-lifting map for ReLU networks with arbitrary DAG architectures satisfies the Frobenius property.
    - **Mechanism**: Verifies closure under the Lie bracket at nonzero parameters (a dense set).
    - **Design Motivation**: Establishes the strongest intrinsic recoverability for almost all initializations.

5. **Linear Networks under Relaxed Balancedness (Theorem 3.8 & 3.9)**

    - **Function**: Extends the intrinsic metric property from balanced ($\lambda=0$) to relaxed balanced ($S = \lambda I$) initialization.
    - **Mechanism**: Derives closed-form expressions for the intrinsic dynamics and proves that relaxed balancedness is a necessary condition (when $r \leq \max(n,m)$).
    - **Design Motivation**: Establishes necessary and sufficient conditions for intrinsic metric properties in linear networks.

### Loss & Training

This paper is a purely theoretical work analyzing continuous-time gradient flow. The loss function takes the form $\ell(\theta) = f(\phi(\theta))$, where $f$ is an arbitrary differentiable function; the results are independent of specific loss functions and datasets.

## Key Experimental Results

### Main Results

This paper makes purely theoretical contributions; the core results are stated as theorems:

| Network Type | Map $\phi$ | Intrinsic Property | Condition |
|---------|-----------|---------|------|
| Arbitrary DAG ReLU network | $\phi_{\text{ReLU}}$ (path-lifting) | Intrinsic Recoverability ✓ | Nonzero parameters (dense set) |
| Two-layer linear network | $\phi_{\text{Lin}} = UV^\top$ | Intrinsic Metric ✓/✗ | Relaxed balanced ✓ / non-relaxed ✗ |
| Deep linear network | $\phi_{\text{Lin}} = U_L \cdots U_1$ | Intrinsic Dynamic ✓ | Relaxed balancedness condition |
| Linear neural ODE | Infinite-depth limit | Intrinsic Dynamic ✓ | Relaxed balanced + closed-form metric |

### Ablation Study

| Comparison Dimension | Prev. SOTA | Ours |
|---------|---------|---------|
| ReLU network depth | Two layers | Arbitrary DAG architecture |
| Linear network initialization | Strict balanced $\lambda = 0$ | Relaxed balanced $S = \lambda I$ |
| Linear network depth | Two layers | Arbitrary depth + infinite depth |
| Completeness of conservation laws | Empirically verified | Theoretically proven (Corollary 3.4) |

### Key Findings

- ReLU networks satisfy the strongest intrinsic recoverability property for a dense set of initializations.
- The known conservation laws (differences of diagonal terms) are complete for ReLU networks.
- Relaxed balancedness is a necessary and sufficient condition for the intrinsic metric property in linear networks.
- A closed-form expression for intrinsic dynamics is derived for three-layer ReLU networks for the first time.
- Linear neural ODEs under relaxed balanced initialization also admit a closed-form intrinsic metric.

## Highlights & Insights

1. **Unified framework**: The three-level hierarchy of definitions clearly reveals the relative strengths of different intrinsic notions.
2. **Favorable properties of ReLU**: Counterintuitively, the piecewise nonlinear structure of ReLU yields a smaller symmetry group and richer conservation laws, making intrinsic dynamics easier to establish than in linear networks.
3. **Power of the kernel inclusion criterion**: Theorem 2.14 provides a concise tool for proving negative results.
4. **Lie-algebraic criterion**: A practical algebraic test based on the Frobenius property avoids direct construction of conservation laws.
5. **Cross-architecture applicability**: The framework uniformly handles ReLU, linear, attention layers, and infinitely deep networks.

## Limitations & Future Work

1. Only continuous gradient flow is analyzed; discrete optimization algorithms (SGD, Adam) are not covered.
2. Only intrinsic dynamics (step i) are established; the extension to mirror flow (step ii) is not addressed.
3. The case $r > \max(n, m)$ for linear networks remains an open problem.
4. Numerical validation experiments are absent.
5. The Frobenius property does not hold for attention layers, requiring indirect analysis.

## Related Work & Insights

- **Arora et al. (2019)**: Balanced initialization and conservation laws → extended in this paper to relaxed balancedness.
- **Marcotte et al. (2023)**: Involutive condition → weakened in this paper to the Frobenius condition.
- **Li et al. (2022)**: Commuting condition (a special case of Frobenius) → mirror flow.
- **Gonon et al. (2024)**: Path-lifting framework → this paper proves it satisfies the Frobenius property.
- **Insights**: Lays the foundation for subsequent analyses of warped mirror flow and implicit bias in practical architectures.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Establishes a complete hierarchical theory; the universal result for ReLU networks is a significant breakthrough.
- **Experimental Thoroughness**: ⭐⭐⭐ Purely theoretical work; theorems are rigorous but numerical validation is lacking.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Definitions and theorems are presented in a coherent progression; the framework is elegant and clear.
- **Value**: ⭐⭐⭐⭐ Provides an important theoretical foundation for understanding implicit bias.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Neural Collapse under Gradient Flow on Shallow ReLU Networks for Orthogonally Separable Data](../../NeurIPS2025/llm_pretraining/neural_collapse_under_gradient_flow_on_shallow_relu_networks_for_orthogonally_se.md)
- [\[NeurIPS 2025\] Generalization Bounds for Rank-sparse Neural Networks](../../NeurIPS2025/llm_pretraining/generalization_bounds_for_rank-sparse_neural_networks.md)
- [\[ICLR 2026\] MoMa: A Simple Modular Deep Learning Framework for Material Property Prediction](moma_a_modular_deep_learning_framework_for_material_property_prediction.md)
- [\[NeurIPS 2025\] Disaggregation Reveals Hidden Training Dynamics: The Case of Agreement Attraction](../../NeurIPS2025/llm_pretraining/disaggregation_reveals_hidden_training_dynamics_the_case_of_agreement_attraction.md)
- [\[NeurIPS 2025\] Alternating Gradient Flows: A Theory of Feature Learning in Two-layer Neural Networks](../../NeurIPS2025/llm_pretraining/alternating_gradient_flows_a_theory_of_feature_learning_in_two-layer_neural_netw.md)

<!-- RELATED:END -->

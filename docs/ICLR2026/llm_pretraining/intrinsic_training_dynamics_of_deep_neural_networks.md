---
title: >-
  [Paper Note] Intrinsic Training Dynamics of Deep Neural Networks
description: >-
  [ICLR 2026][LLM Pretraining][intrinsic dynamics] This paper investigates when parameter space trajectories in deep neural network gradient flow training can be "lifted" to a low-dimensional intrinsic space and represented as an intrinsic Riemannian gradient flow. It proposes an intrinsic recoverability criterion based on conservation laws and generalizes the results to ReLU networks of arbitrary depth and linear networks.
tags:
  - "ICLR 2026"
  - "LLM Pretraining"
  - "intrinsic dynamics"
  - "gradient flow"
  - "conservation laws"
  - "implicit bias"
  - "Riemannian metric"
date: 2026-05-08
content_hash: 53f3393bab1ca172
---

# Intrinsic Training Dynamics of Deep Neural Networks

**Conference**: ICLR 2026  
**arXiv**: [2508.07370](https://arxiv.org/abs/2508.07370)  
**Code**: None  
**Area**: Deep Learning Theory / Optimization Dynamics  
**Keywords**: intrinsic dynamics, gradient flow, conservation laws, implicit bias, Riemannian metric

## TL;DR

This paper investigates when parameter space trajectories in deep neural network gradient flow training can be "lifted" to a low-dimensional intrinsic space and represented as an intrinsic Riemannian gradient flow. It proposes an intrinsic recoverability criterion based on conservation laws and generalizes the results to ReLU networks of arbitrary depth and linear networks.

## Background & Motivation

**Key Challenge of Implicit Bias**: A core challenge in deep learning theory is understanding whether gradient training drives parameters toward certain low-dimensional structures (sparsity, low-rank, etc.), known as the implicit bias problem.

**Lifting Variable Framework**: Many analyses "lift" parameters $\theta$ into a representation $z = \phi(\theta)$ through an architecture-dependent mapping $\phi$. For example, in linear networks $\phi(\theta) = U_L \cdots U_1$, and in ReLU networks $\phi(\theta) = (u_j v_j^\top)_j$.

**Mechanism of Intrinsic Dynamics**: If $z(t)$ can be proven to follow an intrinsic Riemannian gradient flow, convex optimization tools (such as mirror flow) can be utilized to analyze implicit regularization effects.

**Limitations of Prior Work**: The commuting condition in Li et al. (2022) is rarely satisfied in practice; the involutive condition in Marcotte et al. (2023) applies only to two-layer ReLU networks.

**Constraints of Balanced Initialization**: In linear networks, previous intrinsic dynamics results relied on strict balanced initialization conditions $U_{i+1}^\top U_{i+1} = U_i U_i^\top$.

**Goal**: To provide a unified intrinsic dynamics analysis framework for deep ReLU networks with general DAG architectures, linear networks with unbalanced initialization, and infinite-depth linear networks.

## Method

### Overall Architecture

The paper centers on a unified problem: for a gradient flow $\dot{\theta}(t) = -\nabla \ell(\theta(t))$, after lifting parameters to the representation $z(t) = \phi(\theta(t))$, its dynamics can be written as $\dot{z}(t) = -M(\theta(t)) \nabla f(z(t))$, where the metric matrix $M(\theta) = \partial\phi(\theta)\,\partial\phi(\theta)^\top$. The core question is: when can $M(\theta(t))$ be expressed solely using $z(t)$ and the initialization $\theta_0$? Once this is possible, $z$ follows an intrinsic Riemannian gradient flow $\dot{z} = -K_{\theta_0}(z)\nabla f(z)$, allowing analysis of implicit regularization using convex optimization tools (e.g., mirror flow) independent of the original parameter space. To clarify this, the authors first decompose "intrinsicness" into three hierarchical levels from strong to weak and prove their sequential implication. They then use a linear algebra criterion to transform the strongest level into a point-wise verifiable condition. Finally, they apply this criterion to two mainstream architectures: for ReLU networks of arbitrary depth, they bypass explicit construction of conservation laws using Frobenius/Lie algebra criteria; for linear networks, they relax the traditionally strict balanced initialization to a broader family.

> As this is a purely theoretical work, the method essentially consists of a hierarchy of definitions + algebraic criteria + theoretical derivations for two classes of architectures. There is no multi-module pipeline that can be sequenced as a data flow (the core involves algebraic objects like Jacobian null spaces, Lie brackets, and conservation laws, which are not suitable for flowcharts). Therefore, **no framework diagram is provided**. The three key designs below follow the same sequence as the overall architecture: establishing the hierarchy and criteria, then addressing ReLU and linear networks respectively.

### Key Designs

**1. Three-level Intrinsic Hierarchy and Kernel Intersection Criterion: Reducing Global Dynamics to Point-wise Linear Algebra Checks**

To answer "when is $M$ a function of only $z$," the authors construct a three-level hierarchy and prove $\text{Intrinsic Recoverability} \Rightarrow \text{Intrinsic Metric} \Rightarrow \text{Intrinsic Dynamics}$. The weakest, **intrinsic dynamics property** (Def 2.6), only requires the existence of a function $K_{\theta_0}$, depending only on architecture and initialization, such that $M(\theta(t)) = K_{\theta_0}(\phi(\theta(t)))$ holds along the trajectory for all differentiable losses $f$—thus decoupling the metric from the specific task and dataset, making $z(t)$ a self-consistent low-dimensional system. Since "holding along the trajectory" is hard to verify directly, they introduce the **intrinsic metric property** (Def 2.10): utilizing conservation laws $\mathbf{h}$ naturally carried by gradient flow to confine trajectories to the manifold $\mathcal{M}_{\theta_0} = \{\theta : \mathbf{h}(\theta) = \mathbf{h}(\theta_0)\}$. It then suffices for $M(\theta) = K_{\theta_0}(\phi(\theta))$ to hold over the entire $\mathcal{M}_{\theta_0}$. The strongest level, **intrinsic recoverability** (Def 2.15), requires that $\theta$ can be uniquely recovered from $(\phi(\theta), \mathbf{h}(\theta))$, meaning "lifting + conserved quantities" loses no information. Theorem 2.17 proves intrinsic recoverability is equivalent to the kernel intersection condition:

$$\ker\partial\phi(\theta) \cap \ker\partial\mathbf{h}(\theta) = \{0\},$$

This transforms an abstract proposition about global dynamics into a point-wise check of whether two Jacobian null spaces intersect only at the origin—making it computable and decidable. Furthermore, Theorem 2.14 provides a necessary kernel space condition for the intrinsic metric property, used to prove "negative results" (e.g., the intrinsic metric property does not hold for two-layer linear networks or attention layers under naive parameterization).

**2. Frobenius Property of ReLU Networks: Bypassing Explicit Construction of Conservation Laws via Lie Algebra Criteria**

While the kernel intersection criterion is useful, point-wise checking still requires finding all conservation laws $\mathbf{h}$, which is often difficult. The authors instead use a quasi-equivalent (Prop 2.21) algebraic sufficient condition—the **Frobenius property**: the family of vector fields induced by $\phi$ is closed under Lie brackets (this is slightly weaker than the involutive condition in Marcotte et al. 2023). This allows verifying intrinsic recoverability without explicitly constructing conservation laws. Theorem 3.1 proves that for any DAG architecture and ReLU network of any depth, the path-lifting parameterization $\phi_{\text{ReLU}}$ satisfies the Frobenius property on a dense set of non-zero parameters. Thus, Corollary 3.3 establishes the strongest intrinsic recoverability for almost all initializations (Proposition 3.5 provides a specific closed-form characterization for three-layer ReLU intrinsic dynamics). Counter-intuitively, the piecewise structure of ReLU results in a smaller symmetry group and thus richer conservation laws; since known conservation laws (differences of diagonal terms) are already complete for it, intrinsic dynamics are easier to establish for ReLU than for linear networks.

**3. Relaxed Balanced Conditions for Linear Networks: Broadening Balanced Initialization to a Family with Necessary and Sufficient Characterization**

Previous intrinsic dynamics results for linear networks relied on strict balanced initialization $U_{i+1}^\top U_{i+1} = U_i U_i^\top$, which is equivalent to setting all regular conservation laws to zero $\mathbf{h}(\theta_0) = 0$ (denoted $S = 0$). This is nearly impossible to satisfy exactly in practice. The authors relax this to **relaxed balance** (Def 3.6) $S = \lambda I$, allowing adjacent layers to differ by a scalar multiple, thereby expanding the "single point" condition to a whole family of initializations. Under this condition, they prove that relaxed balanced initialization satisfies the intrinsic metric property (Theorem 3.8 for two layers, Theorem 3.11 for arbitrary depth). Conversely, they prove that for configurations like $r \leq \max(n,m)$, relaxed balance is a **necessary** condition for the intrinsic metric property (Theorem 3.9). This provide a necessary and sufficient characterization for linear networks, leading to closed-form expressions for intrinsic dynamics, extended to the limit of infinite-depth linear Neural ODEs (Theorem 3.13).

### Loss & Training

This is a purely theoretical work analyzing continuous-time gradient flow. The function $f$ in the loss $\ell(\theta) = f(\phi(\theta))$ can be any differentiable function; all conclusions are independent of the specific loss form and dataset.

## Key Experimental Results

### Main Results

The core contributions of this paper are presented as theorems:

| Network Type | Mapping $\phi$ | Intrinsic Property | Condition |
| :--- | :--- | :--- | :--- |
| Arbitrary DAG ReLU | $\phi_{\text{ReLU}}$ (path-lifting) | Intrinsic Recoverability ✓ | Non-zero parameters (dense set) |
| 2-layer Linear | $\phi_{\text{Lin}} = UV^\top$ | Intrinsic Metric ✓/✗ | Relaxed balance ✓ / Unrelaxed ✗ |
| Deep Linear | $\phi_{\text{Lin}} = U_L \cdots U_1$ | Intrinsic Dynamics ✓ | Relaxed balance condition |
| Linear Neural ODE | Infinite-depth limit | Intrinsic Dynamics ✓ | Relaxed balance + Closed-form metric |

### Ablation Study

| Dimension | Previous Results | Ours (Extension) |
| :--- | :--- | :--- |
| ReLU Depth | Two layers | Arbitrary DAG architecture |
| Linear Initialization | Strict balance $\lambda = 0$ | Relaxed balance $S = \lambda I$ |
| Linear Layers | Two layers | Arbitrary depth + Infinite depth |
| Conservation Laws | Empirical verification | Theoretical proof (Corollary 3.4) |

### Key Findings

- ReLU networks satisfy the strongest intrinsic recoverability property on a dense set of initializations.
- Known conservation laws (differences of diagonal terms) are complete for ReLU networks.
- The relaxed balance condition is necessary and sufficient for the intrinsic metric property in linear networks.
- A closed-form expression for intrinsic dynamics is provided for three-layer ReLU networks for the first time.
- Linear Neural ODEs under relaxed balanced initialization also possess closed-form intrinsic metrics.

## Highlights & Insights

1.  **Unified Framework**: The three-level progressive definitions clearly reveal the strengths and relationships between different intrinsicness concepts.
2.  **Favorable Properties of ReLU**: Counter-intuitively, the non-linear piecewise structure of ReLU leads to smaller symmetry groups and richer conservation laws, making it easier to establish intrinsic dynamics than for linear networks.
3.  **Power of Kernel Inclusion Criterion**: Theorem 2.14 provides a concise tool for proving "negative results."
4.  **Lie Algebra Criterion**: A practical algebraic testing method based on the Frobenius property avoids the direct construction of conservation laws.
5.  **Cross-Architecture Applicability**: Provides unified treatment for ReLU, linear, attention layers, and infinite-depth networks.

## Limitations & Future Work

1.  Analysis is limited to continuous gradient flow and does not cover discrete optimization algorithms (SGD, Adam).
2.  Only establishes intrinsic dynamics (step i) without progressing to mirror flow (step ii).
3.  The case for linear networks when $r > \max(n, m)$ remains an open problem.
4.  Lacks numerical validation experiments.
5.  The Frobenius property does not hold for attention layers, requiring indirect analysis.

## Related Work & Insights

- **Arora et al. (2019)**: Balanced initialization and conservation laws $\rightarrow$ Generalized to relaxed balanced in this paper.
- **Marcotte et al. (2023)**: Involutive condition $\rightarrow$ Weakened to Frobenius condition in this paper.
- **Li et al. (2022)**: Commuting condition (a special case of Frobenius) $\rightarrow$ mirror flow.
- **Gonon et al. (2024)**: Path-lifting framework $\rightarrow$ This paper proves it satisfies the Frobenius property.
- **Insight**: Lays the theoretical foundation for future analysis of warped mirror flow and implicit bias in practical architectures.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Established a complete hierarchical theory; universal results for ReLU are a major breakthrough.
- **Experimental Thoroughness**: ⭐⭐⭐ Purely theoretical work; theorems are rigorous but lack numerical validation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Definitions and theorems progress logically; the framework is elegant and clear.
- **Value**: ⭐⭐⭐⭐ Provides a significant theoretical cornerstone for understanding implicit bias.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Neural Collapse under Gradient Flow on Shallow ReLU Networks for Orthogonally Separable Data](../../NeurIPS2025/llm_pretraining/neural_collapse_under_gradient_flow_on_shallow_relu_networks_for_orthogonally_se.md)
- [\[NeurIPS 2025\] Generalization Bounds for Rank-sparse Neural Networks](../../NeurIPS2025/llm_pretraining/generalization_bounds_for_rank-sparse_neural_networks.md)
- [\[ICLR 2026\] Time is a Feature: Exploiting Temporal Dynamics in Diffusion Language Models](time_is_a_feature_exploiting_temporal_dynamics_in_diffusion_language_models.md)
- [\[ICML 2025\] Bayesian Neural Scaling Law Extrapolation with Prior-Data Fitted Networks](../../ICML2025/llm_pretraining/bayesian_neural_scaling_law_extrapolation_with_prior-data_fitted_networks.md)
- [\[NeurIPS 2025\] Alternating Gradient Flows: A Theory of Feature Learning in Two-layer Neural Networks](../../NeurIPS2025/llm_pretraining/alternating_gradient_flows_a_theory_of_feature_learning_in_two-layer_neural_netw.md)

</div>

<!-- RELATED:END -->

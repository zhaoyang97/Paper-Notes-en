---
title: >-
  [Paper Note] Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry
description: >-
  [NeurIPS 2025][Optimization][Sparse Autoencoders] This paper reveals a fundamental duality between sparse autoencoder (SAE) architectures and the concept structures they are capable of discovering — each SAE implicitly a…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "Sparse Autoencoders"
  - "Concept Geometry"
  - "Projection Nonlinearity"
  - "Interpretability"
  - "Duality"
date: 2026-05-08
content_hash: dc15d008b43a3133
---

# Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry

**Conference**: NeurIPS 2025
**arXiv**: [2503.01822](https://arxiv.org/abs/2503.01822)  
**Code**: [GitHub](https://github.com/Sai-Sumedh/SaeConceptDuality-SpaDE)  
**Area**: Optimization
**Keywords**: Sparse Autoencoders, Concept Geometry, Projection Nonlinearity, Interpretability, Duality

## TL;DR

This paper reveals a fundamental duality between sparse autoencoder (SAE) architectures and the concept structures they are capable of discovering — each SAE implicitly assumes a particular organization of concepts, and when this assumption is mismatched, concepts are systematically missed. Based on this analysis, the authors propose SpaDE, a novel SAE that accounts for nonlinear separability and dimensional heterogeneity.

## Background & Motivation

Sparse autoencoders (SAEs) have become a central tool in neural network interpretability research, decomposing model representations into overcomplete sets of monosemantic latent variables to enable enumeration and intervention on the concepts computed by the model. Prior work has demonstrated the ability of SAE latent variables to correspond to meaningful concepts such as specific buildings, behaviors, and scripts.

However, a fundamental question has been overlooked: **Do SAEs truly discover all concepts on which a model relies, or are they inherently biased toward certain types of concepts?** Different SAE architectures (ReLU, TopK, JumpReLU) typically achieve similar fidelity/sparsity trade-offs, but do they discover the **same** concepts?

If the answer is no, this may explain the negative results recently observed in SAE research: algorithmic instability and lack of causality. More importantly, it implies that **no universal SAE exists** — the choice of architecture itself embeds assumptions about data structure.

The authors' starting point is to formalize SAEs as a bilevel optimization problem (Claim 3.1), in which the encoder nonlinearity can be uniformly expressed as an orthogonal projection onto some constraint set. This reveals that the geometry of the **receptive field** of each SAE encoder — the region of input space that activates a given latent variable — directly determines what types of concepts it can discover.

## Method

### Overall Architecture

The three dominant SAE variants (ReLU, TopK, JumpReLU) are unified under a "projection nonlinearity" framework. The implicit data assumptions and receptive field geometry of each variant are analyzed, and SpaDE is designed to overcome the identified limitations.

### Key Designs

1. **Unified Projection Nonlinearity Framework (Definition 3.1 & Table 1)**:

    - **ReLU**: $g(\mathbf{v}) = \Pi_{\mathcal{S}}\{\mathbf{v}\}$, $\mathcal{S} = \{\mathbf{y} \geq 0\}$ (projection onto positive orthant)
    - **TopK**: $g(\mathbf{v}) = \Pi_{\mathcal{S}}\{\mathbf{v}\}$, $\mathcal{S} = \{\mathbf{y} \geq 0, \|\mathbf{y}\|_0 \leq k\}$ (projection onto $k$-sparse constraint)
    - **JumpReLU**: ReLU($\mathbf{v} - \theta$) + $\theta \odot H(\mathbf{v} - \theta)$, combining thresholding with positive orthant projection

   Key insight: the essential difference among SAE variants lies in the choice of projection set $\mathcal{S}$.

2. **Bilevel Optimization Formalization (Claim 3.1)**: SAEs solve the following bilevel problem:
   $$\arg\min_{\mathbf{D}, \mathbf{z} \geq 0} \sum_\mathbf{x} \|\mathbf{x} - \mathbf{D}\mathbf{z}\|^2 + \lambda\mathcal{R}(\mathbf{z})$$
   $$\text{s.t.} \quad \mathbf{z} = \mathbf{f}(\mathbf{x}) \in \arg\min_{\pi \in \mathcal{S}} \mathbf{F}(\pi, \mathbf{W}, \mathbf{x})$$
   The inner optimization (determined by the encoder architecture) constrains the solution space of the outer dictionary learning problem.

3. **Receptive Fields and Implicit Assumptions (Definition 3.2 & Table 2)**:

    - **ReLU/JumpReLU**: Receptive field is a half-space → assumes concepts are **linearly separable**
    - **TopK**: Receptive field is a union of hypercones → assumes concepts are **angularly separable** and **dimensionally uniform** (since $k$ is fixed for all inputs)

4. **Two Critical Data Properties**:

    - **Nonlinear Separability**: Concepts of different sizes or magnitudes may not be separable by a hyperplane (e.g., "onion features," linear features of varying magnitude)
    - **Dimensional Heterogeneity**: Different concepts occupy subspaces of different dimensionality (e.g., truth is 1-dimensional, days of the week are 2-dimensional, safety features are high-dimensional)

   Table 3 analyzes the compatibility of each SAE: ReLU/JumpReLU support heterogeneity but not nonlinear separability; TopK supports limited nonlinear separability but not heterogeneity.

5. **SpaDE (Sparsemax Distance Encoder)**: Designed from the duality principle:

    - The projection set is chosen as the probability simplex $\mathcal{S} = \Delta^s = \{\mathbf{x}: \sum_i x_i = 1, \mathbf{x} \geq 0\}$, yielding a Sparsemax nonlinearity that is adaptively sparse (different inputs can activate different numbers of latent variables)
    - The encoder uses Euclidean distance rather than a linear transformation: $\mathbf{z} = \text{Sparsemax}(-\lambda d(\mathbf{x}, \mathbf{W}))$, where $d(\mathbf{x}, \mathbf{W})_i = \|\mathbf{x} - \mathbf{W}_i\|^2$
    - $\mathbf{W}_i$ serves as prototypes; distance-based encoding naturally supports nonlinear separability
    - The outer optimization corresponds to K-Deep Simplex (KDS), with regularization term $\sum_i z_i\|\mathbf{x} - \mathbf{W}_i\|^2$ encouraging prototypes to lie near the data

### Loss & Training

SpaDE is trained with the same reconstruction loss as other SAEs. The hyperparameter $\lambda$ controls the degree of sparsity (analogous to an inverse temperature); as $\lambda \to 0$, the output degenerates to a uniform distribution. Monosemanticity of latent variables is evaluated using the F1 score.

## Key Experimental Results

### Nonlinear Separability Experiment (Fig. 5)

| SAE | Linear Separable Concept F1 | Nonlinearly Separable Concept F1 | Notes |
|---|---|---|---|
| ReLU | 1.0 | ~0.5 | Half-space receptive field cannot isolate nonlinear concepts |
| JumpReLU | 1.0 | ~0.5 | Same as above |
| TopK | ~0.7 | ~0.7 | Neither type is captured perfectly |
| **SpaDE** | **1.0** | **1.0** | Perfectly captures both types of concepts |

### Dimensional Heterogeneity Experiment (Fig. 6)

| SAE | Adaptive Sparsity | High-Dim Concept MSE | Notes |
|---|---|---|---|
| ReLU | ✓ Partial | Low | Adaptive but with cross-concept co-activation |
| JumpReLU | ✓ Partial | Low | Same as above |
| TopK | ✗ | High | Fixed $k$ leads to poor reconstruction of high-dimensional concepts |
| **SpaDE** | **✓ Fully** | **Low** | Precisely matches intrinsic dimensionality |

TopK only reduces normalized MSE below 20% when $k$ exceeds the intrinsic concept dimensionality (e.g., $d=6$ requires $k \geq 8$).

### Formal Language GPT Experiment (Fig. 7)

| SAE | Cross-POS Latent Co-activation | Best F1 Score | Notes |
|---|---|---|---|
| ReLU | High | < 1.0 | Different word classes activate shared latent variables |
| JumpReLU | High | < 1.0 | Same as above |
| TopK | Medium | < 1.0 | Different $k$ values needed for different word classes |
| **SpaDE** | **None** | **1.0** | Perfectly separates all word classes |

### DINOv2 Vision Experiment (Fig. 8)

| SAE | Cross-Class Latent Co-activation | Top-5 F1 Range | Notes |
|---|---|---|---|
| ReLU | Widespread | Large variance | Poor separability across classes |
| JumpReLU | Widespread | Large variance | Same as above |
| TopK | Widespread | Lower | Angular separation insufficient |
| **SpaDE** | **Limited** | **Highest** | Most monosemantic latent variables |

SpaDE identifies interpretable concepts on DINOv2 such as foreground/background, object parts (hands, faces, fins, church windows, dog eyes/ears/noses).

### Key Findings

- Switching SAE architectures may expose entirely new concepts or obscure existing ones — different SAEs are not interchangeable.
- The half-space receptive fields of ReLU/JumpReLU impose an F1 ceiling of approximately 0.5 for nonlinearly separable concepts.
- The fixed sparsity of TopK prevents adaptation to concepts of different intrinsic dimensionality.
- SpaDE's adaptive sparsity combined with distance-based encoding simultaneously addresses both issues.
- Spectral clustering analysis shows that SpaDE's sparse codes better preserve concept boundaries.

## Highlights & Insights

- The "duality" perspective is elegant: rather than searching for a universal SAE, the key is to understand how concepts are organized in data and then select or design architectures that match this organization.
- Unifying SAEs as bilevel optimization with projection nonlinearities is theoretically appealing, reducing the problem of architecture selection to a problem of projection set selection.
- SpaDE exemplifies a "data-property-driven architecture design" paradigm: nonlinear separability → distance-based encoding; dimensional heterogeneity → adaptive sparsity (Sparsemax).
- The experimental design progresses systematically from controlled to naturalistic settings (synthetic → formal language → vision models), yielding a compelling and coherent argument.

## Limitations & Future Work

- SpaDE is not universally optimal — it implicitly assumes that concepts are separated by Euclidean distance, which may not always hold.
- Risk of over-specialization: SpaDE may split a single concept into multiple sub-clusters (two colors appear for Concept 1 in Fig. 5c).
- The analysis considers only mutually exclusive concepts; the setting with overlapping concepts may require different treatment.
- The scalability of SpaDE has not yet been validated at LLM scale.
- Beyond nonlinear separability and dimensional heterogeneity, other critical data properties may remain to be identified and addressed.

## Related Work & Insights

- This work directly responds to the core assumption of Bricken et al. (2023) "Towards Monosemanticity" — whether SAEs can truly discover all concepts.
- The findings are consistent with Fel et al. (2025)'s "Archetypal SAE" regarding SAE instability — instability may stem from a mismatch between architectural assumptions and data structure.
- Inspired by the concept of receptive fields in neuroscience, this paper offers a new perspective for analyzing the selectivity of SAE latent variables.
- An important implication for interpretability research: when using SAEs for model explanation, one should be aware that the discovered concepts are constrained by the implicit assumptions of the chosen SAE architecture.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The SAE architecture–concept geometry duality is a novel and profound insight.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validation progresses systematically from synthetic to formal language to vision, providing a comprehensive evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ The framework is clearly articulated, figures (Fig. 1–2) are intuitive, and mathematical formalism is well integrated with intuition.
- Value: ⭐⭐⭐⭐⭐ The work has a fundamental impact on the direction of SAE interpretability research — shifting focus from blindly pursuing "better SAEs" to understanding the implications of architectural choices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Understanding Adam Requires Better Rotation Dependent Assumptions](understanding_adam_requires_better_rotation_dependent_assumptions.md)
- [\[NeurIPS 2025\] Learning Sparse Approximate Inverse Preconditioners for Conjugate Gradient Solvers on GPUs](learning_sparse_approximate_inverse_preconditioners_for_conjugate_gradient_solve.md)
- [\[ICLR 2026\] Generalization Below the Edge of Stability: The Role of Data Geometry](../../ICLR2026/optimization/generalization_below_the_edge_of_stability_the_role_of_data_geometry.md)
- [\[AAAI 2026\] Tackling Resource-Constrained and Data-Heterogeneity in Federated Learning with Double-Weight Sparse Pack](../../AAAI2026/optimization/tackling_resource-constrained_and_data-heterogeneity_in_federated_learning_with_.md)
- [\[NeurIPS 2025\] Probing Neural Combinatorial Optimization Models](probing_neural_combinatorial_optimization_models.md)

</div>

<!-- RELATED:END -->

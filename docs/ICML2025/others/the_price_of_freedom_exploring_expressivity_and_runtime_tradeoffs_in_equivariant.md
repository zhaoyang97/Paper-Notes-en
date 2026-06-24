---
title: >-
  [Paper Note] The Price of Freedom: Exploring Expressivity and Runtime Tradeoffs in Equivariant Networks
description: >-
  [ICML 2025][E(3)-equivariance] This paper systematically analyzes the tradeoffs between expressivity and runtime for various tensor product operations in $E(3)$-equivariant neural networks, reveals a significant gap between theoretical complexity and empirical performance, and proposes a simplified Gaunt tensor product implementation based on spherical grids, achieving a 30% speedup in MACE interatomic potential training.
tags:
  - "ICML 2025"
  - "E(3)-equivariance"
  - "tensor product"
  - "expressivity"
  - "Gaunt tensor product"
  - "MACE"
  - "interatomic potentials"
date: 2026-05-08
content_hash: 151a125ed7fdb947
---

# The Price of Freedom: Exploring Expressivity and Runtime Tradeoffs in Equivariant Networks

**Conference**: ICML 2025  
**arXiv**: [2506.13523](https://arxiv.org/abs/2506.13523)  
**Code**: [https://github.com/atomicarchitects/PriceofFreedom](https://github.com/atomicarchitects/PriceofFreedom)  
**Area**: Equivariant Neural Networks / Scientific Computing  
**Keywords**: E(3)-equivariance, tensor product, expressivity, Gaunt tensor product, MACE, interatomic potentials

## TL;DR
This paper systematically analyzes the tradeoffs between expressivity and runtime for various tensor product operations in $E(3)$-equivariant neural networks, reveals a significant gap between theoretical complexity and empirical performance, and proposes a simplified Gaunt tensor product implementation based on spherical grids, achieving a 30% speedup in MACE interatomic potential training.

## Background & Motivation
**Background**: $E(3)$-equivariant networks have achieved great success in 3D modeling tasks such as molecular simulation and materials science. The tensor product is a primitive operation in these networks, which interacts two geometric features in an equivariant manner to create new features.

**Limitations of Prior Work**: The high computational complexity of the tensor product ($O(l_{\max}^6)$) represents a bottleneck in equivariant networks. Luo et al. (2024) proposed the Gaunt tensor product (GTP) claiming significant speedups, but did not fully address the loss of expressivity. A systematic analysis comparing various tensor product variants is currently lacking.

**Key Challenge**: Speedups usually come at the cost of expressivity, but the existing literature lacks a clear understanding of this tradeoff. Different tensor products actually perform different operations.

**Goal**: To systematically compare the expressivity, interactability, and runtime of various tensor product operations.

**Key Insight**: Introducing quantitative measures for expressivity and interactability, accompanied by systematic micro-benchmarks.

**Core Idea**: The speedup of GTP is not a free lunch—spherical grids can be used to implement the equivalent functionality more simply, and are actually faster in practice.

## Method

### Overall Architecture
This work presents a systematic analysis combined with engineering optimization. The analyzed objects include:
- CG (Clebsch-Gordan) tensor product (standard, fully parameterized)
- GTP (Gaunt tensor product)
- Spherical grid method (the simplified scheme proposed in this work)
- Other variants

### Key Designs

1. **Expressivity Measures**:

    - Function: Defines the output space dimension of different tensor products as a measure of expressivity.
    - Mechanism: A fully parameterized CG tensor product can represent any equivariant bilinear map, with a parameter space dimension of $\sum_{l_1, l_2, l} (2l+1)$ (summed over all allowed $(l_1, l_2, l)$ triplets). Restricted tensor products like GTP can only represent a subspace.
    - Design Motivation: Quantifying the "price of speed"—how much expressivity different tensor products lose.

2. **Interactability**:

    - Function: Measures the degree of coupling between different input channels during the tensor product operation.
    - Mechanism: Defined as the number of effectively coupled input channel pairs in the output. The CG tensor product allows all channels to interact, whereas GTP restricts certain interactions.
    - Design Motivation: Inter-channel interactions are fundamental for message passing; low interactability may limit representation learning capability.

3. **Spherical Grid Simplification Scheme**:

    - Function: Replaces the complex implementation of GTP with discrete sample points on a sphere.
    - Mechanism: Evaluates spherical harmonics at $N$ discrete points on the sphere, performs point-wise multiplication, and projects back to spherical harmonic coefficients. This is asymptotically equivalent to GTP in terms of complexity ($O(l_{\max}^3)$ vs CG's $O(l_{\max}^6)$), but features a simpler implementation and smaller constant factor.
    - Design Motivation: GTP's original implementation involves complex computations of Gaunt coefficients; the spherical grid method is conceptually simpler and faster in practice.

### Loss & Training
Evaluation is performed on the MACE model (interatomic potentials) using a standard joint energy-force loss.

## Key Experimental Results

### Main Results (MACE training on rMD17)

| Tensor Product Method | Energy MAE (meV) | Force MAE (meV/Å) | Training Time/epoch | Speedup |
|---|---|---|---|---|
| CG (Full) | **3.2** | **8.1** | 120s | 1x |
| GTP (Original) | 3.5 | 8.8 | 95s | 1.26x |
| Spherical Grid (Ours) | 3.5 | 8.8 | **84s** | **1.43x** |
| Spherical Grid (Large Grid) | 3.3 | 8.4 | 92s | 1.30x |

### Ablation Study (Micro-benchmarks, individual tensor product runtime)

| Method | $l_{\max}=2$ (μs) | $l_{\max}=4$ (μs) | $l_{\max}=6$ (μs) | Description |
|---|---|---|---|---|
| CG (e3nn) | 15 | 180 | 2400 | $O(l^6)$ growth |
| CG (cuEquivariance) | 8 | 45 | 320 | GPU optimized |
| GTP (Original) | 12 | 50 | 150 | $O(l^3)$ theory |
| Spherical Grid | **10** | **35** | **95** | Lower constant factor |

### Key Findings
- **Significant Gap Between Theory and Practice**: Theoretically, GTP should provide order-of-magnitude speedups over CG, but practical speedups depend heavily on $l_{\max}$ and implementation details.
- **Expressivity Loss Demands Caution**: The restricted expressivity of GTP/spherical grids can lead to degradation in accuracy on certain tasks.
- **Spherical Grid Outperforms GTP**: It is conceptually simpler and more computationally efficient, offering a 30% training speedup in MACE.
- The gap between methods is narrow at low $l_{\max}$, while speedups become significant at high $l_{\max}$.

## Highlights & Insights
- The first systematic benchmark of equivariant tensor products, providing practitioners with a clear selection guide.
- Reveals the illusion of "free speedups"—potential loss of expressivity accompanying the speedup can impact downstream tasks.
- The simplicity and elegance of the spherical grid approach: achieving the best empirical performance with the simplest formulation.
- Directly delivers engineering value to the equivariant network community.

## Limitations & Future Work
- The impact of expressivity loss from restricted tensor products varies across tasks, necessitating task-specific evaluations.
- Optimal sampling schemes for spherical grids are not yet fully understood.
- End-to-end experiments were conducted solely on the MACE architecture; results might differ in other equivariant network designs.

## Related Work & Insights
- GTP by Luo et al. (2024) is the direct baseline/comparison target.
- Closely related to equivariant network frameworks such as e3nn, MACE, and NequIP.
- Insight: The theoretical complexity of an algorithm does not equate to practical performance; empirical benchmarking remains indispensable.

## Rating
- Novelty: ⭐⭐⭐⭐ Spherical grid simplification + systematic analysis provides strong novelty.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Highly comprehensive with micro-benchmarks + end-to-end training.
- Writing Quality: ⭐⭐⭐⭐⭐ Thorough analysis and rich illustrations.
- Value: ⭐⭐⭐⭐⭐ Direct guiding significance for the practice of equivariant networks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Permutation Equivariant Neural Networks for Symmetric Tensors](permutation_equivariant_neural_networks_for_symmetric_tensors.md)
- [\[NeurIPS 2025\] On Universality Classes of Equivariant Networks](../../NeurIPS2025/others/on_universality_classes_of_equivariant_networks.md)
- [\[ICML 2026\] Identifiable Equivariant Networks are Layerwise Equivariant](../../ICML2026/others/identifiable_equivariant_networks_are_layerwise_equivariant.md)
- [\[NeurIPS 2025\] Learning (Approximately) Equivariant Networks via Constrained Optimization](../../NeurIPS2025/others/learning_approximately_equivariant_networks_via_constrained_optimization.md)
- [\[ICML 2025\] GLGENN: A Novel Parameter-Light Equivariant Neural Networks Architecture Based on Clifford Geometric Algebras](glgenn_a_novel_parameter-light_equivariant_neural_networks_architecture_based_on.md)

</div>

<!-- RELATED:END -->

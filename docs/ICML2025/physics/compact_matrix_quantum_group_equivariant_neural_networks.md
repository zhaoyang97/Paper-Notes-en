---
title: >-
  [Paper Note] Compact Matrix Quantum Group Equivariant Neural Networks
description: >-
  [ICML 2025 (Poster)][Physics & Scientific Computing][Compact Matrix Quantum Groups] This paper extends group equivariant neural networks to the setting of **compact matrix quantum groups**, characterizing the weight matrices of such networks using Woronowicz's formulation of Tannaka-Krein duality, thereby providing a theoretical foundation for learning data on noncommutative geometries.
tags:
  - "ICML 2025 (Poster)"
  - "Physics & Scientific Computing"
  - "Compact Matrix Quantum Groups"
  - "Equivariant Neural Networks"
  - "Tannaka-Krein Duality"
  - "Set Partitions"
  - "Noncommutative Geometry"
  - "$C^*$-Algebras"
date: 2026-05-08
content_hash: ebe899297aa5ff09
---

# Compact Matrix Quantum Group Equivariant Neural Networks

**Conference**: ICML 2025 (Poster)

**arXiv**: [2311.06358](https://arxiv.org/abs/2311.06358)

**Author**: Edward Pearce-Crump

**Area**: Physics / Quantum Groups / Equivariant Neural Networks

**Keywords**: Compact Matrix Quantum Groups, Equivariant Neural Networks, Tannaka-Krein Duality, Set Partitions, Noncommutative Geometry, $C^*$-Algebras

## TL;DR

This paper extends group equivariant neural networks to the setting of **compact matrix quantum groups**, characterizing the weight matrices of such networks using Woronowicz's formulation of Tannaka-Krein duality, thereby providing a theoretical foundation for learning data on noncommutative geometries.

## Background & Motivation

### Background

**Background**: Group equivariant neural networks are highly effective at processing data with explicit group symmetries and have been widely applied in fields such as image recognition and molecular modeling. However, these networks rely on symmetry descriptions on classical geometric spaces, i.e., using **commutative** $C^*$-algebras (algebras of continuous functions on compact groups).

When data resides in **noncommutative geometry** (formally described by noncommutative $C^*$-algebras), traditional group equivariant networks are no longer applicable. For example, symmetries involved in quantum information processing, quantum error correction, and other fields are described by **quantum groups** rather than classical groups.

The core motivation of this paper is:

### Limitations of Prior Work

**Limitations of Prior Work**: Filling the theoretical gap**: extending the theoretical framework of equivariant neural networks from classical groups to quantum groups.

### Key Challenge

**Key Challenge**: Characterizing weight matrices**: providing a complete characterization of weight matrices for "easy" compact matrix quantum groups.

### Proposed Approach

**Proposed Approach**: Discovering new results**: obtaining novel characterizations of weight matrices that have not previously appeared in the machine learning literature, even when degenerating to classical group cases.

## Method

### Overall Architecture

The technical route of the paper is as follows:

1. **Define Compact Matrix Quantum Groups (CMQGs)**: Replace the concept of classical groups with $C^*$-algebras and their comultiplications.
2. **Utilize Woronowicz's Tannaka-Krein Duality**: Convert the representation theory of quantum groups into the characterization of weight spaces.
3. **Focus on "easy" CMQGs**: These quantum groups are defined by **set partitions**, allowing the weight matrices to be completely characterized by combinatorial tools.

### Key Mathematical Tools

The core idea of **Tannaka-Krein Duality** is that a compact matrix quantum group $G$ can be completely recovered from its **representation category** $\text{Rep}(G)$. An equivariant linear map $\phi: V \to W$ needs to satisfy:

$$\phi \circ \rho_V(g) = \rho_W(g) \circ \phi, \quad \forall g \in G$$

In the quantum group setting, the space of equivariant maps (intertwiners) $\text{Hom}_G(V, W)$ consists of linear operators satisfying specific conditions.

The weight matrix of an **easy quantum group** can be represented as:

$$W = \sum_{\pi \in D(k, l)} c_\pi \cdot T_\pi$$

where $D(k,l)$ is the set of set partitions from $k$ upper points to $l$ lower points, $T_\pi$ is the linear map defined by partition $\pi$, and $c_\pi$ are learnable parameters.

### Quantum Group Categories Covered

| Quantum Group Type | Corresponding Partition Category | Is Classical Group? | Prior ML Characterization? |
|:---|:---|:---|:---|
| $O_n$ (Orthogonal Group) | Pair partitions | Yes | Yes |
| $S_n$ (Symmetric Group) | All partitions | Yes | Yes |
| $H_n$ (Hyperoctahedral Group) | Block-symmetric partitions | Yes | Partial |
| $B_n$ (Bistochastic Group) | Block-symmetric partition subset | Yes | No |
| $O_n^+$ (Free Orthogonal Quantum Group) | Non-crossing pair partitions | **No** | No |
| $S_n^+$ (Free Symmetric Quantum Group) | Non-crossing partitions | **No** | No |
| $H_n^+$ (Free Hyperoctahedral Quantum Group) | Non-crossing block-symmetric partitions | **No** | No |
| $B_n^+$ (Free Bistochastic Quantum Group) | Non-crossing block-symmetric subset | **No** | No |

### Loss & Training

This work is purely theoretical and does not involve specific loss function designs. Its core contribution lies in proving the **existence** of equivariant layers and the **characterization theorems** of weight matrices, laying the foundation for future practical applications.

## Key Experimental Results

As a theoretical paper, this work does not include numerical experiments in the traditional sense. Its primary "experiments" consist of the **concrete calculation** of weight matrices for different quantum group types.

### Weight Matrix Dimension Comparison (Different Quantum Groups, $n=2, k=l=2$)

### Main Results

| Quantum Group | Dimension of Intertwiner Space | New Result? |
|:---|:---|:---|
| $O_2$ | 2 | No |
| $S_2$ | 4 | No |
| $H_2$ | 3 | Yes (new characterization) |
| $B_2$ | 2 | Yes (new characterization) |
| $O_2^+$ | 1 | Yes (completely new) |
| $S_2^+$ | 2 | Yes (completely new) |
| $H_2^+$ | 2 | Yes (completely new) |
| $B_2^+$ | 1 | Yes (completely new) |

### Key Findings

1. **Quantum group equivariant networks encompass classical group equivariant networks**: All classical group equivariant neural networks are special cases of quantum group equivariant networks.
2. **Non-crossing partitions impose stronger constraints**: The non-crossing partitions corresponding to quantum groups result in a lower dimension of the intertwiner space and more structured weight matrices.
3. **New characterizations of classical groups**: Completely new weight matrix characterizations are also provided for classical groups such as $H_n$ and $B_n$.

## Highlights & Insights

- **First to extend equivariant neural networks to the quantum group setting**, opening the door to deep learning on noncommutative geometries.
- **Elegant Technique**: Translates representation theory problems into combinatorial ones using Tannaka-Krein duality.
- **Unified Framework**: Unifies the weight characterization of various classical and quantum groups through the language of "easy" quantum groups.
- **Bridging Mathematics and Machine Learning**: Involves deep mathematical tools such as category theory, representation theory, and combinatorics.

## Limitations & Future Work

1. **Purely theoretical work**: No numerical experiments or practical application validations are provided.
2. **Limited to easy quantum groups**: The characterization of weight matrices for non-easy quantum groups remains an open problem.
3. **Practical utility needs validation**: Data on noncommutative geometries is currently rare in machine learning applications, making practical demand unclear.
4. **Computational complexity not discussed**: Compared to classical group equivariant networks, the computational overhead of quantum group equivariant layers is not analyzed.

## Related Work & Insights

- **Cohen & Welling (2016)**: Classical Group Equivariant Convolutional Networks (G-CNNs), which serve as the starting point of this work.
- **Kondor & Trivedi (2018)**: Equivariant networks based on irreducible representations.
- **Maron et al. (2019)**: Higher-order equivariant networks.
- **Banica & Speicher (2009)**: Mathematical theory of easy quantum groups, the core source of tools used in this work.

**Insights**: This work suggests that when data exhibits non-classical symmetries (such as in quantum systems or noncommutative spaces), traditional equivariant network frameworks may not suffice, necessitating more general algebraic tools.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First to generalize equivariant networks to quantum groups.
- **Theoretical Depth**: ⭐⭐⭐⭐⭐ — Involves deep theory such as Tannaka-Krein duality.
- **Practicality**: ⭐⭐ — Lacks experiments and application scenarios.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematically rigorous, but sets a high bar for ML readers.
- **Overall Rating**: 7/10 — Significant theoretical contributions, but practical impact remains to be validated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Accelerating Inference for Multilayer Neural Networks with Quantum Computers](../../ICLR2026/physics/accelerating_inference_for_multilayer_neural_networks_with_quantum_computers.md)
- [\[CVPR 2025\] ATP: Adaptive Threshold Pruning for Efficient Data Encoding in Quantum Neural Networks](../../CVPR2025/physics/atp_adaptive_threshold_pruning_for_efficient_data_encoding_in_quantum_neural_net.md)
- [\[ICML 2025\] Differentiable Stellar Atmospheres with Physics-Informed Neural Networks](differentiable_stellar_atmospheres_with_physics-informed_neural_networks.md)
- [\[ICCV 2025\] ResQ: A Novel Framework to Implement Residual Neural Networks on Analog Rydberg Atom Quantum Computers](../../ICCV2025/physics/resq_a_novel_framework_to_implement_residual_neural_networks_on_analog_rydberg_a.md)
- [\[ICML 2025\] Sum-of-Parts: Self-Attributing Neural Networks with End-to-End Learning of Feature Groups](sum-of-parts_self-attributing_neural_networks_with_end-to-end_learning_of_featur.md)

</div>

<!-- RELATED:END -->

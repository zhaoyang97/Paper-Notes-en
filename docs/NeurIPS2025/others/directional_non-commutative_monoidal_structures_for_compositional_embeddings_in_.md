---
title: >-
  [Paper Note] Directional Non-Commutative Monoidal Structures for Compositional Embeddings in Machine Learning
description: >-
  [NeurIPS 2025][non-commutative algebra] This paper proposes an algebraic framework based on directional non-commutative monoid operators, providing a unified mathematical foundation for multi-dimensional compositional embeddings and unifying SSM recurrence, Transformer self-attention, and RoPE positional encoding as special cases.
tags:
  - "NeurIPS 2025"
  - "non-commutative algebra"
  - "compositional embeddings"
  - "monoidal structure"
  - "interchange law"
  - "positional encoding"
  - "SSM"
  - "Transformer"
date: 2026-05-08
content_hash: 96026ab77276dcfc
---

# Directional Non-Commutative Monoidal Structures for Compositional Embeddings in Machine Learning

**Conference**: NeurIPS 2025
**arXiv**: [2505.15507](https://arxiv.org/abs/2505.15507)  
**Code**: None (purely theoretical work)  
**Area**: Other
**Keywords**: non-commutative algebra, compositional embeddings, monoidal structure, interchange law, positional encoding, SSM, Transformer

## TL;DR
This paper proposes an algebraic framework based on directional non-commutative monoid operators, providing a unified mathematical foundation for multi-dimensional compositional embeddings and unifying SSM recurrence, Transformer self-attention, and RoPE positional encoding as special cases.

## Background & Motivation
- Many structured data types (sequences, images, video) exhibit hierarchical compositional properties along multiple dimensions, yet existing algebraic tools primarily target one-dimensional composition (e.g., free groups, non-commutative semigroups).
- Two-dimensional and higher-dimensional composition lacks a well-established algebraic framework: a 2D array can be composed along rows or columns, and such multi-path composition does not fit traditional one-dimensional algebraic systems.
- Existing ML architectures (Transformer positional encodings, SSM recurrence) implicitly realize some form of composition but lack a rigorous algebraic foundation and a unified perspective.
- Representing tokens as matrices rather than vectors breaks core assumptions of the attention mechanism, creating tension between matrix representations and vector-based learning architectures.

## Core Problem
**How to design an algebraic structure that is compatible with vector-based learning architectures while naturally supporting composition along multiple axes?**

## Method

### One-Dimensional Case
Each element is a tuple $({\bf a}, A)$, where ${\bf a} \in \mathbb{R}^n$ and $A \in GL(n)$. The composition operator is defined as:

$$({\bf a}, A) \circ ({\bf b}, B) := ({\bf a} + A{\bf b}, AB)$$

This operation satisfies associativity but not commutativity, and is essentially the semidirect product of the affine transformation group.

### Multi-Axis Generalization
A $D$-dimensional element is represented as ${\bf x} = ({\bf a}, R_1^{n_1}, R_2^{n_2}, \ldots, R_D^{n_D})$, where $R_i \in GL(n)$ is the transformation matrix for the $i$-th axis and $n_i \in \mathbb{Z}$ denotes the position/extent along that axis. Composition along the $k$-th axis is defined as:

$${\bf x} \circ_k {\bf y} = ({\bf a} + R_k^{n_k} {\bf b},\; R_1^{n_1}, \ldots, R_k^{n_k + m_k}, \ldots, R_D^{n_D})$$

### Four Core Properties
1. **Axis-specific composition operators**: Each axis $i$ has an independent composition operator $\circ_i$.
2. **Associativity along each axis**: $(x \circ_i y) \circ_i z = x \circ_i (y \circ_i z)$.
3. **Global commutativity (interchange law)**: $(x \circ_i y) \circ_j (z \circ_i w) = (x \circ_j z) \circ_i (y \circ_j w)$, which holds if and only if $R_i R_j = R_j R_i$.
4. **Single-axis non-commutativity**: $x \circ_i y \neq y \circ_i x$, preserving directional/ordering information.

### Non-Commutative Self-Attention
The relative transformation from position $q$ to $p$ is defined as $T_{p,q} = R_q R_{q+1} \cdots R_{p-1}$, applied to keys and values:

$$\tilde{K}_{p,q} = T_{p,q} K_q, \quad \tilde{V}_{p,q} = T_{p,q} V_q$$

Attention weights and outputs are computed in the standard manner, but positional information is encoded multiplicatively rather than additively.

### Multi-Dimensional Extension
For $D$-dimensional data points, the relative transformation is $T_{p,q} = \prod_{i=1}^D R_i^{(n_{p,i} - n_{q,i})}$; since the $R_i$ pairwise commute, the order of multiplication is irrelevant.

### Unifying SSM and Transformer
- **SSM recurrence**: $y_k = \sum_{i \le k} C_k (\prod_{j=i}^{k-1} A_j) B_i x_i$
- **Vanilla Transformer**: $y_k = \sum_{i \le k} \alpha_{ik} V_i$
- **Proposed framework**: $y_k = \sum_{i \le k} \alpha_{ik} (\prod_{j=i}^{k-1} R_j) V_i$

SSM is a special case in which attention weights degenerate to implicit (uniform) weights and interactions are constrained to a recurrent structure; the standard Transformer is a special case in which the transformation matrices degenerate to the identity.

### m-Representation: Translation-Invariant Compositional Embeddings
Locally ordered yet globally translation-invariant representations are constructed via sliding windows and rotational transformations. The window embedding is $s_k = \sum_{i=1}^m R^{i-1} a_{k+i-1}$; taking block-wise norms and summing yields the m-representation $v$, which is sensitive only to content changes and invariant to global translation.

### RoPE as a Special Case
When all $R_i = R$ (a fixed block-diagonal rotation matrix), $T_{p,q} = R^{(p-q)}$ depends only on relative position, exactly recovering Rotary Position Embedding. The two-dimensional case with rotation matrices $R_x$ and $R_y$ yields 2D RoPE.

## Key Experimental Results
**This paper is purely theoretical and contains no experiments.** The authors explicitly defer empirical validation to future work.

## Highlights & Insights
- **Strong unification**: A single algebraic framework subsumes SSM, Transformer self-attention, RoPE, and affine transformations as special cases.
- **Rigorous mathematical foundation**: Associativity, commutativity, and invertibility are formally proven.
- **Natural multi-dimensional extension**: The generalization from 1D to 2D/3D is not ad hoc but follows directly from the algebraic structure.
- **Computationally efficient**: When $R_i$ is parameterized as $2 \times 2$ block rotation matrices, matrix multiplication reduces to angle addition, enabling efficient parallel prefix scans.

## Limitations & Future Work
- **Primary weakness: no empirical validation**; it remains unclear whether this algebraic structure yields practical performance gains.
- $R_i \in GL(n)$ incurs large computational and storage overhead in high dimensions, though block-diagonal rotation parameterization mitigates this.
- The interchange law requires $R_i R_j = R_j R_i$, restricting the expressive power of the transformation matrices to commutative matrix families.
- The practical training stability of the framework is unknown.
- Discussion of boundary conditions and degenerate cases (e.g., $R$ approaching a singular matrix) is insufficient.

## Related Work & Insights

| Method | Composition | Multi-dimensional | Algebraic Guarantees |
|--------|------------|-------------------|----------------------|
| RoPE | Rotation matrix multiplication | Extensible but limited | Special case of this framework |
| Matrix space models (Rudolph 2010) | Matrix multiplication | 1D only | Non-commutative semigroup |
| S4/Mamba (SSM) | State recurrence | 1D only (except S4ND) | Special case of this framework |
| Transformer self-attention | Weighted summation | Requires positional encoding | Special case of this framework |
| **Ours** | **Axis-specific non-commutative monoid** | **Naturally multi-dimensional** | **Associativity + interchange law** |

The framework provides theoretical guidance for designing new architectures: one can deliberately choose different $R_i$ parameterizations to obtain specific inductive biases. Positional encoding is elevated from ad hoc design to a level with algebraic guarantees. The interchange law implies that in image/video models, row-wise and column-wise composition should yield consistent results—a testable design principle. The framework may inspire new efficient parallel algorithms, as associativity guarantees the feasibility of prefix scans. The combination of translation invariance and local ordering in the m-representation has potential value for content-matching tasks. The axis-specific concatenation operation $\oplus_k$ provides an algebraically grounded primitive for multimodal fusion and alignment.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (first proposal of a multi-dimensional non-commutative compositional algebra framework of this kind)
- Experimental Thoroughness: ⭐⭐ (no experiments whatsoever)
- Writing Quality: ⭐⭐⭐⭐⭐ (theoretical derivations are clear, though some notation is heavy)
- Value: ⭐⭐⭐⭐ (meaningful theoretical contribution, but the absence of experiments substantially limits practical impact)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Put CASH on Bandits: A Max K-Armed Problem for Automated Machine Learning](put_cash_on_bandits_a_max_k-armed_problem_for_automated_machine_learning.md)
- [\[NeurIPS 2025\] Exploiting Task Relationships in Continual Learning via Transferability-Aware Task Embeddings](exploiting_task_relationships_in_continual_learning_via_transferability-aware_ta.md)
- [\[NeurIPS 2025\] SAD Neural Networks: Divergent Gradient Flows and Asymptotic Optimality via o-minimal Structures](sad_neural_networks_divergent_gradient_flows_and_asymptotic_optimality_via_o-min.md)
- [\[NeurIPS 2025\] Learning non-equilibrium diffusions with Schrödinger bridges: from exactly solvable to simulation-free](learning_non-equilibrium_diffusions_with_schrödinger_bridges_from_exactly_solvab.md)
- [\[NeurIPS 2025\] Equivariance by Contrast: Identifiable Equivariant Embeddings from Unlabeled Finite Group Actions](equivariance_by_contrast_identifiable_equivariant_embeddings_from_unlabeled_fini.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] A Single Architecture for Representing Invariance Under Any Space Group
description: >-
  [ICLR 2026][Space groups] A single architecture (Crystal Fourier Transformer) is proposed that adapts to the invariance requirements of any space group. By analytically deriving the constraints imposed by group operations on Fourier coefficients, the method constructs symmetry-adapted Fourier bases and achieves parameter sharing and zero-shot generalization across all 230 space groups via a dual graph representation of these constraints.
tags:
  - ICLR 2026
  - Space groups
  - symmetry invariance
  - Fourier basis
  - crystal structures
  - zero-shot generalization
date: 2026-05-08
content_hash: 62a2769cfb6c5de1
---

# A Single Architecture for Representing Invariance Under Any Space Group

**Conference**: ICLR 2026
**arXiv**: [2512.13989](https://arxiv.org/abs/2512.13989)
**Code**: None
**Area**: Geometric Deep Learning / Materials Science
**Keywords**: Space groups, symmetry invariance, Fourier basis, crystal structures, zero-shot generalization

## TL;DR
A single architecture (Crystal Fourier Transformer) is proposed that adapts to the invariance requirements of any space group. By analytically deriving the constraints imposed by group operations on Fourier coefficients, the method constructs symmetry-adapted Fourier bases and achieves parameter sharing and zero-shot generalization across all 230 space groups via a dual graph representation of these constraints.

## Background & Motivation

**Background**: Encoding known symmetries into ML models improves accuracy and generalization, yet achieving exact invariance to a specific symmetry group typically requires designing a custom architecture for each group.

**Limitations of Prior Work**: There are 230 space groups in three-dimensional space (crystal symmetries), and designing dedicated architectures for each is not scalable. The problem is further compounded by data scarcity—the largest benchmark dataset (Materials Project) contains only ~200k entries, averaging fewer than 1,000 samples per group, with many groups having almost no data.

**Key Challenge**: Exact group invariance is needed to respect physical constraints, yet cross-group parameter sharing is required to overcome data scarcity.

**Goal**: Develop a single architecture that automatically adapts its weights according to the input space group to enforce the corresponding invariance.

**Key Insight**: Analyze the constraints imposed by group symmetries in Fourier space—group operations introduce phase constraints on reciprocal lattice points, which are encoded as the adjacency matrix of a constraint graph and embedded into neural network layers.

**Core Idea**: Invariance under a space group is equivalent to a set of constraints on Fourier coefficients at reciprocal lattice points. These constraints admit a graph-structured representation that can be encoded into the network, enabling parameter sharing across groups.

## Method

### Overall Architecture
The input is a crystal structure (atomic positions + space group label) and the output is a material property prediction. The core of the architecture is a symmetry-adapted Fourier encoding layer: atomic positions are first encoded in a standard Fourier basis, then multiplied by a precomputed group-dependent adjacency matrix to enforce the constraints, before being passed to Transformer layers.

### Key Designs

1. **Derivation of Group Constraints in Fourier Space (Proposition 3.1 + Theorem 3.2)**:

   - **Function**: Analytically derives the exact constraints imposed by group operations on Fourier coefficients.
   - **Mechanism**: For a $G$-invariant function $f$ and a group operation $\phi(x) = Ax + t$, the Fourier coefficients satisfy $F(\omega) = e^{i2\pi\omega^\top A^\top t} F(A\omega)$. This partitions the reciprocal lattice points into disjoint orbits $\mathcal{O}$, each corresponding to a basis function $e_\mathcal{O}(x) = \sum_{\omega \in \mathcal{O}} w_{\xi \to \omega} e^{i2\pi\omega^\top x}$.
   - **Design Motivation**: The infinite-dimensional problem over continuous function spaces is reduced to a discrete problem over reciprocal lattice points, making the constraints both computable and exact.

2. **Dual Graph Representation and Algorithmic Construction (Algorithm 1)**:

   - **Function**: Represents group constraints as a directed weighted graph over reciprocal lattice points, used to automatically construct symmetry-adapted bases.
   - **Mechanism**: Nodes correspond to frequencies $\omega$; a group operation $\phi$ introduces a directed edge $\omega \to A\omega$ with weight given by the phase factor. Connected components after removing inconsistent self-loops yield phase-consistent orbits, and the products of edge weights give the basis function coefficients.
   - **Design Motivation**: The graph representation translates abstract group-theoretic constraints into a concrete graph-algorithmic problem that can be executed uniformly for any space group.

3. **Crystal Fourier Transformer (CFT)**:

   - **Function**: Uses symmetry-adapted Fourier bases as positional encodings to enable parameter sharing across space groups.
   - **Mechanism**: Atomic positions are first expanded in a standard Fourier basis and then projected onto the invariant subspace via a group-dependent adjacency matrix. The adjacency matrix is precomputed; network weights are shared across all space groups.
   - **Design Motivation**: Different space groups only alter the adjacency matrix (precomputed offline), while the network architecture and parameters remain unchanged—enabling zero-shot generalization.

### Loss & Training
Standard regression/classification losses for material property prediction. Crucially, data from different space groups can be mixed during training, with the model automatically adapting via the adjacency matrix.

## Key Experimental Results

### Main Results

| Task | CFT | Standard PE | CGCNN | Notes |
|------|-----|-------------|-------|-------|
| Formation energy prediction | Competitive | Inferior | Baseline | CFT benefits from exact symmetry |
| Band gap prediction | Competitive | Inferior | Baseline | Same as above |
| Zero-shot generalization | ✓ Succeeds | ✗ Fails | ✗ N/A | Predicts on unseen space groups |

### Ablation Study

| Configuration | Performance | Notes |
|---------------|-------------|-------|
| CFT (full) | Best | Symmetry-adapted Fourier encoding |
| Standard Fourier PE | Inferior | No group constraints |
| No PE | Worst | No positional information |

### Key Findings
- Symmetry-adapted Fourier positional encoding yields significant improvements over standard positional encoding on material property prediction.
- Zero-shot generalization is the central highlight: the model produces reasonable predictions for space groups never encountered during training.
- The positional encoding accurately reflects orbit distances—points within the same orbit are close, while points from different orbits are far apart.
- Precomputation of adjacency matrices is computationally cheap, can be performed offline, and needs to be done only once per group.

## Highlights & Insights
- **From invariant bases to adaptive architectures**: The core insight is that group invariance can be expressed as linear constraints on reciprocal lattice points, which naturally take the form of graph adjacency matrices and can be directly embedded into the matrix multiplication layers of neural networks.
- **A single architecture unifying all 230 space groups**: Rather than designing group-specific networks, the approach uses one shared network combined with group-specific precomputed projection matrices, enabling cross-group knowledge transfer and zero-shot generalization.
- **Theoretical completeness**: It is proved that the constructed basis functions span all continuous $G$-invariant functions in $L^2(\Pi)$—an exact representation, not merely an approximation.

## Limitations & Future Work
- In practice, the basis is approximated by a finite frequency truncation; the choice of cutoff affects expressive power.
- Validation is currently limited to scalar property prediction; extending to equivariant properties (e.g., tensor properties) requires generalizing from invariance to equivariance.
- Performance comparisons against graph neural network baselines (CGCNN, ALIGNN) show competitive rather than substantially superior results.
- Algorithm 1's enumeration of reciprocal lattice points may generate a large number of basis functions under high-frequency cutoffs.

## Related Work & Insights
- **vs. Adams & Orbanz 2023**: That work establishes the existence of $G$-invariant Fourier bases but solves for Laplace eigenfunctions numerically; the present paper provides an analytic construction algorithm.
- **vs. CGCNN/ALIGNN**: These graph networks encode periodicity through augmented graph structures; the present paper encodes space group invariance directly in the positional encoding.
- **vs. Thomas et al. (SE(3)-Transformers)**: SE(3)-equivariant networks handle continuous rotation groups; the present paper addresses discrete crystallographic groups, posing distinct technical challenges.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First single architecture adaptable to any space group; the dual graph representation of Fourier constraints is novel and elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Zero-shot generalization experiments are highly convincing, though quantitative comparisons against state-of-the-art methods do not show a clear lead.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Mathematical derivations build progressively; the exposition from 1D intuition to high-dimensional theory is excellent.
- **Value**: ⭐⭐⭐⭐⭐ — Provides a principled symmetry-handling framework for materials science ML; the zero-shot generalization capability carries significant practical importance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Learning Compact Latent Space for Representing Neural Signed Distance Functions with High-fidelity Geometry Details](../../AAAI2026/others/learning_compact_latent_space_for_representing_neural_signed_distance_functions_.md)
- [\[ICLR 2026\] Agnostics: Learning to Synthesize Code in Any Programming Language with a Universal RL Environment](agnostics_learning_to_code_in_any_programming_language_via_reinforcement_with_a_.md)
- [\[ICLR 2026\] Out of the Shadows: Exploring a Latent Space for Neural Network Verification](out_of_the_shadows_exploring_a_latent_space_for_neural_network_verification.md)
- [\[NeurIPS 2025\] Faithful Group Shapley Value](../../NeurIPS2025/others/faithful_group_shapley_value.md)
- [\[ICCV 2025\] Loss Functions for Predictor-based Neural Architecture Search](../../ICCV2025/others/loss_functions_for_predictor-based_neural_architecture_search.md)

</div>

<!-- RELATED:END -->

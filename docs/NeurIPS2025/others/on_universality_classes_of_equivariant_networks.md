---
title: >-
  [Paper Note] On Universality Classes of Equivariant Networks
description: >-
  [NeurIPS 2025 Spotlight][equivariant neural networks] This paper proves that the separation power of equivariant neural networks (i.e., their ability to distinguish symmetry-inequivalent inputs) is insufficient to fully characterize their approximation capacity—models with identical separation power may possess strictly different approximation abilities. The paper provides a complete characterization of universality classes for shallow invariant networks and establishes suffi…
tags:
  - "NeurIPS 2025 Spotlight"
  - "equivariant neural networks"
  - "universal approximation"
  - "separation power"
  - "symmetry groups"
  - "universality classes"
date: 2026-05-08
content_hash: 241bd36acd181c1a
---

# On Universality Classes of Equivariant Networks

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2506.02293](https://arxiv.org/abs/2506.02293)  
**Code**: Not available  
**Area**: Deep Learning Theory / Equivariant Neural Networks
**Keywords**: equivariant neural networks, universal approximation, separation power, symmetry groups, universality classes

## TL;DR

This paper proves that the separation power of equivariant neural networks (i.e., their ability to distinguish symmetry-inequivalent inputs) is insufficient to fully characterize their approximation capacity—models with identical separation power may possess strictly different approximation abilities. The paper provides a complete characterization of universality classes for shallow invariant networks and establishes sufficient conditions for universality failure.

## Background & Motivation

Equivariant neural networks achieve remarkable empirical success by encoding symmetry into their architectures (e.g., permutation-equivariant GNNs, rotation-equivariant convolutional networks). Existing research on their expressive power has largely focused on **separation power**—the ability to distinguish inputs that are inequivalent under a group action. In graph learning, this is typically formalized via the Weisfeiler-Leman (WL) test hierarchy.

However, separation power addresses only whether distinct inputs can be distinguished, not whether all symmetry-respecting functions can be approximated (i.e., **universality**). Intuitively, separating all inequivalent inputs is a necessary but not sufficient condition for approximating arbitrary functions.

The authors pose a key question: **Is separation power a complete proxy for comparing the expressive power of equivariant networks?** The answer is negative. They construct explicit counterexamples: PointNet and width-1 CNNs share **identical separation power** (both can separate all permutation-inequivalent inputs), yet their approximation capacities are strictly ordered—CNN(filter size 1) $\subsetneq$ PointNet $\subsetneq$ regular representation networks. This demonstrates that separation power is an incomplete proxy for approximation capacity.

## Method

### Overall Architecture

The core theoretical framework represents shallow invariant networks as functions satisfying specific differential constraints, then characterizes universality classes by analyzing the structure of these constraints. Since equivariant networks reduce to invariant networks via projection onto the trivial representation (Remark 11), analyzing the invariant case suffices to obtain corresponding conclusions in the equivariant setting.

### Key Designs

1. **Differential Operator Characterization of Universality Classes (Theorem 13)**: The central theorem establishes an equivalence between universality classes of shallow invariant networks and differential constraints. For layer spaces $M \subseteq \text{Aff}_G(V, W)$ and $N \subseteq \text{Aff}_G(W, \mathbb{R})$, define basis maps $\phi_i: \mathbb{R}^X \to \mathbb{R}^m$ extracted from the equivariant layer structure. A continuous function $f \in \mathcal{U}_\sigma(M, N)$ if and only if $P(\partial_1, \ldots, \partial_d) f = 0$ for all polynomials $P$ vanishing on the row spaces of the basis maps. Intuitively, the equivariant structure of network layers restricts the directions of information flow, inducing unavoidable differential constraints on the function space.

2. **Sufficient Conditions for Universality Failure (Theorems 14 & 15)**: If there exist directions $c_\alpha \in \ker(\phi_\alpha^\top)$ such that

$$D_{c_1} \cdots D_{c_\ell} f \neq 0$$

   then $f \notin \mathcal{U}_\sigma(M, N)$, i.e., the network cannot approximate $f$. The implication is that if $f$ has nonzero mixed partial derivatives along directions "invisible" to the network, approximation fails. Theorem 15 further addresses the special case $n=3$.

3. **Explicit Construction of Equal Separation but Unequal Approximation (Proposition 16)**: The above theory yields three strict inclusions:

$$\mathcal{U}_\sigma(C^1, I) \subsetneq \mathcal{U}_\sigma(\mathbb{R}^n, \mathbb{R}^n, \mathbb{R}) \subsetneq \mathcal{U}_\sigma(\mathbb{R}^n, \mathbb{R}^{S_n}, \mathbb{R})$$

   Proof strategy:
    - **CNN(width 1) cannot approximate** $(x_1 + \cdots + x_n)^n$: since $\partial_n \cdots \partial_1 (x_1 + \cdots + x_n)^n = n! \neq 0$
    - **PointNet can approximate** $(x_1 + \cdots + x_n)^n$: via a summation form $f(x_i, x_1+\cdots+x_n)$
    - **PointNet cannot approximate** $x_1 \cdots x_n$: by constructing direction vectors satisfying the conditions of Theorem 14
    - **Regular representation networks can approximate** $x_1 \cdots x_n$: guaranteed by Theorem 6 (Ravanbakhsh)

4. **Universality under Normal Subgroup Conditions (Theorem 18)**: If the hidden representation arises from the coset space $\mathbb{R}^{G/H}$ of a **normal subgroup** $H$ of $G$, then the corresponding shallow network is separation-constrained universal. For the symmetric group $S_n$ ($n \geq 5$), however, the only nontrivial normal subgroup is the alternating group $A_n$, whose coset space has only 2 elements—too small to be useful. Since all subgroups of abelian groups are normal, cyclic CNNs possess separation-constrained universality.

### Loss & Training

This is a purely theoretical work with no training involved. Results are established via mathematical proof, drawing on variants of the Stone-Weierstrass theorem, differential operator theory, and group representation theory.

## Key Experimental Results

### Summary of Theoretical Results

| Architecture | Symmetry Group | Separation Power | Approximates $(x_1+\cdots+x_n)^n$ | Approximates $x_1 \cdots x_n$ | Sep.-Constrained Universal |
|---|---|---|---|---|---|
| CNN (filter=1) | $\mathbb{Z}_n$ / $S_n$ | Maximal | ✗ | ✗ | ✗ |
| PointNet (shallow) | $S_n$ | Maximal | ✓ | ✗ | ✗ |
| PointNet (depth 3) | $S_n$ | Maximal | ✓ | ✓ | ✓ (Segol & Lipman) |
| Regular repr. $\mathbb{R}^{S_n}$ | $S_n$ | Maximal | ✓ | ✓ | ✓ (Ravanbakhsh) |
| Cyclic CNN (any width) | $\mathbb{Z}_n$ | Maximal | ✓ | — | ✓ (Theorem 18) |

### Structural Factors Affecting Universality

| Factor | Effect on Universality | Remarks |
|---|---|---|
| Separation power | Necessary but not sufficient | Equal separation ≠ equal approximation |
| Depth | Can remedy shallow-network failures | Depth-3 PointNet recovers universality |
| Hidden representation type | Critical | Regular repr. > PointNet repr. > CNN(1) repr. |
| Group structure (normal subgroups) | Determines shallow universality | Abelian groups favorable; symmetric groups unfavorable |

### Key Findings

1. **Separation ≠ Universality**: Three network families share identical separation power yet exhibit strictly ordered approximation capacities—a central new finding in this area.
2. **Depth plays a role beyond parameter efficiency**: In equivariant networks, increasing depth does not merely reduce parameters but fundamentally expands the class of approximable functions.
3. **Group structure determines shallow-network limits**: For the symmetric group—one of the most important symmetry groups—shallow networks face fundamental barriers to universality.
4. **Differential constraint perspective**: The structure of equivariant layers constrains the "smooth directions" of approximable functions via differential operators, providing a novel analytical tool.

## Highlights & Insights

- The paper overturns the implicit assumption in the graph learning community that "separation power" and "expressive power" are equivalent.
- The theoretical contribution is deep: Theorem 13 reformulates neural network approximation as a differential operator annihilation problem, bridging algebra and analysis.
- The proof strategy in Proposition 16 (applying different theorems for different values of $n$) reveals the subtlety of the problem.
- The emergence of the normal subgroup condition exposes the profound influence of group structure—not merely group size—on network expressive power.

## Limitations & Future Work

- Only shallow networks (depth $\leq 2$) are analyzed; universality analysis for deep equivariant networks remains open.
- The interaction between depth and separation (e.g., increased depth improving separation in IGNs) is not addressed.
- No empirical validation of theoretical predictions is provided (the work is purely mathematical).
- For cases not satisfying the normal subgroup condition (e.g., $S_n$), no positive sufficient conditions for universality are given.

## Related Work & Insights

- The results generalize Ravanbakhsh (2020)'s universality result for regular representations to a broader family of representations.
- The correspondence with classical approximation theory (Cybenko and Hornik's universal approximation theorems) in the equivariant setting is made explicit.
- Design implication: when constructing equivariant networks, one must attend not only to what can be distinguished (separation) but also to what can be approximated (universality)—the two objectives may call for different architectural choices.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First systematic decoupling of universality and separation for equivariant networks; the core finding reshapes the community's understanding.
- Experimental Thoroughness: ⭐⭐⭐ — No experiments (purely theoretical), but the theoretical results are sufficiently powerful on their own.
- Writing Quality: ⭐⭐⭐⭐ — Dense mathematical notation, but definitions are clear and examples (PointNet, CNN) are well chosen for reader intuition.
- Value: ⭐⭐⭐⭐⭐ — A foundational contribution to equivariant deep learning theory that will influence the theoretical analysis paradigm for future architecture design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning (Approximately) Equivariant Networks via Constrained Optimization](learning_approximately_equivariant_networks_via_constrained_optimization.md)
- [\[ICML 2026\] Identifiable Equivariant Networks are Layerwise Equivariant](../../ICML2026/others/identifiable_equivariant_networks_are_layerwise_equivariant.md)
- [\[ICML 2025\] Permutation Equivariant Neural Networks for Symmetric Tensors](../../ICML2025/others/permutation_equivariant_neural_networks_for_symmetric_tensors.md)
- [\[ICML 2025\] GLGENN: A Novel Parameter-Light Equivariant Neural Networks Architecture Based on Clifford Geometric Algebras](../../ICML2025/others/glgenn_a_novel_parameter-light_equivariant_neural_networks_architecture_based_on.md)
- [\[ICML 2025\] The Price of Freedom: Exploring Expressivity and Runtime Tradeoffs in Equivariant Networks](../../ICML2025/others/the_price_of_freedom_exploring_expressivity_and_runtime_tradeoffs_in_equivariant.md)

</div>

<!-- RELATED:END -->

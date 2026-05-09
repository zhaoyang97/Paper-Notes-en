---
title: >-
  [Paper Note] Collapsing Taylor Mode Automatic Differentiation
description: >-
  [NeurIPS 2025][Scientific Computing][Taylor mode AD] This paper proposes a *collapsing* optimization technique for Taylor mode automatic differentiation. By rewriting the computation graph to propagate derivative summation operations upward, it substantially accelerates the evaluation of PDE operators (e.g., Laplacian, general linear PDE operators), achieving speeds superior to nested backpropagation while retaining the low-memory advantage of forward-mode AD.
tags:
  - NeurIPS 2025
  - Scientific Computing
  - Taylor mode AD
  - automatic differentiation
  - PDE operators
  - forward mode
  - computation graph optimization
date: 2026-05-08
content_hash: 1f7fb8118e637b11
---

# Collapsing Taylor Mode Automatic Differentiation

**Conference**: NeurIPS 2025
**arXiv**: [2505.13644](https://arxiv.org/abs/2505.13644)
**Code**: To be confirmed
**Area**: Scientific Computing / Automatic Differentiation / PDE
**Keywords**: Taylor mode AD, automatic differentiation, PDE operators, forward mode, computation graph optimization

## TL;DR
This paper proposes a *collapsing* optimization technique for Taylor mode automatic differentiation. By rewriting the computation graph to propagate derivative summation operations upward, it substantially accelerates the evaluation of PDE operators (e.g., Laplacian, general linear PDE operators), achieving speeds superior to nested backpropagation while retaining the low-memory advantage of forward-mode AD.

## Background & Motivation

**Background**: Scientific machine learning (SciML) requires extensive computation of partial differential equation operators such as the Laplacian $\nabla^2 u$ and divergence $\nabla \cdot \mathbf{F}$. The dominant approach computes high-order derivatives via nested backpropagation—applying reverse-mode AD once to obtain first-order derivatives, then again to obtain second-order derivatives.

**Limitations of Prior Work**:
   - Nested backpropagation incurs high computational cost: each additional order of differentiation requires an additional backward pass over the entire computation graph, severely limiting the practicality of SciML.
   - Nested backpropagation demands substantial memory, as intermediate activations must be retained.
   - Recent methods such as the forward Laplacian and randomized Taylor mode AD improve forward-pass efficiency but have not fully addressed the redundant computation inherent in Taylor mode itself.

**Key Challenge**: SciML requires efficient computation of high-order derivatives in PDE operators, yet these operators typically require only **linear combinations** of derivatives (e.g., the Laplacian is the sum of second-order derivatives along each coordinate direction). Standard Taylor mode computes all individual high-order derivative components independently before summing, introducing substantial redundancy.

**Goal**: Identify and eliminate redundancy in Taylor mode AD when evaluating linear PDE operators, and achieve meaningful acceleration through graph rewriting.

**Key Insight**: For linear PDE operators, only a weighted sum of derivative components is ultimately needed. Propagating this summation operation earlier in the computation graph avoids the wasteful pattern of computing each component independently and then aggregating.

**Core Idea**: *Collapse* linear combinations (summations) of derivatives into the Taylor mode propagation process by rewriting the computation graph, thereby eliminating redundant computation.

## Method

### Overall Architecture
The inputs are a neural network and a specification of a linear PDE operator (e.g., Laplacian or a general linear PDE operator). The method applies graph-rewriting optimizations to the Taylor mode AD computation graph: summation operations that would ordinarily be accumulated at the output are distributed across every node in the graph, so that each node propagates a single collapsed scalar rather than a full derivative tensor. The result is a rewritten computation graph that is mathematically equivalent to the original but substantially faster to evaluate.

### Key Designs

1. **Summation Collapsing in Taylor Mode AD**:

    - Function: Propagate the summation operation of a linear PDE operator into each node of the computation graph at the earliest opportunity.
    - Mechanism: Standard Taylor mode maintains a derivative tensor at each node (mixed partial derivatives of all orders). For the Laplacian $\sum_{i=1}^d \frac{\partial^2 f}{\partial x_i^2}$, the standard approach computes all $\frac{\partial^2 f}{\partial x_i^2}$ independently and then sums. The collapsing operation exploits the linearity of Taylor mode propagation rules to directly propagate the post-summation value $\sum_i$ at each node, compressing $d$ independent Taylor coefficients into a single scalar.
    - Design Motivation: PDE operators are linear, and linear operations commute with Taylor propagation—this elementary algebraic fact carries substantial computational significance.

2. **Generalization to Arbitrary Linear PDE Operators**:

    - Function: Support not only the Laplacian but any linear PDE operator of the form $\mathcal{L} = \sum_\alpha c_\alpha \partial^\alpha$.
    - Mechanism: Any linear PDE operator can be expressed as a weighted sum of derivatives, where $c_\alpha$ are scalar coefficients and $\alpha$ is a multi-index. The collapsing technique applies uniformly to any such operator by incorporating the weighting coefficients during propagation.
    - Design Motivation: SciML encompasses far more than the Laplacian (e.g., Helmholtz equation, wave equation, Navier–Stokes equations); the method must be sufficiently general.

3. **Integration with Randomized Taylor Mode**:

    - Function: Apply the collapsing technique jointly with randomized Taylor mode AD.
    - Mechanism: Randomized Taylor mode substitutes random directions for deterministic coordinate directions to estimate PDE operators, and is itself a cost-reduction technique. Collapsing is orthogonal to randomization, and the two can be applied simultaneously for compounded gains.
    - Design Motivation: Combining the two orthogonal optimizations yields greater overall acceleration.

4. **Implementation as a Compiler Optimization**:

    - Function: Since collapsing amounts to propagating a single summation through the computation graph, it is naturally suited for automation by ML compilers.
    - Mechanism: Users express the PDE operator at a high level; the compiler applies the collapsing transformation transparently.
    - Design Motivation: Lowering the barrier to adoption and making the optimization invisible to end users.

## Key Experimental Results

### Main Results
Wall-clock speedup relative to nested backpropagation and standard Taylor mode is evaluated on commonly used PDE operators:

| PDE Operator | Collapsed Taylor Mode | Standard Taylor Mode | Nested Backprop | Speedup |
|---|---|---|---|---|
| Laplacian | Fastest | Second | Slowest | Significant |
| General linear PDE operator | Fastest | Second | Slowest | Significant |
| Randomized + Collapsed | Fastest | — | Slowest | Further gain |

### Key Findings
- **Collapsing consistently accelerates Taylor mode**: Speedups are confirmed across all tested PDE operators.
- **Outperforms nested backpropagation**: Collapsed Taylor mode is faster than both standard Taylor mode and nested backpropagation, while preserving the low-memory advantage of forward-mode AD.
- **Orthogonal gains with randomization**: Collapsing further accelerates Taylor mode that already employs randomization; the two optimizations are complementary.

## Highlights & Insights
- **Conceptual simplicity of the optimization**: The technique reduces to exploiting linearity to exchange the order of summation and propagation—yet it yields substantial computational savings in practice. This exemplifies the "obvious in hindsight, yet previously unexplored" class of optimization.
- **Compiler-level optimization**: The authors argue that such transformations should be applied automatically by ML compilers—a philosophically sound design choice that decouples optimization from user-facing code.
- **Practical value for the SciML ecosystem**: PDE operator evaluation is a central bottleneck in SciML; accelerating it directly improves the usability of the entire field.

## Limitations & Future Work
- **Restricted to linear PDE operators**: The collapsing technique does not directly extend to nonlinear PDE operators (e.g., $|\nabla u|^2$), as summation does not commute with nonlinear operations.
- **Abstract-only information available**: Concrete speedup factors, detailed experimental configurations, and direct comparisons with the forward Laplacian require access to the full paper.
- **Compiler integration**: The authors note that this optimization "should be done by a ML compiler," but the engineering challenges of integrating it into JAX or PyTorch compilers are not elaborated.
- **Scalability to high dimensions**: When the input dimension $d$ is large (e.g., high-dimensional PDEs), the computational cost after collapsing may still be substantial.

## Related Work & Insights
- **vs. Forward Laplacian**: The forward Laplacian is a specialized forward-mode scheme designed specifically for the Laplacian; collapsed Taylor mode is more general, supporting arbitrary linear PDE operators.
- **vs. Nested Backpropagation**: Nested backpropagation is the default approach in current SciML; collapsed Taylor mode provides a more efficient forward-mode alternative.
- **vs. Randomized Taylor Mode**: The two approaches are orthogonal—randomization reduces the number of propagation directions, while collapsing reduces the per-direction cost; they can be composed.

## Rating
- Novelty: ⭐⭐⭐⭐ The optimization is elegant and conceptually clean; while technically straightforward, it delivers practical value to the field.
- Experimental Thoroughness: ⭐⭐⭐ Speedups are evaluated across multiple PDE operators; the NeurIPS camera-ready version is expected to contain comprehensive experiments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Hamiltonian Neural PDE Solvers through Functional Approximation](hamiltonian_neural_pde_solvers_through_functional_approximation.md)
- [\[NeurIPS 2025\] Bayesian Surrogates for Risk-Aware Pre-Assessment of Aging Bridge Portfolios](bayesian_surrogates_for_risk-aware_pre-assessment_of_aging_bridge_portfolios.md)
- [\[NeurIPS 2025\] The Primacy of Magnitude in Low-Rank Adaptation](the_primacy_of_magnitude_in_low-rank_adaptation.md)
- [\[NeurIPS 2025\] From Black Hole to Galaxy: Neural Operator Framework for Accretion and Feedback Dynamics](from_black_hole_to_galaxy_neural_operator_framework_for_accretion_and_feedback_d.md)
- [\[NeurIPS 2025\] From Images to Physics: Probabilistic Inference of Galaxy Parameters and Emission Lines via VAE–Normalizing Flows](one-shot_transfer_learning_for_nonlinear_pdes_with_perturbative_pinns.md)

</div>

<!-- RELATED:END -->

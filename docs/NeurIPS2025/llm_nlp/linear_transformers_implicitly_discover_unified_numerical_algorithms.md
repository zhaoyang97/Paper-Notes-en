---
title: >-
  [Paper Note] Linear Transformers Implicitly Discover Unified Numerical Algorithms
description: >-
  [NeurIPS 2025][LLM (Other)][in-context learning] After training linear Transformers on a masked block matrix completion task, algebraic analysis of the learned weights reveals that the models implicitly converge to the same two-line iterative update rule—EAGLE—under three fundamentally different computational constraints (centralized, distributed, and compute-limited). This rule achieves second-order convergence with only logarithmic dependence on the condition number.
tags:
  - "NeurIPS 2025"
  - "LLM (Other)"
  - "in-context learning"
  - "linear transformer"
  - "numerical algorithm discovery"
  - "matrix completion"
  - "Newton-Schulz"
date: 2026-05-08
content_hash: 959445987722ed28
---

# Linear Transformers Implicitly Discover Unified Numerical Algorithms

**Conference**: NeurIPS 2025
**arXiv**: [2509.19702](https://arxiv.org/abs/2509.19702)  
**Code**: To be confirmed  
**Area**: LLM/NLP
**Keywords**: in-context learning, linear transformer, numerical algorithm discovery, matrix completion, Newton-Schulz

## TL;DR
After training linear Transformers on a masked block matrix completion task, algebraic analysis of the learned weights reveals that the models implicitly converge to the same two-line iterative update rule—EAGLE—under three fundamentally different computational constraints (centralized, distributed, and compute-limited). This rule achieves second-order convergence with only logarithmic dependence on the condition number.

## Background & Motivation

The in-context learning (ICL) capability of Transformers has attracted considerable research attention. Prior work has shown that Transformer weights trained on ICL tasks encode algorithms resembling gradient descent. However, these analyses share several limitations:

**Single computational regime**: Prior work typically studies only a fixed, centralized computational setting, varying the complexity of the regression task rather than the resource constraints.

**Limitations of the gradient descent analogy**: Gradient descent operates in parameter space, whereas Transformers perform data-to-data transformations, with no direct analogue to parameter updates.

**Only first-order methods discovered**: A large body of prior work has only identified first-order optimization rules within Transformers.

This paper poses a core question: **Can a neural network trained solely to impute missing data "invent" a numerical algorithm?** And further: **When computational, communication, or memory budgets are tightened, do Transformers adaptively develop a unified algorithm?**

## Method

### Overall Architecture

The authors design a **masked-block completion** task framework:

Given a low-rank matrix $X = \begin{bmatrix} A & C \\ B & D \end{bmatrix}$ with $\text{rank}(X) = \text{rank}(A)$, the lower-right block $D$ is masked (set to zero), and the Transformer must recover $D$ from the visible blocks $A, B, C$. This formulation unifies Nyström completion and scalar regression.

**Linear attention Transformer architecture**:

$$Z_{\ell+1} = Z_\ell + \sum_h \text{Attn}_\ell^h(Z_\ell)$$

$$\text{Attn}_\ell^h(Z) = (ZW_Q(ZW_K)^\top \odot M)ZW_VW_P^\top$$

where $M$ is a fixed mask that prevents incomplete columns from attending to visible columns.

### Three Computational Regimes

By **only modifying the attention mask and dimensionality constraints**, three resource regimes are instantiated:

1. **Unconstrained (centralized)**: All tokens attend to each other; embedding dimension $k = n + n'$.
2. **Compute-limited**: Query/key matrices are restricted to low rank $k = r \ll n$ (with $r = 5 \approx n/4$ in experiments), reducing per-layer cost from $\Theta(n^2d)$ to $\Theta(nrd)$.
3. **Distributed**: Data is partitioned across $M$ machines; each attention head is assigned to one machine, permitting only local self-attention, with information exchanged via value/projection matrices.

### Key Findings: Emergence of the EAGLE Algorithm

**Algorithm extraction pipeline**: Weight quantization → matrix property detection (sparsity/low-rank structure) → scaling law identification.

The algorithms extracted across all three regimes are **identical**, and are named EAGLE (Emergent Algorithm for Global Low-rank Estimation).

Core two-line update rule (letting $\rho = \|A_\ell\|_2^{-2}$):

$$A_{\ell+1} = A_\ell + \eta\rho \cdot A_\ell(SA_\ell)^\top(SA_\ell)S^\top$$
$$C_{\ell+1} = C_\ell + \gamma\rho \cdot A_\ell(SA_\ell)^\top C_\ell$$

where $\eta \approx 1$, $\gamma \approx 1.9$, and $S$ is an orthogonal sketch matrix (with $S=I_n$ in the centralized setting).

**Discovered weight structure**:
- $W_{QK,\ell} \approx \text{diag}(\alpha_\ell^1 I_n, 0_{n'})$
- $W_{VP,\ell} \approx \text{diag}(\alpha_\ell^2 I_n, \alpha_\ell^3 I_{n'})$
- Each layer requires only 3 scalar parameters, satisfying $\alpha_\ell^1 \alpha_\ell^2 \approx \eta\|A_\ell\|_2^{-2}$

**Connection to Newton-Schulz**: The map $A \mapsto (I - \rho\eta AA^\top)A$ suppresses larger singular values of $A$ more than smaller ones; repeated application progressively conditions the matrix, making it well-conditioned. This is essentially a variant of the Newton-Schulz matrix inversion iteration.

### Loss & Training

The training loss is the mean squared error on the $D$ block:

$$\mathcal{L} = \mathbb{E}[\|D_L - D\|_F^2]$$

Data generation: $X = R_1 R_2^\top / \sqrt{s}$, where rows of $R_1, R_2$ are independently drawn from $\mathcal{N}(0, \Sigma)$ with $\Sigma_{ii} = \alpha^i$ ($\alpha=0.7$), yielding a condition number of approximately $10^3$. Half of the samples are corrupted with Gaussian noise ($\sigma^2=0.01$).

## Key Experimental Results

### Main Results

**Extracted algorithm fidelity** (difference between Transformer activations and EAGLE outputs, Frobenius norm $\times 10^4$):

| Layer | Unconstrained | Distributed | Compute-limited |
|-------|--------------|-------------|-----------------|
| 0 | 0.00 | 0.00 | 0.00 |
| 1 | 2.36 | 0.62 | 0.01 |
| 2 | 5.27 | 1.40 | 0.02 |
| 3 | 3.70 | 1.28 | 0.13 |
| 4 | 2.16 | 1.32 | 0.14 |

Errors remain below $6 \times 10^{-4}$, confirming that EAGLE precisely reproduces the Transformer's layer-wise computation.

**Convergence in the centralized setting** ($d=n=240$, $\kappa=10^2$):

| Method | Iterations to reach $\varepsilon=10^{-20}$ ($\kappa=10^4$) |
|--------|------|
| Gradient Descent (GD) | ~$10^4$ |
| Conjugate Gradient (CG) | ~$10^2$ |
| **EAGLE** | ~$10^0$ (~1–2 steps) |

At $\kappa=10^4$, EAGLE is approximately 100× faster than CG.

### Ablation Study

**Distributed setting**:
- **Sweep over number of machines $M$**: For $M \in \{1,3,5,8\}$, when $\alpha \approx 1$ (worker subspaces are nearly orthogonal), the iteration count is ≤15 and independent of $M$.
- **Sweep over diversity index $\alpha$**: For $\alpha \in \{1, 0.9, 0.36, 0.004\}$, iteration complexity scales linearly with $\alpha^{-1}$, yet remains 10–100× better than GD throughout.

**Compute-limited setting**:
- **Sweep over sketch rank $r$**: For $r \in \{n/8, n/4, n/2, n\}$, the iteration count scales linearly with $n/r$.
- **Wall-clock time per iteration**: 5 ms at $r=30$ vs. 21 ms at $r=240$; computational cost decreases linearly with sketch size.

### Key Findings

1. **Remarkable universality**: Under three entirely different resource constraints, Transformers converge to exactly the same update rule.
2. **Second-order convergence**: EAGLE's iteration complexity is $O(\log\kappa + \log\log(\varepsilon^{-1}))$, far superior to CG's $O(\sqrt{\kappa}\log(\varepsilon^{-1}))$ and GD's $O(\kappa^2\log(\varepsilon^{-1}))$.
3. **Communication efficiency in the distributed setting**: Each machine communicates only $O((d+d')n')$ floats per round (only $2d$ in the scalar prediction case), and Transformers spontaneously discover sparse communication patterns without any explicit communication constraints.
4. **Sketching emerges spontaneously**: The compute-limited model naturally produces a random orthogonal sketch matrix within $W_{QK}$.

## Highlights & Insights

1. **Strong evidence that "Transformers can invent algorithms"**: This is not merely the rediscovery of a gradient descent variant, but a previously unknown method with independent value in numerical analysis.
2. **Data-to-data perspective**: Rather than interpreting Transformer forward passes as parameter updates (the gradient descent view), this work directly characterizes data transformations—a more faithful description of the actual computation.
3. **Logarithmic dependence on condition number**: EAGLE's $\log\kappa$ complexity occupies a previously unexplored speed–accuracy trade-off regime among iterative solvers.
4. **Implications for practical system design**: Low-rank attention constraints are not merely an efficiency trick; they naturally induce principled data compression (sketching) strategies.

## Limitations & Future Work

1. **Restricted to linear attention**: Whether analogous unified algorithms can be extracted from nonlinear (softmax) attention remains an open question.
2. **Numerical stability**: Behavior for $\kappa > 10^8$ has not been tested.
3. **Single training recipe**: Whether alternative training designs can also induce EAGLE is an open question.
4. **Limited problem scale**: Experiments use matrices of dimension $n=d=240$; performance on large-scale problems remains to be verified.
5. **Non-low-rank problems**: The framework assumes low-rank structure; applicability to general matrices is unknown.

## Related Work & Insights

- **Relation to Von Oswald et al. (2023)**: The same update rule in the scalar case ($d'=n'=1$) is referred to as "GD++"; however, the authors argue that the gradient descent interpretation is inaccurate, and that the mechanism is fundamentally a Newton-Schulz conditioning loop.
- **Comparison with Krylov methods**: Classical Nyström approximation typically employs Krylov solvers such as CG; EAGLE offers an entirely different path.
- **Comparison with communication-avoiding Krylov methods**: Distributed EAGLE's communication cost depends on the "data diversity index" $\alpha^{-1}$, which can be substantially smaller than the joint condition number.
- **Broader implications**: This work may open a research direction in which neural networks are trained to discover optimal numerical algorithms tailored to specific computational constraints.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Discovering that Transformers spontaneously converge to a unified numerical algorithm across different resource constraints is highly pioneering.
- Experimental Thoroughness: ⭐⭐⭐⭐ All three settings are validated with both theory and experiments, though problem scales are modest.
- Writing Quality: ⭐⭐⭐⭐ Rigorous structure and clear mathematical exposition, though the high information density makes for demanding reading.
- Value: ⭐⭐⭐⭐⭐ Advances understanding of ICL mechanisms while simultaneously discovering a numerical method with independent value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] In-Context Learning of Linear Dynamical Systems with Transformers: Approximation Bounds and Depth-Separation](in-context_learning_of_linear_dynamical_systems_with_transformers_approximation_.md)
- [\[NeurIPS 2025\] Composing Linear Layers from Irreducibles](composing_linear_layers_from_irreducibles.md)
- [\[NeurIPS 2025\] Strassen Attention, Split VC Dimension and Compositionality in Transformers](strassen_attention_split_vc_dimension_and_compositionality_in_transformers.md)
- [\[NeurIPS 2025\] CAT: Circular-Convolutional Attention for Sub-Quadratic Transformers](cat_circular-convolutional_attention_for_sub-quadratic_transformers.md)
- [\[NeurIPS 2025\] What One Cannot, Two Can: Two-Layer Transformers Provably Represent Induction Heads on Any-Order Markov Chains](what_one_cannot_two_can_two-layer_transformers_provably_represent_induction_head.md)

</div>

<!-- RELATED:END -->

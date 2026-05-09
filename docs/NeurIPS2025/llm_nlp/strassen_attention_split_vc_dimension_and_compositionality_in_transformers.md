---
title: >-
  [Paper Note] Strassen Attention, Split VC Dimension and Compositionality in Transformers
description: >-
  [NeurIPS 2025][LLM/NLP][Transformer theory] This paper introduces the Splitting VC dimension as a theoretical tool to prove fundamental limitations of single-layer softmax Transformers (even with infinite precision) on compositional reasoning tasks, and proposes the Strassen attention mechanism with sub-cubic time complexity to overcome these limitations.
tags:
  - NeurIPS 2025
  - LLM/NLP
  - Transformer theory
  - attention mechanism
  - compositional reasoning
  - VC dimension
  - complexity lower bounds
date: 2026-05-08
content_hash: 818999ac3a8a0b65
---

# Strassen Attention, Split VC Dimension and Compositionality in Transformers

**Conference**: NeurIPS 2025
**arXiv**: [2501.19215](https://arxiv.org/abs/2501.19215)
**Code**: Not released
**Area**: LLM/NLP
**Keywords**: Transformer theory, attention mechanism, compositional reasoning, VC dimension, complexity lower bounds

## TL;DR

This paper introduces the Splitting VC dimension as a theoretical tool to prove fundamental limitations of single-layer softmax Transformers (even with infinite precision) on compositional reasoning tasks, and proposes the Strassen attention mechanism with sub-cubic time complexity to overcome these limitations.

## Background & Motivation

Transformer architectures achieve strong performance on many NLP tasks, yet consistently struggle on tasks requiring compositionality — the ability to combine knowledge components to solve novel problems, widely regarded as the core of systematic generalization in language systems.

Limitations of prior theoretical work:
- Hahn (2020) establishes bounds using hardmax attention, which diverges significantly from the softmax used in practice.
- Sanford et al. (2023) and Peng et al. (2024) employ communication complexity-based approaches that **apply only to finite-precision** (sub-linear bit-width) Transformers.
- This raises a critical question: can higher-precision arithmetic circumvent these limitations?

Furthermore, existing higher-order attention mechanisms (e.g., third-order attention) are theoretically capable of solving these problems but incur $O(n^3)$ time complexity, making them poorly scalable.

The core motivations of this paper are:
1. To establish theoretical limitations for **arbitrary-precision** (including infinite-precision) softmax Transformers.
2. To design a **more efficient** attention mechanism that overcomes these limitations.

## Method

### Overall Architecture

The paper makes two primary contributions: **theoretical lower bounds** (via Splitting VC dimension) and a **novel attention mechanism** (Strassen attention).

### Key Design 1: Splitting VC Dimension

The core idea is to **split** the $n$ arguments of a Boolean function $f: \Sigma^n \to \{0,1\}$ into two parts — one treated as inputs and the other as parameters — thereby casting the function as a hypothesis class and computing its VC dimension.

Given a position set $A \subseteq \{1,\ldots,n\}$, a Boolean matrix $M_f^A$ is constructed as follows:
- Rows are indexed by $w^1 \in \Sigma^A$ (input part).
- Columns are indexed by $w^2 \in \Sigma^B$ (parameter part), where $B = [n] \setminus A$.
- Entries are defined as $M_A^f(w^1, w^2) = f(w^1 \oplus w^2)$.

**Definition**: $\text{split-VC}(f)$ is the maximum VC dimension of the column set of $M_f^A$ over all possible splits $A$.

### Key Design 2: Main Theorem

**Theorem 3.2**: Let $T$ be a single-layer standard-attention Transformer (operating with **real-valued infinite-precision** arithmetic) with embedding dimension $d$, number of attention heads $H$, and output MLP size $L$. If $T$ computes a function $f$, then:

$$\text{size}(T) = \max\{d, H, L\} \geq \text{split-VC}(f)^{\Omega(1)}$$

The proof proceeds by:
1. Decomposing the attention output at an auxiliary token position into a fractional form $\frac{\alpha^{(h)}(w^1) + \beta^{(h)}(w^2) + \gamma^{(h)}}{\lambda^{(h)}(w^1) + \mu^{(h)}(w^2) + \nu^{(h)}}$.
2. Applying the Goldberg–Jerrum theorem to bound the VC dimension of parameterized function classes.
3. This decomposition depends only on the summation structure of attention and is thus precision-independent.

### Key Design 3: Lower Bounds for Three Tasks

| Task | Definition | split-VC Lower Bound |
|------|------------|----------------------|
| Function composition | Given $g,h:[n]\to[n]$, compute $h(g(x))$ | $\geq n$ |
| Match3 | Determine whether $\exists\, j,k$ s.t. $p_i + p_j + p_k \equiv 0 \pmod{m}$ | $\geq n/2$ |
| Boolean matrix product | Compute $(B \circ A)_{ij} = \bigvee_k (A_{ik} \wedge B_{kj})$ | $\geq \sqrt{n}$ |

These results imply that single-layer standard-attention Transformers with $\text{size}(T) = n^{o(1)}$ **cannot** solve any of these tasks.

### Key Design 4: Strassen Attention

Inspired by Strassen's matrix multiplication algorithm, Strassen attention is defined as:

$$a_i = \sum_{j,k} a_{ijk}(v_j \odot v_k)$$

$$a_{ijk} = \text{Softmax}_{j,k}\left(\frac{f_i g_j + g_j h_k + h_k f_i}{\sqrt{d}}\right)$$

where $f_i = W^f x_i$, $g_j = W^g x_j$, and $h_k = W^h x_k$.

Key advantage: the attention scores use **sums of pairwise dot products** rather than full three-index products, enabling decomposition into a constant number of $n \times n$ matrix multiplications.

**Theorem 4.1**: A Strassen attention layer runs in time $O(n^\omega \cdot d)$, where $\omega < 2.372$ is the matrix multiplication exponent — a significant improvement over the $O(n^3)$ complexity of third-order attention.

### Loss & Training

This paper is primarily theoretical. The experimental component trains with the AdamW optimizer at a learning rate of $10^{-3}$ on synthetic data, using identical training configurations across all attention variants for fair comparison.

## Key Experimental Results

### Main Results

Performance of different attention mechanisms across four compositional reasoning tasks:

| Attention Type | Function Composition | Match3 | Boolean Matrix Product | Quotient Boolean Matrix Product | Time Complexity |
|----------------|----------------------|--------|------------------------|---------------------------------|-----------------|
| Standard attention | ✗ Fails | ✗ Fails | ✗ Fails | ✗ Fails | $O(n^2 d)$ |
| Triangular attention | ✗ Fails | — | ✓ Succeeds | ✗ Fails | $O(n^{3/2} d)$* |
| Third-order attention | ✓ Succeeds | ✓ Succeeds | ✓ Succeeds | ✓ Succeeds | $O(n^3 d)$ |
| **Strassen attention** | **✓ Succeeds** | **✓ Succeeds** | **✓ Succeeds** | **✓ Succeeds** | $O(n^\omega d)$ |

*Triangular attention achieves $O(n^{3/2})$ only when the input is already structured as a matrix.

### Ablation Study

- **Training efficiency**: Strassen attention is more efficient in training time and computational resources than third-order attention, as it avoids $O(n^3)$ operations.
- **Effect of embedding dimension**: Experiments confirm the theoretical prediction that standard attention cannot learn these tasks at a fixed small dimension, even with extended training.
- **Quotient Boolean matrix product task**: This newly designed task specifically distinguishes Strassen attention from combinations of standard and triangular attention.

### Key Findings

1. Standard attention fails completely on all four tasks, empirically validating the theoretical lower bounds.
2. Strassen attention converges to the theoretically correct solution on all tasks while being more efficient than third-order attention.
3. Triangular attention can only solve the Boolean matrix product task when the input is naturally matrix-structured.
4. The quotient Boolean matrix product task successfully differentiates Strassen attention from hybrid combinations of standard and triangular attention.

## Highlights & Insights

1. **Precision-independent lower bounds**: This work is the first to prove that even with infinite-precision arithmetic, single-layer standard attention cannot solve certain compositional tasks — establishing a true **expressive power** limitation rather than a precision limitation.
2. **Elegant theoretical tool**: The Splitting VC dimension is a concise yet powerful concept that bridges combinatorial optimization with classical VC dimension from learning theory.
3. **Cross-domain inspiration from Strassen's algorithm**: The ideas underlying Strassen's matrix multiplication algorithm (Strassen, 1969) are ingeniously adapted to the design of attention mechanisms.
4. **Theory–experiment consistency**: All theoretical predictions are precisely confirmed by the experimental results.

## Limitations & Future Work

1. **Single-layer analysis only**: The theoretical results apply solely to single-layer Transformers; expressive power analysis for multi-layer settings remains an open problem.
2. **Experiments limited to synthetic data**: The practical utility of Strassen attention on real-world NLP tasks has not been validated.
3. **Implementation overhead**: Although the theoretical time complexity is superior, the constant factors in fast matrix multiplication algorithms may limit practical speedups.
4. **Gap with modern Transformer architectures**: The analysis does not account for the effects of practical components such as layer normalization and multi-layer interactions.
5. **Trainability of Strassen attention**: Although experiments demonstrate convergence, training stability at scale remains unknown.

## Related Work & Insights

- **Relation to Sanford et al. (2023)**: This work extends their lower bound for the Match3 task to the infinite-precision setting.
- **Relation to Peng et al. (2024)**: Both study function composition, but this paper removes the precision dependence inherent in communication complexity-based approaches.
- **Relation to Bergen et al. (2021) triangular attention**: Experiments show that Strassen attention solves tasks that triangular attention cannot handle.
- **Broader implication**: The Splitting VC dimension methodology may generalize to analyzing expressive power limitations of a wider range of neural network architectures.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Both Splitting VC dimension and Strassen attention are entirely original contributions.
- **Theoretical Depth**: ⭐⭐⭐⭐⭐ — Rigorous proofs revealing deep structural limitations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Synthetic experiments thoroughly validate the theory, though real-task evaluation is absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear organization and precise mathematical exposition.
- **Value**: ⭐⭐⭐ — Currently more theoretical in nature; the practical applicability of Strassen attention warrants further exploration.
- **Overall**: ⭐⭐⭐⭐⭐ (9/10) — A top-tier theoretical contribution that elegantly connects complexity theory with Transformer architecture design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] CAT: Circular-Convolutional Attention for Sub-Quadratic Transformers](cat_circular-convolutional_attention_for_sub-quadratic_transformers.md)
- [\[NeurIPS 2025\] Spectral Conditioning of Attention Improves Transformer Performance](spectral_conditioning_of_attention_improves_transformer_performance.md)
- [\[AAAI 2026\] Vision Transformers are Circulant Attention Learners](../../AAAI2026/llm_nlp/vision_transformers_are_circulant_attention_learners.md)
- [\[NeurIPS 2025\] Linear Transformers Implicitly Discover Unified Numerical Algorithms](linear_transformers_implicitly_discover_unified_numerical_algorithms.md)
- [\[NeurIPS 2025\] Don't Be Lazy: CompleteP Enables Compute-Efficient Deep Transformers](dont_be_lazy_completep_enables_compute-efficient_deep_transformers.md)

</div>

<!-- RELATED:END -->

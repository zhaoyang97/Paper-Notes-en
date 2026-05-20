---
title: >-
  [Paper Note] Composing Linear Layers from Irreducibles
description: >-
  [NeurIPS 2025][LLM/NLP][Clifford algebra] By leveraging Clifford algebra, this work represents linear layers as compositions of bivectors—specifically as rotor sandwich products—requiring only $O(\log^2 d)$ parameters to…
tags:
  - "NeurIPS 2025"
  - "LLM/NLP"
  - "Clifford algebra"
  - "rotor decomposition"
  - "parameter efficiency"
  - "linear layers"
  - "geometric primitives"
date: 2026-05-08
content_hash: 607a6c1a3ee9433a
---

# Composing Linear Layers from Irreducibles

**Conference**: NeurIPS 2025
**arXiv**: [2507.11688](https://arxiv.org/abs/2507.11688)  
**Code**: To be confirmed  
**Area**: LLM/NLP
**Keywords**: Clifford algebra, rotor decomposition, parameter efficiency, linear layers, geometric primitives

## TL;DR
By leveraging Clifford algebra, this work represents linear layers as compositions of bivectors—specifically as rotor sandwich products—requiring only $O(\log^2 d)$ parameters to replace a $d \times d$ dense matrix. When applied to Q/K/V projections in LLM attention layers, performance closely matches the original model and strong baselines.

## Background & Motivation

**Background**: Linear layers account for the vast majority of parameters in modern large language models (e.g., a single 2048×2048 projection in LLaMA's self-attention requires 4M parameters). Parameter-efficient methods such as LoRA and block-Hadamard compress linear layers from different perspectives, but lack a principled understanding of the intrinsic algebraic structure of linear transformations.

**Limitations of Prior Work**: (a) LoRA assumes low-rank structure and constitutes a functional approximation; (b) block-Hadamard assumes structured matrices, also an approximation; (c) both methods approximate dense matrices in a "top-down" manner rather than constructing functionality "bottom-up" from fundamental primitives. A formal algebraic characterization of the minimal irreducible primitives of linear transformations is absent.

**Key Challenge**: Dense matrices require $O(d^2)$ parameters, yet the "essential degrees of freedom" of a linear transformation may be far smaller. The question is whether a minimal set of geometric primitives can be identified whose composition reproduces the functionality of a linear layer.

**Goal**: To describe the algebraic structure of linear transformations using the language of Clifford algebra, employing bivectors as irreducible primitives and composing rotors to achieve extreme parameter compression.

**Key Insight**: In Clifford algebra, linear transformations can be expressed as multivector products of the form $F(x) = \sum_t a_t x b_t$. Restricting $a_t, b_t$ to rotors (elements of the Spin group) and parameterizing rotors via bivectors yields exponential parameter compression.

**Core Idea**: Bivectors serve as the geometric primitives of linear transformations. Through the action of rotor sandwich products on multivector spaces, effective linear transformations can be constructed with only $O(\log^2 d)$ parameters.

## Method

### Overall Architecture

An input vector $x \in \mathbb{R}^d$ is re-encoded as a multivector in the Clifford algebra $\text{Cl}(n)$ (where $n = \log_2 d$), then transformed via the sandwich products $r_t x r_t^\dagger$ of multiple rotors, and finally aggregated to produce the output. The learnable parameters consist solely of a small number of bivector coefficients.

### Key Designs

1. **Clifford Algebra Representation**:

    - Function: Converts linear transformations from matrix representations into multivector product form within Clifford algebra.
    - Mechanism: By Lemma 1 (Hestenes, 2012), any linear function $F: \text{Cl}^k(n) \to \text{Cl}(n)$ can be written as a finite sum $F(x) = \sum_{t=1}^w a_t x b_t$. Restricting $a_t = r_t$, $b_t = r_t^\dagger$ to rotors (elements of Spin(n)) makes $x \mapsto r x r^\dagger$ an orientation-preserving rotation.
    - Key Advantage: Rotors act on the $2^n$-dimensional multivector space (where $n = \log_2 d$), yet require only $\binom{n}{2}$ bivector parameters. For $d = 2048$, a single rotor needs only 55 parameters versus 4M for a dense matrix.

2. **Differentiable Invariant Decomposition Algorithm**:

    - Function: Decomposes a general bivector (which may not be "simple") into a sum of mutually orthogonal, commuting simple bivectors, enabling a closed-form rotor expression.
    - Mechanism: The exponential map of a simple bivector $b = u \wedge v$ has a closed-form solution: $\exp(b) = \cos(\|b\|) + \frac{\sin(\|b\|)}{\|b\|} b$. A general bivector decomposes into a sum of at most $k = \lfloor n/2 \rfloor$ simple bivectors $b = \sum_j b_j$; mutual orthogonality yields $\exp(\sum b_j) = \prod \exp(b_j)$.
    - Algorithm 1 is proposed: simple components are sequentially extracted from the bivector via projection operations, with all steps differentiable through autograd.
    - Design Motivation: Avoids approximation errors from truncating infinite series while maintaining end-to-end differentiability.

3. **Rotor-Gadget Module Design**:

    - Function: Encapsulates rotor operations into a module that can replace dense linear layers.
    - Mechanism: Input $x \in \mathbb{R}^d$ → reshaped into a $2^n$-dimensional multivector → transformed through $w$ rotor sandwich products $\{r_t x r_t^\dagger\}_{t=1}^w$ → output aggregated via learnable mixing coefficients.
    - Width $w$ is a hyperparameter (typical value ~3); total parameter count is $w \times \binom{\log_2 d}{2}$.
    - Application: Replaces Q/K/V projection matrices in LLM attention layers.

### Loss & Training
- Evaluated on LLaMA-2-7B and LLaMA-3-8B.
- All Q/K/V projections in attention layers are replaced with rotor-gadgets.
- Training proceeds via distillation/approximation to recover the functionality of the original layers rather than training from scratch.
- Evaluation metrics: perplexity on WikiText-2 and C4, and multiple downstream zero-shot tasks.

## Key Experimental Results

### Main Results: LLaMA-2-7B Perplexity (log-PPL)

| Method | Params | WikiText-2 Q | WikiText-2 K | WikiText-2 V | C4 Q | C4 K | C4 V |
|--------|--------|-------------|-------------|-------------|------|------|------|
| Original | 100% | 2.575 | 2.575 | 2.575 | 3.151 | 3.151 | 3.151 |
| LR1 (rank-1) | Very few | 2.688 | 3.455 | 4.956 | 3.414 | 4.071 | 5.001 |
| LR4 (rank-4) | Few | 2.658 | 2.729 | 2.880 | 3.390 | 3.315 | 3.504 |
| BH1 (block-Hadamard) | Few | 2.636 | 2.700 | 2.779 | 3.343 | 3.262 | 3.404 |
| **Rotor** | **Very few** | **2.629** | **2.717** | **2.818** | **3.261** | **3.285** | **~3.4** |

The rotor-gadget matches or outperforms the block-Hadamard baseline in most settings, and substantially surpasses naive low-rank approximations.

### Ablation Study: Parameter Count Comparison

| Method | Parameters at d=2048 |
|--------|----------------------|
| Dense matrix | ~4.2M |
| LoRA (rank-4) | ~16K |
| Block-Hadamard | ~11 bits/row |
| **Rotor (w=3)** | **~165** |

The rotor-gadget uses two orders of magnitude fewer parameters than LoRA.

### Key Findings
- The rotor-gadget achieves near-original performance under extreme parameter compression (165 parameters vs. 4.2M), demonstrating that linear transformations possess exploitable algebraic structure.
- The Q projection is easiest to approximate (smallest perplexity increase), while the V projection is hardest (largest increase)—consistent with the functional distinctions among Q/K/V observed in mechanistic interpretability research.
- Width $w=3$ suffices; additional rotors yield diminishing returns.
- LLaMA-3 is harder to approximate than LLaMA-2, possibly because its projection layers encode more complex functionality.

## Highlights & Insights
- **Mathematical Elegance**: Grounding the work in Clifford algebra—a well-established mathematical framework—formalizes the "minimal primitives" of linear layers as bivectors. The $O(\log^2 d)$ parameter complexity arises naturally from the exponential dimensional lift of Clifford algebra ($n$-dimensional space generates a $2^n$-dimensional algebra), yielding a result that is both natural and elegant.
- **Differentiable Invariant Decomposition** constitutes the key technical contribution: it eliminates truncation errors in the exponential map and ensures gradient flow throughout.
- **A New Perspective on Linear Layers**: Rather than asking "how to compress," this work asks "what are the essential degrees of freedom of a linear transformation?"—the answer being a small number of planar rotations.

## Limitations & Future Work
- The approach is not yet a practical drop-in replacement—system-level integration (e.g., custom CUDA kernels) would be required to realize speed advantages.
- Evaluation is limited to Q/K/V projections; the FFN layers, which account for the majority of linear layer parameters, remain untested.
- $d$ must be a power of two for clean embedding decomposition; arbitrary dimensions require padding.
- Training proceeds by distilling/approximating existing weights rather than training from scratch—whether rotor parameterization supports efficient training from random initialization remains unverified.
- Convergence analysis is absent; the optimization landscape of multi-rotor compositions may exhibit local optima.

## Related Work & Insights
- **vs. LoRA**: LoRA assumes low-rank weight updates; rotors assume that transformations can be composed from planar rotations. The two approaches are orthogonal and may be combinable.
- **vs. Block-Hadamard**: Block-Hadamard uses structured orthogonal matrices; rotors also yield orthogonal transformations but with a distinct parameterization. Performance is comparable, but rotors use far fewer parameters.
- **vs. Other Applications of Geometric Algebra in Deep Learning**: Clifford layers in GANs (Ruhe et al.) target equivariance, whereas this work targets parameter efficiency and decomposability.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Introducing Clifford algebra to LLM parameter efficiency represents a genuinely novel perspective with rigorous mathematical foundations.
- Experimental Thoroughness: ⭐⭐⭐ Proof-of-concept level; lacks experiments on training from scratch, broader model coverage, and inference speed comparisons.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are clear, though the presentation poses a high barrier for readers unfamiliar with Clifford algebra.
- Value: ⭐⭐⭐⭐ Opens a new direction for understanding and compressing linear layers, though practical utility requires further validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Linear Transformers Implicitly Discover Unified Numerical Algorithms](linear_transformers_implicitly_discover_unified_numerical_algorithms.md)
- [\[NeurIPS 2025\] In-Context Learning of Linear Dynamical Systems with Transformers: Approximation Bounds and Depth-Separation](in-context_learning_of_linear_dynamical_systems_with_transformers_approximation_.md)
- [\[CVPR 2026\] Composing Concepts from Images and Videos via Concept-prompt Binding](../../CVPR2026/llm_nlp/composing_concepts_from_images_and_videos_via_concept-prompt_binding.md)
- [\[NeurIPS 2025\] Nemotron-Flash: Towards Latency-Optimal Hybrid Small Language Models](nemotron-flash_towards_latency-optimal_hybrid_small_language_models.md)
- [\[NeurIPS 2025\] Adaptive Kernel Design for Bayesian Optimization Is a Piece of CAKE with LLMs](adaptive_kernel_design_for_bayesian_optimization_is_a_piece_of_cake_with_llms.md)

</div>

<!-- RELATED:END -->

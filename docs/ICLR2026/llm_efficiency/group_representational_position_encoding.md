---
title: >-
  [Paper Note] Group Representational Position Encoding (GRAPE)
description: >-
  [ICLR 2026][LLM Efficiency][positional encoding] This paper proposes the GRAPE framework, which unifies the multiplicative (RoPE) and additive (ALiBi/FoX) families of positional encodings in Transformers via group action…
tags:
  - "ICLR 2026"
  - "LLM Efficiency"
  - "positional encoding"
  - "group theory"
  - "RoPE"
  - "ALiBi"
  - "Lie groups"
  - "rotary encoding"
  - "long context"
date: 2026-05-08
content_hash: 26f23b6b71394a03
---

# Group Representational Position Encoding (GRAPE)

**Conference**: ICLR 2026
**arXiv**: [2512.07805](https://arxiv.org/abs/2512.07805)  
**Code**: [github.com/model-architectures/GRAPE](https://github.com/model-architectures/GRAPE)  
**Area**: Signal & Communication
**Keywords**: positional encoding, group theory, RoPE, ALiBi, Lie groups, rotary encoding, long context

## TL;DR

This paper proposes the GRAPE framework, which unifies the multiplicative (RoPE) and additive (ALiBi/FoX) families of positional encodings in Transformers via group actions, proves that RoPE and ALiBi are exact special cases, and introduces a path-integral additive variant GRAPE-AP that outperforms existing methods on downstream tasks.

## Background & Motivation

**Fragmentation of positional encodings**: Existing methods — including absolute encodings (sinusoidal/learned), relative encodings (RoPE), linear biases (ALiBi), and forgetting mechanisms (FoX) — are designed independently, lacking a unified theoretical framework.

**Limitations of RoPE**: RoPE relies on fixed coordinate planes and a log-uniform frequency spectrum, precluding cross-subspace coupling and content-dependent phase warping.

**Absolute encodings break translation equivariance**: Table-based relative encodings introduce window-dependent computational overhead.

**Lack of theoretical guarantees**: Desirable properties such as stability, monotonic distance penalty, and expressivity are scattered across disjoint methods, motivating a unified framework that integrates them.

**Long-context modeling demands**: Long-sequence models require a principled positional geometry design space.

## Method

### Overall Architecture

GRAPE is grounded in Lie group theory, unifying positional encodings as group actions $\mathbf{G}(n) = \exp(n\omega\mathbf{L})$, organized into two families:

- **Multiplicative GRAPE (GRAPE-M)**: Norm-preserving rotations in the special orthogonal group $\mathrm{SO}(d)$
- **Additive GRAPE (GRAPE-A)**: Unipotent actions in the general linear group $\mathrm{GL}$, producing linear biases

### Multiplicative GRAPE

**Core construction**: Rotations are constructed using rank-2 skew-symmetric generators $\mathbf{L} = \mathbf{ab}^\top - \mathbf{ba}^\top \in \mathfrak{so}(d)$:

$$\mathbf{G}(n) = \exp(n\omega\mathbf{L}) \in \mathrm{SO}(d)$$

**Key properties**:
- **Exact relative law**: $\mathbf{G}(n+m) = \mathbf{G}(n)\mathbf{G}(m)$, ensuring attention scores depend only on the offset $j-i$
- **Norm preservation**: $\mathbf{G}(n)^\top\mathbf{G}(n) = \mathbf{I}$
- **Rodrigues closed-form**: $\exp(\mathbf{L}) = \mathbf{I} + \frac{\sin s}{s}\mathbf{L} + \frac{1-\cos s}{s^2}\mathbf{L}^2$, with $O(d)$ complexity requiring no explicit matrix exponentiation

**Multi-subspace GRAPE-M**: $d/2$ rank-2 generators act on orthogonal 2D subspaces. RoPE is exactly recovered when subspaces correspond to standard coordinate pairs with a log-uniform frequency spectrum. Learnable orthogonal bases and non-commutative mixing further extend expressivity.

### Additive GRAPE

**Core construction**: Via homogeneous coordinate lifting into $\mathrm{GL}(d+k)$, a nilpotent generator $\mathbf{A}$ (satisfying $\mathbf{A}^2=\mathbf{0}$) yields a unipotent action:

$$\mathbf{G}_\mathrm{add}(n) = \exp(n\omega\mathbf{A}) = \mathbf{I} + n\omega\mathbf{A}$$

**Exact recovery of ALiBi**: Using a rank-1 nilpotent generator in $\mathrm{GL}(d+2)$, the logit becomes $\mathbf{q}_i^\top\mathbf{k}_j + (j-i)\beta_h$.

**Content-gated variant (GRAPE-A-QK)**: Employs softplus-gated, query/key-dependent slopes:

$$\text{logit} = \mathbf{q}_i^\top\mathbf{k}_j + (j-i)\omega[\text{softplus}(\mathbf{v}^\top\mathbf{q}_i/\sqrt{d}) + \text{softplus}(\mathbf{u}^\top\mathbf{k}_j/\sqrt{d})]$$

**Exact recovery of FoX**: Per-token forgetting scalars $f_t$ correspond to $\omega_t = \log f_t$, and the cumulative bias is consistent with the forgetting bias $D_{ij}$ of FoX.

### Path-Integral Additive GRAPE (GRAPE-AP)

Building on GRAPE-A, a path-integral bias is introduced with an edge potential at each step:

$$\psi_h(t,\ell) = \alpha_h \cdot g\left(\frac{1}{d}\langle\mathbf{p}_{t,h},\, \mathbf{R}_\ell\mathbf{p}_{\ell,h}\rangle\right) \leq 0$$

The path-integral bias is $b_h(t,j) = \sum_{\ell=j+1}^{t}\psi_h(t,\ell)$, which can be combined with multiplicative GRAPE and supports causal constraints and streaming inference.

## Experiments

### Experimental Setup

- Based on nanoGPT / Llama architecture with only the positional encoding replaced
- Dataset: FineWeb-Edu 100B (50B tokens used for training)
- Model scales: Medium (350M, 24 layers, 8 heads) / Large (770M, 36 layers, 10 heads)
- Context length 4096, batch size 480
- Baselines: RoPE, ALiBi, FoX

### Main Results (Medium 350M, 0-shot, 7-task average)

| Method | ARC-E | ARC-C | HellaSwag | PIQA | SciQ | **Avg.** |
|------|-------|-------|-----------|------|------|----------|
| RoPE | 56.36 | 30.38 | 44.65 | 68.77 | 74.40 | 51.73 |
| ALiBi | 58.21 | 29.78 | 45.38 | 70.08 | 78.50 | 52.87 |
| FoX | 58.38 | 30.89 | 45.80 | 69.37 | 78.40 | 52.96 |
| GRAPE-A-QK | 57.95 | **32.00** | 45.77 | 69.37 | 79.00 | 53.00 |
| **GRAPE-AP** | **59.26** | 31.31 | 45.42 | 68.17 | **79.70** | **53.25** |
| GRAPE-AP+KV-shift | 57.32 | 30.55 | **46.18** | 69.10 | 79.60 | **53.46** |

### Main Results (Large 770M, 0-shot, 7-task average)

| Method | ARC-E | ARC-C | HellaSwag | PIQA | SciQ | **Avg.** |
|------|-------|-------|-----------|------|------|----------|
| RoPE | 62.63 | 32.76 | 51.01 | 71.33 | 80.50 | 55.76 |
| ALiBi | 62.67 | 34.39 | 51.33 | 71.11 | 82.70 | 56.44 |
| FoX | 61.07 | 33.11 | 51.85 | 71.27 | 83.70 | 56.30 |
| **GRAPE-AP** | **63.89** | 34.22 | 51.52 | **71.98** | **84.40** | **56.91** |
| FoX+KV-shift | 63.55 | 33.96 | **52.72** | 71.71 | 83.20 | 57.09 |
| GRAPE-AP+KV-shift | 63.72 | 33.11 | 52.29 | 71.65 | 83.50 | 56.86 |

### Key Findings

1. **GRAPE-AP achieves best overall performance without KV-shift**: 350M Avg. 53.25 > FoX 52.96 > RoPE 51.73; 770M Avg. 56.91 > ALiBi 56.44.
2. **Training stability advantage**: RoPE exhibits instability (loss spikes) at 770M scale, while GRAPE maintains stable improvement.
3. **Multiplicative GRAPE-M matches RoPE**: This empirically validates the theoretical equivalence; GRAPE-M itself does not yield significant gains over RoPE.
4. **Additive variants are the primary source of improvement**: The GRAPE-A and GRAPE-AP families consistently outperform purely multiplicative methods.
5. **KV-shift and GRAPE-AP are complementary**: Adding KV-shift further improves the 350M model to 53.46.

## Highlights & Insights

- **Elegant theoretical unification**: The Lie group framework unifies the seemingly disparate RoPE, ALiBi, and FoX as special cases of a single mathematical object, with rigorous proofs provided.
- **Practical efficiency**: The Rodrigues closed-form formula yields $O(d)$ complexity on par with RoPE, with full compatibility for streaming inference and KV-cache.
- **Extensible design space**: The framework naturally gives rise to extensions including learnable orthogonal bases, content-gated slopes, and path-integral biases.
- **Rigorous mathematical exposition**: The group-theoretic perspective provides clear geometric intuition (rotation planes, unipotent translations) for positional encoding design.

## Limitations & Future Work

- **Limited experimental scale**: Validation is restricted to 350M/770M models; experiments at >1B scale are absent, and training covers only 50B tokens.
- **GRAPE-M does not significantly surpass RoPE**: The theoretical advantages of the multiplicative variant — learnable subspaces and non-commutative mixing — do not translate into clear empirical gains.
- **No long-context evaluation**: Training uses only 4096-token contexts; extrapolation to longer sequences is not evaluated, which is precisely the key differentiator between ALiBi and RoPE.
- **Computational overhead of GRAPE-AP underanalyzed**: The edge potential requires per-step inner product computations, and actual inference latency is not reported.
- **Limited downstream task coverage**: Evaluation is restricted to 0-shot language modeling benchmarks, without assessment of generation quality or fine-tuned performance.

## Related Work & Insights

- **RoPE** (Su et al., 2021): An exact special case of GRAPE-M (standard coordinate pairs + log-uniform spectrum).
- **ALiBi** (Press et al., 2021): An exact special case of GRAPE-A in $\mathrm{GL}(d+2)$.
- **Forgetting Transformer (FoX)** (Lin et al., 2025): Shown to be a path-dependent form of GRAPE-A.
- **PaTH Attention** (Yang et al., 2025): Analyzed in the paper as contractive and near-singular, potentially harmful for long-context modeling.
- **NoPE / no positional encoding**: Not discussed within the framework.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The group-theoretic unification is highly elegant; the exact recovery proofs for RoPE/ALiBi/FoX are a standout contribution.
- Experimental Thoroughness: ⭐⭐⭐ — Model scale is modest; long-context and large-model validation are lacking.
- Writing Quality: ⭐⭐⭐⭐ — Mathematical derivations are clear and rigorous, though the dense notation raises the barrier to entry.
- Value: ⭐⭐⭐⭐ — Theoretical contribution is significant, providing a unified principled framework for positional encoding design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Understanding and Improving Length Generalization in Hierarchical Sparse Attention Models](understanding_and_improving_length_generalization_in_hierarchical_sparse_attenti.md)
- [\[ICLR 2026\] When Does Divide and Conquer Work for Long Context LLM? A Noise Decomposition Framework](when_does_divide_and_conquer_work_for_long_context_llm_a_noise_decomposition_fra.md)
- [\[ICLR 2026\] DND: Boosting Large Language Models with Dynamic Nested Depth](dnd_boosting_large_language_models_with_dynamic_nested_depth.md)
- [\[ICLR 2026\] RACE Attention: A Strictly Linear-Time Attention for Long-Sequence Training](race_attention_a_strictly_linear-time_attention_for_long-sequence_training.md)
- [\[ICLR 2026\] Universe Routing: Why Self-Evolving Agents Need Epistemic Control](universe_routing_why_self-evolving_agents_need_epistemic_control.md)

</div>

<!-- RELATED:END -->

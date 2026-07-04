---
title: >-
  [Paper Note] On Expressive Power of Looped Transformers: Theoretical Analysis and Enhancement via Timestep Encoding
description: >-
  [ICML2025][LLM (Other)][Looped Transformer] This work establishes the first approximation rate theory of Looped Transformers regarding the number of loops and the modulus of continuity of target functions. It reveals the loop-architecture-specific approximation error sources (contextual continuity and token continuity) and proposes the Timestep-Modulated Looped Transformer (TMLT) to eliminate this limitation via timestep encoding, achieving consistent improvements across reas…
tags:
  - "ICML2025"
  - "LLM (Other)"
  - "Looped Transformer"
  - "Expressive Power"
  - "Approximation Rate"
  - "Weight Sharing"
  - "Timestep Encoding"
  - "Universal Approximation"
  - "Modulus of Continuity"
date: 2026-05-08
content_hash: a6322f0ec1f25e26
---

# On Expressive Power of Looped Transformers: Theoretical Analysis and Enhancement via Timestep Encoding

**Conference**: ICML2025  
**arXiv**: [2410.01405](https://arxiv.org/abs/2410.01405)  
**Code**: [kevin671/tmlt](https://github.com/kevin671/tmlt)  
**Area**: LLM/NLP  
**Keywords**: Looped Transformer, Expressive Power, Approximation Rate, Weight Sharing, Timestep Encoding, Universal Approximation, Modulus of Continuity

## TL;DR

This work establishes the first approximation rate theory of Looped Transformers regarding the number of loops and the modulus of continuity of target functions. It reveals the loop-architecture-specific approximation error sources (contextual continuity and token continuity) and proposes the Timestep-Modulated Looped Transformer (TMLT) to eliminate this limitation via timestep encoding, achieving consistent improvements across reasoning, in-context learning, and language modeling tasks.

## Background & Motivation

- **Looped Transformer** recursively feeds the output of a fixed-size Transformer layer back to its input, achieving parameter efficiency through weight sharing while the recursive structure empowers it to simulate iterative algorithms.
- The universal approximation properties of standard Transformers have been widely studied (Yun et al., 2020; Kajitsuka & Sato, 2024). However, due to weight-binding constraints, these conclusions **cannot be directly generalized** to Looped Transformers.
- While the approximation rate of weight-tied ReLU networks was only recently established (Zhang et al., 2023), the approximation rate of Looped Transformers remains unknown.
- **Core Problem**: Can Looped Transformers compute contextual mapping? Are they universal approximators? What determines their approximation rate?

## Method

### Theoretical Framework: Three Moduli of Continuity

The core contribution of this paper is the definition of three moduli of continuity, which jointly determine the approximation rate of Looped Transformers:

**1. Sequence Continuity**: Measures the impact of global perturbations in the input sequence on the output.

$$\omega_f(\delta) \coloneqq \sup\{\|f(\mathbf{X}) - f(\mathbf{X}')\|_p : \|\mathbf{X} - \mathbf{X}'\|_2 \leq \delta\}$$

**2. Contextual Continuity**: Fixes a specific token and measures the impact of perturbations in other tokens on this token's output (analogous to the semantic shift of the same word "write" in different sentences like "I write papers" vs "You write books").

$$\omega_f^{\text{cont}}(\delta) \coloneqq \sup_{n,\mathbf{X},\mathbf{X}'}\{\|f(\mathbf{X})_{:,n} - f(\mathbf{X}')_{:,n}\|_p : \|\mathbf{X}-\mathbf{X}'\|_2 \leq \delta,\ \mathbf{X}_{:,n}=\mathbf{X}'_{:,n}\}$$

**3. Token Continuity**: Keeps the context fixed and measures the impact of a single token's perturbation on its own output (analogous to comparing "I write papers" vs "I draft papers").

$$\omega_f^{\text{tok}}(\delta) \coloneqq \sup_{n,\mathbf{X},\mathbf{X}'}\{\|f(\mathbf{X})_{:,n} - f(\mathbf{X}')_{:,n}\|_p : \|\mathbf{X}_{:,n}-\mathbf{X}'_{:,n}\|_2 \leq \delta,\ \mathbf{X}_{:,q}=\mathbf{X}'_{:,q}\ (\forall q \neq n)\}$$

### Main Theorem (Theorem 3.6)

Given a permutation-equivariant continuous function $f$, when the number of loops $r > N$, there exists a single-layer Looped Transformer with only 2 attention heads and head size of 1, such that:

$$\|\mathcal{L}_2 \circ \text{TF}^{\circ r} \circ \mathcal{L}_1 - f\|_{L^p} \leq (Nd)^{1/p}(\omega_f^{\text{tok}}(\delta\sqrt{d}) + \omega_f^{\text{cont}}(\delta\sqrt{Nd})) + \omega_f(\delta\sqrt{Nd}) + \text{lower-order terms}$$

Where $\delta = ((r-N)/2)^{-1/((N+1)d+1)}$, and the parameter size is $O(d)$, which is **independent of the approximation accuracy and sequence length**.

### Three-step Proof

1. **Token-wise Quantization**: FFN maps each token $\mathbf{X}_{:,n} \in [0,1]^d$ to a token ID $z \in \{0,\ldots,\delta^{-d}-1\}$.
2. **Contextual Mapping**: Employs $N$ loops to compute the sequence ID via inner product $\mathbf{u}^\top \mathbf{z}$, ensuring different sequences have distinct IDs.
3. **Function Value Mapping**: Maps the contextual token ID to the target embedding step-by-step using $K-1$ loops.

### TMLT: Introducing Timestep Encoding

Theoretical analysis reveals a limitation of Looped Transformers: the approximation error of the weight-tied FFN is determined by the maximum discrepancy between adjacent contextual token embeddings (Lemma 4.1), preventing precise memorization of target values.

**Solution**: Introduce a timestep $t$-dependent scaling parameter for each loop:

$$\text{FF}(\mathbf{X}) \to \boldsymbol{\eta}(t) \odot \text{FF}(\mathbf{X})$$

Timestep encoding is generated via frequency embeddings + a two-layer MLP (with SiLU activation), which conditions the gain parameters of RMSNorm and the residual scaling coefficients:

$$\boldsymbol{\alpha}_1(t), \boldsymbol{\alpha}_2(t), \boldsymbol{\gamma}_1(t), \boldsymbol{\gamma}_2(t) = \mathbf{W}_5 \cdot \text{SiLU}(\text{TE}(t)) + \mathbf{b}_5$$

Theorem 4.2 proves that after incorporating timestep encoding, the model can **precisely memorize** the target values (with zero error), eliminating the dependence on extra moduli of continuity.

## Key Experimental Results

### Reasoning Tasks (Table 2)

| Task | TF (L=6) | Looped r=32 | TMLT r=32 |
|------|----------|-------------|-----------|
| Sudoku | 0.0 | 87.9 | **90.2** |
| Countdown | 53.8 | 88.1 | **90.5** |

| Task | TF (L=12) | Looped r=100 | TMLT r=100 |
|------|-----------|--------------|------------|
| LCS (100) | 39.8 | 98.2 | **98.6** |
| ED (60) | 41.4 | 47.7 | **88.3** |

- The improvement of TMLT on the ED(60) task is the most significant: 47.7 → 88.3 (+40.6), validating the theoretical predictions.

### In-Context Learning (Table 3)

| Model | MSE ↓ |
|------|-------|
| TF L=12 | 8.6e-03 |
| Looped r=12 | 1.4e-02 |
| TMLT r=12 | **1.7e-03** |

### Language Modeling WikiText-103 (Table 4)

| Model | Train Perplexity | Test Perplexity |
|------|-----------|-----------|
| TF L=12 | 15.9 | 20.5 |
| Looped r=24 | 17.1 | 20.6 |
| TMLT r=24 | **15.9** | **19.6** |

### Key Findings

- Increasing the loop count $r$ consistently improves performance, validating the theoretical approximation rate.
- Timestep encoding brings additional gains across all tasks, particularly on tasks with rapidly changing function values (high modulus of continuity).
- Looped TF can match or even exceed the performance of standard TF with significantly fewer parameters.

## Highlights & Insights

1. **Establishes the first approximation rate for Looped Transformers**, filling a theoretical gap, where the parameter size depends only on the input dimension $O(d)$ and is independent of the approximation accuracy and sequence length.
2. **Highly innovative definitions of three moduli of continuity**: sequence continuity, contextual continuity, and token continuity precisely characterize the different variation modes of sequence-to-sequence functions.
3. **Theory-driven practice**: The limitation of Looped TF (extra dependence on moduli of continuity) is naturally derived from the approximation rate analysis, leading to the targeted TMLT solution.
4. It is proved that even with a weight-tied self-attention mechanism using hardmax, Looped Transformers can still compute contextual mappings.
5. The design of TMLT draws inspiration from adaptive instance normalization (DiT) in diffusion models, demonstrating an elegant cross-domain transfer.

## Limitations & Future Work

- The theoretical analysis is restricted to **single-layer** Looped Transformers, leaving the approximation rate of multi-layer cases unexplored.
- All analyses are confined to **fixed-length** inputs, leaving the length generalization of variable-length sequences unaddressed.
- Timestep encoding introduces additional parameters (frequency embedding MLP + scaling parameter generator), which increases implementation complexity despite the minor overhead relative to total parameter size.
- Lack of characterization of optimal memory capacity—how many samples can the model precisely memorize?
- Training sample sizes for reasoning tasks are extremely large (in the millions), and data efficiency in practical applications remains to be verified.

## Related Work & Insights

- **Yun et al. (2020)**: Universal approximation theorems for standard Transformers, which this work generalizes to the Looped setting.
- **Zhang et al. (2023)**: Approximation rates for weight-tied ReLU networks, based on which this work tackles the additional challenge of contextual mapping.
- **Dehghani et al. (2019)**: Universal Transformer, which first introduced the recursive structure.
- **Saunshi et al. (2025)**: Proved that Looped TF possesses an inductive bias for reasoning tasks.
- **Peebles & Xie (2023)**: Adaptive instance normalization in DiT, which inspired the timestep conditioning design of TMLT.
- **Bae et al. (2025)**: Relaxed weight-tying (layer-wise LoRA), which acts as an alternative method to mitigate weight-tying constraints.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Establishes the approximation rate of Looped Transformers for the first time; the definitions of the three moduli of continuity are highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers three categories of tasks (reasoning/ICL/LM), but the scale is relatively small.
- Writing Quality: ⭐⭐⭐⭐⭐ — Integrates theory and practice naturally with a clear proof structure.
- Value: ⭐⭐⭐⭐ — Highly significant theoretically for understanding Looped/weight-sharing architectures, with promising practical potential for TMLT.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Theoretical Limitations of Ensembles in the Age of Overparameterization](theoretical_limitations_of_ensembles_in_the_age_of_overparameterization.md)
- [\[ACL 2025\] Interactive and Expressive Code-Augmented Planning with Large Language Models](../../ACL2025/llm_nlp/interactive_and_expressive_code-augmented_planning_with_large_language_models.md)
- [\[ACL 2025\] The Impact of Token Granularity on the Predictive Power of Language Model Surprisal](../../ACL2025/llm_nlp/token_granularity_impact.md)
- [\[ACL 2025\] Theory of Mind in Large Language Models: Assessment and Enhancement](../../ACL2025/llm_nlp/theory_of_mind_llm.md)
- [\[ICML 2026\] Differential Syntactic and Semantic Encoding in LLMs](../../ICML2026/llm_nlp/differential_syntactic_and_semantic_encoding_in_llms.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] PaTH Attention: Position Encoding via Accumulating Householder Transformations
description: >-
  [NeurIPS 2025][LLM Evaluation][position encoding] This paper proposes PaTH (Position encoding via accumulating Householder Transformations), a data-dependent multiplicative position encoding scheme that replaces RoPE's static rotation matrices with accumulated Householder transformations, achieving superior theoretical expressiveness and empirical language modeling performance over RoPE.
tags:
  - NeurIPS 2025
  - LLM Evaluation
  - position encoding
  - Householder transformation
  - attention mechanism
  - state tracking
  - RoPE
date: 2026-05-08
content_hash: 80a759c0a7b50500
---

# PaTH Attention: Position Encoding via Accumulating Householder Transformations

**Conference**: NeurIPS 2025
**arXiv**: [2505.16381](https://arxiv.org/abs/2505.16381)
**Code**: [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
**Area**: LLM Evaluation
**Keywords**: position encoding, Householder transformation, attention mechanism, state tracking, RoPE

## TL;DR

This paper proposes PaTH (Position encoding via accumulating Householder Transformations), a data-dependent multiplicative position encoding scheme that replaces RoPE's static rotation matrices with accumulated Householder transformations, achieving superior theoretical expressiveness and empirical language modeling performance over RoPE.

## Background & Motivation

The attention mechanism is inherently permutation-invariant, making position encoding critical for sequence modeling. RoPE has become the standard position encoding scheme in mainstream LLMs, encoding relative positions via rotation matrices $\mathbf{R}$ such that the attention logit takes the form $\mathbf{q}_i^\top \mathbf{R}^{i-j} \mathbf{k}_j$.

However, RoPE suffers from two fundamental limitations:

**Data-independence**: The rotation matrix $\mathbf{R}$ depends solely on the relative position $i-j$ and is entirely unaffected by input content, limiting expressiveness.

**Bounded computational complexity**: RoPE Transformers remain confined to the $\mathsf{TC}^0$ complexity class, rendering them incapable of solving simple synthetic tasks requiring sequential reasoning (e.g., flip-flop language modeling and state tracking tasks).

These limitations suggest that RoPE may be insufficient to fully model the sequential reasoning capabilities required in real-world settings, such as variable tracking in code or entity tracking over long contexts.

## Method

### Overall Architecture

PaTH generalizes multiplicative position encoding to a data-dependent form. The attention logit takes the form:

$$\mathbf{A}_{ij} \propto \exp\left(\mathbf{k}_j^\top \left(\prod_{s=j+1}^{i} \mathbf{H}_s\right) \mathbf{q}_i\right)$$

where $\mathbf{H}_s$ is a data-dependent Householder-like transformation matrix accumulated along the path from position $j$ to position $i$. RoPE is a special case in which $\mathbf{H}_s = \mathbf{R}$ is a static rotation matrix.

### Key Designs

**Householder-like transformation matrix**:

$$\mathbf{H}_t = \mathbf{I} - \beta_t \mathbf{w}_t \mathbf{w}_t^T$$

where:
- $\mathbf{w}_t \in \mathbb{R}^d$: obtained via a low-rank linear layer + short convolution (filter size 3) + L2 normalization
- $\beta_t = 2 \times \text{sigmoid}(\mathbf{u}^\top \mathbf{x}_t + b) \in (0, 2)$: a data-dependent scaling factor

The range $(0, 2)$ for $\beta_t$ allows the transformation matrix to have negative eigenvalues, which is shown to improve state tracking performance.

**Connection to linear RNNs**: PaTH can be viewed as a softmax counterpart of DeltaNet-style linear RNNs. Unrolling the RNN recurrence:

- RNN: $\mathbf{o}_t = \sum_{j=1}^{t} \mathbf{v}_j \left(\mathbf{k}_j^\top \prod_{s=j+1}^{t} \mathbf{H}_s \cdot \mathbf{q}_t\right)$
- PaTH: $\mathbf{o}_t = \frac{1}{Z_t} \sum_{j=1}^{t} \mathbf{v}_j \exp\left(\mathbf{k}_j^\top \prod_{s=j+1}^{t} \mathbf{H}_s \cdot \mathbf{q}_t\right)$

This relationship enables PaTH to combine the associative retrieval capacity of softmax attention with the state tracking capability of linear RNNs.

**Theoretical guarantee** (Theorem 2.1): A single-layer PaTH Transformer (2 heads, $\log n$ precision) can solve $\mathsf{NC}^1$-complete problems, extending the Transformer beyond the $\mathsf{TC}^0$ complexity class.

**PaTH-FoX extension**: PaTH is combined with the Forgetting Transformer (FoX):

$$\mathbf{A}_{ij} \propto \left(\prod_{s=j+1}^{i} f_s\right) \exp\left(\mathbf{k}_j^\top \left(\prod_{s=j+1}^{i} \mathbf{H}_s\right) \mathbf{q}_i\right)$$

analogous to how Gated DeltaNet successfully integrates DeltaNet with Mamba2.

### Loss & Training

The **UT transform** is employed to compactly represent the cumulative product of Householder matrices as:

$$\prod_{t=0}^{L-1} \mathbf{H}_t = \mathbf{I} - \mathbf{W}^\top \mathbf{T}^{-1} \mathbf{W}$$

where $\mathbf{T}^{-1} = (\mathbf{I} + \text{strictLower}(\mathbf{D}\mathbf{W}\mathbf{W}^\top))^{-1} \mathbf{D}$.

In full-matrix form, the attention matrix is expressed as:

$$\widetilde{\mathbf{A}} = \text{lower}(\mathbf{Q}\mathbf{K}^\top) - \text{lower}(\mathbf{Q}\mathbf{W}^\top) \mathbf{T}^{-1} \text{strictLower}(\mathbf{W}\mathbf{K}^\top)$$

A FlashAttention-style tiling algorithm is designed to decompose the global matrix inversion into local inversions, yielding a total complexity of $\mathcal{O}(L^2 d + Ld^2/B)$, which is comparable to standard attention when $B \approx d$.

At inference time, the historical key cache can be updated in-place via $\mathbf{k}_i^{(t)} \leftarrow (\mathbf{I} - \beta_t \mathbf{w}_t \mathbf{w}_t^\top) \mathbf{k}_i^{(t-1)}$, reducing the decoding phase to standard softmax attention decoding and maintaining compatibility with existing inference optimizations such as FlashDecoding and PagedAttention.

## Key Experimental Results

### Main Results

| Model | Wiki. ppl↓ | LMB. ppl↓ | LMB. acc↑ | PIQA↑ | Hella.↑ | Wino.↑ | ARC-e↑ | ARC-c↑ | Avg.↑ |
|-------|-----------|-----------|----------|-------|---------|--------|--------|--------|-------|
| RoPE | 19.01 | 19.77 | 40.4 | 70.2 | 50.3 | 54.9 | 67.2 | 33.3 | 52.7 |
| FoX | 18.33 | 18.28 | 41.7 | 70.8 | 50.9 | 57.1 | 65.7 | 32.6 | 53.1 |
| PaTH | **18.03** | **16.79** | **44.0** | 70.5 | **51.5** | 56.0 | **68.9** | **34.4** | **54.2** |
| PaTH-FoX | **17.35** | **16.23** | 44.1 | **70.8** | **52.2** | **57.1** | 67.3 | 33.9 | 54.2 |

760M parameter models trained on 50B tokens. PaTH surpasses RoPE on nearly all tasks, and PaTH-FoX achieves the lowest perplexity.

### Synthetic Task Experiments

| Model | FFLM ID Sparse | FFLM ID Dense | FFLM OOD |
|-------|---------------|--------------|----------|
| RoPE | 6.9% | 40.3% | 0.01% |
| SBA | 9.6% | 38.9% | 0% |
| FoX | 8.3% | 36.3% | 0% |
| **PaTH** | **0%** | **0.0001%** | **0%** |

PaTH achieves near-perfect performance on FFLM tasks (error rate approaching 0), whereas other methods exhibit error rates as high as 40%.

### Ablation Study

**Long-context benchmarks** (trained at length 4096, evaluated on longer sequences):

| Model | RULER 4K/8K/16K | BABILONG 4K/8K/16K | PhoneBook 2K/4K/8K |
|-------|----------------|--------------------|--------------------|
| RoPE | 35.7/1.3/0.0 | 13.8/0.0/0.0 | 15.6/0.0/0.0 |
| FoX | 41.6/29.5/4.9 | 20.2/8.2/4.4 | 38.5/17.7/— |
| PaTH | 44.6/34.8/18.7 | 24.6/16.8/11.6 | 20.8/0.0/— |
| PaTH-FoX | 42.3/34.0/22.6 | 25.6/19.2/10.0 | **93.8/66.6**/— |

**RoPE→PaTH model conversion**:

| Model | GSM8K | HumanEval | MBPP+ |
|-------|-------|-----------|-------|
| RoPE | 19.9 | 23.1 | 47.1 |
| FoX | 15.5 | 21.3 | 48.2 |
| PaTH | **20.1** | **25.6** | **51.3** |

### Key Findings

1. PaTH exhibits substantial advantages on state tracking tasks, which lie beyond the theoretical reach of RoPE.
2. PaTH-FoX generalizes to 64K tokens despite being trained only on sequences of length 4096, with particularly notable improvements in the code domain.
3. RoPE→PaTH conversion yields inconsistent results on fully trained models and is more effective when applied early in training.

## Highlights & Insights

1. **An elegant bridge from linear RNNs to attention**: State tracking capability from DeltaNet is injected into softmax attention.
2. **Clever application of the UT transform**: The compact representation of Householder matrices enables chunked parallel training.
3. **In-place key cache updates at inference time** eliminate additional storage requirements and preserve compatibility with existing inference infrastructure.
4. **Minimal parameter overhead**: Only a low-rank linear layer, short convolution, and sigmoid gating are added.

## Limitations & Future Work

1. Training throughput decreases by approximately 1.5–2×; further kernel optimization (e.g., via ThunderKittens) is needed.
2. Experiments are conducted primarily on 760M parameter models, lacking validation through from-scratch training at the 7B+ scale.
3. Distillation-based conversion is unstable for fully trained models; improved conversion strategies remain to be explored.
4. Combinations with other attention variants such as Sliding Window Attention have not yet been investigated.

## Related Work & Insights

- Complementary to FoX (which modifies logits additively); their combination as PaTH-FoX yields further improvements.
- Provides a new direction for the research thread of iterative key cache refinement.
- Code has been integrated into the flash-linear-attention library, lowering the barrier for community reproduction.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Reinterprets position encoding from a linear RNN perspective; Householder transformations balance expressiveness and computational efficiency.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers synthetic tasks, language modeling, long-context evaluation, and model conversion, though large-scale training validation is absent.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are rigorous; the narrative flows clearly from motivation through algorithm design to experiments.
- Value: ⭐⭐⭐⭐⭐ A strong candidate to replace RoPE; code is open-sourced and integrated into a mainstream library.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Generalization Error Analysis for Selective State-Space Models Through the Lens of Attention](generalization_error_analysis_for_selective_state-space_models_through_the_lens_.md)
- [\[ICLR 2026\] Spectral Attention Steering for Prompt Highlighting](../../ICLR2026/llm_evaluation/spectral_attention_steering_for_prompt_highlighting.md)
- [\[CVPR 2026\] Anchoring and Rescaling Attention for Semantically Coherent Inbetweening](../../CVPR2026/llm_evaluation/anchoring_and_rescaling_attention_for_semantically_coherent_inbetweening.md)
- [\[ICLR 2026\] Breaking the Correlation Plateau: On the Optimization and Capacity Limits of Attention-Based Regressors](../../ICLR2026/llm_evaluation/breaking_the_correlation_plateau_on_the_optimization_and_capacity_limits_of_atte.md)
- [\[NeurIPS 2025\] PARROT: A Benchmark for Evaluating LLMs in Cross-System SQL Translation](parrot_a_benchmark_for_evaluating_llms_in_cross-system_sql_translation.md)

<!-- RELATED:END -->

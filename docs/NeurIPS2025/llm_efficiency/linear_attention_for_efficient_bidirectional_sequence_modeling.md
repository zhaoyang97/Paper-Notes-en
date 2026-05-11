---
title: >-
  [Paper Note] Linear Attention for Efficient Bidirectional Sequence Modeling
description: >-
  [NeurIPS 2025][LLM Efficiency][linear attention] This paper proposes Lion, a framework that, for the first time, systematically extends linear Transformers to bidirectional sequence modeling. It unifies three equivalent…
tags:
  - "NeurIPS 2025"
  - "LLM Efficiency"
  - "linear attention"
  - "bidirectional modeling"
  - "state space model"
  - "efficient transformer"
  - "RNN"
date: 2026-05-08
content_hash: b790510fcda77843
---

# Linear Attention for Efficient Bidirectional Sequence Modeling

**Conference**: NeurIPS 2025
**arXiv**: [2502.16249](https://arxiv.org/abs/2502.16249)
**Code**: [GitHub](https://github.com/LIONS-EPFL/Lion) (LION Code)
**Area**: LLM Efficiency
**Keywords**: linear attention, bidirectional modeling, state space model, efficient transformer, RNN

## TL;DR

This paper proposes Lion, a framework that, for the first time, systematically extends linear Transformers to bidirectional sequence modeling. It unifies three equivalent representations—full linear attention, bidirectional RNN, and chunkwise parallel—and achieves training speeds up to 10× faster than SSM-based approaches while delivering performance comparable to softmax Transformers on image classification and MLM tasks.

## Background & Motivation

**Background**: Linear Transformers (e.g., RetNet, Mamba-2, GLA) have emerged as efficient alternatives to softmax Transformers for causal sequence modeling, supporting parallel training via matrix multiplication and RNN-style inference. However, their application to bidirectional tasks (e.g., BERT, ViT) has received almost no systematic investigation.

**Limitations of Prior Work**:
   - Existing bidirectional SSMs (e.g., Vim, Hydra) are primarily Mamba-based and naively execute causal scans twice—once forward and once backward.
   - This "dual-scan" approach fails to exploit a natural prior of bidirectional modeling: the entire sequence is available at both training and inference time.
   - As a result, training speed lags far behind softmax Transformers (Vim is 14.95× slower than DeiT).

**Key Challenge**: The efficiency advantage of SSMs in causal tasks (RNN inference) is largely lost in bidirectional settings due to the need for chunking, yet directly computing the full attention matrix reintroduces $\mathcal{O}(L^2)$ complexity.

**Goal**: To provide a systematic framework for extending linear Transformers to bidirectional settings, simultaneously optimizing training speed, inference efficiency, and model performance.

**Key Insight**: Building on the mask structure of causal linear Transformers, the causal mask $\mathbf{M}^C$ (lower triangular) is generalized to a bidirectional mask $\mathbf{M}$ (full matrix), where $\mathbf{M}_{ij}$ equals the product of all decay factors between positions $i$ and $j$.

**Core Idea**: The bidirectional mask is defined as the lower-triangular forward mask plus the upper-triangular backward mask minus the identity matrix (to avoid double-counting the diagonal), thereby converting any causal linear Transformer into its bidirectional counterpart.

## Method

### Overall Architecture

Lion provides three theoretically equivalent representations:
1. **Full Linear Attention**: $\mathbf{Y} = \text{Scale}(\mathbf{Q}\mathbf{K}^\top \odot \mathbf{M})\mathbf{V}$ (fastest training)
2. **Bidirectional RNN**: forward + backward RNN (most memory-efficient inference)
3. **Chunkwise Parallel**: block-wise processing (balanced speed and memory)

### Key Designs

**1. Bidirectional Mask Construction**

$$\mathbf{M} = \mathbf{M}^F + \mathbf{M}^B - \mathbf{I}$$

- $\mathbf{M}^F$: lower triangular (including diagonal), with elements $\mathbf{M}_{ij}^F = \prod_{k=j+1}^i \lambda_k$
- $\mathbf{M}^B$: upper triangular (including diagonal), with elements $\mathbf{M}_{ij}^B = \prod_{k=i+1}^j \lambda_k$
- Subtracting $\mathbf{I}$ prevents double-counting of the diagonal

Three mask types:
- **Selective mask** (input-dependent $\lambda_i$): $\mathbf{M}^F = \text{Tril}(\mathbf{L}^F / \mathbf{L}^{F\top})$
- **Fixed learnable mask**: $\mathbf{M}_{ij} = \lambda^{|i-j|}$ (KMS matrix)
- **All-ones mask** (no decay): $\mathbf{M}_{ij} = 1$

**2. Correct Derivation of Bidirectional RNN (Theorem 3.1)**

Naively summing the two directional RNNs produces imbalanced attention, as the diagonal is counted twice. The correct formulation is:

$$\mathbf{y}_i = \frac{\mathbf{y}_i^F + \mathbf{y}_i^B}{c_i^F + c_i^B}$$

where:
$$\mathbf{y}_i^{F/B} = \mathbf{q}_i^\top (\mathbf{S}_i^{F/B} - \frac{1}{2}\mathbf{k}_i \mathbf{v}_i^\top)$$
$$\mathbf{S}_i^{F/B} = \Lambda_i \mathbf{S}_{i-1}^{F/B} + \mathbf{k}_i \mathbf{v}_i^\top$$

The key correction is subtracting $\frac{1}{2}\mathbf{k}_i \mathbf{v}_i^\top$ to resolve double-counting along the diagonal.

**3. Chunkwise Parallel Form (Theorem 3.2)**

The sequence is divided into $N = L/C$ chunks of size $C$:
- **Intra-chunk**: standard attention computation
- **Inter-chunk**: information propagated via hidden states $\mathbf{S}_{[ij]}$ and $\mathbf{C}_{[ij]}$
- In the bidirectional setting, only inter-chunk computation is required (as the full sequence is available)

### Three Instantiated Models

| Model | Decay Type | Based On |
|-------|-----------|----------|
| **Lion-lit** | $\lambda_i = 1$ (no decay) | Vanilla Linear Transformer |
| **Lion-d** | $\lambda = \sigma(a)$ (fixed learnable) | RetNet |
| **Lion-s** | $\lambda_i = \sigma(\mathbf{W}\mathbf{x}_i + b)$ (selective) | Gated RFA / Mamba-2 |

### Loss & Training

- Training uses the Full Linear Attention form (fastest), with $\mathcal{O}(L^2 d)$ complexity but $\mathcal{O}(1)$ sequential steps.
- Inference switches to the RNN form (most memory-efficient) or the Chunkwise form (balanced).
- Feature map: $\phi(\mathbf{x}) = \frac{\text{SiLU}(\mathbf{x}) + 0.5}{\|\text{SiLU}(\mathbf{x}) + 0.5\|}$
- The attention layers in DeiT/BERT are directly replaced; all other configurations remain unchanged.

## Key Experimental Results

### ImageNet Image Classification (Small Scale, 22M params)

| Model | Top-1 Acc (%) | Training Time (relative to DeiT) |
|-------|--------------|----------------------------------|
| ViT | 72.2 | ×1 |
| DeiT | 79.8 | ×1 |
| Hydra | 78.6 | ×2.50 |
| Vim | 80.3 | **×14.95** |
| Lion-lit | 72.4 | **×0.74** |
| Lion-d | 73.5 | ×1.49 |
| **Lion-d♮** | **79.9** | ×1.66 |
| Lion-s | 74.0 | ×2.03 |
| Lion-s♮ | 79.6 | ×2.72 |

Lion-d♮ surpasses DeiT in accuracy (79.9 vs. 79.8) while training **9× faster** than Vim.

### C4 MLM + GLUE (Large Scale, 334M params)

| Model | MLM Acc | GLUE | Training Time |
|-------|---------|------|---------------|
| BERT | 69.88 | **82.95** | ×1 |
| Hydra | **71.18** | 81.77 | ×3.13 |
| Lion-lit | 67.11 | 80.76 | **×0.95** |
| Lion-d | 68.64 | 81.34 | ×1.10 |
| Lion-s | 69.16 | 81.58 | ×1.32 |

Lion variants train **2–3× faster** than Hydra while achieving performance close to BERT.

### Long Range Arena (LRA)

| Model | PathX | Avg |
|-------|-------|-----|
| Lion-lit | 50.41 | — |
| Lion-d (w/ HIPPO) | 97.28 | 85.63 |
| Lion-s (w/ HIPPO) | **97.99** | **86.07** |

HIPPO initialization proves critical for solving long-range tasks.

### Key Findings

- Lion-lit achieves the fastest training speed (×0.74 relative to DeiT), but its lack of decay weakens long-range modeling.
- Lion-d♮ offers the best accuracy–speed trade-off, surpassing DeiT while training 9× faster than Vim.
- RNN inference memory scales linearly with resolution, compared to quadratic scaling for softmax attention.
- Chunk sizes of 8–16 provide the optimal speed–memory balance.
- The multi-scan strategy (♮) contributes significantly in vision tasks (+6% accuracy) at the cost of increased training time.

## Highlights & Insights

- **First unified framework for bidirectional linear Transformers**: covers 10+ existing models (RetNet, Mamba-2, GLA, HGRN-2, xLSTM, DeltaNet, etc.).
- **Elegant unification of three equivalent representations**: full attention for training, RNN for inference—each exploited for its respective strength.
- **Diagonal double-counting correction in bidirectional RNN**: the seemingly simple $-\frac{1}{2}\mathbf{k}_i\mathbf{v}_i^\top$ correction is a key technical contribution.
- **Pure PyTorch implementation outperforms CUDA-optimized SSMs**: demonstrating the hardware-friendliness of the full attention form.

## Limitations & Future Work

- The full attention form retains $\mathcal{O}(L^2)$ training complexity, making it less favorable than chunked SSMs for very long sequences.
- The non-selective masks in Lion-lit and Lion-d limit expressive capacity.
- Vision tasks rely on the multi-scan strategy, which adds engineering complexity.
- A small performance gap with BERT remains on MLM tasks (−0.72 MLM Acc at large scale).
- Evaluation on additional downstream tasks (e.g., question answering, summarization, biological sequences) is lacking.

## Related Work & Insights

- Vim pioneered bidirectional SSMs for vision but suffers from extremely low training efficiency; Lion addresses this from the training efficiency perspective.
- Hydra's dual-SSD design represents the strongest prior bidirectional SSM; Lion substantially outperforms it in training speed.
- Lion is complementary to Flash Attention: Flash Attention optimizes the I/O efficiency of softmax attention, whereas Lion fundamentally reduces complexity via linear attention.
- Key insight: the prior that the full sequence is available in bidirectional tasks is severely underutilized—it enables direct computation of the full attention matrix without chunking.

## Rating

⭐⭐⭐⭐⭐ (5/5)

A foundational contribution: the paper is the first to unify three representations of bidirectional linear Transformers, with elegant theory (equivalence proofs), high practical value (10× training speedup), and broad coverage (bidirectional extensions of 10+ models).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ZeroS: Zero-Sum Linear Attention for Efficient Transformers](zeros_zero-sum_linear_attention_for_efficient_transformers.md)
- [\[ACL 2026\] Native Hybrid Attention for Efficient Sequence Modeling](../../ACL2026/llm_efficiency/native_hybrid_attention_for_efficient_sequence_modeling.md)
- [\[NeurIPS 2025\] Tiled Flash Linear Attention: More Efficient Linear RNN and xLSTM Kernels](tiled_flash_linear_attention_more_efficient_linear_rnn_and_xlstm_kernels.md)
- [\[ICLR 2026\] RACE Attention: A Strictly Linear-Time Attention for Long-Sequence Training](../../ICLR2026/llm_efficiency/race_attention_a_strictly_linear-time_attention_for_long-sequence_training.md)
- [\[NeurIPS 2025\] Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs](long-context_modeling_with_dynamic_hierarchical_sparse_attention_for_on-device_l.md)

</div>

<!-- RELATED:END -->

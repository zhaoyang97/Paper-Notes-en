---
title: >-
  [Paper Note] Linear Attention for Efficient Bidirectional Sequence Modeling
description: >-
  [NeurIPS 2025][Model Compression][Linear Attention] This paper proposes Lion, a framework that, for the first time, systematically extends linear Transformers to bidirectional sequence modeling. It unifies three equivalent representations—full linear attention, bidirectional RNN, and chunkwise parallel—achieving training speeds up to 10× faster than SSMs while matching softmax Transformer performance.
tags:
  - NeurIPS 2025
  - Model Compression
  - Linear Attention
  - Bidirectional Sequence Modeling
  - State Space Models
  - Efficient Inference
  - Bidirectional RNN
date: 2026-05-08
content_hash: 01c0ead8b3f470fa
---

# Linear Attention for Efficient Bidirectional Sequence Modeling

**Conference**: NeurIPS 2025
**arXiv**: [2502.16249](https://arxiv.org/abs/2502.16249)
**Code**: [GitHub](https://github.com/LIONS-EPFL/Lion)
**Area**: Model Compression / Efficient Sequence Modeling
**Keywords**: Linear Attention, Bidirectional Sequence Modeling, State Space Models, Efficient Inference, Bidirectional RNN

## TL;DR

This paper proposes Lion, a framework that, for the first time, systematically extends linear Transformers to bidirectional sequence modeling. It unifies three equivalent representations—full linear attention, bidirectional RNN, and chunkwise parallel—achieving training speeds up to 10× faster than SSMs while matching softmax Transformer performance.

## Background & Motivation

1. **Background**: Linear Transformers and state space models (SSMs) have emerged as efficient alternatives to softmax Transformers for causal sequence modeling, enabling parallel training via matrix multiplication and efficient inference in RNN form.

2. **Limitations of Prior Work**: Despite their success in causal tasks, linear Transformers still lack a unified framework for bidirectional sequence modeling. Existing bidirectional SSMs (e.g., Bi-Mamba, Vim) mostly apply causal formulations independently in the forward and backward directions (e.g., dual-scan), failing to exploit the natural prior of bidirectional modeling—that the entire sequence is available at both training and inference time.

3. **Key Challenge**: Bidirectional SSMs rely on chunked training for numerical stability (to avoid overflow/underflow from cumulative products of decay factors), making training significantly slower than softmax Transformers. Meanwhile, naively summing the outputs of two linear Transformers leads to "double counting" and attention imbalance.

4. **Goal**: To construct a general framework that enables a broad class of linear Transformers to be efficiently applied to bidirectional sequence modeling.

5. **Key Insight**: Starting from the decay mask of causal linear attention, a symmetric bidirectional mask $\mathbf{M}_{ij}$ is defined as the product of all decay factors between positions $i$ and $j$, from which three equivalent forms—full attention, RNN, and chunkwise parallel—are naturally derived.

6. **Core Idea**: The decay mask of causal linear Transformers can be naturally generalized into a symmetric bidirectional mask. Through lower/upper triangular decomposition, an equivalent bidirectional RNN is obtained that can be trained stably without chunking.

## Method

### Overall Architecture

The Lion framework provides three theoretically equivalent representations for bidirectional linear Transformers:
- **Full Linear Attention**: Maximum training speed; directly computes $\mathbf{Y} = \text{scale}(\mathbf{Q}\mathbf{K}^\top \odot \mathbf{M})\mathbf{V}$
- **Bidirectional RNN**: Highest inference efficiency; runs one forward and one backward RNN pass and merges the results
- **Chunkwise Parallel**: Balances speed and memory

### Key Designs

**1. Bidirectional Mask Construction**

- **Function**: Generalizes the causal mask to the bidirectional setting, encoding relative distance information between positions.
- **Mechanism**: $\mathbf{M}_{ij}$ is defined as the product of all decay factors $\lambda_k$ between positions $i$ and $j$. For selective decay: $\mathbf{M}_{ij} = \prod_{k=\min(i,j)+1}^{\max(i,j)} \lambda_k$; for fixed decay: $\mathbf{M}_{ij} = \lambda^{|i-j|}$; without decay: $\mathbf{M}_{ij} = 1$. The mask is decomposed as $\mathbf{M} = \mathbf{M}^F + \mathbf{M}^B - \mathbf{I}$, where $\mathbf{M}^F$ is lower triangular and $\mathbf{M}^B$ is upper triangular.
- **Design Motivation**: In the causal setting, the mask $\mathbf{M}^C_{ij} = \lambda_{j+1}\lambda_{j+2}\cdots\lambda_i$ encodes relative positional information; the bidirectional scenario naturally extends this to a symmetric form.

**2. Balanced Bidirectional RNN Merging**

- **Function**: Avoids double counting and attention imbalance caused by naively summing forward and backward RNN outputs.
- **Mechanism**: The attention matrix $\mathbf{A} = \mathbf{Q}\mathbf{K}^\top$ is decomposed into $\mathbf{A}^F$ (lower triangular, diagonal halved) and $\mathbf{A}^B$ (upper triangular, diagonal halved), with the mask and scaling factors decomposed accordingly. The final output is $\mathbf{Y} = (\mathbf{C}^F + \mathbf{C}^B)^{-1}(\mathbf{Y}^F + \mathbf{Y}^B)$, where $\mathbf{Y}^F$ and $\mathbf{Y}^B$ are computed by the forward and backward RNNs respectively. The backward pass is implemented by reversing the sequence and reusing the forward RNN.
- **Design Motivation**: Naive summation yields $\mathbf{Y} = ((\mathbf{I} + \mathbf{1}) \odot \mathbf{QK}^\top)\mathbf{V}$, where diagonal elements are counted twice, leading to imbalance.

**3. Three Lion Variants**

- **Function**: Covers different decay types.
- **Mechanism**: Lion-lit (no decay, $\lambda_i = 1$; bidirectional Vanilla Linear Transformer), Lion-d (fixed learnable decay $\lambda = \sigma(a)$; bidirectional RetNet), and Lion-s (selective decay $\lambda_i = \sigma(\mathbf{W}\mathbf{x}_i + b)$; bidirectional GRFA/Mamba2-style).
- **Design Motivation**: Table 1 demonstrates that over a dozen causal linear Transformers can be mapped to bidirectional forms via Lion; the three variants cover representative cases of scalar/diagonal decay.

### Loss & Training

- Full Linear Attention is used during training to maximize speed (comparable to softmax Transformers).
- At inference, the RNN form (highest memory efficiency), full attention (fastest speed), or chunkwise form (balanced) can be selected.
- A shifted normalized SiLU activation is used: $\phi(\mathbf{x}) = \frac{\text{SiLU}(\mathbf{x}) + 0.5}{\|\text{SiLU}(\mathbf{x}) + 0.5\|}$
- Lion is applied by directly replacing attention layers in DeiT / BERT without modifying other hyperparameters.

## Key Experimental Results

### Main Results

**ImageNet-1K Image Classification (Small Scale)**

| Model | Params | Top-1 Acc (%) | Training Time Multiplier ↓ |
|-------|--------|--------------|--------------------------|
| DeiT | 22M | 79.8 | ×1 |
| Hydra | 22M | 78.6 | ×2.50 |
| Vim | 26M | 80.3 | ×14.95 |
| **Lion-s♮** | 22M | **80.5** | **×1.00** |
| Lion-d | 22M | 79.8 | ×0.97 |
| Lion-lit | 22M | 78.9 | ×0.76 |

**ImageNet-1K Image Classification (Base Scale)**

| Model | Params | Top-1 Acc (%) | Training Time Multiplier ↓ |
|-------|--------|--------------|--------------------------|
| DeiT | 86M | 81.8 | ×1 |
| Hydra | 91M | 81.0 | ×2.51 |
| Vim | 98M | 81.9 | ×14.63 |
| **Lion-s♮** | 86M | **82.0** | **×1.01** |

### Ablation Study

| Component | Top-1 Acc (Small) |
|-----------|------------------|
| No decay (Lion-lit) | 78.9 |
| Fixed decay (Lion-d) | 79.8 |
| Selective decay (Lion-s) | 79.6 |
| Lion-s + multi-scan (Lion-s♮) | 80.5 |
| Naive forward+backward sum (imbalanced) | Significant degradation |

### Key Findings

- Lion matches DeiT in training speed while being approximately **15×** faster than Vim and **2.5×** faster than Hydra.
- At Base scale, Lion matches or surpasses the softmax Transformer (DeiT 81.8% vs. Lion-s♮ 82.0%).
- RNN-form inference has memory complexity $O(d^2)$, independent of sequence length.
- On the MLM task (C4 dataset), Lion-s achieves performance comparable to BERT.

## Highlights & Insights

- **Strong unification**: A single framework covers the bidirectional extension of over a dozen linear Transformers, including LinAtt, RetNet, Mamba-2, GLA, HGRN-2, xLSTM, and DeltaNet.
- The paper demonstrates that Full Linear Attention can be trained numerically stably without chunking in the bidirectional setting, since all decay factors are known and can be computed efficiently via cumsum in log space.
- Training and inference are decoupled: full attention is used during training for speed, while the RNN form is used at inference for low memory consumption.

## Limitations & Future Work

- The current work primarily focuses on scalar/diagonal decay ($TC^0$ class); bidirectional extensions for non-diagonal decay (e.g., DeltaNet) are only discussed in the appendix.
- Vision tasks require a multi-scan strategy (Lion-s♮) to compensate for the absence of explicit positional encodings, increasing implementation complexity.
- Evaluation on the LRA long-range dependency benchmark is limited.

## Related Work & Insights

- Causal linear Transformers such as RetNet, Mamba-2, and GLA can all be extended to bidirectional models via Lion.
- Hydra and Vim, as representative existing bidirectional SSMs, employ a dual-scan approach that Lion demonstrates to be suboptimal.
- Insight: The core advantage of bidirectional modeling lies in the availability of the entire sequence; this prior should be fully exploited rather than simply replicating the causal formulation.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First systematic bidirectional extension framework for linear Transformers, with rigorous theoretical derivations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers two major bidirectional tasks (image classification and MLM), but lacks evaluation on broader downstream applications.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear structure, thorough mathematical derivations; the unified mapping table (Table 1) is particularly valuable.
- **Value**: ⭐⭐⭐⭐⭐ — Provides both a theoretical foundation and a practical tool for efficient bidirectional modeling; achieving training speed on par with softmax Transformers is a significant breakthrough.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] RAT: Bridging RNN Efficiency and Attention Accuracy via Chunk-based Sequence Modeling](rat_bridging_rnn_efficiency_and_attention_accuracy_via_chunk-based_sequence_mode.md)
- [\[NeurIPS 2025\] A*-Thought: Efficient Reasoning via Bidirectional Compression for Low-Resource Settings](a-thought_efficient_reasoning_via_bidirectional_compression_for_low-resource_set.md)
- [\[NeurIPS 2025\] Recurrent Attention-based Token Selection for Efficient Streaming Video-LLMs](recurrent_attention-based_token_selection_for_efficient_streaming_video-llms.md)
- [\[ICCV 2025\] FastVAR: Linear Visual Autoregressive Modeling via Cached Token Pruning](../../ICCV2025/model_compression/fastvar_linear_visual_autoregressive_modeling_via_cached_token_pruning.md)
- [\[NeurIPS 2025\] The Structure of Relation Decoding Linear Operators in Large Language Models](the_structure_of_relation_decoding_linear_operators_in_large_language_models.md)

<!-- RELATED:END -->

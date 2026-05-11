---
title: >-
  [Paper Note] QKV Projections Require a Fraction of Their Memory
description: >-
  [ICLR 2026][Model Compression][Training memory compression] This paper proposes PAMM (Point-Approximate Matrix Multiplication), an activation compression technique that approximates QKV projection layer activations by ra…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "Training memory compression"
  - "attention mechanism"
  - "matrix multiplication approximation"
  - "activation compression"
  - "LLM training"
date: 2026-05-08
content_hash: 9a00a4ae49e61ee2
---

# QKV Projections Require a Fraction of Their Memory

**Conference**: ICLR 2026
**arXiv**: [2506.02939](https://arxiv.org/abs/2506.02939)
**Code**: None
**Area**: Model Compression
**Keywords**: Training memory compression, attention mechanism, matrix multiplication approximation, activation compression, LLM training

## TL;DR

This paper proposes PAMM (Point-Approximate Matrix Multiplication), an activation compression technique that approximates QKV projection layer activations by randomly selecting a small number of representative tokens, achieving up to 512× compression without degrading model performance.

## Background & Motivation

During LLM training, QKV projections in the attention layer consume substantial memory: the input $X$ must be retained during the forward pass for use in backpropagation (to compute $\nabla W = X^\top \cdot \nabla Z$). This component can account for up to 20% of the peak GPU memory in the attention block.

Limitations of existing memory optimization methods:
- **Efficient attention** (FlashAttention, etc.): optimizes scaled dot-product attention itself, leaving linear projections untouched
- **Low-rank methods** (CompAct, etc.): compress along the hidden dimension, while redundancy along the sequence dimension is far greater
- **Optimizer state compression**: does not scale with batch size or sequence length

Core insight: **The sequence dimension contains substantial redundancy**. The number of tokens in a training batch $b = BL$ (e.g., 16384) greatly exceeds the hidden dimension $n$ (e.g., 2048); since $\text{rank}(X) \leq n$, theoretically only $n$ basis vectors are needed to represent $X$, yielding a potential compression ratio of 8×.

## Method

### Overall Architecture

PAMM operates in two stages: (1) during the forward pass, $X$ is compressed into a small set of generating points along with auxiliary information; (2) during the backward pass, the compressed representation is used to approximate the gradient $\nabla W$.

### Key Designs

1. **Activation Compression (Compression Stage)**:

    - Randomly sample $k = r \cdot b$ rows from $X \in \mathbb{R}^{b \times n}$ as generating points $C \in \mathbb{R}^{k \times n}$
    - For each point $A_i$, select the best generating point: $f(i) = \arg\max_j |\text{csim}(A_i, C_j)|$ (Lemma 1)
    - Compute the scaling coefficient: $\tilde{A}_i = \alpha(i, f(i)) \cdot C_{f(i)}$, where $\alpha = \frac{\langle A_i, C_j \rangle}{\|C_j\|_2^2}$
    - Neighborhood condition: $\|A_i - \tilde{A}_i\|_2 \leq \varepsilon \|A_i\|_2$; points not satisfying this are discarded

2. **Approximate Matrix Multiplication**:

    - Rather than reconstructing the full $\tilde{A}$, aggregate first: $\tilde{B}_j = \sum_{i:f(i)=j} \alpha_i B_i$
    - Compute $\tilde{O} = C^\top \tilde{B}$, reducing the dimension from $b \times n$ to $k \times n$
    - Introduce a normalization factor $\beta = \frac{b}{b-\eta}$ to ensure the unbiased estimate $\mathbb{E}[\tilde{O}] = O$

3. **Theoretical Guarantees**:

    - **Lemma 2** (sufficient condition for $k$): $k > \frac{b}{n_{\min}} \ln(\frac{b}{\delta})$, requiring only a logarithmic number of generating points
    - Approximation error upper bound: $\|O - \tilde{O}\|_F^2 \leq \|B\|_2^2 (\varepsilon^2 \|A_\mathcal{I}\|_F^2 + \|A_{\bar{\mathcal{I}}}\|_F^2)$
    - In practice, $\varepsilon \to \infty$ (i.e., no neighborhood constraint) yields the best results

### Loss & Training

- PAMM modifies only the backward pass of QKV projections; forward passes and gradients of other layers are unaffected
- Fully compatible with FlashAttention, gradient checkpointing, and LoRA
- Compression ratio $r$ as low as $1/512$ is used in experiments
- In fine-tuning scenarios, $k=1$ (a single generating point) suffices

## Key Experimental Results

### Pre-training Results (LLaMA on C4)

| Model | PAMM r | Val PPL | QKV Memory (MB) | Memory Reduction |
|------|--------|---------|-------------|---------|
| LLaMA-60M | No PAMM | 31.8 | 432 | - |
| LLaMA-60M | 1/512 | 31.6 | 0.85 | **>99%** |
| LLaMA-350M | No PAMM | 18.7 | 1,296 | - |
| LLaMA-350M | 1/512 | 18.5 | 2.53 | **>99%** |
| LLaMA-1B | No PAMM | 15.1 | 2,592 | - |
| LLaMA-1B | 1/512 | 15.0 | 5.06 | **>99%** |

### Fine-tuning Results (RoBERTa-base on GLUE)

| Method | QKV Memory (MB) | GLUE Avg. | Memory Reduction |
|------|-------------|----------|---------|
| Full Fine-Tuning | 288 | 86.28 | - |
| PAMM r=1/128 | 6.75 | 86.11 | **97.7%** |
| PAMM r=1/256 | 3.37 | 86.18 | **98.8%** |

### Throughput Analysis (LLaMA-1B)

| Stage | Baseline (tok/s) | PAMM (tok/s) | Throughput Reduction |
|------|-----------|------------|----------|
| Forward | 247.6K | 235.4K | 4.92% |
| Backward | 141.9K | 138.3K | 2.53% |
| **Total** | 88.4K | 85.2K | **3.61%** |

### Key Findings

- At 512× compression, PPL does not increase but marginally improves (more pronounced in larger models), suggesting that redundant tokens may negatively affect training
- As model size increases, throughput overhead drops from 19.7% (60M) to 2.1% (7B), making PAMM increasingly practical at scale
- PAMM exhibits stable performance across all batch size and sequence length configurations
- Compared to CompAct (which compresses along the hidden dimension), PAMM achieves significantly better performance at high compression ratios

## Highlights & Insights

- Deep insight: redundancy along the sequence dimension far exceeds that along the hidden dimension, which is the fundamental reason high compression ratios are achievable
- Remarkably simple and effective: randomly selected generating points suffice, with no need for complex clustering
- Theoretically rigorous: Lemma 1/2 provide principled guidance for algorithm design
- Fully orthogonal to FlashAttention and compatible for joint use
- Surprising finding: PPL slightly improves at high compression ratios, hinting at a regularization effect

## Limitations & Future Work

- Applied only to QKV projections; activation compression for FFN layers remains unexplored
- The optimal setting for the neighborhood parameter $\varepsilon$ is $\infty$ (i.e., unused), which lacks sufficient theoretical explanation
- The additional computation (cosine similarity matrix + argmax) incurs relatively high overhead for smaller models
- Validation under distributed training (multi-node) settings has not been conducted

## Related Work & Insights

- Key distinction from CompAct: PAMM compresses along the sequence dimension (where redundancy is greater), whereas CompAct compresses along the hidden dimension
- Relationship to gradient checkpointing: complementary — gradient checkpointing reduces the number of stored layers, while PAMM reduces the memory footprint per layer
- Implication: training memory optimization should not focus exclusively on optimizer states and attention mechanisms; activation memory deserves equal attention

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Identifies a novel direction exploiting sequence-dimension redundancy; method is elegantly simple and highly effective
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage of pre-training, fine-tuning, throughput, and ablation studies
- Writing Quality: ⭐⭐⭐⭐⭐ Theory and experiments are well integrated with clear illustrations
- Value: ⭐⭐⭐⭐⭐ A practically deployable memory optimization tool for LLM training

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LLMs Encode Their Failures: Predicting Success from Pre-Generation Activations](llms_encode_their_failures_predicting_success_from_pre-generation_activations.md)
- [\[ICLR 2026\] LightMem: Lightweight and Efficient Memory-Augmented Generation](lightmem_lightweight_and_efficient_memory-augmented_generation.md)
- [\[ICCV 2025\] A Good Teacher Adapts Their Knowledge for Distillation](../../ICCV2025/model_compression/a_good_teacher_adapts_their_knowledge_for_distillation.md)
- [\[ACL 2026\] Mem^p: Exploring Agent Procedural Memory](../../ACL2026/model_compression/memp_exploring_agent_procedural_memory.md)
- [\[ICCV 2025\] MSQ: Memory-Efficient Bit Sparsification Quantization](../../ICCV2025/model_compression/msq_memory-efficient_bit_sparsification_quantization.md)

</div>

<!-- RELATED:END -->

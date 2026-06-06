---
title: >-
  [Paper Note] MonarchAttention: Zero-Shot Conversion to Fast, Hardware-Aware Structured Attention
description: >-
  [NeurIPS 2025][LLM/NLP][efficient attention] This paper proposes MonarchAttention, which leverages the structured properties of Monarch matrices and employs alternating optimization over a variational form of softmax to…
tags:
  - "NeurIPS 2025"
  - "LLM/NLP"
  - "efficient attention"
  - "Monarch matrices"
  - "sub-quadratic attention"
  - "structured matrices"
  - "hardware-aware"
  - "zero-shot conversion"
date: 2026-05-08
content_hash: e6d5056d16bdadc4
---

# MonarchAttention: Zero-Shot Conversion to Fast, Hardware-Aware Structured Attention

**Conference**: NeurIPS 2025
**arXiv**: [2505.18698](https://arxiv.org/abs/2505.18698)  
**Code**: [GitHub](https://github.com/cjyaras/monarch-attention)  
**Area**: LLM/NLP
**Keywords**: efficient attention, Monarch matrices, sub-quadratic attention, structured matrices, hardware-aware, zero-shot conversion

## TL;DR

This paper proposes MonarchAttention, which leverages the structured properties of Monarch matrices and employs alternating optimization over a variational form of softmax to approximate attention at $\Theta(N\sqrt{N}d)$ complexity. The method enables zero-shot replacement of attention layers in pretrained Transformers without any additional training, while achieving 1.4×–8.2× speedups over FlashAttention-2 on GPU.

## Background & Motivation

The core attention mechanism in Transformers has $\Theta(N^2 d)$ quadratic time complexity, which is a critical bottleneck for long-sequence training and inference. Existing sub-quadratic attention methods fall into several categories:

**Low-rank methods** (linear attention, Performer, etc.): hardware-friendly but unsuitable as direct drop-in replacements for pretrained models, since attention matrices often exhibit strong diagonal structure.

**Sparse methods** (LSH, fixed sparse masks): data-dependent sparsity patterns are difficult to implement efficiently on GPU.

**Low-rank + sparse**: combined approaches improve accuracy but incur significant overhead.

These methods either require training from scratch or fine-tuning (limiting transferability), or fail to achieve practical GPU speedups (large gap between theoretical complexity and actual performance). The core motivation of MonarchAttention is to simultaneously achieve **transferability** (zero-shot replacement) and **hardware efficiency** (exploiting GPU tensor cores).

## Method

### Overall Architecture

MonarchAttention aims to find a Monarch matrix $\mathbf{M} \in \mathbb{R}^{N \times N}$ such that $\mathbf{M} \approx \text{softmax}(\mathbf{Q}\mathbf{K}^\top)$, and then efficiently compute $\mathbf{O} = \mathbf{M}\mathbf{V}$.

**Monarch matrix definition**: Given $N = m \times b$ (typically $m = b = \sqrt{N}$), a Monarch matrix is defined as $\mathbf{M} = \mathbf{P}^\top \mathbf{B}$, where $\mathbf{P}$ is a transpose permutation matrix and $\mathbf{B}$ is a block rank-one matrix. Each block $\mathbf{B}_{jk} = \mathbf{L}_{jk}\mathbf{R}_{kj}^\top$ requires only $\Theta(N\sqrt{N})$ storage, and matrix multiplication requires only $\Theta(N\sqrt{N}d)$ operations.

### Key Designs

**Variational form of softmax**: Exploiting the variational characterization of softmax:

$$\text{softmax}(\mathbf{z}) = \arg\max_{\mathbf{a} \in \Delta^N} \langle \mathbf{a}, \mathbf{z} \rangle + H(\mathbf{a})$$

where $H(\mathbf{a}) = -\sum_i \mathbf{a}_i \log \mathbf{a}_i$ is the Shannon entropy. The computation of the attention matrix is reformulated as an optimization problem:

$$\sigma(\mathbf{Q}\mathbf{K}^\top) = \arg\max_{\mathbf{A} \in \Delta^{N \times N}} f(\mathbf{A}; \mathbf{Q}, \mathbf{K}) := \langle \mathbf{A}, \mathbf{Q}\mathbf{K}^\top \rangle + H(\mathbf{A})$$

**Exploiting low-dimensional structure**: When $\mathbf{A}$ has Monarch structure, the objective decomposes into multiple independent small-scale subproblems:

$$f(\mathbf{P}^\top \mathbf{B}; \mathbf{Q}, \mathbf{K}) = \sum_{j,k} f(\mathbf{B}_{jk}; \tilde{\mathbf{Q}}_j, \mathbf{K}_k)$$

Due to the block rank-one structure, the entropy term is also separable, and each subproblem requires only $\Theta((m+b)d)$ operations.

**Alternating maximization**: The objective is concave in $\mathbf{R}$ given fixed $\mathbf{L}$ (and vice versa), yielding closed-form updates via KKT conditions:

$$\mathbf{R}^{(t)} = \text{softmax}_i(\mathbf{Z}_R^{(t)}), \quad \mathbf{L}^{(t)} = \text{softmax}_k(\mathbf{Z}_L^{(t)})$$

With initialization $\mathbf{L}_{jkl}^{(0)} = \delta_{kl}$ (identity matrix), $T$ steps of alternating optimization yield $\mathbf{M}^{(T)} \approx \sigma(\mathbf{Q}\mathbf{K}^\top)$.

### Loss & Training

MonarchAttention is an **inference-time** optimization method that involves no training loss. Its core procedure relies on alternating maximization of a variational objective to approximate the attention matrix.

**IO-optimized implementation**: $\mathbf{L}$ and $\mathbf{R}$ are not materialized in HBM; only state variables $\alpha_R, \alpha_L, c_R, c_L$ are maintained (additional $\Theta(Nd)$ memory). All intermediate values are materialized only in on-chip SRAM, achieving IO savings analogous to FlashAttention but with effective sequence length $\sqrt{N}$, eliminating the need for tiling along the sequence dimension. The optimal IO complexity is $\Theta(Nd)$, improving upon FlashAttention's worst-case $O(N^2 d^2 / S)$.

## Key Experimental Results

### Main Results

| Task | Model | Method | FLOPs Reduction | Performance Drop |
|------|-------|--------|-----------------|-----------------|
| Image Classification (ImageNet) | ViT-B (87M) | MonarchAttention | 80% | Top-5 accuracy drop of only 5% |
| Image Classification (ImageNet) | ViT-B | MonarchAttention | 50% | On par with baseline |
| QA (SQuAD) | RoBERTa-B (125M) | MonarchAttention | 60% | F1 drop of only 10 points |
| QA (SQuAD) | RoBERTa-B | MonarchAttention | 35% | On par with baseline |
| Summarization (BookSum) | BART-B (139M) | MonarchAttention (N=8192) | Similar FLOPs as softmax N=2048 | ROUGE-1 +0.75, ROUGE-L +0.5 |

**Image Generation (DiT-XL 675M, ImageNet)**:

| Replaced Layers | Method | FLOPs (×10⁹) | FID ↓ | sFID ↓ |
|-----------------|--------|--------------|-------|--------|
| All | Nyströmformer | 3.30 | 5.97 | 13.47 |
| All | MonarchAttention | 3.44 | **2.82** | **5.09** |
| First half | Nyströmformer | 5.88 | 8.17 | 19.01 |
| First half | MonarchAttention | 5.95 | **0.39** | **0.66** |
| Second half | Nyströmformer | 5.88 | 6.76 | 13.58 |
| Second half | MonarchAttention | 5.95 | **1.98** | **3.36** |

### Ablation Study

**Speed benchmarks (NVIDIA A40 GPU)**:

| Sequence Length N | MonarchAttention vs. FlashAttention-2 Speedup |
|-------------------|-----------------------------------------------|
| 256 | 1.4× |
| 4096 | 4.5× |
| 16384 | 8.2× |

Effect of iteration count $T$ on ViT accuracy: larger $T \in \{1, 2, 3\}$ yields higher accuracy at greater FLOPs cost; $T=1$ already achieves a favorable accuracy–efficiency trade-off.

### Key Findings

1. MonarchAttention substantially outperforms low-rank baselines (Performer, Nyströmformer) across all tasks, with particularly large margins on image generation.
2. On the summarization task, MonarchAttention achieves a better ROUGE vs. FLOPs trade-off than softmax attention by efficiently processing longer sequences.
3. Replacing only the first half of DiT layers yields FID of 0.39, suggesting that attention in earlier layers is more amenable to Monarch matrix approximation.
4. Even at short sequences (N=256), a 1.4× speedup is achieved thanks to a fully fused Triton kernel implementation.

## Highlights & Insights

1. **Elegant use of the variational form**: Reformulating softmax approximation as constrained optimization avoids direct computation of the $N \times N$ attention matrix.
2. **Entropy separability of block rank-one structure**: The block rank-one property of Monarch matrices enables separable computation of the entropy term, reducing complexity from $\Theta(mb)$ to $\Theta(m+b)$—a key theoretical contribution.
3. **Hardware-aware design**: Monarch matrices exploit GPU tensor cores via batched dense matrix multiplications, avoiding the memory-unfriendly access patterns of FFT-based algorithms.
4. **IO complexity strictly superior to FlashAttention**: $\Theta(Nd)$ in the worst case vs. $O(N^2 d^2 / S)$.

## Limitations & Future Work

1. **Not applicable to autoregressive generation**: Cannot be used for token-by-token decoding in decoders, as no complete attention matrix exists to approximate.
2. **Difficulty integrating causal masks**: How to efficiently incorporate causal masking into MonarchAttention remains unclear.
3. **Uniform block size allocation**: Monarch matrices assign equal capacity to each block, yet different regions of the attention matrix may vary in complexity.
4. **Scalability**: For extremely long sequences where $\sqrt{N}d > S$ (SRAM size), tiling along the sequence dimension is still required.

## Related Work & Insights

- **FlashAttention**: MonarchAttention can be viewed as a block-sparse generalization of FlashAttention, where each optimization step is expressible as a FlashAttention-style computation.
- **Monarch Mixer**: Although it also uses Monarch matrices for token mixing, it is not data-dependent; MonarchAttention preserves the data-dependent nature of attention.
- **$\alpha$-entmax**: Produces sparser attention matrices than softmax and may be better suited for block rank-one approximation, representing a promising future direction.
- The method is applicable to diffusion language models (non-autoregressive) and for accelerating the prefill stage.

## Rating

- ⭐⭐⭐⭐ **Novelty**: The combination of the variational form of softmax with Monarch structured matrices for sub-quadratic attention approximation is highly elegant.
- ⭐⭐⭐⭐ **Experimental Thoroughness**: Covers diverse architectures and tasks across vision (ViT, DiT) and language (RoBERTa, BART), providing strong empirical support.
- ⭐⭐⭐⭐⭐ **Value**: Zero-shot replacement + practical GPU speedups + open-source code yield extremely high engineering value.
- ⭐⭐⭐ **Limitations & Future Work**: Inapplicability to autoregressive LLM inference—the most dominant current use case—limits broader impact.

**Overall**: ⭐⭐⭐⭐ (4/5) — A high-quality work with elegant theory, solid experiments, and strong engineering practicality. The central limitation is incompatibility with autoregressive decoding, though the method remains highly valuable for prefill, encoder-only models, and diffusion models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CoPS: Conditional Prompt Synthesis for Zero-Shot Anomaly Detection](../../CVPR2026/llm_nlp/cops_conditional_prompt_synthesis_for_zero-shot_anomaly_detection.md)
- [\[CVPR 2026\] Boosting Quantitive and Spatial Awareness for Zero-Shot Object Counting](../../CVPR2026/llm_nlp/boosting_quantitive_and_spatial_awareness_for_zero-shot_object_counting.md)
- [\[AAAI 2026\] Soft Filtering: Guiding Zero-Shot Composed Image Retrieval with Prescriptive and Proscriptive Prompts](../../AAAI2026/llm_nlp/soft_filtering_guiding_zero-shot_composed_image_retrieval_with_prescriptive_and_.md)
- [\[NeurIPS 2025\] Spectral Conditioning of Attention Improves Transformer Performance](spectral_conditioning_of_attention_improves_transformer_performance.md)
- [\[ICML 2026\] SLAY: Geometry-Aware Spherical Linearized Attention with Yat-Kernel](../../ICML2026/llm_nlp/slay_geometry-aware_spherical_linearized_attention_with_yat-kernel.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] CAT: Circular-Convolutional Attention for Sub-Quadratic Transformers
description: >-
  [NeurIPS 2025][LLM/NLP][Circulant convolution] CAT replaces the $N \times N$ attention matrix in standard self-attention with a circulant matrix generated from an $N$-dimensional vector…
tags:
  - "NeurIPS 2025"
  - "LLM/NLP"
  - "Circulant convolution"
  - "FFT attention"
  - "sub-quadratic complexity"
  - "softmax preservation"
  - "EIT framework"
date: 2026-05-08
content_hash: f82c1e8d1c524307
---

# CAT: Circular-Convolutional Attention for Sub-Quadratic Transformers

**Conference**: NeurIPS 2025
**arXiv**: [2504.06704](https://arxiv.org/abs/2504.06704)
**Code**: Unavailable (not released by authors)
**Area**: LLM/NLP / Transformer Efficiency
**Keywords**: Circulant convolution, FFT attention, sub-quadratic complexity, softmax preservation, EIT framework

## TL;DR

CAT replaces the $N \times N$ attention matrix in standard self-attention with a circulant matrix generated from an $N$-dimensional vector, leveraging FFT to achieve $O(N \log N)$ attention computation. While strictly preserving the softmax row-normalization structure, CAT matches or surpasses standard attention on ImageNet-1k (avg pool, CLIP-L accuracy 0.694 vs. 0.646) and WikiText-103 masked LM (PPL 8.32 vs. 9.82).

## Background & Motivation

**Background**: The standard self-attention mechanism in Transformers has $O(N^2)$ complexity, limiting scalability for long-sequence tasks. Existing efficient attention approaches follow two main directions: (1) Linear Transformers (e.g., Performer) approximate softmax with kernel functions, reducing complexity to $O(N)$ but sacrificing the softmax structure at the cost of training instability; (2) sparse attention methods (e.g., BigBird, Longformer) apply softmax only over local token subsets, losing global context coverage.

**Limitations of Prior Work**: Kernel approximations in Linear Transformers frequently cause numerical instability and accuracy degradation. Sparse methods require careful tuning of block sizes and sparsity patterns, with re-tuning needed across different sequence lengths. SSM-based methods such as Mamba abandon the attention mechanism entirely, making them incompatible with existing Transformer training stacks and inference optimizations (FlashAttention, KV Cache).

**Key Challenge**: All existing methods trade off between reducing complexity and preserving the representational capacity and compatibility of softmax attention — either abandoning softmax for efficiency or retaining softmax but introducing sequence-length-dependent hyperparameters.

**Goal**: To find an approach that strictly preserves global softmax row normalization (mathematically consistent with standard attention), achieves sub-$O(N^2)$ computational complexity, and introduces no additional hyperparameters.

**Key Insight**: The authors observe that each row of the attention matrix is essentially a weighted aggregation over all tokens, and that circulant matrices naturally satisfy the structure where "each row is a cyclic shift of the same weight vector" — with circulant matrix-vector multiplication computable via FFT in $O(N \log N)$.

**Core Idea**: Replace the $N \times N$ attention matrix produced by $QK^\top$ with a circulant matrix derived from a single vector passed through softmax, and leverage FFT to achieve sub-quadratic global attention.

## Method

### Overall Architecture

CAT serves as a drop-in replacement for standard multi-head self-attention. Given input $\mathbf{X} \in \mathbb{R}^{N \times D}$, standard attention requires three projection matrices $\mathbf{W_Q}, \mathbf{W_K}, \mathbf{W_V}$ (totaling $3D^2$ parameters), whereas CAT requires only $\mathbf{W_A} \in \mathbb{R}^{D \times 1}$ (attention projection) and $\mathbf{W_V} \in \mathbb{R}^{D \times D}$ (value projection), amounting to $(D+H)D$ parameters (where $H$ is the number of heads).

### Key Designs

1. **Circulant Attention Matrix Construction**:

    - Function: Replaces the $N \times N$ attention weight matrix with a circulant matrix generated from an $N$-dimensional vector.
    - Mechanism: A single projection $\mathbf{Z} = \mathbf{X} \mathbf{W_A} \in \mathbb{R}^{N \times 1}$ yields an attention score vector, which is passed through row-wise softmax to obtain $\mathbf{Z}^\star = \text{softmax}(\mathbf{Z})$. A circulant matrix $\text{circ}(\mathbf{Z}^\star)$ is then constructed, where the $i$-th row is the $i$-th cyclic shift of $\mathbf{Z}^\star$. A key property is that $\text{softmax}(\text{circ}(\mathbf{Z})) \equiv \text{circ}(\text{softmax}(\mathbf{Z}))$ — the circulant operator and softmax commute — so each row remains a valid softmax-normalized weight distribution.
    - Design Motivation: Circulant matrices constitute the "simplest structure that preserves softmax row normalization" — each row shares the same weight set, differing only in position. This simultaneously endows the model with explicit relative positional encoding, since attention weights depend only on the relative offset between tokens.

2. **FFT-Accelerated Computation**:

    - Function: Leverages the circular convolution theorem to accelerate $O(N^2)$ matrix multiplication to $O(N \log N)$.
    - Mechanism: Circulant matrix-vector multiplication is equivalent to circular convolution. By the convolution theorem, $\text{circ}(\mathbf{Z}^\star) \mathbf{V} = \text{IFFT}[\text{FFT}(\mathbf{Z}^\star) \odot \text{FFT}(\mathbf{V})]$, where $\odot$ denotes element-wise Hadamard product. FFT/IFFT costs $O(N \log N)$ and the Hadamard product costs $O(N)$, yielding total complexity $O(N \log N)$.
    - Design Motivation: This is the only sub-quadratic attention computation that is exact rather than approximate — circulant matrix multiplication and FFT are equivalent to floating-point precision.

3. **CAT-Alter Hybrid Architecture**:

    - Function: Alternates between CAT layers and standard attention layers throughout the network.
    - Mechanism: Half of the attention layers are replaced by CAT and the remaining half retain standard attention, arranged in alternating fashion. CAT layers provide efficient $O(N \log N)$ computation and cyclic-shift regularization, while standard attention layers retain fully flexible global interaction.
    - Design Motivation: Pure CAT occasionally underperforms standard attention on tasks requiring highly flexible global interactions (e.g., causal LM with token pooling). The hybrid approach captures the advantages of both, analogous to Transformer/SSM hybrid architectures such as Jamba.

### Loss & Training

CAT does not alter the training objective — standard cross-entropy classification loss is used for vision tasks and standard cross-entropy LM loss for language modeling. Attention layers are directly substituted during training with identical hyperparameters. ImageNet-1k training runs for 50 epochs (4 × V100); WikiText-103 training also runs for 50 epochs; both use AdamW. In the causal LM setting, $\mathbf{Z}$ must be shifted to maintain causality, which degrades complexity back to $O(N^2)$.

## Key Experimental Results

### Main Results

| Task / Model | Pool/LM Type | Method | Learnable Params | Complexity | Metric |
|---|---|---|---|---|---|
| ImageNet CLIP-L | avg pool | Attention | $3D^2$ | $O(N^2)$ | Acc 0.646 |
| ImageNet CLIP-L | avg pool | **CAT** | $(D+H)D$ | $O(N\log N)$ | **Acc 0.694** (+4.8%) |
| ImageNet CLIP-L | avg pool | CAT-Alter | $(2D+H/2)D$ | $O(N^2)$ | Acc 0.681 |
| ImageNet CLIP-L | token | Attention | $3D^2$ | $O(N^2)$ | Acc 0.574 |
| ImageNet CLIP-L | token | CAT-Alter | $(2D+H/2)D$ | $O(N^2)$ | **Acc 0.593** |
| WikiText-103 GPT-2 | masked | Attention | $3D^2$ | $O(N^2)$ | PPL 9.82 |
| WikiText-103 GPT-2 | masked | **CAT** | $(D+H)D$ | $O(N\log N)$ | **PPL 8.32** (−15.3%) |
| WikiText-103 GPT-2 | causal | Attention | $3D^2$ | $O(N^2)$ | PPL 27.84 |
| WikiText-103 GPT-2 | causal | CAT-Alter | $(2D+H/2)D$ | $O(N^2)$ | PPL 27.68 |

### Ablation Study

Comparison of Q/K/V parameterization variants (CLIP-L, avg pool):

| Circulant Convolution Variant | Learnable Params | Complexity | Acc ↑ |
|---|---|---|---|
| qkv (Averaged-Key) | $3D^2$ | $O(N\log N)$ | 0.696 |
| **qv (CAT, default)** | $(D+H)D$ | $O(N\log N)$ | **0.694** |
| q only | $(N+H)D$ | $O(N\log N)$ | 0.637 |
| v only | $(N+D)D$ | $O(N\log N)$ | 0.625 |
| Standard Attention | $3D^2$ | $O(N^2)$ | 0.646 |

### Key Findings

- **CAT substantially outperforms standard attention under avg pooling**: A gain of 4.8% on CLIP-L (0.694 vs. 0.646), indicating that the cyclic-shift structure of circulant convolution is particularly effective in global token mixing scenarios.
- **Remarkable gains in masked LM**: On GPT-2, PPL drops from 9.82 to 8.32 (−15.3%); on Transformer-XL, from 13.94 to 10.28 (−26.3%), suggesting that the circulant structure has a natural affinity for masked prediction.
- **The qv combination offers the best trade-off**: Accuracy is within 0.2% of retaining full qkv, while parameter count drops from $3D^2$ to $(D+H)D$, approximately a two-thirds reduction.
- **Causal settings remain a weakness**: The causal constraint forces CAT back to $O(N^2)$, with accuracy slightly below standard attention, necessitating the CAT-Alter hybrid.
- **Linear Attention training collapses**: NaN losses appear repeatedly on CLIP-L, corroborating the importance of softmax preservation.

## Highlights & Insights

- **The commutativity of circulant matrices and softmax** is the most elegant theoretical insight in the paper: $\text{softmax}(\text{circ}(\mathbf{Z})) = \text{circ}(\text{softmax}(\mathbf{Z}))$. This means that applying softmax to an $N$-dimensional vector directly yields a valid $N \times N$ attention matrix, resolving the tension between softmax preservation and complexity reduction in a single step.
- **The perspective of reducing attention coefficients from $N^2$ to $N$**: Standard attention generates $N^2$ coefficients at runtime, whereas CAT generates only $N$ and obtains the rest via cyclic shifts. This not only reduces computation but may also serve as a regularizer, mitigating overfitting.
- **Resonance with the ViT avg pooling trend**: Recent work has shown that avg pooling often outperforms class token pooling in ViTs; CAT performs strongest precisely under avg pooling, hinting at a deeper affinity between the circulant structure and global token mixing.

## Limitations & Future Work

- **Causal LM cannot maintain sub-quadratic complexity**: The causal mask breaks the circulant structure, degrading CAT to $O(N^2)$. This is the most significant limitation — GPT-style generative models cannot benefit from the efficiency gains.
- **Practical speedup is limited**: At $N=256$, FFT overhead offsets theoretical gains, and the gather-based implementation remains $O(N^2)$ (albeit with ~10% speedup). Custom GPU kernels or much longer sequences are needed to realize the $O(N \log N)$ advantage in practice.
- **All tokens share the same attention weight set**: Each row of the circulant matrix is merely a shift of the same weights; the model cannot learn truly token-specific attention patterns, which may be a bottleneck for tasks requiring highly heterogeneous attention.
- **Not validated on ultra-long sequence tasks**: ImageNet ViT uses $N=256$ and WikiText uses $N=256$ — both very short. CAT's theoretical advantages need to be verified on long-text or long-video tasks with $N > 10{,}000$.
- **No open-source code**: Reproducibility is limited.

## Related Work & Insights

- **vs. Performer / Linear Attention**: Performer approximates softmax with random features to achieve $O(N)$ but disrupts the softmax structure, causing training instability. CAT strictly preserves softmax and is more stable, at the cost of $O(N \log N)$ rather than $O(N)$ complexity.
- **vs. BigBird / Longformer**: Sparse methods preserve softmax but only over token subsets, losing global coverage and requiring hyperparameter tuning. CAT covers all token pairs with zero additional hyperparameters.
- **vs. Mamba (SSM)**: Mamba entirely abandons attention and is incompatible with Transformer inference stacks. CAT is a drop-in replacement, fully compatible with FlashAttention, KV Cache, etc. Notably, CAT's qv fusion design bears conceptual similarity to Mamba's input-driven control.
- **Resonance with hybrid architectures such as Jamba**: The alternating strategy in CAT-Alter parallels Jamba's SSM + Attention mixing, both reflecting the difficulty of a single mechanism being optimal across all settings.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The idea of replacing the attention matrix with a circulant matrix is elegant with a solid theoretical foundation; the EIT framework definition offers meaningful guidance.
- Experimental Thoroughness: ⭐⭐⭐ Coverage spans vision and language tasks, but sequence lengths are short, and long-sequence validation and error bars are absent.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear and ablation designs are well-motivated, though some concepts are repeated verbosely.
- Value: ⭐⭐⭐⭐ Provides a novel perspective for efficient attention design, though the causal setting limitation reduces practical applicability to LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Strassen Attention, Split VC Dimension and Compositionality in Transformers](strassen_attention_split_vc_dimension_and_compositionality_in_transformers.md)
- [\[AAAI 2026\] Vision Transformers are Circulant Attention Learners](../../AAAI2026/llm_nlp/vision_transformers_are_circulant_attention_learners.md)
- [\[NeurIPS 2025\] msf-CNN: Patch-based Multi-Stage Fusion with Convolutional Neural Networks for TinyML](msf-cnn_patch-based_multi-stage_fusion_with_convolutional_neural_networks_for_ti.md)
- [\[NeurIPS 2025\] Spectral Conditioning of Attention Improves Transformer Performance](spectral_conditioning_of_attention_improves_transformer_performance.md)
- [\[NeurIPS 2025\] Linear Transformers Implicitly Discover Unified Numerical Algorithms](linear_transformers_implicitly_discover_unified_numerical_algorithms.md)

</div>

<!-- RELATED:END -->

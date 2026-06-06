---
title: >-
  [Paper Note] Share Your Attention: Transformer Weight Sharing via Matrix-Based Dictionary Learning
description: >-
  [AAAI 2026][Model Compression][weight sharing] Inspired by dictionary learning, this paper proposes the MASA framework, which decomposes the attention projection matrices (Q/K/V/O) across Transformer layers into linear c…
tags:
  - "AAAI 2026"
  - "Model Compression"
  - "weight sharing"
  - "dictionary learning"
  - "transformer compression"
  - "attention compression"
  - "parameter efficiency"
date: 2026-05-08
content_hash: d46212b66ebe6911
---

# Share Your Attention: Transformer Weight Sharing via Matrix-Based Dictionary Learning

**Conference**: AAAI 2026
**arXiv**: [2508.04581](https://arxiv.org/abs/2508.04581)  
**Code**: [https://github.com/mts-ai/MASA](https://github.com/mts-ai/MASA)  
**Area**: Model Compression
**Keywords**: weight sharing, dictionary learning, transformer compression, attention compression, parameter efficiency

## TL;DR

Inspired by dictionary learning, this paper proposes the MASA framework, which decomposes the attention projection matrices (Q/K/V/O) across Transformer layers into linear combinations of shared matrix atoms, achieving performance on par with or superior to the original Transformer at a 66.7% attention parameter compression ratio.

## Background & Motivation

Large language models (LLMs) impose substantial computational and memory demands during deployment. Existing compression techniques primarily focus on **intra-block optimization** (e.g., low-rank approximation, attention head pruning), yet the repetitive layer structure of Transformers implies significant **cross-layer redundancy**—a dimension that remains largely unexplored beyond KV caching.

Specifically, a Transformer with $L$ layers and hidden dimension $d$ requires $\mathcal{O}(L \cdot d^2)$ parameters, with attention modules alone accounting for roughly half the parameters in base models such as LLaMA and Mistral. While recent methods including GQA, Sequential-sharing, and Repeat-all-over explore cross-layer sharing, they either suffer notable performance degradation (especially on reasoning tasks) or lack a principled framework for capturing cross-layer statistical patterns.

The core motivation of MASA is that **cross-layer statistical regularities exist among the attention weight matrices of different Transformer layers and can be captured by dictionary learning**, allowing each layer's weights to be represented as linear combinations of a small set of shared matrix atoms, thereby enabling substantial parameter compression.

## Method

### Overall Architecture

MASA maintains independent dictionary pools for each of the four projection types Q, K, V, and O. Each dictionary contains $S$ shared matrix atoms $\mathbf{D}_s \in \mathbb{R}^{d \times h}$. The weight matrix for each projection type at each layer is reconstructed by a weighted sum of the shared atoms using a layer-specific coefficient vector $\mathbf{c}_l \in \mathbb{R}^S$:

$$\hat{\mathbf{W}}_l = \sum_{s=1}^{S} c_{ls} \mathbf{D}_s$$

Both the dictionary atoms and the linear coefficients are jointly learned via backpropagation on the training loss, without any auxiliary dictionary learning loss.

### Key Designs

1. **Matrix Atom Sharing Mechanism**:

    - Function: Represents each layer's attention projection matrix as a linear combination of shared matrix atoms.
    - Mechanism: Drawing on classical dictionary learning from signal processing, each weight matrix is treated as a "signal" reconstructed from a shared "dictionary."
    - Design Motivation: Unlike low-rank methods that impose a uniform rank constraint across all layers, MASA allows the effective rank to vary across layers and projection types, more flexibly capturing cross-layer redundancy.
    - Compression ratio formula: $r \approx 1 - S/L$; setting $S = L/3$ achieves 66.7% compression.

2. **MLP-Parameterized Mixing Coefficients (Embedding-based Coefficient Parameterization)**:

    - Function: Assigns each Transformer block an independent trainable embedding vector, from which a 3-layer MLP predicts the mixing coefficients $\mathbf{c}_l$.
    - Mechanism: Decouples the optimization of mixing coefficients from direct gradient updates, smoothing the training process.
    - Design Motivation: Reduces gradient variance and acts as implicit regularization; the MLP and embeddings are discarded after training, retaining only the final coefficient matrix $\mathbf{C}$ with no additional inference overhead.

3. **Matrix PCA Adaptation for Pretrained Models**:

    - Function: Enables training-free compression of existing pretrained LLMs.
    - Mechanism: Analytically solves for the optimal orthogonal matrix basis (Matrix PCA), then improves reconstruction accuracy via a grouping strategy and local residual refinement.
    - Grouping strategy: Uses KL divergence to measure changes in adjacent layer output distributions, clustering functionally similar layers to share a dictionary.
    - Local refinement: Applies Cholesky whitening to the residual $\Delta \mathbf{W}_l$ followed by low-rank approximation, with rank budget adaptively allocated according to the matrix role (Q/K/V/O).

### Loss & Training

- Training loss: Standard language modeling cross-entropy loss with no auxiliary objectives.
- Optimizer: AdamW ($\beta_1=0.9$, $\beta_2=0.999$, weight decay=0.1).
- Follows Chinchilla-optimal training: number of training tokens equals 20× the model parameter count.
- Linear warmup for the first 10% of steps, followed by cosine annealing.
- Gradient clipping at global norm 1.0.
- FlashAttention used to accelerate long-sequence training.

## Key Experimental Results

### Main Results

| Model (Scale) | Method | Attn. Compression | AVG Acc (%) ↑ | WikiText PPL ↓ | LAMBADA PPL ↓ |
|--------------|--------|-------------------|---------------|----------------|---------------|
| Transformer-S (110M) | Vanilla | 0% | 33.48 | 76.11 | 167.39 |
| Transformer-S | MASA-QKV | 50% | **34.43** | **72.08** | **112.23** |
| Transformer-S | MASA-QKVO | 66.7% | 33.74 | 72.82 | 133.62 |
| Transformer-S | Low-Rank | 66.7% | 32.27 | 83.25 | 264.52 |
| Transformer-S | GQA | 41.7% | 33.34 | 78.41 | 187.71 |
| Transformer-L (729M) | Vanilla | 0% | 42.12 | 30.88 | 20.73 |
| Transformer-L | MASA-QKV | 50% | 41.74 | **30.83** | 22.08 |
| Transformer-L | MASA-QKVO | 66.7% | 41.30 | 31.34 | 21.21 |

### Pretrained Model Compression Results

| Model | Method | Compression | AVG Acc (%) ↑ | WikiText PPL ↓ |
|-------|--------|-------------|---------------|----------------|
| Llama 3.2 1B | Vanilla | 0% | 57.61 | 11.57 |
| Llama 3.2 1B | SVD-LLM | 20% | 53.11 | 15.08 |
| Llama 3.2 1B | Matrix PCA (Ours) | 20% | **55.34** | **12.61** |
| Llama 3.1 8B | Vanilla | 0% | 70.93 | 7.33 |
| Llama 3.1 8B | Matrix PCA (Ours) | 20% | **70.09** | **7.84** |

### Ablation Study

| Configuration | Key Metric | Notes |
|--------------|-----------|-------|
| Separate vs. QV-shared dictionary | Separate: 34.43% vs. QV-Common: 33.95% | Independent dictionaries per projection type perform best |
| With/without O projection compression | QKV: 34.43% vs. QKVO: 33.74% | O projection is more sensitive; Q/K/V are more compressible |
| Dictionary size $S=2 \to 8$ | Acc 33.82%→33.94%, PPL 74.79→70.66 | Larger dictionaries yield consistent improvement |
| Large-scale training (65B tokens) | MASA-QKV vs. Vanilla: Acc gap only 0.23% | MASA remains competitive under large-scale training |

### Key Findings

- **MASA-QKV (50% compression) can outperform an uncompressed Transformer**: accuracy improves by ~1% and perplexity decreases substantially on small models.
- **Q/K/V projections are more compressible than the O projection**: even at high compression of Q/K/V ($S=2$, 62.5%), performance is comparable to the vanilla model.
- **Negligible computational overhead**: MASA-QKVO incurs only ~8.3% throughput reduction compared to the original model (1240 vs. 1352 tokens/sec).
- **Effective on ViTs**: 66.7% attention parameter compression on CIFAR-10/100 and TinyImageNet yields performance on par with or better than vanilla models.
- **Pretrained model adaptation**: 20% attention compression of Llama 3.1 8B retains approximately 99% of downstream accuracy.

## Highlights & Insights

- **Theoretical elegance**: Recasts attention compression as a dictionary learning problem, establishing a principled connection between classical signal processing and Transformer efficiency.
- **Plug-and-play**: Requires no distillation, regularization, or architectural modifications; trains with a standard optimizer while preserving the original training pipeline.
- **Sophisticated pretrained adaptation design**: The combination of Matrix PCA, KL divergence-based grouping, and adaptive rank allocation significantly outperforms SVD-LLM without any fine-tuning.
- **Practical design principle**: Prioritize compression of Q/K/V projections while preserving the independence of the O projection—offering valuable empirical guidance for future Transformer compression work.

## Limitations & Future Work

- Only attention modules are compressed; FFN modules (accounting for the other half of parameters) are not addressed, and joint compression warrants exploration.
- Matrix PCA adaptation for larger models (>8B) requires more refined grouping strategies.
- Redundancy among dictionary atoms grows with $S$; dictionary sparsification or rank-constrained learning could be introduced.
- Combination with other compression techniques such as quantization remains unexplored.
- Validation is primarily on language modeling; performance under instruction tuning and downstream fine-tuning scenarios has yet to be examined.

## Related Work & Insights

- **GQA** (Ainslie et al., 2023): Performs parameter sharing at the KV-head level, but is limited to within a single layer.
- **Basis Sharing** (Wang et al., ICLR 2025): Shares singular vectors via SVD but lacks layer-adaptive control.
- **Repeat-all-over / Sequential-sharing**: Deterministically repeats weights across layers; overly rigid, leading to performance degradation on reasoning tasks.
- MASA unifies dictionary learning with Transformer design, providing a continuous spectrum between "full sharing" and "full independence."

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing dictionary learning for cross-layer Transformer weight sharing is a novel perspective with a solid theoretical foundation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers LLMs at three scales, Vision Transformers, and pretrained model adaptation, with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with good integration of theoretical derivation and empirical results.
- Value: ⭐⭐⭐⭐ Provides a practical Transformer compression solution and a clear design principle (prioritize QKV compression) with meaningful implications for future work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Revisiting Weight Regularization for Low-Rank Continual Learning](../../ICLR2026/model_compression/revisiting_weight_regularization_for_low-rank_continual_learning.md)
- [\[ICML 2026\] Provably Learning Attention with Queries](../../ICML2026/model_compression/provably_learning_attention_with_queries.md)
- [\[CVPR 2026\] Elastic Weight Consolidation Done Right for Continual Learning](../../CVPR2026/model_compression/elastic_weight_consolidation_done_right_for_continual_learning.md)
- [\[NeurIPS 2025\] Spark Transformer: Reactivating Sparsity in FFN and Attention](../../NeurIPS2025/model_compression/spark_transformer_reactivating_sparsity_in_ffn_and_attention.md)
- [\[ICML 2026\] Task-Driven Subspace Decomposition for Knowledge Sharing and Isolation in LoRA-based Continual Learning](../../ICML2026/model_compression/task-driven_subspace_decomposition_for_knowledge_sharing_and_isolation_in_lora-b.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Polar Sparsity: High Throughput Batched LLM Inferencing with Scalable Contextual Sparsity
description: >-
  [NeurIPS 2025][LLM/NLP][LLM inference] This paper reveals a "polarity shift" phenomenon in LLM inference sparsity — MLP layer sparsity vanishes as batch size increases…
tags:
  - "NeurIPS 2025"
  - "LLM/NLP"
  - "LLM inference"
  - "contextual sparsity"
  - "attention head sparsity"
  - "batched inference"
  - "GPU kernel"
date: 2026-05-08
content_hash: 08de4ea6fa92e2da
---

# Polar Sparsity: High Throughput Batched LLM Inferencing with Scalable Contextual Sparsity

**Conference**: NeurIPS 2025
**arXiv**: [2505.14884](https://arxiv.org/abs/2505.14884)  
**Code**: [susavlsh10/Polar-Sparsity](https://github.com/susavlsh10/Polar-Sparsity)  
**Area**: LLM/NLP
**Keywords**: LLM inference, contextual sparsity, attention head sparsity, batched inference, GPU kernel

## TL;DR

This paper reveals a "polarity shift" phenomenon in LLM inference sparsity — MLP layer sparsity vanishes as batch size increases, while attention head sparsity remains stable and batch-invariant. Based on this insight, the authors design Selective Head Attention and corresponding GPU kernels, achieving up to 2.2× end-to-end speedup in large-batch inference.

## Background & Motivation

Contextual activation sparsity is a promising direction for accelerating LLM inference: each token activates only a small fraction of model parameters. However, **existing methods do not scale to large-batch inference** — which is critical in real-world deployments.

The root cause lies in two conflicting observations:

**MLP sparsity rapidly diminishes as batch size grows**: the union of active neurons across sequences in a batch quickly approaches dense computation.

**Attention layers become the bottleneck as batch size and sequence length increase**: attention latency grows linearly with batch size, eventually dominating end-to-end latency.

**Prior work only optimizes single-query inference**: methods such as DejaVu and PowerInfer lose their benefits at large batch sizes.

Taking OPT-66B as an example, as batch size increases from 1 to 64, MLP layers become efficient due to batching, whereas attention layer latency grows nearly linearly, emerging as the new bottleneck.

## Method

### Overall Architecture

The core insight of Polar Sparsity is that **the locus of sparsity shifts in polarity from MLP layers to attention layers**.

- **Small batch**: MLP sparsity is effective and attention overhead is low → conventional activation sparsity methods work well.
- **Large batch**: MLP sparsity disappears and attention becomes the primary bottleneck → attention head sparsity is required.

The system comprises two core components:
1. **Dynamic sparsity for MLP layers**: Selective GEMM kernel with a dynamic per-layer top-k strategy.
2. **Stable sparsity for attention layers**: Selective Head Attention (SHA) kernel.

### Key Designs

**Dynamic MLP Sparsity**:

For a batched input $\mathbf{x} \in \mathbb{R}^{B \times 1 \times d}$, the sparse MLP computation is:

$$\text{MLP}_{S_B}(\mathbf{x}) = \sigma(\mathbf{x} W_{1, S_B}) W_{2, S_B}^\top$$

where $S_B \subseteq [D]$ is the union of active neurons across all sequences in the batch.

- A lightweight two-layer feedforward network serves as the router to predict neuron activations.
- A **dynamic top-k mechanism** is proposed: different layers use different $k$ values, determined offline via a greedy algorithm to achieve a target recall of 99%.
- A GPU kernel (Selective GEMM) is designed to fuse indexing and matrix multiplication, avoiding gather-scatter overhead.

**Attention Head Sparsity**:

Key observation: for each token, only a small number of attention heads contribute significantly to the output; the influence of the remaining heads is negligible.

$$\text{Attention}(Q_{b,i}, K_{b,i}, V_{b,i}) = \text{softmax}\left(\frac{Q_{b,i} K_{b,i}^\top}{\sqrt{d_h}}\right) V_{b,i}$$

Since attention is computed independently per sequence, **head sparsity is invariant to batch size** — this is its fundamental advantage over MLP sparsity.

Empirical findings include:
- Perplexity increases slowly within 50% head sparsity when the most important heads are retained.
- **Larger models exhibit higher head sparsity**: OPT-66B incurs only a 5% perplexity increase at 30% head activation.
- Layer 0 consistently has the highest attention importance → dense attention is applied to Layer 0.

**Selective Head Attention (SHA) Kernel**:

A sparsity-aware kernel modified from FlashAttention:
- Accepts a batch head index tensor recording active head indices per batch entry.
- Each CUDA thread-block handles one batch entry and one head.
- Only active heads trigger read/write operations, reducing memory I/O and computation.
- Group sparsity is applied for GQA models.

### Loss & Training

Router training:
- 400K tokens are sampled from the Wikitext-2 training set.
- MLP router: two-layer feedforward network trained with binary cross-entropy loss and AdamW optimizer.
- Attention router: single-layer fully connected network, supervised by top-k labels derived from the L2 norm of attention outputs.

## Key Experimental Results

### Main Results

**Zero-shot benchmark evaluation** (at key sparsity thresholds):

| Model | COPA | OBQA | PIQA | RTE | WG | HS | MMLU | ARC-E | ARC-C | Avg |
|-------|------|------|------|-----|-----|-----|------|-------|-------|-----|
| OPT 66B | 0.85 | 0.304 | 0.787 | 0.603 | 0.690 | 0.557 | 0.263 | 0.711 | 0.369 | 0.570 |
| OPT 66B + PS-0.3 | 0.83 | 0.296 | 0.769 | 0.592 | 0.677 | 0.546 | 0.264 | 0.693 | 0.361 | 0.560 |
| LLaMA 2 7B | 0.87 | 0.314 | 0.781 | 0.628 | 0.690 | 0.572 | 0.418 | 0.763 | 0.433 | 0.608 |
| LLaMA 2 7B + PS-0.5 | 0.89 | 0.312 | 0.779 | 0.552 | 0.687 | 0.568 | 0.356 | 0.762 | 0.439 | 0.594 |
| LLaMA 3.1 70B | 0.92 | 0.370 | 0.831 | 0.697 | 0.799 | 0.665 | 0.753 | 0.872 | 0.606 | 0.724 |
| LLaMA 3.1 70B + PS-0.625 | 0.91 | 0.340 | 0.823 | 0.729 | 0.793 | 0.650 | 0.732 | 0.853 | 0.590 | 0.712 |

Average accuracy degradation across all models at their respective key sparsity thresholds is less than 1%.

**Comparison with other sparse methods** (LLaMA-2-7B):

| Method | COPA | PIQA | WG | HS | MMLU(5) | ARC-E | ARC-C |
|--------|------|------|-----|-----|---------|-------|-------|
| Dense | 0.87 | 0.781 | 0.690 | 0.572 | 0.458 | 0.763 | 0.433 |
| ReLUfication | 0.83 | 0.779 | 0.686 | 0.548 | 0.386 | 0.738 | 0.396 |
| CATS-50% | — | 0.769 | 0.675 | 0.571 | 0.421 | 0.744 | 0.412 |
| TEAL-50% | — | 0.778 | 0.673 | — | 0.405 | — | — |
| **PolarSparse-50%** | **0.89** | 0.779 | **0.687** | 0.568 | 0.381 | **0.762** | **0.439** |

### Ablation Study

**Throughput improvements**:

| Model | Config | Batch=1 Speedup | Large-Batch Speedup |
|-------|--------|-----------------|---------------------|
| OPT-66B | PS-0.3 | ~1× | **2.2×** |
| LLaMA-2-7B | PS-0.5 | ~1× | **1.85×** |
| LLaMA-3.1-70B | PS-0.625 | ~1× | **1.51×** |

**GPU kernel performance**:
- Selective GEMM: up to **5.5×** speedup (vs. dense baseline).
- Selective FlashAttention: **2.8×** speedup at 30% sparsity.
- Both kernels exhibit near-linear sparsity-to-speedup scaling.

### Key Findings

1. **The sparsity polarity shift has been quantitatively validated**: at batch=64, OPT-66B MLP union activation rate approaches 100%, while head sparsity remains unchanged.
2. **Larger models are better suited for head sparsity**: OPT-66B loses only 5% perplexity at 30% activation.
3. **This work is the first to demonstrate that contextual sparsity scales to large batches**: all prior work is effective only at batch=1.
4. **GQA models require higher sparsity thresholds** (62.5% vs. 30–50%), as KV cache sharing within groups inherently weakens group-level sparsity.

## Highlights & Insights

1. **The core insight is concise yet profound**: the polarity shift of sparsity from MLP to attention is an intuitive but previously overlooked observation with strong practical implications.
2. **The system design is end-to-end complete**: from router training to GPU kernels to full system integration.
3. **Broad model coverage**: effectiveness is validated on OPT, LLaMA-2/3, Mistral, and Qwen.
4. **Seamless integration with existing inference frameworks**: built on FlashAttention Triton kernels with CUDA Graphs support.
5. **The batch-invariant property** makes it naturally suitable for batched serving in production environments.

## Limitations & Future Work

1. Limited benefit in small-batch inference, where GPU utilization is insufficient to realize sparsity advantages.
2. The fixed top-k strategy lacks flexibility; input- and layer-adaptive dynamic strategies may perform better.
3. Combining head sparsity with token sparsity could yield multiplicative speedups and warrants further exploration.
4. Evaluation is limited to context lengths up to 16K; performance under million-token contexts remains unverified.
5. Group sparsity for GQA models is less effective than head sparsity for MHA models.
6. Only greedy decoding is supported; sparsity patterns under beam search and speculative decoding may differ.

## Related Work & Insights

- Complementary to DejaVu (primarily MLP sparsity): Polar Sparsity demonstrates greater advantages at large batch sizes.
- MoA/MoH-style MoE head routing achieves only theoretical FLOP reductions without actual speedup; this work delivers kernel-level practical acceleration.
- A promising future direction is task-aware, query-sensitive routing — allocating different numbers of heads to hard and easy queries within the same batch — which may enable lossless sparse inference.

## Rating

- Novelty: ⭐⭐⭐⭐ The polarity shift observation is not difficult to discover in isolation, but systematically translating it into an engineering-ready acceleration solution is a significant contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple model families and scales, includes accuracy, throughput, and kernel microbenchmarks, with detailed comparisons against multiple baselines.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear and experiments are comprehensive, though technical details could be presented more concisely.
- Value: ⭐⭐⭐⭐⭐ A directly deployable inference acceleration solution that is the first to address the failure of contextual sparsity under large-batch settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Resting Neurons, Active Insights: Robustify Activation Sparsity for Large Language Models](../../ICML2026/llm_nlp/resting_neurons_active_insights_robustify_activation_sparsity_for_large_language.md)
- [\[NeurIPS 2025\] Detecting High-Stakes Interactions with Activation Probes](detecting_high-stakes_interactions_with_activation_probes.md)
- [\[AAAI 2026\] Scalable and Accurate Graph Reasoning with LLM-Based Multi-Agents](../../AAAI2026/llm_nlp/scalable_and_accurate_graph_reasoning_with_llm-based_multi-agents.md)
- [\[NeurIPS 2025\] Reparameterized LLM Training via Orthogonal Equivalence Transformation](reparameterized_llm_training_via_orthogonal_equivalence_transformation.md)
- [\[NeurIPS 2025\] Q♯: Provably Optimal Distributional RL for LLM Post-Training](qsharp_provably_optimal_distributional_rl_for_llm_post-training.md)

</div>

<!-- RELATED:END -->

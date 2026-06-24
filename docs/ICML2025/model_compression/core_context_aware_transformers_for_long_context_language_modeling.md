---
title: >-
  [Paper Note] Core Context Aware Transformers for Long Context Language Modeling
description: >-
  [ICML 2025][Model Compression][Long Context Modeling] Proposes Core Context Aware (CCA) Attention, which dynamically compresses input tokens into a small number of core tokens through globality-aware pooling, combined with a locality-preserving module to capture adjacent fine-grained information. It achieves plug-and-play replacement of standard self-attention, yielding a 7.9× speedup and 46% GPU memory savings under 128K context while maintaining modeling performance.
tags:
  - "ICML 2025"
  - "Model Compression"
  - "Long Context Modeling"
  - "Efficient Attention"
  - "KV Cache Compression"
  - "Core Context"
  - "Linear Complexity"
date: 2026-05-08
content_hash: b5cee3e930646a5e
---

# Core Context Aware Transformers for Long Context Language Modeling

**Conference**: ICML 2025  
**arXiv**: [2412.12465](https://arxiv.org/abs/2412.12465)  
**Code**: None  
**Area**: Model Compression  
**Keywords**: Long Context Modeling, Efficient Attention, KV Cache Compression, Core Context, Linear Complexity

## TL;DR

Proposes Core Context Aware (CCA) Attention, which dynamically compresses input tokens into a small number of core tokens through globality-aware pooling, combined with a locality-preserving module to capture adjacent fine-grained information. It achieves plug-and-play replacement of standard self-attention, yielding a 7.9× speedup and 46% GPU memory savings under 128K context while maintaining modeling performance.

## Background & Motivation

When the context length $L$ of Transformers scales to extreme values (e.g., 128K), the self-attention mechanism faces two core challenges:

**Surge in redundant information**: Not all contexts are equally important to the target token. Attention scores exhibit highly sparse distributions across most layers and heads, leaving a large amount of computation wasted on redundant context.

**Computational and storage overhead**: The computational complexity of standard self-attention is $O(L^2)$, and KV cache storage scales as $O(L)$, posing a severe bottleneck in ultra-long context regimes.

Limitations of prior work:

- **StreamingLLM / LM-Infinite**: Only retain attention for a few tokens at the beginning and the end, completely ignoring information exchange among middle tokens, which severely degrades performance on QA tasks requiring full-text understanding.
- **MInference**: Employs fixed sparse patterns determined offline (such as A-shape, vertical-slash, block-sparse), failing to dynamically adapt to different inputs.
- **LongLoRA**: Grouped attention with shift strategies, but inter-group communication remains constrained.
- **Linear Attention (RetNet, etc.)**: Requires training from scratch, preventing the reuse of pretrained LLM knowledge.

The authors observe that there is a distinct division between "core context" and "redundant context" in attention scores (as illustrated in Figure 1), where the core context is concentrated on a small number of high-attention tokens. This observation inspires the design idea of dynamically identifying and focusing on core context.

## Method

### Overall Architecture

CCA-Attention consists of two complementary modules integrated through a differentiable fusion strategy:

1. **Globality-aware Pooling Module**: Groups the input token sequence and dynamically compresses each group into a single core token. These core tokens are used instead of original tokens for attention computation to capture long-range global dependencies.
2. **Locality-preserving Module**: Preserves $s$ adjacent tokens of the query token for fine-grained attention, supplementing local information lost by the global module.
3. **Differentiable Fusion**: Concatenates the Key-Value pairs of both modules and computes attention using a unified softmax, ensuring that each token can access information from all preceding tokens.

CCA-Attention is plug-and-play: its input and output dimensions are perfectly aligned with standard self-attention, allowing it to directly replace attention layers in pretrained LLMs with only minimal fine-tuning.

### Key Designs

#### 1. Globality-Aware Pooling: Dynamic Core Token Generation

Given an input sequence $\mathbf{X} = [\mathbf{x}_1; \mathbf{x}_2; \dots; \mathbf{x}_L]$, it is divided into $m = \lfloor L/g \rfloor$ groups, each containing $g$ tokens.

For the $i$-th group, the query of the last token in the group $\mathbf{x}_{ig}$ is utilized to evaluate the importance of each token within the group, generating the core token $\mathbf{c}_i$ via weighted pooling:

$$\mathbf{c}_i = \text{softmax}\left(\frac{\mathbf{Q}_{ig} \mathbf{K}_{\mathcal{I}_i}^{\prime\top}}{\sqrt{d}}\right) \mathbf{X}_{\mathcal{I}_i}^{G}$$

where $\mathbf{Q}_{ig} = \mathbf{x}_{ig} \mathbf{W}^Q$, $\mathbf{K}_{\mathcal{I}_i}^{\prime} = \mathbf{X}_{\mathcal{I}_i}^{G} \mathbf{W}^K$.

**Design Motivation**: Attention map visualizations demonstrate that important tokens consistently receive high attention scores from subsequent tokens, making the use of the last token in a group as the "evaluator" a natural and effective choice.

The core token sequence $\mathbf{C} = [\mathbf{c}_1; \dots; \mathbf{c}_m]$ compresses the original $L \times d$ representation to $m \times d$, significantly reducing the computational workload of subsequent attention and the size of the KV cache.

The global Key and Value are computed using the core tokens:

$$\mathbf{K}^G = \mathbf{C} \mathbf{W}^K, \quad \mathbf{V}^G = \mathbf{C} \mathbf{W}^V$$

For query $\mathbf{Q}_i$, the global attention only accesses the core tokens preceding the index $j = \max(0, \lfloor(i-s)/g\rfloor)$, excluding nearby tokens (which are handled by the local module).

#### 2. Locality-Preserving Module: Fine-Grained Local Attention

Each query $\mathbf{Q}_i$ performs full attention on at least $s$ preceding tokens. The local window size is adaptively adjusted to $s + ((i-s) \bmod g)$ to ensure that all tokens are covered by the attention compute without omissions.

$$\mathbf{K}_{\mathcal{U}_i}^L = [\mathbf{K}_k^L; \cdots; \mathbf{K}_i^L], \quad k = \max(1, i - s - ((i-s) \bmod g))$$

**Critical Design**: The local module shares the projection parameters $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V$ with the global module, introducing no extra parameters.

#### 3. Differentiable Fusion Strategy

The K/V from both the global and local modules are concatenated to compute the attention output in a unified manner through a single softmax:

$$\mathbf{Att}_i = \text{softmax}\left(\frac{\mathbf{Q}_i [\widetilde{\mathbf{K}}_{\mathcal{T}_i}^G; \widetilde{\mathbf{K}}_{\mathcal{U}_i}^L]^\top}{\sqrt{d}}\right) [\widetilde{\mathbf{V}}_{\mathcal{T}_i}^G; \widetilde{\mathbf{V}}_{\mathcal{U}_i}^L]$$

This concatenated fusion (instead of a weighted sum) ensures that local and global information compete under a unified softmax normalization, achieving an end-to-end differentiable and adaptive allocation. The paper proves (Proposition 1) that CCA-Attention is equivalent to a specific variant of full attention, where each token can access information from all preceding tokens, ensuring information integrity.

#### 4. Flexibility at Inference Stage

- During inference, $g$ and $s$ can be dynamically adjusted to generate different model variants for efficiency-accuracy trade-offs, adapting to varying traffic scenarios.
- Implemented with Triton kernels to accelerate parallel computation in both training and inference.
- The KV cache only stores $\mathbf{K}^G, \mathbf{V}^G$ of core tokens and $\mathbf{K}^L, \mathbf{V}^L$ within the local window, reducing storage complexity from $O(L)$ to $O(L/g + s)$.

### Loss & Training

CCA-Attention supports three training strategies:

| Strategy | Description | Applicable Scenarios |
|------|------|----------|
| **Training from Scratch** | Full training of CCA-Attention on large-scale corpora | Pursuing optimal performance, with abundant resources |
| **Full Fine-Tuning** | Fine-tuning all parameters based on pretrained LLM weights | Balance between performance and efficiency |
| **Partial Fine-Tuning** | Fine-tuning only $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V$ | Resource-constrained, fast adaptation |

Core Advantage: Unlike linear attention (which requires training from scratch), CCA-Attention can directly leverage the knowledge of pretrained LLMs, requiring only minimal fine-tuning to deploy.

## Key Experimental Results

### Main Results

Results on LLaMA2-7B-32K on the LongBench-E benchmark (32K context):

| Method | Average Score | First-token Latency (s) | GPU Memory (GB) | Speedup | GPU Memory Savings |
|------|--------|------------------|----------|--------|----------|
| Vanilla Self-Attention | 22.11 | 9.15 | 35.58 | 1.0× | — |
| StreamingLLM | 14.95 | 5.75 | 22.94 | 1.6× | 35%↓ |
| LM-Infinite | 18.76 | 4.72 | 26.35 | 1.9× | 26%↓ |
| MInference | 21.14 | 4.20 | 33.52 | 2.2× | 6%↓ |
| **CCA-LLM (Ours)** | **21.86** | **2.59** | **19.12** | **3.5×** | **46%↓** |

Results under 64K context on LLaMA2-7B-80K:

| Method | Average Score | First-token Latency (s) | GPU Memory (GB) | Speedup | GPU Memory Savings |
|------|--------|------------------|----------|--------|----------|
| Vanilla Self-Attention | 22.42 | 32.43 | 60.03 | 1.0× | — |
| StreamingLLM | 14.94 | 9.04 | 37.45 | 3.6× | 37%↓ |
| LM-Infinite | 21.20 | 8.27 | 41.54 | 3.9× | 31%↓ |
| MInference | 22.08 | 8.14 | 54.09 | 4.0× | 10%↓ |
| **CCA-LLM (Ours)** | **22.24** | **6.42** | **33.86** | **5.7×** | **44%↓** |

Validation on newer models (LLaMA3.1-8B-128K and Qwen2.5-7B-128K, 32K context):

| Method | Base Model | Average Score | Speedup | GPU Memory Savings |
|------|----------|--------|--------|----------|
| Vanilla | LLaMA3.1-8B | 37.93 | 1.0× | — |
| MInference | LLaMA3.1-8B | 37.74 | 1.9× | 11%↓ |
| **CCA-LLM** | **LLaMA3.1-8B** | **37.81** | **3.1×** | **49%↓** |
| Vanilla | Qwen2.5-7B | 38.38 | 1.0× | — |
| MInference | Qwen2.5-7B | 36.72 | 2.2× | 8%↓ |
| **CCA-LLM** | **Qwen2.5-7B** | **38.08** | **3.9×** | **45%↓** |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|----------|------|
| Global Module Only | Significant performance drop | Lack of local fine-grained information, insufficient short-range dependency modeling |
| Local Module Only | Degradation on long-range tasks | Fails to capture long-distance semantic correlations |
| Global + Local Concatenated Fusion | Best performance | Competitive fusion under unified softmax outperforms weighted summation |
| $g$=8, $s$=512 | High accuracy, moderate speed | Smaller group size preserves more information |
| $g$=64, $s$=512 | Fastest speed, slight decrease in accuracy | Larger compression ratio, suitable for extreme efficiency scenarios |
| Partial Fine-Tuning vs Full Fine-Tuning | Small accuracy gap | Fine-tuning only QKV weights achieves performance close to full fine-tuning |

### Key Findings

1. **Superior efficiency-performance trade-off compared to all baselines**: CCA-LLM achieves the maximum speedup and memory savings while keeping average scores close to full attention. Although MInference maintains accuracy, its memory savings are only 6-11%, far below CCA's 44-49%.
2. **Severe accuracy degradation in StreamingLLM / LM-Infinite**: These methods gain efficiency by discarding intermediate tokens, but their average scores drop from 22 to 14-18, rendering them almost unusable for multi-document QA and summarization tasks.
3. **Strong cross-model generalization**: Consistent advantages are demonstrated across three models with different architectures: LLaMA2, LLaMA3.1, and Qwen2.5.
4. **Up to 7.9× speedup achieved in 128K context**: As the context length increases, the efficiency advantages of CCA-Attention become more prominent.

## Highlights & Insights

1. **Dynamic generation of core tokens**: Unlike static sparse patterns (e.g., fixed strided attention in BigBird), CCA dynamically generates core tokens using intra-group weighted pooling, allowing the compressed representation to adaptively preserve the most critical information. This is the core novelty of this work.
2. **Plug-and-play design**: CCA-Attention maintains the same input/output interfaces and parameter dimensions as standard self-attention, enabling it to directly replace the attention layers in pretrained LLMs with only partial fine-tuning of QKV weights. This significantly lowers deployment barriers.
3. **Complementary global-local philosophy**: The dual-module design of coarse-grained global information + fine-grained local information, integrated via concatenation + unified softmax fusion, elegantly ensures information integrity (each token can indirectly access all preceding tokens).
4. **Dynamic hyperparameter tuning during inference**: Flexible adjustment of $g$ and $s$ allows the same model to switch efficiency modes in different scenarios, which is highly practical for real-world deployments.

## Limitations & Future Work

1. **Inevitable information loss from core token generation**: Although dynamic, intra-group weighted pooling is still lossy compression. For tasks requiring precise citation of source details (such as exact numbers and code snippets), crucial information may be lost.
2. **Lack of theoretical guidance for choosing group size $g$**: The optimal value of $g$ can vary across different tasks and models, currently relying primarily on empirical tuning.
3. **Hardware dependency of Triton kernels**: The custom Triton implementation restricts generalizability across different GPU architectures and inference frameworks.
4. **Compatibility with GQA/MQA**: The paper does not explicitly discuss the performance of CCA-Attention under grouped-query attention (GQA) or multi-query attention (MQA) architectures.
5. **Explorable directions**: Extension of the core token concept to multimodal scenarios (visual token compression) or combination with KV cache eviction strategies (e.g., H2O) for further compression.

## Related Work & Insights

- **Efficient Attention Lineage**: Longformer → BigBird → LongLoRA → MInference → CCA, showing a trend from static sparse patterns to dynamic adaptation.
- **KV Cache Compression**: Complementary to KV cache eviction methods like H2O and SnapKV; CCA compresses at the attention mechanism level, whereas they compress from the cache management perspective.
- **Context Compression**: Conceptually similar to gist tokens (Mu et al.) and AutoCompressor (Chevalier et al.), but CCA's core tokens require no extra auxiliary networks or special tokens.
- **Insights**: The concept of core tokens can be extended to multimodal scenarios (e.g., visual token compression) and the expert selection problem within MoE architectures.

## Rating

- Novelty: ⭐⭐⭐⭐ — The dual-module concept of global pooling + local preservation holds some novelty, but represents a natural extension of sparse attention.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validated across multiple models and benchmarks with complete ablations, but lacks detailed analysis of more 128K+ scenarios.
- Writing Quality: ⭐⭐⭐⭐⭐ — Motivation is clear, mathematical derivations are rigorous, and illustrations are intuitive.
- Value: ⭐⭐⭐⭐ — The combination of plug-and-play capability and substantial efficiency gains possesses strong practical value, though more real-world deployment verification is needed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models](lacache_ladder-shaped_kv_caching_for_efficient_long-context_modeling_of_large_la.md)
- [\[ACL 2026\] Latent-Condensed Transformer for Efficient Long Context Modeling](../../ACL2026/model_compression/latent-condensed_transformer_for_efficient_long_context_modeling.md)
- [\[NeurIPS 2025\] Compress, Gather, and Recompute: REFORMing Long-Context Processing in Transformers](../../NeurIPS2025/model_compression/compress_gather_and_recompute_reforming_long-context_processing_in_transformers.md)
- [\[ICML 2025\] RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression](rocketkv_accelerating_long-context_llm_inference_via_two-stage_kv_cache_compress.md)
- [\[ICML 2025\] Context Tuning for In-Context Optimization](context_tuning_for_in-context_optimization.md)

</div>

<!-- RELATED:END -->

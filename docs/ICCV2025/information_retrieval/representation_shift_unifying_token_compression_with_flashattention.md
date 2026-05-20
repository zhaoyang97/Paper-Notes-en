---
title: >-
  [Paper Note] Representation Shift: Unifying Token Compression with FlashAttention
description: >-
  [ICCV 2025][Information Retrieval & RAG][Token Pruning] This paper proposes Representation Shift, a training-free and model-agnostic token importance metric that measures the magnitude of representational change before a…
tags:
  - "ICCV 2025"
  - "Information Retrieval & RAG"
  - "Token Pruning"
  - "FlashAttention"
  - "Representation Shift"
  - "Vision Transformer"
  - "Model Acceleration"
date: 2026-05-08
content_hash: ebdaf9c0417ed599
---

# Representation Shift: Unifying Token Compression with FlashAttention

**Conference**: ICCV 2025
**arXiv**: [2508.00367](https://arxiv.org/abs/2508.00367)  
**Code**: [https://github.com/mlvlab/Representation-Shift](https://github.com/mlvlab/Representation-Shift)  
**Area**: Information Retrieval
**Keywords**: Token Pruning, FlashAttention, Representation Shift, Vision Transformer, Model Acceleration

## TL;DR

This paper proposes Representation Shift, a training-free and model-agnostic token importance metric that measures the magnitude of representational change before and after a network layer, enabling — for the first time — compatibility between token compression and FlashAttention, achieving up to 5.5× speedup on video understanding and image classification tasks.

## Background & Motivation

Existing token compression methods primarily rely on attention maps to assess token importance, e.g., by using the attention scores of the class token or globally averaged attention scores. However, FlashAttention avoids explicitly materializing the attention map to reduce HBM memory accesses, making these methods incompatible with FlashAttention. This creates a fundamental conflict: FlashAttention yields substantial speedups (1.5× on DeiT-S, 2.7× on UMT-B), yet cannot be combined with training-free token compression methods. Additionally, some learnable token compression methods require fine-tuning, limiting their generalizability.

The core motivation is to identify a token importance metric that (1) does not depend on attention maps, (2) requires no additional training, and (3) is simultaneously compatible with FlashAttention and diverse architectures (Transformer, CNN, SSM).

## Method

### Overall Architecture

The core idea is elegantly simple: within a given network layer, the representational change (i.e., Representation Shift) of each token before and after the layer transformation is computed. Tokens with smaller shifts are regarded as redundant and pruned. Since only the layer input and output are required — with no dependence on attention maps — the method is naturally compatible with FlashAttention.

### Key Designs

1. **Definition of Representation Shift**: Given input tokens $\mathbf{x} \in \mathbb{R}^{L \times C}$ and a layer transformation $F(\cdot)$, the importance score is defined as $s = \Delta\mathbf{x} = \mathcal{D}(F(\mathbf{x}), \mathbf{x})$, where $\mathcal{D}$ is a distance metric. The underlying assumption is that informative tokens tend to exhibit larger representational changes, as the network amplifies their core information or suppresses redundant signals, whereas tokens with smaller changes are likely less relevant to the target task.

2. **Operation Choice**: The authors compare three computation sites: (i) through the Attention layer, (ii) through the MLP layer, and (iii) through the entire Attention Block. Experiments show that **the MLP layer yields the best representation shift**. This is because the Attention layer is responsible for cross-token information exchange and produces diffuse transformations, whereas the MLP operates independently on each token, generating more discriminative representational changes that capture token-specific contributions. The final formulation is $\Delta\mathbf{x} = \|\text{MLP}(\text{LN}(\mathbf{x}')) - \mathbf{x}'\|_2$.

3. **Distance Metric**: L2 norm, L1 norm, and cosine distance are compared. The **L2 distance** proves most stable and robust across layers and models. Cosine similarity degrades in deeper layers; L1 shows a slight advantage in early layers but is generally inferior to L2. L2 distance is therefore adopted as the default metric.

### Loss & Training

The method is entirely **training-free** and introduces no learnable parameters. Token pruning is applied directly at inference: after computing Representation Shift scores at a designated layer, tokens with the lowest scores are discarded. The method can also be combined with existing token merging approaches such as vid-TLDR by simply replacing their importance metric.

## Key Experimental Results

### Main Results

**Video-Text Retrieval (UMT, 7 benchmarks)**

| Model | Method | Throughput (vid/s) | Speedup | MSRVTT R@1 | ActivityNet R@1 | DiDeMo R@1 |
|------|------|---------|--------|-----------|----------------|-----------|
| UMT-B | Base | 32 | - | 50.0 | 57.2 | 62.1 |
| UMT-B | Attn | 57 | 1.78× | 47.6 | 54.2 | 57.7 |
| UMT-B | **Ours** | **175** | **5.47×** | 48.0 | 50.3 | 56.9 |
| UMT-L | Base | 12 | - | 58.7 | 65.6 | 70.8 |
| UMT-L | Attn | 23 | 1.91× | 50.2 | 53.2 | 58.2 |
| UMT-L | **Ours** | **66** | **5.50×** | **56.5** | **62.9** | **67.3** |

**Image Classification (ImageNet1K, DeiT)**

| Model | Method | Acc-1 | Throughput | GFLOPs |
|------|------|-------|--------|--------|
| DeiT-S | Base | 79.8 | 3002 | 4.6 |
| DeiT-S | Attn | 72.1 | 4844 | 3.0 |
| DeiT-S | **Ours** | **77.8** | **5948** | 3.0 |
| DeiT-B | Base | 81.8 | 1037 | 17.6 |
| DeiT-B | Attn | 76.9 | 2065 | 11.5 |
| DeiT-B | **Ours** | **79.6** | **2428** | 11.5 |

### Ablation Study

**Operation and Distance Metric Ablation (DeiT-S, ImageNet1K)**

| Operation | Acc after Layer 0 Pruning | Acc after Layer 4 Pruning | Acc after Layer 8 Pruning |
|---------|----------------|----------------|----------------|
| Attention | ~79.5 | ~76.5 | ~73.0 |
| Entire Block | ~79.5 | ~77.5 | ~74.0 |
| **MLP** | **~79.5** | **~78.0** | **~75.0** |

| Distance Metric | Layer 0 | Layer 4 | Layer 8 |
|---------|------|------|------|
| Cosine | ~79.5 | ~76.0 | ~72.0 |
| L1 | ~79.5 | ~77.5 | ~74.5 |
| **L2** | **~79.5** | **~78.0** | **~75.0** |

**Generalization to CNN and SSM (ImageNet1K)**

| Model | Method | Acc | Throughput | GFLOPs |
|------|------|-----|--------|--------|
| ResNet-34 | Base | 73.2 | 5811 | 3.7 |
| ResNet-34 | Line-wise | 72.8 | 7112 | 2.5 |
| ResNet-50 | Base | 76.1 | 2927 | 4.1 |
| ResNet-50 | Line-wise | **76.4** | 3553 | 2.7 |
| ViM-T | Base | 76.1 | 1603 | 1.5 |
| ViM-T | Ours | 75.5 | 1754 | 1.3 |

### Key Findings

- On UMT-L, Representation Shift outperforms attention-based pruning by an average of 7.2% in R@1 while achieving approximately 2.8× higher throughput.
- UMT-L + Representation Shift (66 vid/s) is both faster and more accurate than the UMT-B baseline (32 vid/s), demonstrating that a large model with token compression can outperform a smaller model used directly.
- Token compression is extended to CNNs (ResNet) and SSMs (ViM) for the first time; notably, line-wise pruning on ResNet-50 even improves accuracy (76.1 → 76.4).

## Highlights & Insights

- **Extreme simplicity**: The core idea reduces to a single formula — computing the L2 distance across the MLP layer. No additional parameters, no training, no attention maps, and negligible computational overhead.
- **First unification of FlashAttention and token compression**: The paper resolves the long-standing incompatibility between two major acceleration paradigms, yielding multiplicative speedup when combined.
- **Model-agnostic generality**: Applicable beyond ViT to CNNs and SSMs, making Representation Shift a universal token importance metric.
- **Interpretable visualization**: Representation shift maps computed at intermediate DeiT layers naturally highlight foreground objects, resembling saliency detection.

## Limitations & Future Work

- On certain video benchmarks (MSVD, ActivityNet) with UMT-B, non-trivial performance degradation is observed, suggesting that pruning ratios and layer selection need task-specific tuning.
- Token pruning in CNNs is constrained to row/column-level pruning due to the 2D grid requirement of convolution, offering less flexibility than ViT-based pruning.
- Deeper integration of Representation Shift with token merging methods (e.g., ToMe) remains unexplored.
- Applicability to multimodal large language models (e.g., LLaVA) has not been validated.

## Related Work & Insights

- Compared to attention-based methods such as EViT and BAT, Representation Shift does not rely on attention maps and thus applies to a broader range of architectures.
- When combined with vid-TLDR by replacing its importance metric, the method simultaneously benefits from token merging and FlashAttention acceleration.
- This work opens a new direction for large model inference acceleration: a similar approach could be explored for assessing token importance in the KV cache of LLMs.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Minimalist yet effective; first to unify FlashAttention and token compression
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers video retrieval, video QA, and image classification; validated across ViT, CNN, and SSM architectures
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with thorough ablations and rich figures
- **Value**: ⭐⭐⭐⭐⭐ Highly practical; applicable to existing models with minimal code changes

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Scaling Language-Centric Omnimodal Representation Learning](../../NeurIPS2025/information_retrieval/scaling_language-centric_omnimodal_representation_learning.md)
- [\[ICLR 2026\] Token-Guard: Towards Token-Level Hallucination Control via Self-Checking Decoding](../../ICLR2026/information_retrieval/token-guard_towards_token-level_hallucination_control_via_self-checking_decoding.md)
- [\[NeurIPS 2025\] HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation](../../NeurIPS2025/information_retrieval/hypergraphrag_retrieval-augmented_generation_via_hypergraph-structured_knowledge.md)
- [\[ACL 2026\] CodePromptZip: Code-specific Prompt Compression for Retrieval-Augmented Generation in Coding Tasks with LMs](../../ACL2026/information_retrieval/codepromptzip_code-specific_prompt_compression_for_retrieval-augmented_generatio.md)
- [\[ICLR 2026\] TokMem: One-Token Procedural Memory for Large Language Models](../../ICLR2026/information_retrieval/tokmem_one-token_procedural_memory_for_large_language_models.md)

</div>

<!-- RELATED:END -->

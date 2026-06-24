---
title: >-
  [Paper Note] Efficient Length-Generalizable Attention via Causal Retrieval for Long-Context Language Modeling
description: >-
  [ICML 2025][LLM Efficiency][Length Generalization] This paper proposes the Grouped Cross-Attention (GCA) mechanism, which integrates chunk-level causal retrieval into the attention mechanism to achieve an end-to-end learnable retriever. The constructed Differentiable Retrieval-based Transformer (DRT) achieves near-perfect accuracy on the passkey retrieval test with a 16M context, achieving length generalization up to 1000 times the training length.
tags:
  - "ICML 2025"
  - "LLM Efficiency"
  - "Length Generalization"
  - "Grouped Cross-Attention"
  - "Causal Retrieval"
  - "Long-Context Modeling"
  - "Differentiable Retrieval"
date: 2026-05-08
content_hash: 59f6302ffff46398
---

# Efficient Length-Generalizable Attention via Causal Retrieval for Long-Context Language Modeling

**Conference**: ICML 2025  
**arXiv**: [2410.01651](https://arxiv.org/abs/2410.01651)  
**Code**: [https://github.com/ant-research/long-context-modeling](https://github.com/ant-research/long-context-modeling)  
**Area**: LLM Efficiency  
**Keywords**: Length Generalization, Grouped Cross-Attention, Causal Retrieval, Long-Context Modeling, Differentiable Retrieval

## TL;DR
This paper proposes the Grouped Cross-Attention (GCA) mechanism, which integrates chunk-level causal retrieval into the attention mechanism to achieve an end-to-end learnable retriever. The constructed Differentiable Retrieval-based Transformer (DRT) achieves near-perfect accuracy on the passkey retrieval test with a 16M context, achieving length generalization up to 1000 times the training length.

## Background & Motivation
**Background**: Transformers perform exceptionally well in NLP, but processing long contexts beyond the pre-training window still faces the dual challenges of length generalization and quadratic complexity.

**Limitations of Prior Work**: Most long-range language models rely on expanding the attention window during post-training, which significantly increases computational and memory overhead. Sliding window methods can extrapolate but fail to capture long-range dependencies outside the window.

**Key Challenge**: Retrieval-Augmented Language Models (RLMs) can access long-range information with a fixed window. However, existing RLMs rely on pre-trained external retrievers (e.g., BM25, Contriever). The retrieved chunks might not be useful for causal language modeling, and the retriever cannot propagate gradients through the autoregressive loss.

**Goal**: How to enable the retriever to learn end-to-end to retrieve the historical chunks that are most helpful for predicting the next chunk.

**Key Insight**: Incorporate retrieval scores as weights in the next-token prediction (rather than using them solely for selection), thereby allowing them to receive gradients from the autoregressive loss.

**Core Idea**: GCA generalizes the token-to-token attention paradigm in self-attention to a chunk-to-chunk retrieval and fusion paradigm, where retrieval scores act as a soft choice to fuse the cross-attention outputs of each chunk.

## Method

### Overall Architecture
DRT consists of $N$ Transformer-like layers. The input sequence is equally divided into multiple chunks (size $S=64$), with a special LMK token inserted at the end of each chunk for summarization. The lower layers are standard Transformer layers, while the upper layers additionally include the GCA module. The upper layers are further divided into $G$ groups, with each group performing independent retrieval. The outputs of the lower layers are processed by a bidirectional Transformer encoder to generate chunk representations and landmark representations, which are shared across all upper layers.

### Key Designs

1. **Grouped Cross-Attention (GCA)**:

    - **Function**: Computes Cross-Attention for each token in the current chunk with respect to each retrieved chunk, and then fuses the results using the retrieval scores as weights.
    - **Mechanism**: GCA independently computes the CA output for each retrieved chunk, and then performs a weighted fusion using softmax-normalized retrieval scores.
    - **Design Motivation**: The key difference from Chunked Cross-Attention (CCA)—CCA concatenates all retrieved chunks and applies a unified softmax, with retrieval scores not participating in the computation; GCA applies softmax to each chunk independently, with retrieval scores acting as a soft choice involved in the prediction, thereby allowing gradient backpropagation.
    - KV projection parameters are shared across layers to save parameters and memory.

2. **Causal Retrieval**:

    - **Function**: Learns to retrieve historical chunks that most effectively reduce the autoregressive loss of the next chunk.
    - **Mechanism**: Computes relevance scores between the landmark representation of the current chunk and those of historical chunks, selecting the top-$k$.
    - Upper layers are split into $G$ groups, where each group retrieves independently; higher-level groups can perform multi-hop retrieval based on the retrieval results of the previous group.
    - **Design Motivation**: RPT relies on an external reference LM to tag high-quality chunks for training the retriever, which has poor scalability. GCA naturally embeds the retriever into the attention structure for end-to-end training.

3. **Gumbel Top-k Sampling**:

    - **Function**: Adds Gumbel noise to the retrieval scores before performing top-$k$ during training to balance exploration and exploitation.
    - **Mechanism**: High-scoring chunks are still the most likely to be selected, but low-scoring chunks also have opportunities to be explored.
    - **Design Motivation**: Pure top-$k$ selection may get stuck in local optima; Gumbel noise increases training diversity.

4. **Memory-Offloaded Inference**:

    - **Function**: Offloads historical chunk representations to CPU memory and loads them back to GPU during retrieval.
    - The GPU memory complexity is significantly reduced.
    - Each retrieval is triggered $G$ times only after generating $S$ tokens, resulting in minimal swapping overhead.

### Loss & Training
- Standard next-token prediction loss
- Sliding window self-attention ($W=512$) + top-$k$ retrieval ($K=8$, $S=64$), with an attention span of 512 tokens
- Triton-based hardware-aware GCA implementation
- Training complexity scales near-linearly due to chunk-level implementation

## Key Experimental Results

### Main Results (Language Modeling Perplexity, 350M Model, 16K Training/Evaluation)

| Model | Training Overhead | top-k | Window | PG19 valid | PG19 test | ArXiv valid | ArXiv test |
|------|---------|-------|------|-----------|-----------|-------------|------------|
| BaseLM (SW+ALiBi) | 1x | - | 512 | 14.55 | 13.68 | 3.06 | 3.06 |
| BaseLM (+2 layers) | 1.15x | - | 658 | 14.23 | 13.37 | 2.95 | 2.94 |
| Landmark Attn | 1.5x | 4 | 768 | 14.10 | 13.21 | 3.02 | 3.02 |
| DRT_ret x1 | 1.22x | 8 | 512 | 14.05 | 13.21 | 2.89 | 2.89 |
| **DRT_ret x2** | **1.24x** | 8 | 512 | **14.02** | **13.18** | **2.85** | **2.85** |

### Single Passkey Retrieval Accuracy (128M Model)

| Model | 4K | 16K | 64K | 128K | 256K | 16M |
|------|------|------|------|-------|------|------|
| BaseLM (+2 layers) | 15.37 | 3.89 | 0.0 | - | - | - |
| Landmark Attn | 99.82 | 97.88 | 0.00 | 0.00 | - | - |
| **DRT_ret x1** | 98.50 | 98.59 | 100 | **100** | **100** | **100** |
| **DRT_ret x2** | 99.65 | 99.65 | 100 | **100** | **100** | **100** |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| w/o Triton | Training overhead 1.45x | Triton implementation reduces GCA overhead by ~16% |
| w/o Gumbel top-k | Slightly higher PPL (14.36 vs 14.05) | Gumbel noise effectively improves retrieval quality |
| w/ Contriever | PPL 14.55 vs 14.05 | End-to-end causal retrieval outperforms fixed external retrievers by a large margin |
| w/ random retriever | PPL 14.53 vs 14.05 | Random retrieval performs similarly to external retrievers |

### Key Findings
- DRT is the first attention mechanism to achieve perfect passkey retrieval at a context length of 16M (1000x of the training length).
- Multiple retrievals ($G=2$) significantly outperform single retrieval on 2-hop NIAH (88.52% vs 41.07%).
- End-to-end causal retrieval significantly outperforms the external retriever (Contriever), which behaves similarly to or even worse than random retrieval.
- Inference Efficiency: DRT's inference time and memory overhead are an order of magnitude lower than those of Landmark Attn.

## Highlights & Insights
- GCA elegantly embeds the retrieval operation into the attention mechanism, solving the core bottleneck of the inability to propagate gradients to retrieval scores.
- Chunk-level retrieval, rather than token-level retrieval, is key to length generalization, as chunks provide richer semantic information.
- Case studies validate the concept of causal retrieval: the model retrieves not only semantically similar content but also information beneficial for predicting the next chunk.
- Achieving 1000x length generalization is a significant milestone.

## Limitations & Future Work
- Currently validated only on 128M–350M models; its effectiveness on larger language models remains to be explored.
- The chunk size $S$ is fixed at 64; adaptive chunk partitioning may yield better results.
- Although CPU offloading is feasible, its impact on throughput during large-batch inference needs to be evaluated.
- There is still a performance gap on 2-hop NIAH in short contexts (1K) compared to Landmark Attn (41% vs 91%).

## Related Work & Insights
- Contrast with RPT: RPT requires an external reference LM to annotate chunks for retriever training, while DRT learns in an end-to-end manner.
- Contrast with Landmark Attention: LA performs top-$k$ retrieval at every token and every layer, which incurs high computational overhead and fails to extrapolate.
- The softmax-off-by-one trick in GCA allows tokens to ignore all retrieved chunks, increasing flexibility.
- The concept of causal retrieval can be extended to external knowledge base retrieval scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Elegant design of the GCA mechanism and novel concept of causal retrieval)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Multiple datasets, multiple tasks, complete ablations, and case studies)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure and intuitive illustrations)
- Value: ⭐⭐⭐⭐⭐ (1000x length generalization is a major breakthrough)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Long-Short Alignment for Effective Long-Context Modeling in LLMs](long-short_alignment_for_effective_long-context_modeling_in_llms.md)
- [\[ICML 2025\] Curse of High Dimensionality Issue in Transformer for Long-context Modeling](curse_of_high_dimensionality_issue_in_transformer_for_long-context_modeling.md)
- [\[ACL 2025\] SEAL: Scaling to Emphasize Attention for Long-Context Retrieval](../../ACL2025/llm_efficiency/seal_scaling_to_emphasize_attention_for_long-context_retrieval.md)
- [\[ACL 2025\] Literary Evidence Retrieval via Long-Context Language Models](../../ACL2025/llm_efficiency/literary_evidence_retrieval_via_long-context_language_models.md)
- [\[ACL 2025\] Squeezed Attention: Accelerating Long Context Length LLM Inference](../../ACL2025/llm_efficiency/squeezed_attention_accelerating_long_context_length_llm_inference.md)

</div>

<!-- RELATED:END -->

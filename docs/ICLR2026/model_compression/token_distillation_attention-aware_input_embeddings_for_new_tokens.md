---
title: >-
  [Paper Note] Token Distillation: Attention-Aware Input Embeddings for New Tokens
description: >-
  [ICLR 2026][Model Compression][Vocabulary Expansion] This paper proposes Token Distillation, a method that distills multi-subword interaction information encoded across all Transformer layers into a single token embedding, enabling high-quality initialization of new token embeddings without requiring a pretrained hypernetwork and outperforming existing approaches.
tags:
  - ICLR 2026
  - Model Compression
  - Vocabulary Expansion
  - Token Embedding Initialization
  - Knowledge Distillation
  - Domain Adaptation
  - Language Adaptation
date: 2026-05-08
content_hash: e1e7d81cbe62469c
---

# Token Distillation: Attention-Aware Input Embeddings for New Tokens

**Conference**: ICLR 2026
**arXiv**: [2505.20133](https://arxiv.org/abs/2505.20133)
**Code**: [https://github.com/konstantinjdobler/token-distillation](https://github.com/konstantinjdobler/token-distillation)
**Area**: Model Compression
**Keywords**: Vocabulary Expansion, Token Embedding Initialization, Knowledge Distillation, Domain Adaptation, Language Adaptation

## TL;DR

This paper proposes Token Distillation, a method that distills multi-subword interaction information encoded across all Transformer layers into a single token embedding, enabling high-quality initialization of new token embeddings without requiring a pretrained hypernetwork and outperforming existing approaches.

## Background & Motivation

- **Static vocabulary problem**: Pretrained language models rely on fixed tokenizers that over-segment domain-specific or novel-language vocabulary, leading to performance degradation and increased computational overhead.
- **Fundamental limitations of existing initialization methods**:
    - The subword averaging approach exploits only the embedding matrix, ignoring functional knowledge encoded in Transformer layers.
    - For example, the individual subword embeddings of `<_pal><at><able>` do not carry the semantics of `<_palatable>`.
    - The semantics of multi-subword spans are progressively constructed through the attention and feed-forward layers during contextualization (i.e., neural detokenization).
- **Core insight**: Effective new token embeddings must capture information stored across all Transformer layers, not merely the embedding matrix.

## Method

### Overall Architecture

Given a new token $t^{\star}$ and its original subword decomposition $[t_1, \dots, t_n]$, Token Distillation directly optimizes the new embedding $\mathbf{e}^{\star}$ such that the hidden states produced when using the single new token closely match those produced when using the original multi-subword sequence.

### Key Design: Hidden-State Distillation Objective

The optimization minimizes the MSE of hidden states at a specified layer:

$$\min_{\mathbf{e}^{\star} \in \mathbb{R}^d} \mathbb{E}_{s \sim S} \left[ \frac{1}{|\mathcal{M}(s_\tau, s_{\tau^{\star}})|} \sum_{(i,j) \in \mathcal{M}(s_\tau, s_{\tau^{\star}})} \left\| \mathcal{H}_{\mathbf{e}^{\star}}^{(l)}(s_{\tau^{\star}})_i - \mathcal{H}^{(l)}(s_\tau)_j \right\|_2^2 \right]$$

- $\mathcal{H}^{(l)}(s_\tau)$: layer-$l$ hidden states under the original tokenization (teacher).
- $\mathcal{H}_{\mathbf{e}^{\star}}^{(l)}(s_{\tau^{\star}})$: hidden states under the new token embedding (student).
- $\mathcal{M}(s_\tau, s_{\tau^{\star}})$: alignment position mapping, restricted to positions that attend to the new token.
- In practice, the final-layer hidden states are used.

### Context Retrieval

Two strategies are employed to obtain training contexts:

1. **Primary method**: Efficient retrieval of segments containing the target token from a corpus using the Aho-Corasick algorithm.
2. **Fallback**: Prompting the model with the new token to generate text containing the target word.

### Output Embedding Handling

- Token Distillation optimizes only input embeddings, since new tokens fall outside the teacher model's prediction vocabulary.
- Output embeddings can be further trained with a next-token prediction (NTP) objective or initialized to zero vectors.
- The method can be combined with $\alpha$NTP, which dynamically down-weights the NTP loss to avoid interference.

### Efficiency

- Only 25 context segments per new token are required.
- Contexts are truncated to 50 tokens in length.
- 2,500 new tokens can be initialized on a single GPU in under 10 minutes.

## Key Experimental Results

### Main Results: Biomedical Domain Adaptation (Average over 8 Models)

| Method | Average Accuracy |
|--------|-----------------|
| Original tokenization | 66.5 |
| Random | 57.5 |
| Subword Mean | 60.8 |
| NTP (new embeddings only) | 63.0 |
| ZeTT (pretrained hypernetwork) | — (partial model coverage) |
| **Token Distillation** | **64.6** |
| **Token Distillation + αNTP** | **64.7** |

### Definition Generation Quality (LLM-as-Judge)

| Method | Similarity Avg | Correctness Avg |
|--------|---------------|-----------------|
| Random | 0.0 | 0.1 |
| Subword Mean | 16.6 | 18.6 |
| NTP | 52.0 | 59.4 |
| ZeTT | — | — |
| **Token Distillation** | **68.5** | **74.4** |
| **Token Distillation + αNTP** | **76.7** | **83.3** |

### French Language Adaptation

| Method | Mistral-7B | Llama3-8B | Llama3-8B-i | Avg |
|--------|-----------|-----------|-------------|-----|
| Original | 69.5 | 69.4 | 72.1 | 73.2 |
| Subword Mean | 56.3 | 58.4 | 61.7 | 61.5 |
| NTP | 64.7 | 67.0 | 70.1 | 70.8 |
| **Token Distillation** | **68.5** | **68.9** | **72.9** | **72.9** |

### Key Findings

- Token Distillation consistently outperforms NTP and subword averaging across all 8 models, and surpasses ZeTT without requiring hypernetwork pretraining.
- Definition generation experiments confirm that distilled embeddings achieve higher semantic quality and completeness.
- Freezing original embeddings and updating only the new embeddings (NTP variant) outperforms full embedding fine-tuning.
- Tied-embedding models (e.g., Llama3.2-3B) may exhibit norm explosion; adding $\alpha$NTP regularization mitigates this issue.
- In French language adaptation, Token Distillation can even surpass the original tokenization baseline (Llama3-8B-i).

## Highlights & Insights

- **Theoretically well-motivated**: Identifies the fundamental flaw of existing methods in ignoring Transformer-layer knowledge.
- **Extremely lightweight**: Requires only 25 text segments per token and processes 2,500 new tokens in 10 minutes.
- **No auxiliary model required**: Relies solely on the target model itself, with no pretrained hypernetwork needed.
- **Broad model coverage**: Evaluated across 3B–8B models, base/instruct variants, and tied/untied embedding configurations.

## Limitations & Future Work

- Only input embeddings are learned; output embeddings require separate handling.
- Norm instability may arise for tied-embedding models.
- The choice of the final-layer hidden states as the distillation target has not been thoroughly explored for optimality.
- A small amount of in-context text containing each new token is required, limiting applicability in fully zero-resource scenarios.
- Compared to hypernetwork-based methods, initialization is slower at inference time due to gradient-based optimization rather than a single forward pass.

## Related Work & Insights

- **Gradient-free methods**: Subword averaging, weighted linear combinations (WECHSEL, FVT, etc.) — all neglect Transformer-layer knowledge.
- **Gradient-based methods**: NTP embedding tuning and hypernetwork ZeTT — the former has an indirect optimization objective, while the latter requires expensive pretraining.
- **Token-to-Words**: Uses PatchScopes to identify the layer at which subwords are unified into a single representation, but requires training a mapping module.
- **Token Distillation**: Requires no layer localization and directly captures information from all layers through distillation.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ★★★★☆ |
| Theoretical Depth | ★★★★☆ |
| Experimental Thoroughness | ★★★★★ |
| Value | ★★★★☆ |
| Writing Quality | ★★★★★ |

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] FASA: Frequency-aware Sparse Attention](fasa_frequency-aware_sparse_attention.md)
- [\[ICLR 2026\] TurboBoA: Faster and Exact Attention-aware Quantization without Backpropagation](turboboa_faster_and_exact_attention-aware_quantization_without_backpropagation.md)
- [\[ICLR 2026\] AgilePruner: An Empirical Study of Attention and Diversity for Adaptive Visual Token Pruning in LVLMs](agilepruner_an_empirical_study_of_attention_and_diversity_for_adaptive_visual_to.md)
- [\[NeurIPS 2025\] A Token is Worth over 1,000 Tokens: Efficient Knowledge Distillation through Low-Rank Clone](../../NeurIPS2025/model_compression/a_token_is_worth_over_1000_tokens_efficient_knowledge_distillation_through_low-r.md)
- [\[ICLR 2026\] TiTok: Transfer Token-level Knowledge via Contrastive Excess to Transplant LoRA](titok_transfer_token-level_knowledge_via_contrastive_excess_to_transplant_lora.md)

<!-- RELATED:END -->

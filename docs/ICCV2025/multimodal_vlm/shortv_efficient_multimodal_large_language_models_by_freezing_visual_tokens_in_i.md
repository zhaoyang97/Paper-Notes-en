---
title: >-
  [Paper Note] ShortV: Efficient Multimodal Large Language Models by Freezing Visual Tokens in Ineffective Layers
description: >-
  [ICCV 2025][Multimodal VLM][multimodal large language models] This work identifies significant **layer-level redundancy** in MLLMs—most layers contribute minimally to the transformation of visual tokens—and proposes ShortV: freezing visual tokens (skipping their attention and FFN computations) in approximately 60% of layers. On LLaVA-NeXT-13B, this achieves a 50% reduction in FLOPs with negligible performance degradation. The method is training-free and orthogonal to token pruning approaches, allowing them to be combined.
tags:
  - ICCV 2025
  - Multimodal VLM
  - multimodal large language models
  - inference efficiency
  - layer redundancy
  - visual token freezing
  - training-free
  - MLLM acceleration
date: 2026-05-08
content_hash: a8147dead295d6db
---

# ShortV: Efficient Multimodal Large Language Models by Freezing Visual Tokens in Ineffective Layers

**Conference**: ICCV 2025
**arXiv**: 2504.00502
**Code**: [https://github.com/icip-cas/ShortV](https://github.com/icip-cas/ShortV)
**Area**: Multimodal VLM
**Keywords**: multimodal large language models, inference efficiency, layer redundancy, visual token freezing, training-free, MLLM acceleration

## TL;DR

This work identifies significant **layer-level redundancy** in MLLMs—most layers contribute minimally to the transformation of visual tokens—and proposes ShortV: freezing visual tokens (skipping their attention and FFN computations) in approximately 60% of layers. On LLaVA-NeXT-13B, this achieves a 50% reduction in FLOPs with negligible performance degradation. The method is training-free and orthogonal to token pruning approaches, allowing them to be combined.

## Background & Motivation

The computational overhead of MLLMs (e.g., LLaVA-1.5, LLaVA-NeXT) stems from two primary factors:

**Large LLM backbone**: 7B–13B parameters

**Long visual token sequences**: 576–2880 tokens per image

**Prior work perspective**: Methods such as FastV identify redundancy from the **token dimension**—many visual tokens are deemed unimportant and are pruned. These approaches reduce the number of tokens.

**Novel perspective of this paper**: Redundancy is examined from the **layer dimension**—does a given visual token truly need to be updated at every layer?

**Key finding**: In text-only LLMs, approximately 25% of layers are ineffective for text tokens. However, in MLLMs, the **modality gap**—the distributional discrepancy between visual and text tokens in the embedding space—may lead the model to adopt fundamentally different processing strategies for each modality.

## Method

### Layer Contribution (LC) Metric

To quantify the contribution of a given layer to a specific token type, the authors propose the LC metric:

1. Replace layer $i$ with a **sparse layer** in which target tokens (e.g., visual tokens) are frozen—their hidden states remain unchanged.
2. In the sparse layer, frozen tokens do not participate in query computation and bypass the FFN.
3. The KL divergence between the original and modified model outputs is computed to measure the impact of freezing:

$$LC_i^X = KL(logits(M), logits(\mathcal{M}_i^X))$$

where $M$ is the original model and $\mathcal{M}_i^X$ is the model with token $X$ frozen at layer $i$. A lower LC indicates that the layer is less important for that token type.

### Why Not Use Alternative Metrics?

- **Perplexity**: Unsuitable for visual tokens, since even complete removal of visual tokens allows MLLMs to generate plausible text (perplexity changes minimally), while visual task performance degrades severely.
- **Cosine similarity**: Measures only the similarity between a layer's input and output, ignoring the layer's position in the network—small transformations in early layers affect all subsequent layers, whereas the same transformation in later layers does not. Experiments confirm that cosine similarity systematically overestimates redundancy in shallow layers and underestimates it in deep layers.

### ShortV Approach

1. Construct a small calibration dataset (2,000 samples from Flickr30K and GQA).
2. Compute the LC score for each layer with respect to visual tokens.
3. Rank layers by LC score in ascending order and select the $N$ lowest-scoring layers.
4. Replace these layers with ShortV layers (visual tokens frozen).

**ShortV layer design** (as illustrated in Figure 4):
- Only **text tokens** pass through the $W_Q$ and $W_O$ projection matrices and the FFN.
- Visual tokens do not serve as queries (they do not attend to other tokens).
- Visual tokens bypass the FFN.
- Visual token keys and values are retained so that text tokens can still attend to visual tokens.

### FLOPs Analysis

Original layer: $FLOPs = 2(t+v)(4h+3m)h + 4(t+v)^2 h$

ShortV layer: $FLOPs^* = 2t(4h+3m)h + 4vh^2 + 4t(t+v)h$

Overall FLOPs ratio: $r = \frac{(L-N) \times FLOPs + N \times FLOPs^*}{L \times FLOPs}$

## Key Experimental Results

### Main Results: Comparison with Other Training-Free Methods

| Model | Method | TFLOPs | FLOPs Ratio | MME ↑ | MMBench ↑ | MMMU ↑ | MMStar ↑ | SEED ↑ | GQA ↑ | Flickr30K ↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5-7B | Vanilla | 8.5 | 100% | 1510.7 | 64.1 | 36.3 | 33.7 | 66.1 | 61.9 | 74.9 |
| LLaVA-1.5-7B | FastV (K=2,R=50%) | 4.9 | 58% | 1475.6 | 64.3 | 35.8 | 32.4 | 65.4 | 60.2 | 67.5 |
| LLaVA-1.5-7B | VTW (K=16) | 4.7 | 55% | 1497.0 | 64.0 | 36.1 | 32.8 | 66.2 | 55.1 | 44.5 |
| LLaVA-1.5-7B | **ShortV (N=19)** | **4.7** | **55%** | **1503.1** | **64.8** | **36.2** | **33.3** | **66.2** | **60.9** | **71.3** |
| LLaVA-NeXT-13B | Vanilla | 81.8 | 100% | 1570.0 | 69.3 | 35.9 | 39.9 | 71.9 | 65.7 | 66.7 |
| LLaVA-NeXT-13B | FastV (K=2,R=50%) | 42.1 | 51% | 1546.4 | 68.5 | 35.9 | 39.6 | 71.5 | 62.9 | 66.0 |
| LLaVA-NeXT-13B | VTW (K=20) | 41.7 | 51% | 1569.4 | 69.1 | 34.8 | 39.8 | 71.8 | 61.5 | 56.6 |
| LLaVA-NeXT-13B | **ShortV (N=24)** | **41.0** | **50%** | **1553.0** | **70.2** | **36.2** | **39.9** | **71.8** | **63.6** | **67.5** |

ShortV matches or outperforms all baselines across both models at comparable or lower computational cost. VTW exhibits severe degradation on Flickr30K (44.5–56.6), while ShortV incurs virtually no loss.

### Ablation Study: Layer Selection Strategies

| Strategy | FLOPs Ratio | MMBench | MMMU | SEED-Bench | GQA |
|---|---|---|---|---|---|
| Vanilla | 100% | 64.0 | 36.3 | 66.1 | 61.9 |
| Random | 55% | 58.4 | 33.6 | 60.5 | 56.1 |
| Cosine Similarity | 55% | 60.8 | 34.2 | 62.7 | 59.5 |
| **LC (Ours)** | **55%** | **64.8** | **36.2** | **66.2** | **60.9** |

The LC metric substantially outperforms both random selection and cosine similarity. While cosine similarity outperforms random selection, it still fails to match the original model's performance.

### Ablation Study: Freezing Different Token Types

| Frozen Tokens | MMBench | MMMU | SEED-Bench | GQA |
|---|---|---|---|---|
| None (Vanilla) | 64.0 | 36.3 | 66.1 | 61.9 |
| Text tokens | 2.1 | 23.7 | 8.9 | 2.9 |
| All tokens | 1.3 | 26.6 | 0.8 | 0.0 |
| Random tokens | 1.5 | 22.9 | 5.5 | 2.3 |
| **Visual tokens (Ours)** | **64.8** | **36.2** | **66.2** | **60.9** |

Freezing text tokens causes **catastrophic performance collapse** (GQA: 61.9 → 2.9), whereas freezing visual tokens has negligible impact—providing strong evidence that MLLM layers process visual and text tokens in fundamentally different ways.

### Compatibility: ShortV + FastV

| Method | FLOPs Ratio | MMBench | MMMU | SEED-Bench | GQA |
|---|---|---|---|---|---|
| Vanilla | 100% | 64.0 | 36.3 | 66.1 | 61.9 |
| FastV | 58% | 64.3 | 35.8 | 65.4 | 60.2 |
| ShortV | 55% | 64.8 | 36.2 | 66.2 | 60.9 |
| **ShortV + FastV** | **29%** | **64.2** | **37.1** | **65.1** | **59.3** |

Combining both methods reduces FLOPs to 29% while maintaining reasonable performance.

### Key Findings

1. **60% of layers can freeze visual tokens without performance degradation**, far exceeding the ~25% threshold observed for text tokens in text-only LLMs.
2. Practical speedup: ShortV (N=24) achieves **1.52× acceleration** on LLaVA-NeXT-13B, compared to only 1.31× for FastV.
3. Even with 80% of layers frozen, the model retains >90% of original performance.

## Highlights & Insights

1. **Modal asymmetry in layer-level redundancy**: This is a profound and somewhat unexpected finding—MLLM layers exhibit far greater redundancy for visual tokens than for text tokens, reflecting the low efficiency of current MLLM architectures in processing visual information.
2. **Design philosophy of the LC metric**: By directly measuring the impact of freezing on model output, the LC metric avoids the systematic biases inherent in proxy metrics such as cosine similarity.
3. **A new dimension orthogonal to token pruning**: Token pruning reduces the number of tokens; ShortV reduces the per-token computation. The two can be combined to achieve 29% FLOPs.
4. **Training-free with no parameter updates**: ShortV can be applied as a plug-and-play module to any MLLM.

## Limitations & Future Work

1. **Validation limited to the LLaVA series**: Other architectures such as Qwen-VL and InternVL have not been evaluated.
2. **Visual tokens retained as KV**: Although visual tokens do not serve as queries, their keys and values still participate in attention computation; this portion of computation is not saved.
3. **LC computation requires a small calibration dataset**: Although only 2,000 samples are needed, the process must be repeated for each new model.
4. **Evaluation restricted to image understanding tasks**: Longer-sequence scenarios such as video understanding and long-document comprehension remain unexplored.
5. **Implicit assumption of input-agnostic layer inefficiency**: The method uses a fixed layer selection regardless of input, yet different image types may require different layers for effective processing.

## Related Work & Insights

- **FastV**: Identifies token-level redundancy and applies pruning; ShortV identifies layer-level redundancy and applies freezing—combining these two complementary dimensions suggests substantial room for efficiency gains in MLLM visual processing.
- **LLM layer pruning**: Men et al. find that ~25% of layers can be removed in text-only LLMs, whereas ShortV shows ~60% of layers can be frozen for visual tokens in MLLMs—the modality gap induces substantially greater redundancy.
- **SAISA / NAAViT**: The ShortV layer design draws inspiration from architectures that integrate multimodal cross-attention within self-attention.
- **Implications for MLLM architecture design**: Given that most layers are unimportant for visual tokens, it may be worth designing architectures from the outset that process visual information in only a small subset of layers.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: ⭐⭐⭐⭐ — The LC metric and the finding of modal asymmetry in layer-level redundancy are valuable contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive ablations covering layer selection strategies, token type freezing, and method compatibility, with multi-model validation.
- **Value**: ⭐⭐⭐⭐⭐ — Training-free, practical speedup of 1.5×+, and composable with FastV.
- **Writing Quality**: ⭐⭐⭐⭐ — The argumentation is logically coherent, progressing seamlessly from discovery to method to validation.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] CompCap: Improving Multimodal Large Language Models with Composite Captions](compcap_improving_multimodal_large_language_models_with_composite_captions.md)
- [\[ICCV 2025\] LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models](llavaprumerge_adaptive_token_reduction_for_efficient_large_m.md)
- [\[ICCV 2025\] FALCON: Resolving Visual Redundancy and Fragmentation in High-resolution Multimodal Large Language Models via Visual Registers](falcon_resolving_visual_redundancy_and_fragmentation_in_high.md)
- [\[ICCV 2025\] BASIC: Boosting Visual Alignment with Intrinsic Refined Embeddings in Multimodal Large Language Models](basic_boosting_visual_alignment_with_intrinsic_refined_embeddings_in_multimodal_.md)
- [\[ICCV 2025\] SimpleVQA: Multimodal Factuality Evaluation for Multimodal Large Language Models](simplevqa_multimodal_factuality_evaluation_for_multimodal_large_language_models.md)

<!-- RELATED:END -->

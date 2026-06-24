---
title: >-
  [Paper Note] RoToR: Towards More Reliable Responses for Order-Invariant Inputs
description: >-
  [ACL 2025][positional bias] Proposes RoToR, a zero-shot, order-invariant language model based on global ordering and circular position encoding allocation. It achieves stable order invariance by minimizing position ID modifications and designs a Selective Routing mechanism to adaptively handle mixed input types.
tags:
  - "ACL 2025"
  - "positional bias"
  - "order invariance"
  - "positional encoding"
  - "causal language models"
  - "selective routing"
date: 2026-05-08
content_hash: d163a45d670edc56
---

# RoToR: Towards More Reliable Responses for Order-Invariant Inputs

**Conference**: ACL 2025  
**arXiv**: [2502.08662](https://arxiv.org/abs/2502.08662)  
**Code**: Yes ([https://github.com/soyoung97/RoToR](https://github.com/soyoung97/RoToR))  
**Area**: Others  
**Keywords**: positional bias, order invariance, positional encoding, causal language models, selective routing

## TL;DR

Proposes RoToR, a zero-shot, order-invariant language model based on global ordering and circular position encoding allocation. It achieves stable order invariance by minimizing position ID modifications and designs a Selective Routing mechanism to adaptively handle mixed input types.

## Background & Motivation

Language models are highly sensitive to input order, but in many practical scenarios, the order of list-like inputs (such as table rows, multiple-choice options, and retrieved document collections) should be irrelevant. This "positional bias" problem is widely recognized:

- In LLM-as-a-judge scenarios, models show up to a 75% preference for the first response.
- On MMLU, simply changing the order of options can shift model rankings by up to 8 positions.
- The "lost-in-the-middle" phenomenon: information in the middle positions is heavily ignored.

Existing zero-shot order-invariant methods suffer from two key limitations:

**Training-inference distribution mismatch**: PCW completely isolates attention between segments, while PINE dynamically reallocates position IDs for each query token. This results in a position encoding allocation that deviates significantly from the pre-training distribution.

**Inability to adapt to mixed inputs**: Practical questions (such as MMLU) contain both order-insensitive options (e.g., A, B, C) and order-sensitive options (e.g., "None of the above"). Existing methods apply a single strategy to both.

The core idea of RoToR is to: **replace the query-wise dynamic sorting of PINE with a global ordering + circular allocation**, significantly reducing position ID perturbations; meanwhile, selective routing is introduced to adaptively handle both order-sensitive and order-insensitive inputs.

## Method

### Overall Architecture

RoToR consists of two stages:
1. **RoToR Core**: A new position ID allocation scheme that uses global ordering and circular arrangement to achieve order invariance.
2. **Selective Routing**: Adaptively selects the output based on the confidence of the original model and the order-invariant model.

### Key Designs

1. **Global Ordering**

    - **Function**: Determines a unified permutation order for all input segments, rather than PINE's query-wise permutation.
    - **Core Idea**:
        - Provides three global ordering algorithms:
       - **Lexicographical**: Based on the lexicographical order of token sequences, with minimal overhead.
       - **MonoT5**: Uses a pointwise reranker to sort based on relevance to the question.
       - **Frequency**: Normalized sorting based on inverse token frequency.
        - The sorting result is shared across all query tokens, all layers, and all attention heads.
    - **Design Motivation**:
        - PINE recalculates sorting for each query token, leading to $O(n^2d)$ extra computation and frequent position ID changes.
        - Global ordering only needs to be performed once, reducing complexity to $O(nk\log k)$, and consistent position ID allocation minimizes distribution shift.

2. **Circular Arrangement**

    - **Function**: Simulates bidirectional attention in causal LMs, allowing each segment to "see" all other segments.
    - **Core Idea**:
        - Given the global ordering A→B→O→K→G, construct a directed circular graph.
        - When segment B is used as a query, put B at the end according to the circular order: O→K→G→A→B.
        - When segment K is used as a query: G→A→B→O→K.
        - Key: All suffix and generated tokens are concatenated using the front and back parts of the global ordering, **no longer changing position IDs token-by-token**.
    - **Design Motivation**:
        - In causal attention, tokens at the end of the sequence can see all preceding tokens.
        - Circular arrangement allows each segment to take turns in the "last position," achieving de facto bidirectional access.
        - Unlike PINE, the position IDs of suffix tokens remain constant, significantly reducing OOD risk.

3. **Selective Routing**

    - **Function**: Adaptively chooses between using the output of the original model or the order-invariant model.
    - **Core Idea**:
        - The original model and the RoToR model generate answers and their confidences (maximum token probabilities) for the same input, respectively.
        - If original model confidence + bias α > RoToR confidence, the answer from the original model is chosen; otherwise, RoToR is selected.
        - α = 0.2 (determined via validation set search), leaning slightly towards the original model.
    - **Design Motivation**:
        - In practical tasks (e.g., MMLU), some options are order-sensitive (e.g., "None of the above").
        - The order-invariant model might perform worse on these options.
        - Confidence-based routing allows adaptive selection of the most suitable model.

### Computational Complexity Analysis

| Method | Extra Computational Overhead |
|------|------------|
| PINE | $O(n^2d + nk\log k)$ (Recalculates RoPE-free attention + sorting for each query) |
| RoToR (Lexicographical) | $O(nk\log k)$ (Single global sorting) |
| RoToR (Radix Sort Optimized) | $O(nk)$ |

## Key Experimental Results

### Main Results (Lost in the Middle Benchmark, best_subspan_em %)

| Method | ndoc=10 | ndoc=20 | ndoc=30 |
|------|---------|---------|---------|
| **Llama-3.1-8B-Instruct** | | | |
| Original | 50.2~54.7 | 51.0~54.8 | 43.5~56.8 |
| PCW | 11.9~12.4 | 3.7~4.0 | 1.8~2.3 |
| Set-Based Prompting | 42.5 | 26.3 | 14.1 |
| PINE | 58.6~59.0 | 55.5~56.2 | 53.7~54.8 |
| **RoToR-lexical** | **61.4~61.6** | **59.6~61.4** | **59.0~59.5** |
| RoToR-MonoT5 | 61.2~61.4 | 60.7~61.2 | 60.7~60.9 |
| **Llama-3.1-70B-Instruct** | | | |
| Original | 65.7~66.2 | 64.3~66.2 | — |
| PINE | 67.5~67.9 | 65.5~65.9 | — |
| **RoToR** | **69.3~69.6** | **67.6~67.9** | — |

### KGQA Experiment (N=30 segments, best_subspan_em %)

| Method | Llama-8B Acc. | Llama-70B Acc. | Qwen-4B Acc. | Qwen-7B Acc. |
|------|-------------|---------------|-------------|-------------|
| Original | 50.2 | 61.6 | 30.7 | 31.5 |
| PINE | 51.5 | 63.1 | 31.6 | 32.3 |
| **RoToR** | **53.1** | **63.6** | **32.0** | **34.3** |
| RoToR-MonoT5 | 51.6 | — | 32.3 | 32.9 |

### Key Findings

1. **RoToR consistently outperforms PINE across all models and settings**: improving by an average of 2-5 percentage points on the LitM benchmark.
2. **Excellent order invariance**: After shuffling segment order, RoToR exhibits extremely small standard deviation (0.02-0.11), which is significantly better than the Original model (0.07-0.75).
3. **Simple lexicographical order is sufficient**: It does not require complex MonoT5 sorting; RoToR-lexical already brings significant benefits.
4. **Computational overhead is far lower than PINE**: Eliminates the $O(n^2d)$ term, making the advantage more pronounced as the number of segments k increases.
5. **PCW and Set-Based Prompting almost fail when the number of segments increases**: At ndoc=30, PCW obtains only 2%.
6. **Selective routing is effective**: Helps handle order-sensitive special options in MMLU.

## Highlights & Insights

- **Clever naming**: RoToR is a palindrome, echoing the theme of "order invariance," while also hinting at "Rotary."
- **Simplicity is power**: The global ordering + circular arrangement scheme is extremely simple, yet mathematically guarantees order invariance.
- **Unique OOD perspective**: Frames the positional bias problem as a training-inference distribution mismatch and mitigates it through minimal modifications.
- **Experimental insights**: Discerned that in bfloat16 precision, attention scores in PINE produce many tied values, leading to non-deterministic sorting, which is an important practical insight.

## Limitations & Future Work

1. **Inability to handle completely arbitrary input structures**: Still requires explicit segment partitioning.
2. **Selective routing requires two forward passes**: Increases inference cost during practical deployment.
3. **Global ordering does not guarantee optimality**: Although lexicographical sorting is simple and effective, placing relevant documents closer might be better (the advantage of MonoT5 sorting).
4. **Limited large-scale experiments**: Due to resource constraints, experiments for the 70B model were not conducted at ndoc=30.
5. **Lack of direct validation in high-impact scenarios such as LLM-as-a-judge**.

## Related Work & Insights

- PINE is the most direct predecessor. RoToR eliminates its core defect of query-wise sorting through global ordering.
- Conceptually linked to order-invariance methods in Set/Graph ML (Murphy et al., 2019), but circular allocation and its application to pre-trained LMs represent novel contributions.
- The idea of selective routing can be generalized to other scenarios requiring "heterogeneous processing" (e.g., deciding whether to use retrieval results in RAG).
- Techniques for adapting RoPE and causal attention can inspire other work requiring modifications to the attention mechanism.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of circular arrangement and global ordering is simple and novel. Introducing the OOD perspective to positional bias analysis is a unique contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers three major task categories (LitM / KGQA / MMLU) across 5 model sizes and multiple sorting algorithm variants, including variance and time analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured, vivid diagrams, with a direct and intuitive comparison to PINE.
- **Value**: ⭐⭐⭐⭐ — Solves a persistent and practical problem in decoder-only LMs with a neat and practical approach.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Using Source-Side Confidence Estimation for Reliable Translation into Unfamiliar Languages](using_source-side_confidence_estimation_for_reliable_translation_into_unfamiliar.md)
- [\[ACL 2025\] Enhancing Transformers for Generalizable First-Order Logical Entailment](enhancing_fol_entailment.md)
- [\[ACL 2025\] SEOE: A Scalable and Reliable Semantic Evaluation Framework for Open Domain Event Detection](seoe_semantic_eval.md)
- [\[ACL 2025\] Are Any-to-Any Models More Consistent Across Modality Transfers Than Specialists?](are_any-to-any_models_more_consistent_across_modality_transfers_than_specialists.md)
- [\[ACL 2025\] PopAlign: Diversifying Contrasting Patterns for a More Comprehensive Alignment](popalign_diversifying_contrasting_patterns_for_a_more_comprehensive_alignment.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] TPA: Next Token Probability Attribution for Detecting Hallucinations in RAG
description: >-
  [ACL 2026][RAG hallucination detection] This paper proposes TPA, a framework that mathematically decomposes the generation probability of each token in an LLM into contributions from seven sources (Query, RAG Context, Past Token, Self Token, FFN, Final LayerNorm, and Initial Embedding), and combines part-of-speech (POS) tagging for feature aggregation to achieve state-of-the-art hallucination detection in RAG settings.
tags:
  - ACL 2026
  - RAG hallucination detection
  - probability attribution
  - residual stream decomposition
  - part-of-speech tagging
  - attention mechanism
date: 2026-05-08
content_hash: 07900d656dfb02be
---

# TPA: Next Token Probability Attribution for Detecting Hallucinations in RAG

**Conference**: ACL 2026
**arXiv**: [2512.07515](https://arxiv.org/abs/2512.07515)
**Code**: None
**Area**: Information Retrieval / Hallucination Detection
**Keywords**: RAG hallucination detection, probability attribution, residual stream decomposition, part-of-speech tagging, attention mechanism

## TL;DR

This paper proposes TPA, a framework that mathematically decomposes the generation probability of each token in an LLM into contributions from seven sources (Query, RAG Context, Past Token, Self Token, FFN, Final LayerNorm, and Initial Embedding), and combines part-of-speech (POS) tagging for feature aggregation to achieve state-of-the-art hallucination detection in RAG settings.

## Background & Motivation

**Background**: RAG mitigates LLM hallucinations by retrieving external knowledge, but models may still ignore or misinterpret retrieved information. Existing detection methods either rely on heuristic proxy signals (e.g., consistency checking, semantic entropy) or focus on binary conflicts between FFN and RAG context.

**Limitations of Prior Work**: (1) Proxy-signal methods measure only the "symptoms" of hallucination (e.g., output variance, surface-level confidence) without addressing architectural root causes, and fail on confidently incorrect outputs. (2) Prior internal-analysis work (e.g., ReDeEP) considers only the binary FFN-vs-RAG conflict, neglecting the influence of other critical components such as LayerNorm and the user query.

**Key Challenge**: A high FFN contribution to token probability does not always indicate hallucination — it is normal for function words (e.g., "the", "of") but highly suspicious for named entities. Existing methods cannot distinguish this grammatical difference.

**Goal**: To establish a complete token probability attribution framework covering all additive components of the Transformer, and to capture grammatical-dimension anomalies by incorporating POS information.

**Key Insight**: The additive structure of the Transformer residual stream is exploited to precisely decompose final token probabilities into incremental contributions from each component.

**Core Idea**: Token probability = Initial Embedding contribution + per-layer Attention contributions + per-layer FFN contributions + Final LayerNorm adjustment. Attention contributions are further allocated to four sources (Query / RAG / Past / Self) via attention weights, and features are formed by aggregating these attributions by POS category.

## Method

### Overall Architecture

TPA proceeds in three steps: (1) **Coarse-grained decomposition** — a probing function (logit lens) decomposes token probability into contributions from Initial Embedding, per-layer Attention, per-layer FFN, and Final LayerNorm; (2) **Fine-grained attribution** — Attention contributions are allocated to individual attention heads in logit space, then attributed to the four sources (Query / RAG / Past / Self) via attention weights, forming a seven-dimensional attribution vector; (3) **Syntax-aware feature engineering** — attribution scores are aggregated by POS category (nouns, verbs, numerals, etc.) to construct detection features.

### Key Designs

1. **Complete Probability Decomposition (Theorem 1)**:

    - Function: Precisely decomposes the final generation probability of a token into a sum of contributions from each component.
    - Mechanism: A probing function $\Phi(\mathbf{h}, y) = [\text{Softmax}(\mathbf{h} \mathbf{W}_U)]_y$ is defined to map any intermediate hidden state to a token probability. Each component's contribution is defined as the difference in probing probability before and after applying that component: $\Delta P_{att}^{(l)} = \Phi(\mathbf{h}_{mid}^{(l)}, y) - \Phi(\mathbf{h}^{(l-1)}, y)$. By telescoping summation, all differences sum exactly to the final probability.
    - Design Motivation: This is an exact decomposition (not an approximation) that loses no information. Compared to prior work focusing solely on FFN, it covers previously neglected components such as LayerNorm and the initial embedding.

2. **Logit-Space Attention Head Attribution**:

    - Function: Allocates the per-layer attention contribution to individual heads and then to the four input sources.
    - Mechanism: Because of Softmax nonlinearity, directly decomposing attention head contributions in probability space is infeasible. The method instead operates in logit space, where each head's logit contribution $\Delta z_{h,y}^{(l)}$ can be computed exactly (by projecting the head output onto the unembedding vector). Probability contributions are then allocated to heads using exponential logit ratios. Each head's contribution is subsequently distributed to the Query / RAG / Past / Self sources according to attention weights.
    - Design Motivation: A first-order Taylor expansion provides the theoretical basis (Proposition 1); logit space is linear and thus admits additive decomposition.

3. **POS-Aware Feature Aggregation**:

    - Function: Captures anomalies in attribution patterns across different grammatical categories.
    - Mechanism: POS tagging is applied to the generated response, and the seven-dimensional attribution vector of each token is averaged within each POS category, yielding a $7 \times |\text{POS}|$-dimensional feature vector. For example, a low RAG contribution for nouns or an anomalously high LayerNorm contribution for numerals are strong hallucination indicators.
    - Design Motivation: Normal attribution patterns differ substantially across POS categories — function words naturally depend on FFN/LayerNorm, whereas content words should be primarily driven by RAG. Neglecting POS distinctions obscures these critical signals.

### Loss & Training

A lightweight classifier (e.g., XGBoost) is trained on the attribution features. The entire attribution computation can be completed in a single teacher-forced forward pass (non-autoregressive), yielding high computational efficiency.

## Key Experimental Results

### Main Results

TPA achieves state-of-the-art performance across 5 LLMs (Llama2-7B/13B, Llama3-8B, Mistral-7B, Qwen3-8B) and multiple RAG hallucination detection benchmarks, outperforming prior methods based on consistency checking, semantic entropy, and internal probing.

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| Full TPA (7 sources + POS) | SOTA | Complete attribution + POS aggregation |
| w/o POS aggregation | Significant drop | Validates the importance of POS distinction |
| FFN + RAG only (binary) | Drop | Validates the value of full-component coverage |
| w/o LayerNorm | Drop | LayerNorm is a newly identified important signal source |

### Key Findings

- **LayerNorm is a neglected hallucination signal**: SHAP analysis reveals that an excessively high LayerNorm contribution for numerals (NUM) is a strong hallucination indicator — a pattern entirely undetectable by the conventional FFN-vs-RAG framework.
- **POS distinction is critical**: Low RAG contribution and high FFN contribution for nouns signal hallucination, yet the same pattern is entirely normal for function words. Without POS aggregation, the detector cannot distinguish these two cases.
- **Cross-architecture generalization**: TPA performs consistently across Llama2/3, Mistral, and Qwen3, demonstrating that attribution patterns are universal properties of the Transformer architecture.
- **Single forward pass**: Unlike consistency- or entropy-based methods that require multiple sampling passes, TPA requires only one teacher-forced forward pass, yielding high inference efficiency.

## Highlights & Insights

- **Paradigm shift from "detecting symptoms" to "diagnosing root causes"**: Rather than relying on output-level proxy signals, TPA directly analyzes the actual contribution of each component during generation, providing a more reliable detection foundation.
- **Mathematical elegance of exact decomposition**: The telescoping summation property of the residual stream enables an exact (non-approximate) probability decomposition, grounded in solid theoretical foundations.
- **Novel findings on LayerNorm**: This work is the first to reveal the role of Final LayerNorm in hallucination generation, broadening the understanding of internal Transformer mechanisms.

## Limitations & Future Work

- The framework assumes that RAG-retrieved context is correct and relevant, and does not address hallucinations caused by retrieval errors.
- The POS tagger itself may be noisy on generated text, potentially degrading feature quality.
- A classifier must be trained, making the approach not fully unsupervised.
- While fine-grained attribution is performed at the token level, the final output is a response-level detection signal; token-level hallucination localization is not provided.

## Related Work & Insights

- **vs. ReDeEP**: ReDeEP analyzes only the binary conflict between FFN and RAG context. TPA extends the analysis to all seven sources and discovers previously overlooked signals such as LayerNorm.
- **vs. Semantic Entropy / Consistency Checking**: These methods measure output-level symptoms; TPA directly analyzes the internal generation mechanism, making it more robust to confidently incorrect outputs.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The exact seven-source probability decomposition combined with POS aggregation constitutes an entirely new detection paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validation across 5 models is thorough; SHAP analysis provides interpretability.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematical derivations are rigorous and figures are clear.
- Value: ⭐⭐⭐⭐⭐ Provides a new analytical framework and state-of-the-art method for RAG hallucination detection.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Detecting RAG Extraction Attack via Dual-Path Runtime Integrity Game](detecting_rag_extraction_attack_via_dual-path_runtime_integrity_game.md)
- [\[ACL 2026\] Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation](stable-rag_mitigating_retrieval-permutation-induced_hallucinations_in_retrieval-.md)
- [\[ICLR 2026\] Token-Guard: Towards Token-Level Hallucination Control via Self-Checking Decoding](../../ICLR2026/information_retrieval/token-guard_towards_token-level_hallucination_control_via_self-checking_decoding.md)
- [\[ACL 2026\] Context Attribution with Multi-Armed Bandit Optimization](context_attribution_with_multi-armed_bandit_optimization.md)
- [\[ACL 2026\] Is Agentic RAG Worth It? An Experimental Comparison of RAG Approaches](is_agentic_rag_worth_it_an_experimental_comparison_of_rag_approaches.md)

<!-- RELATED:END -->

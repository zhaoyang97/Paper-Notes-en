---
title: >-
  [Paper Note] TPA: Next Token Probability Attribution for Detecting Hallucinations in RAG
description: >-
  [ACL 2026][LLM Safety][RAG Hallucination Detection] This paper proposes the TPA framework, which mathematically decomposes the generation probability of each token in LLMs into contributions from seven sources (Query…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "RAG Hallucination Detection"
  - "Probability Attribution"
  - "Residual Stream Decomposition"
  - "POS Tagging"
  - "Attention Mechanism"
date: 2026-05-08
content_hash: ddafbf805da153a6
---

# TPA: Next Token Probability Attribution for Detecting Hallucinations in RAG

**Conference**: ACL 2026  
**arXiv**: [2512.07515](https://arxiv.org/abs/2512.07515)  
**Code**: None  
**Area**: Information Retrieval / Hallucination Detection  
**Keywords**: RAG Hallucination Detection, Probability Attribution, Residual Stream Decomposition, POS Tagging, Attention Mechanism

## TL;DR

This paper proposes the TPA framework, which mathematically decomposes the generation probability of each token in LLMs into contributions from seven sources (Query, RAG Context, Past Token, Self Token, FFN, Final LayerNorm, and Initial Embedding). By aggregating these features with Part-of-Speech (POS) tagging, it achieves SOTA hallucination detection performance in RAG scenarios.

## Background & Motivation

**Background**: RAG mitigates LLM hallucinations by retrieving external knowledge, yet models may still ignore or misinterpret the retrieved information. Existing detection methods either rely on heuristic proxy signals (e.g., consistency checks, semantic entropy) or focus on binary conflicts between FFNs and RAG context.

**Limitations of Prior Work**: (1) Proxy signal methods only measure "symptoms" of hallucinations (e.g., output variance, surface confidence) rather than architectural root causes, making them ineffective against confident errors. (2) Previous internal analysis work (e.g., ReDeEP) focuses solely on the binary conflict between FFN and RAG, overlooking the influence of other critical components like LayerNorm and user queries.

**Key Challenge**: High FFN contribution to token probability does not always indicate a hallucination—it is normal for functional words ("the", "of") but highly suspicious for named entities. Existing methods cannot distinguish these grammatical differences.

**Goal**: Establish a complete token probability attribution framework that covers all additive components of the Transformer while incorporating POS information to capture anomalies at the grammatical level.

**Key Insight**: Leverage the additive structure of the Transformer residual stream to precisely decompose the final token probability into contribution increments from each component.

**Core Idea**: Token probability = Initial Embedding contribution + Attention contribution per layer + FFN contribution per layer + Final LayerNorm adjustment. Attention contributions are further allocated to four sources (Query/RAG/Past/Self) based on attention weights. These are then aggregated by POS tags to form detection features.

## Method

### Overall Architecture

TPA consists of three steps: (1) Coarse-grained decomposition—using a detection function (logit lens) to decompose token probability into four categories: Initial Embedding, Attention by layer, FFN by layer, and Final LayerNorm; (2) Fine-grained attribution—allocating Attention contributions to individual attention heads in the logit space, then attributing them to Query/RAG/Past/Self sources based on attention weights to form a seven-dimensional attribution vector; (3) POS-aware feature engineering—aggregating attribution scores by POS tags (nouns, verbs, numerals, etc.) to construct detection features.

### Key Designs

1.  **Full Probability Decomposition (Theorem 1)**:
    - **Function**: Precisely decomposes the final generation probability of a token into the sum of contributions from all components.
    - **Mechanism**: Defines a probing function $\Phi(\mathbf{h}, y) = [\text{Softmax}(\mathbf{h} \mathbf{W}_U)]_y$ to map any intermediate state to a token probability. The contribution of each component is defined as the difference in probing probability before and after applying that component: $\Delta P_{att}^{(l)} = \Phi(\mathbf{h}_{mid}^{(l)}, y) - \Phi(\mathbf{h}^{(l-1)}, y)$. Due to the telescopic sum, all differences sum exactly to the final probability.
    - **Design Motivation**: This is a precise decomposition (not an approximation) that loses no information. Unlike prior work focusing only on FFNs, this covers neglected components like LayerNorm and Initial Embeddings.

2.  **Logit-space Attention Head Attribution**:
    - **Function**: Allocates the attention contribution of each layer to individual attention heads, and subsequently to the four input sources.
    - **Mechanism**: Direct decomposition of attention head contributions in probability space is infeasible due to Softmax non-linearity. By moving to the logit space, the logit contribution of each head $\Delta z_{h,y}^{(l)}$ can be calculated exactly (projecting head output onto the unembedding vector). Probability contributions are then allocated to heads via exponential logit ratios. Each head's contribution is further distributed to Query/RAG/Past/Self based on attention weights.
    - **Design Motivation**: First-order Taylor expansion provides the theoretical basis (Proposition 1); the logit space is linear, allowing for additive decomposition.

3.  **POS-aware Feature Aggregation**:
    - **Function**: Captures attribution pattern anomalies across different grammatical categories.
    - **Mechanism**: Performs POS tagging on generated responses and averages the seven-dimensional attribution vectors for each token by its POS category, forming a $7 \times |POS|$ feature vector. For instance, a low RAG contribution for nouns or an unusually high LayerNorm contribution for numerals are strong signals of hallucination.
    - **Design Motivation**: Normal attribution patterns vary significantly by POS—functional words naturally rely on FFN/LayerNorm, while content words should be primarily driven by RAG. Failing to distinguish POS would drown out these critical signals.

### Loss & Training

A lightweight classifier (e.g., XGBoost) is trained on the attribution features. The entire attribution calculation can be completed via a single teacher-forced forward pass (non-autoregressive), ensuring high computational efficiency.

## Key Experimental Results

### Main Results

TPA achieves SOTA performance on multiple RAG hallucination detection benchmarks across 5 LLMs (Llama2-7B/13B, Llama3-8B, Mistral-7B, Qwen3-8B), surpassing previous methods based on consistency, semantic entropy, and internal probing.

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Full TPA (7 sources + POS) | SOTA | Full attribution + POS aggregation |
| w/o POS Aggregation | Significant drop | Validates the criticality of POS distinction |
| FFN + RAG Only (Binary) | Drop | Validates the value of covering all components |
| w/o LayerNorm | Drop | LayerNorm is a newly discovered important signal source |

### Key Findings

- **LayerNorm is an overlooked hallucination signal source**: SHAP analysis shows that excessive LayerNorm contribution for numerals (NUM) is a strong hallucination indicator—something the traditional FFN vs. RAG framework cannot capture.
- **POS distinction is vital**: Low RAG and high FFN contributions for nouns are hallucination signals, but the same pattern is perfectly normal for functional words. Without POS aggregation, the detector cannot distinguish between these cases.
- **Cross-architecture generalization**: TPA performs consistently across Llama2/3, Mistral, and Qwen3, suggesting that attribution patterns are universal features of the Transformer architecture.
- **Single forward pass**: Unlike consistency or entropy methods requiring multiple samples, TPA requires only one teacher-forced forward pass, offering high inference efficiency.

## Highlights & Insights

- **Paradigm shift from "detecting symptoms" to "diagnosing causes"**: TPA moves away from relying on output-level proxy signals to directly analyzing the actual contribution of each component during generation, providing a more reliable foundation for detection.
- **Mathematical elegance of precise decomposition**: Utilizes the telescopic sum property of the residual stream to achieve exact (non-approximate) probability decomposition with a solid theoretical foundation.
- **New discovery regarding LayerNorm**: For the first time, it reveals the role of Final LayerNorm in hallucination generation, expanding the understanding of Transformer internal mechanisms.

## Limitations & Future Work

- Assumes the retrieved RAG context is correct and relevant; does not handle hallucinations caused by retrieval errors.
- The POS tagger itself may be noisy on generated text, affecting feature quality.
- Requires training a classifier; it is not a fully unsupervised detection method.
- Fine-grained attribution is performed at the token level but eventually aggregated for response-level detection; it does not yet provide token-level hallucination localization.

## Related Work & Insights

- **vs. ReDeEP**: ReDeEP only analyzes binary conflicts between FFN and RAG context. TPA expands this to all seven sources, discovering neglected signals like LayerNorm.
- **vs. Semantic Entropy / Consistency Checks**: These methods measure output-level symptoms, whereas TPA directly analyzes internal generation mechanisms, making it more robust against "confident errors."

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Precise seven-source probability decomposition + POS aggregation is a brand-new detection paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Thoroughly validated across 5 models; SHAP analysis provides interpretability.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Rigorous mathematical derivation and clear illustrations.
- **Value**: ⭐⭐⭐⭐⭐ Provides a new analytical framework and SOTA method for RAG hallucination detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] FinGround: Detecting and Grounding Financial Hallucinations via Atomic Claim Verification](finground_detecting_and_grounding_financial_hallucinations_via_atomic_claim_veri.md)
- [\[ACL 2026\] Detecting RAG Extraction Attack via Dual-Path Runtime Integrity Game](detecting_rag_extraction_attack_via_dual-path_runtime_integrity_game.md)
- [\[AAAI 2026\] WaterMod: Modular Token-Rank Partitioning for Probability-Balanced LLM Watermarking](../../AAAI2026/llm_safety/watermod_modular_token-rank_partitioning_for_probability-balanced_llm_watermarki.md)
- [\[ACL 2026\] Two Pathways to Truthfulness: On the Intrinsic Encoding of LLM Hallucinations](two_pathways_to_truthfulness_on_the_intrinsic_encoding_of_llm_hallucinations.md)
- [\[ACL 2026\] De-Anonymization at Scale via Tournament-Style Attribution](de-anonymization_at_scale_via_tournament-style_attribution.md)

</div>

<!-- RELATED:END -->

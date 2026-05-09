---
title: >-
  [Paper Note] SCURank: Ranking Multiple Candidate Summaries with Summary Content Units for Enhanced Summarization
description: >-
  [ACL 2026][Model Compression][Summary Ranking] This paper proposes SCURank, a ranking framework based on Summary Content Units (SCUs). It extracts SCUs from candidate summaries, estimates information importance via cross-summary clustering, and scores candidates by informativeness. SCURank replaces unstable LLM-based direct ranking and coarse-grained ROUGE-based ranking. Combined with BRIO contrastive learning in a multi-LLM distillation setting, it significantly improves the summarization performance of distilled models.
tags:
  - ACL 2026
  - Model Compression
  - Summary Ranking
  - Content Units
  - Contrastive Learning
  - Multi-LLM Distillation
  - Informativeness
date: 2026-05-08
content_hash: 255d2c7a7b124832
---

# SCURank: Ranking Multiple Candidate Summaries with Summary Content Units for Enhanced Summarization

**Conference**: ACL 2026
**arXiv**: [2604.19185](https://arxiv.org/abs/2604.19185)
**Code**: [https://github.com/IKMLab/SCURank](https://github.com/IKMLab/SCURank)
**Area**: Text Summarization / Model Distillation
**Keywords**: Summary Ranking, Content Units, Contrastive Learning, Multi-LLM Distillation, Informativeness

## TL;DR

This paper proposes SCURank, a ranking framework based on Summary Content Units (SCUs). It extracts SCUs from candidate summaries, estimates information importance via cross-summary clustering, and scores candidates by informativeness. SCURank replaces unstable LLM-based direct ranking and coarse-grained ROUGE-based ranking. Combined with BRIO contrastive learning in a multi-LLM distillation setting, it significantly improves the summarization performance of distilled models.

## Background & Motivation

**Background**: LLMs demonstrate strong performance on summarization tasks but incur high deployment costs. Distilling LLM summarization capabilities into smaller models such as BART has become a prominent trend. The BRIO framework trains small models via contrastive learning to distinguish good from poor summaries, making the ranking quality of candidate summaries critical.

**Limitations of Prior Work**: (1) Direct LLM-based ranking (e.g., GPTRank) is unstable — research has shown that LLMs are unreliable and inconsistent in text comparison and ranking tasks; (2) Classical metrics such as ROUGE only measure n-gram overlap and offer insufficient discriminative power for high-quality summaries; (3) Distilling from a single LLM introduces model-specific biases, limiting the diversity of generation patterns.

**Key Challenge**: The differences among high-quality summaries lie in information selection and coverage rather than surface lexical overlap. A ranking method capable of measuring informativeness rather than surface matching is needed.

**Goal**: (1) Design a summarization ranking method grounded in information content rather than direct comparison or surface overlap; (2) Validate the effectiveness of distilling from multiple diverse LLMs.

**Key Insight**: Return to the core objective of summarization — information retention. SCUs (Summary Content Units) are used as atomic representations of information, and the importance of each SCU is estimated through cross-summary clustering.

**Core Idea**: The quality of a summary is determined by the richness and importance of the information it contains — the more important SCUs a summary includes, the better the summary.

## Method

### Overall Architecture

SCURank follows a three-step pipeline: (1) **SCU Extraction** — gpt-4o-mini is used to extract concise, self-contained, and unique information units from each candidate summary; (2) **SCU Aggregation** — all SCUs are encoded into vectors using sentence-transformers and semantically similar SCUs are grouped via HDBSCAN clustering; (3) **Summary Scoring** — the importance of each SCU is determined by the size of its cluster (more summaries sharing the same SCU indicates higher importance), and the summary score equals the sum of its SCU importances divided by summary length. The resulting rankings are used for BRIO contrastive learning to train the distilled model.

### Key Designs

1. **SCU Extraction and Aggregation**:

    - **Function**: Decompose summaries into atomic information units and estimate the importance of each unit.
    - **Mechanism**: An LLM extracts SCUs from each summary (e.g., "Obama received the Nobel Peace Prize in 2009"). These SCUs are then encoded into vectors using all-mpnet-base-v2 and clustered via HDBSCAN, which automatically determines the number of clusters. Cluster size reflects information importance — the more independently summaries contain the same information, the more critical that information is.
    - **Design Motivation**: The LLM is only used for SCU extraction (a structured task with high reliability), thereby avoiding the instability of direct LLM-based ranking. HDBSCAN requires no preset number of clusters, adapting naturally to varying amounts of semantic information.

2. **Informativeness Scoring**:

    - **Function**: Compute an informativeness score for each summary based on the SCU distribution.
    - **Mechanism**: The score of summary $s_i$ = $\sum$ cluster sizes of its SCUs / summary length. Division by length prevents bias toward longer summaries. This score directly reflects "how much important information the summary contains."
    - **Design Motivation**: ROUGE measures surface overlap while GPTRank is unstable. The informativeness scoring of SCURank provides a concrete, stable, and interpretable ranking criterion.

3. **Multi-LLM Distillation**:

    - **Function**: Distill from summaries generated by multiple diverse LLMs to increase diversity.
    - **Mechanism**: For each document, multiple LLMs (GPT-4o, Claude, Gemini, etc.) generate candidate summaries, which are uniformly ranked by SCURank and fed into BRIO training to distill the target model. Summaries from different LLMs exhibit distinct content selection preferences and writing styles.
    - **Design Motivation**: Single-LLM distillation inherits model-specific biases. Multi-LLM distillation provides richer training signals and enhances the model's abstractive capacity.

### Loss & Training

The BRIO framework is used for contrastive learning: higher-ranked summaries serve as positive samples and lower-ranked ones as negative samples. BRIO jointly trains generation and evaluation capabilities. SCU extraction uses gpt-4o-mini, and encoding uses all-mpnet-base-v2.

## Key Experimental Results

### Main Results

**Comparison of Distilled Model Summarization Performance**

| Ranking Method | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore |
|----------------|---------|---------|---------|-----------|
| ROUGE Ranking | Baseline | Baseline | Baseline | Baseline |
| GPTRank | Slightly better | Slightly better | Unstable | Unstable |
| **SCURank** | **Best** | **Best** | **Best** | **Best** |

### Ablation Study

| Configuration | Performance | Notes |
|---------------|-------------|-------|
| Single-LLM Distillation | Baseline | Distilled from one LLM only |
| Multi-LLM Distillation + ROUGE Ranking | Improved | Diversity is beneficial |
| Multi-LLM Distillation + SCURank | **Best** | Informativeness ranking + diversity |
| HDBSCAN vs. K-Means | HDBSCAN better | Advantage of adaptive cluster count |

### Key Findings

- SCURank consistently outperforms ROUGE-based and GPTRank-based ranking across all evaluation metrics and datasets.
- Multi-LLM distillation enhances the abstractive capacity of distilled models (less copying, more paraphrasing).
- LLMs are reliable for SCU extraction (structured output) but unreliable for direct ranking tasks.
- SCURank rankings align more closely with human judgments of summary quality.
- Length normalization is critical — without it, longer summaries are systematically favored.

## Highlights & Insights

- Shifting the ranking focus from "surface matching" to "information retention" represents the correct direction for summarization evaluation.
- LLMs are reliable for structured tasks (SCU extraction) but unreliable for judgment tasks (ranking) — this distinction provides guidance for the appropriate use of LLMs in evaluation pipelines.
- HDBSCAN's adaptive clustering is well-suited to the natural grouping of information units.

## Limitations & Future Work

- SCU extraction still relies on LLMs, incurring non-trivial cost.
- Informativeness does not capture all aspects of summary quality — coherence, readability, and other dimensions are not directly modeled.
- Validation is limited to news summarization datasets.
- Future work could explore combining SCURank with fluency and coherence metrics.

## Related Work & Insights

- **vs. GPTRank**: Relies on direct LLM ranking, which is unstable; SCURank uses LLMs only for SCU extraction, with ranking based on deterministic information statistics.
- **vs. ROUGE**: Measures surface n-gram overlap with insufficient discriminative power for high-quality summaries; SCURank measures semantic-level information coverage.
- **vs. Nawrath et al. (2024)**: Proposes SGUs for evaluation purposes; SCURank extends this idea to ranking and distillation applications.

## Rating

- Novelty: ⭐⭐⭐⭐ Applying SCUs to ranking is a natural yet effective extension.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset evaluation, comparison of multiple ranking methods, and complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Method is clearly presented with intuitive pipeline diagrams.
- Value: ⭐⭐⭐⭐ Provides a more reliable ranking solution for summarization distillation.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Calibrated Speculative Decoding: Frequency-Guided Candidate Selection for Efficient Inference](calibrated_speculative_decoding_frequency-guided_candidate_selection_for_efficie.md)
- [\[ICCV 2025\] Local Dense Logit Relations for Enhanced Knowledge Distillation](../../ICCV2025/model_compression/local_dense_logit_relations_for_enhanced_knowledge_distillation.md)
- [\[ICCV 2025\] DuoLoRA: Cycle-Consistent and Rank-Disentangled Content-Style Personalization](../../ICCV2025/model_compression/duolora_cycle-consistent_and_rank-disentangled_content-style_personalization.md)
- [\[ACL 2026\] Training-Free Test-Time Contrastive Learning for Large Language Models](training-free_test-time_contrastive_learning_for_large_language_models.md)
- [\[ACL 2026\] DASH-KV: Accelerating Long-Context LLM Inference via Asymmetric KV Cache Hashing](dash-kv_accelerating_long-context_llm_inference_via_asymmetric_kv_cache_hashing.md)

<!-- RELATED:END -->

---
title: >-
  [Paper Note] SCURank: Ranking Multiple Candidate Summaries with Summary Content Units for Enhanced Summarization
description: >-
  [ACL 2026][Text Generation][Summary Ranking] This paper proposes SCURank, a ranking framework based on Summary Content Units (SCUs). It extracts SCUs, estimates information importance via cross-summary clustering…
tags:
  - "ACL 2026"
  - "Text Generation"
  - "Summary Ranking"
  - "Summary Content Units"
  - "Contrastive Learning"
  - "Multi-LLM Distillation"
  - "Informativeness"
date: 2026-05-08
content_hash: 609b3b8e144a8a23
---

# SCURank: Ranking Multiple Candidate Summaries with Summary Content Units for Enhanced Summarization

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.19185](https://arxiv.org/abs/2604.19185)  
**Code**: [https://github.com/IKMLab/SCURank](https://github.com/IKMLab/SCURank)  
**Area**: Text Summarization / Model Distillation  
**Keywords**: Summary Ranking, Summary Content Units, Contrastive Learning, Multi-LLM Distillation, Informativeness

## TL;DR

This paper proposes SCURank, a ranking framework based on Summary Content Units (SCUs). It extracts SCUs, estimates information importance via cross-summary clustering, and ranks candidate summaries by information richness. This approach replaces unstable direct LLM ranking and coarse-grained ROUGE ranking. When combined with BRIO contrastive learning in multi-LLM distillation scenarios, it significantly improves the performance of distilled summarization models.

## Background & Motivation

**Background**: LLMs perform exceptionally well in summarization but suffer from high deployment costs. Distilling LLM summarization capabilities into smaller models like BART has become a trend. The BRIO framework uses contrastive learning to train small models to distinguish between good and bad summaries, where the quality of candidate summary ranking is critical.

**Limitations of Prior Work**: (1) Direct LLM ranking (e.g., GPTRank) is unstable—research shows LLMs are unreliable and inconsistent in text comparison and ranking; (2) Classical metrics like ROUGE only measure n-gram overlap, showing insufficient discrimination for high-quality summaries; (3) Distilling from a single LLM introduces model-specific bias, limiting the diversity of generation patterns.

**Key Challenge**: The difference between high-quality summaries lies in information selection and coverage rather than surface lexical overlap. A ranking method that measures information richness instead of surface matching is required.

**Goal**: (1) Design a summary ranking method based on information content rather than direct comparison or surface overlap; (2) Verify the effectiveness of distillation from multiple diverse LLMs.

**Key Insight**: Return to the core objective of summarization—information retention. Use SCUs (Summary Content Units) as atomic representations of information and estimate the importance of each SCU through cross-summary clustering.

**Core Idea**: The quality of a summary is determined by the richness and importance of the information content it contains—the more important SCUs present in a summary, the better the summary.

## Method

### Overall Architecture

The SCURank process consists of three steps: (1) SCU Extraction—using gpt-4o-mini to extract short, independent, and unique information units from each candidate summary; (2) SCU Aggregation—encoding all SCUs into vectors using sentence-transformers and clustering similar SCUs via HDBSCAN; (3) Summary Scoring—the importance of each SCU is determined by the size of the cluster containing it (shared by more summaries = more important). The summary score is calculated as the sum of its SCU importance divided by summary length. Ranking results are used for BRIO contrastive learning to train the distilled model.

### Key Designs

1.  **SCU Extraction and Aggregation**:
    - **Function**: Decomposes summaries into atomic information units and estimates the importance of each unit.
    - **Mechanism**: LLMs extract SCUs (e.g., "Obama won the Nobel Peace Prize in 2009") from each summary. Then, all-mpnet-base-v2 encodes SCUs into vectors, and HDBSCAN identifies clusters automatically. Cluster size reflects information importance—the more summaries independently contain the same information, the more critical that information is.
    - **Design Motivation**: LLMs are used only for SCU extraction (a structured task with high reliability), avoiding the instability of direct LLM ranking. HDBSCAN does not require a predefined number of clusters, adapting to varying amounts of semantic information.

2.  **Information Richness Scoring**:
    - **Function**: Calculates an information richness score for each summary based on SCU distribution.
    - **Mechanism**: The score of summary $s_i = \sum \text{cluster size containing its SCUs} / \text{summary length}$. Dividing by length prevents bias toward longer summaries. This score directly reflects "how much important information the summary contains."
    - **Design Motivation**: ROUGE measures surface overlap, and GPTRank is unstable. SCURank's information richness score provides a concrete, stable, and interpretable ranking criterion.

3.  **Multi-LLM Distillation**:
    - **Function**: Distills from summaries produced by multiple different LLMs to increase diversity.
    - **Mechanism**: For the same document, multiple LLMs (GPT-4o, Claude, Gemini, etc.) generate candidate summaries. These are ranked uniformly by SCURank and fed into BRIO for training. Summaries generated by multiple LLMs exhibit different content selection preferences and writing styles.
    - **Design Motivation**: Single-LLM distillation inherits specific biases. Multi-LLM distillation provides richer training signals and enhances the model's abstractive capabilities.

### Loss & Training

The BRIO framework is used for contrastive learning: high-ranking summaries serve as positive samples, while low-ranking ones serve as negative samples. BRIO simultaneously trains generation and evaluation capabilities. SCU extraction uses gpt-4o-mini, and encoding uses all-mpnet-base-v2.

## Key Experimental Results

### Main Results

**Comparison of Distilled Model Summarization Performance**

| Ranking Method | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore |
| :--- | :--- | :--- | :--- | :--- |
| ROUGE Ranking | Baseline | Baseline | Baseline | Baseline |
| GPTRank | Slight Gain | Slight Gain | Unstable | Unstable |
| **SCURank** | **Best** | **Best** | **Best** | **Best** |

### Ablation Study

| Configuration | Effect | Description |
| :--- | :--- | :--- |
| Single-LLM Distillation | Baseline | Distilled from only one LLM |
| Multi-LLM Distillation + ROUGE | Gain | Diversity is beneficial |
| Multi-LLM Distillation + SCURank | **Best** | Informativeness ranking + Diversity |
| HDBSCAN vs K-Means | HDBSCAN Better | Advantage of adaptive cluster numbers |

### Key Findings

- SCURank consistently outperforms ROUGE and GPTRank ranking across all evaluation metrics and datasets.
- Multi-LLM distillation enhances the abstractive ability of the distilled model (less copying, more paraphrasing).
- LLMs are reliable for SCU extraction (structured output) but unreliable for direct ranking tasks.
- SCURank rankings align more closely with human judgments of summary quality.
- Length normalization is critical—without it, longer summaries are systematically favored.

## Highlights & Insights

- Shifting the ranking focus from "surface matching" back to "information retention" is the correct direction for summarization evaluation.
- The distinction that LLMs are reliable for structured tasks (SCU extraction) but unreliable for judgment tasks (ranking) provides guidance for the proper use of LLMs in evaluation.
- The adaptive clustering of HDBSCAN is well-suited for the natural grouping of information units.

## Limitations & Future Work

- SCU extraction still depends on LLMs, involving certain costs.
- Information richness does not represent the entirety of summary quality—coherence and readability are not directly modeled.
- Validated only on news summarization datasets.
- Future work could explore combining SCURank with fluency/coherence metrics.

## Related Work & Insights

- **vs GPTRank**: Relies on direct LLM ranking, which is unstable; SCURank uses LLM only for extraction, with ranking based on deterministic information statistics.
- **vs ROUGE**: Measures surface n-gram overlap, lacking discriminative power for high-quality summaries; SCURank measures semantic-level information coverage.
- **vs Nawrath et al. (2024)**: Proposed SGU for evaluation; SCURank extends this to ranking and distillation applications.

## Rating

- Novelty: ⭐⭐⭐⭐ Using SCU for ranking is a natural yet effective extension.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete with multiple datasets, ranking method comparisons, and ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear methodology and intuitive flowcharts.
- Value: ⭐⭐⭐⭐ Provides a more reliable ranking solution for summarization distillation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ThreadSumm: Summarization of Nested Discourse Threads Using Tree of Thoughts](threadsumm_summarization_of_nested_discourse_threads_using_tree_of_thoughts.md)
- [\[ACL 2026\] FACTS: Table Summarization via Offline Template Generation with Agentic Workflows](facts_table_summarization_via_offline_template_generation_with_agentic_workflows.md)
- [\[ACL 2026\] Adaptive Planning for Multi-Attribute Controllable Summarization with Monte Carlo Tree Search](adaptive_planning_for_multi-attribute_controllable_summarization_with_monte_carl.md)
- [\[ACL 2026\] In-depth Research Impact Summarization through Fine-Grained Temporal Citation Analysis](in-depth_research_impact_summarization_through_fine-grained_temporal_citation_an.md)
- [\[ACL 2026\] Can You Make It Sound Like You? Post-Editing LLM-Generated Text for Personal Style](can_you_make_it_sound_like_you_post-editing_llm-generated_text_for_personal_styl.md)

</div>

<!-- RELATED:END -->

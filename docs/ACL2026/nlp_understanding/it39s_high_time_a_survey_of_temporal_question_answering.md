---
title: >-
  [Paper Note] It's High Time: A Survey of Temporal Question Answering
description: >-
  [ACL 2026][NLP Understanding][Temporal Question Answering] This paper provides a comprehensive survey of Temporal Question Answering (TQA), proposing a unified analysis framework based on three dimensions: corpus tempora…
tags:
  - "ACL 2026"
  - "NLP Understanding"
  - "Temporal Question Answering"
  - "Temporal Reasoning"
  - "Retrieval-Augmented Generation"
  - "Large Language Models"
  - "Survey"
date: 2026-05-08
content_hash: 8f1a4b00c45bfe8b
---

# It's High Time: A Survey of Temporal Question Answering

**Conference**: ACL 2026  
**arXiv**: [2505.20243](https://arxiv.org/abs/2505.20243)  
**Code**: [https://github.com/DataScienceUIBK/TemporalQA-Survey](https://github.com/DataScienceUIBK/TemporalQA-Survey)  
**Area**: Information Retrieval / Temporal Question Answering  
**Keywords**: Temporal Question Answering, Temporal Reasoning, Retrieval-Augmented Generation, Large Language Models, Survey

## TL;DR

This paper provides a comprehensive survey of Temporal Question Answering (TQA), proposing a unified analysis framework based on three dimensions: corpus temporality, question temporality, and model temporal capabilities. It systematically reviews the evolution of TQA methods from rule-based pipelines to the Transformer/LLM era, benchmark datasets, and evaluation strategies, while identifying future challenges.

## Background & Motivation

**Background**: Time is a fundamental dimension of information generation, retrieval, and understanding. With the explosion of time-stamped content such as news, social media, and knowledge bases, there is a burgeoning need for question answering systems that can handle temporal constraints and contexts. TQA has evolved from rule-based pipelines to systems powered by Transformers and LLMs.

**Limitations of Prior Work**: TQA faces unique challenges: (1) Temporal ambiguity resolution—fuzzy expressions like "recently" or "post-war" require contextual anchoring; (2) Cross-temporal reasoning—understanding causal and sequential relationships between events; (3) Knowledge volatility—facts evolve over time, and static corpora or pre-trained models cannot answer time-sensitive queries; (4) Temporal intent may be implicit, requiring systems to infer the appropriate time range.

**Key Challenge**: Existing surveys either focus on general QA/IR or only on a narrow aspect of temporal processing. A recent TQA survey (Campos et al., 2014) predates modern temporal language models, RAG systems, and large-scale temporal benchmarks, leaving a significant knowledge gap.

**Goal**: To provide a comprehensive survey of TQA, covering systems built on unstructured text and offering a comparison framework for unified datasets, tasks, and methods.

**Key Insight**: A three-dimensional analysis framework is proposed—corpus temporality (synchronic vs. diachronic), question temporality (explicit/implicit intent, temporal direction, reasoning complexity), and model temporal capabilities (temporal language modeling, temporal-aware retrieval, temporal reasoning)—serving as the organizational principle for the entire paper.

**Core Idea**: The core challenge of TQA lies in the "mismatch" between these three dimensions—failure occurs when corpus temporality, question temporality, and model capabilities are not aligned.

## Method

### Overall Architecture

The survey is organized along the three-dimensional framework: (1) Corpus dimension—distinguishing synchronic corpora (documents from a single point in time) from diachronic corpora (collections of documents across time) and analyzing their respective impacts on TQA; (2) Question dimension—categorized into explicit/implicit temporal intent, past/present/future direction, and simple/multi-hop reasoning complexity; (3) Model dimension—covering temporal language modeling (how to encode temporal knowledge), temporal-aware retrieval (how to retrieve temporal-related documents), and temporal reasoning (how to perform temporal logic reasoning).

### Key Designs

1.  **Corpus Temporality Analysis**:
    - **Function**: Distinguishes the different requirements that synchronic and diachronic corpora impose on TQA systems.
    - **Mechanism**: In synchronic corpora (e.g., Wikipedia snapshots), temporal relationships of events must be inferred from document-internal structures; in diachronic corpora (e.g., news archives), timelines are derived directly from the temporal distribution of the document collection. Relative time expressions such as "today" or "next week" must be anchored to document publication dates for correct interpretation.
    - **Design Motivation**: This distinction explains why certain TQA methods are effective on one type of corpus but fail on another.

2.  **TQA Datasets and Benchmark Classification**:
    - **Function**: Systematically categorizes the features and coverage of existing TQA datasets.
    - **Mechanism**: Classifies datasets by knowledge source (News/Wikipedia/Freebase), creation method (crowdsourcing/automatic generation), answer type (extractive/free-form), time range, and support for multi-hop reasoning. Representative datasets such as ComplexTempQA ($100$ million+ questions) and ArchivalQA ($532,000$ news items over 20 years) are identified.
    - **Design Motivation**: In the absence of a unified classification framework, comparisons between disparate datasets lack systematicity.

3.  **TQA Methods in the LLM Era**:
    - **Function**: Reviews the latest TQA methods based on Transformers/LLMs.
    - **Mechanism**: Key advances include: (a) Temporal language modeling—injecting temporal awareness by pre-training on timestamped text (e.g., TempLM, TEMPLAMA); (b) Temporal-aware RAG—introducing temporal filtering and re-ranking in the retrieval stage; (c) Continual temporal adaptation—adapting to knowledge updates through continuous pre-training. While powerful, LLMs still suffer from knowledge decay (limited knowledge of events after training data cutoffs) and insufficient temporal reasoning capabilities.
    - **Design Motivation**: The widespread application of LLMs necessitates an urgent understanding of their temporal reasoning capabilities and limitations.

### Loss & Training

As a survey paper, specific training procedures are not proposed. Instead, the article reviews three types of training paradigms: (1) Temporal-augmented pre-training—explicitly encoding timestamp information in the corpus; (2) Temporal-aware fine-tuning—fine-tuning models on temporal QA data; (3) Continual learning—preventing knowledge decay by continuously training on data from new time periods.

## Key Experimental Results

### Main Results

**Statistics of Major TQA Datasets**

| Dataset | # Questions | Source | Answer Type | Time Range | Multi-hop |
| :--- | :--- | :--- | :--- | :--- | :--- |
| NewsQA | 119k | News | Free-form | 2007-2015 | ✗ |
| TimeQA | 41.2k | Wiki | Extractive | 1367-2018 | ✗ |
| ComplexTempQA | 100.2M | Wiki | Extractive | 1987-2023 | ✓ |
| ArchivalQA | 532k | News | Extractive | 1987-2007 | ✗ |
| TempLAMA | 50k | News | Extractive | 2010-2020 | ✓ |

### Ablation Study

**Typical Performance Comparison of LLMs on Temporal Reasoning Tasks**

| Model/Method | TempLAMA | TimeQA | Description |
| :--- | :--- | :--- | :--- |
| GPT-4 (zero-shot) | ~$40\%$ | ~$55\%$ | Baseline, no temporal enhancement |
| + Temporal-aware RAG | ~$60\%$ | ~$70\%$ | Retrieves temporal-related documents |
| + Continual Adaptation | ~$55\%$ | ~$65\%$ | Continual training on new data |
| Specialized Temporal Model | ~$65\%$ | ~$72\%$ | Temporal-augmented pre-training |

### Key Findings

- The primary bottlenecks for LLMs in temporal reasoning are: (1) Knowledge cutoffs leading to inaccurate answers for recent events; (2) Instability in understanding implicit temporal expressions ("recently", "not long ago").
- RAG is currently the most effective method to address insufficient temporal knowledge in LLMs, though temporal-aware retrieval strategies remain immature.
- Multi-hop temporal reasoning (e.g., "Who was president after event $X$ but before event $Y$?") remains the greatest challenge.
- Existing datasets mainly cover the past; benchmarks for future-oriented temporal QA are virtually non-existent.
- Temporal reasoning for synchronic versus diachronic corpora requires different modeling strategies, yet existing methods rarely distinguish between them.

## Highlights & Insights

- The three-dimensional analysis framework (Corpus × Question × Model) provides clear organizational principles for understanding TQA, which is a methodology transferable to surveys in other domains.
- The survey offers comprehensive coverage from rule-based systems to the LLM era, providing a complete evolutionary landscape of the TQA field.
- Identified key gaps—such as future-oriented temporal QA and continuous adaptation on diachronic corpora—set a clear direction for subsequent research.

## Limitations & Future Work

- The scope is limited to TQA on unstructured text, excluding temporal knowledge graph QA and semi-structured table QA.
- Some quantitative comparisons are based on synthetic estimates; direct comparisons across different datasets and settings are limited.
- Future challenges: (1) Future-oriented temporal reasoning; (2) Reasoning over temporally inconsistent documents; (3) Alleviating knowledge decay.
- It is recommended to develop continuously updated benchmarks to evaluate TQA systems longitudinally.

## Related Work & Insights

- **vs. Campos et al. (2014)**: The previous TQA survey, which predates the Transformer era; this paper fills a decade-long gap.
- **vs. Kolomiyets & Moens (2011)**: A general QA survey with limited temporal coverage; ours focuses exclusively on the temporal dimension.
- **vs. Zhu et al. (2025)**: A recent survey focusing on general QA/IR with shallow analysis of temporal reasoning; ours provides an in-depth analysis of temporal reasoning.

## Rating

- **Novelty**: ⭐⭐⭐ The three-dimensional framework is innovative for a survey, though as a review it doesn't introduce new methods.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Systematic comparison of a vast number of datasets and methods.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear organization, systematic taxonomy, and informative charts.
- **Value**: ⭐⭐⭐⭐ Fills a ten-year survey gap in the TQA field and serves as an important reference for researchers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Table Question Answering in the Era of Large Language Models: A Comprehensive Survey](table_question_answering_in_the_era_of_large_language_models_a_comprehensive_sur.md)
- [\[ACL 2026\] Beyond Chunking: Discourse-Aware Hierarchical Retrieval for Long Document Question Answering](beyond_chunking_discourse-aware_hierarchical_retrieval_for_long_document_questio.md)
- [\[ACL 2026\] AdapTime: Enabling Adaptive Temporal Reasoning in Large Language Models](adaptime_enabling_adaptive_temporal_reasoning_in_large_language_models.md)
- [\[ACL 2026\] ASTRA: Adaptive Semantic Tree Reasoning Architecture for Complex Table Question Answering](astra_adaptive_semantic_tree_reasoning_architecture_for_complex_table_question_a.md)
- [\[ACL 2026\] Unveiling Temporal Framing in News Text](uncovering_temporal_framing_in_the_news.md)

</div>

<!-- RELATED:END -->

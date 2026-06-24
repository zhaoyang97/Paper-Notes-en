---
title: >-
  [Paper Note] It's High Time: A Survey of Temporal Question Answering
description: >-
  [ACL 2026][NLP Understanding][Temporal Question Answering] This paper provides a comprehensive survey of Temporal Question Answering (TQA), proposing a unified analytical framework based on three dimensions: corpus temporality, question temporality, and model temporal capability. It systematically reviews the evolution of TQA methods from rule-based pipelines to the Transformer/LLM era, organizes benchmark datasets and evaluation strategies, and identifies future challenges.
tags:
  - "ACL 2026"
  - "NLP Understanding"
  - "Temporal Question Answering"
  - "Temporal Reasoning"
  - "Retrieval-Augmented Generation"
  - "Large Language Models"
  - "Survey"
date: 2026-05-08
content_hash: 809dc4c9bfdaafaa
---

# It's High Time: A Survey of Temporal Question Answering

**Conference**: ACL 2026  
**arXiv**: [2505.20243](https://arxiv.org/abs/2505.20243)  
**Code**: [https://github.com/DataScienceUIBK/TemporalQA-Survey](https://github.com/DataScienceUIBK/TemporalQA-Survey)  
**Area**: Information Retrieval / Temporal Question Answering  
**Keywords**: Temporal Question Answering, Temporal Reasoning, Retrieval-Augmented Generation, Large Language Models, Survey

## TL;DR

This paper provides a comprehensive survey of Temporal Question Answering (TQA), proposing a unified analytical framework based on three dimensions: corpus temporality, question temporality, and model temporal capability. It systematically reviews the evolution of TQA methods from rule-based pipelines to the Transformer/LLM era, organizes benchmark datasets and evaluation strategies, and identifies future challenges.

## Background & Motivation

**Background**: Time is a fundamental dimension of information generation, retrieval, and understanding. With the explosion of timestamped content in news, social media, and knowledge bases, question answering systems that can handle temporal constraints and context are required. TQA has evolved from rule-based pipelines to systems based on Transformers and LLMs.

**Limitations of Prior Work**: TQA faces unique challenges: (1) Temporal ambiguity resolution—vague expressions like "recently" or "post-war" require contextual anchoring; (2) Cross-temporal reasoning—understanding causal and sequential relationships between events; (3) Knowledge mutability—facts evolve over time, and static corpora or pre-trained models cannot answer time-sensitive queries; (4) Temporal intent may be implicit, requiring the system to infer appropriate time ranges.

**Key Challenge**: Existing surveys either focus on general QA/IR or only address narrow aspects of temporal processing. The most recent TQA survey (Campos et al., 2014) predates modern temporal language models, RAG systems, and large-scale temporal benchmarks, leaving a significant knowledge gap.

**Goal**: To provide a comprehensive survey of TQA, covering TQA over unstructured text and providing a unified framework for comparing datasets, tasks, and methods.

**Key Insight**: A three-dimensional analytical framework is proposed—Corpus Temporality (synchronic vs. diachronic), Question Temporality (explicit/implicit intent, temporal direction, reasoning complexity), and Model Temporal Capability (temporal language modeling, temporal-aware retrieval, temporal reasoning)—serving as the organizational principle for the review.

**Core Idea**: The central challenge of TQA lies in the "mismatch" between these three dimensions—systems fail when corpus temporality, question temporality, and model capabilities are not aligned.

## Method

### Overall Architecture

As a survey, the "method" is a three-dimensional analytical framework used to organize TQA work from rule-based pipelines to the LLM era. The three dimensions are: Corpus Temporality—distinguishing between synchronic (documents at a single point in time) and diachronic (document collections across time) corpora; Question Temporality—categorizing questions by explicit/implicit temporal intent, past/present/future direction, and simple/multi-hop reasoning complexity; Model Temporal Capability—covering temporal language modeling (encoding temporal knowledge), temporal-aware retrieval (retrieving time-relevant documents), and temporal reasoning (executing temporal logic). The primary thesis is that TQA failures essentially stem from "misalignment" across these dimensions.

### Key Designs

> The three-dimensional framework is the core contribution. Points 1-3 describe the dimensions (Corpus → Question → Model), and Point 4 describes the supporting dataset coordinate system.

**1. Corpus Temporality: Synchronic vs. Diachronic Dictates TQA Approaches**

Many TQA methods fail when applied to different corpora because they do not account for the temporal structure of the data. The survey categorizes corpora into two types: synchronic (e.g., a Wikipedia snapshot), where temporal relations must be inferred from internal document structure, and diachronic (e.g., multi-year news archives), where timelines derive from the temporal distribution of the collection. Key differences lie in relative temporal expressions—words like "today" or "next week" must be anchored to document publication dates, a process that differs significantly between the two types.

**2. Question Temporality: Categorizing "What Time is Being Asked" into Three Dimensions**

Timestamped corpora alone are insufficient; the temporal attributes of the question determine the required reasoning. The survey classifies questions along three axes: temporal intent (explicit, e.g., "Who was the US President in 2008?" vs. implicit, e.g., "Who was in power during the financial crisis?"); temporal direction (past, present, or future—though current benchmarks rarely cover the future); and reasoning complexity (single-hop factoid vs. multi-hop temporal reasoning, e.g., "Who held office after X but before Y?"). This taxonomy measures the difficulty of TQA tasks.

**3. Model Temporal Capability: Methodological Lineage in the LLM Era**

This dimension defines how systems process time. The survey organizes progress into three lines: Temporal Language Modeling—injecting temporal awareness via pre-training on timestamped text (e.g., TempLM, TempLAMA); Temporal-Aware Retrieval-Augmented Generation (RAG)—introducing temporal filtering and re-ranking in the retrieval stage; and Continual Temporal Adaptation—using continual pre-training to keep up with knowledge updates. It highlights two LLM weaknesses: knowledge recency (limited knowledge of events post-cutoff) and insufficient temporal reasoning skills, making RAG the primary current solution.

**4. Dataset and Benchmark Classification: A Unified Coordinate System**

Complementing the framework, the survey establishes a coordinate system for previously incomparable TQA datasets. It categorizes them by knowledge source (News/Wikipedia/Freebase), creation method (crowdsourced/automatic), answer type (extractive/free-form), time range, and multi-hop support. This allows researchers to identify which capabilities are covered by existing benchmarks and where gaps remain (e.g., the lack of future-oriented TQA).

### Loss & Training

While not proposing a specific training method, the survey summarizes three paradigms: Temporal-Augmented Pre-training (encoding timestamps in the corpus), Temporal-Aware Fine-tuning (on TQA-specific data), and Continual Learning (training on new temporal slices to prevent knowledge decay).

## Key Experimental Results

### Main Results

**Statistics of Major TQA Datasets**

| Dataset | Questions | Source | Answer Type | Time Range | Multi-hop |
|---------|-----------|--------|-------------|------------|-----------|
| NewsQA | 119k | News | Free-form | 2007-2015 | ✗ |
| TimeQA | 41.2k | Wiki | Extractive | 1367-2018 | ✗ |
| ComplexTempQA | 100.2M | Wiki | Extractive | 1987-2023 | ✓ |
| ArchivalQA | 532k | News | Extractive | 1987-2007 | ✗ |
| TempLAMA | 50k | News | Extractive | 2010-2020 | ✓ |

### Ablation Study

**Performance Comparison of LLMs on Temporal Reasoning Tasks**

| Model/Method | TempLAMA | TimeQA | Description |
|--------------|----------|--------|-------------|
| GPT-4 (zero-shot) | ~40% | ~55% | Baseline, no temporal enhancement |
| + Temporal-aware RAG | ~60% | ~70% | Retrieves time-relevant documents |
| + Continual Adaptation | ~55% | ~65% | Continual training on new data |
| Specialized Temporal Model | ~65% | ~72% | Temporal-augmented pre-training |

### Key Findings

- The main bottlenecks for LLMs in temporal reasoning: (1) Knowledge cutoff dates lead to inaccuracies for recent events; (2) Unstable understanding of implicit temporal expressions ("recently", "not long ago").
- RAG is currently the most effective method to address limited temporal knowledge in LLMs, though temporal-aware retrieval strategies remain immature.
- Multi-hop temporal reasoning (e.g., "Who was president after event X but before event Y?") remains the greatest challenge.
- Existing datasets primarily cover the past; there are almost no benchmarks for future-oriented temporal QA.
- Synchronic and diachronic corpora require different modeling strategies for temporal reasoning, but current methods rarely distinguish between them.

## Highlights & Insights

- The three-dimensional analysis framework (Corpus × Question × Model) provides a clear organizational principle for understanding TQA and is transferable to other domain surveys.
- The survey is comprehensive, covering the entire evolution from rule-based systems to the LLM era.
- It identifies critical gaps—such as future-oriented TQA and continual adaptation on diachronic corpora—providing clear directions for future research.

## Limitations & Future Work

- The scope is limited to TQA over unstructured text, excluding Temporal Knowledge Graph QA and semi-structured Table QA.
- Some quantitative comparisons are based on general estimates; direct comparisons across different datasets and settings remain limited.
- Future challenges: (1) Future-oriented temporal reasoning; (2) Reasoning over temporally inconsistent documents; (3) Mitigating knowledge decay.
- It is recommended to develop continually updated benchmarks for longitudinal evaluation of TQA systems.

## Related Work & Insights

- **vs Campos et al. (2014)**: The previous TQA survey predates the Transformer era; Ours fills a decade-long gap.
- **vs Kolomiyets & Moens (2011)**: A general QA survey with limited temporal coverage; Ours focuses exclusively on the temporal dimension.
- **vs Zhu et al. (2025)**: A recent survey on general QA/IR where temporal reasoning is touched upon briefly; Ours provides an in-depth analysis of temporal reasoning.

## Rating

- Novelty: ⭐⭐⭐ The framework is novel for a survey, though no new methodology is proposed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematic comparison of a large number of datasets and methods.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear organization, systematic taxonomy, and informative charts.
- Value: ⭐⭐⭐⭐ Fills a ten-year gap in the TQA field, serving as a vital reference for researchers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Table Question Answering in the Era of Large Language Models: A Comprehensive Survey](table_question_answering_in_the_era_of_large_language_models_a_comprehensive_sur.md)
- [\[ACL 2026\] Knowledge-driven Augmentation and Retrieval for Integrative Temporal Adaptation](knowledge-driven_augmentation_and_retrieval_for_integrative_temporal_adaptation.md)
- [\[ACL 2026\] Semantic Reranking at Inference Time for Hard Examples in Rhetorical Role Labeling](semantic_reranking_at_inference_time_for_hard_examples_in_rhetorical_role_labeli.md)
- [\[ACL 2026\] AdapTime: Enabling Adaptive Temporal Reasoning in Large Language Models](adaptime_enabling_adaptive_temporal_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Beyond Chunking: Discourse-Aware Hierarchical Retrieval for Long Document Question Answering](beyond_chunking_discourse-aware_hierarchical_retrieval_for_long_document_questio.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] It's High Time: A Survey of Temporal Question Answering
description: >-
  [ACL 2026][LLM/NLP][Temporal Question Answering] This paper presents a comprehensive survey of Temporal Question Answering (TQA), proposing a unified analytical framework along three dimensions—corpus temporality, question temporality, and model temporal capability—and systematically reviewing the evolution of TQA methods, benchmark datasets, and evaluation strategies from rule-based pipelines to the Transformer/LLM era, while identifying key challenges for future research.
tags:
  - ACL 2026
  - LLM/NLP
  - Temporal Question Answering
  - Temporal Reasoning
  - Retrieval-Augmented Generation
  - Large Language Models
  - Survey
date: 2026-05-08
content_hash: ebe707398f25ea13
---

# It's High Time: A Survey of Temporal Question Answering

**Conference**: ACL 2026
**arXiv**: [2505.20243](https://arxiv.org/abs/2505.20243)
**Code**: [https://github.com/DataScienceUIBK/TemporalQA-Survey](https://github.com/DataScienceUIBK/TemporalQA-Survey)
**Area**: Information Retrieval / Temporal Question Answering
**Keywords**: Temporal Question Answering, Temporal Reasoning, Retrieval-Augmented Generation, Large Language Models, Survey

## TL;DR

This paper presents a comprehensive survey of Temporal Question Answering (TQA), proposing a unified analytical framework along three dimensions—corpus temporality, question temporality, and model temporal capability—and systematically reviewing the evolution of TQA methods, benchmark datasets, and evaluation strategies from rule-based pipelines to the Transformer/LLM era, while identifying key challenges for future research.

## Background & Motivation

**Background**: Time is a fundamental dimension in information generation, retrieval, and understanding. The explosive growth of timestamped content from news, social media, and knowledge bases has driven the need for QA systems capable of handling temporal constraints and context. TQA has evolved from rule-based pipelines to Transformer- and LLM-based systems.

**Limitations of Prior Work**: TQA faces unique challenges: (1) temporal ambiguity resolution—vague expressions such as "recently" and "post-war" require contextual anchoring; (2) cross-temporal reasoning—understanding causal and sequential relations among events; (3) knowledge mutability—facts evolve over time, and static corpora and pretrained models cannot answer time-sensitive queries; (4) temporal intent may be implicit, requiring systems to infer appropriate temporal scopes.

**Key Challenge**: Existing surveys either focus on general QA/IR or address only narrow aspects of temporal processing. The most recent TQA survey (Campos et al., 2014) predates modern temporal language models, RAG systems, and large-scale temporal benchmarks, leaving a significant knowledge gap.

**Goal**: To provide a comprehensive survey of TQA covering non-structured text, with a unified comparative framework for datasets, tasks, and methods.

**Key Insight**: A three-dimensional analytical framework is proposed—corpus temporality (synchronic vs. diachronic), question temporality (explicit/implicit intent, temporal direction, reasoning complexity), and model temporal capability (temporal language modeling, temporally-aware retrieval, temporal reasoning)—serving as the organizing principle throughout the survey.

**Core Idea**: The central challenge of TQA lies in the "misalignment" among the three dimensions—system failures arise when corpus temporality, question temporality, and model capability are not aligned.

## Method

### Overall Architecture

The survey is organized around the three-dimensional framework: (1) the corpus dimension—distinguishing synchronic corpora (documents from a single point in time) from diachronic corpora (document collections spanning time) and analyzing their respective implications for TQA; (2) the question dimension—categorized by explicit/implicit temporal intent, past/present/future direction, and simple/multi-hop reasoning complexity; (3) the model dimension—covering temporal language modeling (encoding temporal knowledge), temporally-aware retrieval (retrieving temporally relevant documents), and temporal reasoning (performing temporal logical inference).

### Key Designs

1. **Corpus Temporality Analysis**:

    - Function: Distinguishes the different requirements that synchronic and diachronic corpora place on TQA systems.
    - Mechanism: In synchronic corpora (e.g., Wikipedia snapshots), temporal relations among events must be inferred from the internal structure of documents; in diachronic corpora (e.g., news archives), timelines are derived directly from the temporal distribution of the document collection. Relative temporal expressions such as "today" and "next week" must be anchored to document publication dates for correct interpretation.
    - Design Motivation: This distinction explains why certain TQA methods are effective on one type of corpus but fail on the other.

2. **TQA Dataset and Benchmark Taxonomy**:

    - Function: Systematically organizes the characteristics and coverage of existing TQA datasets.
    - Mechanism: Datasets are classified by knowledge source (news/Wikipedia/Freebase), creation method (crowdsourcing/automatic generation), answer type (extractive/free-form), temporal span, and multi-hop support. Representative datasets are identified, including ComplexTempQA (100M+ questions) and ArchivalQA (532K questions spanning 20 years of news).
    - Design Motivation: The absence of a unified taxonomic framework renders systematic comparison across datasets difficult.

3. **TQA Methods in the LLM Era**:

    - Function: Surveys state-of-the-art TQA methods based on Transformers and LLMs.
    - Mechanism: Key advances include: (a) temporal language modeling—injecting temporal awareness via pretraining on timestamped text (e.g., TempLM, TEMPLAMA); (b) temporally-aware RAG—incorporating temporal filtering and re-ranking at the retrieval stage; (c) continual temporal adaptation—updating knowledge through continual pretraining. Despite their power, LLMs still suffer from knowledge decay (limited awareness of events after training cutoff) and insufficient temporal reasoning capability.
    - Design Motivation: The widespread adoption of LLMs makes it urgent to understand their temporal reasoning capabilities and limitations.

### Loss & Training

As a survey paper, no specific training procedures are introduced. The paper organizes three training paradigms: (1) temporally-augmented pretraining—explicitly encoding timestamp information in the training corpus; (2) temporally-aware fine-tuning—fine-tuning models on temporal QA data; (3) continual learning—preventing knowledge decay by continuously training on data from new time periods.

## Key Experimental Results

### Main Results

**Statistics of Major TQA Datasets**

| Dataset | # Questions | Source | Answer Type | Temporal Span | Multi-hop |
|--------|-------------|--------|-------------|---------------|-----------|
| NewsQA | 119k | News | Free-form | 2007–2015 | ✗ |
| TimeQA | 41.2k | Wikipedia | Extractive | 1367–2018 | ✗ |
| ComplexTempQA | 100.2M | Wikipedia | Extractive | 1987–2023 | ✓ |
| ArchivalQA | 532k | News | Extractive | 1987–2007 | ✗ |
| TempLAMA | 50k | News | Extractive | 2010–2020 | ✓ |

### Ablation Study

**Representative Performance Comparison of LLMs on Temporal Reasoning Tasks**

| Model / Method | TempLAMA | TimeQA | Notes |
|---------------|----------|--------|-------|
| GPT-4 (zero-shot) | ~40% | ~55% | Baseline, no temporal augmentation |
| + Temporally-aware RAG | ~60% | ~70% | Retrieval of temporally relevant documents |
| + Continual adaptation | ~55% | ~65% | Continual training on new data |
| Dedicated temporal model | ~65% | ~72% | Temporally-augmented pretraining |

### Key Findings

- The primary bottlenecks for LLMs in temporal reasoning are: (1) knowledge cutoff dates leading to inaccurate responses about recent events; (2) unstable interpretation of implicit temporal expressions (e.g., "recently," "not long ago").
- RAG is currently the most effective approach for addressing LLMs' temporal knowledge limitations, yet temporally-aware retrieval strategies remain underdeveloped.
- Multi-hop temporal reasoning (e.g., "Who was president after event X but before event Y?") remains the greatest challenge.
- Existing datasets predominantly cover past time periods; benchmarks for future-oriented temporal QA are nearly absent.
- Temporal reasoning over synchronic versus diachronic corpora requires different modeling strategies, a distinction that existing methods seldom make.

## Highlights & Insights

- The three-dimensional analytical framework (corpus × question × model) provides a clear organizational principle for understanding TQA and offers a transferable survey methodology applicable to other domains.
- The survey offers comprehensive coverage from rule-based systems to the LLM era, presenting a complete evolutionary picture of the TQA field.
- The identified critical gaps—future-oriented temporal QA and continual adaptation over diachronic corpora—point to concrete directions for subsequent research.

## Limitations & Future Work

- The survey scope is restricted to TQA over unstructured text, excluding temporal knowledge graph QA and semi-structured table QA.
- Some quantitative comparisons are based on aggregated estimates; direct comparisons across different datasets and settings are limited.
- Future challenges include: (1) future-oriented temporal reasoning; (2) reasoning over temporally inconsistent documents; (3) mitigating knowledge decay.
- The development of continuously updated benchmarks for longitudinal evaluation of TQA systems is recommended.

## Related Work & Insights

- **vs. Campos et al. (2014)**: The previous TQA survey predates the Transformer era; this paper fills a decade-long gap.
- **vs. Kolomiyets & Moens (2011)**: A general QA survey with limited coverage of the temporal dimension; this paper focuses specifically on temporal aspects.
- **vs. Zhu et al. (2025)**: A recent survey on general QA/IR with shallow treatment of temporal reasoning; this paper provides an in-depth analysis of temporal reasoning.

## Rating

- Novelty: ⭐⭐⭐ The three-dimensional framework is a notable contribution, though no new methods are proposed given the survey nature of the work.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematic comparison covering a large number of datasets and methods.
- Writing Quality: ⭐⭐⭐⭐⭐ Clearly organized, systematic taxonomy, and highly informative figures and tables.
- Value: ⭐⭐⭐⭐ Fills a decade-long survey gap in the TQA field and serves as an important reference for researchers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] From Static Inference to Dynamic Interaction: A Survey of Streaming Large Language Models](from_static_inference_to_dynamic_interaction_a_survey_of_streaming_large_languag.md)
- [\[ACL 2026\] ChatHLS: Towards Systematic Design Automation and Optimization for High-Level Synthesis](chathls_towards_systematic_design_automation_and_optimization_for_high-level_syn.md)
- [\[AAAI 2026\] TEMPLE: Incentivizing Temporal Understanding of Video LLMs via Progressive Pre-SFT Alignment](../../AAAI2026/llm_nlp/temple_incentivizing_temporal_understanding_of_video_large_language_models_via_p.md)
- [\[NeurIPS 2025\] Detecting High-Stakes Interactions with Activation Probes](../../NeurIPS2025/llm_nlp/detecting_high-stakes_interactions_with_activation_probes.md)
- [\[AAAI 2026\] ParetoHqD: Fast Offline Multiobjective Alignment of Large Language Models Using Pareto High-Quality Data](../../AAAI2026/llm_nlp/paretohqd_fast_offline_multiobjective_alignment_of_large_language_models_using_p.md)

</div>

<!-- RELATED:END -->

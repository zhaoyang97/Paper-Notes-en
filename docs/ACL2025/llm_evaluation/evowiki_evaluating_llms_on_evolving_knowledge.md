---
title: >-
  [Paper Note] EvoWiki: Evaluating LLMs on Evolving Knowledge
description: >-
  [ACL 2025][LLM Evaluation][Knowledge Evolution] This paper proposes EvoWiki, an automatically updatable dynamic evaluation benchmark that categorizes knowledge into three levels: stable, evolved, and uncharted. It systematically evaluates the ability of LLMs to utilize evolving knowledge, revealing synergistic effects when combining Retrieval-Augmented Generation (RAG) and Continual Learning (CL).
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Knowledge Evolution"
  - "Dynamic Benchmark"
  - "RAG"
  - "Continual Learning"
  - "Wikidata"
  - "Multi-hop Reasoning"
date: 2026-05-08
content_hash: 648ea1704095ec45
---

# EvoWiki: Evaluating LLMs on Evolving Knowledge

**Conference**: ACL 2025  
**arXiv**: [2412.13582](https://arxiv.org/abs/2412.13582)  
**Code**: [GitHub](https://github.com/wtangdev/EvoWiki)  
**Area**: LLM Evaluation / Knowledge Utilization  
**Keywords**: Knowledge Evolution, Dynamic Benchmark, RAG, Continual Learning, Wikidata, Multi-hop Reasoning  

## TL;DR

This paper proposes EvoWiki, an automatically updatable dynamic evaluation benchmark that categorizes knowledge into three levels: stable, evolved, and uncharted. It systematically evaluates the ability of LLMs to utilize evolving knowledge, revealing synergistic effects when combining Retrieval-Augmented Generation (RAG) and Continual Learning (CL).

## Background & Motivation

- **Problem Definition**: LLMs have a knowledge cut-off date, while world knowledge continues to evolve (fact updates, emergence of new information). How can the capability of LLMs to utilize dynamically changing knowledge be evaluated?
- **Limitations of Prior Work**: (1) Traditional benchmarks (e.g., NaturalQA, HotpotQA) are static and insensitive to temporal changes; (2) Dynamic benchmarks (e.g., LiveBench, RealtimeQA) focus on data freshness but fail to systematically distinguish different evolutionary levels of knowledge; (3) Static golden answers may become outdated as knowledge updates, leading to false negatives.
- **Key Motivation**: There is a need for a benchmark that can automatically update and distinguish between different knowledge evolution stages (known and unchanged vs. known but changed vs. completely new) to accurately evaluate LLMs' knowledge utilization capabilities.
- **Key Challenge**: (1) Newly released LLMs face the risk of test set contamination; (2) Utilizing knowledge of different evolutionary levels varies in difficulty, which requires fine-grained attribute support for in-depth analysis; (3) The benchmark must be automatically updatable to keep pace with the continuous evolution of both knowledge and models.

## Method

### Overall Architecture

EvoWiki is constructed based on the Wikidata knowledge graph and Wikipedia text sources. It identifies knowledge evolution statuses by comparing knowledge graph snapshots at different timestamps. The entire dataset revolves around three timestamps: init-time (2021.09, where LLM knowledge has achieved sufficient coverage), cutoff-time (2024.01, the LLM knowledge cutoff), and current-time (2024.05, the evaluation time).

### Key Designs

1. **Three-level Knowledge Evolution Classification**:
    - **Stable**: Facts that remain unchanged from init-time to current-time $\rightarrow$ evaluates the baseline performance of LLMs on existing knowledge.
    - **Evolved**: Facts that existed before init-time but changed between cutoff-time and current-time $\rightarrow$ evaluates whether LLMs can recognize knowledge updates (e.g., a person's spouse has changed).
    - **Uncharted**: Brand-new facts appearing after cutoff-time $\rightarrow$ evaluates the ability of LLMs to acquire new knowledge, naturally remaining contamination-free.

2. **Multi-dimensional Attribute Annotation**:
    - **Referenced Context**: Fact triples are anchored to Wikipedia pages through distant supervision, ensuring every question is supported by a verifiable document.
    - **Multi-hop Reasoning**: Extending from 1-hop to 3-hop reasoning questions to test knowledge integration and reasoning capabilities.
    - **Popularity**: Measures how popular a piece of knowledge is using Wikipedia page views, analyzing the impact of popularity on model performance.

3. **Automatically Updatable Mechanism**: The construction pipeline is based on the continuous updates of Wikidata/Wikipedia. Timestamps can be flexibly adjusted to fit the knowledge cutoff dates of different LLMs, supporting automated data updates without human intervention.

### Loss & Training

No model training is involved; this is a benchmark and evaluation work. Standard settings are used for evaluating RAG and CL.

## Experiments

### Main Results: RAG vs CL vs Combined Methods (Llama-3.1-8B)

| Method | Stable 1-hop | Stable Multi-hop | Evolved 1-hop | Evolved Multi-hop | Uncharted 1-hop | Uncharted Multi-hop |
|------|------------|------------|-------------|-------------|---------------|---------------|
| Open-book (Upper Bound) | 86.87 | 56.40 | 75.24 | 60.30 | 83.52 | 51.32 |
| Closed-book | 31.61 | 22.17 | 6.96 | 13.99 | 10.84 | 17.90 |
| BM25 Retrieval | 59.41 | 14.42 | 36.13 | 13.85 | 44.93 | 15.47 |
| Contriever Retrieval | 77.90 | 19.37 | 48.99 | 17.85 | 72.69 | 21.42 |
| SFT + Closed-book | 36.97 | 24.41 | 8.53 | 17.34 | 15.15 | 20.59 |
| **SFT + Contriever** | **82.85** | **24.02** | **57.22** | **20.22** | **78.85** | **24.84** |
| **SFT + Open-book** | **92.10** | **60.22** | **80.78** | **62.90** | **89.34** | **55.07** |

RAG performs strongly on 1-hop queries but remains weak on multi-hop ones; CL offers consistent but limited improvements; the RAG+CL combination achieves the best performance.

### Ablation Study: Human Evaluation of Dataset Quality

| Metric | Stable (1-hop/All) | Evolved (1-hop/All) | Uncharted (1-hop/All) |
|------|-------------------|--------------------|-----------------------|
| Fluency | 99.17 / 95.69 | 94.58 / 95.56 | 95.00 / 95.42 |
| Answerability | 96.67 / 94.44 | 94.17 / 95.69 | 92.92 / 92.64 |
| Accuracy | 97.92 / 93.19 | 93.33 / 94.58 | 91.67 / 90.97 |

Human evaluation verifies that the dataset achieves high quality of 90%+ across fluency, answerability, and accuracy.

### Key Findings

- **Evolved knowledge is the most challenging**: In closed-book settings, the 1-hop accuracy of evolved knowledge is only 6.96% (Llama-3.1-8B), indicating that LLMs heavily tend to output outdated answers.
- **RAG is effective for 1-hop but remains weak for multi-hop**: Contriever comes close to Open-book on Stable 1-hop (77.90 vs. 86.87) but drops to 19.37 on multi-hop (vs. Open-book 56.40), because multi-hop reasoning requires retrieving and integrating multiple document segments.
- **CL provides stable but modest improvements**: SFT on Closed-book only improves Evolved from 6.96% to 8.53%, and Uncharted from 10.84% to 15.15%.
- **Synergistic effect of RAG + CL**: SFT + Contriever chemistry outperforms individual RAG or CL on most metrics, e.g., improving Stable 1-hop from 77.90 (Contriever) to 82.85 (SFT + Contriever).
- **Larger retrieval corpora degrade performance**: Expanding the retrieval corpus (BM25_large_corpus) actually leads to worse performance due to increased distractor information.

## Highlights & Insights

- The ingenious three-level knowledge classification (stable/evolved/uncharted) provides both baseline controls and frontier challenges, offering a comprehensive framework to evaluate LLM knowledge utilization.
- The fully automated, updatable mechanism prevents the benchmark from becoming outdated—simply adjusting the timestamps adapts it to newly released LLMs.
- Multi-dimensional attributes (context, multi-hop, popularity) support fine-grained analysis instead of merely reporting a single aggregated score.
- The discovery of the synergistic effect between RAG + CL provides future directions for combining both approaches to adapt to knowledge evolution.

## Limitations & Future Work

- The dataset is primarily based on English Wikidata/Wikipedia, lacking multilingual versions.
- The knowledge domain is restricted to human entities (entity type = human); the evolution patterns in other domains (science, technology, geography, etc.) might differ.
- Multi-hop reasoning is limited to a maximum of 3 hops, and automatically generated 3-hop questions may degrade into shallow 2-hop reasoning.
- Using GPT-4o-mini for question polishing might introduce systematic biases.
- The categorization of Evolved and Uncharted rely on Wikidata edit timestamps, while Wikidata itself contains update latency and noise.

## Related Work & Insights

- **Temporal QA Benchmarks**: TimeQA (Chen et al., 2021), TEMPLAMA (Dhingra et al., 2021), etc., evaluate time-aware reasoning; EvoWiki goes a step further by distinguishing different evolutionary levels of knowledge.
- **Dynamic Benchmarks**: RealtimeQA (Kasai et al., 2023), LiveBench (White et al., 2024) focus on data freshness and contamination mitigation. EvoWiki adds knowledge categorization and multi-dimensional attributes on top of this.
- **Knowledge Conflicts**: Ying et al. (2024) revealed that LLMs tend to prioritize internal knowledge even when external knowledge is correct. The Evolved category in EvoWiki serves as an ideal testbed for such conflicts.
- **RAG and CL**: Lewis et al. (2020) proposed RAG; Tang et al. (2024) proposed a generator-reader framework. This work systematically compares their performances under knowledge evolution scenarios.

## Rating

| Dimension | Score (1-10) |
|------|-----------|
| Novelty | 7 |
| Experimental Thoroughness | 8 |
| Writing Quality | 8 |
| Value | 8 |
| **Overall Score** | **7.8** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] GuessArena: Guess Who I Am? A Self-Adaptive Framework for Evaluating LLMs in Domain-Specific Knowledge and Reasoning](guessarena_guess_who_i_am_a.md)
- [\[ACL 2025\] SANSKRITI: A Comprehensive Benchmark for Evaluating Language Models' Knowledge of Indian Culture](sanskriti_a_comprehensive_benchmark_for_evaluating_language_models_knowledge_of_.md)
- [\[ACL 2025\] Towards Objective Fine-tuning: How LLMs' Prior Knowledge Causes Potential Poor Calibration?](towards_objective_fine-tuning_how_llms_prior_knowledge_causes_potential_poor_cal.md)
- [\[ACL 2025\] AntiLeakBench: Preventing Data Contamination by Automatically Constructing Benchmarks with Updated Real-World Knowledge](antileakbench_preventing_data_contamination_by_automatically_constructing_benchm.md)
- [\[ACL 2025\] From Tools to Teammates: Evaluating LLMs in Multi-Session Coding Interactions](from_tools_to_teammates_evaluating_llms_in_multi-session_coding_interactions.md)

</div>

<!-- RELATED:END -->

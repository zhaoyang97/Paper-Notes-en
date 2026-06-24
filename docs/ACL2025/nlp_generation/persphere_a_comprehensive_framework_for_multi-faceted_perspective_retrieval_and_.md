---
title: >-
  [Paper Note] PerSphere: A Comprehensive Framework for Multi-Faceted Perspective Retrieval and Summarization
description: >-
  [Text Generation] > This paper proposes the PerSphere benchmark dataset and the MURS (Multi-faceted perspective retrieval and summarization) task, which aims to retrieve and comprehensively summarize multi-faceted perspectives on controversial issues from a document collection. It also proposes HierSphere, a hierarchical multi-agent summarization system, to alleviate challenges related to long contexts and perspective extraction.
tags:
  - "Text Generation"
date: 2026-05-08
content_hash: d356819dca769fce
---

# PerSphere: A Comprehensive Framework for Multi-Faceted Perspective Retrieval and Summarization

| Information | Content |
|------|------|
| Conference | ACL 2025 |
| arXiv | [2412.12588](https://arxiv.org/abs/2412.12588) |
| Code | [GitHub](https://github.com/LuoXiaoHeics/PerSphere) |
| Area | nlp_generation (Retrieval-Augmented Generation × Argument Mining × Information Synthesis) |
| Keywords | multi-faceted summarization, perspective retrieval, echo chamber, RAG, multi-agent |

## TL;DR

> This paper proposes the PerSphere benchmark dataset and the MURS (Multi-faceted perspective retrieval and summarization) task, which aims to retrieve and comprehensively summarize multi-faceted perspectives on controversial issues from a document collection. It also proposes HierSphere, a hierarchical multi-agent summarization system, to alleviate challenges related to long contexts and perspective extraction.

## Background & Motivation

- **Echo Chamber Issue**: Social platforms and recommendation algorithms increasingly trap users in echo chambers, leading to a biased understanding of various issues and worsening perspective polarization and the spread of misinformation.
- **Limitations of Prior Work**:
    - Argumentative summarization does not consider the "comprehensiveness" of content.
    - RAG methods focus on relevance but ignore the diversity of perspective coverage.
    - Perspectrum (Chen et al., 2019) assumes perspectives can be extracted prior to queries, which is impractical for real-world applications.
- **Core Needs**: Users do not seek a single unified "correct answer" but rather wish to obtain a balanced, comprehensive argumentative summary on controversial issues.
- **Task Definition**: Given a query and a document set D, the task is to retrieve comprehensive relevant documents and then summarize two opposing stances and their respective non-overlapping perspectives, complete with document citations.

## Method

### Task Formalization

The task is split into a two-step pipeline:

1. **Comprehensive Document Retrieval**: Retrieve $k$ documents from the document set D that comprehensively cover all perspectives.
2. **Multi-Faceted Summarization**: From the retrieved documents, summarize:
    - Two opposing claims ($c_0, c_1$)
    - Multiple non-overlapping perspectives under each claim
    - The corresponding document citations for each perspective

Formalization: $q \rightarrow c_0: \{p_{0,j}, [D_{0,j}]\}; c_1: \{p_{1,j}, [D_{1,j}]\}$

### Dataset Construction

#### Theperspective Subset (185 Instances)
- Data source: Editorial articles from THEPERSPECTIVE website.
- Covers topics including life, sports, politics, entertainment, and technology.
- Each query has two opposing stances, with 2-4 perspectives per stance.
- **Characteristics**: Each perspective corresponds to only one evidence document (relatively simple).
- Uses GPT-4-Turbo to complete incomplete perspective sentences.
- The document set was expanded from 1,103 to 4,107 (by adding irrelevant documents from Perspectrum to increase diversity).

#### Perspectrumx Subset (878 Instances)
- A more challenging version constructed based on the Perspectrum dataset.
- **Characteristics**: Each perspective can be supported by multiple documents ($|D_j^i| >= 1$).
- The number of documents and perspectives follows a Poisson-like distribution, exhibiting high variance.
- Total of 8,092 documents, with an average document length of 168.5 words.

### Evaluation Metrics

**Retrieval Metrics**:
- **Recall@k**: The proportion of relevant documents among the $k$ retrieved documents.
- **Cover@k**: The coverage of perspectives by the retrieved documents (focusing more on the comprehensiveness at the perspective level than Recall).

**Summarization Metrics**:
- **GPT-4 Score**: Uses a specially designed prompt to have GPT-4 evaluate the summarization quality (on a scale of 1-10).
- Evaluation criteria include: uniqueness of perspectives, absence of informational overlap, consistency with the claim, etc.
- Meta-evaluation confirms that the Pearson correlation coefficient between GPT-4 scores and human ratings is 0.70.

### HierSphere: Hierarchical Multi-Agent Summarization System

Designed based on two core challenges identified through analysis (long context and perspective extraction):

1. **Multiple Local Agents**: Divided the retrieved documents into multiple groups, with each group's local summary generated by an agent.
2. **Editorial Agent**: Fuses the local summaries, merges semantically identical perspectives, and refines them based on "one-sentence perspective" exemplars.

## Experiments

### Retrieval Experiments

| Retriever | Theperspective Recall@20 | Perspectrumx Recall@20 | Perspectrumx Cover@20 |
|--------|--------------------------|------------------------|----------------------|
| BM25 | 82.35 | 56.27 | 64.43 |
| E5-large | 88.89 | 61.18 | 70.17 |
| GTR-large | 94.80 | 65.68 | 72.98 |
| Ada-002 | 95.34 | 68.80 | 74.16 |
| GritLM | 96.77 | 70.58 | 77.01 |

- GritLM is consistently optimal across all metrics.
- The retrieval difficulty of Theperspective is significantly lower than that of Perspectrumx (single doc-per-perspective vs. multi-doc).

### Summarization Experiments

| Retriever | Summarization Model | Theperspective @20 | Perspectrumx @20 |
|--------|---------|-------------------|----------------|
| BM25 | GPT-4-Turbo | 7.74 | 5.49 |
| GritLM | GPT-4-Turbo | 8.16 | 6.04 |
| GritLM | Claude-3-Sonnet | 8.05 | 6.25 |
| Golden | Claude-3-Sonnet | 8.80 | 7.28 |

### Key Findings

1. **Multi-document support increases task difficulty**: Summarization scores on Perspectrumx are significantly lower than those on Theperspective.
2. **More documents are not necessarily better**: Using 20 documents is sometimes worse than using 10, and performance drops significantly with 100 documents — models are prone to introducing irrelevant knowledge and generating redundant perspectives.
3. **Document order affects results**: Reversing or randomly shuffling the document order degrades summarization quality, indicating that current LLMs tend to pay more attention to the beginning information in long contexts.
4. **Perspective extraction bottleneck**: Cover@20 reaches 96.77%, but Rp@20 is only 77.35% (the actual extractability of perspectives). This explains that the document covering perspectives does not mean the model can successfully extract those perspectives.
5. **HierSphere is effective**: For example, LLaMA-3.1-70B improves from 5.11 to 5.89 (+0.78) on Perspectrumx, with contributions from both long-context mitigation and perspective refinement.

### Human Evaluation

- Summarization Quality: The Pearson correlation between GPT-4 scores and human ratings is $r = 0.70$, with GPT-4 tending to overestimate.
- Perspective Extraction Entailment Judgment: The agreement between GPT-4 and humans is 83%-86%, validating the effectiveness of automatic evaluation.

## Highlights & Insights

1. **Filling the task gap**: For the first time, perspective "comprehensiveness" is integrated into the core evaluation dimension of retrieval and summarization, moving beyond the traditional RAG paradigm that only focuses on relevance.
2. **Cover@k metric design**: Reflects multi-faceted coverage better than standard Recall, which is of great significance for multi-perspective scenarios.
3. **Revealing the core bottlenecks in LLM summarization**: Long-context processing, one-sentence perspective extraction, and sensitivity to document order — these findings have universal reference value for the broader RAG community.
4. **Simplicity and effectiveness of HierSphere**: The multi-agent divide-and-conquer along with editorial fusion approach is simple yet directly and effectively alleviates the long-context bottleneck.
5. **Two complementary sub-datasets**: Theperspective (single-document/perspective, easier) and Perspectrumx (multi-document/perspective, harder) provide multi-tiered evaluation.

## Limitations & Future Work

1. **Dataset scale**: A total of 1,064 instances, which is relatively small for a retrieval task.
2. **Retrieval is not the main research focus**: Although retrieval itself poses a challenge to current models, the paper does not dive deep into retrieval improvement solutions.
3. **Summarization evaluation relies heavily on GPT-4**: Although automatic evaluation is validated by its correlation with human ratings, GPT-4 tends to overestimate, and the meta-evaluation sample size is small.
4. **No handling of ultra-large-scale document collections**: When the document collection reaches the scale of millions, coarse-to-fine retrieval and ranking are required, a scenario not explored in the paper.
5. **Pro/Con binary opposition assumption**: Controversies in the real world may have more than two stances, and the binary stance framework might be overly simplified.

## Related Work & Insights

- **Argument Mining**: Stance detection (Rinott et al., 2015), evidence detection (Ein-Dor et al., 2020), argument summarization and clustering (Ajjour et al., 2019; Syed et al., 2023). ArgSum (Li et al., 2024) focuses on persuasiveness rather than comprehensiveness.
- **RAG**: RAPTOR (Sarthi et al., 2024) and GraphRAG (Edge et al., 2024) enhance overall understanding but only focus on QA tasks.
- **Perspectrum (Chen et al., 2019)**: The most relevant prior work, but it assumes the perspective pool can be extracted prior to queries, which is impractical.

## Rating ⭐⭐⭐⭐

The task definition is of practical significance (combating echo chambers), the dataset construction is standardized, and the evaluation framework is comprehensive. The experimental findings (long-context bottlenecks, difficulty in perspective extraction, and sensitivity to document order) provide broad reference value for the RAG field. While the improvements of HierSphere are modest, the direction is correct. The primary limitations are the small dataset scale and the oversimplification of the binary opposition assumption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Structured Language Generation Model: Loss Calibration and Formatted Decoding for Efficient Text](../../AAAI2026/nlp_generation/structured_language_generation_model_loss_calibration_and_formatted_decoding_for.md)
- [\[ACL 2025\] Multi-document Summarization through Multi-document Event Relation Graph Reasoning in LLMs](event_graph_bias_mitigation_summarization.md)
- [\[ICLR 2026\] Antislop: A Comprehensive Framework for Identifying and Eliminating Repetitive Patterns in Language Models](../../ICLR2026/nlp_generation/antislop_a_comprehensive_framework_for_identifying_and_eliminating_repetitive_pa.md)
- [\[ACL 2025\] ATGen: A Framework for Active Text Generation](atgen_a_framework_for_active_text_generation.md)
- [\[ACL 2025\] Towards Better Open-Ended Text Generation: A Multicriteria Evaluation Framework](towards_better_open-ended_text_generation_a_multicriteria_evaluation_framework.md)

</div>

<!-- RELATED:END -->

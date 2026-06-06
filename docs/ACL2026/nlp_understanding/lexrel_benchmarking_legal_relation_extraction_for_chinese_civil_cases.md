---
title: >-
  [Paper Note] LexRel: Benchmarking Legal Relation Extraction for Chinese Civil Cases
description: >-
  [ACL 2026][NLP Understanding][Legal Relation Extraction] Ours constructs the first structured classification system for Chinese civil legal relations (9 major domains…
tags:
  - "ACL 2026"
  - "NLP Understanding"
  - "Legal Relation Extraction"
  - "Chinese Civil Cases"
  - "Legal Knowledge Graph"
  - "Benchmark"
  - "Relation Taxonomy"
date: 2026-05-08
content_hash: bc8a81f7700c3820
---

# LexRel: Benchmarking Legal Relation Extraction for Chinese Civil Cases

**Conference**: ACL 2026  
**arXiv**: [2512.12643](https://arxiv.org/abs/2512.12643)  
**Code**: [GitHub](https://github.com/thunlp/LexRel)  
**Area**: LLM Evaluation / Legal NLP  
**Keywords**: Legal Relation Extraction, Chinese Civil Cases, Legal Knowledge Graph, Benchmark, Relation Taxonomy

## TL;DR

Ours constructs the first structured classification system for Chinese civil legal relations (9 major domains, 265 relation types) and proposes the LexRel benchmark (1,140 expert-annotated samples). It evaluates the capabilities of mainstream LLMs on legal relation extraction tasks, identifying significant limitations in current models while demonstrating the gain effect of legal relation information on downstream legal AI tasks.

## Background & Motivation

**Background**: Legal relations are the fundamental units of analysis in Chinese civil cases, referring to relations between individuals prescribed by legal norms. In judicial practice, legal professionals frequently rely on legal relations for legal information retrieval, statute prediction, and case outcome analysis. However, legal relations have been long overlooked in the legal AI field, particularly lacking systematic research in the context of Chinese civil law.

**Limitations of Prior Work**: Current information extraction in legal AI mainly targets factual entities (e.g., persons, objects, contracts) or general social relations (e.g., employment, ownership), ignoring legal relations as concepts rooted in statutory rules and judicial practice, which differ essentially from ordinary semantic associations in natural language. Furthermore, legal relations are almost never explicitly expressed in judgment documents and usually must be inferred from factual descriptions. Existing legal relation schemas are often too coarse-grained, categorizing only at the broad level of civil rights and obligations.

**Key Challenge**: The lack of a fine-grained, structured legal relation classification system and high-quality annotated data makes it impossible to systematically evaluate and improve the capabilities of AI models in legal relation understanding.

**Goal**: (1) Establish the first comprehensive legal relation classification system covering Chinese civil law; (2) Define the legal relation extraction task and construct an expert-annotated benchmark dataset; (3) Evaluate the legal relation extraction capabilities of mainstream LLMs; (4) Validate the gains of legal relation information for downstream tasks.

**Key Insight**: Starting from legal theory and combining judicial practice with expert guidance, the system is constructed before computational annotation and evaluation are performed, ensuring both legal normativity and AI utility.

## Method

### Overall Architecture

The construction of LexRel is divided into three levels: (1) **Schema Design**: Construction of a hierarchical legal relation taxonomy and argument definitions; (2) **Task Definition and Data Annotation**: Definition of the legal relation extraction task (type extraction + argument extraction), using an "LLM pre-annotation + expert correction" pipeline to build the benchmark; (3) **Model Evaluation**: Evaluation of multiple SOTA LLMs under zero-shot and relation-enhanced settings.

### Key Designs

1.  **Hierarchical Legal Relation Taxonomy**
    *   **Function**: Provides the first fine-grained structured description of legal relations in Chinese civil law.
    *   **Mechanism**: Constructed through two stages—(a) Extracting relation expressions from judgment documents via keyword matching and filtering 123 candidate types into 6 domains based on legal terminology literature; (b) Review and expansion by two senior legal experts, adding 3 new domains (instrument relations, letter of credit relations, independent guarantee relations), resulting in a final coverage of **9 domains and 265 relation types**.
    *   **Design Motivation**: Combines empirical induction from judgment data with normative grounding in legal theory, ensuring the taxonomy is both data-supported and conceptually rigorous.

2.  **Legal Relation Extraction Task Definition**
    *   **Function**: Formalizes legal relation identification as a computable NLP task.
    *   **Mechanism**: Divided into two sub-tasks—**Type Extraction**: Identifying legal relation types from factual text $x$ as $\hat{r} = f_{\text{type}}(x), \hat{r} \in \mathcal{R}$; **Argument Extraction**: Extracting subject, object, and content $(\hat{S}, \hat{O}, \hat{c}) = f_{\text{arg}}(x, \hat{r})$ based on predicted types.
    *   **Design Motivation**: Legal relations are usually implicit in judgments and require inference from factual descriptions, which is more challenging than conventional relation extraction.

3.  **LLM + Expert Hybrid Annotation Pipeline**
    *   **Function**: Controls costs while ensuring annotation quality.
    *   **Mechanism**: DeepSeek-V3 first extracts candidate legal relation types and arguments from full judgments as draft annotations, followed by verification and correction by 6 legal professional annotators supervised by a senior legal AI expert. After removing 60 samples where no legal relation could be identified, 1,140 annotated samples were obtained.
    *   **Design Motivation**: Pure manual annotation of legal relations is extremely costly and requires expertise; LLM assistance significantly reduces the annotation burden.

### Evaluation Strategy

*   **Zero-shot Baselines**: Direct evaluation of LLM capabilities for type and argument extraction on LexRel.
*   **Relation Enhancement (RE) Baselines**: Using GPT-4o or DeepSeek-R1 to generate training data from full judgment texts (including legal analysis sections) for SFT on small open-source models.
*   **Argument Evaluation**: LLM-as-Judge (DeepSeek-V3) is utilized, with manual verification showing accuracies of 95.4% (subject), 96.9% (object), and 81.0% (content).

## Key Experimental Results

### Main Results

| Model | Method | Type Micro-F1 | Type Macro-F1 | Argument Micro-F1 | Argument Macro-F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| o3-mini | zero-shot | **0.762** | **0.441** | **0.382** | **0.129** |
| DeepSeek-R1 | zero-shot | 0.693 | 0.376 | 0.268 | 0.065 |
| GPT-4o | zero-shot | 0.670 | 0.314 | 0.224 | 0.068 |
| Claude-Sonnet-4 | zero-shot | 0.590 | 0.330 | 0.258 | 0.088 |
| Qwen3-14B | RE w/ R1 | 0.733 | 0.430 | 0.381 | 0.146 |
| Qwen3-8B | RE w/ R1 | 0.675 | 0.337 | 0.304 | 0.098 |
| Llama3.1-8B | zero-shot | 0.250 | 0.052 | 0.027 | 0.006 |

### Downstream Task Gain

| Model | Case Analysis | Case Analysis + LR | Consult | Consult + LR | Damage Calc | Damage Calc + LR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MiniCPM4-8B | 32.0 | **45.0** | 7.6 | **8.4** | 65.0 | **76.8** |
| DeepSeek-V3 | 66.2 | **68.2** | 16.4 | **16.5** | 85.0 | **86.4** |
| GPT-4o | 55.8 | **56.6** | 18.2 | **19.2** | 84.4 | **85.8** |

### Key Findings

*   Reasoning LLMs (o3-mini, DeepSeek-R1) significantly outperform non-reasoning models in zero-shot settings, indicating that legal relation extraction requires strong reasoning capabilities.
*   Argument extraction (max micro-F1 only 0.382) is much more difficult than type extraction (0.762). Macro-F1 is consistently lower than micro-F1 for all models, highlighting poor performance on long-tail relation types.
*   SFT can significantly improve small model performance (e.g., InternLM3-8B argument extraction improved from 0.048 to 0.323); Qwen3-14B + RE nearly matches o3-mini zero-shot levels.
*   Legal relation distribution in LexRel exhibits long-tail characteristics, highly consistent with the distribution of causes of action in 26.6 million real civil judgments (where the top 25 causes account for 80% of cases).
*   Legal relation information consistently brings gains to downstream tasks; MiniCPM4-8B improved from 32.0 to 45.0 (+13.0) in case analysis and from 65.0 to 76.8 (+11.8) in damage calculation.

## Highlights & Insights

*   **First Comprehensive Civil Legal Relation System**: 265 relation types across 9 domains, built through combining legal theory and data-driven methods, filling a major gap in legal AI.
*   **High Task Value**: Legal relations are implicit in judgments and must be inferred from facts, posing a severe challenge to the legal reasoning capabilities of LLMs.
*   **In-depth Long-tail Analysis**: Validates the distributional representativeness of LexRel by comparing it with the cause-of-action distribution of 26.6 million real judgments.
*   **Practical Value Proven via Downstream Gains**: Even if current model extraction accuracy is not yet perfect, introducing legal relation information still enhances performance in downstream tasks.

## Limitations & Future Work

*   The schema and dataset focus on Chinese civil law; direct transfer to other legal systems requires localized adaptation.
*   Synthetic training data generation and LLM-as-Judge evaluation use DeepSeek series models, posing a potential model-family coupling issue.
*   Argument extraction performance remains low (max macro-F1 only 0.146), with long-tail relation type extraction being the primary bottleneck.
*   The scale of 1,140 samples is relatively small, especially considering the data sparsity across 265 relation types.

## Related Work & Insights

*   **vs Legal Knowledge Graphs**: Existing legal KGs mainly capture general social relations (e.g., employment, kinship). LexRel focuses on relations under legal norms (e.g., creditor-debtor relations, contractual obligations), which are closer to judicial practice.
*   **vs General Relation Extraction**: Legal relation extraction is harder than general RE because legal relations are often not explicitly expressed in text and require inference combined with legal knowledge.
*   **vs Legal Benchmarks like LawBench**: LexRel focuses on an overlooked yet fundamental capability—legal relation identification—making it an important supplement to existing legal evaluations.

## Rating

*   **Novelty**: ⭐⭐⭐⭐ Systematically defines and evaluates Chinese civil legal relation extraction for the first time, filling an important gap.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated 12 models across two settings, including long-tail analysis and downstream task validation, though the dataset size is small.
*   **Writing Quality**: ⭐⭐⭐⭐ Handles the intersection of legal science and NLP well, with a clear construction process for the taxonomy.
*   **Value**: ⭐⭐⭐⭐ Provides vital benchmark resources for the legal AI community and promotes research in legal relation modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] HCRE: LLM-based Hierarchical Classification for Cross-Document Relation Extraction](hcre_llm-based_hierarchical_classification_for_cross-document_relation_extractio.md)
- [\[ACL 2026\] DiZiNER: Disagreement-guided Instruction Refinement via Pilot Annotation Simulation for Zero-shot Named Entity Recognition](diziner_disagreement-guided_instruction_refinement_via_pilot_annotation_simulati.md)
- [\[ACL 2026\] MSMO-ABSA: Multi-Scale and Multi-Objective Optimization for Cross-Lingual Aspect-Based Sentiment Analysis](msmo-absa_multi-scale_and_multi-objective_optimization_for_cross-lingual_aspect-.md)
- [\[ACL 2026\] Unveiling Temporal Framing in News Text](uncovering_temporal_framing_in_the_news.md)
- [\[ACL 2026\] Beyond Chunking: Discourse-Aware Hierarchical Retrieval for Long Document Question Answering](beyond_chunking_discourse-aware_hierarchical_retrieval_for_long_document_questio.md)

</div>

<!-- RELATED:END -->

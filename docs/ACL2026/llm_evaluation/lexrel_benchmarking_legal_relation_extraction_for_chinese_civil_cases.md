---
title: >-
  [Paper Note] LexRel: Benchmarking Legal Relation Extraction for Chinese Civil Cases
description: >-
  [ACL 2026][LLM Evaluation][legal relation extraction] This work introduces the first structured taxonomy of legal relations in Chinese civil law (9 domains, 265 relation types) and presents LexRel…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "legal relation extraction"
  - "Chinese civil cases"
  - "legal knowledge graph"
  - "benchmark"
  - "relation classification schema"
date: 2026-05-08
content_hash: 31fbf9b28083d380
---

# LexRel: Benchmarking Legal Relation Extraction for Chinese Civil Cases

**Conference**: ACL 2026  
**arXiv**: [2512.12643](https://arxiv.org/abs/2512.12643)  
**Code**: [GitHub](https://github.com/thunlp/LexRel)  
**Area**: LLM Evaluation / Legal NLP  
**Keywords**: legal relation extraction, Chinese civil cases, legal knowledge graph, benchmark, relation classification schema

## TL;DR

This work introduces the first structured taxonomy of legal relations in Chinese civil law (9 domains, 265 relation types) and presents LexRel, a benchmark comprising 1,140 expert-annotated instances. The benchmark is used to evaluate leading LLMs on legal relation extraction, revealing significant limitations of current models on this task, while also demonstrating that incorporating legal relation information yields consistent gains on downstream legal AI tasks.

## Background & Motivation

**Background**: Legal relations are the fundamental unit of analysis in Chinese civil cases, referring to relationships between individuals as regulated by legal norms. Legal practitioners routinely rely on legal relations for legal information retrieval, statute prediction, and case outcome analysis. However, legal relations have long been neglected in the legal AI literature, particularly in the context of Chinese civil law where systematic study is lacking.

**Limitations of Prior Work**: Existing information extraction in legal AI primarily targets factual entities (e.g., persons, objects, contracts) or general social relations (e.g., employment, ownership), overlooking legal relations as a distinct concept rooted in statutory rules and judicial practice rather than ordinary semantic associations in natural language. Moreover, legal relations are almost never explicitly stated in judicial decisions and must typically be inferred from factual descriptions. Existing legal relation schemas tend to be coarse-grained, operating only at the level of broad categories of civil rights and obligations.

**Key Challenge**: The absence of a fine-grained, structured legal relation taxonomy and high-quality annotated data prevents systematic evaluation and improvement of AI models' capacity for legal relation understanding.

**Goal**: (1) Establish the first comprehensive legal relation taxonomy covering Chinese civil law; (2) define the legal relation extraction task and construct an expert-annotated benchmark dataset; (3) evaluate the legal relation extraction capabilities of leading LLMs; (4) validate the utility of legal relation information for downstream tasks.

**Key Insight**: Starting from jurisprudential theory and combining judicial practice with expert guidance, the authors first construct the taxonomy and then proceed to computational annotation and evaluation, balancing legal rigor with AI practicality.

## Method

### Overall Architecture

LexRel is constructed at three levels: (1) **Schema Design**: building a hierarchical legal relation taxonomy and defining argument roles; (2) **Task Definition & Data Annotation**: formalizing the legal relation extraction task (type extraction + argument extraction) and constructing the benchmark via an "LLM pre-annotation + expert correction" pipeline; (3) **Model Evaluation**: evaluating multiple state-of-the-art LLMs under zero-shot and relation-augmented settings.

### Key Designs

1. **Hierarchical Legal Relation Taxonomy**

    - Function: Provides the first fine-grained structured characterization of legal relations in Chinese civil law.
    - Mechanism: Constructed in two stages — (a) relation expressions are extracted from judicial decisions via keyword matching, yielding 123 candidate types drawn from legal terminology references, grouped into 6 domains; (b) two senior legal scholars review and expand the schema, adding 3 new domains (negotiable instrument relations, letter of credit relations, and independent guarantee relations), resulting in a final taxonomy of **9 domains and 265 relation types**.
    - Design Motivation: Combining empirical induction from judgment data with normative grounding in legal theory ensures that the taxonomy is both data-supported and conceptually rigorous.

2. **Legal Relation Extraction Task Definition**

    - Function: Formalizes legal relation recognition as a computable NLP task.
    - Mechanism: Decomposed into two subtasks — **Type Extraction**: identifying the legal relation type $\hat{r} = f_{\text{type}}(x), \hat{r} \in \mathcal{R}$ from factual text $x$; **Argument Extraction**: given the predicted relation type, extracting the subject, object, and content $(\hat{S}, \hat{O}, \hat{c}) = f_{\text{arg}}(x, \hat{r})$.
    - Design Motivation: Legal relations are typically implicit in judicial decisions and must be inferred from factual descriptions, making this task more challenging than conventional relation extraction.

3. **LLM + Expert Hybrid Annotation Pipeline**

    - Function: Ensures annotation quality while controlling cost.
    - Mechanism: DeepSeek-V3 is first used to extract candidate legal relation types and arguments from full judgment texts as draft annotations, which are then verified and corrected by 6 law-trained annotators under the supervision of a senior legal AI expert. After removing 60 samples in which no legal relation could be identified, the final benchmark contains 1,140 annotated instances.
    - Design Motivation: Purely manual annotation of legal relations is prohibitively costly and demands specialized expertise; LLM assistance substantially reduces the annotation burden.

### Evaluation Strategy

- **Zero-shot Baseline**: LLMs are evaluated directly on LexRel for type and argument extraction without task-specific training.
- **Relation-Enhanced Baseline (RE)**: GPT-4o or DeepSeek-R1 generates training data from complete judgment texts (including legal analysis sections), which is used for SFT of open-source smaller models.
- **Argument Evaluation**: LLM-as-Judge (DeepSeek-V3) is employed, with human validation achieving accuracy rates of 95.4% (subject), 96.9% (object), and 81.0% (content).

## Key Experimental Results

### Main Results

| Model | Method | Type micro-F1 | Type macro-F1 | Arg. micro-F1 | Arg. macro-F1 |
|------|------|------------------|------------------|------------------|------------------|
| o3-mini | zero-shot | **0.762** | **0.441** | **0.382** | **0.129** |
| DeepSeek-R1 | zero-shot | 0.693 | 0.376 | 0.268 | 0.065 |
| GPT-4o | zero-shot | 0.670 | 0.314 | 0.224 | 0.068 |
| Claude-Sonnet-4 | zero-shot | 0.590 | 0.330 | 0.258 | 0.088 |
| Qwen3-14B | RE w/ R1 | 0.733 | 0.430 | 0.381 | 0.146 |
| Qwen3-8B | RE w/ R1 | 0.675 | 0.337 | 0.304 | 0.098 |
| Llama3.1-8B | zero-shot | 0.250 | 0.052 | 0.027 | 0.006 |

### Downstream Task Gains

| Model | Case Analysis | Case Analysis+LR | Consultation | Consultation+LR | Damage Calc. | Damage Calc.+LR |
|------|---------|------------|------|---------|---------|------------|
| MiniCPM4-8B | 32.0 | **45.0** | 7.6 | **8.4** | 65.0 | **76.8** |
| DeepSeek-V3 | 66.2 | **68.2** | 16.4 | **16.5** | 85.0 | **86.4** |
| GPT-4o | 55.8 | **56.6** | 18.2 | **19.2** | 84.4 | **85.8** |

### Key Findings

- Reasoning-oriented LLMs (o3-mini, DeepSeek-R1) significantly outperform non-reasoning models in the zero-shot setting, indicating that legal relation extraction demands substantial reasoning capability.
- Argument extraction (highest micro-F1: 0.382) is considerably more difficult than type extraction (0.762); the macro-F1 of all models falls well below their micro-F1, revealing severely inadequate performance on long-tail relation types.
- SFT substantially improves smaller model performance (e.g., InternLM3-8B argument extraction improves from 0.048 to 0.323); Qwen3-14B with RE nearly matches the zero-shot performance of o3-mini.
- The distribution of legal relations in LexRel exhibits a long-tail pattern, closely mirroring the long-tail distribution of case types observed across 26.6 million real civil judgments (the top 25 case types account for 80% of all cases).
- Legal relation information consistently improves downstream task performance: MiniCPM4-8B improves from 32.0 to 45.0 (+13.0) on case analysis and from 65.0 to 76.8 (+11.8) on damage calculation.

## Highlights & Insights

- **First comprehensive civil legal relation taxonomy**: 265 relation types spanning 9 domains, constructed by integrating legal theory with data-driven methods, filling a critical gap in legal AI.
- **Task challenge highlighted**: Legal relations are implicit in judicial decisions and must be inferred from factual descriptions, posing a stringent challenge to LLMs' legal reasoning capabilities.
- **In-depth long-tail analysis**: The representativeness of LexRel's distribution is validated by comparison with the case-type distribution of 26.6 million real judgments.
- **Downstream gains demonstrate practical value**: Even at current extraction accuracy levels, incorporating legal relation information consistently improves downstream task performance.

## Limitations & Future Work

- The schema and dataset focus on Chinese civil law; direct transfer to other legal systems requires localization and adaptation.
- Both synthetic training data generation and LLM-as-Judge evaluation rely on the DeepSeek model family, introducing potential model-family coupling.
- Argument extraction performance remains low (highest macro-F1: 0.146), with long-tail relation types as the primary bottleneck.
- The scale of 1,140 samples is relatively small, particularly given the data sparsity across 265 relation types.

## Related Work & Insights

- **vs. Legal Knowledge Graphs**: Existing legal KGs primarily capture general social relations (e.g., employment, kinship); LexRel targets relations governed by legal norms (e.g., creditor-debtor obligations, contractual duties), which are more closely aligned with judicial practice.
- **vs. General Relation Extraction**: Legal relation extraction is more challenging than general RE, as legal relations are typically not explicitly expressed in text and require inference grounded in legal knowledge.
- **vs. LawBench and similar legal benchmarks**: LexRel focuses on a neglected yet foundational capability — legal relation recognition — and constitutes an important complement to existing legal evaluation benchmarks.

## Rating

- Novelty: ⭐⭐⭐⭐ The first systematic definition and evaluation of Chinese civil legal relation extraction, filling an important gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluates 12 models under two settings, with long-tail analysis and downstream task validation, though the dataset scale is relatively small.
- Writing Quality: ⭐⭐⭐⭐ The integration of legal scholarship and NLP is handled appropriately, and the taxonomy construction process is clearly presented.
- Value: ⭐⭐⭐⭐ Provides an important benchmark resource for the legal AI community and advances the research agenda on legal relation modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Exploring the Capability Boundaries of LLMs in Mastering of Chinese Chouxiang Language](exploring_the_capability_boundaries_of_llms_in_mastering_of_chinese_chouxiang_la.md)
- [\[ACL 2026\] Capabilities and Evaluation Biases of Large Language Models in Classical Chinese Poetry Generation: A Case Study on Tang Poetry](capabilities_and_evaluation_biases_of_large_language_models_in_classical_chinese.md)
- [\[ACL 2026\] E2EDev: Benchmarking Large Language Models in End-to-End Software Development Task](e2edev_benchmarking_large_language_models_in_end-to-end_software_development_tas.md)
- [\[ICLR 2026\] Benchmarking Overton Pluralism in LLMs](../../ICLR2026/llm_evaluation/benchmarking_overton_pluralism_in_llms.md)
- [\[ACL 2026\] ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition](researchbench_benchmarking_llms_in_scientific_discovery_via_inspiration-based_ta.md)

</div>

<!-- RELATED:END -->

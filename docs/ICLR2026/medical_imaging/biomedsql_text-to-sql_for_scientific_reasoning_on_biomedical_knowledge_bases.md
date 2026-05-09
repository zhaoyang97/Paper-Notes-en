---
title: >-
  [Paper Note] BiomedSQL: Text-to-SQL for Scientific Reasoning on Biomedical Knowledge Bases
description: >-
  [ICLR 2026 (Gen2 Workshop)][Medical Imaging][Text-to-SQL] This paper introduces BiomedSQL, the first benchmark specifically designed to evaluate the scientific reasoning capabilities of Text-to-SQL systems on biomedical knowledge bases. It comprises 68,000 question/SQL/answer triples and reveals a substantial gap between the best-performing model (GPT-o3-mini, 62.6%) and domain experts (90%).
tags:
  - ICLR 2026 (Gen2 Workshop)
  - Medical Imaging
  - Text-to-SQL
  - Biomedical Knowledge Base
  - Scientific Reasoning
  - BigQuery
  - LLM Evaluation
date: 2026-05-08
content_hash: 938082e76fc2f2df
---

# BiomedSQL: Text-to-SQL for Scientific Reasoning on Biomedical Knowledge Bases

**Conference**: ICLR 2026 (Gen2 Workshop)  
**arXiv**: [2505.20321](https://arxiv.org/abs/2505.20321)  
**Code**: [https://github.com/NIH-CARD/biomedsql](https://github.com/NIH-CARD/biomedsql)  
**Area**: Medical Imaging  
**Keywords**: Text-to-SQL, Biomedical Knowledge Base, Scientific Reasoning, BigQuery, LLM Evaluation

## TL;DR
This paper introduces BiomedSQL, the first benchmark specifically designed to evaluate the scientific reasoning capabilities of Text-to-SQL systems on biomedical knowledge bases. It comprises 68,000 question/SQL/answer triples and reveals a substantial gap between the best-performing model (GPT-o3-mini, 62.6%) and domain experts (90%).

## Background & Motivation

Modern biomedical research increasingly relies on large-scale structured databases, requiring frequent queries over electronic health records, high-throughput experimental data, and population-scale studies. Natural language interfaces—particularly Text-to-SQL systems—hold promise for enabling non-technical researchers to access these resources.

**Limitations of Prior Work**: Existing Text-to-SQL systems treat query generation as a "syntactic translation" task, mapping question structures to SQL templates without deep domain understanding. In biomedical settings, this abstraction breaks down—domain experts routinely ask questions such as "Which SNPs are significantly associated with Alzheimer's disease?" or "Which approved drugs target genes upregulated in Parkinson's disease?"—queries that implicitly encode domain-specific knowledge such as statistical thresholds (e.g., GWAS significance at $p < 5\times10^{-8}$), drug approval processes, and cross-modal causal reasoning.

**Root Cause**: General-purpose Text-to-SQL benchmarks (e.g., Spider, BIRD) do not evaluate scientific reasoning; EHR-oriented benchmarks (e.g., EHRSQL) focus on temporal logic and patient retrieval rather than the reasoning required for scientific discovery.

**Starting Point**: Construct the first large-scale benchmark specifically targeting the evaluation of scientific reasoning in biomedical Text-to-SQL.

## Method

### Overall Architecture
Input: a natural-language biomedical question plus database schema information. Output: an LLM-generated SQL query → execution to retrieve results → generation of a natural-language answer. Evaluation covers both SQL execution accuracy and natural-language answer quality.

### Key Designs

1. **Relational Database Construction**:

    - Integrates 10 core tables sourced from the OpenTargets Platform (gene–disease–drug associations) and ChEMBL (bioactive molecules and pharmacological data).
    - Incorporates GWAS summary statistics for Alzheimer's disease and Parkinson's disease (including p-values, rsIDs, allele frequencies, etc.).
    - Integrates causal inference data from omicSynth (multi-omics biomarkers derived via Mendelian randomization).
    - All data are uploaded to Google BigQuery in Parquet format.

2. **SQL Annotation and Augmentation**:

    - Domain experts manually authored gold-standard SQL queries for 40 seed questions.
    - These 40 queries are automatically expanded to 68,000 QA pairs via templatization and entity substitution.
    - All generated SQL queries are executed on BigQuery to obtain ground-truth results.

3. **BMSQL Multi-Step Agent**:

    - A custom iterative Text-to-SQL architecture that emulates expert query workflows.
    - Step 1: Schema analysis to identify relevant tables and columns.
    - Step 2: Generation of an initial SQL query.
    - Step 3: Syntax error correction if needed (up to 3 retries).
    - Step 4: Application of statistical threshold filters (e.g., p-value significance).
    - Step 5: Natural-language answer generation based on two sets of execution results.
    - Optional additional inference-time compute steps.

### Scientific Reasoning Taxonomy
Three major reasoning categories:
1. **Operationalizing Implicit Scientific Conventions**: Requires inferring GWAS significance thresholds, effect directionality, etc.
2. **Integrating Missing Contextual Knowledge**: Requires understanding drug approval status, clinical trial phases, etc.
3. **Executing Complex Multi-Hop Reasoning**: Requires chaining relationships across multiple tables.

### Evaluation Metrics
- Execution Accuracy (EX): exact-match rate of SQL execution results.
- Jaccard Index (JAC): intersection-over-union of result sets.
- Syntax Error Rate (SER): rate of syntactically invalid queries.
- BioScore (LLM-as-judge): Response Quality Rate (RQR) + Safety Rate (SR).

## Key Experimental Results

### Main Results

| Model | EX↑ | JAC↑ | RQR↑ | SR↑ | SER↓ |
|-------|-----|------|------|-----|------|
| Domain Expert | 90.0 | 90.0 | 95.0 | - | - |
| GPT-o3-mini | 53.5 | 60.4 | 73.3 | 29.4 | 0.0 |
| GPT-4o | 46.9 | 54.7 | 71.2 | 26.1 | 1.3 |
| Claude-3.7-sonnet | - | - | - | 43.0 | - |
| Qwen-2.5-Coder-32B | 40.8 | - | - | - | - |
| **BMSQL-GPT-o3-mini** | **62.6** | **69.2** | **83.1** | 38.0 | 2.6 |
| **BMSQL-Gemini** | - | - | **84.6** | - | - |

### Interaction Paradigm Experiments

| Method | EX↑ | JAC↑ | RQR↑ |
|--------|-----|------|------|
| ReAct-GPT-o3-mini | 56.2 | 64.8 | 73.6 |
| Index-GPT-o3-mini | - | - | - (highest SR 66.9%) |
| BMSQL-GPT-o3-mini | 62.6 | 69.2 | 83.1 |

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|-----------|------|
| Combo prompt (3-rows + 3-shot + stat-instruct) | EX +5.5%, RQR +4.5% | Token consumption increases ~3× |
| 1-pass vs. 3-pass (inference-time compute) | EX 62.6→61.7, RQR 83.1→85.5 | Marginal gains from additional reasoning steps |
| Adding table row samples alone | Negligible improvement | Schema understanding > content memorization |

### Key Findings
- The best-performing model, BMSQL-GPT-o3-mini, achieves only 62.6% EX, leaving a ~30% gap relative to the expert baseline of 90%.
- Join, Similarity Search, and Multi-Filter query types are the most challenging.
- Additional inference-time compute steps yield minimal improvement, primarily correcting syntax errors rather than restructuring query logic.
- Schema-level understanding is more important than memorizing raw data rows.
- The smaller Qwen-2.5-Coder model outperforms much larger LLaMA models on several metrics.

## Highlights & Insights
- BiomedSQL is the first Text-to-SQL benchmark focused on scientific reasoning in the biomedical domain, filling an important gap.
- The 68,000-question scale is substantial, and each question requires implicit domain knowledge reasoning.
- The multi-dimensional evaluation framework is well-designed, combining SQL execution metrics with natural-language answer quality.
- BMSQL's multi-stage design emulates expert query workflows and substantially outperforms single-step approaches.
- The work reveals significant deficiencies in current LLMs' ability to operationalize domain-specific scientific conventions.
- Use of BigQuery simulates real production environments, enhancing relevance for practical deployment.

## Limitations & Future Work
- Gold SQL queries are not uniquely correct; multiple semantically equivalent SQL expressions may exist.
- General Text-to-SQL systems such as DIN-SQL and DAIL-SQL are not evaluated due to incompatibility with the BigQuery dialect.
- Reliance on the BigQuery cloud-specific dialect limits direct comparability with other benchmarks.
- Dataset expansion via templates may introduce systematic biases.
- Domain coverage is largely restricted to neurodegenerative diseases (Alzheimer's and Parkinson's).

## Related Work & Insights
- Complements general-purpose Text-to-SQL benchmarks such as Spider and BIRD by focusing on the scientific reasoning dimension.
- Differs from clinical benchmarks such as EHRSQL and MIMICSQL, which target patient retrieval rather than scientific knowledge discovery.
- Related to scientific reasoning benchmarks such as SciFact and EntailmentBank, but evaluates SQL generation capability.
- The proposed approach could be extended to knowledge base querying in other domains (e.g., materials science, environmental science).

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] MedAgentGym: A Scalable Agentic Training Environment for Code-Centric Reasoning in Biomedical Data Science](medagentgym_agentic_training_biomedical.md)
- [\[NeurIPS 2025\] CGBench: Benchmarking Language Model Scientific Reasoning for Clinical Genetics Research](../../NeurIPS2025/medical_imaging/cgbench_benchmarking_language_model_scientific_reasoning_for_clinical_genetics_r.md)
- [\[ICLR 2026\] Augmenting Representations with Scientific Papers](augmenting_representations_with_scientific_papers.md)
- [\[NeurIPS 2025\] Mind the Gap: Aligning Knowledge Bases with User Needs to Enhance Mental Health Retrieval](../../NeurIPS2025/medical_imaging/mind_the_gap_aligning_knowledge_bases_with_user_needs_to_enhance_mental_health_r.md)
- [\[ICLR 2026\] Tracing Pharmacological Knowledge in Large Language Models](tracing_pharmacological_knowledge_in_large_language_models.md)

<!-- RELATED:END -->

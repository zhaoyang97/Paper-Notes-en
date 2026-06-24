---
title: >-
  [Paper Note] MDBench: A Synthetic Multi-Document Reasoning Benchmark Generated with Knowledge Guidance
description: >-
  [ACL 2025][LLM Evaluation][Multi-document reasoning] Proposes MDBench, a multi-document reasoning QA benchmark synthesized through a "structured knowledge &rarr; LLM-assisted enhancement &rarr; natural text generation" pipeline. It controllably injects cross-document dependencies and poses a significant challenge to frontier LLMs (the best model achieves only ~60% EM).
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Multi-document reasoning"
  - "synthetic data"
  - "knowledge-guided generation"
  - "QA benchmark"
date: 2026-05-08
content_hash: 50297e353f504be9
---

# MDBench: A Synthetic Multi-Document Reasoning Benchmark Generated with Knowledge Guidance

**Conference**: ACL 2025  
**arXiv**: [2506.14927](https://arxiv.org/abs/2506.14927)  
**Code**: [GitHub](https://github.com/jpeper/MDBench)  
**Area**: NLP / Multi-Document Reasoning / Evaluation Benchmark  
**Keywords**: Multi-document reasoning, synthetic data, knowledge-guided generation, LLM evaluation, QA benchmark

## TL;DR

Proposes MDBench, a multi-document reasoning QA benchmark synthesized through a "structured knowledge &rarr; LLM-assisted enhancement &rarr; natural text generation" pipeline. It controllably injects cross-document dependencies and poses a significant challenge to frontier LLMs (the best model achieves only ~60% EM).

## Background & Motivation

Multi-document reasoning is a core and increasingly crucial capability in LLM applications, where models must synthesize information from multiple text sources for reasoning and question answering. However, current LLM evaluations face three critical issues:

**Scarcity of Benchmarks**: Although LLMs' capability to process long contexts is rapidly improving, benchmarks specifically testing multi-document reasoning are sparse. Existing datasets like HotPotQA and MuSiQue primarily target multi-hop QA, but they often rely on public data and suffer from extremely high annotation costs.

**Risk of Data Contamination**: Static, manually annotated benchmarks are highly susceptible to being "seen" by LLMs during pre-training, rendering the evaluation ineffective.

**Annotation Cost**: Annotating multi-document scenarios involves long-text comprehension, leading to prohibitively high human annotation costs and making scalability difficult.

The core idea proposed by the authors is: instead of starting directly from natural text, they use **compressed structured knowledge** (tabular data) as seeds, inject reasoning dependencies through LLM-assisted editing, and then transform the structured knowledge into natural language document sets. This "structure-first, generate-later" approach simultaneously solves the challenges of controllability, novelty, and scalability.

## Method

### Overall Architecture

The generation pipeline of MDBench consists of four steps:

1. **Acquire Seed Knowledge** &rarr; 2. **Knowledge Enhancement** (injecting reasoning dependencies) &rarr; 3. **Natural Text Generation** &rarr; 4. **Automated Quality Verification**

### Key Designs

1. **Seed Knowledge Source (Step 1)**: Wikipedia tables from the TabFact dataset are utilized, specifically selecting tables with 5-17 rows and 3-9 columns. Each row eventually corresponds to a document in the generated document set, where structured attributes are provided by table columns and row-to-row relationships naturally align with cross-document dependencies.

2. **Knowledge Enhancement (Step 2)**: This is the core innovation of the pipeline. GPT-4o is employed to edit tables and inject five categories of multi-document reasoning challenges:

    - **Multi-hop reasoning**: Requires chain deduction across multiple rows (documents).
    - **Numerical reasoning**: Involves numerical calculations and comparisons.
    - **Temporal reasoning**: Deals with temporal dependencies and sequential relationships.
    - **Knowledge aggregation**: Requires aligning, comparing, or contrasting information from multiple sources.
    - **Soft reasoning**: Requires making inferences under uncertainty (e.g., cross-document entity linking).
   
   The enhancement process includes two types of demonstrations: (a) reasoning skill demonstrations (showing simple and difficult examples for each skill), and (b) knowledge editing demonstrations (showing how to edit a simple table into a complex QA example). Corresponding question-answer pairs are also generated during this step.

3. **Document Set Generation (Step 3)**: Each row of the enhanced table is independently transformed into a natural language document. The generation of each document is conditioned on the enhanced table, column names, and corresponding row content, ensuring logical consistency in the generated documents. The length increases by approximately 9 times from the structured table to natural text (256 &rarr; 2397 tokens).

4. **Automatic Quality Verification (Step 4)**: A two-tier verification mechanism is introduced:

    - Target consistency check: Verifies whether each component (seed table, editing plan, edited table, document set) aligns with instructions.
    - **Oracle Self-Consistency Check**: Employs GPT-4o to answer questions under three "known-answer" variations (original table + editing plan / generated table / generated document set). A sample is retained only if the answers from all three variations are consistent. This filter retains approximately 32% of the generated samples.

### Dataset Statistics

The final benchmark contains 1,000 multi-document QA samples (300 human-verified, 700 machine-verified). The validation rate of human verification reaches 87%. On average, each document set contains 8.31 documents, with an average document length of 268 tokens, totaling an average context of 2,397 tokens.

## Key Experimental Results

### Main Results: Multi-Document Reasoning (Exact Match)

| Model | Zero-shot | ZS CoT | 1-shot | 1-shot CoT | Overall |
|------|-----------|--------|--------|------------|------|
| LLaMA-3-8B | 42.7 | 43.1 | 39.3 | 38.9 | 41.0 |
| LLaMA-3-70B | 51.2 | 50.2 | 45.0 | 49.3 | 48.9 |
| GPT-3.5-Turbo | 49.8 | 37.4 | 44.5 | 36.5 | 42.1 |
| Claude-3.5-Sonnet | 59.2 | 56.4 | 58.3 | 55.9 | 57.5 |
| Gemini-2.5-Flash | 57.8 | 58.3 | 58.8 | 60.2 | 58.8 |
| **GPT-4o** | **59.7** | **62.1** | 59.7 | 58.3 | **60.0** |
| GPT-o1 | 57.3 | 60.7 | 59.7 | 59.2 | 59.2 |

### Ablation Study: Document vs. Table Reasoning

| Model | Document EM | Table EM | Difference |
|------|---------|---------|------|
| GPT-4o | 60.0 | 71.2 | -11.2 |
| GPT-o1 | 59.2 | 67.9 | -8.7 |
| Claude-3.5-Sonnet | 57.5 | 66.9 | -9.4 |
| LLaMA-3-70B | 48.9 | 52.4 | -3.5 |
| LLaMA-3-8B | 41.0 | 35.8 | +5.2 |

Document shuffling ablation (GPT-4o Zero-shot EM):

| Condition | GPT-4o | GPT-3.5 |
|------|--------|---------|
| Original | 59.7 | 49.8 |
| Remove separators | 55.5 | 45.0 |
| Shuffle order | 53.1 | 39.3 |
| Remove separators + Shuffle | 50.2 | 41.7 |

### Key Findings

1. **MDBench poses a significant challenge to frontier models**: The best EM is only 60% (GPT-4o), and the best Accuracy is 82.2% (GPT-o1), indicating that multi-document reasoning remains far from solved.
2. **Surface form affects reasoning**: All models experience a substantial performance drop when transitioning from structured tables to natural text documents (GPT-4o decreases by 11.2% EM). This indicates that multi-document reasoning is influenced not only by underlying logical complexity but also by the interference of surface forms.
3. **Sensitivity to document ordering**: Removing separators and shuffling document order both lead to degraded performance, confirming the existence of real cross-document temporal and sequential dependencies in MDBench.
4. **CoT benefits strong models but has limited effects on weaker ones**: GPT-4o and Gemini-2.5-Flash benefit from CoT, whereas LLaMA-3-8B and GPT-3.5 show little to no improvement.
5. **LLaMA exhibits anomaly weakness on tabular reasoning**: LLaMA-3-8B performs even worse in table reasoning than in document reasoning, which might relate to its capability in processing structured formats.

## Highlights & Insights

1. **The "structure-first, generate-later" pipeline design is elegant**: Using tables as an intermediate representation naturally controls cross-document dependencies, resolving a core challenge in constructing multi-document benchmarks.
2. **Oracle self-consistency validation**: Utilizing information inherent in the construction process itself to verify consistency provides a low-cost, high-quality approach to automated validation.
3. **Provides a table-vs-document comparison perspective**: Discrepancies in performance for the exact same reasoning problem across two surface forms reveal the entanglement between "understanding" and "format adaptation" in LLM reasoning.
4. **Controllable reasoning types**: Reasoning skill types are specified through demonstrations, allowing future extensions to easily introduce new reasoning challenges.

## Limitations & Future Work

1. **Narrow data domains**: Seed data originates from Wikipedia tables, covering general domains like sports, politics, and technology, while lacking specialized fields such as law or medicine.
2. **Reliance on GPT-4o for generation and validation**: This might introduce specific preferences and limitations characteristic of this model.
3. **Non-parallel multilingualism**: While English is used, multilingual translation/extension would require reconsidering cultural differences and linguistic properties.
4. **Counterfactual adjustments might not be extreme enough**: Current "mild counterfactual adjustments" may not fully eliminate the risk of pre-existing data contamination.
5. **Small scale with 1,000 samples**: Establishing robust model rankings may require larger-scale benchmarks.

## Related Work & Insights

- **HotPotQA / MuSiQue / FanOutQA**: Classic multi-document QA benchmarks, but heavily reliant on human annotation.
- **MuSR**: Generated multi-step reasoning benchmarks using neurosymbolic methods, sharing a similar concept but targeting single-document contexts.
- **TabFact / Chain-of-Tables**: Research in tabular reasoning, which inspired the use of tables as intermediate representations in this work.
- **Synthetic Benchmark Trend**: As data contamination becomes a growing concern, LLM-driven synthetic benchmark creation is becoming an important research direction.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The knowledge-guided synthetic pipeline design is novel, and the Oracle self-consistency check is highly ingenious.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers multiple model families, varied prompting strategies, table-vs-document comparative studies, and document shuffling ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The pipeline is clearly described, diagrams are intuitive, and experimental analyses provide depth.
- **Value**: ⭐⭐⭐⭐ — Provides a controllable, scalable new method for multi-document reasoning evaluation, making a practical contribution to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] KITAB-Bench: A Comprehensive Multi-Domain Benchmark for Arabic OCR and Document Understanding](kitab-bench_a_comprehensive_multi-domain_benchmark_for_arabic_ocr_and_document_u.md)
- [\[ACL 2025\] READoc: A Unified Benchmark for Realistic Document Structured Extraction](readoc_a_unified_benchmark_for_realistic_document_structured_extraction.md)
- [\[ACL 2025\] MARS: Benchmarking the Metaphysical Reasoning Abilities of Language Models with a Multi-task Evaluation Dataset](mars_benchmarking_the_metaphysical_reasoning_abilities_of_language_models_with_a.md)
- [\[ACL 2025\] SANSKRITI: A Comprehensive Benchmark for Evaluating Language Models' Knowledge of Indian Culture](sanskriti_a_comprehensive_benchmark_for_evaluating_language_models_knowledge_of_.md)
- [\[ACL 2025\] GuessArena: Guess Who I Am? A Self-Adaptive Framework for Evaluating LLMs in Domain-Specific Knowledge and Reasoning](guessarena_guess_who_i_am_a.md)

</div>

<!-- RELATED:END -->

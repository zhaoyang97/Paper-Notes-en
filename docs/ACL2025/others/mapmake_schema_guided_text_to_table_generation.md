---
title: >-
  [Paper Note] Map&Make: Schema Guided Text to Table Generation
description: >-
  [ACL 2025][Text-to-Table] This work proposes the Map&Make method, which first deconstructs unstructured text into propositional atomic statements (Map phase) and then derives the table schema and populates data based on them (Make phase), significantly improving text-to-table quality and interpretability on both Rotowire and Livesum scenarios.
tags:
  - "ACL 2025"
  - "Text-to-Table"
  - "Schema-Guided"
  - "Propositional Decomposition"
  - "Structured Summarization"
  - "Information Extraction"
date: 2026-05-08
content_hash: d4f11dcccdc7b665
---

# Map&Make: Schema Guided Text to Table Generation

**Conference**: ACL 2025  
**arXiv**: [2505.23174](https://arxiv.org/abs/2505.23174)  
**Code**: None  
**Area**: Other  
**Keywords**: Text-to-Table, Schema-Guided, Propositional Decomposition, Structured Summarization, Information Extraction

## TL;DR

This work proposes the Map&Make method, which first deconstructs unstructured text into propositional atomic statements (Map phase) and then derives the table schema and populates data based on them (Make phase), significantly improving text-to-table quality and interpretability on both Rotowire and Livesum scenarios.

## Background & Motivation

**Background**: Text-to-table generation is an important task in information retrieval, aiming to transform dense, unstructured text into interpretable, structured tables. Existing approaches mostly utilize LLMs or sequence-to-sequence models to generate tables directly from text, relying on the models to implicitly understand the textual structure.

**Limitations of Prior Work**: Existing approaches suffer from two critical limitations. First, they lack explicit guidance on "what to extract" and "how to organize information," leading to unstable table structures, inconsistent column names, and missing or redundant data. Second, LLMs are prone to hallucinations when directly generating tables—creating non-existent data, especially when handling complex multi-table scenarios.

**Key Challenge**: Information in text is implicit, nested, and vague, whereas tables require explicit, flat, and precise information. Transitioning from implicit to explicit requires intermediate stages, yet existing methods attempt to achieve this in a single step, crossing a massive semantic gap.

**Goal**: To design a staged framework that bridges the semantic gap between unstructured text and structured tables through an intermediate representation (propositional atomic statements).

**Key Insight**: Inspired by propositional logic, complex text can be decomposed into a series of atomic propositions, each describing a single fact. Such fine-grained decomposition makes subsequent schema inference and data filling more controllable.

**Core Idea**: To replace direct text-to-table mapping with a three-step pipeline of "propositional atomic statement decomposition $\rightarrow$ schema inference $\rightarrow$ table filling," making each step interpretable and verifiable.

## Method

### Overall Architecture

Map&Make consists of two core phases. The **Map phase** decomposes the input text into a set of propositional atomic statements, with each statement describing an independent unit of fact. The set of decomposed statements forms a "fact checklist" of the text. The **Make phase** automatically infers the table schema (i.e., column names and table structure) from the fact checklist, and then extracts data from the atomic statements to fill the table based on the schema.

### Key Designs

1. **Propositional Decomposition**:

    - **Function**: Deconstructing complex text into minimal information units.
    - **Mechanism**: Using LLMs to decompose paragraphs into a series of short atomic propositions. Each proposition contains only one subject-verb-object relationship or a decimal fact. For example, "LeBron James scored 15 points and grabbed 4 rebounds in the second quarter" would be decomposed into two atomic statements: "LeBron James scored 15 points in the second quarter" and "LeBron James grabbed 4 rebounds in the second quarter." Semantic faithfulness to the original text is preserved during decomposition without adding hypothetical information.
    - **Design Motivation**: Fine-grained decomposition allows each piece of information to be verified independently, significantly reducing ambiguity and the risk of hallucination during subsequent table filling. Meanwhile, atomic statements naturally map to individual table cells.

2. **Schema Inference**:

    - **Function**: Automatically determining the column names and structure of tables.
    - **Mechanism**: When a reference schema is available (schema-guided), predefined column names are directly used; in the open-schema setting (open schema), latent table columns are auto-derived by clustering the predicates/attributes in the atomic statements. For complex multi-table scenarios (e.g., players' stats + team stats in Rotowire), atomic statements are grouped based on subject type, and schemas are inferred independently for each group.
    - **Design Motivation**: The explicit schema inference step ensures the consistency and completeness of the table structure, avoiding the issue of chaotic column names during direct generation.

3. **Hallucination Correction and Quality Control**:

    - **Function**: Improving the factual accuracy of the generated tables.
    - **Mechanism**: On the Rotowire dataset, the authors carefully identified and corrected hallucination errors in the original annotations (such as incorrect statistical figures), establishing a cleaner evaluation benchmark. During the generation phase, by mapping atomic statements one-to-one to table cells, they ensured that every filled value was supported by the source text. For scenarios requiring numerical aggregation (e.g., statistical summaries in Livesum), explicit calculation steps are used instead of relying on the model's implicit inference.
    - **Design Motivation**: Hallucination is a core issue in text-to-table tasks. Through the traceable mapping from atomic statements to cells, each data point can be traced back and verified.

### Loss & Training

Map&Make primarily relies on LLM prompt engineering and does not involve model fine-tuning. The decomposition rules, schema formats, and filling instructions are described in detail in the prompts. Few-shot exemplars are used to guide the LLM in understanding the task requirements.

## Key Experimental Results

### Main Results

| Dataset | Method | F1 | Precision | Recall | NTD↓ |
|--------|------|-----|-----------|--------|------|
| Rotowire | GPT-4 Direct | 48.3 | 52.1 | 45.0 | 0.42 |
| Rotowire | Map&Make | 63.7 | 67.2 | 60.5 | 0.28 |
| Rotowire | Previous SOTA | 55.2 | 58.6 | 52.1 | 0.35 |
| Livesum | GPT-4 Direct | 41.5 | 44.8 | 38.7 | 0.51 |
| Livesum | Map&Make | 56.2 | 60.1 | 52.8 | 0.34 |
| Livesum | Previous SOTA | 47.8 | 50.3 | 45.6 | 0.43 |

### Ablation Study

| Configuration | Rotowire F1 | Livesum F1 | Description |
|------|------------|------------|------|
| Full Map&Make | 63.7 | 56.2 | Full framework |
| w/o Decomposition | 52.1 | 44.3 | Infer schema directly from text |
| w/o Schema Inference | 57.4 | 49.6 | Use fixed schema |
| w/o Hallucination Fix | 59.8 | 53.1 | Without correcting annotation errors |
| Random Decomposition | 48.9 | 41.7 | Random splitting instead of semantic decomposition |

### Key Findings
- **Propositional decomposition makes the greatest contribution**: Removing the decomposition step leads to an F1 drop of over 11% (Rotowire) and 12% (Livesum), highlighting the critical role of the intermediate representation.
- **More significant improvement on Livesum**: This dataset requires numerical aggregation. Propositional decomposition clarifies the computation steps and reduces arithmetic errors.
- **Rotowire annotation quality issues**: After correcting the hallucination errors in the original annotations, the absolute scores of all methods improved, indicating that the quality of the evaluation benchmark itself significantly impacts research conclusions.
- **Effective under open schema**: Even without a predefined table schema, the quality of automatically inferred schemas remains highly satisfactory, demonstrating the generalizability of the framework.

## Highlights & Insights
- **Propositional decomposition serves as a general intermediate step**: This approach is not limited to text-to-table and can be generalized to any task requiring structured details extracted from unstructured text, such as knowledge graph construction, event extraction, and relation extraction. Decomposition into atomic statements essentially standardizes information granularity.
- **A responsible approach to correcting evaluation set annotation errors**: While many works choose to bypass identified issues in evaluation datasets, this paper directly corrects the errors and releases the reformed version, providing the community with a more reliable benchmark.
- **High interpretability**: Each step yields explicit intermediate outputs, allowing precise identification of whether errors originate from the decomposition, schema inference, or table filling phase during debugging and error analysis.

## Limitations & Future Work
- **Dependence on LLM decomposition quality**: If the proposition decomposition by the LLM contains errors (omissions or inaccuracy), these errors cascade to subsequent stages.
- **High computational cost**: The three steps—decomposition, schema inference, and filling—all require individual LLM calls, making the inference overhead multiple times higher than direct methods.
- **Validation limited to English datasets**: The performance of propositional decomposition in multilingual scenarios has not been evaluated.
- **Limited support for complex nested tables**: For multi-layer nested table structures, the current schema inference may be inadequate.
- Future work can explore lightweight decomposition models to replace large LLMs, as well as multilingual propositional decomposition.

## Related Work & Insights
- **vs TAPAS/TaPEx**: These methods are designed for table understanding (Table QA) in the opposite direction—answering questions from tables. Map&Make focuses on "constructing tables from text," which can be viewed as an upstream task of table understanding.
- **vs Traditional Information Extraction (IE)**: IE methods typically require predefined schemas (such as relation triples), whereas Map&Make can automatically infer schemas, offering higher flexibility. However, in terms of precision, dedicated IE models may still perform better in specific domains.
- **vs Chain-of-Table**: Chain-of-Table focuses on multi-step reasoning for table manipulation, while Map&Make targets the table generation phase. The two can be cascaded.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of using propositional decomposition as an intermediate representation is simple yet effective, and the schema-guided design has practical value.
- Experimental Thoroughness: ⭐⭐⭐⭐ The comparative experiments and ablation studies on the two datasets are comprehensive, though dataset diversity could be further expanded.
- Writing Quality: ⭐⭐⭐⭐ The method is described clearly, with explicit motivations for each step of the pipeline.
- Value: ⭐⭐⭐⭐ Inspiring for structured information extraction, and the propositional decomposition approach is widely applicable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] TabXEval: Why this is a Bad Table? An eXhaustive Rubric for Table Evaluation](tabxeval_why_this_is_a_bad_table_an_exhaustive_rubric_for_table_evaluation.md)
- [\[ACL 2025\] Towards Text-Image Interleaved Retrieval](towards_text-image_interleaved_retrieval.md)
- [\[ACL 2025\] MapQaTor: An Extensible Framework for Efficient Annotation of Map-Based QA Datasets](mapqator_an_extensible_framework_for_efficient_annotation_of_map-based_qa_datase.md)
- [\[ACL 2025\] DAPE V2: Process Attention Score as Feature Map for Length Extrapolation](dape_v2_process_attention_score_as_feature_map_for_length_extrapolation.md)
- [\[ACL 2025\] FRACTAL: Fine-Grained Scoring from Aggregate Text Labels](fractal_fine-grained_scoring_from_aggregate_text_labels.md)

</div>

<!-- RELATED:END -->

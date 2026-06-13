---
title: >-
  [Paper Note] When TableQA Meets Noise: A Dual Denoising Framework for Complex Questions and Large Tables
description: >-
  [ACL 2026][LLM/NLP][TableQA] By decomposing semantic units in questions and constructing evidence trees for transparent table pruning…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "TableQA"
  - "Data Denoising"
  - "LLM Reasoning"
  - "Evidence Filtering"
  - "Table Pruning"
date: 2026-05-08
content_hash: f9534e0cc8354288
---

# When TableQA Meets Noise: A Dual Denoising Framework for Complex Questions and Large Tables

**Conference**: ACL 2026  
**arXiv**: [2509.17680](https://arxiv.org/abs/2509.17680)  
**Code**: Not provided  
**Area**: LLM / NLP / TableQA  
**Keywords**: TableQA, Data Denoising, LLM Reasoning, Evidence Filtering, Table Pruning

## TL;DR
By decomposing semantic units in questions and constructing evidence trees for transparent table pruning, the EnoTab framework achieves significant performance gains when handling complex questions and ultra-large tables, effectively mitigating the negative impact of noisy data on reasoning through a dual denoising mechanism.

## Background & Motivation

**Background**: With advancements in LLM reasoning capabilities, Table Question Answering (TableQA) has become a core task in NLP. However, in practical applications (such as finance and healthcare), question complexity and table scale have increased significantly, leading to a substantial rise in data noise.

**Limitations of Prior Work**: Two core problems exist:

- Complex questions contain spurious correlations (e.g., "in Vienna" when such data is absent from the table), which easily mislead LLMs.
- In ultra-large tables, only 1-2% of rows are relevant to the answer, while the remaining data acts as noise that interferes with reasoning.

**Key Challenge**: Existing methods struggle to balance "accurate denoising" and "retaining necessary information." Program-generator-based table pruning methods often rely on black-box decisions; once an error occurs in deletion, the process must be entirely restarted. Question decomposition methods are prone to misjudging spurious correlations.

**Goal**: Propose a framework that can effectively identify irrelevant parts of a question while making the table pruning process transparent.

**Key Insight**: The authors observe that effective TableQA requires two capabilities: (1) Relevance filtering—identifying and ignoring spurious correlations in questions; (2) Table pruning—removing irrelevant data while retaining all information required for the answer. These two capabilities are independent yet complementary.

**Core Idea**: Through the explicit extraction and evaluation of "evidence"—the minimum semantic unit—question denoising and table denoising are implemented separately. In the table pruning stage, an Evidence Tree is introduced as an observable execution path, making every step verifiable and reversible.

## Method

### Overall Architecture

EnoTab operates in three stages:

1. **Evidence Generation and Evaluation**: The LLM decomposes complex questions into minimum semantic units (evidence), where each evidence $e=(area, condition, action)$ represents an attribute constraint in the question. For example, "cities in Tel Aviv" corresponds to $(District\ \text{column}, Tel\ Aviv, \text{string match})$. Reliability is filtered through multi-round consistency checks and usability tests.
2. **Evidence Tree Construction and Execution**: A binary tree is constructed based on the reliable evidence set, where leaf nodes are evidence (executing a single filter) and internal nodes are logical relations (AND/OR). The pruned sub-table is obtained through post-order traversal. If an AND node produces an empty table, a rollback mechanism is triggered.
3. **Answer Generation**: The cleaned evidence and sub-tables are sent to the LLM to generate the final answer.

### Key Designs

1. **Dual Evaluation of Evidence Consistency and Usability**:

    - **Function**: Determine if each decomposed semantic unit is truly relevant to the answer reasoning.
    - **Mechanism**: Instead of relying directly on LLM scores, two objective criteria are used. Consistency is checked via 5 rounds of generation—evidence appearing stably ($S \geq 0.8$) is considered important. Usability is checked via a toolkit $\mathcal{P}$ to see if the evidence matches actual data in the table. Only evidence passing both tests is retained.
    - **Design Motivation**: Addressing the "spurious correlation" pain point in question decomposition. Consistency mimics the stability of human reasoning, while usability ensures evidence is supported by table data, preventing hallucinations.

2. **Post-order Rollback Mechanism of Evidence Tree**:

    - **Function**: Detect and repair abnormal states (e.g., empty tables) during pruning to prevent accidental deletion of key data.
    - **Mechanism**: During post-order traversal, if an AND node results in an empty table, an And2Or operation is executed: changing AND to OR to obtain a broader superset. If it remains empty or the validator $M_i$ determines the information is incomplete, the system rolls back to the previous node's result and retries (up to 2 times).
    - **Design Motivation**: Unlike black-box SQL methods that require a full restart upon error, every step in an Evidence Tree is observable. The key insight of And2Or is that for TableQA, recall is more important than precision (missing evidence cannot be recovered, but extra rows can be filtered later).

3. **Two-stage Retrieval to Optimize Evidence Generation Efficiency**:

    - **Function**: Efficiently select k=10 representative rows from ultra-large tables as context for evidence generation.
    - **Mechanism**: Stage one uses LSH for coarse keyword filtering. Stage two calculates the semantic similarity between candidate rows and the question: $Score(r,Q) = 0.7 \cdot S_{sem}(r,Q) + 0.3 \cdot S_{lex}(r,Q)$, where $S_{sem}$ uses pre-trained embedding models and $S_{lex}$ uses edit distance.
    - **Design Motivation**: Direct evidence generation on a full table is computationally expensive and error-prone; using representative samples maintains quality while significantly reducing costs.

### Loss & Training

This method is entirely training-free, utilizing off-the-shelf LLMs (GPT-4o/4o-mini/LLaMA) and tools (embedding models, symbolic matching tools).

## Key Experimental Results

### Main Results

Evaluated on two large-scale table datasets (STQA-L for natural ultra-large tables, STQA-N for noise-injected tables) against several baselines:

| Method | STQA-N (GPT-4o) | STQA-N (GPT-4o-mini) | STQA-L (GPT-4o) | STQA-L (GPT-4o-mini) | Average Gain |
|------|---------|---------|---------|---------|---------|
| TabLaP | 73.6 | 70.8 | 69.1 | 65.9 | - |
| **EnoTab (Ours)** | **80.3** | **78.2** | **75.3** | **72.5** | **+8.7%** |
| Relative Gain | +8.3% | +9.5% | +8.2% | +9.1% | - |

### Ablation Study

| Configuration | STQA-N Drop | STQA-L Drop | Description |
|------|---------|---------|---------|
| Full Model | - | - | - |
| W/O Consistency Eval | -6.3% | -5.6% | Spurious evidence is retained |
| W/O Usability Eval | -8.2% | -6.8% | Evidence not in the table is applied |
| W/O And2Or Rollback | -4.2% | -3.8% | Empty tables cannot recover; data loss |
| W/O Table Validator | -4.2% | -2.5% | No integrity check; may miss answer data |

Conclusion: Consistency and usability evaluations contribute the most (5-8% drops each), indicating that the two-stage question denoising is the core of the framework.

### Standard Benchmark Results

Performance on WikiTQ and TabFact:

| Method | WikiTQ | TabFact | Average |
|------|--------|---------|------|
| Chain-of-Table | 67.1 | 84.2 | 75.7 |
| TabLaP | 72.8 | 86.9 | 79.9 |
| **EnoTab (Ours)** | **74.6** | **89.2** | **81.9** |
| Relative Gain | +1.8% | +2.3% | +2.0% |

Table compression analysis: EnoTab achieves significant compression while maintaining accuracy. STQA-L dropped from an average of 8,967 tokens to 2,176 (75.7% compression), and STQA-N dropped from 26,742 tokens to 2,893 (89.2% compression).

### Key Findings

- **Adaptability to Complex Questions**: In WikiTQ, categorized by GPT-4o difficulty levels (Easy/Medium/Hard/ExtraHard), EnoTab maintains a clear advantage in ExtraHard scenarios.
- **Cross-model Robustness**: Comparing closed-source (GPT-4o/4o-mini) and open-source models (LLaMA-2-70B/Qwen-1.5-70B), end-to-end QA dropped 15% from closed to open-source, while EnoTab only dropped 3-5%.
- **Noise Resistance**: After injecting distracting content into WikiTQ, end-to-end QA dropped by over 20%, while EnoTab remained stable.

## Highlights & Insights

- **Innovation of Evidence as Minimum Unit**: Unlike traditional decomposition that breaks down a question into sub-questions, this paper decomposes it into $(column, condition, action)$ triples. This fine-grained abstraction significantly reduces judgment complexity.
- **Observable Table Pruning Path**: The Evidence Tree transforms a black-box pruning process into explicit binary tree execution, where every node is verifiable and every step is reversible. Combined with the And2Or rollback, it finds a practical balance between precision and recall.
- **Training-free Plug-and-play Solution**: Based entirely on off-the-shelf LLMs and tool combinations, it requires no fine-tuning or new parameters, greatly reducing deployment costs.

## Limitations & Future Work

- **Limitations in Handling Composite Values**: Insufficient handling of special formats like "1-1" (win-loss records) or "251-32=189" (composite calculations).
- **Single Table Assumption**: Currently focused on single-table QA. Performance on multi-table relational questions remains unclear.
- **Validator Limitations**: The integrity validator $M_i$ sees performance degradation on certain structured data, and rollback attempts are limited.
- **Future Directions**: (1) Enhance AND node discrimination to reduce reliance on And2Or; (2) Extend to joint reasoning in multi-table scenarios; (3) Develop structured handling for composite value types.

## Related Work & Insights

**vs. Question Decomposition Methods** (Dater/Chain-of-Table): These methods also decompose questions but at a coarser granularity (sub-questions), making it difficult to identify spurious correlations. EnoTab avoids this via fine-grained evidence units (triplets) and objective consistency evaluation.

**vs. Table Pruning Methods** (H-Star/TabSQLify): These methods prune via SQL/Python programs, where black-box characteristics make errors hard to detect or correct. EnoTab's Evidence Tree makes the pruning logic explicit, observable, and verifiable.

**Insights**: The triangular combination of fine-grained decomposition + objective evaluation + observable execution is highly relevant for other tasks involving structured data and complex reasoning, such as code generation and knowledge graph queries.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The design of evidence as the minimum unit is innovative and addresses transparency pain points.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive analysis across 4 datasets, 3 classes of baselines, full ablation, and cross-model adaptation.
- **Writing Quality**: ⭐⭐⭐⭐ Logical and detailed, though some sections are slightly verbose.
- **Value**: ⭐⭐⭐⭐⭐ Addresses real pain points in practical applications (complex questions + large tables + noise) with no training required and easy deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MulDimIF: A Multi-Dimensional Constraint Framework for Evaluating and Improving Instruction Following in Large Language Models](muldimif_a_multi-dimensional_constraint_framework_for_evaluating_and_improving_i.md)
- [\[ACL 2026\] From Fallback to Frontline: When Can LLMs be Superior Annotators of Human Perspectives?](from_fallback_to_frontline_when_can_llms_be_superior_annotators_of_human_perspec.md)
- [\[ACL 2026\] EVE: A Domain-Specific LLM Framework for Earth Intelligence](eve_a_domain-specific_llm_framework_for_earth_intelligence.md)
- [\[ICLR 2026\] d²Cache: Accelerating Diffusion-Based LLMs via Dual Adaptive Caching](../../ICLR2026/llm_nlp/d2cache_accelerating_diffusion-based_llms_via_dual_adaptive_caching.md)
- [\[ACL 2026\] When Gradients Collide: Failure Modes of Multi-Objective Prompt Optimization for LLM Judges](when_gradients_collide_failure_modes_of_multi-objective_prompt_optimization_for_.md)

</div>

<!-- RELATED:END -->

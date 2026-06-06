---
title: >-
  [Paper Note] ASTRA: Adaptive Semantic Tree Reasoning Architecture for Complex Table Question Answering
description: >-
  [ACL2026][NLP Understanding][Complex Table QA] ASTRA adaptively reconstructs complex tables into semantic trees and employs a dual-mode reasoning approach—combining textual tree navigation and symbolic code execution—to…
tags:
  - "ACL2026"
  - "NLP Understanding"
  - "Complex Table QA"
  - "Semantic Tree"
  - "Table Serialization"
  - "Symbolic Reasoning"
  - "Structured Retrieval"
date: 2026-05-08
content_hash: d0111f6cce0ac6ca
---

# ASTRA: Adaptive Semantic Tree Reasoning Architecture for Complex Table Question Answering

**Conference**: ACL2026  
**arXiv**: [2604.08999](https://arxiv.org/abs/2604.08999)  
**Code**: https://github.com/zjukg/ASTRA  
**Area**: Table Question Answering / LLM Reasoning  
**Keywords**: Complex Table QA, Semantic Tree, Table Serialization, Symbolic Reasoning, Structured Retrieval  

## TL;DR
ASTRA adaptively reconstructs complex tables into semantic trees and employs a dual-mode reasoning approach—combining textual tree navigation and symbolic code execution—to answer questions. It achieves accuracies of 91.6%, 81.9%, and 90.1% on AIT-QA, SSTQA, and HiTab respectively, surpassing strong LLMs and existing table structuring methods.

## Background & Motivation
**Background**: To process table question answering, LLMs typically convert 2D tables into 1D text formats such as Markdown, HTML, triplets, relational tables, or tree structures. For simple flat tables, these serialization methods are sufficient for LLMs to complete various QA tasks.

**Limitations of Prior Work**: Complex tables often feature hierarchical headers, merged cells, irregular sub-tables, and implicit semantic dependencies. The paper identifies four categories of issues in existing methods: structural neglect, the 2D-to-1D representation gap, hallucinations caused by black-box numerical reasoning, and the inability of fixed schemas to adapt to heterogeneous tables.

**Key Challenge**: LLMs favor natural language-like input, yet critical tabular information resides in 2D structures and hierarchical relationships. Direct conversion to text scatters the structure, while conversion to relational tables or triplets may result in sparsity, redundancy, or loss of hierarchical semantics.

**Goal**: ASTRA aims to construct an intermediate representation that preserves explicit hierarchies and semantic contexts while remaining accessible for LLM retrieval and code execution, thereby improving accuracy, interpretability, and efficiency in complex TableQA.

**Key Insight**: The authors select the "Semantic Tree" as a unified representation. Tree nodes and paths preserve header hierarchies, entity-attribute relationships, and cell origins. This structure can be retrieved via natural language or converted into Python dictionaries for programmatic reasoning.

**Core Idea**: First, AdaSTR adaptively reconstructs a semantic tree based on table scale and content density. Then, DuTR performs parallel textual navigation and symbolic execution on the same tree. Finally, an answer selector chooses the more credible response.

## Method

### Overall Architecture
ASTRA decomposes complex TableQA into two phases. The first phase is AdaSTR (Adaptive Semantic Tree Reconstruction), which transforms the original table $T$ into a semantic tree $\tilde{T}$. The second phase is DuTR (Dual-Mode Tree Reasoning), which concurrently runs textual tree navigation and symbolic tree operations on the semantic tree to generate two candidate answers, followed by a final selection via a lightweight LLM.

The focus of this workflow is "representation first": rather than forcing an LLM to perform hard reasoning on a Markdown table, the table is rewritten into a tree with explicit parent-child relationships, semantic paths, and an executable structure.

### Key Designs
1. **AdaSTR: Adaptive Semantic Tree Reconstruction**:
    - **Function**: Converts complex, heterogeneous, and irregular tables into semantic trees while preserving hierarchical and semantic dependencies.
    - **Mechanism**: First, Header Identification & Normalization merges vertical dependencies (e.g., transforming an isolated "Percent" into "Yukon-Percent"). Then, Hierarchy Identification uses an LLM to extract hidden semantic groups (e.g., "Regional Statistics") from normalized headers. Finally, one of three construction strategies (DSP, SRE, or PSS) is selected based on table type.
    - **Design Motivation**: A single fixed serialization strategy cannot handle all complex tables. Medium-sized tables are parsed directly by LLMs, text-dense tables utilize coordinate placeholders with backfilling, and large-scale repetitive tables are handled via programmatically generated loops.

2. **Evaluator-Guided Refinement Loop**:
    - **Function**: Reduces structural hallucinations and information omission during LLM tree construction.
    - **Mechanism**: An Evaluator checks structural integrity and information coverage, verifying if paths align with table coordinates and how many cells are correctly mapped. If the score falls below a threshold, feedback is provided to the LLM for iterative refinement over several rounds.
    - **Design Motivation**: If the semantic tree is constructed incorrectly, all subsequent reasoning will be flawed; thus, quality control at the representation layer is essential.

3. **DuTR: Dual-Mode Tree Reasoning**:
    - **Function**: Combines the strengths of natural language semantic retrieval and code execution.
    - **Mechanism**: The textual mode adaptively selects between Leaf-to-Root or Root-to-Leaf navigation. The former suits aggregation questions by pulling context upward from relevant leaves, while the latter suits lookup questions by traversing downward via global path guidance. The symbolic mode abstracts the semantic tree into a structural skeleton, directing the LLM to generate code for selection, aggregation, and comparison, with a self-correction loop to fix execution errors.
    - **Design Motivation**: Textual reasoning excels at semantic localization, while symbolic execution is superior for numerical computation; complex TableQA often requires both.

### Loss & Training
ASTRA is a training-free method. For fair comparison, all training-free methods including AdaSTR, DuTR, E5, EEDP, GraphOTTER, and ST-Raptor utilize DeepSeek-V3-250324 as the backbone. Evaluation uses GPT-5 as a binary judge to determine if the predicted answer is equivalent to the gold answer and to calculate Accuracy.

## Key Experimental Results

### Main Results
ASTRA achieved strong results across three complex TableQA datasets, notably outperforming both strong models and intermediate representation baselines on SSTQA and HiTab.

| Method | AIT-QA Acc | SSTQA Acc | HiTab Acc | Description |
| :--- | :--- | :--- | :--- | :--- |
| DeepSeek-V3 | 78.5 | 63.2 | 82.0 | Strong open-source LLM with standard text serialization |
| GPT-4o | 80.6 | 66.4 | 78.6 | Strong closed-source model |
| o3 | 89.1 | 78.2 | 85.3 | Strong reasoning model |
| GraphOTTER | 90.4 | 71.5 | 88.8 | Triplet/Graph-based intermediate representation |
| ST-Raptor | 62.7 | 71.1 | 49.0 | Physical tree structure, weaker generalization |
| **ASTRA Adaptive Selection** | **91.6** | **81.9** | **90.1** | **Final method** |
| ASTRA Oracle | 93.5 | 86.1 | 94.1 | Upper bound for ideal selection of text/symbolic answers |

The complementarity of Textual Reasoning and Symbolic Reasoning is evident: on SSTQA, the textual mode (79.8%) outperformed the symbolic mode (75.3%), whereas on HiTab, the symbolic mode (89.3%) outperformed the textual mode (82.2%). This confirms that semantic-intensive and numerical-aggregation problems require different reasoning paths.

### Ablation Study
| Module / Configuration | Metric | Results | Description |
| :--- | :--- | :--- | :--- |
| AdaSTR Full | Avg Coverage / Min Coverage | 0.929 / 0.738 | Best tree construction coverage |
| w/o Evaluator-Guided | Avg Coverage / Min Coverage | 0.745 / 0.473 | Significant info loss without feedback loop |
| w/o Synthesis Strategies | Avg Coverage / Min Coverage | 0.795 / 0.153 | Static DSP struggles with large complex tables |
| Adaptive Tree Navigation | SSTQA Acc | 79.84 | Full textual navigation |
| w/o Embedding Model | SSTQA Acc | 71.34 | Lost 8.50 points without semantic path guidance |
| Force Root-to-Leaf | SSTQA Acc | 77.23 | Fixed strategy inferior to dynamic switching |
| Force Leaf-to-Root | SSTQA Acc | 75.65 | Fixed strategy inferior to dynamic switching |
| Symbolic Tree Manipulation | SSTQA Acc | 75.26 | Full symbolic reasoning |
| w/o Code Examples | SSTQA Acc | 70.42 | Examples are critical for program generation |
| Textual Serialization | SSTQA Acc | 63.20 | Original text serialization is weaker |
| Semantic Tree Direct Prompting | SSTQA Acc | 70.55 | 7.35 point gain just by changing representation |

### Key Findings
- The semantic tree representation itself is highly valuable: Direct Prompting alone improved accuracy from 63.20 to 70.55 without complex reasoning.
- Textual and Symbolic modes are complementary rather than redundant. Semantic queries favor textual path retrieval, while numerical aggregations favor code execution.
- ASTRA's online QA latency is significantly lower than ST-Raptor and GraphOTTER. For example, on AIT-QA, ASTRA tree/QA takes 29.18/7.80s, compared to 55.73/31.18s for ST-Raptor. ASTRA's write-once/read-many architecture amortizes construction costs as query counts increase.
- The Evaluator-Guided Loop is triggered in only ~7% of cases, suggesting most tables can be structured in a single round while retaining correction capabilities for difficult samples.

## Highlights & Insights
- The paper addresses the root problem of complex TableQA: the issue is not LLM intelligence, but rather that input representations destroy table structure. The semantic tree acts as an intermediate language consumable by both LLMs and programs.
- The DSP/SRE/PSS construction modes are practical, avoiding the high costs and hallucination risks of forcing LLMs to output entire trees for all table types.
- The most insightful aspect is the Textual-Symbolic dual-mode: a single structured representation serving both semantic retrieval and verifiable computation is more robust than simple prompt engineering.

## Limitations & Future Work
- For very simple flat tables, semantic tree reconstruction may introduce unnecessary overhead compared to direct text serialization. ASTRA's advantage lies primarily in complex hierarchical tables.
- The method relies on text and structural parsing, neglecting visual cues like background colors, bold text, borders, and layout emphasis, which often carry implicit semantics in real-world reports.
- Evaluation relies on GPT-5 as a judge. While better at handling semantic equivalence, it may introduce bias; future work could include human sampling or task-specific exact matching.
- The Answer Selector is a lightweight LLM; it may still be misled when both textual and symbolic answers are incorrect or only partially correct.

## Related Work & Insights
- **vs Markdown/HTML Serialization**: Traditional text serialization is simple but loses hierarchies and cell dependencies; ASTRA explicitly preserves these via tree paths.
- **vs GraphOTTER**: While GraphOTTER's triplet approach mitigates fixed schema issues, it scatters cell relationships; ASTRA's tree preserves parent-child structures and semantic context.
- **vs ST-Raptor**: ST-Raptor demonstrates tree utility but focuses on physical layout and rule-based construction; ASTRA uses LLMs to mine semantic hierarchies for irregular tables.
- **Insight**: Table representations for LLMs should not just aim to be "compressed into text" but should strive to be "both human-readable and program-executable."

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of semantic trees and dual-mode reasoning is comprehensive, with the adaptive construction strategy offering significant engineering value.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensive evaluation across three complex datasets, including construction and reasoning ablations and efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-defined problems and challenges; some appendix details remain important for replication.
- Value: ⭐⭐⭐⭐⭐ Provides direct insights for complex TableQA, document intelligence, and LLM tool-based reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Table Question Answering in the Era of Large Language Models: A Comprehensive Survey](table_question_answering_in_the_era_of_large_language_models_a_comprehensive_sur.md)
- [\[ACL 2026\] AdapTime: Enabling Adaptive Temporal Reasoning in Large Language Models](adaptime_enabling_adaptive_temporal_reasoning_in_large_language_models.md)
- [\[ACL 2026\] It's High Time: A Survey of Temporal Question Answering](it39s_high_time_a_survey_of_temporal_question_answering.md)
- [\[ACL 2026\] Beyond Chunking: Discourse-Aware Hierarchical Retrieval for Long Document Question Answering](beyond_chunking_discourse-aware_hierarchical_retrieval_for_long_document_questio.md)
- [\[ACL 2026\] SAM-NER: Semantic Archetype Mediation for Zero-Shot Named Entity Recognition](sam-ner_semantic_archetype_mediation_for_zero-shot_named_entity_recognition.md)

</div>

<!-- RELATED:END -->

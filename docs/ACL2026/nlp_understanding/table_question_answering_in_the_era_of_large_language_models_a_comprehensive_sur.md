---
title: >-
  [Paper Note] Table Question Answering in the Era of Large Language Models: A Comprehensive Survey
description: >-
  [ACL 2026][NLP Understanding][Table Question Answering] This paper provides a comprehensive survey of Table Question Answering (TQA) research in the LLM era. It systematically categorizes task settings across five dimens…
tags:
  - "ACL 2026"
  - "NLP Understanding"
  - "Table Question Answering"
  - "LLM"
  - "Survey"
  - "Table Understanding"
  - "Complex Reasoning"
date: 2026-05-08
content_hash: 5b7f5e24b612672f
---

# Table Question Answering in the Era of Large Language Models: A Comprehensive Survey

**Conference**: ACL 2026  
**arXiv**: [2510.09671](https://arxiv.org/abs/2510.09671)  
**Code**: None  
**Area**: Reinforcement Learning  
**Keywords**: Table Question Answering, LLM, Survey, Table Understanding, Complex Reasoning

## TL;DR
This paper provides a comprehensive survey of Table Question Answering (TQA) research in the LLM era. It systematically categorizes task settings across five dimensions (table format, question complexity, answer format, modality, and domain) and organizes modeling methods based on core challenges (table understanding, complex queries, large inputs, data heterogeneity, and knowledge integration). Covering 277 papers, it also forward-lookingly discusses emerging directions such as reinforcement learning and interpretability.

## Background & Motivation

**Background**: Table Question Answering (TQA) aims to answer natural language questions based on tabular data and is one of the most widely researched tabular tasks in the LLM era. Task settings are diverse, covering text/image tables, single/multiple tables, and simple retrieval/complex reasoning variants.

**Limitations of Prior Work**: Existing surveys either focus solely on text tables while ignoring image tables, focus on table representation without discussing modeling methods, or emphasize Agent settings while neglecting fine-tuning methods. No survey provides a systematic TQA task classification framework nor covers emerging directions in the LLM era such as reinforcement learning and new evaluation paradigms.

**Key Challenge**: TQA research is growing rapidly but lacks a systematic organizational framework—different papers focus on different sub-problems and use different evaluation methods, making it difficult for researchers to fully comprehend the current state and open problems of the field.

**Goal**: To provide a comprehensive TQA survey that systematizes existing research across four levels: task classification, modeling methods, evaluation methods, and emerging directions.

**Key Insight**: Organize methods through a "challenge-driven" approach—grouping modeling methods by the core challenges they attempt to solve rather than by technical routes.

**Core Idea**: Construct a two-dimensional classification framework—the task dimension describes "what to do" from five perspectives, and the method dimension describes "how to do it" from five challenges, forming a research landscape.

## Method

### Overall Architecture
The survey is organized into four dimensions: (1) Task settings and resources—categorizing existing benchmarks by table format/structure/quantity, question complexity, answer format, modality, and domain; (2) Modeling methods—grouped by five major challenges (table understanding, complex queries, large input processing, data heterogeneity, and knowledge integration); (3) Evaluation methods—covering traditional metrics and new LLM-as-judge approaches; (4) Emerging directions—reinforcement learning, interpretability, etc.

### Key Designs

1.  **Five-dimensional Task Taxonomy**:
    - **Function**: Systematically describes the diversity of the TQA task space.
    - **Mechanism**: (a) Table dimension: text vs. image, flat vs. hierarchical, single vs. multi-table; (b) Question dimension: retrieval-based vs. reasoning-based; (c) Answer dimension: short span vs. free-form; (d) Modality dimension: pure table vs. table + text/image/knowledge graph; (e) Domain dimension: open-domain vs. closed-domain.
    - **Design Motivation**: Existing surveys usually classify by only one or two dimensions, failing to capture the full diversity of TQA tasks.

2.  **Challenge-driven Method Classification**:
    - **Function**: Organizes literature by the core problems the methods attempt to solve.
    - **Mechanism**: Five major challenges—(a) Table understanding (visual table modeling + text table representation); (b) Complex query processing (fine-tuning methods: RL, distillation, curriculum learning vs. non-fine-tuning: Agent workflows, ReAct prompting, code generation); (c) Large input processing (table compression, block retrieval); (d) Data heterogeneity (cross-modal fusion); (e) Knowledge integration (external knowledge enhancement).
    - **Design Motivation**: Researchers typically look for methods when facing specific challenges; grouping by challenge is more practical than grouping by technology.

3.  **Forward-looking Analysis of Emerging Directions**:
    - **Function**: Identifies research opportunities that have not yet been fully explored.
    - **Mechanism**: (a) Application of reinforcement learning in TQA—including process reward models, GRPO, etc.; (b) Interpretability—verifiability of Chain-of-Thought reasoning; (c) New evaluation paradigms—LLM-as-judge, uncertainty-aware evaluation.
    - **Design Motivation**: Assists researchers in quickly locating high-value, low-competition research directions.

### Loss & Training
As a survey paper, it does not involve specific training. The text summarizes commonly used strategies in fine-tuning methods: SFT on synthetic data, RL optimization using GRPO/PPO, and process reward model training.

## Key Experimental Results

### Main Results
The paper does not include original experiments but systematically organizes comparisons of various methods on major benchmarks. Taking WikiTableQuestions as an example:

| Method Type | Representative Method | Accuracy Range |
| :--- | :--- | :--- |
| Fine-tuning + Code Gen | TableLlama, CABINET | 55-68% |
| Non-fine-tuning Agent | ReAct + Code Tools | 60-72% |
| RL Enhanced | GRPO + Process Reward | 70-78% |
| Commercial API | GPT-4 + Few-shot | 65-74% |

### Ablation Study
A comparison of table representation formats summarized in the paper:

| Representation Format | Advantage | Disadvantage |
| :--- | :--- | :--- |
| Markdown | Compact, LLM-friendly | Truncation of large tables |
| JSON | Clear structure | High token consumption |
| SQL Database | Precise queries | Requires predefined schema |
| Spreadsheet | Flexible, supports formulas | Complex formatting |

### Key Findings
- No single table representation format is optimal across all datasets and models—Markdown and JSON each have advantages in different scenarios.
- Visual table understanding remains significantly more difficult than text table understanding; even OCR pipelines are inferior to models fine-tuned directly on table images.
- RL methods (such as GRPO + process reward models) are the most promising recent directions, significantly outperforming SFT on complex reasoning tasks.
- Agent workflows combined with code execution are the most effective strategy in non-fine-tuning scenarios, but they depend on powerful base models.

## Highlights & Insights
- The challenge-driven classification framework is the greatest contribution of this survey—allowing researchers to quickly locate "how others solved the problem I am facing." This organization is more practical than grouping by technical routes (such as "prompt-based methods" vs. "fine-tuning methods").
- It provides the first systematic coverage of RL applications in TQA—including GRPO, process reward models, and Best-of-N selection, offering a comprehensive starting point for researchers in this direction.
- The reflection on evaluation methods is valuable—it points out that traditional exact match metrics cannot evaluate free-form answers, and LLM-as-judge is a necessary but cautious alternative.

## Limitations & Future Work
- Restricted by the survey deadline, it may miss the latest work (especially the rapid developments in 2026).
- Lacks systematic quantitative comparison experiments—direct comparison is difficult as different works use different settings.
- Coverage of multilingual TQA and domain-specific (e.g., finance, medical) TQA is limited.
- Future directions: A unified framework for multi-table reasoning, and end-to-end RL optimization for table + code + reasoning.

## Related Work & Insights
- **vs. Dong et al. (2022)**: Earlier surveys only covered methods from the pre-LLM era; this paper comprehensively covers new paradigms in the LLM era such as fine-tuning, Agents, and RL.
- **vs. Wu et al. (2025)**: That survey focuses on table representation; this paper covers both representation and modeling methods and introduces challenge-driven classification.

## Rating
- **Novelty**: ⭐⭐⭐ The survey itself contains no original methods, but the classification framework is innovative.
- **Experimental Thoroughness**: ⭐⭐⭐ No original experiments, but literature coverage is comprehensive (277 papers).
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear structure, intuitive classification charts, and high readability.
- **Value**: ⭐⭐⭐⭐ Provides a one-stop reference for TQA researchers, with particularly high value in the forward-looking analysis of emerging directions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] It's High Time: A Survey of Temporal Question Answering](it39s_high_time_a_survey_of_temporal_question_answering.md)
- [\[ACL 2026\] The Imperfective Paradox in Large Language Models](the_imperfective_paradox_in_large_language_models.md)
- [\[ACL 2026\] AdapTime: Enabling Adaptive Temporal Reasoning in Large Language Models](adaptime_enabling_adaptive_temporal_reasoning_in_large_language_models.md)
- [\[ACL 2026\] ASTRA: Adaptive Semantic Tree Reasoning Architecture for Complex Table Question Answering](astra_adaptive_semantic_tree_reasoning_architecture_for_complex_table_question_a.md)
- [\[ACL 2026\] Beyond Chunking: Discourse-Aware Hierarchical Retrieval for Long Document Question Answering](beyond_chunking_discourse-aware_hierarchical_retrieval_for_long_document_questio.md)

</div>

<!-- RELATED:END -->

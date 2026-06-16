---
title: >-
  [Paper Note] Table Question Answering in the Era of Large Language Models: A Comprehensive Survey
description: >-
  [ACL 2026][NLP Understanding][LLM] This paper provides a comprehensive survey of Table Question Answering (TQA) research in the LLM era. It systematically categorizes task settings across five dimensions (table format, question complexity, answer format, modality, and domain) and organizes modeling approaches based on core challenges (table understandin
tags:
  - ACL 2026
  - NLP Understanding
  - LLM
date: 2026-05-08
content_hash: 847db68c18605a5a
---
# Table Question Answering in the Era of Large Language Models: A Comprehensive Survey

**Conference**: ACL 2026  
**arXiv**: [2510.09671](https://arxiv.org/abs/2510.09671)  
**Code**: None  
**Area**: Reinforcement Learning  
**Keywords**: Table Question Answering, LLM, Survey, Table Understanding, Complex Reasoning

## TL;DR
This paper provides a comprehensive survey of Table Question Answering (TQA) research in the LLM era. It systematically categorizes task settings across five dimensions (table format, question complexity, answer format, modality, and domain) and organizes modeling approaches based on core challenges (table understanding, complex queries, large inputs, data heterogeneity, and knowledge integration). Covering 277 papers, it also provides forward-looking discussions on emerging directions such as reinforcement learning and interpretability.

## Background & Motivation

**Background**: Table Question Answering (TQA) aims to answer natural language questions based on tabular data and is one of the most widely studied tabular tasks in the LLM era. Task settings are diverse, covering variants such as text/image tables, single/multiple tables, and simple retrieval/complex reasoning.

**Limitations of Prior Work**: Existing surveys either focus solely on text tables while ignoring image tables, focus on table representation without discussing modeling methods, or emphasize Agent settings while neglecting fine-tuning approaches. No survey provides a systematic TQA task classification framework or covers emerging LLM-era directions like reinforcement learning and new evaluation paradigms.

**Key Challenge**: TQA research is growing rapidly but lacks a systematic organizational framework—different papers focus on different sub-problems and use different evaluation methods, making it difficult for researchers to fully comprehend the current state and open problems of the field.

**Goal**: To provide a comprehensive TQA survey that systematizes existing research from four levels: task classification, modeling methods, evaluation methods, and emerging directions.

**Key Insight**: Organize methodological classifications using a "challenge-driven" approach—grouping modeling methods by the core challenges they attempt to solve rather than by technical routes.

**Core Idea**: Construct a two-dimensional classification framework—the task dimension describes "what to do" from five perspectives, and the method dimension describes "how to do it" across five challenges, forming a complete research landscape.

## Method

### Overall Architecture
The survey is organized across four dimensions: (1) Task settings and resources—classifying existing benchmarks by table format/structure/quantity, question complexity, answer format, modality, and domain; (2) Modeling methods—grouped by five major challenges (table understanding, complex queries, large input processing, data heterogeneity, and knowledge integration); (3) Evaluation methods—covering traditional metrics and new LLM-as-judge paradigms; (4) Emerging directions—reinforcement learning, interpretability, etc.

### Key Designs

**1. Five-dimensional Task Taxonomy: Characterizing "What TQA is doing" via five orthogonal perspectives**

Existing surveys often only approach the task from one or two dimensions, failing to capture the full diversity of TQA tasks. This paper establishes a five-dimensional coordinate system for the task space: (a) Table dimension—text tables vs. image tables, flat vs. hierarchical, single table vs. multi-table; (b) Question dimension—retrieval-based vs. reasoning-based; (c) Answer dimension—short span vs. free-form; (d) Modality dimension—pure tables vs. tables combined with text/images/knowledge graphs; (e) Domain dimension—open-domain vs. closed-domain. These five perspectives are mutually orthogonal, allowing any benchmark to be positioned within this coordinate system to facilitate horizontal comparison.

**2. Challenge-driven Method Classification: Organizing literature by "which core difficulty is being solved" rather than technical route**

Researchers usually search for methods with a specific pain point in mind, so grouping by technical routes ("prompt-based" vs. "fine-tuning") is less effective. This paper categorizes modeling work by five major challenges: (a) Table understanding—visual table modeling + text table representation; (b) Complex query processing—split into fine-tuning routes (RL, distillation, curriculum learning) and non-fine-tuning routes (Agent workflows, ReAct prompting, code generation); (c) Large input processing—table compression, chunked retrieval; (d) Data heterogeneity—cross-modal fusion; (e) Knowledge integration—external knowledge enhancement. This organization allows readers with specific problems, such as "my table is too large for the context," to directly find relevant solutions.

**3. Prospective Analysis of Emerging Directions: Identifying high-value yet uncrowded research gaps**

Another value of this survey is helping researchers find promising directions. This paper specifically identifies three under-explored areas: (a) Applications of Reinforcement Learning in TQA, including Process Reward Models, GRPO, etc.; (b) Interpretability, focusing on the verifiability of Chain-of-Thought reasoning; (c) New evaluation paradigms, such as LLM-as-judge and uncertainty-aware evaluation. This section is the first to systematically cover RL applications in TQA, providing a roadmap for new researchers.

### Loss & Training
As a survey paper, it does not involve specific training. It summarizes commonly used strategies in fine-tuning methods: SFT on synthetic data, RL using GRPO/PPO optimization, and the training of Process Reward Models.

## Key Experimental Results

### Main Results
While the paper does not contain original experiments, it systematically organizes comparisons of methods across major benchmarks. For WikiTableQuestions:

| Method Type | Representative Methods | Accuracy Range |
|-------------|------------------------|----------------|
| Fine-tuning + Code Gen | TableLlama, CABINET | 55-68% |
| Non-fine-tuning Agent | ReAct + Code Tools | 60-72% |
| RL Enhanced | GRPO + Process Reward | 70-78% |
| Commercial API | GPT-4 + Few-shot | 65-74% |

### Ablation Study
The paper summarizes comparisons of table representation formats:

| Representation Format | Advantages | Disadvantages |
|-----------------------|------------|---------------|
| Markdown | Compact, LLM-friendly | Truncation for large tables |
| JSON | Clear structure | High token consumption |
| SQL Database | Precise querying | Requires predefined schema |
| Spreadsheet | Flexible, supports formulas | Complex formatting |

### Key Findings
- No single table representation format is optimal across all datasets and models—Markdown and JSON have advantages in different scenarios.
- Visual table understanding remains significantly more difficult than text table understanding; even OCR pipelines are inferior to models directly fine-tuned on table images.
- RL methods (e.g., GRPO + Process Reward Models) are currently the most promising direction, significantly outperforming SFT on complex reasoning tasks.
- Agent workflows + code execution represent the most effective strategy for non-fine-tuning scenarios, but they rely on powerful base models.

## Highlights & Insights
- The challenge-driven classification framework is the primary contribution—allowing researchers to quickly locate how others solved specific problems. This is more practical than technical route grouping.
- First systematic coverage of RL in TQA—including GRPO, Process Reward Models, Best-of-N selection, etc., providing a comprehensive starting point for researchers in this direction.
- Reflection on evaluation methods is valuable—noting that traditional exact match metrics cannot evaluate free-form answers and that LLM-as-judge is a necessary but cautious alternative.

## Limitations & Future Work
- Due to the survey cutoff time, it may miss very recent work (especially rapid developments in 2026).
- Lack of systematic quantitative comparative experiments—direct comparison is difficult as different works use different settings.
- Limited coverage of multilingual TQA and domain-specific TQA (e.g., finance, healthcare).
- Future directions: Unified frameworks for multi-table reasoning, and end-to-end RL optimization combining tables, code, and reasoning.

## Related Work & Insights
- **vs. Dong et al. (2022)**: Early surveys only covered pre-LLM methods; this paper covers new paradigms like fine-tuning, Agents, and RL in the LLM era.
- **vs. Wu et al. (2025)**: That survey focuses on table representation, while this paper covers both representation and modeling methods and introduces challenge-driven classification.

## Rating
- Novelty: ⭐⭐⭐ (The survey itself contains no original method, but the framework is innovative)
- Experimental Thoroughness: ⭐⭐⭐ (No original experiments, but comprehensive literature coverage with 277 papers)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear structure, intuitive classification charts, high readability)
- Value: ⭐⭐⭐⭐ (Provides a one-stop reference for TQA researchers, especially the forward-looking analysis of emerging directions)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] It's High Time: A Survey of Temporal Question Answering](it39s_high_time_a_survey_of_temporal_question_answering.md)
- [\[ACL 2026\] The Imperfective Paradox in Large Language Models](the_imperfective_paradox_in_large_language_models.md)
- [\[ACL 2025\] BQA: Body Language Question Answering Dataset for Video Large Language Models](../../ACL2025/nlp_understanding/bqa_body_language_question_answering_dataset_for_video_large_language_models.md)
- [\[ACL 2026\] AdapTime: Enabling Adaptive Temporal Reasoning in Large Language Models](adaptime_enabling_adaptive_temporal_reasoning_in_large_language_models.md)
- [\[ACL 2026\] ASTRA: Adaptive Semantic Tree Reasoning Architecture for Complex Table Question Answering](astra_adaptive_semantic_tree_reasoning_architecture_for_complex_table_question_a.md)

</div>

<!-- RELATED:END -->

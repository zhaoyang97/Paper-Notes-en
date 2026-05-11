---
title: >-
  [Paper Note] Table Question Answering in the Era of Large Language Models: A Comprehensive Survey
description: >-
  [ACL 2026][Reinforcement Learning][Table Question Answering] This paper presents a comprehensive survey of Table Question Answering (TQA) research in the era of large language models. It systematically categorizes task s…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "Table Question Answering"
  - "LLM"
  - "Survey"
  - "Table Understanding"
  - "Complex Reasoning"
date: 2026-05-08
content_hash: 3498b5534709f92b
---

# Table Question Answering in the Era of Large Language Models: A Comprehensive Survey

**Conference**: ACL 2026
**arXiv**: [2510.09671](https://arxiv.org/abs/2510.09671)
**Code**: None
**Area**: Reinforcement Learning
**Keywords**: Table Question Answering, LLM, Survey, Table Understanding, Complex Reasoning

## TL;DR
This paper presents a comprehensive survey of Table Question Answering (TQA) research in the era of large language models. It systematically categorizes task settings along five dimensions (table format, question complexity, answer format, modality, and domain), organizes modeling approaches around five core challenges (table understanding, complex queries, large input handling, data heterogeneity, and knowledge integration), covers 277 papers, and provides forward-looking discussions on emerging directions such as reinforcement learning and interpretability.

## Background & Motivation

**Background**: Table Question Answering (TQA) aims to answer natural language questions based on tabular data, and is one of the most widely studied table-related tasks in the LLM era. Task settings are highly diverse, encompassing textual and image-based tables, single and multi-table scenarios, and a spectrum from simple retrieval to complex reasoning.

**Limitations of Prior Work**: Existing surveys either focus solely on textual tables while neglecting image-based tables, address table representations without discussing modeling methods, or emphasize agentic settings while ignoring fine-tuning approaches. No prior survey provides a systematic TQA task taxonomy or covers emerging directions in the LLM era such as reinforcement learning and novel evaluation paradigms.

**Key Challenge**: TQA research is growing rapidly but lacks a systematic organizational framework. Different papers address different sub-problems and adopt different evaluation protocols, making it difficult for researchers to obtain a comprehensive understanding of the field's current state and open problems.

**Goal**: To provide a comprehensive TQA survey that systematically organizes existing research along four dimensions: task taxonomy, modeling methods, evaluation protocols, and emerging directions.

**Key Insight**: Organizing the method taxonomy in a *challenge-driven* manner—grouping modeling approaches by the core challenge they address rather than by technical paradigm.

**Core Idea**: Constructing a two-dimensional classification framework in which the task dimension characterizes *what to do* from five perspectives, and the method dimension characterizes *how to do it* from five challenge angles, jointly forming a panoramic view of the research landscape.

## Method

### Overall Architecture
The survey is organized along four dimensions: (1) **Task Settings and Resources** — existing benchmarks are categorized from five perspectives: table format/structure/quantity, question complexity, answer format, modality, and domain; (2) **Modeling Methods** — approaches are grouped by five major challenges: table understanding, complex queries, large input handling, data heterogeneity, and knowledge integration; (3) **Evaluation Methods** — covering both traditional metrics and emerging LLM-as-judge approaches; (4) **Emerging Directions** — including reinforcement learning and interpretability.

### Key Designs

1. **Five-Dimensional Task Taxonomy**:

    - **Function**: Systematically characterizes the diversity of the TQA task space.
    - **Mechanism**: (a) Table dimension: textual vs. image-based, flat vs. hierarchical, single vs. multi-table; (b) Question dimension: retrieval-type vs. reasoning-type; (c) Answer dimension: short-span vs. free-form; (d) Modality dimension: table-only vs. table combined with text/image/knowledge graph; (e) Domain dimension: open-domain vs. closed-domain.
    - **Design Motivation**: Existing surveys typically classify along only one or two dimensions, failing to capture the full diversity of TQA tasks.

2. **Challenge-Driven Method Taxonomy**:

    - **Function**: Organizes the literature according to the core problem each method addresses.
    - **Mechanism**: Five major challenges — (a) Table understanding (visual table modeling and textual table representation); (b) Complex query processing (fine-tuning approaches: RL, distillation, curriculum learning; training-free approaches: agent workflows, ReAct prompting, code generation); (c) Large input handling (table compression, chunking-based retrieval); (d) Data heterogeneity (cross-modal fusion); (e) Knowledge integration (external knowledge augmentation).
    - **Design Motivation**: Researchers typically seek methods in response to specific challenges; grouping by challenge is more practically useful than grouping by technical paradigm.

3. **Prospective Analysis of Emerging Directions**:

    - **Function**: Identifies research opportunities that remain underexplored.
    - **Mechanism**: (a) Reinforcement learning for TQA — including process reward models and GRPO; (b) Interpretability — verifiability of chain-of-thought reasoning; (c) Novel evaluation paradigms — LLM-as-judge and uncertainty-aware evaluation.
    - **Design Motivation**: Helps researchers quickly identify high-value yet low-competition research directions.

### Loss & Training
As a survey paper, no original training is involved. The paper summarizes commonly used strategies in fine-tuning approaches, including SFT on synthetic data, RL optimization via GRPO/PPO, and process reward model training.

## Key Experimental Results

### Main Results
No original experiments are conducted; however, the paper systematically compiles comparisons of major methods on key benchmarks. Using WikiTableQuestions as an example:

| Method Type | Representative Methods | Accuracy Range |
|---|---|---|
| Fine-tuning + Code Generation | TableLlama, CABINET | 55–68% |
| Training-free Agent | ReAct + Code Tools | 60–72% |
| RL-enhanced | GRPO + Process Reward | 70–78% |
| Commercial API | GPT-4 + Few-shot | 65–74% |

### Ablation Study
The paper summarizes a comparison of table representation formats:

| Representation Format | Advantages | Disadvantages |
|---|---|---|
| Markdown | Compact, LLM-friendly | Truncation for large tables |
| JSON | Clear structure | High token consumption |
| SQL Database | Precise querying | Requires predefined schema |
| Spreadsheet | Flexible, formula-capable | Complex formatting |

### Key Findings
- No single table representation format is optimal across all datasets and models — Markdown and JSON each hold advantages in different scenarios.
- Visual table understanding remains significantly more challenging than textual table understanding; even OCR pipelines underperform models fine-tuned directly on table images.
- RL methods (e.g., GRPO combined with process reward models) represent the most promising recent direction, substantially outperforming SFT on complex reasoning tasks.
- Agent workflows combined with code execution constitute the most effective strategy in training-free settings, but rely on strong backbone models.

## Highlights & Insights
- The challenge-driven taxonomy is the survey's most significant contribution — enabling researchers to quickly identify how others have addressed a given problem. This organizational approach is more practically useful than grouping by technical paradigm (e.g., "prompt-based" vs. "fine-tuning-based" methods).
- This is the first survey to systematically cover RL applications in TQA — including GRPO, process reward models, and Best-of-N selection — providing a comprehensive starting point for researchers in this direction.
- The critical reflection on evaluation methods is valuable: the paper highlights that traditional exact-match metrics are inadequate for evaluating free-form answers, and that LLM-as-judge is a necessary yet cautiously applicable alternative.

## Limitations & Future Work
- The survey's coverage cutoff may cause some recent work to be missed, particularly given rapid developments in 2026.
- A systematic quantitative comparison experiment is absent — direct comparison is difficult due to heterogeneous experimental setups across papers.
- Coverage of multilingual TQA and domain-specific TQA (e.g., finance, healthcare) is limited.
- Future directions include: unified frameworks for multi-table reasoning, and end-to-end RL optimization integrating table understanding, code generation, and reasoning.

## Related Work & Insights
- **vs. Dong et al. (2022)**: An earlier survey covering only pre-LLM-era methods; the present work comprehensively covers LLM-era paradigms including fine-tuning, agentic approaches, and RL.
- **vs. Wu et al. (2025)**: That survey focuses on table representations; the present work covers both representations and modeling methods, and introduces a challenge-driven taxonomy.

## Rating
- **Novelty**: ⭐⭐⭐ — No original methods are proposed, but the taxonomic framework is innovative.
- **Experimental Thoroughness**: ⭐⭐⭐ — No original experiments, but literature coverage is comprehensive (277 papers).
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear structure, intuitive classification figures, and strong readability.
- **Value**: ⭐⭐⭐⭐ — Serves as a one-stop reference for TQA researchers; the prospective analysis of emerging directions is particularly valuable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] A Survey of Reinforcement Learning for Large Language Models under Data Scarcity: Challenges and Solutions](a_survey_of_reinforcement_learning_for_large_language_models_under_data_scarcity.md)
- [\[ICLR 2026\] GraphOmni: A Comprehensive and Extensible Benchmark Framework for Large Language Models on Graph-theoretic Tasks](../../ICLR2026/reinforcement_learning/graphomni_a_comprehensive_and_extensible_benchmark_framework_for_large_language_.md)
- [\[CVPR 2026\] ReAG: Reasoning-Augmented Generation for Knowledge-based Visual Question Answering](../../CVPR2026/reinforcement_learning/reag_reasoning-augmented_generation_for_knowledge-based_visual_question_answerin.md)
- [\[ACL 2026\] Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments](feedback-driven_tool-use_improvements_in_large_language_models_via_automated_bui.md)
- [\[ACL 2026\] From Passive Metric to Active Signal: The Evolving Role of Uncertainty Quantification in Large Language Models](from_passive_metric_to_active_signal_the_evolving_role_of_uncertainty_quantifica.md)

</div>

<!-- RELATED:END -->

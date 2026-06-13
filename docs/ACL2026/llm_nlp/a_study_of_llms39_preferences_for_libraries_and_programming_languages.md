---
title: >-
  [Paper Note] A Study of LLMs' Preferences for Libraries and Programming Languages
description: >-
  [ACL 2026 (Findings)][LLM/NLP][Code generation preference] This is the first systematic study of preference behaviors toward libraries and programming languages in code generation across 8 LLMs. It finds that LLMs heavil…
tags:
  - "ACL 2026 (Findings)"
  - "LLM/NLP"
  - "Code generation preference"
  - "Library selection bias"
  - "Programming language preference"
  - "LLM behavior analysis"
  - "Technical diversity"
date: 2026-05-08
content_hash: e7bf7ed035b88845
---

# A Study of LLMs' Preferences for Libraries and Programming Languages

**Conference**: ACL 2026 (Findings)  
**arXiv**: [2503.17181](https://arxiv.org/abs/2503.17181)  
**Code**: [GitHub](https://github.com/itsluketwist/llm-code-bias)  
**Area**: LLM/NLP  
**Keywords**: Code generation preference, Library selection bias, Programming language preference, LLM behavior analysis, Technical diversity

## TL;DR

This is the first systematic study of preference behaviors toward libraries and programming languages in code generation across 8 LLMs. It finds that LLMs heavily favor popular libraries like NumPy (45% of uses are unnecessary) and the Python language (58% of high-performance tasks still select Python), with natural language recommendations inconsistent with actual code choices.

## Background & Motivation

**Background**: LLMs have made significant progress in code generation, but existing evaluations primarily focus on functional correctness and syntactic validity, ignoring the critical design decisions LLMs make—which libraries to select and which programming languages to use.

**Limitations of Prior Work**: Developers often do not specify specific libraries when prompting LLMs, and many end-users lack the expertise to judge whether an LLM's language choice is appropriate. This implies that the technical preferences of LLMs may profoundly impact the diversity of the software ecosystem.

**Key Challenge**: LLMs should select the most suitable technology stack based on task requirements, but the frequency distribution in training data may cause them to systematically bias toward popular technologies, even when they are not optimal.

**Goal**: Quantify preference patterns in library and programming language selection and evaluate the rationality and potential risks of these preferences.

**Key Insight**: Design three sets of experiments—library selection for benchmark tasks, library/language selection for project initialization, and consistency testing between natural language recommendations and actual code behavior.

**Core Idea**: LLMs exhibit a significant "familiarity preference" in code generation, prioritizing popular technologies over those best suited for the task.

## Method

### Overall Architecture

The study consists of three sets of experiments covering two dimensions (library and language) × two scenarios (benchmark tasks and project initialization), plus a consistency check. Eight diverse LLMs (GPT-4o-mini, GPT-3.5-turbo, Claude-3.5 Sonnet/Haiku, Llama-3.2-3B, Mistral-7B, Qwen-2.5-Coder, DeepSeek-LLM) were used, with 3-100 responses generated per task to reduce randomness.

### Key Designs

1.  **Library Preference Experiment (Experiment 1)**:
    *   **Function**: Quantify Python library selection preferences when no library is specified.
    *   **Mechanism**: Use 525 tasks from BigCodeBench (filtering out tasks mentioning ground-truth libraries in prompts), require LLMs to generate code using external libraries, and compare library usage frequency with ground-truth.
    *   **Design Motivation**: Developers often ask LLMs to write code without specifying libraries; understanding behavior patterns in this scenario is essential.

2.  **Language Preference Experiment (Experiment 2)**:
    *   **Function**: Quantify programming language selection preferences when no language is specified.
    *   **Mechanism**: Test benchmark tasks using six language-agnostic datasets (Multi-HumanEval, MBXP, AixBench, CoNaLa, APPS, CodeContests). Additionally, design 5 project initialization tasks for high-performance scenarios (concurrent web servers, cross-platform GUI, low-latency trading platforms, etc.) where Python is considered suboptimal.
    *   **Design Motivation**: Test whether LLMs select appropriate languages based on task characteristics (e.g., high-performance needs) or default to Python.

3.  **Recommendation Consistency Experiment (Experiment 3)**:
    *   **Function**: Verify whether the technologies LLMs recommend in natural language match those they actually use in code generation.
    *   **Mechanism**: Have LLMs rank the "best" libraries/languages in text, then compare these with actual usage frequency rankings from Experiments 1/2 using Kendall's $\tau_b$ coefficient.
    *   **Design Motivation**: If an LLM "knows" the optimal choice (correct NL recommendation) but fails to use it in code (different actual preference), it suggests the preference is embedded in generation behavior rather than a lack of knowledge.

### Loss & Training

This is an empirical study and does not involve model training. All LLMs used default API parameters, new sessions were used for each interaction to avoid cache bias, and no system prompts were used to reflect baseline behavior.

## Key Experimental Results

### Main Results

| Finding | Specific Data | Impact |
| :--- | :--- | :--- |
| NumPy Overuse | NumPy used in 192 out of 305 tasks (63%) where it was unnecessary | Severe bias |
| Lack of Diversity | Each LLM used only 32-39 different libraries | Ecosystem homogenization |
| Python Preference | Python still chosen in 58% of high-performance tasks | Technical mismatch |
| Rust Absence | Rust usage rate 0% in high-performance projects | Extreme preference |
| NL-Code Inconsistency | Kendall's $\tau_b$ is extremely low | Inconsistency between knowledge and action |

### Ablation Study

| Configuration | Key Metrics | Description |
| :--- | :--- | :--- |
| Prompt Sensitivity | Preference patterns unchanged | Similar results across different prompt strictness levels |
| Cross-LLM Consistency | Top-3 libraries identical | All 8 LLMs shared the same top three libraries (NumPy, pandas, Matplotlib) |

### Key Findings
*   Library usage distributions across all LLMs are highly similar, with the top three being consistent (NumPy > pandas > Matplotlib), regardless of model size or open/closed-source status.
*   Even when tasks explicitly require high performance (low-latency trading, parallel processing), Python remains dominant and Rust is entirely absent.
*   Consistency between the technology stacks LLMs "recommend" and what they actually use is extremely low, indicating preferences are rooted in generation behavior rather than at the knowledge level.

## Highlights & Insights
*   The finding that "LLMs know what is better but do not necessarily act on it" is significant, suggesting preferences in code generation may stem from training data distribution rather than reasoning.
*   The study serves as a warning for the software ecosystem: large-scale use of LLMs may create a positive feedback loop—preference for popular libraries → generation of more code using those libraries → more training data → stronger preference.
*   The experimental design is concise yet powerful, with the three experiments forming a complete chain of evidence.

## Limitations & Future Work
*   Only 8 LLMs were tested, excluding the latest reasoning-enhanced models (e.g., o1, DeepSeek-R1).
*   Library analysis was in-depth for Python, but library preferences in other languages were not explored.
*   No specific debiasing methods were proposed; the work remains primarily descriptive of the phenomenon.
*   Future research could investigate how fine-tuning and RLHF affect technical preferences.

## Related Work & Insights
*   **vs LLM Social Bias Research**: Expands bias analysis from social dimensions to technical dimensions, opening a new research direction.
*   **vs Code Generation Evaluation**: Introduces "quality of design decisions" as an overlooked but important evaluation dimension.
*   **vs Tool Recommendation Systems**: Reveals the limitations of LLMs as "implicit recommendation systems."

## Rating
*   Novelty: ⭐⭐⭐⭐⭐ First systematic study of LLM technical preferences, opening new directions.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Design is complete with 8 models, multiple scenarios, and consistency checks.
*   Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definition, concise experiments, and impactful findings.
*   Value: ⭐⭐⭐⭐ Provides important warnings for both LLM developers and users.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Understanding Structured Financial Data with LLMs: A Case Study on Fraud Detection](understanding_structured_financial_data_with_llms_a_case_study_on_fraud_detectio.md)
- [\[ICML 2026\] Deep Networks Learn to Parse Uniform-Depth Context-Free Languages from Local Statistics](../../ICML2026/llm_nlp/deep_networks_learn_to_parse_uniform-depth_context-free_languages_from_local_sta.md)
- [\[NeurIPS 2025\] Are Language Models Efficient Reasoners? A Perspective from Logic Programming](../../NeurIPS2025/llm_nlp/are_language_models_efficient_reasoners_a_perspective_from_logic_programming.md)
- [\[NeurIPS 2025\] EnCompass: Enhancing Agent Programming with Search Over Program Execution Paths](../../NeurIPS2025/llm_nlp/encompass_enhancing_agent_programming_with_search_over_program_execution_paths.md)
- [\[ACL 2026\] VOYAGER: A Training Free Approach for Generating Diverse Datasets using LLMs](voyager_a_training_free_approach_for_generating_diverse_datasets_using_llms.md)

</div>

<!-- RELATED:END -->

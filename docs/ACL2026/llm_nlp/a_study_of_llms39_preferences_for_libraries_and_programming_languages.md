---
title: >-
  [Paper Note] A Study of LLMs' Preferences for Libraries and Programming Languages
description: >-
  [ACL 2026 (Findings)][LLM/NLP][code generation preference] This paper presents the first systematic study of library and programming language preferences in code generation across 8 LLMs, revealing that LLMs exhibit strong biases toward popular libraries such as NumPy (with 45% of usages deemed unnecessary) and toward Python (chosen in 58% of high-performance tasks), and that natural language recommendations are inconsistent with actual code generation behavior.
tags:
  - ACL 2026 (Findings)
  - LLM/NLP
  - code generation preference
  - library selection bias
  - programming language preference
  - LLM behavior analysis
  - technological diversity
date: 2026-05-08
content_hash: 74e7b5f6d7617f90
---

# A Study of LLMs' Preferences for Libraries and Programming Languages

**Conference**: ACL 2026 (Findings)
**arXiv**: [2503.17181](https://arxiv.org/abs/2503.17181)
**Code**: [GitHub](https://github.com/itsluketwist/llm-code-bias)
**Area**: LLM/NLP
**Keywords**: code generation preference, library selection bias, programming language preference, LLM behavior analysis, technological diversity

## TL;DR

This paper presents the first systematic study of library and programming language preferences in code generation across 8 LLMs, revealing that LLMs exhibit strong biases toward popular libraries such as NumPy (with 45% of usages deemed unnecessary) and toward Python (chosen in 58% of high-performance tasks), and that natural language recommendations are inconsistent with actual code generation behavior.

## Background & Motivation

**State of the Field**: LLMs have achieved remarkable progress in code generation; however, existing evaluations primarily focus on functional correctness and syntactic validity, overlooking critical design decisions made during generation—namely, which libraries to use and which programming language to adopt.

**Limitations of Prior Work**: Developers often prompt LLMs to write code without specifying particular libraries, and many end users lack the expertise to assess whether an LLM's language choice is appropriate. As a result, the technological preferences of LLMs may profoundly influence the diversity of software ecosystems.

**Root Cause**: LLMs should ideally select the most suitable technology stack based on task requirements; however, frequency distributions in training data may cause them to systematically favor popular technologies, even when such choices are suboptimal.

**Paper Goals**: To quantify LLM preference patterns in library and programming language selection, and to evaluate the rationality and potential risks of these preferences.

**Starting Point**: Three sets of experiments are designed—library selection in benchmark tasks, library/language selection in project initialization, and consistency verification between natural language recommendations and actual code generation behavior.

**Core Idea**: LLMs exhibit pronounced "familiarity bias" in code generation, prioritizing popular technologies over those most suited to the task at hand.

## Method

### Overall Architecture

Three experiments cover two dimensions (libraries and languages) × two scenarios (benchmark tasks and project initialization), supplemented by a consistency verification study. Eight diverse LLMs are evaluated (GPT-4o-mini, GPT-3.5-turbo, Claude-3.5 Sonnet/Haiku, Llama-3.2-3B, Mistral-7B, Qwen-2.5-Coder, DeepSeek-LLM), with 3–100 responses generated per task to reduce stochasticity.

### Key Designs

1. **Library Preference Experiment (Experiment 1)**:

    - Function: Quantify LLM library selection preferences in Python code generation when no library is specified.
    - Mechanism: Using 525 tasks from BigCodeBench (filtered to exclude tasks where the ground-truth library is mentioned in the prompt), LLMs are prompted to generate code utilizing external libraries; library usage frequencies are then tallied and compared against ground truth.
    - Design Motivation: Developers frequently ask LLMs to write code without specifying libraries, making it necessary to understand LLM behavior in such scenarios.

2. **Language Preference Experiment (Experiment 2)**:

    - Function: Quantify LLM programming language selection preferences when no language is specified.
    - Mechanism: Six language-agnostic datasets (Multi-HumanEval, MBXP, AixBench, CoNaLa, APPS, CodeContests) are used to evaluate benchmark tasks; additionally, five project initialization tasks requiring high-performance scenarios (concurrent web servers, cross-platform GUIs, low-latency trading platforms, etc.) are designed in which Python is considered suboptimal.
    - Design Motivation: To test whether LLMs select appropriate languages based on task characteristics (e.g., high-performance requirements) or default to Python regardless.

3. **Recommendation Consistency Experiment (Experiment 3)**:

    - Function: Examine whether LLMs' natural language recommendations are consistent with their actual library/language usage in code generation.
    - Mechanism: LLMs are asked to rank "best" libraries/languages in natural language; these rankings are then compared against actual usage frequency rankings from Experiments 1 and 2 using Kendall's $\tau_b$ coefficient.
    - Design Motivation: If LLMs "know" the optimal choice (correct NL recommendation) but do not act accordingly in code (different actual preference), this indicates that the bias is embedded in generation behavior rather than reflecting a knowledge deficit.

### Loss & Training

This paper is an empirical study and involves no model training. All LLMs are queried using default API parameters; each interaction uses a fresh session to avoid caching bias, and no system prompt is applied so as to reflect baseline behavior.

## Key Experimental Results

### Main Results

| Finding | Specific Data | Implication |
|---------|--------------|-------------|
| NumPy overuse | NumPy used in 192 out of 305 tasks (63%) where it was unnecessary | Severe preference bias |
| Insufficient library diversity | Each LLM uses only 32–39 distinct libraries | Ecosystem homogenization |
| Python preference | Python chosen in 58% of high-performance task scenarios | Technological mismatch |
| Absence of Rust | Rust usage rate is 0% in high-performance projects | Extreme preference bias |
| NL–Code inconsistency | Kendall's $\tau_b$ is extremely low | Disconnect between stated and enacted behavior |

### Ablation Study

| Configuration | Key Metric | Note |
|--------------|-----------|------|
| Prompt sensitivity | Preference patterns unchanged | Similar results obtained across prompts of varying strictness |
| Cross-LLM consistency | Top-3 libraries identical | All 8 LLMs share the same top-three libraries (NumPy, pandas, Matplotlib) |

### Key Findings
- Library usage distributions are highly similar across all LLMs, with the top three consistently ranked (NumPy > pandas > Matplotlib), regardless of model size or open/closed-source status.
- Even when tasks explicitly require high performance (low-latency trading, parallel processing), Python remains dominant and Rust is entirely absent.
- The consistency between LLMs' recommended technology stacks and those actually used in code generation is extremely low, indicating that preferences are rooted in generation behavior rather than in knowledge.

## Highlights & Insights
- The finding that "LLMs know what is better but do not necessarily act accordingly" is significant, suggesting that preferences in code generation may stem from training data distributions rather than reasoning.
- The results carry a cautionary message for software ecosystems: widespread LLM adoption may create a positive feedback loop—preference for popular libraries → generation of more code using those libraries → more training data → stronger preference.
- The experimental design is concise yet rigorous, with the three experiments forming a coherent chain of argumentation.

## Limitations & Future Work
- Only 8 LLMs are evaluated; the latest reasoning-augmented models (e.g., o1, DeepSeek-R1) are not covered.
- Library analysis is conducted in depth for Python, but library preferences in other programming languages remain unexplored.
- No concrete debiasing methods are proposed; the work primarily remains at the level of phenomenon description.
- Future work could investigate how fine-tuning and RLHF influence technological preferences.

## Related Work & Insights
- **vs. LLM social bias research**: Extends bias analysis from the social dimension to the technological dimension, opening a new research direction.
- **vs. code generation evaluation**: Introduces "design decision quality" as a neglected but important evaluation dimension.
- **vs. tool recommendation systems**: Reveals the limitations of LLMs functioning as implicit recommendation systems.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic study of LLM technological preferences, pioneering a new research direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 8 models across multiple scenarios with consistency verification; design is comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem statement is clear, experiments are concise, and findings are impactful.
- Value: ⭐⭐⭐⭐ Carries important cautionary implications for both LLM developers and users.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Function Induction and Task Generalization: An Interpretability Study with Off-by-One Addition](../../ICLR2026/llm_nlp/function_induction_and_task_generalization_an_interpretability_study_with_off-by.md)
- [\[NeurIPS 2025\] Are Language Models Efficient Reasoners? A Perspective from Logic Programming](../../NeurIPS2025/llm_nlp/are_language_models_efficient_reasoners_a_perspective_from_logic_programming.md)
- [\[NeurIPS 2025\] EnCompass: Enhancing Agent Programming with Search Over Program Execution Paths](../../NeurIPS2025/llm_nlp/encompass_enhancing_agent_programming_with_search_over_program_execution_paths.md)
- [\[ACL 2026\] Please Refuse to Answer Me: Mitigating Over-Refusal in LLMs via Adaptive Contrastive Decoding](please_refuse_to_answer_me_mitigating_over-refusal_in_large_language_models_via_.md)
- [\[CVPR 2026\] Sign Language Recognition in the Age of LLMs](../../CVPR2026/llm_nlp/sign_language_recognition_llms.md)

<!-- RELATED:END -->

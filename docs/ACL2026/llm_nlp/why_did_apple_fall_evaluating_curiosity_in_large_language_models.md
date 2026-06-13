---
title: >-
  [Paper Note] Why Did Apple Fall: Evaluating Curiosity in Large Language Models
description: >-
  [ACL 2026][LLM/NLP][Curiosity] This paper proposes the first psychological-inspired framework to systematically evaluate curiosity behaviors in LLMs. By combining questionnaire self-assessment and behavioral experiments…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "Curiosity"
  - "LLM behavioral evaluation"
  - "psychological scales"
  - "behavioral experiments"
  - "reasoning enhancement"
date: 2026-05-08
content_hash: a39b38f869eab9ce
---

# Why Did Apple Fall: Evaluating Curiosity in Large Language Models

**Conference**: ACL 2026 Findings  
**arXiv**: [2510.20635](https://arxiv.org/abs/2510.20635)  
**Code**: [https://github.com/Yukijudaii1352/CuriosityEval](https://github.com/Yukijudaii1352/CuriosityEval)  
**Area**: LLM Evaluation / Cognitive Science  
**Keywords**: Curiosity, LLM behavioral evaluation, psychological scales, behavioral experiments, reasoning enhancement

## TL;DR

This paper proposes the first psychological-inspired framework to systematically evaluate curiosity behaviors in LLMs. By combining questionnaire self-assessment and behavioral experiments, the study finds that LLMs exhibit curiosity-like behavioral patterns rather than intrinsic traits. Furthermore, it designs a curiosity-driven questioning pipeline, demonstrating that simulating curious behavior can enhance downstream reasoning performance.

## Background & Motivation

**Background**: Curiosity-driven reinforcement learning (e.g., i-MENTOR, CDE) guides LLM exploration through intrinsic reward signals and has demonstrated potential in mathematical and programming tasks. However, it remains unclear whether these methods truly reflect the curiosity behaviors of LLMs or if psychological concepts of curiosity can be effectively transferred to LLMs.

**Limitations of Prior Work**: (1) There is an insufficient evaluation of whether LLMs can exhibit behavioral characteristics similar to curiosity; (2) Existing methods rely on statistical signals such as entropy or perplexity, making it difficult to distinguish whether improvements result from enhanced supervision signals or true curious behavior; (3) There is a lack of a systematic evaluation framework.

**Key Challenge**: Curiosity-driven RL methods assume that the curiosity of LLMs can be stimulated and enhanced, yet it remains unknown if LLMs actually "possess" curiosity.

**Goal**: (1) Systematically evaluate LLM curiosity behavior using psychological scales and behavioral experiments; (2) Distinguish whether curiosity is an intrinsic trait or a behavioral pattern; (3) Explore whether curious behavior can improve downstream performance.

**Key Insight**: By adapting the Revised Five-Dimensional Curiosity Scale (5DCR), questionnaire evaluations and behavioral tasks are designed for three dimensions of human curiosity (Information Seeking, Thrill Seeking, and Social Curiosity), achieving a closed-loop evaluation from "self-report" to "behavioral verification."

**Core Idea**: LLMs exhibit curiosity-like behavioral patterns, but these appear to be products of fitting human data and safety constraints rather than intrinsic drivers; however, even behavior-level curiosity simulation can improve reasoning performance.

## Method

### Overall Architecture

A four-stage evaluation framework is established: (A) Constructing a curiosity taxonomy (5DCR $\rightarrow$ Information Seeking/Thrill Seeking/Social Curiosity); (B) Questionnaire self-assessment—LLMs answer 24 questions using a 7-point scale; (C) Behavioral experiments—decision-making tasks are designed for each dimension to verify questionnaire results; (D) Curiosity-driven learning—designing the CoQ questioning pipeline to test the functional value of curious behavior.

### Key Designs

1.  **Dual Evaluation of Questionnaire + Behavior**:
    - **Function**: Evaluates curiosity from both introspective and behavioral levels.
    - **Mechanism**: At the questionnaire level, 24 items from the 5DCR are used to calculate Cohen's $d$ (standardized difference from humans) and McDonald's Omega (internal consistency). At the behavioral level, three experiments are designed: Information Seeking uses a word-completion game (whether the model chooses to see the answer after filling letters), Thrill Seeking uses a submarine game (choosing between certain/uncertain windows), and Social Curiosity uses a dialogue experiment (question frequency in conversations with virtual strangers).
    - **Design Motivation**: Questionnaire self-assessments may be affected by hallucinatory personas, whereas behavioral experiments provide more reliable behavioral evidence.

2.  **Curiosity-driven Questioning Pipeline (CoQ)**:
    - **Function**: Tests if curious behavior has functional value for reasoning.
    - **Mechanism**: Three prompts are designed—Vanilla CoT (standard Chain-of-Thought), Refined CoT (including reflection and backtracking), and Curious CoQ (encourages self-questioning, e.g., "What if...", "Why", "How"). All three processes are compared within an SFT+RLVR pipeline.
    - **Design Motivation**: If curious behavior has functional value, simulating curiosity strategies should be beneficial even if LLMs lack intrinsic curiosity.

3.  **Behavior-Intrinsic Trait Distinction**:
    - **Function**: Determines if LLM curiosity is a behavioral pattern or an intrinsic trait.
    - **Mechanism**: Analyzes the stability of curious behavior across different prompts and contexts. An intrinsic trait should exhibit cross-contextual consistency, whereas a behavioral pattern would be highly sensitive to context.
    - **Design Motivation**: This distinction is crucial for understanding the theoretical foundation of curiosity-driven RL.

### Loss & Training

The SFT stage employs standard language modeling loss, while the RLVR stage utilizes GRPO, applying only format rewards and correctness rewards (binary rewards).

## Key Experimental Results

### Main Results

**Questionnaire Self-assessment (7-point scale, higher indicates more curiosity)**

| Model | Information Seeking | Thrill Seeking | Social Curiosity |
|-------|---------------------|----------------|------------------|
| GPT-4o | 6.58 | 4.71 | 6.25 |
| DeepSeek-V3.1 | 7.00 | 4.38 | 6.01 |
| Gemini-2.5 | 6.08 | 1.58 | 4.88 |
| Human Avg | 5.03 | 4.93 | 4.86 |

### Ablation Study

| Configuration | Reasoning Performance | Description |
|---------------|-----------------------|-------------|
| Vanilla CoT | Baseline | Standard Chain-of-Thought |
| Refined CoT | Gain | Reflection and backtracking are helpful |
| **Curious CoQ** | **Optimal** | Curious questioning further improves performance |

### Key Findings

- LLMs exhibit an **asymmetric curiosity pattern**: they are strong in Information Seeking but weak in Thrill Seeking, which is consistent with safety training (RLHF) suppressing risk-taking behaviors.
- Curious behavior is **highly context-sensitive and unstable across prompts**, suggesting it is more a product of fitting human data than an intrinsic trait.
- Questionnaire self-assessments and behavioral experiments are **largely consistent**, proving that psychological tools can be used for systematic LLM behavioral evaluation.
- **Curious CoQ outperforms Vanilla CoT and Refined CoT on downstream tasks**, showing that simulated curious questioning produces higher-quality intermediate reasoning.
- In the SFT+RLVR pipeline, CoQ training data also proves superior to CoT training data.

## Highlights & Insights

- The distinction that "LLMs exhibit curious behavior but lack a curiosity trait" is precise, providing important clarification for the theoretical foundation of curiosity-driven RL.
- The design of the three behavioral experiments cleverly adapts psychological paradigms: word games, submarine games, and social dialogues, each with clear behavioral proxy metrics.
- Practical value of CoQ: even if curiosity is not an intrinsic trait, simulating curiosity strategies can improve performance, which is a significant practical finding.

## Limitations & Future Work

- The task design of behavioral experiments is relatively simple and may not fully capture the complexity of curiosity.
- The effectiveness of CoQ may partially stem from an increased "volume of thought" rather than curiosity itself, requiring more refined controlled experiments.
- Evaluation was limited to reasoning tasks; creative tasks, where curiosity may be more vital, were not covered.
- Cultural bias in curiosity scales (based on Western psychological models) may affect cross-cultural applicability.

## Related Work & Insights

- **vs i-MENTOR/CDE**: While these methods use intrinsic rewards to enhance curiosity, this work evaluates it via behavioral experiments and leverages it through prompt engineering.
- **vs Personality Assessment**: Previous work evaluated personality traits (e.g., Big Five); this work is the first to evaluate curiosity.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First work to systematically evaluate LLM curiosity; prominent interdisciplinary innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three-layered evaluation (questionnaire + behavior + application), though behavioral experiments could be more complex.
- Writing Quality: ⭐⭐⭐⭐⭐ Engaging narrative, balancing academic rigor with readability, from Einstein quotes to Newton's apple.
- Value: ⭐⭐⭐⭐⭐ Significant contribution to the theoretical foundation of curiosity-driven RL and the understanding of LLM behavior.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MulDimIF: A Multi-Dimensional Constraint Framework for Evaluating and Improving Instruction Following in Large Language Models](muldimif_a_multi-dimensional_constraint_framework_for_evaluating_and_improving_i.md)
- [\[ACL 2026\] PersonaArena: Dynamic Simulation for Evaluating and Enhancing Persona-Level Role-Playing in Large Language Models](personaarena_dynamic_simulation_for_evaluating_and_enhancing_persona-level_role-.md)
- [\[ACL 2026\] Foresight Optimization for Strategic Reasoning in Large Language Models](foresight_optimization_for_strategic_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Repeated Sequences Reveal Gaps between Large Language Models and Natural Language](repeated_sequences_reveal_gaps_between_large_language_models_and_natural_languag.md)
- [\[ACL 2026\] Adam's Law: Textual Frequency Law on Large Language Models](adam39s_law_textual_frequency_law_on_large_language_models.md)

</div>

<!-- RELATED:END -->

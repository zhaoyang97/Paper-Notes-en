---
title: >-
  [Paper Note] BIG-Bench Extra Hard
description: >-
  [ACL 2025][LLM (Other)][LLM Evaluation] To address the issue of BIG-Bench Hard being saturated by state-of-the-art models, Google DeepMind introduces BIG-Bench Extra Hard (BBEH), replacing the corresponding tasks in BBH with 23 significantly harder tasks. The strongest general model achieves only 9.8% (harmonic mean) while the strongest reasoning model reaches 44.8%, revealing a huge gap in LLMs' general reasoning capabilities.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "LLM Evaluation"
  - "General Reasoning"
  - "BIG-Bench"
  - "Benchmark Saturation"
  - "Reasoning Ability"
date: 2026-05-08
content_hash: 6f54300e638f1a69
---

# BIG-Bench Extra Hard

**Conference**: ACL 2025  
**arXiv**: [2502.19187](https://arxiv.org/abs/2502.19187)  
**Code**: [github.com/google-deepmind/bbeh](https://github.com/google-deepmind/bbeh)  
**Area**: Others  
**Keywords**: LLM Evaluation, General Reasoning, BIG-Bench, Benchmark Saturation, Reasoning Ability

## TL;DR

To address the issue of BIG-Bench Hard being saturated by state-of-the-art models, Google DeepMind introduces BIG-Bench Extra Hard (BBEH), replacing the corresponding tasks in BBH with 23 significantly harder tasks. The strongest general model achieves only 9.8% (harmonic mean) while the strongest reasoning model reaches 44.8%, revealing a huge gap in LLMs' general reasoning capabilities.

## Background & Motivation

Current evaluations of LLM reasoning capabilities are heavily biased toward mathematics and programming, whereas reasoning actually encompasses a broad range of cognitive skills, including logical deduction, spatiotemporal understanding, commonsense reasoning, and humor comprehension. For a long time, BIG-Bench Hard (BBH) has served as the de facto standard for evaluating general reasoning, but the following issues have led to its gradual obsolescence:

**Performance Saturation**: State-of-the-art models have reached 90%+ accuracy on BBH, failing to differentiate between model capabilities.

**High Random Baseline**: 8 out of 23 tasks are binary classification, and 5 out of 23 tasks have no more than 5 options.

**Shortcut Vulnerability**: Some tasks can be bypassed without reasoning through simple rules (e.g., three 'L' commands automatically mean a triangle).

**Short Input Length**: The macro-average input length of BBH tasks is only around 700 characters.

**Few Reasoning Steps**: Most problems require only a small number of reasoning steps.

**Limited Skill Coverage**: Although the variety of skills is broad, it can still be significantly expanded.

BBEH aims to drastically increase the difficulty and expand the required reasoning skills while retaining the diversity advantages of BBH.

## Method

### Overall Architecture

BBEH replaces the original tasks in BBH one-by-one with 23 new tasks. Each new task:
- Lies in the same reasoning domain.
- Tests similar (or more) reasoning skills.
- Is significantly more difficult.
- Contains 200 samples per task (120 for Disambiguation QA).

### Key Designs

1. **Significant Expansion of Required Reasoning Skills**: Building upon the original 11 categories of skills in BBH (temporal understanding, spatial geometry, commonsense, humor, causality, world knowledge, logical deduction, linguistic knowledge, counting/filtering, data structures & algorithms, arithmetic), 12 new high-order skill requirements have been added:

    - Many-hop reasoning
    - Very long-range dependency
    - Going against strong prior
    - Learning on the fly
    - Dealing with distractors
    - Long-context
    - Needle in a haystack
    - Finding errors in reasoning traces
    - Inductive reasoning
    - Constraint satisfaction
    - Compositional understanding
    - Knowledge-intensive reasoning

2. **Semi-Adversarial Difficulty Calibration**: Two reference models—Gemini 1.5 Flash (general) and Gemini-2.0-Flash-Thinking-Exp (reasoning)—were selected. Task difficulty was iteratively increased until the accuracy of both reference models dropped below 70%. Models were generally treated as black boxes, but model strategies were analyzed when necessary (e.g., after discovering that a model used Python to execute boolean expressions directly, the tasks were modified to replace True/False with natural language sub-expressions).

3. **Exemplary Task Upgrade Examples**:

    - **Boolean Expressions**: Replaces "True" with textual sub-expressions like "The capital of Canada is Ottawa" to prevent models from using code execution.
    - **Buggy Tables**: Upgraded from simple table queries to understanding and reconstructing large, flawed tables.
    - **Object Counting**: Upgraded from simple counting in a short list to counting specific types with heavily distracting elements in extremely long lists.
    - **Word Sorting**: Upgraded from standard alphabetical sorting to sorting with a modified alphabetical order (violating strong priors) + locating sorting errors.

### Dataset Properties

- **Input Length**: The macro-average context length of BBEH is approximately 6 times that of BBH.
- **Required Reasoning Budget**: Using the output length of Gemini 2.0 Flash as a proxy, BBEH requires approximately 7 times the thinking budget of BBH.
- **Random Baseline**: The overall random baseline for BBEH is 8.4%, significantly lower than BBH.
- **BBEH Mini**: Contains 460 samples (20 per task) for rapid and low-cost experimentation.

## Key Experimental Results

### Main Results (BBEH Harmonic Mean Accuracy)

| Model | Type | BBEH Harmonic Mean ↑ |
|------|------|-------------|
| Qwen-2.5-7B-Instruct | General | 2.4% |
| Llama 3.1 8B Instruct | General | 3.0% |
| Gemma2 27B IT | General | 3.6% |
| Gemma3 27B | General | 4.5% |
| Gemini 2.0 Flash-Lite | General | 4.9% |
| Gemini 2.0 Flash | General | 8.0% |
| GPT-4o | General | **9.8%** |
| Distill R1 Qwen 32B | Reasoning | 6.0% |
| DeepSeek R1 | Reasoning | 5.2% |
| o3-mini (high) | Reasoning | **44.8%** |

### Single-Task Results Highlights

| Task | GPT-4o | o3-mini | DeepSeek R1 | Description |
|------|--------|---------|-------------|------|
| Buggy Tables | 3.5 | 59.5 | 4.5 | Significant advantage for reasoning models |
| Object Counting | 11.0 | 90.0 | 76.5 | Reasoning models excel dramatically in counting |
| Object Properties | 1.5 | 56.5 | 0.0 | o3-mini stands out |
| Temporal Sequences | 0.5 | 68.5 | 0.0 | o3-mini stands out |
| NYCC (Humor) | 23.0 | 16.0 | 20.0 | GPT-4o leads in humor comprehension |
| SARC Triples (Sarcasm) | 38.5 | 24.0 | 28.5 | GPT-4o leads |
| Causal Understanding | 54.0 | 54.0 | 54.5 | Performance is comparable across models in causal understanding |

### Key Findings

1. **General Reasoning Remains Highly Challenging**: The strongest general model, GPT-4o, achieves a harmonic mean of only 9.8%, indicating that even state-of-the-art LLMs still have immense room for improvement in general reasoning.
2. **Imbalanced Advantages of Reasoning Models**: o3-mini exhibits a massive advantage in formal problems such as counting, planning, arithmetic, and data structures. However, in "soft" reasoning skills like commonsense, humor, sarcasm, and causal reasoning, it even underperforms compared to general models.
3. **Similar Influence of Model Size**: Larger models show significant improvement in formal reasoning but minor improvements in soft reasoning.
4. **Impact of Context Length and Thinking Budget**: The advantage of o3-mini increases with the context length and the required thinking budget.
5. **Imbalanced Performance of DeepSeek R1**: Its micro-average accuracy outperforms general models, but because it scores extremely low on certain tasks, its harmonic mean is lower than that of the two general models.
6. **Different Models Excel at Different Reasoning Types**: No single model consistently leads across all tasks.

## Highlights & Insights

1. **Choice of Evaluation Metric**: Using the harmonic mean instead of the arithmetic mean or micro-average penalizes model "unbalancedness" more effectively, reflecting true general reasoning capability—which is an important methodological contribution.
2. **Revealing the True Boundaries of Reasoning Models**: Current reasoning models have achieved leaps in formal problems (e.g., AIME2024 jumping from 13.4% to 87.3%), but have made little progress on tasks requiring soft reasoning, such as commonsense, humor, and causality.
3. **Systemic Benchmark Design Methodology**: The design process of preserving BBH's strengths + fixing its flaws + semi-adversarial calibration provides a paradigm for constructing future benchmarks.
4. **Proxy for Real-World Reasoning Capability**: Tasks in BBEH are closer to real-world scenarios (long context, multiple distractors, requiring multi-step reasoning), making it a better proxy for reflecting actual reasoning capabilities than pure math/coding benchmarks.

## Limitations & Future Work

- **Reference Model Bias**: Semi-adversarial construction inevitably targets the weaknesses of the specific reference models, raising fairness concerns for evaluations of non-reference models.
- **Temporality of Static Benchmarks**: As models continue to advance, BBEH will eventually become saturated.
- **Text-Only Limitation**: It does not evaluate multimodal reasoning capabilities.
- **Limited Sample Size**: With only 200 samples per task, the statistical significance may be insufficient.
- **Lack of Process-Based Evaluation**: It only assesses final answer accuracy, failing to analyze the quality of the model's intermediate reasoning process.
- **Evaluation Cost**: The output lengths of reasoning models on BBEH are very long, making the evaluation cost significantly higher than that of BBH.

## Related Work & Insights

- **Succession to BBH**: Directly replaces the 23 tasks of BBH, maintaining consistency in domains.
- **Complementarity with Math/Coding Benchmarks**: Benchmarks like AIME and GSM8K evaluate mathematical reasoning, whereas BBEH evaluates a broader range of general reasoning.
- **Inspiration from Inverse Scaling**: Certain tasks specifically test whether models can act against existing priors (e.g., modified alphabetical sorting), aligning with inverse scaling research.
- **Guidance for Model Development**: The differentiated performance across various types of reasoning capabilities provides a guide for balancing abilities in model training.

## Rating

- Novelty: ⭐⭐⭐⭐ Clever task design, comprehensive skill coverage, with innovative semi-adversarial construction methodology.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple mainstream model series with rich analytical dimensions (model size, type, context length, etc.).
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, rigorous structure, in-depth analysis, and outstanding visualization.
- Value: ⭐⭐⭐⭐⭐ Fills the gap in general reasoning evaluation and will serve as an important benchmark in the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Zero-Shot Belief: A Hard Problem for LLMs](zero-shot_belief_a_hard_problem_for_llms.md)
- [\[ACL 2025\] LR²Bench: Evaluating Long-chain Reflective Reasoning Capabilities of Large Language Models via Constraint Satisfaction Problems](lr2bench_evaluating_long-chain_reflective_reasoning_capabilities_of_large_langua.md)
- [\[ACL 2025\] DICE-Bench: Evaluating the Tool-Use Capabilities of Large Language Models in Multi-Round, Multi-Party Dialogues](dice-bench_evaluating_the_tool-use_capabilities_of_large_language_models_in_mult.md)
- [\[ACL 2026\] Big AI is Accelerating the Metacrisis: What Can We Do?](../../ACL2026/llm_nlp/big_ai_is_accelerating_the_metacrisis_what_can_we_do.md)
- [\[ACL 2025\] Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights](unintended_harms_of_value-aligned_llms_psychological_and_empirical_insights.md)

</div>

<!-- RELATED:END -->

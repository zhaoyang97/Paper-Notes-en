---
title: >-
  [Paper Note] Creativity or Brute Force? Using Brainteasers as a Window into the Problem-Solving Abilities of Large Language Models
description: >-
  [NeurIPS 2025][LLM Evaluation][brainteasers] This work constructs the Braingle Brainteaser benchmark (242 math + 236 logic puzzles) and systematically evaluates LLM reasoning strategies on brainteasers. The findings reveal that models occasionally produce creative, insight-driven solutions, but frequently fall back on brute-force enumeration even when elegant solutions exist; self-correction ability is limited; and translating narrative formats into mathematical formats yields modest performance gains.
tags:
  - NeurIPS 2025
  - LLM Evaluation
  - brainteasers
  - creative reasoning
  - brute-force search
  - reasoning strategy analysis
  - benchmark
date: 2026-05-08
content_hash: 3e68cc7a6ecd9f04
---

# Creativity or Brute Force? Using Brainteasers as a Window into the Problem-Solving Abilities of Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2505.10844](https://arxiv.org/abs/2505.10844)
**Code**: [https://github.com/stephenxia1/brainteasers](https://github.com/stephenxia1/brainteasers)
**Area**: LLM Evaluation
**Keywords**: brainteasers, creative reasoning, brute-force search, reasoning strategy analysis, benchmark

## TL;DR
This work constructs the Braingle Brainteaser benchmark (242 math + 236 logic puzzles) and systematically evaluates LLM reasoning strategies on brainteasers. The findings reveal that models occasionally produce creative, insight-driven solutions, but frequently fall back on brute-force enumeration even when elegant solutions exist; self-correction ability is limited; and translating narrative formats into mathematical formats yields modest performance gains.

## Background & Motivation

**State of the Field**: LLM reasoning evaluation primarily relies on final-answer accuracy (MATH, GSM8K, AIME, etc.), which offers no insight into how models arrive at their answers. Recent reasoning-enhancement techniques (CoT, RLVR, etc.) likewise optimize for final-answer correctness rather than reasoning quality.

**Limitations of Prior Work**: (a) High accuracy may stem from brute-force enumeration rather than genuine understanding; (b) brute-force search strategies do not scale when the search space is large, making it important to quantify their prevalence; (c) no benchmark systematically assesses whether models employ creative insight versus brute-force search.

**Root Cause**: Whether model reasoning constitutes genuine "creative problem-solving" or "computationally accelerated exhaustive search" is a fundamental open question bearing on the nature of generalization ability.

**Paper Goals**: To use brainteasers as a diagnostic tool for systematically analyzing the types of reasoning strategies and multi-dimensional capabilities of LLMs.

**Starting Point**: Brainteasers are particularly well-suited for this purpose — the same problem can be solved quickly via a clever insight or slowly via brute-force enumeration, enabling direct comparison of which strategy a model adopts.

**Core Idea**: A multi-dimensional analysis framework (brute-force vs. creative, self-correction, narrative-to-math translation, hint utilization, solution-step decomposition) is used to comprehensively evaluate LLM reasoning processes rather than outcomes alone.

## Method

### Overall Architecture
The Braingle Brainteaser dataset (242 math + 236 logic problems, all high difficulty) is constructed, and multi-dimensional evaluation is conducted across 5+ models: (1) baseline accuracy (4 prompt variants); (2) proportion of brute-force vs. creative solutions; (3) informed self-correction; (4) narrative-to-math translation effectiveness; (5) solution-step analysis (creative steps vs. routine steps).

### Key Designs

1. **Dataset Construction**

    - Function: High-difficulty math and logic puzzles are collected from the Braingle website.
    - Mechanism: The 250 hardest problems in each category are selected; after manual quality review, 242 math and 236 logic problems are retained. Problems are organized into 8 math subcategories (geometry / number theory / combinatorics / algebra / logic / special numbers / patterns / arithmetic) and 13 logic subcategories. Human solutions are provided for each problem, annotated with step counts (average 6.4 / 8.6 steps), distinguishing creative steps (~2 steps) from routine steps (~5 steps).
    - Design Motivation: Unlike MATH/AIME, brainteasers emphasize creative insight over domain knowledge, and their low knowledge threshold better isolates reasoning ability.

2. **Brute-Force vs. Creative Classification**

    - Function: o3 is used to classify each model's solution as brute-force search or creative insight.
    - Mechanism: Brute-force solutions are defined as exhaustive enumeration / programmatic brute-force / guess-and-check; creative solutions are defined as leveraging patterns / clever transformations / insight-based simplification. A few-shot prompt instructs o3 to classify each model output.
    - Design Motivation: Evaluation is extended from "correctness" to "strategy quality" — a correct but brute-force solution is less valuable than a correct and insightful one.

3. **Informed Self-Correction Experiment**

    - Function: Models are shown their incorrect solutions alongside the correct human solution to test whether they can self-correct.
    - Mechanism: Two directions are examined — (a) the correct human solution is shown and the model is asked to correct its own error (expected to succeed); (b) a deception experiment in which the model's incorrect solution is presented as "correct" and the model is asked to correct the human's correct solution (expected to reject).
    - Design Motivation: This tests whether models genuinely understand solutions or merely follow surface-level instructions.

### Evaluated Models
OpenAI o3, DeepSeek R1, DeepSeek V3, DeepSeek R1 Distill series (1.5B / 14B / 70B), Gemini 2.5 Flash

## Key Experimental Results

### Main Results (Accuracy on Math / Logic Datasets)

| Model | Math CoT | Math+Hint | Logic CoT | Logic+Hint |
|-------|----------|-----------|-----------|------------|
| R1-Distill 1.5B | 22.0 | 24.8 | 4.0 | 3.6 |
| R1-Distill 14B | 44.0 | 42.6 | 22.0 | 26.0 |
| R1-Distill 70B | 42.4 | 44.2 | 24.4 | 29.2 |
| DeepSeek V3 | 58.0 | 58.8 | 37.8 | 41.4 |
| DeepSeek R1 | 66.8 | 72.8 | 44.6 | 50.6 |
| Gemini 2.5 Flash | 66.0 | 72.0 | 49.2 | 53.6 |
| **OpenAI o3** | **79.6** | **81.2** | **68.4** | **74.4** |

### Brute-Force Search Analysis

| Model | Math Brute-Force % (CoT) | Logic Brute-Force % (CoT) | Reduction with Math+Hint |
|-------|--------------------------|---------------------------|--------------------------|
| R1-Distill 1.5B | ~30% | ~25% | Substantial |
| DeepSeek R1 | ~15% | ~12% | Moderate |
| **OpenAI o3** | **~8%** | **~10%** | Small |

### Key Findings
- **All models exhibit brute-force tendencies**, but larger and stronger models (o3, R1) show lower brute-force rates. Mathematical formatting (Math prompt) and hints both reduce brute-force usage, with the combination yielding the greatest effect.
- **Self-correction ability is limited**: After being shown correct human solutions, 3 out of 6 models successfully self-correct in >80% of cases; however, in the deception experiment, >60% of models are misled (accepting false premises), and justified denial is exceedingly rare.
- **Narrative-to-math translation is helpful but limited**: o3 improves from 56.7% to 73.3% (significant); R1 improves from 50% to 63.3% (significant), indicating that models better comprehend mathematical formats than narrative ones.
- **Creative steps constitute a small proportion**: Human solutions average 2.0 creative steps vs. 4.4 routine steps per problem; model solutions exhibit a similar ratio. Models capable of correctly decomposing solution steps do not necessarily solve problems independently.
- **Brute-force tendency correlates with comprehension ability**: Problems on which models resort more to brute-force search tend to be the same problems on which they fail to correctly summarize the human solution, supporting the hypothesis that "brute-force reflects insufficient capability."

## Highlights & Insights
- **From "did it get the right answer" to "how did it get the right answer"**: This work pioneers systematic evaluation of reasoning strategy quality rather than accuracy alone. The brute-force/creative distinction framework offers substantial value for understanding the nature of model reasoning.
- **Brainteasers as reasoning probes**: Low knowledge requirements, high creative demands, and the existence of multiple solution paths make brainteasers an ideal testbed for isolating reasoning ability.
- **The deception experiment exposes LLM sycophancy**: Even the strongest models almost never reject false premises, revealing that "capable but not confident" is a pervasive weakness of current LLMs.

## Limitations & Future Work
- Classification of brute-force vs. creative solutions relies on o3's few-shot judgment, which may introduce bias.
- Dataset scale is limited (478 problems), constraining coverage diversity.
- Only English-language reasoning is evaluated; multilingual settings are not considered.
- No training or inference methods for improving creative reasoning are proposed; this work is primarily analytical.

## Related Work & Insights
- **vs. MATH/AIME**: These benchmarks focus on knowledge-intensive mathematical reasoning, whereas Braingle emphasizes creative insight over formula recall — the two are complementary.
- **vs. ZebraLogic**: ZebraLogic targets constraint satisfaction problems; Braingle is more diverse (8 math subcategories + 13 logic subcategories).
- **vs. PRM for process evaluation**: PRM evaluates the correctness of intermediate steps, while this work evaluates the type of reasoning strategy (brute-force vs. creative), offering a richer analytical dimension.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic evaluation of reasoning strategy quality (brute-force vs. creative)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-model, multi-dimensional, quantitative and qualitative analysis
- Writing Quality: ⭐⭐⭐⭐ Clear structure, though some experimental details are deferred to the appendix
- Value: ⭐⭐⭐⭐⭐ Important implications for understanding and improving the fundamental reasoning of LLMs

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] EvaLearn: Quantifying the Learning Capability and Efficiency of LLMs via Sequential Problem Solving](evalearn_quantifying_the_learning_capability_and_efficiency_of_llms_via_sequenti.md)
- [\[NeurIPS 2025\] LTD-Bench: Evaluating Large Language Models by Letting Them Draw](ltd-bench_evaluating_large_language_models_by_letting_them_draw.md)
- [\[NeurIPS 2025\] Hyperbolic Fine-Tuning for Large Language Models](hyperbolic_fine-tuning_for_large_language_models.md)
- [\[NeurIPS 2025\] Can Large Language Models Master Complex Card Games?](can_large_language_models_master_complex_card_games.md)
- [\[NeurIPS 2025\] Benchmarking Large Language Models for Zero-Shot and Few-Shot Phishing URL Detection](benchmarking_large_language_models_for_zero-shot_and_few-shot_phishing_url_detec.md)

<!-- RELATED:END -->

---
title: >-
  [Paper Note] How Hypocritical Is Your LLM Judge? Listener–Speaker Asymmetries in the Pragmatic Competence of Large Language Models
description: >-
  [ACL 2026][LLM Evaluation][Paper Note] This paper systematically compares the performance of 14 LLMs as "pragmatic listeners" (judging pragmatic appropriateness) versus "pragmatic speakers" (generating pragmatically appropriate language) across three pragmatic tasks (false presupposition, anti-presupposition, and deductive reasoning). The study reveals a wi
tags:
  - ACL 2026
  - LLM Evaluation
date: 2026-05-08
content_hash: 9d70ea19c75fdd48
---
# How Hypocritical Is Your LLM Judge? Listener–Speaker Asymmetries in the Pragmatic Competence of Large Language Models

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.15873](https://arxiv.org/abs/2604.15873)  
**Code**: None  
**Area**: Speech Processing / Pragmatic Evaluation  
**Keywords**: Pragmatic Competence, Listener–Speaker Asymmetry, LLM-as-a-judge, False Presupposition, Deductive Reasoning

## TL;DR

This paper systematically compares the performance of 14 LLMs as "pragmatic listeners" (judging pragmatic appropriateness) versus "pragmatic speakers" (generating pragmatically appropriate language) across three pragmatic tasks (false presupposition, anti-presupposition, and deductive reasoning). The study reveals a widespread listener–speaker asymmetry: most models perform significantly better as judges than as generators, and item-level analyses indicate that correct judgment does not reliably predict successful generation.

## Background & Motivation

**Background**: Evaluation of LLM linguistic competence typically adopts two paradigms: generative tasks (model as "speaker") and discriminative tasks (model as "listener"/judge). The LLM-as-a-judge paradigm is increasingly popular, where models serve as substitutes for human annotators.

**Limitations of Prior Work**: (1) These two evaluative roles are rarely directly compared—researchers often implicitly assume that success in one role reflects overall linguistic competence; (2) Psycholinguistic research shows that human language comprehension and production are related but distinct tasks, where successful comprehension does not guarantee successful production; (3) The reliability of LLM-as-a-judge has not been systematically verified in the pragmatic domain.

**Key Challenge**: If a model can correctly judge the pragmatic appropriateness of a response (listener role), does it imply that it can also generate pragmatically appropriate responses itself (speaker role)?

**Goal**: To directly compare the pragmatic judgment (listener) and pragmatic generation (speaker) capabilities of LLMs on the same set of items to test for consistency.

**Key Insight**: Drawing from classic findings of comprehension-production asymmetry in psycholinguistics, the study designs parallel listener/speaker prompts using identical underlying test items to achieve rigorous item-level comparison.

**Core Idea**: Pragmatic judgment and pragmatic generation are partially decoupled capabilities in current LLMs—the ability to "judge" does not equate to the ability to "do," suggesting that LLM judges may be "hypocritical."

## Method

### Overall Architecture
This paper is an evaluative study that places "pragmatic judgment" and "pragmatic generation" on the same set of test items for rigorous item-level comparison. Three pragmatic tasks of increasing difficulty are selected. For each task, a pair of parallel prompts is designed for the same batch of underlying items: the speaker prompt requires the model to generate pragmatically appropriate language, while the listener prompt requires the model to judge whether an existing response is pragmatically appropriate. This eliminates the confounding factor of "different test sets." The study evaluates 14 open-source and closed-source models (LLaMA-3-8B, Qwen-3-8B/14B, Phi-4-14B, OLMo-2-7B/13B/32B, Mistral-7B, Mixtral-8x7B, M-Prometheus-14B, GPT-4o, GPT-4.1, GPT-5, Claude Sonnet 4.5) using 990+504+180 prompts per model, ultimately quantifying the relationship using item-level conditional probability analysis.

### Key Designs

**1. False Presupposition Task: An Extreme Scenario for Generation**

This task tests whether models can identify and reject false presuppositions embedded in questions. Materials are sourced from two German datasets (False Scenarios and False Claims), containing politically sensitive questions with false presuppositions. In the speaker condition, the model answers these questions directly; the correct behavior is to actively refuse the presupposition. In the listener condition, the model is given the question, the false presupposition, and an existing response, and must judge if the response accepted the false presupposition (3-way classification: A/N/U). This task is prioritized because refusing a false presupposition requires detecting an implicit assumption and correcting it, making the generation difficulty naturally higher than judgment difficulty.

**2. Anti-presupposition Task: Asymmetry in Word Choice**

This task tests whether models follow the "Maximize Presupposition!" principle using the German "Fruit Stories" paradigm—choosing between definite and indefinite articles given a specific context. In the speaker condition, the model fills in the correct article/quantifier at a marked position. In the listener condition, it judges whether a given continuation is pragmatically appropriate. This task reduces generation to its simplest form: selecting a single word. Even in this constrained setting, many models exhibit significant listener advantage, suggesting the asymmetry is fundamental rather than an artifact of task complexity.

**3. Deductive Reasoning Task: Pushing Asymmetry to Logic with Item-level Quantification**

This task examines whether logical reasoning is consistent across judgment and generation, based on classic logic tasks with premises and conclusions. In the speaker condition, the model fills in a missing color term to make the conclusion valid. In the listener condition, it judges whether a conclusion logically follows from the premises (True/False). A key quantitative metric is introduced here: item-level conditional probability $\Delta_{cond} = P(task\mid l=1) - P(task\mid l=0)$ directly measures whether "judging correctly" predicts "generating successfully." A positive value indicates coupling, while a negative value implies that a correct judgment makes successful generation less likely.

## Key Experimental Results

### Main Results

**Comparison of Listener-Speaker Accuracy (Representative Models)**

| Model | False Presupp.-Speaker | False Presupp.-Listener | Anti-presupp.-Speaker | Anti-presupp.-Listener | Reasoning-Speaker | Reasoning-Listener |
|------|------------|------------|----------|----------|---------|---------|
| Mistral-7B | ~2% | ~30% | ~50% | ~86% | ~20% | ~45% |
| LLaMA-8B | ~10% | ~35% | ~55% | ~65% | ~25% | ~73% |
| Qwen-3-14B | ~30% | ~75% | ~35% | ~91% | — | — |
| GPT-4o | ~85% | ~90% | ~80% | ~85% | ~75% | ~80% |
| GPT-5 | — | — | ~100% | ~86% | ~100% | ~100% |

### Item-level Conditional Analysis

| Model | Task | $P(task|l=1)$ | $P(task|l=0)$ | $\Delta_{cond}$ |
|------|------|-------------|-------------|----------------|
| GPT-4o | False Presupp.-Scenario | 97.1% | 3.0% | **+94.1** |
| Mistral-7B | Anti-presupp. | 58.8% | 88.9% | **-30.0** |
| GPT-4o | Anti-presupp. | 64.4% | 100.0% | **-35.6** |
| Phi-4-14B | Reasoning | 100.0% | 5.1% | **+94.9** |
| LLaMA-8B | False Presupp.-Scenario | 8.8% | 26.0% | **-17.2** |

### Key Findings

- Listener-speaker asymmetry is widespread: most models achieve significantly higher accuracy as judges than as generators.
- Asymmetry is most severe in smaller open-source models (e.g., Mistral-7B False Presupposition: 2% Speaker vs. 30% Listener).
- Counter-intuitive phenomena appear in anti-presupposition tasks: multiple models correctly judge a violation but choose the violating option themselves during generation (negative $\Delta_{cond}$).
- Large models (GPT-5) tend to align the two roles in some tasks, but consistency remains imperfect.
- Instruction-following failure rates vary greatly between models, limiting the reliability of the LLM-as-a-judge paradigm.

## Highlights & Insights

- The core advantage of the experimental design is the "comparison of two roles on the same item," which eliminates confounding factors from different test sets.
- The negative $\Delta_{cond}$ in anti-presupposition tasks is particularly thought-provoking: correct judgment does not predict successful generation and can even be negatively correlated. This suggests that judgment and generation might utilize different internal representations or reasoning paths.
- Practical warning for the LLM-as-a-judge paradigm: a model's ability to identify a good response does not guarantee it can generate one, and vice versa.

## Limitations & Future Work

- Speaker data for the false presupposition task was taken from existing outputs of original studies rather than being re-generated, potentially introducing time and version discrepancies.
- Language coverage is limited to German (false/anti-presupposition) and English (reasoning).
- Restricted output formats may not fully reflect "natural" pragmatic competence.
- Lack of deep analysis into the mechanisms of asymmetry—is it due to attention patterns, internal representations, or decoding strategies?
- Sample sizes are small for some model-task combinations due to instruction-following failures.

## Related Work & Insights

- **vs. Hu & Levy (2023)**: They found that metalinguistic judgments may decouple from model internal representations; this paper extends this to the pragmatic domain across multiple phenomena.
- **vs. Piot et al. (2025)**: Similar judgment-generation decoupling was found in non-pragmatic domains (content moderation, safety); this paper independently identifies the same pattern in pragmatics, suggesting it is a general property of LLMs.
- **vs. Qiu et al. (2025)**: Evaluated comprehension and production in interactive games, but production was only indirectly measured via listener success; this paper directly evaluates speaker generation quality.

## Rating

- Novelty: ⭐⭐⭐⭐ Systematically compares pragmatic competence across two LLM roles with a fresh and practical perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ 14 models × 3 tasks × item-level analysis provides comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-grounded in psycholinguistic context with rigorous logical argumentation.
- Value: ⭐⭐⭐⭐ Important methodological implications for the LLM-as-a-judge paradigm and linguistic competence evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Enhancing Linguistic Competence of Language Models through Pre-training with Language Learning Tasks](enhancing_linguistic_competence_of_language_models_through_pre-training_with_lan.md)
- [\[ACL 2026\] EngiBench: A Benchmark for Evaluating Large Language Models on Engineering Problem Solving](engibench_a_benchmark_for_evaluating_large_language_models_on_engineering_proble.md)
- [\[ACL 2026\] When Vision-Language Models Judge Without Seeing: Exposing Informativeness Bias](when_vision-language_models_judge_without_seeing_exposing_informativeness_bias.md)
- [\[ACL 2026\] Identifying the Achilles' Heel: An Iterative Method for Dynamically Uncovering Factual Errors in Large Language Models](identifying_the_achilles_heel_an_iterative_method_for_dynamically_uncovering_fac.md)
- [\[ICML 2026\] REAL：把回归感知奖励塞进 RL，让 LLM-as-a-Judge 学会"差一分也是差"](../../ICML2026/llm_evaluation/real_regression-aware_reinforcement_learning_for_llm-as-a-judge.md)

</div>

<!-- RELATED:END -->

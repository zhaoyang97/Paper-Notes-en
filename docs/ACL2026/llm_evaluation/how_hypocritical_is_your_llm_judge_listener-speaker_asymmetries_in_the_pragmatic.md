---
title: >-
  [Paper Note] How Hypocritical Is Your LLM Judge? Listener–Speaker Asymmetries in the Pragmatic Competence of Large Language Models
description: >-
  [ACL 2026][LLM Evaluation][Paper Note] This paper systematically compares 14 LLMs as "pragmatic listeners" (judging pragmatic appropriateness) and "pragmatic speakers" (generating pragmatically appropriate language) across three pragmatic tasks (false presuppositions, anti-presuppositions, and deductive reasoning). The study reveals a widespread listener-sp
tags:
  - ACL 2026
  - LLM Evaluation
date: 2026-05-08
content_hash: aef9481a8f2af9dd
---
# How Hypocritical Is Your LLM Judge? Listener–Speaker Asymmetries in the Pragmatic Competence of Large Language Models

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.15873](https://arxiv.org/abs/2604.15873)  
**Code**: None  
**Area**: Speech Processing / Pragmatic Evaluation  
**Keywords**: Pragmatic competence, listener-speaker asymmetries, LLM-as-a-judge, false presuppositions, deductive reasoning

## TL;DR

This paper systematically compares 14 LLMs as "pragmatic listeners" (judging pragmatic appropriateness) and "pragmatic speakers" (generating pragmatically appropriate language) across three pragmatic tasks (false presuppositions, anti-presuppositions, and deductive reasoning). The study reveals a widespread listener-speaker asymmetry: most models perform significantly better as judges than as generators, and item-level analysis demonstrates that correct judgment does not reliably predict successful generation.

## Background & Motivation

**Background**: Evaluation of LLM linguistic competence typically adopts two paradigms: generative tasks (model as "speaker") and discriminative tasks (model as "listener"/judge). The LLM-as-a-judge paradigm is increasingly prevalent, with models serving as substitutes for human annotators.

**Limitations of Prior Work**: (1) These two evaluative roles are rarely compared directly—researchers implicitly assume that success in one role reflects overall linguistic competence; (2) Psycholinguistic research suggests that human language comprehension and production are related but distinct tasks, where successful comprehension does not guarantee successful production; (3) The reliability of LLM-as-a-judge has not been systematically verified in the pragmatic domain.

**Key Challenge**: If a model can correctly judge the pragmatic appropriateness of a response (listener role), does it imply the model can generate pragmatically appropriate responses itself (speaker role)?

**Goal**: To directly compare the pragmatic judgment (listener) and pragmatic generation (speaker) capabilities of LLMs on the same set of items to test their consistency.

**Key Insight**: Drawing on classic psycholinguistic findings of comprehension-production asymmetries, the authors design parallel listener/speaker prompts using identical underlying test items to achieve rigorous item-level comparison.

**Core Idea**: Pragmatic judgment and pragmatic generation are partially dissociated abilities in current LLMs—"knowing how to judge" does not equal "knowing how to act," suggesting that LLM judges may be "hypocritical."

## Method

### Overall Architecture
This is an evaluation study designed to test whether "judging" equals "doing" through a rigorous item-level comparison of pragmatic judgment and generation. The approach involves selecting three pragmatic tasks of increasing difficulty and designing parallel prompt pairs for the same underlying test items. The speaker prompt requires the model to generate pragmatically appropriate language, while the listener prompt requires the model to judge whether an existing response is pragmatically appropriate. Since the underlying items are identical, accuracy differences between roles exclude confounding factors from variable test sets.

### Key Designs

**1. False Presupposition Task: Extreme Scenarios for Generation**

This task tests whether models can identify and reject false presuppositions embedded in questions. Materials are sourced from two German datasets (False Scenarios and False Claims) containing politically sensitive questions with false presuppositions. In the speaker condition, models answer these questions directly, where correct behavior is to actively reject the presupposition. In the listener condition, models are given the question, the false presupposition, and an existing answer to judge if the response accepted the false presupposition (3-way classification: A/N/U) compared against human labels. Rejection of false presuppositions requires detecting hidden assumptions and active correction, making the generation difficulty naturally higher than judgment.

**2. Anti-presupposition Task: Asymmetry in Word Choice**

This task evaluates adherence to the "Maximize Presupposition!" principle using the German "Fruit Stories" paradigm, where models choose between definite and indefinite articles given a context. The speaker condition involves filling in the correct article/quantifier at a marked position. The listener condition involves judging the pragmatic appropriateness of a given continuation. This task compresses generation to its simplest form (single word selection), yet many models still exhibit a significant listener advantage.

**3. Deductive Reasoning Task: Logical Scaling and Item-level Quantification**

This task examines consistency in logical reasoning between judgment and generation. In the speaker condition, models fill in missing color terms to make a conclusion valid. In the listener condition, models judge if a conclusion logically follows from premises (True/False). A key quantitative metric is introduced: the item-level conditional probability $\Delta_{cond} = P(task\mid l=1) - P(task\mid l=0)$, which measures whether "correct judgment" predicts "successful generation." A positive value indicates coupled abilities, while a negative value suggests that correct judgment might even correlate with failed generation.

The evaluation covers 14 models, with a total of 990+504+180 prompts per model across three tasks.

## Key Experimental Results

### Main Results

**Listener-Speaker Accuracy Comparison (Representative Models)**

| Model | False Pres.-Speaker | False Pres.-Listener | Anti-pres.-Speaker | Anti-pres.-Listener | Reasoning-Speaker | Reasoning-Listener |
|------|------------|------------|----------|----------|---------|---------|
| Mistral-7B | ~2% | ~30% | ~50% | ~86% | ~20% | ~45% |
| LLaMA-8B | ~10% | ~35% | ~55% | ~65% | ~25% | ~73% |
| Qwen-3-14B | ~30% | ~75% | ~35% | ~91% | — | — |
| GPT-4o | ~85% | ~90% | ~80% | ~85% | ~75% | ~80% |
| GPT-5 | — | — | ~100% | ~86% | ~100% | ~100% |

### Item-level Conditional Analysis

| Model | Task | $P(task|l=1)$ | $P(task|l=0)$ | $\Delta_{cond}$ |
|------|------|-------------|-------------|----------------|
| GPT-4o | False Pres.-Scenario | 97.1% | 3.0% | **+94.1** |
| Mistral-7B | Anti-presupposition | 58.8% | 88.9% | **-30.0** |
| GPT-4o | Anti-presupposition | 64.4% | 100.0% | **-35.6** |
| Phi-4-14B | Reasoning | 100.0% | 5.1% | **+94.9** |
| LLaMA-8B | False Pres.-Scenario | 8.8% | 26.0% | **-17.2** |

### Key Findings

- Listener-speaker asymmetries are widespread: most models achieve significantly higher accuracy as judges than as generators.
- Asymmetry is most severe in small open-source models (e.g., Mistral-7B False Presupposition: Speaker 2% vs. Listener 30%).
- Counterintuitive phenomena emerge in anti-presupposition tasks: several models correctly identify violations but choose the violating option when generating themselves (negative $\Delta_{cond}$).
- Larger models (GPT-5) tend toward alignment in certain tasks, though consistency remains imperfect.
- High variance in instruction-following failure rates across models limits the reliability of the LLM-as-a-judge paradigm.

## Highlights & Insights

- The core strength of the experimental design is the "comparison of two roles on the same item," which eliminates confounding variables introduced by different test sets.
- The negative $\Delta_{cond}$ in anti-presupposition tasks is particularly thought-provoking: correct judgment does not predict successful generation and can even be negatively correlated. This suggests that judgment and generation might utilize different internal representations or reasoning paths.
- Practical warning for the LLM-as-a-judge paradigm: a model's ability to identify a good response does not guarantee its ability to generate one, and vice-versa.

## Limitations & Future Work

- Speaker data for the false presupposition task was taken from existing outputs of original studies rather than being re-generated, potentially introducing temporal and versioning discrepancies.
- Language coverage is limited to German (false/anti-presuppositions) and English (reasoning).
- Constrained output formats may not fully reflect "natural" pragmatic competence.
- Lack of deep analysis into the mechanisms of asymmetry—whether due to attention patterns, internal representations, or decoding strategies.
- Small sample sizes for certain model-task combinations due to instruction-following failures.

## Related Work & Insights

- **vs. Hu & Levy (2023)**: They found metalinguistic judgments may dissociate from internal representations; this paper extends this to the pragmatic domain across multiple phenomena.
- **vs. Piot et al. (2025)**: Identified similar judgment-generation dissociation in non-pragmatic domains (content moderation, safety); this paper finds the same pattern in pragmatics, suggesting it is a general property of LLMs.
- **vs. Qiu et al. (2025)**: Evaluated comprehension and production in interactive games, but measured production only indirectly; this paper directly assesses speaker generation quality.

## Rating

- Novelty: ⭐⭐⭐⭐ Systematic comparison of LLM pragmatic roles provides a fresh and practical perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage across 14 models and 3 tasks with item-level analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Strong psycholinguistic grounding and rigorous logical argumentation.
- Value: ⭐⭐⭐⭐ Significant methodological implications for the LLM-as-a-judge paradigm and linguistic competence evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Enhancing Linguistic Competence of Language Models through Pre-training with Language Learning Tasks](enhancing_linguistic_competence_of_language_models_through_pre-training_with_lan.md)
- [\[ICLR 2026\] Do LLM Agents Know How to Ground, Recover, and Assess? Evaluating Epistemic Competence in Information-Seeking Agents](../../ICLR2026/llm_evaluation/do_llm_agents_know_how_to_ground_recover_and_assess_evaluating_epistemic_compete.md)
- [\[ACL 2026\] EngiBench: A Benchmark for Evaluating Large Language Models on Engineering Problem Solving](engibench_a_benchmark_for_evaluating_large_language_models_on_engineering_proble.md)
- [\[ACL 2026\] When Vision-Language Models Judge Without Seeing: Exposing Informativeness Bias](when_vision-language_models_judge_without_seeing_exposing_informativeness_bias.md)
- [\[ACL 2026\] Identifying the Achilles' Heel: An Iterative Method for Dynamically Uncovering Factual Errors in Large Language Models](identifying_the_achilles_heel_an_iterative_method_for_dynamically_uncovering_fac.md)

</div>

<!-- RELATED:END -->

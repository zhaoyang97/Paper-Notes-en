---
title: >-
  [Paper Note] How Hypocritical Is Your LLM Judge? Listener–Speaker Asymmetries in the Pragmatic Competence of Large Language Models
description: >-
  [ACL 2026][LLM Evaluation][Pragmatic competence] This paper systematically compares the performance of 14 LLMs as "pragmatic listeners" (judging pragmatic appropriateness) and "pragmatic speakers" (generating pragmatical…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Pragmatic competence"
  - "Listener–Speaker asymmetry"
  - "LLM-as-a-judge"
  - "False presupposition"
  - "Deductive reasoning"
date: 2026-05-08
content_hash: 494fe7514285578c
---

# How Hypocritical Is Your LLM Judge? Listener–Speaker Asymmetries in the Pragmatic Competence of Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.15873](https://arxiv.org/abs/2604.15873)  
**Code**: None  
**Area**: Speech Processing / Pragmatic Evaluation  
**Keywords**: Pragmatic competence, Listener–Speaker asymmetry, LLM-as-a-judge, False presupposition, Deductive reasoning

## TL;DR

This paper systematically compares the performance of 14 LLMs as "pragmatic listeners" (judging pragmatic appropriateness) and "pragmatic speakers" (generating pragmatically appropriate language) through three pragmatic tasks (false presupposition, antipresupposition, and deductive reasoning). It identifies a widespread listener–speaker asymmetry: most models perform significantly better as judges than as generators, and item-level analysis indicates that correct judgment does not reliably predict successful generation.

## Background & Motivation

**Background**: Evaluation of LLM linguistic competence typically follows two paradigms: generative tasks (model as "speaker") and discriminative tasks (model as "listener"/judge). The LLM-as-a-judge paradigm is increasingly popular, where models serve as proxies for human annotators.

**Limitations of Prior Work**: (1) These two evaluative roles are almost never directly compared—researchers implicitly assume that success in one role reflects overall linguistic competence; (2) Psycholinguistic research indicates that language comprehension and production are related but distinct tasks in humans, and successful comprehension does not guarantee successful production; (3) The reliability of LLM-as-a-judge has not been systematically validated in the pragmatic domain.

**Key Challenge**: If a model can correctly judge the pragmatic appropriateness of a response (listener role), does it imply that it can also generate pragmatically appropriate responses itself (speaker role)?

**Goal**: To directly compare the pragmatic judgment (listener) and pragmatic generation (speaker) capabilities of LLMs on the same set of items, testing whether the two are consistent.

**Key Insight**: Drawing on classic findings of comprehension–production asymmetry in psycholinguistics, this study designs parallel listener/speaker prompts using identical underlying test items to achieve rigorous item-level comparison.

**Core Idea**: Pragmatic judgment and pragmatic generation are partially dissociated capabilities in current LLMs; "being able to judge" does not equate to "being able to do." LLM judges may be "hypocritical."

## Method

### Overall Architecture

Three pragmatic tasks are selected. For each task, parallel speaker prompts (requiring generation) and listener prompts (requiring judgment) are designed for the same test items. 14 LLMs (including open-source and closed-source) are evaluated. The accuracy for both roles is calculated, followed by item-level conditional analysis.

### Key Designs

1.  **False Presuppositions**:
    - **Function**: Tests whether the model can identify and reject false presuppositions within questions.
    - **Mechanism**: Utilizes two German datasets (False Scenarios and False Claims) containing politically sensitive questions with false presuppositions. Speaker condition: The model directly answers questions with false presuppositions; the correct behavior is to reject the presupposition. Listener condition: The model is provided with the question, the false presupposition, and an existing answer, then judges whether the answer accepted the false presupposition (3-way classification: A/N/U), compared against human annotations.
    - **Design Motivation**: Rejecting false presuppositions requires detecting implicit assumptions and actively correcting them—generation is far more difficult than judgment, making it an ideal scenario for testing listener–speaker asymmetry.

2.  **Antipresuppositions**:
    - **Function**: Tests whether the model follows the "Maximize Presupposition!" principle.
    - **Mechanism**: Utilizes the German "Fruit Stories" paradigm—given a context, the model must choose between a definite or indefinite article. Speaker condition: Fill in the correct article/quantifier at the marked position. Listener condition: Judge whether a given continuation is pragmatically appropriate. Even in highly constrained generation settings (choosing a single word), many models exhibit a significant listener advantage.
    - **Design Motivation**: This is the "simplest" generation task (word selection); if asymmetry exists here, it indicates a fundamental issue.

3.  **Deductive Reasoning**:
    - **Function**: Tests the consistency of logical reasoning in evaluation versus generation.
    - **Mechanism**: Based on classic logical reasoning tasks, providing premises and conclusions. Speaker condition: Fill in the missing color word that makes the conclusion valid. Listener condition: Judge whether the given conclusion logically follows from the premises (True/False). Item-level analysis—conditional probability $\Delta_{cond} = P(task|l=1) - P(task|l=0)$—measures whether correct judgment predicts successful generation.
    - **Design Motivation**: Deductive reasoning involves both pragmatic and logical competence, allowing for a test of whether asymmetry crosses different cognitive dimensions.

### Loss & Training

This is an evaluative study with no training involved. 14 models were evaluated (LLaMA-3-8B, Qwen-3-8B/14B, Phi-4-14B, OLMo-2-7B/13B/32B, Mistral-7B, Mixtral-8x7B, M-Prometheus-14B, GPT-4o, GPT-4.1, GPT-5, Claude Sonnet 4.5). A total of 990 + 504 + 180 prompts per model were used.

## Key Experimental Results

### Main Results

**Comparison of Listener-Speaker Accuracy (Representative Models)**

| Model | False Presupposition-Speaker | False Presupposition-Listener | Antipresupposition-Speaker | Antipresupposition-Listener | Reasoning-Speaker | Reasoning-Listener |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Mistral-7B | ~2% | ~30% | ~50% | ~86% | ~20% | ~45% |
| LLaMA-8B | ~10% | ~35% | ~55% | ~65% | ~25% | ~73% |
| Qwen-3-14B | ~30% | ~75% | ~35% | ~91% | — | — |
| GPT-4o | ~85% | ~90% | ~80% | ~85% | ~75% | ~80% |
| GPT-5 | — | — | ~100% | ~86% | ~100% | ~100% |

### Item-level Conditional Analysis

| Model | Task | $P(task|l=1)$ | $P(task|l=0)$ | $\Delta_{cond}$ |
| :--- | :--- | :--- | :--- | :--- |
| GPT-4o | False Presupposition-Scenario | 97.1% | 3.0% | **+94.1** |
| Mistral-7B | Antipresupposition | 58.8% | 88.9% | **-30.0** |
| GPT-4o | Antipresupposition | 64.4% | 100.0% | **-35.6** |
| Phi-4-14B | Reasoning | 100.0% | 5.1% | **+94.9** |
| LLaMA-8B | False Presupposition-Scenario | 8.8% | 26.0% | **-17.2** |

### Key Findings

- Listener–speaker asymmetry is widespread: accuracy for most models as judges is significantly higher than as generators.
- Asymmetry is most severe in small open-source models (e.g., Mistral-7B False Presupposition: Speaker 2% vs. Listener 30%).
- A counterintuitive phenomenon appears in the antipresupposition task: multiple models correctly identify violations but choose the violating option when generating themselves (negative $\Delta_{cond}$).
- Large models (GPT-5) tend toward alignment between the two roles in certain tasks but are still not perfectly consistent.
- Instruction-following failure rates vary greatly across models, limiting the reliability of LLM-as-a-judge.

## Highlights & Insights

- The core strength of the experimental design is the "comparison of two roles on the same item," which eliminates confounding factors from different test sets.
- The negative $\Delta_{cond}$ in the antipresupposition task is particularly thought-provoking: correct judgment not only fails to predict successful generation but may even be negatively correlated. This suggests that judgment and generation might utilize different internal representations or reasoning paths.
- Practical warning for the LLM-as-a-judge paradigm: A model's ability to recognize a good response does not mean it can generate one, and vice versa.

## Limitations & Future Work

- Speaker data for the false presupposition task was taken from existing outputs of original studies rather than being re-generated, potentially introducing temporal and versioning discrepancies.
- Language coverage is limited, using only German (false presupposition, antipresupposition) and English (reasoning).
- Constrained output formats might not fully reflect "natural" pragmatic competence.
- The mechanisms of asymmetry—whether due to attention patterns, internal representations, or decoding strategies—were not analyzed in depth.
- Sample sizes for certain model-task combinations were small due to instruction-following failures.

## Related Work & Insights

- **vs Hu & Levy (2023)**: They found that metalinguistic judgments can dissociate from internal model representations; this study extends this finding to the pragmatic domain across multiple phenomena.
- **vs Piot et al. (2025)**: Similar judgment–generation dissociations were found in non-pragmatic domains (content moderation, safety); this study independently discovers the same pattern in pragmatics, suggesting this is a general property of LLMs.
- **vs Qiu et al. (2025)**: Evaluated comprehension and production in interactive games, but production competence was only indirectly measured via listener success rate; this study directly evaluates speaker generation quality.

## Rating

- Novelty: ⭐⭐⭐⭐ Systematically compares two LLM roles in pragmatic competence; the perspective is novel and practically significant.
- Experimental Thoroughness: ⭐⭐⭐⭐ 14 models × 3 tasks × item-level analysis, comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐⭐ Psycholinguistic background is well-explained with rigorous logical argumentation.
- Value: ⭐⭐⭐⭐ Significant methodological implications for the LLM-as-a-judge paradigm and linguistic competence evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Enhancing Linguistic Competence of Language Models through Pre-training with Language Learning Tasks](enhancing_linguistic_competence_of_language_models_through_pre-training_with_lan.md)
- [\[ACL 2026\] Contrastive Decoding Mitigates Score Range Bias in LLM-as-a-Judge](contrastive_decoding_mitigates_score_range_bias_in_llm-as-a-judge.md)
- [\[ACL 2026\] When Vision-Language Models Judge Without Seeing: Exposing Informativeness Bias](when_vision-language_models_judge_without_seeing_exposing_informativeness_bias.md)
- [\[ACL 2026\] EngiBench: A Benchmark for Evaluating Large Language Models on Engineering Problem Solving](engibench_a_benchmark_for_evaluating_large_language_models_on_engineering_proble.md)
- [\[ACL 2026\] Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models](challenging_the_boundaries_of_reasoning_an_olympiad-level_math_benchmark_for_lar.md)

</div>

<!-- RELATED:END -->

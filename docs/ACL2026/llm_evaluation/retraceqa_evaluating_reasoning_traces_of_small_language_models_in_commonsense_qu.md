---
title: >-
  [Paper Note] ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering
description: >-
  [ACL 2026][LLM Evaluation][LLM-as-Judge] This paper proposes ReTraceQA, the first reasoning process evaluation benchmark for commonsense reasoning tasks. It includes 2421 expert-annotated step-level error localizations and classifications, revealing that 14–24% of SLMs provide correct answers despite flawed reasoning. When reasoning-aware evaluation replaces
tags:
  - ACL 2026
  - LLM Evaluation
  - LLM-as-Judge
date: 2026-05-08
content_hash: ff211ae7372fe103
---
# ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering

**Conference**: ACL 2026  
**arXiv**: [2510.09351](https://arxiv.org/abs/2510.09351)  
**Code**: [https://github.com/SapienzaNLP/ReTraceQA](https://github.com/SapienzaNLP/ReTraceQA)  
**Area**: LLM Evaluation/Commonsense Reasoning  
**Keywords**: Reasoning Process Evaluation, Small Language Models, Commonsense Reasoning, Process Reward Models, LLM-as-Judge

## TL;DR

This paper proposes ReTraceQA, the first reasoning process evaluation benchmark for commonsense reasoning tasks. It includes 2421 expert-annotated step-level error localizations and classifications, revealing that 14–24% of SLMs provide correct answers despite flawed reasoning. When reasoning-aware evaluation replaces answer-only evaluation, SLM performance drops by up to 25 percentage points.

## Background & Motivation

**Background**: Small Language Models (SLMs, $\leq$ 10B parameters) are performing increasingly well on various commonsense reasoning benchmarks. However, current evaluation practices rely almost exclusively on the correctness of the final answer—as long as the model prediction matches the ground truth, it is considered correct, disregarding whether the reasoning process is sound.

**Limitations of Prior Work**: (1) Models can reach the correct answer through invalid reasoning paths (e.g., shortcut reasoning, accidental correctness from false premises); answer-only evaluation artificially inflates performance metrics. (2) Existing reasoning process evaluation benchmarks (ProcessBench, MR-Ben, etc.) focus on mathematics and science; the process evaluation for commonsense reasoning is entirely blank. (3) Process Reward Models (PRMs) and LLM judges are primarily used for Best-of-N selection to optimize performance rather than for scrutinizing whether correct answers are derived from valid reasoning.

**Key Challenge**: There is a significant gap between the high scores of SLMs on leaderboards and their actual reasoning capabilities—correct answers do not equate to correct reasoning, yet current evaluation systems cannot distinguish between the two.

**Goal**: To build the first step-level reasoning process evaluation benchmark for commonsense reasoning, quantify the extent to which answer-only evaluation overestimates SLM capabilities, and evaluate the performance of LLMs as reasoning judges and PRMs in the commonsense domain.

**Key Insight**: Focus on "process errors"—instances where the answer is correct but the reasoning process is flawed. Establish a gold standard through expert annotation and use it to measure the reliability of automated evaluation methods.

**Core Idea**: Generate Chain-of-Thought (CoT) reasoning chains using 7 SLMs across 4 commonsense reasoning datasets. Three PhD-level experts annotated step-level error locations and categories (Misinterpretation/Hallucination/Reasoning), constructing a benchmark of 2421 instances to evaluate LLM judges and PRMs in both reference-free and reference-based settings.

## Method

### Overall Architecture

ReTraceQA transforms "commonsense reasoning process evaluation" into a quantifiable first-error localization problem. First, 7 instruction-tuned SLMs (Llama 3.2/3.1, Qwen 2.5, Phi-4-mini) generate CoT reasoning chains zero-shot on four datasets: CSQA, OBQA, QASC, and StrategyQA. These chains are segmented into steps, and balanced sampling is applied to ensure a mix of correct/incorrect chains, model sources, and unique questions. Finally, three PhD-level experts annotated the first error step and its category for each chain, resulting in 2421 gold instances. Once the benchmark was established, LLM judges and PRMs were evaluated chain-by-chain in reference-free and reference-based settings to see if they could replicate expert judgments.

### Key Designs

**1. Three-level Hierarchical Error Taxonomy: Categorizing Three Distinct Failures by Cognitive Level**

Answer-only evaluation is distorted because it lumps "misunderstanding the question," "misremembering facts," and "logical leaps" into a single binary result. This paper defines three mutually exclusive error categories with a priority ranking from "grounding to reasoning": Misinterpretation belongs to the grounding layer, involving misunderstandings of the question, options, or task requirements (e.g., citing non-existent options); Hallucination belongs to the content layer, introducing empirically false or unverifiable world knowledge, used when the logical structure might hold but the factual "bricks" are wrong (e.g., "wolves do not survive in the Arctic"); Reasoning belongs to the reasoning layer, involving invalid logical leaps between correct premises (e.g., knowing "salt lowers the freezing point" but concluding "this makes ice form more easily"). This hierarchy points diagnostic information directly toward specific improvements—fixing facts, logic, or comprehension.

**2. First Error Localization Task Definition: Tracking only the first error to avoid cascade ambiguity**

Given a question $q$ and a reasoning chain $S = [s_0, s_1, \ldots, s_n]$, the task requires predicting an index $i \in \{-1, 0, \ldots, n\}$, where $i = -1$ indicates a fully correct chain and $i \geq 0$ indicates the first error occurs at step $s_i$. Only the first error is annotated because once a step is wrong, subsequent steps are built on false premises, making their individual "correctness" ambiguous and impossible to attribute cleanly. This definition aligns with ProcessBench, facilitating direct cross-domain comparison (math vs. commonsense) while converging evaluation into a single calculable localization accuracy.

**3. Dual-Axis Evaluation Framework (Reference-free / Reference-based × Judge / PRM): Separating Deployment and Evaluation Scenarios**

The reliability of the same evaluator may differ significantly between scenarios, so they are cross-tested along two axes. The reference-free setting provides only the reasoning chain without the ground truth answer, corresponding to real-world deployment like training feedback or Best-of-N selection. The reference-based setting additionally provides the correct answer, corresponding to offline evaluation. Both settings are measured using three metrics: correct (accuracy in identifying fully correct chains), error (accuracy in localizing the first error step), and their harmonic mean F1. This reveals the relative strengths of LLM judges and PRMs while distinguishing the ability to judge overall correctness from the more difficult task of precise error localization.

### Loss & Training

As a benchmark paper, no new models were trained. LLM judges reused slightly adapted ProcessBench prompt templates. PRMs utilized thresholded judgments from sigmoid outputs or F1-maximized thresholds. Open-source models used greedy decoding; o1-mini and DeepSeek-R1 used a temperature of 1.0 due to API constraints.

## Key Experimental Results

### Main Results

| Model | CSQA F1 | OBQA F1 | QASC F1 | StrategyQA F1 | Avg F1 |
|------|---------|---------|---------|-------------|--------|
| **Ref-based LLM Judge** | | | | | |
| o1-mini | 65.7 | 79.2 | 74.2 | 78.3 | 74.4 |
| GPT-4o | 67.9 | 76.6 | 66.2 | 65.3 | 69.0 |
| Qwen2.5-72B | 64.7 | 69.9 | 69.7 | 67.3 | 67.9 |
| Gemini-2.0-Flash | 65.2 | 74.5 | 68.4 | 62.4 | 67.6 |
| DeepSeek-R1 | 57.4 | 56.4 | 56.7 | 47.2 | 54.4 |
| **Ref-free PRM** | | | | | |
| Qwen2.5-Math-PRM-7B | 33.8 | 42.8 | 48.6 | 37.4 | 40.7 |
| Math-Shepherd-PRM-7B | 8.0 | 11.5 | 17.9 | 28.4 | 16.5 |

| SLM Model | Answer-only Acc | Reasoning-aware Acc | Performance Inflation $\Delta$ |
|---------|------------|-------------|---------|
| Qwen2.5-7B | 81.0 | 67.5 | 13.5 |
| Llama-3.1-8B | 76.3 | 63.1 | 13.2 |
| Qwen2.5-3B | 70.4 | 48.5 | 22.0 |
| Llama-3.2-1B | 49.0 | 23.4 | 25.6 |
| Average | 68.3 | 49.7 | 18.6 |

### Ablation Study

| Dataset | Process Error Rate (Correct Answer, Incorrect Reasoning) |
|--------|---------------------------|
| CSQA | 16.3% |
| OBQA | 14.7% |
| QASC | 16.6% |
| StrategyQA | 24.0% |
| Average | 17.9% |

### Key Findings

- **17.9% of correct answers stem from flawed reasoning**: On average, one out of every 5–6 "correct" answers has an incorrect reasoning process, reaching up to 24% on StrategyQA. This indicates that answer-only evaluation severely overestimates SLM capabilities.
- **Reasoning-aware evaluation leads to significant performance drops**: Using o1-mini as a reasoning judge, average SLM accuracy dropped from 68.3% to 49.7% (a 18.6pp decrease); the worst-performing Llama-3.2-1B dropped from 49.0% to 23.4% (a 25.6pp decrease).
- **Hallucination is the primary failure mode for SLM reasoning**: Hallucination errors account for 41.9%–62.5% of all errors, followed by reasoning errors (27.9%–35.4%), while misinterpretation errors are the least frequent (9.6%–24.1%). SLMs understand questions but frequently manufacture false "facts."
- **Math PRMs do not transfer to commonsense reasoning**: The strongest math PRM achieved an average F1 of only 40.7%, compared to 74.4% for the best LLM judge, indicating extremely limited generalization for PRMs.
- **LLM judges are better at overall judgment than error localization**: The "correct" score for detecting overall chain validity is much higher than the "error" score for localizing specific error steps, showing that precise reasoning error localization remains an open challenge.
- **Errors frequently occur in intermediate steps (Steps 3-4)**: Early context establishment is usually successful, with errors emerging during the intermediate reasoning stages. o1-mini’s prediction distribution aligns closely with human annotations but tends to over-attribute errors to later steps.

## Highlights & Insights

- **First to quantify the severity of "correct answer $\neq$ correct reasoning" in commonsense tasks**: A 17.9% process error rate and up to 25pp performance inflation serve as a wake-up call to the community—leaderboard scores are nearly 19 percentage points higher than actual capabilities.
- **Practical value of hierarchical error taxonomy**: The error distribution pattern (Hallucination > Reasoning > Misinterpretation) clearly reveals that the core weakness of SLMs lies in factual grounding rather than logic or comprehension, providing a clear guide for improvement.
- **Warning on cross-domain transfer**: The failure of math PRMs in commonsense reasoning (average F1 of only 21.1% in reference-free settings) proves that "mathematical reasoning $\neq$ general reasoning," calling for the development of domain-specific process reward models.
- **Extremely high annotation quality**: Annotated by three PhD-level experts with a Fleiss's Kappa of 0.84 ("almost perfect agreement"), providing a reliable gold standard for the field.

## Limitations & Future Work

- Only SLMs with $\leq$ 10B parameters were evaluated; the reasoning trace quality of larger models was not addressed.
- "Correctness" in commonsense reasoning can be subjective—different annotators may disagree on the "correctness" of certain world knowledge.
- Only zero-shot CoT was used to generate reasoning chains; reasoning quality under few-shot or other prompting strategies remains unexplored.
- Future work is needed to build specialized PRMs for commonsense reasoning rather than relying on transfers from the mathematical domain.
- The work could be extended to more reasoning domains such as legal, ethical, or social reasoning.

## Related Work & Insights

- **vs. ProcessBench**: ProcessBench only covers error localization in mathematical reasoning; ReTraceQA is the first to extend process evaluation to the commonsense domain.
- **vs. MR-Ben/MR-GSM8K**: These benchmarks provide error localization, explanation, and correction but are similarly limited to math/science; ReTraceQA demonstrates that commonsense reasoning requires a different evaluation framework.
- **vs. MMErroR**: MMErroR evaluates the diagnostic ability of VLMs on given erroneous reasoning chains; ReTraceQA evaluates the process-level quality of reasoning chains generated by the SLMs themselves, making the two complementary.
- **vs. PRM (Math-Shepherd/Qwen2.5-Math-PRM)**: ReTraceQA experiments prove that math PRMs cannot transfer to commonsense reasoning, highlighting the necessity of domain-specific evaluation.

## Rating

- Novelty: ⭐⭐⭐⭐ First step-level reasoning process benchmark for commonsense reasoning; clear task definition.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive analysis including 5 PRMs, 8 LLM judges, dual settings (ref-free/ref-based), and downstream evaluation of 7 SLMs.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rigorous task definition, and detailed statistical analysis.
- Value: ⭐⭐⭐⭐ Reveals severe flaws in answer-only evaluation and provides practical benchmarks and tools for reasoning-aware evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Evaluating Legal Reasoning Traces with Legal Issue Tree Rubrics](evaluating_legal_reasoning_traces_with_legal_issue_tree_rubrics.md)
- [\[ACL 2026\] ReCoQA: A Benchmark for Tool-Augmented and Multi-Step Reasoning in Real Estate Question and Answering](recoqa_a_benchmark_for_tool-augmented_and_multi-step_reasoning_in_real_estate_qu.md)
- [\[ACL 2026\] Evaluating Reasoning Models for Queries with Presuppositions](evaluating_reasoning_models_for_queries_with_presuppositions.md)
- [\[ACL 2026\] Language Models Don't Know What You Want: Evaluating Personalization in Deep Research Needs Real Users](language_models_dont_know_what_you_want_evaluating_personalization_in_deep_resea.md)
- [\[ACL 2026\] Question Difficulty Estimation for Large Language Models via Answer Plausibility Scoring](question_difficulty_estimation_for_large_language_models_via_answer_plausibility.md)

</div>

<!-- RELATED:END -->

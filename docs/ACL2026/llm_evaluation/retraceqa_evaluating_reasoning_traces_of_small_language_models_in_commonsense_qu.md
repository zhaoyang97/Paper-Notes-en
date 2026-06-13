---
title: >-
  [Paper Note] ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering
description: >-
  [ACL 2026][LLM Evaluation][Reasoning process evaluation] Ours proposes ReTraceQA, the first reasoning process evaluation benchmark for commonsense reasoning tasks. It contains 2…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Reasoning process evaluation"
  - "Small Language Models"
  - "Commonsense reasoning"
  - "Process Reward Models"
  - "LLM-as-Judge"
date: 2026-05-08
content_hash: ff55f85691e285a1
---

# ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering

**Conference**: ACL 2026  
**arXiv**: [2510.09351](https://arxiv.org/abs/2510.09351)  
**Code**: [https://github.com/SapienzaNLP/ReTraceQA](https://github.com/SapienzaNLP/ReTraceQA)  
**Area**: LLM Evaluation/Commonsense Reasoning  
**Keywords**: Reasoning process evaluation, Small Language Models, Commonsense reasoning, Process Reward Models, LLM-as-Judge

## TL;DR

Ours proposes ReTraceQA, the first reasoning process evaluation benchmark for commonsense reasoning tasks. It contains 2,421 step-level error localization and error classification labels annotated by experts. It reveals that 14-24% of SLMs provide correct answers despite incorrect reasoning processes. When replacing answer-only evaluation with reasoning-aware evaluation, SLM performance drops by up to 25 percentage points.

## Background & Motivation

**Background**: Small Language Models (SLMs, ≤10B parameters) are performing increasingly well on various commonsense reasoning benchmarks. However, current evaluation practices rely almost entirely on the correctness of the final answer—as long as the model prediction matches the gold answer, it is considered correct, completely ignoring whether the reasoning process is rational.

**Limitations of Prior Work**: (1) Models can reach correct answers through invalid reasoning paths (e.g., shortcut reasoning, accidental correctness from false premises), and answer-only evaluation artificially inflates performance metrics; (2) Existing reasoning process evaluation benchmarks (ProcessBench, MR-Ben, etc.) focus on math/science domains, leaving a complete vacuum in process evaluation for commonsense reasoning; (3) Process Reward Models (PRM) and LLM judges are primarily used for Best-of-N selection to optimize performance rather than for auditing whether correct answers were obtained via valid reasoning paths.

**Key Challenge**: There is a significant gap between the high scores of SLMs on leaderboards and their true reasoning capabilities—correct answers do not equate to correct reasoning, yet current evaluation systems cannot distinguish between the two.

**Goal**: To construct the first step-level reasoning process evaluation benchmark for commonsense reasoning, quantify the degree of overestimation in SLM capabilities by answer-only evaluation, and evaluate the performance of LLMs as reasoning judges and PRMs in the commonsense reasoning domain.

**Key Insight**: Focus on "process errors"—instances where the answer is correct but the reasoning process is flawed. Establish a gold standard through expert annotation and use it to measure the reliability of automated evaluation methods.

**Core Idea**: Generate CoT reasoning chains using 7 SLMs on 4 commonsense reasoning datasets. Three PhD-level experts annotate step-level error locations and error categories (Misinterpretation/Hallucination/Reasoning) to build a benchmark of 2,421 instances. LLM judges and PRMs are evaluated under both reference-free and reference-based settings.

## Method

### Overall Architecture

The construction process of ReTraceQA includes: (1) Selecting questions from four commonsense reasoning datasets: CSQA, OBQA, QASC, and StrategyQA; (2) Generating reasoning chains via zero-shot CoT using 7 SLMs (instruct-tuned versions of Llama 3.2/3.1, Qwen 2.5, and Phi-4-mini); (3) Performing step segmentation on reasoning chains; (4) Applying balanced sampling to ensure a balance of correct/incorrect chains, models, and unique questions; (5) Three experts annotating the position of the first error step and the error category for each chain.

### Key Designs

1. **Three-level Hierarchical Error Taxonomy**:
    - Function: Mutually exclusive classification of reasoning errors based on cognitive levels.
    - Mechanism: Defines three categories from low to high levels—Misinterpretation (Grounding level: misunderstanding the question, option meanings, or task requirements, including citing non-existent options or providing multiple answers); Hallucination (Content level: introducing empirically false or unverifiable world knowledge, used only when the logical structure might be correct but factual "building blocks" are wrong, e.g., "Wolves do not survive in Arctic regions"); Reasoning (Reasoning level: making invalid logical jumps between correct premises, e.g., correctly stating "salt lowers the freezing point" but incorrectly inferring "this makes ice form more easily"). Priority follows "Grounding to Reasoning" during classification.
    - Design Motivation: Distinguishing between three fundamentally different failure modes—"not understanding the question," "not knowing facts," and "unable to reason logically"—to provide targeted diagnostic information for improving SLMs.

2. **First-Error Localization Task Definition**:
    - Function: Formalizing reasoning process evaluation into a quantifiable task.
    - Mechanism: Given a question $q$ and a reasoning chain $S = [s_0, s_1, \ldots, s_n]$, predict an index $i \in \{-1, 0, \ldots, n\}$, where $i = -1$ indicates all steps are correct, and $i \geq 0$ indicates the first error occurs at step $s_i$. Only the first error is considered because subsequent steps build on a false premise, making their correctness ambiguous.
    - Design Motivation: Maintaining consistency with ProcessBench task definitions facilitates cross-domain comparison, and first-error localization avoids ambiguity in cascading error attribution.

3. **Dual-axis Evaluation Framework (Reference-free + Reference-based × Judge + PRM)**:
    - Function: Comprehensively evaluate the performance of automated reasoning evaluation methods in commonsense reasoning.
    - Mechanism: The reference-free setting (providing only the reasoning chain, no correct answer) tests the reliability of LLM judges and PRMs for training feedback/Best-of-N selection; the reference-based setting (providing both the correct answer and reasoning chain) tests their ability as evaluation tools. Both settings use correct (accuracy in identifying fully correct chains), error (accuracy in locating the first error step), and F1 (harmonic mean of both) for evaluation.
    - Design Motivation: The reference-free setting reflects actual deployment scenarios (gold answers unavailable during training), while the reference-based setting reflects evaluation scenarios. Combining both reveals strengths and weaknesses of different models under different conditions.

### Loss & Training

Ours is a benchmark paper and does not involve model training. LLM judges use slightly adapted ProcessBench prompt templates. PRMs use thresholding of sigmoid outputs or threshold selection maximized by F1. All open-source models use greedy decoding; o1-mini and DeepSeek-R1 use a temperature of 1.0 due to API constraints.

## Key Experimental Results

### Main Results

| Model | CSQA F1 | OBQA F1 | QASC F1 | StrategyQA F1 | Average F1 |
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

| SLM Model | Answer-only Acc. | Reasoning-aware Acc. | Performance Inflation Δ |
|---------|------------|-------------|---------|
| Qwen2.5-7B | 81.0 | 67.5 | 13.5 |
| Llama-3.1-8B | 76.3 | 63.1 | 13.2 |
| Qwen2.5-3B | 70.4 | 48.5 | 22.0 |
| Llama-3.2-1B | 49.0 | 23.4 | 25.6 |
| Average | 68.3 | 49.7 | 18.6 |

### Ablation Study

| Dataset | Process Error Rate (Answer Correct but Reasoning Incorrect) |
|--------|---------------------------|
| CSQA | 16.3% |
| OBQA | 14.7% |
| QASC | 16.6% |
| StrategyQA | 24.0% |
| Average | 17.9% |

### Key Findings

- **17.9% of correct answers stem from incorrect reasoning**: On average, one in every 5-6 "correct" answers has a flawed reasoning process, reaching as high as 24% on StrategyQA, indicating that answer-only evaluation severely overestimates SLM capabilities.
- **Reasoning-aware evaluation leads to a significant performance drop**: After using o1-mini as a reasoning judge, the average accuracy of SLMs dropped from 68.3% to 49.7% (a 18.6pp decrease), with the worst-performing Llama-3.2-1B dropping from 49.0% to 23.4% (a 25.6pp decrease).
- **Hallucination is the primary failure mode in SLM reasoning**: Hallucination errors account for 41.9%-62.5% of all errors, followed by reasoning errors (27.9%-35.4%), while misinterpretation errors are the least frequent (9.6%-24.1%). SLMs can understand questions but often fabricate false "facts."
- **Math PRMs fail to transfer to commonsense reasoning**: The strongest math PRM achieved an average F1 of only 40.7%, whereas the strongest LLM judge reached 74.4%, demonstrating extremely limited generalization of PRMs.
- **LLM judges excel at holistic judgment but struggle with error localization**: Correctness scores for detecting overall chain validity are much higher than error scores for locating specific error steps, indicating that precise localization of reasoning errors remains an open challenge.
- **Errors frequently occur in intermediate steps (Steps 3-4)**: Early context establishment is usually successful; errors appear during intermediate reasoning stages. o1-mini's prediction distribution aligns closely with human annotations but shows a tendency to over-attribute errors to later steps.

## Highlights & Insights

- **First quantification of the severity of "Answer correct ≠ Reasoning correct" in commonsense reasoning**: A 17.9% process error rate and performance inflation of up to 25pp serve as a warning to the community—leaderboard scores are nearly 19 percentage points higher than actual capabilities.
- **Practical value of hierarchical error taxonomy**: The error distribution pattern of Hallucination > Reasoning > Misinterpretation clearly reveals that the core weakness of SLMs lies in factual grounding rather than logical reasoning or question understanding, providing clear guidance for improvement.
- **Warning on cross-domain transfer**: The failure of math PRMs in commonsense reasoning (average F1 only 21.1% in reference-free settings) proves that "mathematical reasoning ≠ general reasoning," calling for the construction of domain-specific process reward models.
- **Extremely high annotation quality**: Annotated by three PhD-level experts with a Fleiss's Kappa of 0.84 ("almost perfect agreement"), providing a reliable gold standard for the field.

## Limitations & Future Work

- Only SLMs with ≤10B parameters were evaluated; the reasoning process quality of larger models remains unaddressed.
- The "correctness" of commonsense reasoning itself is subjective—different annotators may disagree on the "correctness" of certain world knowledge.
- Only zero-shot CoT was used to generate reasoning chains; reasoning quality under few-shot or other prompting strategies was not explored.
- Future work should build dedicated PRMs for commonsense reasoning rather than relying on transfers from the mathematical domain.
- Expansion to more reasoning domains (legal, ethical, social reasoning, etc.).

## Related Work & Insights

- **vs ProcessBench**: ProcessBench only covers error localization in mathematical reasoning; ReTraceQA extends process evaluation to the commonsense reasoning domain for the first time.
- **vs MR-Ben/MR-GSM8K**: These benchmarks provide error localization, explanation, and correction but are similarly limited to math/science; ReTraceQA demonstrates that commonsense reasoning requires a different evaluation framework.
- **vs MMErroR**: MMErroR evaluates the diagnostic ability of VLMs on given incorrect reasoning chains; ReTraceQA evaluates the process-level reliability of reasoning chains generated by SLMs themselves. The two are complementary.
- **vs PRM (Math-Shepherd/Qwen2.5-Math-PRM)**: ReTraceQA experiments prove that math PRMs cannot transfer to commonsense reasoning, highlighting the necessity of domain-specific evaluation.

## Rating

- Novelty: ⭐⭐⭐⭐ First step-level reasoning process benchmark for commonsense reasoning with clear task definitions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive analysis involving 5 PRMs + 8 LLM judges, dual ref-free/ref-based settings, and downstream evaluations of 7 SLMs.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rigorous task definitions, and detailed statistical analysis.
- Value: ⭐⭐⭐⭐ Reveals severe flaws in answer-only evaluation and provides a practical benchmark and tools for reasoning-aware evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Evaluating Legal Reasoning Traces with Legal Issue Tree Rubrics](evaluating_legal_reasoning_traces_with_legal_issue_tree_rubrics.md)
- [\[ACL 2026\] ReCoQA: A Benchmark for Tool-Augmented and Multi-Step Reasoning in Real Estate Question and Answering](recoqa_a_benchmark_for_tool-augmented_and_multi-step_reasoning_in_real_estate_qu.md)
- [\[ACL 2026\] Evaluating Reasoning Models for Queries with Presuppositions](evaluating_reasoning_models_for_queries_with_presuppositions.md)
- [\[ACL 2026\] Evaluating Temporal Consistency in Multi-Turn Language Models](evaluating_temporal_consistency_in_multi-turn_language_models.md)
- [\[ACL 2026\] Language Models Don't Know What You Want: Evaluating Personalization in Deep Research Needs Real Users](language_models_dont_know_what_you_want_evaluating_personalization_in_deep_resea.md)

</div>

<!-- RELATED:END -->

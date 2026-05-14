---
title: >-
  [Paper Note] ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering
description: >-
  [ACL 2026][LLM Evaluation][Reasoning process evaluation] This paper introduces ReTraceQA, the first reasoning process evaluation benchmark for commonsense question answering, comprising 2…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Reasoning process evaluation"
  - "small language models"
  - "commonsense reasoning"
  - "process reward models"
  - "LLM-as-Judge"
date: 2026-05-08
content_hash: 28d69383b891e1f8
---

# ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering

**Conference**: ACL 2026
**arXiv**: [2510.09351](https://arxiv.org/abs/2510.09351)
**Code**: [https://github.com/SapienzaNLP/ReTraceQA](https://github.com/SapienzaNLP/ReTraceQA)
**Area**: LLM Evaluation / Commonsense Reasoning
**Keywords**: Reasoning process evaluation, small language models, commonsense reasoning, process reward models, LLM-as-Judge

## TL;DR

This paper introduces ReTraceQA, the first reasoning process evaluation benchmark for commonsense question answering, comprising 2,421 instances annotated by domain experts with step-level error localization and error categorization. The benchmark reveals that 14–24% of SLMs produce correct answers via flawed reasoning, and that replacing answer-only evaluation with reasoning-aware evaluation reduces SLM performance by up to 25 percentage points.

## Background & Motivation

**Background**: Small language models (SLMs, ≤10B parameters) achieve increasingly strong performance on commonsense reasoning benchmarks. However, current evaluation practices rely almost entirely on final answer correctness—a prediction is deemed correct as long as it matches the gold answer, with no consideration of whether the underlying reasoning is sound.

**Limitations of Prior Work**: (1) Models can arrive at correct answers via invalid reasoning paths (e.g., shortcut reasoning or accidental correctness from flawed premises), causing answer-only evaluation to artificially inflate performance metrics. (2) Existing reasoning process evaluation benchmarks (ProcessBench, MR-Ben, etc.) focus on mathematics and science, leaving commonsense reasoning process evaluation entirely unaddressed. (3) Process reward models (PRMs) and LLM judges have primarily been used for Best-of-N selection to optimize performance, rather than to examine whether correct answers are obtained through valid reasoning paths.

**Key Challenge**: There is a substantial gap between the high benchmark scores achieved by SLMs and their true reasoning capabilities—answer correctness does not imply reasoning correctness, yet current evaluation frameworks cannot distinguish between the two.

**Goal**: To construct the first step-level reasoning process evaluation benchmark for commonsense reasoning, quantify the degree to which answer-only evaluation overestimates SLM capabilities, and assess the performance of LLM judges and PRMs in the commonsense reasoning domain.

**Key Insight**: The paper focuses on *process errors*—instances where the answer is correct but the reasoning is flawed—and establishes a gold standard through expert annotation, which is then used to measure the reliability of automated evaluation methods.

**Core Idea**: Seven SLMs generate chain-of-thought reasoning traces across four commonsense reasoning datasets. Three PhD-level experts annotate each trace with step-level error locations and error categories (Misinterpretation / Hallucination / Reasoning), producing a 2,421-instance benchmark. LLM judges and PRMs are then evaluated under both reference-free and reference-based settings.

## Method

### Overall Architecture

The construction pipeline of ReTraceQA consists of: (1) selecting questions from four commonsense reasoning datasets—CSQA, OBQA, QASC, and StrategyQA; (2) generating reasoning traces via zero-shot CoT using seven instruction-tuned SLMs (Llama 3.2/3.1, Qwen 2.5, Phi-4-mini); (3) segmenting reasoning traces into discrete steps; (4) applying balanced sampling to ensure balance across correct/incorrect traces, models, and question uniqueness; and (5) having three expert annotators label the position of the first erroneous step and its error category for each trace.

### Key Designs

1. **Three-Level Hierarchical Error Taxonomy**

    - **Function**: Provides mutually exclusive classification of reasoning errors according to cognitive level.
    - **Mechanism**: Three error types are defined from lower to higher levels—*Misinterpretation* (grounding layer: misunderstanding the question, options, or task requirements, including referencing non-existent options or providing multiple answers); *Hallucination* (content layer: introducing empirically incorrect or unverifiable world knowledge, applied only when the logical structure may be intact but the factual "building blocks" are wrong, e.g., "wolves do not inhabit Arctic regions"); and *Reasoning* (inference layer: making invalid logical leaps between correct premises, e.g., correctly stating "salt lowers the freezing point" but incorrectly inferring "this makes ice easier to form"). Classification follows a grounding-to-reasoning priority order.
    - **Design Motivation**: To distinguish three fundamentally different failure modes—failing to understand the question, lacking factual knowledge, and failing to reason logically—thereby providing targeted diagnostic information for improving SLMs.

2. **First-Error Localization Task Formulation**

    - **Function**: Formalizes reasoning process evaluation as a quantifiable task.
    - **Mechanism**: Given a question $q$ and a reasoning trace $S = [s_0, s_1, \ldots, s_n]$, the model predicts an index $i \in \{-1, 0, \ldots, n\}$, where $i = -1$ indicates all steps are correct and $i \geq 0$ indicates the first error occurs at step $s_i$. Only the first error is targeted, as subsequent steps build on erroneous premises, making their correctness ambiguous.
    - **Design Motivation**: Maintaining a task definition consistent with ProcessBench facilitates cross-domain comparison, and first-error localization avoids the ambiguity of cascading error attribution.

3. **Dual-Axis Evaluation Framework (Reference-Free + Reference-Based × Judge + PRM)**

    - **Function**: Comprehensively assesses automated reasoning evaluation methods in the commonsense domain.
    - **Mechanism**: The reference-free setting (providing only the reasoning trace, without the gold answer) tests the reliability of LLM judges and PRMs as training feedback or Best-of-N selection signals. The reference-based setting (providing both the gold answer and the reasoning trace) tests their utility as evaluation tools. Both settings are evaluated using *correct* (accuracy in identifying fully correct traces), *error* (accuracy in localizing the first erroneous step), and *F1* (harmonic mean of the two).
    - **Design Motivation**: The reference-free setting reflects real deployment scenarios (gold answers are unavailable during training), while the reference-based setting reflects evaluation scenarios. Their combination reveals the strengths and weaknesses of different models under different conditions.

### Loss & Training

This paper presents an evaluation benchmark and does not involve model training. LLM judges use a slightly adapted version of the ProcessBench prompt template. PRMs employ either thresholded decisions from sigmoid-activated outputs or threshold selection via F1 maximization. All open-source models use greedy decoding; o1-mini and DeepSeek-R1 use temperature 1.0 due to API constraints.

## Key Experimental Results

### Main Results

| Model | CSQA F1 | OBQA F1 | QASC F1 | StrategyQA F1 | Avg. F1 |
|-------|---------|---------|---------|---------------|---------|
| **Reference-Based LLM Judges** | | | | | |
| o1-mini | 65.7 | 79.2 | 74.2 | 78.3 | 74.4 |
| GPT-4o | 67.9 | 76.6 | 66.2 | 65.3 | 69.0 |
| Qwen2.5-72B | 64.7 | 69.9 | 69.7 | 67.3 | 67.9 |
| Gemini-2.0-Flash | 65.2 | 74.5 | 68.4 | 62.4 | 67.6 |
| DeepSeek-R1 | 57.4 | 56.4 | 56.7 | 47.2 | 54.4 |
| **Reference-Free PRMs** | | | | | |
| Qwen2.5-Math-PRM-7B | 33.8 | 42.8 | 48.6 | 37.4 | 40.7 |
| Math-Shepherd-PRM-7B | 8.0 | 11.5 | 17.9 | 28.4 | 16.5 |

| SLM | Answer-Only Accuracy | Reasoning-Aware Accuracy | Inflation Δ |
|-----|---------------------|--------------------------|-------------|
| Qwen2.5-7B | 81.0 | 67.5 | 13.5 |
| Llama-3.1-8B | 76.3 | 63.1 | 13.2 |
| Qwen2.5-3B | 70.4 | 48.5 | 22.0 |
| Llama-3.2-1B | 49.0 | 23.4 | 25.6 |
| Average | 68.3 | 49.7 | 18.6 |

### Ablation Study

| Dataset | Process Error Rate (Correct Answer, Flawed Reasoning) |
|---------|------------------------------------------------------|
| CSQA | 16.3% |
| OBQA | 14.7% |
| QASC | 16.6% |
| StrategyQA | 24.0% |
| Average | 17.9% |

### Key Findings

- **17.9% of correct answers stem from flawed reasoning**: On average, roughly one in every five to six "correct" responses involves an erroneous reasoning process, reaching 24% on StrategyQA, demonstrating that answer-only evaluation substantially overestimates SLM capabilities.
- **Reasoning-aware evaluation yields substantial performance drops**: Using o1-mini as a reasoning judge, average SLM accuracy drops from 68.3% to 49.7% (a decrease of 18.6 pp), with Llama-3.2-1B dropping from 49.0% to 23.4% (a decrease of 25.6 pp).
- **Hallucination is the dominant failure mode in SLM reasoning**: Hallucination errors account for 41.9%–62.5% of all errors, followed by reasoning errors (27.9%–35.4%), with misinterpretation errors being the least frequent (9.6%–24.1%). SLMs generally understand the question but frequently fabricate false "facts."
- **Mathematical PRMs do not transfer to commonsense reasoning**: The strongest mathematical PRM achieves an average F1 of only 40.7%, compared to 74.4% for the strongest LLM judge, indicating highly limited generalization of PRMs across domains.
- **LLM judges excel at holistic judgment but struggle with error localization**: Models achieve substantially higher *correct* scores (detecting fully correct traces) than *error* scores (localizing specific erroneous steps), suggesting that precise localization of reasoning errors remains an open challenge.
- **Errors are most frequent in intermediate steps (steps 3–4)**: Early context-setting steps are generally successful; errors emerge at the intermediate reasoning stage. The prediction distribution of o1-mini closely matches human annotations but shows a tendency to over-attribute errors to later steps.

## Highlights & Insights

- **First quantification of the severity of "correct answer ≠ correct reasoning" in commonsense reasoning**: A process error rate of 17.9% and performance inflation of up to 25 pp serve as a warning to the community—leaderboard scores overestimate true capabilities by nearly 19 percentage points on average.
- **Practical value of the hierarchical error taxonomy**: The error distribution pattern of Hallucination > Reasoning > Misinterpretation clearly identifies factual grounding—rather than logical inference or question comprehension—as the core weakness of SLMs, providing actionable guidance for improvement.
- **A cautionary finding on cross-domain transfer**: The poor performance of mathematical PRMs on commonsense reasoning (average F1 of only 21.1% in the reference-free setting) demonstrates that "mathematical reasoning ≠ general reasoning," motivating the development of domain-specific process reward models.
- **High annotation quality**: Three PhD-level expert annotators achieve a Fleiss's Kappa of 0.84 ("almost perfect agreement"), establishing a reliable gold standard for the field.

## Limitations & Future Work

- Only SLMs with ≤10B parameters are evaluated; the reasoning process quality of larger models is not addressed.
- The "correctness" of commonsense reasoning is inherently subjective—annotators may disagree on whether certain world knowledge claims are correct.
- Only zero-shot CoT is used to generate reasoning traces; few-shot prompting and other strategies are not explored.
- Future work should develop PRMs specifically designed for commonsense reasoning, rather than relying on transfer from the mathematical domain.
- The framework can be extended to additional reasoning domains (legal, ethical, social reasoning, etc.).

## Related Work & Insights

- **vs. ProcessBench**: ProcessBench addresses error localization only in mathematical reasoning; ReTraceQA is the first to extend process evaluation to commonsense reasoning.
- **vs. MR-Ben/MR-GSM8K**: These benchmarks provide error localization, explanation, and correction, but are likewise restricted to mathematics and science. ReTraceQA demonstrates that commonsense reasoning requires a distinct evaluation framework.
- **vs. MMErroR**: MMErroR evaluates VLMs' ability to diagnose given erroneous reasoning traces; ReTraceQA evaluates step-level process assessment of reasoning traces generated by SLMs themselves. The two are complementary.
- **vs. PRMs (Math-Shepherd/Qwen2.5-Math-PRM)**: Experiments in ReTraceQA demonstrate that mathematical PRMs cannot transfer to commonsense reasoning, highlighting the necessity of domain-specific evaluation.

## Rating

- Novelty: ⭐⭐⭐⭐ First step-level reasoning process evaluation benchmark for commonsense reasoning, with a well-defined task formulation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five PRMs, eight LLM judges, reference-free/reference-based dual settings, and downstream evaluation across seven SLMs, yielding a comprehensive analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear paper structure, rigorous task definition, and detailed statistical analysis.
- Value: ⭐⭐⭐⭐ Exposes critical deficiencies of answer-only evaluation and provides a practical benchmark and tools for reasoning-aware assessment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Closing the Modality Reasoning Gap for Speech Large Language Models](closing_the_modality_reasoning_gap_for_speech_large_language_models.md)
- [\[NeurIPS 2025\] Small Language Models as Compiler Experts: Auto-Parallelization for Heterogeneous Systems](../../NeurIPS2025/llm_evaluation/small_language_models_as_compiler_experts_auto-parallelization_for_heterogeneous.md)
- [\[ACL 2026\] Are They Lovers or Friends? Evaluating LLMs' Social Reasoning in English and Korean Dialogues](are_they_lovers_or_friends_evaluating_llms39_social_reasoning_in_english_and_kor.md)
- [\[AAAI 2026\] ConInstruct: Evaluating Large Language Models on Conflict Detection and Resolution in Instructions](../../AAAI2026/llm_evaluation/coninstruct_evaluating_large_language_models_on_conflict_detection_and_resolutio.md)
- [\[ICLR 2026\] Predicting LLM Reasoning Performance with Small Proxy Model](../../ICLR2026/llm_evaluation/predicting_llm_reasoning_performance_with_small_proxy_model.md)

</div>

<!-- RELATED:END -->

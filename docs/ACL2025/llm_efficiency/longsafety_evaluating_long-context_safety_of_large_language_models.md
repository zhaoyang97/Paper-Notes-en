---
title: >-
  [Paper Note] LongSafety: Evaluating Long-Context Safety of Large Language Models
description: >-
  [ACL 2025][LLM Efficiency][Long-context safety] Proposes LongSafety, the first LLM safety evaluation benchmark specifically tailored for open-ended long-context tasks. It covers 7 safety categories and 6 task types across 1,543 test cases. The evaluation reveals that most models achieve a safety rate below 55%, and safety capabilities in short contexts do not transfer well to long-context scenarios.
tags:
  - "ACL 2025"
  - "LLM Efficiency"
  - "Long-context safety"
  - "safety evaluation benchmark"
  - "multi-agent evaluation"
  - "LLM safety"
  - "long text"
date: 2026-05-08
content_hash: bbe8b647e7ddccc4
---

# LongSafety: Evaluating Long-Context Safety of Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2502.16971](https://arxiv.org/abs/2502.16971)  
**Code**: [github.com/thu-coai/LongSafety](https://github.com/thu-coai/LongSafety)  
**Area**: LLM Efficiency  
**Keywords**: Long-context safety, safety evaluation benchmark, multi-agent evaluation, LLM safety, long text

## TL;DR

Proposes LongSafety, the first LLM safety evaluation benchmark specifically tailored for open-ended long-context tasks. It covers 7 safety categories and 6 task types across 1,543 test cases. The evaluation reveals that most models achieve a safety rate below 55%, and safety capabilities in short contexts do not transfer well to long-context scenarios.

## Background & Motivation

With advancements in long-sequence processing, LLMs have demonstrated remarkable capabilities in understanding and generating long texts. However, safety issues in long-context scenarios (such as implicit alignment risks within verbose content and cognitive disruption to models) have become increasingly prominent, yet systematic evaluation tools remain lacking.

Existing long-context benchmarks (e.g., LongBench, InfiniteBench, RULER) primarily focus on general capability evaluation and do not cover safety issues. Meanwhile, current safety benchmarks (e.g., SafetyBench, Red Team, AdvBench) are typically restricted to short-context query tasks of a few hundred words, failing to evaluate long-context models that process documents spanning thousands of words. The only parallel work, LongSafetyBench, utilizes a multiple-choice format, which is insufficient for evaluating generation safety—an aspect far more critical for generative models.

This research gap motivates the authors to propose LongSafety, the first comprehensive safety evaluation benchmark for **open-ended** long-context tasks.

## Method

### Overall Architecture

The construction process of LongSafety consists of three core stages: data collection (gathering long safety-related documents from the internet), instruction curation (writing instructions for each document that could trigger safety violations), and a multi-agent evaluation framework (assessing the safety of model responses).

The problem is formulated as follows: given a long context $C$ and a safety instruction $I$, the model generates a response $R$, which is evaluated for safety. Since instructions can be appended either before or after the context, a test case is considered passed only if the responses under both configurations are safe.

### Key Designs

1. **Safety Taxonomy (7 safety categories)**: It covers Toxicity Content, Biased Opinion, Physical & Mental Harm, Illegal Activities, Unethical Activities, Privacy & Property, and Sensitive Topics. This taxonomy is revised from prior frameworks by Sun et al. and Zhang et al., and adapted for long-context scenarios.

2. **Diverse Task Types (6 types)**: Includes Question Answering (QA), Generation, Brainstorming, Summarization, Rewrite, and Role-playing. The first five are derived from Ouyang et al., and Role-playing is introduced as a new task type to further expand the coverage of long-context tasks.

3. **Multi-Agent Evaluation Framework**: Comprises three specialized roles driven by LLMs:

    - **Risk Analyzer**: Analyzes hidden safety risks in the instructions and generates a reference set of behaviors likely to lead to safe/unsafe responses.
    - **Context Summarizer**: Generates concise summaries for long contexts, capturing key information and highlighting instruction-relevant content to filter out distractors.
    - **Safety Judge**: Integrates the risk analysis and context summary to make a binary decision (safe/unsafe) on the model's response.
   
   Through collaborative multi-perspective analysis, this framework achieves 92% accuracy on the test set, significantly outperforming single-agent evaluators.

4. **Data Collection Pipeline**: 

    - Crowdsourced workers search for relevant documents on the internet using predefined safety keywords.
    - Workers extract plain text content (multiple documents can be combined to form a long context).
    - Three safety instructions of different task types are written for each context.
    - The instructions most likely to trigger safety issues are retained, while samples with inconsistencies between context and instructions are filtered out.

### Loss & Training

Since LongSafety is an evaluation benchmark rather than a training method, it does not involve loss function design. Evaluation is conducted using the newly proposed $SR_{long}$ metric: a test instance is marked safe only when the model's responses under both prefix and suffix instruction placements are judged as safe.

## Key Experimental Results

### Main Results

| Model | $SR_{long}$ | $SR_{short}$ | Drop |
|------|-------------|--------------|----------|
| Claude-3.5-haiku | 77.7% | 89.9% | -12.2% |
| Claude-3.5-sonnet | 76.8% | 94.0% | -17.2% |
| GPT-4-turbo | 48.3% | 84.3% | -36.0% |
| GPT-4o | 40.4% | 73.7% | -33.3% |
| GPT-4o mini | 37.1% | 64.2% | -27.1% |
| Qwen2.5-72B | 31.3% | 72.2% | -40.9% |
| Llama-3.1-8B | 13.4% | 74.2% | -60.8% |

### Ablation Study

| Configuration | Evaluation Accuracy | Description |
|------|-----------|------|
| Full multi-agent framework | 92% | Collaborative work of three agents |
| Without Context Summarizer | 90% | Decreased by 2% but still outperforms a single Judge |
| Single GPT-4o mini Judge | <90% | Only uses Safety Judge |
| Llama-Guard-3 | Lowest | Traditional safety guardrail |

### Key Findings

- **Misalignment between Short and Long-Context Safety**: Models with high safety rankings in short contexts can perform poorly in long contexts. For instance, Llama-3.1-8B ranks 2nd in short-context safety but experiences a drop of over 60% in long contexts, falling to the second-to-last position.
- **Sensitive Topics are Most Challenging**: All open-source models achieve a safety rate below 20% on "Sensitive Topics", and most closed-source models also fall below 50%.
- **Generation-Oriented Tasks are Riskier**: Generative tasks such as Generation, Brainstorming, Summarization, and Rewrite exhibit an average $SR_{long}$ of under 30%, whereas QA tasks reach 46.3%.
- **Context Relevance Exacerbates Risk**: Safety risks are more pronounced when the context is relevant than when it is irrelevant.
- **Input Length Affects Safety**: Safety risks escalate further as the input sequence length increases.
- **Claude-3.5 Series Significantly Leads**: It is the only model family with an average safety rate exceeding 55%.

## Highlights & Insights

- **Filling an Important Gap**: This work presents the first systematic safety evaluation of LLMs in open-ended long-context tasks. The open-ended format reflects real-world scenarios more accurately than multiple-choice questions.
- **Discovery of Safety Non-Transferability**: Short-context safety does not predict long-context safety. This poses a major challenge to current model deployment practices that rely primarily on short-context safety validation.
- **Elegant Multi-Agent Evaluator Design**: Decomposing the evaluation task into risk analysis, context summarization, and safety determination simplifies the process and exemplifies a divide-and-conquer philosophy.
- **Thorough Data Statistics**: Covers 1,543 test cases with an average length of 5,424 words, spanning $7 \times 6 = 42$ combinations of safety categories and task types.

## Limitations & Future Work

- The benchmark primarily targets English content, lacking multi-lingual evaluation.
- Context lengths mostly cluster around thousands of words, without covering ultra-long context scenarios (e.g., 100K+ tokens).
- The multi-agent evaluator relies on GPT-4o mini, which may introduce systemic biases.
- The work does not propose concrete mitigation methods for long-context safety, serving primarily as a diagnostic study.
- The keyword-driven data collection method might leave coverage blind spots.
- Future research could explore long-context-specific safety alignment training strategies.

## Related Work & Insights

- LongBench (Bai et al., 2024) and InfiniteBench (Zhang et al., 2024a) focus on general long-context capabilities, while this work complements them with a safety dimension.
- SafetyBench (Zhang et al., 2023a) and SALAD-Bench (Li et al., 2024) are traditional short-context safety benchmarks, which this study extends to long contexts.
- LongSafetyBench (Huang et al., 2024) similarly targets long-context safety but employs a multiple-choice setup, whereas the open-ended formats in this study are more challenging.
- The paradigm of the multi-agent evaluation framework can be generalized to other complex NLG evaluation tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ First open-ended long-context safety benchmark, though it is primarily an evaluation framework rather than a methodological innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 16 models, 7 safety categories, 6 task types, multi-dimensional analyses, and evaluator comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rich tables and figures, and well-structured analysis.
- Value: ⭐⭐⭐⭐⭐ Highlights the severe state of long-context safety, offering crucial guidance for safety evaluation and alignment research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LongReward: Improving Long-context Large Language Models with AI Feedback](longreward_improving_long-context_large_language_models_with_ai_feedback.md)
- [\[ACL 2025\] CNNSum: Exploring Long-Context Summarization with Large Language Models in Chinese Novels](cnnsum_exploring_long-context_summarization_with_large_language_models_in_chines.md)
- [\[ACL 2025\] Dynamic Chunking and Selection for Reading Comprehension of Ultra-Long Context in Large Language Models](dynamic_chunking_and_selection_for_reading_comprehension_of_ultra-long_context_i.md)
- [\[ACL 2025\] Ref-Long: Benchmarking the Long-Context Referencing Capability of Long-Context Language Models](ref-long_benchmarking_the_long-context_referencing_capability_of_long-context_la.md)
- [\[ACL 2025\] How to Train Long-Context Language Models (Effectively)](train_long_context_effectively.md)

</div>

<!-- RELATED:END -->

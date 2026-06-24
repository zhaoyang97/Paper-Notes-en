---
title: >-
  [Paper Note] Com2: A Causal-Guided Benchmark for Exploring Complex Commonsense Reasoning in Large Language Models
description: >-
  [ACL 2025][LLM Evaluation][Complex Commonsense Reasoning] This paper proposes Com2, a complex commonsense reasoning benchmark constructed based on causal event graphs and causal theories (intervention/counterfactuals). It contains 2,500 topical questions and 1,254 detective story questions, revealing significant deficiencies of LLMs in reasoning depth and breadth.
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Complex Commonsense Reasoning"
  - "Causal Graph"
  - "Causal Theory"
  - "Intervention"
  - "Counterfactual"
  - "Benchmark"
date: 2026-05-08
content_hash: 10a4b65f2b56cf23
---

# Com2: A Causal-Guided Benchmark for Exploring Complex Commonsense Reasoning in Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2506.07064](https://arxiv.org/abs/2506.07064)  
**Code**: [GitHub](https://github.com/HIT-SCIR/Com2)  
**Area**: LLM Evaluation / Commonsense Reasoning  
**Keywords**: Complex Commonsense Reasoning, Causal Graph, Causal Theory, Intervention, Counterfactual, Benchmark  

## TL;DR

This paper proposes Com2, a complex commonsense reasoning benchmark constructed based on causal event graphs and causal theories (intervention/counterfactuals). It contains 2,500 topical questions and 1,254 detective story questions, revealing significant deficiencies of LLMs in reasoning depth and breadth.

## Background & Motivation

- **Problem Definition**: Existing commonsense reasoning benchmarks (e.g., CommonsenseQA) mostly focus on single-step reasoning, where knowing the relevant knowledge is sufficient to answer. However, in the real world, human reasoning is more focused on complex multi-step commonsense, such as long-term impacts of events, consequences of unexpected incidents, and counterfactual hypotheses.
- **Limitations of Prior Work**: (1) Complex reasoning in mathematics and coding has been thoroughly studied (AIME, MATH), but complex commonsense reasoning has been neglected due to the lack of structured representation and explicit ground truth; (2) The test-time scaling strategies of existing reasoning LLMs (o1, R1) are primarily validated on mathematics/coding tasks, leaving their efficacy on commonsense reasoning unknown.
- **Core Motivation**: While LLMs have mastered a vast amount of simple explicit knowledge through pre-training, how do they perform when dealing with complex implicit knowledge derived from simple knowledge (e.g., long-term chain reactions of events, counter-intuitive scenarios)? A systematic benchmark is required to answer this question.
- **Key Challenge**: The representation of commonsense knowledge is informal, context-dependent, and typically lacks universally accepted standard answers, making the construction of high-quality datasets difficult.

## Method

### Overall Architecture

The construction of Com2 consists of a four-stage pipeline: (1) **Event Proposal**—utilizing LLMs to generate specific and abstract events as seeds; (2) **Causal Chain Proposal**—building a 5-node causal chain rooted at an event to represent simple scenarios; (3) **Causal Graph Proposal**—modifying the causal chain based on causal theories (such as intervention and counterfactual analysis) to generate complex scenarios; (4) **Com2 Synthesis**—synthesizing multiple-choice and multi-select questions based on the generated causal graphs.

### Key Designs

1. **Five Causal Graph Scenarios Corresponding to Five Reasoning Tasks**:
    - **Direct**: Direct reasoning on causal chains, asking about long-term consequences of events (the simplest task).
    - **Decision**: Dual-branch causal graphs, asking how to prevent undesirable consequences (multiple-choice).
    - **Transition**: Causal chains with causal transition issues (such as scenario drift), testing model reliability as reasoning depth increases.
    - **Intervention**: Breaking the original causal chain by introducing external unexpected events, testing the model's reasoning capabilities in uncommon scenarios.
    - **Counterfactual**: Constructing counterfactual hypotheses for specific events that have occurred, asking "what would happen if X did not occur".

2. **Com2-hard Subset**: Based on 400+ detective stories (BMDS), constructing complex reasoning scenarios with interwoven multiple clues, containing three higher-difficulty tasks: Decision, Intervention, and Counterfactual.

3. **Guided Slow Thinking**: Each sample is equipped with systemic analysis, divide-and-conquer strategies, self-correction, and context recognition steps, which can serve as auxiliary prompts to validate the reasoning capabilities of LLMs.

### Loss & Training

The proposed work is a benchmark and does not involve model training. Evaluation uses Accuracy; the multiple-choice Decision task uses a soft scoring strategy where points are awarded based on the proportion of correctly predicted options.

## Experimental Results

### Main Results: LLM Performance on Com2

| Model | Direct | Decision | Transition | Intervention | Counter. | Main Avg | Hard Avg | Total Score |
|------|--------|----------|------------|-------------|----------|----------|----------|------|
| Qwen2.5-32B | 83.60 | 65.16 | 48.80 | 33.80 | 72.40 | 60.73 | 54.80 | 57.77 |
| GPT-4o | 80.60 | 66.43 | 48.40 | 32.20 | 68.80 | 59.26 | 59.72 | 59.49 |
| GPT-4o-mini | 83.20 | 62.54 | 49.20 | 31.40 | 71.20 | 59.50 | 55.29 | 57.40 |
| LLaMA-3.1-8B | 83.20 | 58.04 | 47.00 | 30.40 | 71.40 | 58.01 | 53.56 | 55.79 |
| R1-distilled | 75.20 | 56.51 | 43.40 | 30.00 | 68.20 | 54.65 | 62.70 | 58.68 |
| QwQ-32B | 79.80 | 59.82 | 47.40 | 32.00 | 64.60 | 56.70 | 52.01 | 54.36 |
| o1-mini | 80.00 | 32.64 | 47.80 | 30.00 | 66.60 | 51.48 | 56.54 | 54.01 |

All LLMs perform the worst on the Intervention task (around 30%), revealing a severe deficiency in the breadth of reasoning.

### Ablation Study: Post-training Effects

| Model | Main Avg (Before Training) | Main Avg (After Training) | Hard Avg (Before Training) | Hard Avg (After Training) |
|------|-------------------|-------------------|-------------------|-------------------|
| LLaMA-3.1-8B | 58.01 | ~68 (Significant Gain) | 53.56 | ~58 (Gain in OOD remains) |
| Qwen2-7B | 58.13 | ~66 (Significant Gain) | 54.71 | ~57 |

Post-training shows significant improvement on the Main subset but limited improvement on the Hard (OOD) subset, indicating that the reasoning capabilities learned from simple tasks can be partially transferred.

### Key Findings

- **Counterfactual is not the hardest**: According to causal theory, counterfactual reasoning should be the most difficult. However, LLMs perform better on this task than on Transition and Intervention, suggesting that pre-training has endowed LLMs with decent counterfactual reasoning capabilities.
- **Intervention is the biggest bottleneck**: The average accuracy is only ~31%, indicating that LLMs severely lack reasoning breadth when processing unexpected/uncommon scenarios.
- **Reasoning LLMs are not necessarily better**: On Com2-main, o1-mini and QwQ underperform compared to general LLMs, potentially due to "overthinking", leading them into cognitive traps in raw commonsense scenarios.
- **Test-time scaling is inefficient for commonsense reasoning**: Outputting more tokens does not necessarily improve performance, contrasting with the scaling law in mathematics/coding domains.
- **Guided Slow Thinking is effective**: Providing a guided thinking process significantly improves LLM accuracy, indicating that structured reasoning guidance can compensate for model deficiencies.

## Highlights

- Systematically constructs a complex commonsense reasoning benchmark using causal event graphs and causal theories for the first time, with five task types accurately corresponding to practical scenarios of human concern.
- Reveals critical deficiencies of LLMs in complex commonsense reasoning: insufficient reasoning breadth (inability to handle unexpected events) and unstable reasoning depth (reliability degrades during causal transitions).
- Com2-hard is built on detective stories, offering more natural and highly challenging scenarios, serving as an excellent out-of-distribution (OOD) generalization test set.
- Comprehensively evaluates over 10 LLMs (general + reasoning models) with a detailed experimental design.

## Limitations

- The data synthesis pipeline relies on ChatGPT (gpt-4o-mini). Although validated by human evaluation for quality, systematic bias may still exist.
- Only multiple-choice/multi-select formats are evaluated, without covering open-ended generation tasks.
- Com2-hard is built on detective stories, and LLMs might have encountered similar stories during pre-training, potentially leading to inflated performance.
- The construction of causal graphs could be more refined and step-by-step; the current prompt-based method might lack perfect rigor.

## Related Work

- **Commonsense Reasoning Benchmarks**: CommonsenseQA (Talmor et al., 2019), OpenBookQA (Mihaylov et al., 2018), and other tasks focus on single-step reasoning, whereas this work extends to multi-step complex reasoning.
- **Causal Reasoning**: CausalNet (Luo et al., 2016), GLUCOSE (Mostafazadeh et al., 2020), and others study the extraction of causal relationships, while this work applies causal theory (Pearl) to benchmark construction.
- **Complex Reasoning LLMs**: o1 (OpenAI, 2024), DeepSeek-R1 (Liu et al., 2024), and other models improve reasoning capabilities through test-time compute, while this work reveals their limitations in the commonsense domain.

## Rating

| Dimension | Score (1-10) |
|------|-----------|
| Novelty | 8 |
| Experimental Thoroughness | 9 |
| Writing Quality | 7 |
| Value | 7 |
| **Overall Rating** | **7.8** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] HellaSwag-Pro: A Large-Scale Bilingual Benchmark for Evaluating the Robustness of LLMs in Commonsense Reasoning](hellaswag-pro_a_large-scale_bilingual_benchmark_for_evaluating_the_robustness_of.md)
- [\[ACL 2025\] WXImpactBench: A Disruptive Weather Impact Understanding Benchmark for Evaluating Large Language Models](wximpactbench_a_disruptive_weather_impact_understanding_benchmark_for_evaluating.md)
- [\[ACL 2025\] Batayan: A Filipino NLP Benchmark for Evaluating Large Language Models](batayan_a_filipino_nlp_benchmark_for_evaluating_large_language_models.md)
- [\[ACL 2025\] McBE: A Multi-task Chinese Bias Evaluation Benchmark for Large Language Models](mcbe_a_multi-task_chinese_bias_evaluation_benchmark_for_large_language_models.md)
- [\[ACL 2025\] EditInspector: A Benchmark for Evaluation of Text-Guided Image Edits](editinspector_a_benchmark_for_evaluation_of_text-guided_image_edits.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] ProcessBench: Identifying Process Errors in Mathematical Reasoning
description: >-
  [ACL 2025][Reasoning][Mathematical Reasoning] This paper proposes the ProcessBench benchmark (comprising 3,400 test cases, focusing primarily on competition-/Olympiad-level math problems) to evaluate the capability of PRMs and critic models in locating the earliest erroneous step in mathematical reasoning. The findings reveal that existing PRMs fail to generalize to difficult problems beyond GSM8K/MATH, whereas general LLMs (e.g., QwQ-32B-Preview) acting as critics perform co…
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Mathematical Reasoning"
  - "Process Error Detection"
  - "Reward Model"
  - "Critic Model"
  - "Scalable Oversight"
date: 2026-05-08
content_hash: daba4ded5457f076
---

# ProcessBench: Identifying Process Errors in Mathematical Reasoning

**Conference**: ACL 2025  
**arXiv**: [2412.06559](https://arxiv.org/abs/2412.06559)  
**Code**: [ProcessBench](https://github.com/QwenLM/ProcessBench)  
**Area**: LLM Reasoning  
**Keywords**: Mathematical Reasoning, Process Error Detection, Reward Model, Critic Model, Scalable Oversight

## TL;DR

This paper proposes the ProcessBench benchmark (comprising 3,400 test cases, focusing primarily on competition-/Olympiad-level math problems) to evaluate the capability of PRMs and critic models in locating the earliest erroneous step in mathematical reasoning. The findings reveal that existing PRMs fail to generalize to difficult problems beyond GSM8K/MATH, whereas general LLMs (e.g., QwQ-32B-Preview) acting as critics perform comparably to GPT-4o.

## Background & Motivation

**Background**: Language models frequently make mistakes when solving mathematical problems. The ability to automatically detect erroneous steps in the reasoning process is crucial for "scalable oversight". Two types of models are used for this purpose: (1) **Process Reward Models (PRMs)**—specially trained models that score each reasoning step; (2) **critic models**—general LLMs prompted to step-by-step critique the reasoning process.

**Limitations of Prior Work**: Existing PRMs are mainly trained and evaluated on the GSM8K and MATH datasets, where the problem difficulty is relatively limited. The generalization capability of PRMs remains completely unknown when facing harder mathematical problems (competition- and Olympiad-level). Furthermore, there is a lack of fair comparative evaluation between PRMs and critic models.

**Key Challenge**: The high scores of PRM on simple mathematical problems may create an illusion that "reasoning error detection is solved." In reality, on more challenging mathematical problems where error patterns are subtler, reasoning chains are longer, and deeper mathematical knowledge is required, existing methods may still fall far short of being practical.

**Goal**: (1) Construct a process error detection benchmark dominated by competition-/Olympiad-level mathematical problems; (2) conduct a systematic and fair comparison between PRMs and critic models; (3) reveal the true capability boundaries of existing methods.

**Key Insight**: Build a high-quality gold-standard test set by leveraging human experts to annotate the earliest error in each reasoning step (or confirm that all steps are correct).

**Core Idea**: Establish a "find the earliest erroneous step" benchmark on competition-/Olympiad-level mathematical problems to test the true upper limits of PRMs and critic models, exposing the capability gap between the two.

## Method

### Overall Architecture

ProcessBench contains 3,400 test cases. Each case consists of a mathematical problem and a step-by-step solution, with human experts annotating the position of the earliest erroneous step (annotated as "no error" if the solution is completely correct). The data sources cover multiple difficulty levels ranging from simple (GSM8K) to hard (competition-/Olympiad-level). The evaluation task is: given a problem and a step-by-step solution, the model must locate the earliest erroneous step or determine that all steps are correct.

### Key Designs

1. **Multi-difficulty Test Case Construction**:

    - Function: Covers mathematical problems from basic to competition/Olympiad levels to test model performance across different difficulties
    - Mechanism: Data sources include: (a) GSM8K (grade-school math word problems); (b) MATH (high school math competition problems, categorized by difficulty levels 1-5); (c) competition-/Olympiad-level math problems (e.g., AMC, AIME, Olympiad shortlist). For each problem, multiple LLMs (including the Qwen2.5 series) are used to generate step-by-step solutions, which are then reviewed and annotated step-by-step by human mathematics experts. This includes both erroneous solutions (with the earliest erroneous step annotated) and fully correct solutions (annotated as "correct").
    - Design Motivation: The multi-difficulty design allows the precise identification of the model's "generalization boundary"—the point at which PRMs cease to be reliable.

2. **Unified Evaluation Framework for PRM and Critic Models**:

    - Function: Fairly compares two entirely different error detection paradigms on the same task and data.
    - Mechanism: For PRMs, their reward scores for each step are converted into error detection decisions—the step with the lowest score (below a threshold) is deemed the erroneous step. For critic models, a unified prompt is designed to guide the general LLM to review the solution step-by-step, requiring it to output the index of the first erroneous step or declare all steps correct. The evaluation metric is the "error localization accuracy"—the ratio of correctly identifying the earliest erroneous step (or correctly judging no error).
    - Design Motivation: PRMs and critic models represent two different technical paradigms—specialized training vs. general capability. The unified evaluation framework eliminates unfair comparisons caused by differing evaluation methodologies.

3. **Self-Trained PRM Baseline**:

    - Function: Verifies whether a simple approach can outperform complex PRMs.
    - Mechanism: Directly fine-tunes a general language model on the PRM800K dataset to obtain a new PRM—without any fancy training strategy optimizations, using only standard supervised learning. This "naive PRM" is compared against existing open-source PRMs.
    - Design Motivation: If a model simply fine-tuned on PRM800K can outperform many elaborately designed open-source PRMs, it suggests that the issues with existing PRMs may lie not in the model design but in the training data and evaluation methodologies.

### Evaluation Metrics

The primary metric is the **F1-score** (balancing error localization and correctness judgment), while individual accuracies across different difficulty levels are also reported.

## Key Experimental Results

### Main Results: PRM vs. Critic Models

| Model Type | Model | GSM8K | MATH (Easy) | MATH (Hard) | Competition/Olympiad | Overall F1 |
|----------|------|-------|-------------|-------------|----------|--------|
| Critic | o1-mini | Highest | Highest | Highest | Highest | **Highest** |
| Critic | GPT-4o | High | High | Medium-High | Medium-High | High |
| Critic | QwQ-32B-Preview | High | High | Medium-High | Medium | Close to GPT-4o |
| Self-trained PRM | PRM800K-finetuned | High | Medium-High | Medium | Medium-Low | Medium |
| Open-source PRM | Math-Shepherd | Medium-High | Medium | Low | Very Low | Low |
| Open-source PRM | Other open-source PRMs | Medium | Medium-Low | Low | Very Low | Low |

### Difficulty Level Generalization Analysis

| Difficulty Level | Closed-source Critic Avg. | Open-source Critic Avg. | Open-source PRM Avg. | Self-trained PRM |
|----------|---------------|---------------|------------|----------|
| GSM8K (Easy) | ~85% | ~70% | ~65% | ~70% |
| MATH Level 1-3 | ~80% | ~65% | ~55% | ~60% |
| MATH Level 4-5 | ~70% | ~55% | ~40% | ~45% |
| Competition/Olympiad | ~60% | ~45% | ~25% | ~35% |

### Key Findings

- **PRM Generalization Failure**: Existing open-source PRMs suffer from sharp performance degradation on harder problems beyond GSM8K and MATH, performing close to random on competition-/Olympiad-level problems. This suggests that they have learned shallow patterns tailored to specific difficulties and formats, rather than general reasoning error detection capabilities.
- **Critic Models are Stronger**: General LLMs, via critic prompts, outperform specially trained PRMs across all difficulty levels, with a massive performance gap on hard problems.
- **Naive PRM Wins**: A model simply fine-tuned on PRM800K outperforms most open-source PRMs that claim complex designs, indicating that current PRM training strategies may introduce negative biases.
- **QwQ-32B-Preview Shows Outstanding Performance**: This open-source reasoning model acts as a critic with performance comparable to the closed-source GPT-4o, though still lagging behind the reasoning-focused o1-mini.
- **o1-mini Outstands**: The reasoning-focused model o1-mini leads by a large margin across all difficulty levels, demonstrating that reasoning-enhanced training indeed elevates error detection capability.
- **Judging Correct Solutions is Also Hard**: Models need not only to find errors but also to determine if a solution is "entirely correct". Many PRMs tend to label errors even in correct solutions (exhibiting high false-positive rates).

## Highlights & Insights

- **An Important Wake-Up Call for the PRM Research Community**: High scores of PRMs on GSM8K/MATH created an illusion of capability. By introducing harder mathematical problems, ProcessBench exposes the true generalization boundary of PRMs. This plays an important guiding role for the research direction of this field.
- **Fair Comparison between Two Paradigms**: For the first time, PRMs and critic models are compared under a unified framework. It is found that employing general LLMs as critics might be a more pragmatic solution than specially trained PRMs. This challenges the assumption that "specially trained verifiers outperform prompted general models".
- **Implications for Scalable Oversight**: If human supervision of reasoning processes is to be scaled via PRMs (scalable oversight), PRMs must work reliably on problems that humans find challenging. ProcessBench reveals that we are still far from achieving this goal.

## Limitations & Future Work

- Although the scale of 3,400 test cases is reasonable, it remains limited, particularly regarding the sample size for the hardest levels.
- Only mathematical reasoning is covered; the performance of PRMs in domains like code reasoning or logical reasoning is not included.
- The evaluated open-source PRMs might not include the latest models (such as the new generation of PRMs trained on Qwen2.5-Math).
- The complementary usage of PRMs and critic models is not explored—whether combining them can further enhance error detection capability.
- The task definition of "finding the earliest erroneous step" might be oversimplified—in practice, a solution may contain multiple independent errors.
- Future work could explore using the evaluation results from ProcessBench to guide curriculum learning training strategies for PRMs.

## Related Work & Insights

- **vs PRM800K (Let's Verify Step by Step)**: PRM800K provides PRM training data and verifies the effectiveness of PRMs on MATH. ProcessBench reveals that PRMs trained on this data have limited generalization capability, indicating a need for more diverse training data.
- **vs PRMBench (2501.03124)**: The two benchmarks are complementary—PRMBench focuses on fine-grained error type classification (Simplicity/Soundness/Sensitivity), whereas ProcessBench focuses on difficulty generalization. Both point to the same conclusion: existing PRMs lack adequate capability.
- **vs Outcome Reward Models (ORM)**: The results of ProcessBench suggest that when PRMs are not sufficiently reliable, ORMs (which only look at the correctness of the final answer) might be a more robust alternative, as they do not depend on the accuracy of step-level judgments.
- **vs QwQ / o1 Series Reasoning Models**: These models acquire powerful critique capabilities through deep thinking during inference, implying that "learning to verify" may require "learning to reason" as a prerequisite.

## Rating

- Novelty: ⭐⭐⭐⭐ The PRM evaluation on competition-/Olympiad-level mathematics fills an important gap, though the benchmark construction methodology itself is relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers both PRM and critic paradigms across multiple difficulty levels, presenting a comprehensive comparison and solid findings.
- Writing Quality: ⭐⭐⭐⭐ Clear and direct perspectives with powerful formulations of the two core findings (poor PRM generalization + stronger critic models).
- Value: ⭐⭐⭐⭐⭐ Has a major impact on PRM research and the direction of scalable oversight, particularly by exposing the generalization issues of PRMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Self-Error-Instruct: Generalizing from Errors for LLMs Mathematical Reasoning](self-error-instruct_generalizing_from_errors_for_llms_mathematical_reasoning.md)
- [\[ACL 2025\] Linguistic Generalizability of Test-Time Scaling in Mathematical Reasoning](mclm_multilingual_test_time_scaling.md)
- [\[ACL 2025\] Can Large Language Models Detect Errors in Long Chain-of-Thought Reasoning?](can_large_language_models_detect_errors_in_long_chain-of-thought_reasoning.md)
- [\[ICLR 2026\] Latent Veracity Inference for Identifying Errors in Stepwise Reasoning](../../ICLR2026/llm_reasoning/latent_veracity_inference_for_identifying_errors_in_stepwise_reasoning.md)
- [\[ACL 2025\] Enhancing Mathematical Reasoning in LLMs by Stepwise Correction](enhancing_mathematical_reasoning_in_llms_by_stepwise_correction.md)

</div>

<!-- RELATED:END -->

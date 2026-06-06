---
title: >-
  [Paper Note] SessionIntentBench: A Multi-Task Inter-Session Intention-Shift Modeling Benchmark
description: >-
  [ACL 2026][LLM Evaluation][Shopping Intent] This paper proposes SessionIntentBench, a multi-task benchmark evaluating L(V)LM's ability to understand inter-step intention shifts in e-commerce shopping sessions. It compris…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Shopping Intent"
  - "Session Modeling"
  - "E-commerce Recommendation"
  - "Intent Drift"
date: 2026-05-08
content_hash: 08d33462dd3028f1
---

# SessionIntentBench: A Multi-Task Inter-Session Intention-Shift Modeling Benchmark

**Conference**: ACL 2026  
**arXiv**: [2507.20185](https://arxiv.org/abs/2507.20185)  
**Code**: None  
**Area**: LLM Evaluation  
**Keywords**: Shopping Intent, Session Modeling, E-commerce Recommendation, Intent Drift, LLM Evaluation

## TL;DR

This paper proposes SessionIntentBench, a multi-task benchmark evaluating L(V)LM's ability to understand inter-step intention shifts in e-commerce shopping sessions. It comprises four progressive sub-tasks (Intent Purchase Likelihood Estimation, Attribute Regularization, Intent Validation Comparison, and Intent Evolution Modeling), with 1.9 million intent entries and 1.13 million intent trajectories. Experiments show that over 20 current L(V)LMs underperform in capturing complex session intentions.

## Background & Motivation

**Background**: User intent modeling is crucial in e-commerce. Existing methods either analyze user profiles and purchase history or perform single-purchase intent inference using surface information like product titles and prices. Shopping sessions record user interactions across a series of browsing activities.

**Limitations of Prior Work**: (1) Existing works cover only single dimensions of sessions or intentions, failing to model them jointly; (2) Relying solely on product titles and images as reasoning clues neglects rich product metadata; (3) There is a lack of automated intent data construction pipelines and systematic evaluation benchmarks.

**Key Challenge**: In complex multi-step shopping sessions, user intent changes dynamically (e.g., from red sneakers → white casual shoes → low-priced shoes), but LLMs cannot effectively connect scattered information within the session to track such intent drifts.

**Goal**: (1) Design the concept of an intent tree and an automated data construction pipeline; (2) Build a multi-task benchmark to evaluate L(V)LM's cross-session intent understanding; (3) Verify the performance gains for LLMs when explicit intent information is injected.

**Key Insight**: Intent modeling is decomposed into four progressive sub-tasks—ranging from validating intent-product alignment and checking key attributes to comparing adjacent products and predicting future exploration directions.

**Core Idea**: An "intention tree" is used to structuredly represent the branching and evolution of intentions within a session. Intent metadata is automatically generated via multi-step L(V)LM prompting to build a scalable intent modeling benchmark.

## Method

### Overall Architecture

The SessionIntentBench construction pipeline consists of four stages: (1) Multimodal Attribute Extraction—using GPT-4o-mini to extract standardized attributes from product text and images; (2) Intent Generation—gradually inferring user intent lists along the session timeline to form an intention tree (5 branches for each of the first 5 steps, then 1 branch per step); (3) Intent Drift Metadata Analysis—extracting key attributes and comparisons between adjacent products; (4) Manual Annotation—quality verification of sampled subsets by AMT workers.

### Key Designs

1.  **Intention Tree Construction**:
    - **Function**: Structuredly represents the branching and evolution of user intentions within a session.
    - **Mechanism**: Using the product sequence $P_1, P_2, ..., P_T$ in the session as the backbone, the LLM infers 5 possible intentions at each time step to form a tree structure. After the 5th step, only 1 intention is inferred to control exponential growth. This results in 1.13 million intent trajectories (paths from root to leaf).
    - **Design Motivation**: Real-world user purchase intent is diverse; the intention tree represents "multiple reasonable intent hypotheses that may exist under the same interaction history."

2.  **Four-Task Evaluation System**:
    - **Function**: Evaluates LLM intent understanding from four complementary perspectives.
    - **Mechanism**: Task 1 tests matching between inferred intent and new products; Task 2 checks if key attributes are reflected in new products; Task 3 evaluates if adjacent product comparisons reasonably explain intent shifts; Task 4 predicts whether to continue recommending similar products, products with different features in the same category, or cross-category exploration. All four tasks output scores from 0-3.
    - **Design Motivation**: A single task cannot comprehensively evaluate intent understanding; integrated assessment across alignment, regularization, comparison, and prediction is necessary.

3.  **Intent Injection Experiments**:
    - **Function**: Verifies the performance boost of explicit intent information on LLM decision-making.
    - **Mechanism**: Inferred intent information (e.g., "The user may be looking for low-priced white sneakers") is added to the prompt to compare model performance with and without intent information.
    - **Design Motivation**: If intent information improves performance, it indicates that LLMs currently lack the ability to autonomously extract intent from raw sessions.

### Loss & Training

The benchmark evaluation mainly utilizes zero-shot and few-shot prompting without specialized training. Fine-tuning experiments use SFT to fine-tune Llama-3.1-8B and Llama-3.2-3B on the training set. Manual annotation uses Amazon Mechanical Turk with multi-round screening to ensure quality.

## Key Experimental Results

### Main Results

**Zero-shot L(V)LM Performance (Accuracy %)**

| Model | Task 1 Acc | Task 2 Acc | Task 3 Acc | Task 4 Acc |
| :--- | :--- | :--- | :--- | :--- |
| Random | 50.00 | 50.00 | 50.00 | 54.38 |
| Majority | 62.30 | 54.35 | 71.80 | 63.15 |
| Qwen-2.5-7B | 58.62 | 51.02 | 70.59 | 40.07 |
| LLaVA-v1.6-vicuna-7b | 62.01 | 46.93 | 71.27 | 37.21 |
| Mistral-7B-v0.3 | 62.17 | 47.65 | 71.30 | 39.61 |

### Ablation Study

| Configuration | Effect | Description |
| :--- | :--- | :--- |
| Zero-shot | Baseline level | Most models perform near or below the majority baseline. |
| Few-shot | Minor gain | However, performance on some tasks actually decreases. |
| Fine-tuning (SFT) | Mixed results | Improvements on some tasks, but no comprehensive gain. |
| + Intent Injection | Significant gain | Demonstrates the value of explicit intent information. |

### Key Findings

- Performance of 20+ L(V)LMs across the four tasks is generally near or below the majority baseline, indicating that current models cannot effectively understand session intentions.
- Task 2 (Attribute Regularization) is the most subjective and has the lowest inter-annotator agreement.
- Multimodal models (LVLM) do not perform better than text-only LLMs, indicating product image information is not effectively utilized.
- Intent injection experiments prove that LLM performance improves significantly when intent is explicitly provided, suggesting the bottleneck lies in intent extraction rather than reasoning.
- Inconsistent fine-tuning results suggest that session intent understanding requires deeper reasoning rather than pattern memorization.

## Highlights & Insights

- The intention tree concept formalizes implicit user mental states into a computable tree structure, providing a new representation paradigm for intent modeling.
- The four-task evaluation system is cleverly designed, forming a progressive assessment from alignment → validation → comparison → prediction.
- The data scale is massive (1.9 million intent entries), yet construction costs are controlled through LLM automation and manual sample validation.

## Limitations & Future Work

- Intent generation depends on LLMs (GPT-4o-mini), limiting quality to the LLM's capabilities.
- Manual annotation only covers a sampled subset; the full data quality has not been comprehensively verified.
- The 0-3 scoring criteria for the four tasks are somewhat subjective, particularly for Task 2.
- Future work could explore integrating intent modeling into end-to-end recommendation system training.

## Related Work & Insights

- **vs Amazon-M2 (Jin et al., 2023)**: Amazon-M2 provides raw session data; SessionIntentBench adds intent metadata and evaluation tasks on top of it.
- **vs Sun et al. (2024)**: They optimize recommendations using intent ranking prompts; this work focuses on evaluating LLM intent understanding capabilities.
- **vs Xu et al. (2024)**: They model intentions in co-purchase behaviors but only cover single interactions; this work models intent evolution across sessions.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The intention tree and four-task evaluation system are meaningful new contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 20+ models, multiple evaluation settings, and manual annotation validation.
- **Writing Quality**: ⭐⭐⭐⭐ Task definitions are clear, though notation is heavy.
- **Value**: ⭐⭐⭐⭐ Provides the first systematic benchmark for e-commerce intent modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Multi-Task Reinforcement Learning for Enhanced Multimodal LLM-as-a-Judge](multi-task_reinforcement_learning_for_enhanced_multimodal_llm-as-a-judge.md)
- [\[ACL 2026\] Modeling Multi-Dimensional Cognitive States in Large Language Models under Cognitive Crowding](modeling_multi-dimensional_cognitive_states_in_large_language_models_under_cogni.md)
- [\[ACL 2026\] SciImpact: A Multi-Dimensional, Multi-Field Benchmark for Scientific Impact Prediction](sciimpact_a_multi-dimensional_multi-field_benchmark_for_scientific_impact_predic.md)
- [\[ACL 2026\] Reward Modeling for Scientific Writing Evaluation](reward_modeling_for_scientific_writing_evaluation.md)
- [\[ACL 2026\] DiningBench: A Hierarchical Multi-view Benchmark for Perception and Reasoning in the Dietary Domain](diningbench_a_hierarchical_multi-view_benchmark_for_perception_and_reasoning_in_.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] SessionIntentBench: A Multi-Task Inter-Session Intention-Shift Modeling Benchmark
description: >-
  [ACL 2026][LLM Evaluation][shopping intent] This paper proposes SessionIntentBench, a multi-task benchmark for evaluating the ability of L(V)LMs to understand inter-session intention shifts in e-commerce shopping sessions. It comprises four progressively structured subtasks—intent-purchase likelihood estimation, attribute normalization, intent verification contrast, and intent evolution modeling—constructed from 1.9 million intent entries and 1.13 million intent trajectories. Experiments on 20+ L(V)LMs demonstrate that current models perform poorly at capturing complex session-level user intent.
tags:
  - ACL 2026
  - LLM Evaluation
  - shopping intent
  - session modeling
  - e-commerce recommendation
  - intent shift
  - large language model evaluation
date: 2026-05-08
content_hash: 3a453704ee6e24ad
---

# SessionIntentBench: A Multi-Task Inter-Session Intention-Shift Modeling Benchmark

**Conference**: ACL 2026
**arXiv**: [2507.20185](https://arxiv.org/abs/2507.20185)
**Code**: None
**Area**: LLM Evaluation
**Keywords**: shopping intent, session modeling, e-commerce recommendation, intent shift, large language model evaluation

## TL;DR

This paper proposes SessionIntentBench, a multi-task benchmark for evaluating the ability of L(V)LMs to understand inter-session intention shifts in e-commerce shopping sessions. It comprises four progressively structured subtasks—intent-purchase likelihood estimation, attribute normalization, intent verification contrast, and intent evolution modeling—constructed from 1.9 million intent entries and 1.13 million intent trajectories. Experiments on 20+ L(V)LMs demonstrate that current models perform poorly at capturing complex session-level user intent.

## Background & Motivation

**State of the Field**: User intent modeling is critical in e-commerce. Existing approaches either analyze user profiles and purchase histories or rely on surface-level information such as product titles and prices for single-purchase intent inference. Shopping sessions record user interaction behaviors across a sequence of browsing activities.

**Limitations of Prior Work**: (1) Existing work addresses either sessions or intent in isolation, without joint modeling; (2) only product titles and images are used as reasoning cues, neglecting rich product metadata; (3) automated intent data construction pipelines and systematic evaluation benchmarks are lacking.

**Root Cause**: In complex multi-step shopping sessions, user intent is dynamic (e.g., red sneakers → white casual shoes → low-price shoes), yet LLMs cannot effectively connect dispersed session signals to track such intent shifts.

**Paper Goals**: (1) Design the concept of an intent tree and an automated data construction pipeline; (2) build a multi-task benchmark to evaluate L(V)LMs' cross-session intent understanding; (3) validate the performance gains achieved by injecting explicit intent information into LLMs.

**Starting Point**: Intent modeling is decomposed into four progressively structured subtasks—from verifying intent–product alignment, to checking key attributes, to contrasting adjacent products, to predicting future exploration directions.

**Core Idea**: An intent tree structure is used to represent the branching and evolution of intent within a session. Multi-step prompting of L(V)LMs automatically generates intent metadata, enabling the construction of a scalable intent modeling benchmark.

## Method

### Overall Architecture

The SessionIntentBench construction pipeline consists of four stages: (1) **Multimodal attribute extraction**—GPT-4o-mini is used to extract standardized attributes from product text and images; (2) **Intent generation**—user intent lists are incrementally inferred along the session timeline to form an intent tree (five branches per step for the first five steps, then one branch per step thereafter); (3) **Intent-shift metadata analysis**—key attributes and adjacent product contrasts are extracted; (4) **Human annotation**—AMT annotators validate quality on a sampled subset.

### Key Designs

1. **Intent Tree Construction**:

    - **Function**: Structurally represents the branching and evolution of user intent throughout a session.
    - **Mechanism**: Using the product sequence $P_1, P_2, ..., P_T$ in the session as a backbone, an LLM infers five candidate intents at each timestep, forming a tree structure. After step 5, only one intent is inferred per step to control exponential growth. This yields 1.13 million intent trajectories (paths from root to leaf).
    - **Design Motivation**: Real users hold diverse purchase intents; the intent tree can represent "multiple plausible intent hypotheses co-existing under the same interaction history."

2. **Four-Task Evaluation Framework**:

    - **Function**: Evaluates LLMs' intent understanding from four complementary perspectives.
    - **Mechanism**: Task 1 examines whether an inferred intent matches a new product; Task 2 checks whether key attributes are reflected in a new product; Task 3 assesses whether the contrast between adjacent products reasonably explains an intent shift; Task 4 predicts whether to continue recommending similar products, products of the same category with different attributes, or cross-category exploration. All four tasks produce scores on a 0–3 scale.
    - **Design Motivation**: A single task cannot comprehensively assess intent understanding—evaluation must span intent–product alignment, attribute normalization, contrastive verification, and evolution prediction.

3. **Intent Injection Experiments**:

    - **Function**: Validates the performance gains from providing explicit intent information to LLMs.
    - **Mechanism**: Inferred intent information (e.g., "the user may be looking for low-price white sneakers") is inserted into the prompt, and model performance across the four tasks is compared with and without this information.
    - **Design Motivation**: If intent injection improves performance, it indicates that the bottleneck lies in intent extraction from raw sessions rather than in downstream reasoning.

### Loss & Training

The benchmark evaluation primarily employs zero-shot and few-shot prompting, with no dedicated training. Fine-tuning experiments apply SFT on the training set for Llama-3.1-8B and Llama-3.2-3B. Human annotation is conducted via Amazon Mechanical Turk with multi-round filtering to ensure annotation quality.

## Key Experimental Results

### Main Results

**Zero-Shot L(V)LM Performance (Accuracy %)**

| Model | Task 1 Acc | Task 2 Acc | Task 3 Acc | Task 4 Acc |
|---|---|---|---|---|
| Random | 50.00 | 50.00 | 50.00 | 54.38 |
| Majority | 62.30 | 54.35 | 71.80 | 63.15 |
| Qwen-2.5-7B | 58.62 | 51.02 | 70.59 | 40.07 |
| LLaVA-v1.6-vicuna-7b | 62.01 | 46.93 | 71.27 | 37.21 |
| Mistral-7B-v0.3 | 62.17 | 47.65 | 71.30 | 39.61 |

### Ablation Study

| Configuration | Effect | Notes |
|---|---|---|
| Zero-shot | Baseline level | Most models near or below majority |
| Few-shot | Marginal improvement | Some tasks even degrade |
| Fine-tuning (SFT) | Mixed results | Improves some tasks but not comprehensively |
| + Intent injection | Significant improvement | Confirms the value of explicit intent information |

### Key Findings

- 20+ L(V)LMs generally perform at or below the majority baseline across all four tasks, confirming that current models cannot effectively understand session-level intent.
- Task 2 (attribute normalization) is the most subjective subtask and exhibits the lowest inter-annotator agreement.
- Multimodal models (LVLMs) do not outperform text-only LLMs, indicating that product image information is not being effectively utilized.
- Intent injection experiments demonstrate that explicitly providing intent information yields significant performance gains, suggesting the bottleneck is intent extraction rather than reasoning.
- Fine-tuning effects are inconsistent, likely because session intent understanding requires deeper reasoning capabilities rather than pattern memorization.

## Highlights & Insights

- The intent tree concept formalizes implicit user mental states into a computable tree structure, offering a new representational paradigm for intent modeling.
- The four-task evaluation framework is elegantly designed, forming a progressive assessment chain: alignment → verification → contrast → prediction.
- The dataset is large-scale (1.9 million intent entries) yet constructed at manageable cost through LLM automation combined with sampled human verification.

## Limitations & Future Work

- Intent generation relies on an LLM (GPT-4o-mini), and its quality is bounded by that model's capabilities.
- Human annotation covers only a sampled subset; the quality of the full dataset has not been comprehensively validated.
- The 0–3 scoring criteria across the four tasks carry notable subjectivity, particularly for Task 2.
- Future work may explore integrating intent modeling into end-to-end recommendation system training.

## Related Work & Insights

- **vs. Amazon-M2 (Jin et al., 2023)**: Amazon-M2 provides raw session data; SessionIntentBench augments it with intent metadata and evaluation tasks.
- **vs. Sun et al. (2024)**: Their work optimizes recommendations via intent-ranking prompts, whereas this paper focuses on evaluating LLMs' intent understanding capabilities.
- **vs. Xu et al. (2024)**: They model co-purchase behavioral intent but only cover single interactions; this paper models intent evolution across sessions.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The intent tree and four-task evaluation framework constitute meaningful novel contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 20+ models, multiple evaluation settings, and human annotation validation.
- **Writing Quality**: ⭐⭐⭐⭐ Task definitions are clear, though notation is dense.
- **Value**: ⭐⭐⭐⭐ Provides the first systematic benchmark for e-commerce intent modeling.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Task-Aware LLM Routing with Multi-Level Task-Profile-Guided Data Synthesis for Cold-Start Scenarios](task-aware_llm_routing_with_multi-level_task-profile-guided_data_synthesis_for_c.md)
- [\[ACL 2026\] SciImpact: A Multi-Dimensional, Multi-Field Benchmark for Scientific Impact Prediction](sciimpact_a_multi-dimensional_multi-field_benchmark_for_scientific_impact_predic.md)
- [\[ACL 2026\] Modeling Multi-Dimensional Cognitive States in Large Language Models under Cognitive Crowding](modeling_multi-dimensional_cognitive_states_in_large_language_models_under_cogni.md)
- [\[ACL 2026\] ODUTQA-MDC: A Task for Open-Domain Underspecified Tabular QA with Multi-turn Dialogue-based Clarification](odutqa-mdc_a_task_for_open-domain_underspecified_tabular_qa_with_multi-turn_dial.md)
- [\[ACL 2026\] MADE: A Living Benchmark for Multi-Label Text Classification with Uncertainty Quantification](made_a_living_benchmark_for_multi-label_text_classification_with_uncertainty_qua.md)

<!-- RELATED:END -->

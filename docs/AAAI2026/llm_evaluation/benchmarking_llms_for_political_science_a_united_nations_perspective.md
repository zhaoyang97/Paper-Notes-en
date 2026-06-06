---
title: >-
  [Paper Note] Benchmarking LLMs for Political Science: A United Nations Perspective
description: >-
  [AAAI 2026][LLM Evaluation][UN Security Council] This paper presents UNBench, the first comprehensive LLM evaluation benchmark for political science grounded in UN Security Council records from 1994 to 2024. It encompass…
tags:
  - "AAAI 2026"
  - "LLM Evaluation"
  - "UN Security Council"
  - "political science benchmark"
  - "voting simulation"
  - "resolution prediction"
  - "diplomatic text generation"
date: 2026-05-08
content_hash: 4b68681130a93bcd
---

# Benchmarking LLMs for Political Science: A United Nations Perspective

**Conference**: AAAI 2026
**arXiv**: [2502.14122](https://arxiv.org/abs/2502.14122)  
**Code**: [GitHub](https://github.com/yueqingliang1/UNBench)  
**Area**: LLM Evaluation / Political Science
**Keywords**: UN Security Council, political science benchmark, voting simulation, resolution prediction, diplomatic text generation

## TL;DR

This paper presents UNBench, the first comprehensive LLM evaluation benchmark for political science grounded in UN Security Council records from 1994 to 2024. It encompasses four interrelated tasks—resolution drafting, voting simulation, adoption prediction, and representative statement generation—to systematically assess LLMs' ability to understand and simulate complex political dynamics.

## Background & Motivation

**Background**: LLMs have achieved remarkable progress in natural language processing, yet their application in high-stakes political decision-making scenarios remains underexplored. Existing benchmarks (MMLU, BIG-Bench, etc.) include politically relevant tasks but in a fragmented manner, lacking systematic evaluation that reflects real multilateral decision-making processes.

**Limitations of Prior Work**: Political science demands capabilities beyond semantic understanding—predicting coalition dynamics, interpreting ambiguous diplomatic language, and balancing national interests against global norms. Existing LLM evaluations do not cover these higher-order requirements.

**Key Challenge**: UN resolutions, once adopted, become binding international law, and a single Security Council veto can block passage entirely. Quantitative evaluation of model performance in such high-stakes scenarios is entirely absent.

**Goal**: To construct the first unified LLM evaluation benchmark covering the full UN decision-making pipeline (drafting → voting → deliberation), systematically assessing model capabilities in political reasoning, voting prediction, and diplomatic text generation.

**Key Insight**: Systematic extraction of data from publicly available UNSC records, with four interrelated tasks designed to cover the three stages of a resolution's lifecycle.

**Core Idea**: Leverage thirty years of authentic UN Security Council records to build an end-to-end benchmark that comprehensively evaluates LLMs' political reasoning capabilities through four progressively structured tasks.

## Method

### Overall Architecture

UNBench is organized around three stages of a UN resolution's lifecycle, yielding four tasks: the drafting stage (Task 1: Co-Penholder Judgement), the voting stage (Task 2: Voting Simulation; Task 3: Draft Adoption Prediction), and the deliberation stage (Task 4: Statement Generation). Data are sourced from official UNSC records spanning 1994–2024, comprising draft texts, voting records, and diplomatic speeches.

### Key Designs

1. **Task 1: Co-Penholder Judgement**: Given an anonymized resolution draft and its lead sponsor, the model selects the most likely co-penholder from 2–5 candidate countries. This tests the model's ability to understand resolution topics, infer diplomatic alignment, and reason about multilateral cooperation strategies. The task is formatted as multiple-choice with 355,126 instances.

2. **Task 2: Voting Simulation**: The LLM is prompted to role-play as a specific country's representative and cast a vote (in favor / against / abstain) on a given resolution. The model must integrate resolution content, national interest priorities, geopolitical alignments, and veto power considerations. The task contains 17,430 voting instances.

3. **Task 3: Draft Adoption Prediction**: The model predicts whether a resolution will ultimately be adopted. Unlike the individual-level voting in Task 2, this task requires holistic reasoning over the collective dynamics of all 15 Council members, accounting for veto threats, supporting coalitions, and historical precedents. The dataset comprises 1,978 resolutions, of which 98 were not adopted.

4. **Task 4: Statement Generation**: Given a resolution and its voting outcome, the model generates a diplomatic statement on behalf of a country's representative. This evaluates the model's ability to produce persuasive language under political constraints, requiring alignment between national interests, voting rationale, and diplomatic register. The dataset contains 7,394 statements.

### Loss & Training

For classification tasks (Tasks 1–3), a time-based train/test split is applied to simulate the realistic scenario of predicting future events from historical data. BERT and DeBERTa are fine-tuned for 3 epochs with a learning rate of $5 \times 10^{-5}$. LLMs perform inference with temperature $= 0.0$.

## Key Experimental Results

### Main Results

Primary results across the four UNBench tasks:

| Model | Task1 Bal.ACC | Task2 Bal.ACC | Task2 Mac.F1 | Task3 Bal.ACC | Task4 ROUGE | Task4 CosSim |
|-------|------|------|------|------|------|------|
| BERT | 0.011 | 0.537 | 0.396 | 0.333 | / | / |
| DeBERTa | 0.010 | 0.500 | 0.527 | 0.333 | / | / |
| Llama-3.1-8B | 0.665 | 0.530 | 0.168 | 0.357 | 0.039 | 0.355 |
| Mistral-7B | 0.563 | 0.426 | 0.268 | 0.529 | 0.194 | 0.575 |
| **GPT-4o** | **0.726** | **0.823** | **0.696** | **0.677** | 0.199 | 0.619 |
| DeepSeek-V3 | 0.695 | 0.724 | 0.655 | 0.668 | **0.207** | **0.623** |
| Qwen2.5-7B | 0.642 | 0.699 | 0.375 | 0.578 | 0.201 | 0.623 |

### Ablation Study

Effects of increasing the number of candidate options in Task 1 (Co-Penholder Judgement):

- 2 options: GPT-4o achieves the highest accuracy, significantly outperforming other models.
- 5 options: Accuracy declines across all models, with GPT-4o maintaining its lead.
- Traditional models (BERT/DeBERTa) nearly completely fail on Task 1 (~0.01), indicating that political reasoning requires LLM-scale architectures.

Dataset statistics: 30-year span, 515 adopted resolutions, 98 non-adopted resolutions, 7,394 diplomatic statements, covering 204 countries.

### Key Findings

- GPT-4o achieves the strongest performance across predictive tasks (Tasks 1–3), demonstrating superior geopolitical reasoning capabilities.
- Traditional text classification models completely fail on Task 1 (Bal.ACC ~0.01), confirming that the task requires knowledge-level reasoning beyond text comprehension.
- All models yield low ROUGE scores on Task 4 (<0.21), exposing limitations in aligning with precise diplomatic terminology.
- Llama-3.2-3B surpasses GPT-4o on Macro-F1 for Task 3 (0.402 vs. 0.363) while achieving lower Bal.ACC, revealing metric divergence under class imbalance.
- DeepSeek-V3 exhibits the strongest performance on generation tasks, achieving the highest scores on both ROUGE and semantic similarity.

## Highlights & Insights

- UNBench is the first systematic LLM evaluation benchmark covering the full UN decision-making pipeline, filling a significant gap in the political science domain.
- The four tasks progress from selection to prediction to generation, forming a comprehensive and coherent evaluation framework.
- The 30-year dataset captures the evolution of the international order, enabling temporal analysis.
- The complete failure of traditional NLP models on political reasoning tasks underscores the irreplaceable role of LLM-scale knowledge reasoning capabilities.

## Limitations & Future Work

- Severe class imbalance: 1,880 adopted versus 98 non-adopted resolutions, posing challenges for Task 3 evaluation.
- Coverage is limited to English, with multilingual diplomatic scenarios not considered.
- Task 4 evaluation relies solely on ROUGE and cosine similarity, lacking human evaluation of diplomatic text quality.
- Geopolitical biases embedded in model outputs on this data are not assessed (complementary to adjacent work on bias evaluation).
- In-depth analysis of model performance variation across different time periods or regional issue areas is absent.

## Related Work & Insights

- **vs. MMLU/BIG-Bench**: General-purpose benchmarks cover political science in a fragmented manner; UNBench provides end-to-end evaluation grounded in authentic multilateral scenarios.
- **vs. Harvard Dataverse UN datasets**: The latter contains only voting statistics, lacking draft texts and debate records; UNBench covers the full pipeline.
- **vs. Nation-Level Bias paper**: Both utilize UNSC data but from different perspectives—UNBench assesses capability while the bias paper assesses fairness, making them highly complementary.

## Rating

- Novelty: ⭐⭐⭐⭐ First LLM benchmark for UN political science with innovative task design, though the core methodology is relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model comparisons and multi-task evaluation are provided, but additional SOTA models and human evaluation are lacking.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, formalized task definitions, and detailed descriptions of the data construction process.
- Value: ⭐⭐⭐⭐ Provides important infrastructure for AI × political science research, with 30 years of data spanning multiple shifts in the international order.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DARE-bench: Evaluating Modeling and Instruction Fidelity of LLMs in Data Science](../../ICLR2026/llm_evaluation/dare-bench_evaluating_modeling_and_instruction_fidelity_of_llms_in_data_science.md)
- [\[ICLR 2026\] Benchmarking Overton Pluralism in LLMs](../../ICLR2026/llm_evaluation/benchmarking_overton_pluralism_in_llms.md)
- [\[ACL 2026\] Personalized Benchmarking: Evaluating LLMs by Individual Preferences](../../ACL2026/llm_evaluation/personalized_benchmarking_evaluating_llms_by_individual_preferences.md)
- [\[ACL 2026\] ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition](../../ACL2026/llm_evaluation/researchbench_benchmarking_llms_in_scientific_discovery_via_inspiration-based_ta.md)
- [\[ACL 2026\] BizCompass: Benchmarking the Reasoning Capabilities of LLMs in Business Knowledge and Applications](../../ACL2026/llm_evaluation/bizcompass_benchmarking_the_reasoning_capabilities_of_llms_in_business_knowledge.md)

</div>

<!-- RELATED:END -->

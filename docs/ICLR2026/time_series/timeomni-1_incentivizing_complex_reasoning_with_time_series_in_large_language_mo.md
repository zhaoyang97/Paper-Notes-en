---
title: >-
  [Paper Note] TimeOmni-1: Incentivizing Complex Reasoning with Time Series in Large Language Models
description: >-
  [ICLR 2026][Time Series][Time Series Reasoning] TimeOmni-1 proposes the first unified time series reasoning model, leveraging TSR-Suite (the first reasoning-oriented time series dataset suite) and a two-stage training paradigm (SFT for injecting temporal priors + RL for refining reasoning), achieving significant improvements over GPT-4.1 across multiple time series reasoning tasks.
tags:
  - ICLR 2026
  - Time Series
  - Time Series Reasoning
  - LLM
  - Reinforcement Learning
  - Multi-task Joint Training
  - Causal Discovery
date: 2026-05-08
content_hash: 49c6ff40efb5fc63
---

# TimeOmni-1: Incentivizing Complex Reasoning with Time Series in Large Language Models

**Conference**: ICLR 2026
**arXiv**: [2509.24803](https://arxiv.org/abs/2509.24803)
**Code**: [GitHub](https://github.com/AntonGuan/TimeOmni-1)
**Area**: Human Understanding / Time Series
**Keywords**: Time Series Reasoning, LLM, Reinforcement Learning, Multi-task Joint Training, Causal Discovery

## TL;DR

TimeOmni-1 proposes the first unified time series reasoning model, leveraging TSR-Suite (the first reasoning-oriented time series dataset suite) and a two-stage training paradigm (SFT for injecting temporal priors + RL for refining reasoning), achieving significant improvements over GPT-4.1 across multiple time series reasoning tasks.

## Background & Motivation

Time series understanding is transitioning from basic pattern analysis to advanced reasoning, yet two major bottlenecks persist:

**Scarcity of High-Quality Data**: Existing time series QA datasets (e.g., Time-MQA) remain at a superficial question-answering level, suffering from critical issues — (a) questions are overly simplistic, with negligible performance gaps between reasoning and non-reasoning models; (b) insufficient context forces models to guess rather than reason.

**Lack of Viable Reasoning Pathways**: It remains unclear which tasks genuinely require time series reasoning capabilities. Existing methods are confined to narrow tasks (e.g., TimeMaster trains six models on six datasets separately) without cross-task transferability.

The authors propose two core **design principles**:
- **Principle 1 — QA Must Reward Reasoning**: Reasoning models should substantially outperform non-reasoning models, i.e., $\bar{S}(M_{RM}) \gg \bar{S}(M_{NRM})$
- **Principle 2 — Context Must Be Sufficient**: Adequate time series input $X$ and auxiliary context $C$ must be provided to eliminate ambiguity.

## Method

### Overall Architecture

TimeOmni-1 adopts a **two-stage curriculum learning** framework:

- **Stage 1 (SFT)**: Injects time series reasoning priors — supervised fine-tuning with human-guided chain-of-thought (CoT), establishing three core temporal reasoning capabilities in the LLM: perception, extrapolation, and decision-making.
- **Stage 2 (RL)**: Refines reasoning — employs GRPO with task-customized reward functions to convert imitative priors into robust reasoning behavior.

### Key Designs

1. **TSR-Suite (Time Series Reasoning Suite)**: The first comprehensive time series reasoning dataset, covering 4 atomic tasks and 3 major reasoning capabilities:

    - **Perception**: Task 1 — Scene Understanding (attribution of a single series); Task 2 — Causal Discovery (causal relationships among multiple series)
    - **Extrapolation**: Task 3 — Event-Aware Forecasting (inferring future trends under event perturbations)
    - **Decision-making**: Task 4 — Decision Making (integrating perception and extrapolation for action selection)

   The suite contains 23K+ samples, of which 2.3K are carefully curated through a human-guided hierarchical annotation pipeline, spanning 10 domains.

2. **Hierarchical CoT Annotation Pipeline**: A three-step annotation process — (a) an LLM Analyzer generates reasoning chains under human-guided templates (Step-1 CoT); (b) human experts verify contextual sufficiency and author expert reasoning chains for LLM error cases (Step-2 CoT); (c) an LLM Rewriter normalizes the expert reasoning chains.

3. **Task-Customized RL Reward Design**:

    - Format reward $\mathcal{R}_{format}$: enforces the `<think></think><answer></answer>` structure
    - Discrete tasks (Tasks 1, 2, 4): exact-match accuracy $\mathcal{R}_{discrete} \in \{0,1\}$
    - Sequential task (Task 3): count reward $\mathcal{R}_{count}=0.1$ (for correct prediction sequence length) + exponential-decay-mapped normalized MAE reward

4. **Multi-task Joint Training**: All tasks are trained within a single unified model. Two complementary experiments validate cross-task gains:

    - **Progressive Capability Transfer**: Without directly training on the decision-making task, training on perception + extrapolation alone raises decision accuracy from 25.5% to 31.3%.
    - **Progressive Capability Supplementation**: Incrementally incorporating prerequisite tasks in joint training raises decision accuracy from 40.9% to 47.9%.

### Loss & Training

- Stage 1: Standard SFT loss (cross-entropy) using human-guided CoT data.
- Stage 2: GRPO (Group Relative Policy Optimization) with reward functions:
    - Tasks 1/2/4: $R = \mathcal{R}_{format} + \mathcal{R}_{discrete}$
    - Task 3: $R = \mathcal{R}_{format} + \mathcal{R}_{count} + \text{exp-decay}(\text{MAE})$

## Key Experimental Results

### Main Results

**ID/OOD evaluation across four tasks (ACC %, Task 3 reports MAE↓):**

| Method | Scene Understanding (ID) | Causal Discovery (ID) | Event Forecasting (ID/MAE) | Decision-Making (ID) |
|--------|--------------------------|----------------------|---------------------------|----------------------|
| GPT-4.1 | 85.5 | 28.7 | 13.79 | 25.5 |
| Qwen2.5-7B | 48.5 | 21.6 | 23.28 | 25.5 |
| Time-R1 | 30.9 | 30.2 | 17.61 | 27.8 |
| **TimeOmni-1** | **90.7** | **69.3** | **14.30** | **47.9** |

TimeOmni-1 surpasses GPT-4.1 by 40.6% on causal discovery (ID) and by 22.4% on the decision-making task.

### Ablation Study

| Configuration | Causal Discovery (ID) | Decision-Making (ID) | Note |
|---------------|-----------------------|----------------------|------|
| Base model | 21.6 | 25.5 | LLM lacks temporal priors |
| ANS-SFT (answer supervision) | 30.5 | 51.0 | Fits answer distribution only, no reasoning |
| CoT-SFT (Stage 1) | 67.7 | 40.9 | CoT injection significantly improves causal discovery |
| CoT-SFT+RL (Stage 2) | 69.3 | 47.9 | RL refinement yields further gains |
| Single-task training (CoT-SFT+RL) | 67.5 | 40.9 | Joint training outperforms single-task |

### Key Findings

- **LLMs Inherently Lack Temporal Reasoning Priors**: The base model achieves only 21.6% on causal discovery (near random chance at 33.3%), and RL alone cannot establish this capability.
- **Human-Guided Templates Are Critical**: GPT-4.1 achieves 28.7% on zero-shot causal discovery, rising to 71.1% with human-guided templates.
- **Joint Training Yields Mutual Benefits**: Cross-task joint training outperforms single-task training on all tasks, supporting a "train once, transfer across tasks" paradigm.
- **General Reasoning Capabilities Are Preserved**: TimeOmni-1 achieves an average accuracy improvement of 16.5% over the base model on general reasoning benchmarks including DROP, GPQA, and ReClor.
- **Success Rate (SR)**: TimeOmni-1 achieves SR ≥ 93.8% across all tasks, substantially outperforming existing time series-specific models (e.g., ChatTS achieves SR = 0% on event forecasting).

## Highlights & Insights

1. **Systematic Definition of Time Series Reasoning Tasks**: The first work to explicitly articulate "reasoning necessity" and "context sufficiency" as design principles, establishing a task framework that genuinely demands reasoning.
2. **Perception → Extrapolation → Decision-Making as a Progressive Capability Pathway**: Reflects the cognitive logic of "understand first, then predict, then act," demonstrating depth in task design.
3. **Complementary Roles of the Two-Stage Training**: SFT instills "knowing how to think," while RL refines "thinking more accurately" — both stages are indispensable.
4. **Empirical Evidence for Cross-Task Positive Transfer**: Through carefully designed progressive experiments, the work demonstrates that the three core temporal reasoning capabilities are intrinsically interconnected.

## Limitations & Future Work

1. **Limited Data Scale**: TSR-Suite contains only 23K samples (with 2.3K manually annotated), which is small compared to general NLP datasets.
2. **Narrow Task Coverage**: The 4 tasks focus predominantly on classification and forecasting, lacking more diverse reasoning tasks such as anomaly detection and trend interpretation.
3. **Persistent OOD Generalization Gap**: Event forecasting OOD MAE reaches 145.53 (vs. 14.30 in-domain), indicating that cross-domain generalization requires further improvement.
4. **Base Model Constraints**: Validation is limited to Qwen2.5-7B; scaling behavior with larger models remains unexplored.
5. **Reasoning Chain Quality Depends on Human Templates**: Data construction relies heavily on human-guided templates, limiting scalability.

## Related Work & Insights

- **Time-R1** is the closest existing time series reasoning model but is limited to classical forecasting; TimeOmni-1 extends the scope to multi-task reasoning.
- **DeepSeek-R1** demonstrates that RL can enhance reasoning capabilities; TimeOmni-1 introduces this paradigm to the time series domain.
- **Time-MQA** is large in scale but suffers from overly simple tasks and insufficient context; TSR-Suite is designed to address these shortcomings directly.
- The work offers important insights for the field of **time series intelligence**: general-purpose temporal models require the injection of reasoning priors rather than relying solely on pattern fitting.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First systematic construction of a time series reasoning task framework + first unified time series reasoning model
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Four-task ID/OOD evaluation + progressive experiments + ablation study + general capability assessment — highly comprehensive
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with design-principle-driven exposition; some figures and tables are overly dense in information
- **Value**: ⭐⭐⭐⭐⭐ Opens a new direction for time series reasoning; dataset, model, and code are fully open-sourced

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Reasoning on Time-Series for Financial Technical Analysis](reasoning_on_time-series_for_financial_technical_analysis.md)
- [\[ACL 2026\] Learning Uncertainty from Sequential Internal Dispersion in Large Language Models](../../ACL2026/time_series/learning_uncertainty_from_sequential_internal_dispersion_in_large_language_model.md)
- [\[NeurIPS 2025\] PlanU: Large Language Model Reasoning through Planning under Uncertainty](../../NeurIPS2025/time_series/planu_large_language_model_reasoning_through_planning_under_uncertainty.md)
- [\[ICLR 2026\] VoT: Event-Driven Reasoning and Multi-Level Alignment Unlock the Value of Text for Time Series Forecasting](unlocking_the_value_of_text_event-driven_reasoning_and_multi-level_alignment_for.md)
- [\[ICLR 2026\] EDINET-Bench: Evaluating LLMs on Complex Financial Tasks using Japanese Financial Statements](edinet-bench_evaluating_llms_on_complex_financial_tasks_using_japanese_financial.md)

<!-- RELATED:END -->

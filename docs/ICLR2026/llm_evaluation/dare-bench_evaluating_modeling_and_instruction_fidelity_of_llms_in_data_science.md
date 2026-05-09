---
title: >-
  [Paper Note] DARE-bench: Evaluating Modeling and Instruction Fidelity of LLMs in Data Science
description: >-
  [ICLR 2026][LLM Evaluation][data science benchmark] DARE-bench is a large-scale verifiable benchmark for data science tasks, comprising 6,300 Kaggle-derived tasks that support evaluation across two dimensions—ML modeling and instruction following—along with training data for SFT and RL. SFT improves Qwen3-32B by 1.83×, while RL improves Qwen3-4B by more than 8×.
tags:
  - ICLR 2026
  - LLM Evaluation
  - data science benchmark
  - instruction following
  - ML modeling
  - RLVR
  - LLM agent
date: 2026-05-08
content_hash: 626e1966ef39c3e8
---

# DARE-bench: Evaluating Modeling and Instruction Fidelity of LLMs in Data Science

**Conference**: ICLR 2026
**arXiv**: [2602.24288](https://arxiv.org/abs/2602.24288)
**Code**: [https://github.com/Snowflake-Labs/dare-bench](https://github.com/Snowflake-Labs/dare-bench)
**Area**: LLM Evaluation
**Keywords**: data science benchmark, instruction following, ML modeling, RLVR, LLM agent

## TL;DR
DARE-bench is a large-scale verifiable benchmark for data science tasks, comprising 6,300 Kaggle-derived tasks that support evaluation across two dimensions—ML modeling and instruction following—along with training data for SFT and RL. SFT improves Qwen3-32B by 1.83×, while RL improves Qwen3-4B by more than 8×.

## Background & Motivation

**Background**: LLMs are increasingly deployed as data science agents (data loading, transformation, and modeling), yet existing benchmarks (DS-1000, DSBench, MLE-bench, etc.) suffer from significant shortcomings: most evaluate only final answer accuracy while ignoring process fidelity; verifiable ground truth is absent; task scales are small (hundreds of instances); and no training data is provided.

**Limitations of Prior Work**: (a) Lack of standardized process-aware evaluation—whether the agent truly follows the specified DS pipeline; (b) scarcity of accurately annotated training data, limiting applicability of SFT and RL; (c) existing benchmarks are predominantly sourced from Kaggle competitions, resulting in narrow domain coverage (e.g., missing time-series forecasting).

**Key Challenge**: Evaluating process fidelity requires deterministic ground truth, yet DS tasks are inherently stochastic and environment-dependent. How can process evaluation be made verifiable?

**Goal**: (a) Construct a large-scale, verifiable, training-ready DS benchmark; (b) cover two complementary capabilities—instruction following and ML modeling; (c) support RLVR (reinforcement learning with verifiable rewards) training.

**Key Insight**: The high reproducibility of data science workflows is leveraged—by controlling random seeds and providing explicit instructions, faithful execution of a process yields deterministic outputs, enabling outcome-based verification of process fidelity.

**Core Idea**: Through engineered determinism (fixed seeds + sandboxed execution + reference solutions + verifiable ground truth), DS process evaluation is transformed into automatically verifiable, outcome-based assessment.

## Method

### Overall Architecture
DARE-bench encompasses three task families (classification, regression, and time-series forecasting), each with two variants. The evaluation pipeline proceeds as follows: given a natural-language problem and structured input files, an LLM generates and executes code within a sandbox to produce predictions, which are then automatically scored against the ground truth. Dataset construction follows a four-stage automated pipeline: data sourcing → LLM-assisted task design → post-processing → sandbox validation.

### Key Designs

1. **Dual-Variant Task Design (IF + MM / XF + CF)**:

    - Function: Each dataset yields two complementary evaluation tasks.
    - **IF (Instruction Following)**: Requires the LLM to faithfully reproduce a reference workflow (specifying model type, hyperparameters, and preprocessing steps), evaluating process fidelity. Fixing random seeds ensures that faithful execution produces a uniquely determined output.
    - **MM (ML Modeling)**: Imposes no methodological constraints and evaluates only final prediction performance, assessing ML modeling capability.
    - **XF/CF (Time-Series)**: XF retains exogenous features; CF retains only timestamps and entity columns (the classical forecasting setting).
    - Design Motivation: IF simulates the scenario of "strictly executing a data scientist's prescribed pipeline," while MM simulates "the client cares only about final performance." The two variants are complementary and both reflect real-world needs.

2. **Automated Data Construction Pipeline**:

    - Function: Automatically constructs standardized ML tasks from Kaggle datasets.
    - Core Procedure: (1) Dataset Sourcing—retrieves Kaggle datasets and metadata via API and web scraping; (2) LLM-Assisted Task Design—LLM determines whether the dataset supports a predictive task and identifies target columns and features; (3) Post-Processing—data splitting, noise injection for IF tasks, resampling and entity validation for time-series tasks; (4) Finalization—sandbox validation to confirm task solvability.
    - Design Motivation: Overcomes the bottleneck of manual annotation; LLMs handle only auxiliary content (descriptions, metadata, rule extraction) and do not generate the training signal itself.

3. **Verifiable Reward Design**:

    - Function: Enables DARE-bench to support RLVR training.
    - Mechanism: Ground truth for IF tasks is derived from the execution output of reference solutions under fixed seeds; ground truth for MM tasks comes from original dataset labels. Both are deterministic numerical values amenable to automatic scoring.
    - Design Motivation: This is the critical enabler for RL training—no human judgment or LLM-as-judge is required; verification is purely outcome-based.

### Loss & Training
- SFT: Qwen3-32B/4B is fine-tuned on the DARE-bench training split.
- RL: The GRPO algorithm is applied with DARE-bench verifiable rewards, without requiring preference data.
- Evaluation Metrics: accuracy/F1 for classification, R²/RMSE for regression, SMAPE/MAE for time-series forecasting.

## Key Experimental Results

### Main Results

| Model | Baseline Score | SFT Score | RL Score | Gain |
|-------|---------------|-----------|----------|------|
| gpt-o4-mini | ~45 | - | - | Strongest closed-source model still struggles with ML modeling |
| Qwen3-32B | 23.25 | ~42.5 (1.83×) | - | Substantial SFT improvement |
| Qwen3-4B | 4.39 | ~25 | **37.40 (8.5×)** | Remarkable RL gain |

### Ablation Study

| Task Type | Key Finding |
|-----------|-------------|
| Classification-IF | Failure to follow seed/hyperparameter instructions is the primary cause of failure |
| Classification-MM | Under open-ended modeling, LLMs frequently select suboptimal models |
| Time-series-CF | Most challenging subtask; even strong models perform poorly |
| RL vs. SFT | RL substantially outperforms SFT on small models; SFT suffices for large models |

### Key Findings
- **Even gpt-o4-mini struggles on ML modeling tasks**, indicating that the data science capabilities of current LLMs are far from mature.
- **Instruction following is the primary bottleneck**: models frequently deviate from specified procedures (ignoring designated seeds, altering preprocessing steps, etc.), causing IF task failures.
- **RL yields exceptionally large gains on small models**: Qwen3-4B improves from 4.39 to 37.40 (8.5×), demonstrating that verifiable rewards combined with RL are highly effective for enhancing DS agent capabilities.
- **Time-series forecasting is the greatest weakness**: all models perform worst on the CF variant, suggesting that LLMs lack deep knowledge of time-series modeling.

## Highlights & Insights
- **Transforming process fidelity into outcome-based evaluation**: By exploiting the reproducibility of data science workflows, "whether the process is correct" is converted to "whether deterministic outputs match," elegantly resolving the objectivity problem in process evaluation.
- **Unified training and evaluation**: 95% of the 6,300 tasks are available for training and 5% for testing, genuinely supporting a "train on benchmark" paradigm.
- **Dramatic RL gains on small models**: An 8× improvement demonstrates that task-specific RL training can substantially unlock the potential of small models, with significant practical implications for DS agent deployment.
- **Scale far exceeding comparable benchmarks**: 6,300 tasks versus the largest comparable benchmark SWE-bench (21K tasks, but not DS-specific), representing an order-of-magnitude improvement within the DS evaluation landscape.

## Limitations & Future Work
- Kaggle datasets may not represent the complexity of real-world industrial data science problems.
- The "determinism" of IF tasks relies on complete control of random seeds; many real-world DS pipelines cannot be fully reproduced.
- Multi-turn interaction and iterative modeling scenarios are not evaluated—actual DS workflows typically require multiple experimental iterations.
- The choice of evaluation metrics for time-series tasks (e.g., SMAPE vs. MAE) may affect model rankings.

## Related Work & Insights
- **vs. DSBench/MLE-bench**: Larger scale (6,300 vs. 540/75 tasks), includes training data, and covers time-series forecasting.
- **vs. SWE-bench**: SWE-bench targets software engineering; DARE-bench targets data science—the two are strongly complementary.
- **vs. RLVR (DeepSeek-R1, etc.)**: DARE-bench provides a new RLVR domain beyond mathematics and code—verifiable rewards for DS tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ The IF+MM dual-variant evaluation and DS process fidelity verification constitute novel contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-model evaluation with SFT and RL experiments across 6,300 tasks at considerable scale.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with detailed pipeline descriptions.
- Value: ⭐⭐⭐⭐ Fills an important gap in DS agent training and evaluation; RL training results are particularly compelling.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] TabStruct: Measuring Structural Fidelity of Tabular Data](tabstruct_measuring_structural_fidelity_of_tabular_data.md)
- [\[ICLR 2026\] Truthfulness Despite Weak Supervision: Evaluating and Training LLMs Using Peer Prediction](truthfulness_despite_weak_supervision_evaluating_and_training_llms_using_peer_pr.md)
- [\[AAAI 2026\] Benchmarking LLMs for Political Science: A United Nations Perspective](../../AAAI2026/llm_evaluation/benchmarking_llms_for_political_science_a_united_nations_perspective.md)
- [\[NeurIPS 2025\] On Evaluating LLM Alignment by Evaluating LLMs as Judges](../../NeurIPS2025/llm_evaluation/on_evaluating_llm_alignment_by_evaluating_llms_as_judges.md)
- [\[NeurIPS 2025\] RDB2G-Bench: A Comprehensive Benchmark for Automatic Graph Modeling of Relational Databases](../../NeurIPS2025/llm_evaluation/rdb2g-bench_a_comprehensive_benchmark_for_automatic_graph_modeling_of_relational.md)

<!-- RELATED:END -->

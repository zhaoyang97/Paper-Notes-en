---
title: >-
  [Paper Note] Scaling Generalist Data-Analytic Agents
description: >-
  [ICLR 2026][LLM Reasoning][Data-Analytic Agent] This paper proposes DataMind — a complete training framework for data-analytic agents — achieving diverse query synthesis via fine-grained task taxonomy with recursive diff…
tags:
  - "ICLR 2026"
  - "LLM Reasoning"
  - "Data-Analytic Agent"
  - "Agent Training"
  - "Multi-turn Code Execution"
  - "Data Synthesis"
  - "SFT+RL"
date: 2026-05-08
content_hash: c5a156f07cc058c0
---

# Scaling Generalist Data-Analytic Agents

**Conference**: ICLR 2026
**arXiv**: [2509.25084](https://arxiv.org/abs/2509.25084)  
**Code**: [GitHub](https://github.com/zjunlp/DataMind)  
**Area**: LLM Reasoning
**Keywords**: Data-Analytic Agent, Agent Training, Multi-turn Code Execution, Data Synthesis, SFT+RL

## TL;DR

This paper proposes DataMind — a complete training framework for data-analytic agents — achieving diverse query synthesis via fine-grained task taxonomy with recursive difficulty composition, ensuring data quality through knowledge-augmented trajectory sampling and self-consistency filtering, employing a dynamic SFT+RL mixed training strategy, and implementing a memory-efficient asynchronous rollout framework. The resulting DataMind-14B achieves a 71.16% average score across multiple benchmarks, establishing a new state of the art and surpassing GPT-5 and DeepSeek-V3.1.

## Background & Motivation

**Background**: Data-analytic agents discover actionable insights by generating code to process, model, and compute over data, serving as key catalysts for AI-driven scientific discovery and automated decision support. Existing data-analytic agents (DS-Agent, AutoKaggle, Data Interpreter, etc.) rely almost entirely on closed-source models through prompt engineering and multi-agent scaffolding.

**Limitations of Prior Work**:
- **Insufficient training data**: Publicly available data analysis benchmarks provide only limited test sets without step-by-step trajectory annotations, making them unsuitable for direct training use.
- **Unclear training strategy**: How to allocate steps and maintain stability under the conventional SFT-then-RL paradigm for long-horizon agent training remains an open question.
- **Unstable multi-turn code execution**: Data files and code interpreters involve complex memory management; parallel agent rollout combined with multi-turn code generation frequently crashes under limited memory resources.
- **Capability gap in open-source models**: The few existing open-source trained models (TableLLM, Table-R1) handle only simple table understanding tasks and fail when faced with large-scale data files in diverse formats and long-horizon multi-step reasoning.

**Key Challenge**: High-quality training demands large-scale diverse trajectory data, a stable training strategy, and reliable environment interaction — yet all three present unique challenges in data analysis settings, where task formats are heterogeneous (csv/xlsx/sqlite), reasoning chains are long, and code execution carries side effects.

**Goal**: The paper proposes DataMind, an end-to-end scalable data synthesis and agent training framework that systematically addresses the three challenges above.

## Method

### Overall Architecture

The DataMind pipeline consists of four key modules:
1. **Data Synthesis**: File collection → task taxonomy → query synthesis → trajectory sampling & filtering → DataMind-12K
2. **Training Strategy**: Dynamically weighted joint optimization of SFT loss and DAPO (RL) loss
3. **Rollout Engineering**: Asynchronous interaction + chunked code maintenance + secure sandbox
4. **Reward Design**: Format reward + answer reward (model-as-judge) + length penalty

Agents follow the ReAct paradigm: Thought → Action (Python/SQL code) → Observation (execution feedback), with a maximum of $\mathcal{T}=10$ turns.

### Key Design 1: Fine-Grained Task Taxonomy and Recursive Difficulty Composition

- Data analysis tasks are categorized into **18 fine-grained classes** (data cleaning, descriptive statistics, correlation analysis, time-series analysis, anomaly detection, etc.), each accompanied by 4–6 exemplar queries as few-shot demonstrations.
- **Recursive easy-to-hard composition**: The output of one task type is used as input to the next, iterated 2–5 times to progressively escalate difficulty and create multi-hop analytical challenges that far exceed the demands of any single task type.
- Data files are sourced from Kaggle (3,400 csv + 560 xlsx) and BIRD/OmniSQL (1,954 sqlite).

### Key Design 2: Knowledge-Augmented Trajectory Sampling and Self-Consistency Filtering

Trajectory sampling employs a two-level quality assurance scheme:

**Level 1 — Knowledge-Augmented Sampling**:
- High-level workflow knowledge $k$ is manually designed for each task category to guide an expert model (DeepSeek-V3.1) in generating trajectories.
- $\mathcal{N}=3$ independent trajectories are sampled per query.

**Level 2 — Self-Consistency Filtering**:
- A judge model (GPT-4o-mini) verifies whether the final answers across $\mathcal{N}$ trajectories are consistent.
- Among consistent trajectories, the most concise and accurate one is selected as the training instance.
- Inconsistent trajectories are returned to the agent along with the judge's chain-of-thought feedback for reflective revision, followed by re-filtering.
- Rule-based filtering enforces format compliance, length constraints (answer < 1,024 tokens), and linguistic integrity.

The final dataset is **DataMind-12K** (11,707 high-quality trajectories).

### Key Design 3: Dynamic Mixed SFT+RL Training

The conventional SFT-then-RL paradigm presents a dilemma: excessive SFT rigidifies reasoning patterns and suppresses RL exploration; premature RL leaves the model incapable of generating effective rollout groups.

This work adopts joint optimization:

$$\mathcal{L}_{\text{Final}}(\theta) = \gamma \cdot \mathcal{L}_{\text{SFT}}(\theta) + (1-\gamma) \cdot \mathcal{L}_{\text{DAPO}}(\theta)$$

- $\gamma$ is **dynamically scheduled**: initialized at a large value (0.9) to enable knowledge absorption from expert data, then gradually annealed to a small value (0.05) to encourage exploration.
- The SFT loss is computed only on tokens generated by the agent, masking environment feedback tokens.
- The DAPO algorithm is used for RL, employing decoupled clipping and dynamic sampling.
- Training begins with a cold start on DataMind-12K.

**Void Turns Filtering**: Loss is masked for entire trajectories containing invalid turns (i.e., turns that produce neither effective code nor an answer), preventing trajectory collapse caused by distributional drift.

### Key Design 4: Asynchronous Multi-Turn Rollout Engineering

- **Asynchronous interaction**: Model generation and code execution are decoupled across different samples, preventing simultaneous GPU and CPU memory spikes.
- **Chunked code maintenance**: Following a notebook-style paradigm, each step generates only the current code snippet; prior snippets are concatenated at execution time, avoiding the memory overhead of maintaining a global variable pool.
- **Safety controls**: Each trajectory runs in an isolated environment with CPU time and peak memory limits, and unsafe function calls are filtered.

## Key Experimental Results

### Main Results: Multi-Benchmark Performance Comparison

| Model Type | Method | DABench pass@1 | TableBench pass@1 | BIRD pass@1 | Avg pass@1 |
|-----------|--------|---------------|-------------------|-------------|------------|
| **Closed-source** | GPT-4o | 76.39 | 64.97 | 50.20 | 63.85 |
| **Closed-source** | o4-mini | 79.12 | 71.03 | 57.04 | 69.06 |
| **Closed-source** | DeepSeek-R1 | 78.73 | 68.96 | 55.80 | 67.83 |
| **Closed-source** | DeepSeek-V3.1 | 81.32 | 72.52 | 57.89 | 70.58 |
| **Closed-source** | GPT-5 | 78.21 | 69.93 | 60.17 | 69.44 |
| Open-source-7B | ReAct (Qwen-Coder-7B) | 15.05 | 11.70 | 7.02 | 11.26 |
| Open-source-7B | TableLLM | 36.71 | 41.01 | 11.99 | 29.90 |
| Open-source-7B | Table-R1 | 42.54 | 56.36 | 10.69 | 36.53 |
| Open-source-7B | **DataMind-7B** | **77.30** | **67.60** | **59.41** | **68.10** |
| Open-source-14B | ReAct (Qwen-Coder-14B) | 71.21 | 56.96 | 41.76 | 56.64 |
| Open-source-14B | TableLLM | 38.26 | 46.44 | 20.99 | 35.23 |
| Open-source-14B | **DataMind-14B** | **80.29** | **70.95** | **62.23** | **71.16** |

Key findings:
- DataMind-14B achieves a 71.16% average score, **surpassing all closed-source models** (including GPT-5 at 69.44% and DeepSeek-V3.1 at 70.58%).
- DataMind-7B achieves 68.10%, the best among all open-source models.
- Specialized models (OmniSQL/SQL-R1) remain competitive on BIRD but exhibit sharp performance drops on other benchmarks.
- DataMind is trained on only 12K samples, far fewer than baselines (TableLLM: 20K; OmniSQL: 2.5M).

### Ablation Study: Training Strategy Comparison

| Training Strategy | Avg pass@1 | Avg pass@3 |
|------------------|------------|------------|
| SFT only | 62.54 | 73.74 |
| zero-RL (no SFT) | 58.03 | 71.72 |
| SFT-then-RL | 63.42 | 75.46 |
| **SFT-and-RL (dynamic $\gamma$)** | **68.10** | **79.07** |

Key insights:
- SFT alone improves the baseline from 11.26% to 62.54% — **data quality accounts for the majority of the performance gain**.
- zero-RL underperforms SFT alone — the 7B model's limited multi-step reasoning capacity prevents it from independently producing high-quality rollout trajectories.
- SFT-then-RL yields only marginal improvement and is prone to training instability.
- The dynamic mixed strategy yields an additional 5.56-point improvement by balancing knowledge absorption with exploration.

### Data and Filtering Analysis

| Filtering Strategy | Effect |
|-------------------|--------|
| Con-select (self-consistency + best selection) | Baseline setting |
| Non-select (retain all consistent trajectories) | Superior on DABench — trajectory diversity is more beneficial |
| Random-select (randomly choose a consistent trajectory) | Comparable to con-select — judge preference may reduce diversity |
| Non-con (no consistency filtering) | **Significant degradation across all metrics** — answer quality is the critical guarantee for trajectory quality |

Core finding: **Self-consistency filtering is more critical than best-trajectory selection** — answer correctness ensures the intrinsic quality of trajectories, and diverse reasoning paths are more beneficial for model learning than a single "best" path.

## Highlights & Insights

### Strengths
- **End-to-end engineering completeness**: The system encompasses data synthesis, training strategy, and rollout engineering, with independent innovations at each stage.
- **Deep insights**: Findings such as the SFT loss serving simultaneously as a stabilizer and a potential source of collapse during RL training, and self-consistency filtering being more important than best-trajectory selection, carry strong practical guidance.
- **Compelling results**: A 14B model trained on only 12K samples outperforms GPT-5 and specialized models trained on 2.5M samples.
- **Training dynamics analysis**: The "raising a child" analogy provides an intuitive explanation of the dynamic weight scheduling from SFT to RL.

### Limitations & Future Work
- Evaluation relies on GPT-4o-mini as the judge; using the same judge for both training and evaluation introduces potential bias (though cross-validation reports a Pearson correlation of 0.96).
- The 18-category task taxonomy is manually designed; category boundaries and coverage may have omissions.
- Experiments are conducted only at the 7B and 14B scales; behavior at larger or smaller model sizes remains unknown.
- The chunked code maintenance strategy may become less efficient in scenarios with long dependency chains (e.g., cross-turn variable references).

### Rating
⭐⭐⭐⭐⭐ — An exemplary systematic engineering contribution: the problem is clearly defined, the solution is comprehensive, experiments are rigorous, and insights are deep. This work offers strong reference value to the agent training community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Understanding the Role of Training Data in Test-Time Scaling](understanding_the_role_of_training_data_in_test-time_scaling.md)
- [\[ICLR 2026\] Agentified Assessment of Logical Reasoning Agents](agentified_assessment_of_logical_reasoning_agents.md)
- [\[ICLR 2026\] Estimating the Empowerment of Language Model Agents](estimating_the_empowerment_of_language_model_agents.md)
- [\[ICLR 2026\] DESIGNER: Design-Logic-Guided Multidisciplinary Data Synthesis for LLM Reasoning](designer_design-logic-guided_multidisciplinary_data_synthesis_for_llm_reasoning.md)
- [\[ICLR 2026\] Adaptive Social Learning via Mode Policy Optimization for Language Agents](adaptive_social_learning_via_mode_policy_optimization_for_language_agents.md)

</div>

<!-- RELATED:END -->

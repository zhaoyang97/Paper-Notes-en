---
title: >-
  [Paper Note] Jupiter: Enhancing LLM Data Analysis Capabilities via Notebook and Inference-Time Value-Guided Search
description: >-
  [AAAI 2026][LLM Reasoning][Data Analysis] This work constructs the NbQA dataset (38K task-solution pairs extracted from real Jupyter Notebooks) and proposes the Jupiter framework (modeling data analysis as a state-level…
tags:
  - "AAAI 2026"
  - "LLM Reasoning"
  - "Data Analysis"
  - "Jupyter Notebook"
  - "Value-Guided Search"
  - "MCTS"
  - "Test-Time Compute"
date: 2026-05-08
content_hash: 9cf97680be5c9db8
---

# Jupiter: Enhancing LLM Data Analysis Capabilities via Notebook and Inference-Time Value-Guided Search

**Conference**: AAAI 2026
**arXiv**: [2509.09245](https://arxiv.org/abs/2509.09245)  
**Code**: [https://github.com/microsoft/Jupiter](https://github.com/microsoft/Jupiter)  
**Area**: LLM Reasoning
**Keywords**: Data Analysis, Jupyter Notebook, Value-Guided Search, MCTS, Test-Time Compute

## TL;DR
This work constructs the NbQA dataset (38K task-solution pairs extracted from real Jupyter Notebooks) and proposes the Jupiter framework (modeling data analysis as a state-level search problem with PUCT search guided by a value model), enabling Qwen2.5-14B to achieve 86.38% on InfiAgent-DABench, surpassing GPT-4o (85.99%), and improving Qwen2.5-7B on DSBench from 63.51% to 89.19%.

## Background & Motivation

**Background**: LLMs are increasingly applied to data analysis tasks, yet existing approaches either focus on isolated analysis stages or rely on proprietary models (e.g., GPT-4o) and complex agent architectures (AutoGen, TaskWeaver). Open-source models perform poorly on multi-step data analysis.

**Limitations of Prior Work**:
- **Data scarcity**: Large-scale, high-quality training data for data analysis is lacking. Existing datasets are mostly synthesized by LLMs, limiting quality and diversity.
- **Weak reasoning capability**: Data analysis requires multi-step reasoning and tool use; LLMs frequently err in complex analysis chains and cannot self-correct.
- **Agent systems depend on closed-source models**: Frameworks such as AutoGen achieve strong results but require GPT-4-level models.

**Key Challenge**: Open-source small models (7B–14B) fall far short of closed-source large models on data analysis, due to the lack of high-quality training data and effective inference-time search strategies.

**Goal**: (1) Construct high-quality training data from real Notebooks; (2) Design inference-time value-guided search to improve multi-step data analysis accuracy.

**Key Insight**: Jupyter Notebooks naturally encode a multi-step structure of question–code–execution result, making them an ideal training source for data analysis. Meanwhile, the success of MCTS combined with value models on reasoning tasks can be transferred to the data analysis setting.

**Core Idea**: Real Notebook data + value-model-guided inference-time search = open-source small models matching GPT-4o on data analysis.

## Method

### Overall Architecture
Two major components: (1) **NbQA Dataset Construction** — extracting high-quality task-solution pairs from 1.6M real Jupyter Notebooks; (2) **Jupiter Inference Framework** — modeling data analysis as a search tree, using MCTS to collect trajectories for training a value model, and applying the value model to guide search at inference time.

### Key Designs

1. **NbQA Dataset Construction Pipeline**:

    - **Function**: Extract standardized data analysis task-solution pairs from real Jupyter Notebooks.
    - **Mechanism**: Coarse filtering (valid structure, successful execution, sufficient complexity) → GPT-4o mini quality scoring (1–5) → GPT-4o fine-grained extraction (standardized format) → GPT-4o mini automatic review. The final dataset contains 38,635 task pairs, of which 6,845 include complete data dependencies.
    - Covers 8 task categories: statistical summarization, distribution analysis, correlation analysis, anomaly detection, data preprocessing, feature engineering, machine learning, and visualization.
    - **Design Motivation**: Real Notebook solutions are more diverse and practical than LLM-synthesized ones, directly reflecting the actual working patterns of data scientists.

2. **Jupiter Search Framework**:

    - **Function**: Model data analysis as a state-level search problem and construct a search tree for exploration.
    - **Mechanism**: The root node represents the original problem; each node represents a Notebook state (accumulated thought-action pairs with execution results); leaf nodes represent errors or candidate answers. Node selection follows the PUCT formula: $\text{PUCT}(s') = Q(s') + c_{\text{puct}} \cdot P(s') \cdot \frac{\sqrt{N_{\text{parent}}}}{1+N(s')}$. During training, MCTS ($c_{\text{puct}}>0$) is used to collect diverse trajectories; during inference, the exploration term is disabled ($c_{\text{puct}}=0$), relying solely on the value model for guidance.
    - **Design Motivation**: Data analysis is a sequential decision-making problem; the search tree enables systematic exploration of diverse analysis paths.

3. **Value Model Training**:

    - **Function**: Train a regression head to predict the Q-value of each Notebook state.
    - **Mechanism**: A regression head is appended to the fine-tuned LLM to output a scalar in $[-1, 1]$. Normalized Q-values from MCTS-collected trajectories serve as supervision signals. Training runs for 3 epochs with batch size 4 and learning rate $1\text{e-}4$.
    - **Design Motivation**: The value model compresses the knowledge acquired during MCTS — namely, which paths are more likely to succeed — into an evaluation function, eliminating the need for full MCTS at inference time.

### Loss & Training
- **SFT stage**: The base model is supervised fine-tuned on NbQA data.
- **Value model training**: Regression loss; input is the full multi-turn dialogue; output is the normalized Q-value from MCTS trajectories.
- GRPO-based RL was attempted but underperformed SFT, likely due to sparse rewards and issues with rule-based correctness checking.

## Key Experimental Results

### Main Results

InfiAgent-DABench (40 iterations):

| Model | Framework | Accuracy |
|-------|-----------|----------|
| GPT-4o | TaskWeaver | 85.99% |
| GPT-4o | ReAct | 81.32% |
| Qwen2.5-72B | ReAct | 75.88% |
| **Qwen2.5-14B (SFT)** | **Jupiter** | **86.38%** |
| **Qwen2.5-7B (SFT)** | **Jupiter** | **77.82%** |

DSBench Data Modeling:

| Model | Success Rate |
|-------|-------------|
| GPT-4 (AutoGen) | 87.84% |
| Qwen2.5-7B (Jupiter) | **89.19%** |
| Qwen2.5-14B (Jupiter) | **98.65%** |

### Ablation Study

| Configuration | Qwen2.5-7B Acc | Qwen2.5-14B Acc |
|---------------|----------------|-----------------|
| w/o VM, w/o ExpTerm | 70.04% | 79.38% |
| w/o VM, with ExpTerm | 68.87% | 75.88% |
| with VM, with ExpTerm | 68.87% | 74.71% |
| **with VM, w/o ExpTerm** | **77.82%** | **86.38%** |

### Key Findings
- **Value model is critical**: Including versus excluding the value model yields a 7–8 point difference.
- **Exploration term is harmful**: Disabling the exploration term improves accuracy by 8–11 percentage points, as the search space for data analysis is sparse and precise guidance outperforms blind exploration.
- **NbQA data quality is high**: SFT alone improves Mistral-7B by 56.81% (from 2.33% to 59.14%), demonstrating the exceptional value of real Notebook data.
- **Generalizes to mathematical reasoning**: On the AIME math competition, Qwen2.5-7B improves from 0% to 33.3% (OR) / 20% (voting).

## Highlights & Insights
- **Real Notebooks > Synthetic Data**: Training data refined from 1.6M real Jupyter Notebooks substantially outperforms LLM-synthesized data, as it reflects genuine data scientist workflows.
- **Disabling exploration at inference time**: Unlike AlphaGo, data analysis tasks have a sparse search space; once a strong value model is available, greedy search outperforms exploratory search.
- **Open-source small models outperform closed-source large models**: The 14B model surpasses GPT-4o, demonstrating that high-quality data combined with effective search can compensate for the gap in model scale.

## Limitations & Future Work
- **RL training underperforms**: GRPO-based RL falls short of SFT, possibly because correctness-checking rules are too coarse (partial correctness receives no reward).
- **Limited value model generalization**: The value model is effective within its training domain (data analysis), but cross-domain effectiveness (e.g., mathematical reasoning) remains uncertain.
- **High search overhead**: Forty iterations of search incur substantial inference-time cost, motivating the need for more efficient search strategies.
- **Narrow task coverage**: Only structured data analysis is addressed; unstructured data processing, deep learning, and related tasks are excluded.

## Related Work & Insights
- **vs. Data Interpreter**: Data Interpreter is a complex agent system, whereas Jupiter is a streamlined search + value model framework that achieves superior results on smaller models.
- **vs. AlphaMath/MCTS-RL**: In mathematical reasoning, MCTS exploration discovers novel solution paths; in data analysis, disabling exploration yields better performance — the nature of the task determines the appropriate search strategy.
- **vs. ReAct**: ReAct performs single-trajectory reasoning without search; Jupiter systematically explores multiple analysis paths via a search tree.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of Notebook data extraction and value-guided search constitutes a practical innovation, though individual components are not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Two benchmarks, multiple model sizes, detailed ablations, hyperparameter analysis, and cross-domain generalization tests.
- **Writing Quality**: ⭐⭐⭐⭐ Method description is clear and the data construction pipeline is detailed, though the paper is lengthy.
- **Value**: ⭐⭐⭐⭐⭐ Both the NbQA dataset and the Jupiter framework are open-sourced, representing significant contributions to the data analysis automation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Trade-offs in Large Reasoning Models: An Empirical Analysis of Deliberative and Adaptive Reasoning over Foundational Capabilities](trade-offs_in_large_reasoning_models_an_empirical_analysis_of_deliberative_and_a.md)
- [\[NeurIPS 2025\] SolverLLM: Solving Optimization Problems via Test-Time Scaling with LLM-Guided Search](../../NeurIPS2025/llm_reasoning/solverllm_leveraging_test-time_scaling_for_optimization_problem_via_llm-guided_s.md)
- [\[ICLR 2026\] DESIGNER: Design-Logic-Guided Multidisciplinary Data Synthesis for LLM Reasoning](../../ICLR2026/llm_reasoning/designer_design-logic-guided_multidisciplinary_data_synthesis_for_llm_reasoning.md)
- [\[AAAI 2026\] Evaluating, Synthesizing, and Enhancing for Customer Support Conversation](evaluating_synthesizing_and_enhancing_for_customer_support_conversation.md)
- [\[AAAI 2026\] RPM-MCTS: Knowledge-Retrieval as Process Reward Model with Monte Carlo Tree Search for Code Generation](rpm-mcts_knowledge-retrieval_as_process_reward_model_with_monte_carlo_tree_searc.md)

</div>

<!-- RELATED:END -->

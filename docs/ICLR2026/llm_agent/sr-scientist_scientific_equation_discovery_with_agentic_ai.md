---
title: >-
  [Paper Note] SR-Scientist: Scientific Equation Discovery With Agentic AI
description: >-
  [ICLR 2026][LLM Agent][symbolic regression] This paper proposes the SR-Scientist framework, which elevates LLMs from simple equation proposers to autonomous AI scientists. By leveraging a code interpreter tool for data a…
tags:
  - "ICLR 2026"
  - "LLM Agent"
  - "symbolic regression"
  - "agentic AI"
  - "equation discovery"
  - "reinforcement-learning"
  - "scientific discovery"
date: 2026-05-08
content_hash: 8b6498f776eda8c3
---

# SR-Scientist: Scientific Equation Discovery With Agentic AI

**Conference**: ICLR 2026
**arXiv**: [2510.11661](https://arxiv.org/abs/2510.11661)
**Code**: [GitHub](https://github.com/GAIR-NLP/SR-Scientist)
**Area**: LLM Agent
**Keywords**: symbolic regression, agentic AI, equation discovery, reinforcement-learning, scientific discovery

## TL;DR

This paper proposes the SR-Scientist framework, which elevates LLMs from simple equation proposers to autonomous AI scientists. By leveraging a code interpreter tool for data analysis and equation evaluation, the framework autonomously discovers scientific equations through long-horizon interactions, with reinforcement learning further enhancing its capabilities.

## Background & Motivation

Symbolic Regression (SR) aims to discover interpretable mathematical expressions from observational data and is a foundational task in scientific discovery. Existing approaches fall into three main categories:
- **Genetic Programming (GP) methods**: e.g., PySR and GPLearn, which perform combinatorial search over expression trees
- **Deep learning methods**: e.g., E2E, NeSymReS, and DSR, which train neural networks to map numerical data to expressions
- **LLM-augmented methods**: e.g., LLM-SR and LaSR, which embed LLMs within GP pipelines as equation proposers

Key limitations of existing LLM-based approaches include:
1. LLMs serve only as equation generators within fixed pipelines, **lacking autonomy**
2. They cannot directly analyze observational data via tools to extract insights
3. Most works focus solely on inference and do not explore **self-improvement** through methods such as RL

The core motivation of this paper is to build a scientific discovery framework centered on Agentic AI, transforming the LLM from a passive tool into an autonomous agent capable of driving the entire discovery lifecycle.

## Method

### Overall Architecture

The inference framework of SR-Scientist adopts an iterative design (Algorithm 1):

1. **Each iteration** sets a precision target $G_i$ (based on MAPE)
2. The LLM agent operates under the ReAct framework, alternating between reasoning and tool invocation: $(r_1, \mathcal{T}_1, o_1), (r_2, \mathcal{T}_2, o_2), \ldots$
3. An Experience Buffer propagates the best equations across iterations
4. Upon meeting the stopping criterion, the best equation is submitted

The objective function uses MAPE (Mean Absolute Percentage Error):

$$\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - f(\mathbf{x}_i)}{y_i} \right|$$

### Key Designs

**Tool Design**: The code interpreter is encapsulated into two core tools:
- **Data Analyzer $T_1$**: Linked to observational data, allowing the agent to write code for statistical analysis, residual analysis, and various forms of data exploration
- **Equation Evaluator $T_2$**: Accepts equation skeletons with constant placeholders, internally optimizes constants via the BFGS algorithm, and reports performance

**Experience Buffer**: Maintains $E = \{(e_i, s_i)\}_{i=1}^{N}$ to record explored equations and their MAPE scores. At the start of each iteration, the top-$K$ equations are retrieved as in-context examples, elegantly circumventing LLM context length constraints.

**Long-Horizon Optimization**: Each iteration allows the agent up to $M=25$ rounds of interaction (exceeding 20 turns), providing sufficient time for data analysis and equation refinement.

### Loss & Training

The training framework adopts the GRPO algorithm with a log-linear reward function:

$$\mathcal{R} = \text{clip}\left(\frac{\lg s_{\max} - \lg s}{\lg s_{\max} - \lg s_{\text{goal}}}, 0, 1\right)$$

where $s$ is the MAPE of the best equation, $s_{\max}=100\%$, and $s_{\text{goal}}=0.1\%$. This continuous reward design avoids the sparsity of binary rewards. Training data is constructed via a hybrid strategy combining rule-based and model-based synthesis, covering four domains: materials science, chemistry, biology, and physics.

## Key Experimental Results

### Main Results

Accuracy results on the LSR-Synth benchmark (129 problems across 4 disciplines):

| Method | Overall Acc₀.₀₁ | Overall Acc₀.₀₀₁ | Materials Acc₀.₀₁ | Chemistry Acc₀.₀₁ | Biology Acc₀.₀₁ | Physics Acc₀.₀₁ |
|--------|-----------------|-------------------|-------------------|-------------------|-----------------|-----------------|
| PySR | 29.46 | 14.47 | 53.33 | 25.93 | 16.67 | 25.76 |
| LLM-SR (Qwen-480B) | 41.08 | 18.09 | 80.00 | 36.11 | 30.56 | 28.79 |
| SR-Scientist (GPT-120B) | **63.57** | **49.35** | 74.67 | **81.48** | **66.67** | 40.91 |
| SR-Scientist (GLM) | 48.32 | 25.06 | 81.33 | 45.37 | 40.28 | 36.37 |
| SR-Scientist (Qwen-480B) | 49.09 | 24.55 | **86.67** | 40.74 | 50.00 | 34.09 |
| SR-Scientist (30B) | 32.30 | 16.02 | 81.33 | 22.22 | 22.22 | 18.18 |
| SR-Scientist (30B+RL) | 40.92 | 20.69 | 85.33 | 37.38 | 29.17 | 25.00 |

**Core Finding**: SR-Scientist outperforms all baselines by 6%–35% across all four backbone models, with GPT-OSS-120B achieving the highest overall performance. RL training yields consistent and significant improvements across all disciplines.

### Ablation Study

| Method | Acc₀.₀₁ | Acc₀.₀₀₁ |
|--------|---------|----------|
| SR-Scientist (GPT) | 63.57 | 49.35 |
| w/o Data Analyzer $T_1$ | 35.66 | 16.28 |
| w/o Experience Buffer | 57.36 | 41.86 |
| w/o top-k (random sampling) | 58.14 | 41.86 |

Key findings from the ablation:
- The data analysis tool has the largest impact on the GPT model (a drop of ~28 percentage points)
- The experience buffer has the largest impact on the Qwen model (a drop of 13.4 percentage points)
- Top-k sampling outperforms random sampling

### Key Findings

1. **Symbolic Accuracy**: SR-Scientist achieves the best performance in fully recovering ground-truth equations (SA=7.75–8.00), surpassing PySR (4.65) and LLM-SR (5.43)
2. **Noise Robustness**: SR-Scientist consistently outperforms other methods under Gaussian noise with varying standard deviations
3. **OOD Generalization**: Discovered equations maintain the best performance on out-of-distribution test data
4. **Optimal Interaction Length**: 25 rounds is the optimal value; too few (10 rounds) is insufficient for thorough exploration, while too many yields diminishing returns
5. **Tool Usage Behavioral Differences**: GPT-series models tend to directly write residual analysis code, whereas Qwen/GLM models more frequently use descriptive statistics

## Highlights & Insights

1. **Paradigm Shift**: Transforming the LLM from a passive equation proposer to an autonomous AI scientist represents a significant conceptual advance in scientific discovery
2. **Elegant Experience Buffer Design**: A simple heap structure resolves the LLM context length limitation while enabling cross-iteration knowledge transfer
3. **Continuous Reward Design**: Leveraging the continuous measurability of equation performance, the log-linear reward avoids sparsity and is better suited to this task than the binary rewards typical of math or code tasks
4. **Minimal Human Pipeline Principle**: The agent freely determines its own workflow; different backbone models exhibit distinct analytical strategies (e.g., GPT favors residual analysis, Qwen favors statistical analysis)
5. **Effective RL Self-Improvement**: The 30B model trained with RL approaches the performance of larger non-RL models, validating the feasibility of agent self-improvement

## Limitations & Future Work

1. Only text-based models are used; multimodal inputs (e.g., chart analysis) are not exploited
2. Performance degrades significantly under noisy conditions
3. The agent may repeatedly explore previously identified poor-performing equations across iterations; the memory system has room for improvement
4. Although the evaluation set is designed to mitigate memorization, LSR-Synth remains synthetic data and may not fully reflect the complexity of real-world scientific discovery

## Related Work & Insights

- Compared to works such as FunSearch (Romera-Paredes et al., 2024) and AlphaEvolve, SR-Scientist places greater emphasis on agent autonomy and long-horizon interaction
- The combination of experience buffer and GRPO provides a blueprint for RL training of scientific discovery agents
- The modular framework design (pluggable tools, swappable backbone models) is highly extensible and can be generalized to other scientific discovery tasks

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Introducing the agentic AI paradigm into symbolic regression, combined with RL-based self-improvement, constitutes a significant contribution
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across 4 disciplines, 5 backbone models, and multiple metrics including accuracy, generalization, noise robustness, and symbolic accuracy
- **Practicality**: ⭐⭐⭐⭐ — Open-source code and modular framework, though the reliance on extensive LLM calls entails non-trivial computational cost
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure and well-specified algorithms, though some sections could be more concise
- **Overall Rating**: ⭐⭐⭐⭐ (4/5)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents](newtonbench_benchmarking_generalizable_scientific_law_discovery_in_llm_agents.md)
- [\[ICLR 2026\] The Controllability Trap: A Governance Framework for Military AI Agents](the_controllability_trap_a_governance_framework_for_military_ai_agents.md)
- [\[ICLR 2026\] Toward a Dynamic Stackelberg Game-Theoretic Framework for Agentic AI Defense Against LLM Jailbreaking](toward_a_dynamic_stackelberg_game-theoretic_framework_for_agentic_ai_defense_aga.md)
- [\[ICLR 2026\] Inherited Goal Drift: Contextual Pressure Can Undermine Agentic Goals](inherited_goal_drift_contextual_pressure_can_undermine_agentic_goals.md)
- [\[ACL 2026\] How Adversarial Environments Mislead Agentic AI](../../ACL2026/llm_agent/how_adversarial_environments_mislead_agentic_ai.md)

</div>

<!-- RELATED:END -->

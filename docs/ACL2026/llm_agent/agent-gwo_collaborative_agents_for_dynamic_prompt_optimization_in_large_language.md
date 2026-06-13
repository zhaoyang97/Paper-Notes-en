---
title: >-
  [Paper Note] Agent-GWO: Collaborative Agents for Dynamic Prompt Optimization in Large Language Models
description: >-
  [ACL 2026][LLM Agent][Prompt Optimization] Ours proposes Agent-GWO, which introduces the leader-follower mechanism of the Grey Wolf Optimizer (GWO) into a multi-agent framework to jointly optimize prompt templates and de…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Prompt Optimization"
  - "Grey Wolf Optimizer"
  - "Multi-Agent"
  - "Reasoning Enhancement"
  - "Decoding Hyperparameters"
date: 2026-05-08
content_hash: 0a732ab7e5dd7d4c
---

# Agent-GWO: Collaborative Agents for Dynamic Prompt Optimization in Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.18612](https://arxiv.org/abs/2604.18612)  
**Code**: Coming soon  
**Area**: LLM Agent  
**Keywords**: Prompt Optimization, Grey Wolf Optimizer, Multi-Agent, Reasoning Enhancement, Decoding Hyperparameters

## TL;DR

Ours proposes Agent-GWO, which introduces the leader-follower mechanism of the Grey Wolf Optimizer (GWO) into a multi-agent framework to jointly optimize prompt templates and decoding hyperparameters (temperature, top-p, etc.). It consistently outperforms existing prompt optimization methods across 11 mathematical and mixed reasoning benchmarks.

## Background & Motivation

**Background**: Prompting strategies such as Chain-of-Thought (CoT) have significantly improved LLM performance in complex reasoning tasks, but high-quality reasoning still relies heavily on manually designed static prompts. Methods like ToT, GoT, and AoT improve reasoning trajectories under fixed prompts, but the fragility of the prompts themselves remains unresolved.

**Limitations of Prior Work**: (1) Reasoning performance is highly sensitive to prompt phrasing, example order, and context perturbations, where minor changes can lead to large performance fluctuations; (2) existing automatic prompt optimization methods usually adopt single-agent local search and cannot simultaneously optimize prompts and decoding hyperparameters; (3) the joint space of prompts and decoding configurations (temperature, top-p, repetition penalty, etc.) is massive, making manual trial-and-error extremely inefficient.

**Key Challenge**: Reasoning quality is controlled by both the prompt template and the decoding configuration, yet existing methods either optimize only the prompt (ignoring decoding configurations) or rely on a fixed prompt (tuning only decoding parameters), lacking a unified optimization framework.

**Goal**: Automatically discover more stable and task-adapted prompt-decoding configuration pairs at inference-time without additional training.

**Key Insight**: Each agent is defined as a combination of a prompt template and decoding hyperparameters, transforming the prompt optimization problem into a swarm intelligence optimization problem. The Grey Wolf Optimizer (GWO) naturally possesses a leader-follower hierarchical structure, making it suitable for guiding collaborative swarm searches.

**Core Idea**: Use the $\alpha/\beta/\delta$ leader mechanism of GWO to iteratively optimize a population composed of multiple agents (each agent = prompt + decoding parameters). The three best-performing agents guide the updates of the remaining agents, eventually converging to a robust optimal reasoning configuration.

## Method

### Overall Architecture

Agent-GWO organizes $N$ agents into a population, where each agent is defined by a shared frozen LLM, an independent prompt template, and a set of decoding hyperparameters (temperature, top-p, frequency penalty, presence penalty, maximum length). During the iterative optimization process: (1) All agents process validation set problems in parallel to generate reasoning chains and answers; (2) fitness is evaluated based on exact-match accuracy to select the three leaders, $\alpha/\beta/\delta$; (3) the hyperparameters of non-elite agents move toward the leaders, and prompts are updated via LLM-driven lightweight editing; (4) after $K$ rounds, the configuration of the final $\alpha$ agent is returned for direct use in inference.

### Key Designs

1. **Agent Structure and Hyperparameter Sampling**:

    - **Function**: Define the search space and initialize a diverse agent population.
    - **Mechanism**: Each agent $Agent_j = (\eta_j, prompt_j)$, where $\eta_j = \{T_j, p_j, F_j, E_j, M_j\}$ represents temperature, top-p, frequency penalty, presence penalty, and maximum length, respectively. During initialization, parameters are sampled from a normal distribution and clipped to legal ranges, e.g., $T_j \sim \mathcal{N}(\mu_t, \sigma_t^2)$ clipped to $[a_t, b_t]$, ensuring population diversity.
    - **Design Motivation**: Unify prompts and decoding parameters into inheritable agent configurations, allowing both to be jointly optimized within the same framework.

2. **GWO Leader-Follower Update Mechanism**:

    - **Function**: Guide the population to collaboratively search for the optimal configuration.
    - **Mechanism**: In each round, the three agents with the highest fitness are selected as $\alpha/\beta/\delta$ leaders. Non-elite agents' hyperparameters converge toward the leaders via weighted averaging: $\eta_j^{(k)} = w_\alpha X_\alpha + w_\beta X_\beta + w_\delta X_\delta$, where $w_\alpha > w_\beta > w_\delta$ ensures the best agent has the most influence. Prompts are updated via a `PromptAdaptation` function—using a fixed system instruction to let the LLM perform lightweight editing (step reordering, semantic equivalent rewriting, format adjustment) on the current prompt by referencing the three leaders' prompts.
    - **Design Motivation**: The hierarchical guidance of GWO has been proven to balance exploration and exploitation in continuous optimization. Treating hyperparameters as a continuous space and prompts as a discrete space allows updates via weighted averaging and LLM editing, respectively.

3. **Multi-dimensional Fitness Evaluation**:

    - **Function**: Robustly evaluate agent quality.
    - **Mechanism**: The primary evaluation uses exact-match accuracy on the validation set. When accuracies are identical, an LLM-judge ranks agents based on three dimensions: logical consistency $s_{logic}$, creativity $s_{creativity}$, and completeness $s_{complete}$, with weights $(0.5, 0.2, 0.3)$ calibrated on 200 manually annotated samples.
    - **Design Motivation**: Pure accuracy is prone to ties on small batches; auxiliary scoring ensures the stability of leader selection.

## Key Experimental Results

### Main Results

| Backbone | Method | GSM8K | MATH | SVAMP | MultiArith | 11-Task Avg |
|----------|------|-------|------|-------|------------|-----------|
| GPT-4o-mini | CoT | 85.4% | 74.8% | 84.7% | 89.5% | 74.6% |
| GPT-4o-mini | AoT | 95.0% | 83.6% | 91.5% | 92.6% | 83.3% |
| GPT-4o-mini | **Agent-GWO** | **95.9%** | **80.2%** | **92.3%** | **95.3%** | **84.0%** |
| Qwen-7B | CoT | 77.5% | 65.8% | 82.7% | 84.6% | 62.2% |
| Qwen-7B | **Agent-GWO** | **89.1%** | **74.1%** | **90.1%** | **93.3%** | **69.9%** |
| Gemma-12B | CoT | 83.5% | 72.8% | 79.3% | 82.7% | 74.0% |
| Gemma-12B | **Agent-GWO** | **92.8%** | **82.1%** | **90.9%** | **95.9%** | **82.4%** |

### Ablation Study

| Configuration | Avg Accuracy | Description |
|------|-----------|------|
| Agent-GWO (n=5, K=10) | 84.0% | Full model |
| Optimize prompt only (Fixed decoding) | Decrease | Decoding parameter optimization contributes |
| Optimize decoding only (Fixed prompt) | Decrease | Prompt optimization contributes |
| Random search instead of GWO | Decrease | GWO guidance is superior to random |
| n=3 agents | Slight decrease | Number of agents affects search sufficiency |

### Key Findings

- Agent-GWO achieved the highest average accuracy across all three backbones and was optimal or sub-optimal on most individual tasks.
- Compared to CoT, the Gain was larger on the small model (Qwen-7B, +7.7pp), indicating that optimization is more helpful for weaker models.
- Jointly optimizing prompts and decoding parameters outperformed optimizing either individually, validating the necessity of the unified framework.
- The default configuration of $n=5, K=10$ provides a good balance between computational budget and performance.

## Highlights & Insights

- **Prompt optimization as swarm intelligence search**: By unifying prompt templates and decoding hyperparameters into agent configurations, the joint space is systematically searched via population optimization, which is more efficient than manual trial-and-error or single-agent search.
- **Natural mapping of GWO hierarchy**: The $\alpha/\beta/\delta$ leaders naturally correspond to the "current best/second-best/third-best" configurations. Followers converge toward leaders via weighted updates while maintaining diversity, resulting in an elegant design.
- **Inference-time adaptation without training**: The entire process is completed on a frozen model, and the optimized configuration can be used directly for inference, making it highly practical.

## Limitations & Future Work

- The optimization process requires multiple evaluations of all agents on the validation set, with computational costs proportional to $n \times K$.
- The quality of LLM-driven prompt editing depends on the design of system instructions, and the editing space is limited to "lightweight editing."
- Validation was only performed on reasoning tasks (math/logic); defining fitness for open-ended generation tasks is more difficult.
- The default configuration of 5 agents and 10 iterations might not be optimal for all scenarios; adaptive adjustment strategies remain to be explored.

## Related Work & Insights

- **vs AFlow**: AFlow also performs automatic workflow optimization, but its search space is limited to workflow structures. Agent-GWO searches both prompts and decoding parameters, covering a larger search space.
- **vs CoT-SC**: Self-consistency improves robustness through multi-sample voting but does not optimize the prompt itself. Agent-GWO directly optimizes the generation configuration to fundamentally improve reasoning quality.
- **vs AoT (Atom-of-Thought)**: AoT is the strongest competitor for Agent-GWO, enhancing performance through atomized reasoning. Agent-GWO slightly outperforms it on most tasks, and the two are orthogonal and can be combined.

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing GWO to prompt optimization is a novel cross-domain combination, though metaheuristic optimization with LLMs has precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive experiments with 11 benchmarks, 3 backbones, and 7 baselines.
- Writing Quality: ⭐⭐⭐⭐ The framework description is clear, though theoretical analysis could be deeper.
- Value: ⭐⭐⭐⭐ Provides a practical inference-time optimization solution, particularly valuable in resource-constrained scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ToolWeaver: Weaving Collaborative Semantics for Scalable Tool Use in Large Language Models](../../ICLR2026/llm_agent/toolweaver_weaving_collaborative_semantics_for_scalable_tool_use_in_large_langua.md)
- [\[ACL 2026\] ImplicitMemBench: Measuring Unconscious Behavioral Adaptation in Large Language Models](implicitmembench_measuring_unconscious_behavioral_adaptation_in_large_language_m.md)
- [\[ACL 2026\] AnchorMem: Anchored Facts with Associative Contexts for Building Memory in Large Language Models](anchormem_anchored_facts_with_associative_contexts_for_building_memory_in_large_.md)
- [\[ACL 2026\] Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments](feedback-driven_tool-use_improvements_in_large_language_models_via_automated_bui.md)
- [\[ACL 2026\] Lightweight LLM Agent Memory with Small Language Models](lightweight_llm_agent_memory_with_small_language_models.md)

</div>

<!-- RELATED:END -->

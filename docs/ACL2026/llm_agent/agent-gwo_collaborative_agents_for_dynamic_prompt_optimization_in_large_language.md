---
title: >-
  [Paper Note] Agent-GWO: Collaborative Agents for Dynamic Prompt Optimization in Large Language Models
description: >-
  [ACL 2026][LLM Agent][Prompt optimization] This paper proposes Agent-GWO, which integrates the leader-follower hierarchy of the Grey Wolf Optimizer (GWO) into a multi-agent framework to jointly optimize prompt templates…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Prompt optimization"
  - "Grey Wolf Optimizer"
  - "multi-agent"
  - "reasoning enhancement"
  - "decoding hyperparameters"
date: 2026-05-08
content_hash: a4795e842869b899
---

# Agent-GWO: Collaborative Agents for Dynamic Prompt Optimization in Large Language Models

**Conference**: ACL 2026
**arXiv**: [2604.18612](https://arxiv.org/abs/2604.18612)  
**Code**: Coming soon  
**Area**: LLM Agent
**Keywords**: Prompt optimization, Grey Wolf Optimizer, multi-agent, reasoning enhancement, decoding hyperparameters

## TL;DR

This paper proposes Agent-GWO, which integrates the leader-follower hierarchy of the Grey Wolf Optimizer (GWO) into a multi-agent framework to jointly optimize prompt templates and decoding hyperparameters (temperature, top-p, etc.), consistently outperforming existing prompt optimization methods across 11 mathematical and mixed reasoning benchmarks.

## Background & Motivation

**Background**: Prompting strategies such as Chain-of-Thought (CoT) have substantially improved LLM performance on complex reasoning tasks; however, high-quality reasoning still relies heavily on manually designed static prompts. Methods such as ToT, GoT, and AoT improve reasoning trajectories under fixed prompts but leave unresolved the inherent fragility of the prompts themselves.

**Limitations of Prior Work**: (1) Reasoning performance is highly sensitive to prompt wording, example ordering, and contextual perturbations, where minor changes can cause substantial performance degradation; (2) existing automatic prompt optimization methods typically employ single-agent local search and cannot simultaneously optimize both prompts and decoding hyperparameters; (3) the joint space of prompt templates and decoding configurations (temperature, top-p, repetition penalty, etc.) is vast, rendering manual trial-and-error extremely inefficient.

**Key Challenge**: Reasoning quality is jointly governed by both the prompt template and the decoding configuration, yet existing methods either optimize only the prompt (ignoring decoding settings) or tune only decoding parameters under a fixed prompt, lacking a unified optimization framework.

**Goal**: Automatically discover more stable and task-adaptive prompt–decoding configuration pairs at inference time, without any additional training.

**Key Insight**: Each agent is defined as a combination of a prompt template and decoding hyperparameters, recasting prompt optimization as a swarm intelligence optimization problem. GWO's inherent leader-follower hierarchical structure is well-suited to guiding collaborative population-based search.

**Core Idea**: The α/β/δ leader mechanism of GWO drives iterative optimization over a population of agents—each comprising a prompt and decoding parameters—where the three best-performing agents guide the update of the remaining agents, ultimately converging to a robust optimal inference configuration.

## Method

### Overall Architecture

Agent-GWO organizes $N$ agents into a population, where each agent is defined by a shared frozen LLM, an individual prompt template, and a set of decoding hyperparameters (temperature, top-p, frequency penalty, presence penalty, and maximum length). The iterative optimization proceeds as follows: (1) all agents process validation-set problems in parallel to generate reasoning chains and answers; (2) fitness is evaluated via exact-match accuracy, and the top three agents are designated as α/β/δ leaders; (3) non-elite agents' hyperparameters are updated toward the leaders, and prompts are revised via lightweight LLM-driven editing; (4) after $K$ rounds, the final α agent's configuration is returned for direct use at inference time.

### Key Designs

1. **Agent Structure and Hyperparameter Sampling**:

    - Function: Define the search space and initialize a diverse agent population.
    - Mechanism: Each agent is defined as $Agent_j = (\eta_j, prompt_j)$, where $\eta_j = \{T_j, p_j, F_j, E_j, M_j\}$ denotes temperature, top-p, frequency penalty, presence penalty, and maximum length, respectively. Parameters are initialized by sampling from a normal distribution and clipping to valid ranges, e.g., $T_j \sim \mathcal{N}(\mu_t, \sigma_t^2)$ clipped to $[a_t, b_t]$, ensuring population diversity.
    - Design Motivation: Unifying prompts and decoding parameters as inheritable agent configurations enables joint optimization of both within a single framework.

2. **GWO Leader-Follower Update Mechanism**:

    - Function: Guide the population to collaboratively search for the optimal configuration.
    - Mechanism: At each iteration, the three agents with the highest fitness are designated as α/β/δ leaders. Non-elite agents' hyperparameters are updated via a weighted average toward the leaders: $\eta_j^{(k)} = w_\alpha X_\alpha + w_\beta X_\beta + w_\delta X_\delta$, where $w_\alpha > w_\beta > w_\delta$ ensures the best agent has the greatest influence. Prompts are updated via a PromptAdaptation function—a fixed system instruction that instructs the LLM to perform lightweight edits to the current prompt (step reordering, semantically equivalent paraphrasing, format adjustment) by referencing the three leaders' prompts.
    - Design Motivation: GWO's hierarchical guidance has been shown to balance exploration and exploitation in continuous optimization. Hyperparameters are treated as a continuous space updated via weighted averaging, while prompts are treated as a discrete space updated via LLM-driven editing.

3. **Multi-Dimensional Fitness Evaluation**:

    - Function: Robustly assess agent quality.
    - Mechanism: Primary evaluation uses exact-match accuracy on the validation set. When accuracy is tied, an LLM-judge scores agents along three dimensions—logical consistency $s_{logic}$, creativity $s_{creativity}$, and completeness $s_{complete}$—with weights $(0.5, 0.2, 0.3)$ calibrated on 200 manually annotated samples.
    - Design Motivation: Pure accuracy on small batches frequently yields ties; the auxiliary scoring ensures stable leader selection.

## Key Experimental Results

### Main Results

| Backbone | Method | GSM8K | MATH | SVAMP | MultiArith | 11-Task Avg |
|----------|--------|-------|------|-------|------------|-------------|
| GPT-4o-mini | CoT | 85.4% | 74.8% | 84.7% | 89.5% | 74.6% |
| GPT-4o-mini | AoT | 95.0% | 83.6% | 91.5% | 92.6% | 83.3% |
| GPT-4o-mini | **Agent-GWO** | **95.9%** | **80.2%** | **92.3%** | **95.3%** | **84.0%** |
| Qwen-7B | CoT | 77.5% | 65.8% | 82.7% | 84.6% | 62.2% |
| Qwen-7B | **Agent-GWO** | **89.1%** | **74.1%** | **90.1%** | **93.3%** | **69.9%** |
| Gemma-12B | CoT | 83.5% | 72.8% | 79.3% | 82.7% | 74.0% |
| Gemma-12B | **Agent-GWO** | **92.8%** | **82.1%** | **90.9%** | **95.9%** | **82.4%** |

### Ablation Study

| Configuration | Avg Accuracy | Notes |
|---------------|-------------|-------|
| Agent-GWO (n=5, K=10) | 84.0% | Full model |
| Prompt-only optimization (fixed decoding params) | Decreased | Decoding parameter optimization contributes |
| Decoding-only optimization (fixed prompt) | Decreased | Prompt optimization contributes |
| Random search replacing GWO | Decreased | GWO-guided search outperforms random |
| n=3 agents | Slightly decreased | Agent count affects search thoroughness |

### Key Findings

- Agent-GWO achieves the highest average accuracy across all three backbones and is optimal or near-optimal on most individual tasks.
- Gains over CoT are largest on the smaller model (Qwen-7B, +7.7 pp), indicating that optimization is especially beneficial for weaker models.
- Jointly optimizing prompts and decoding parameters outperforms optimizing either in isolation, validating the necessity of a unified framework.
- The default configuration of n=5, K=10 achieves a favorable balance between computational budget and performance.

## Highlights & Insights

- **Prompt optimization as swarm intelligence search**: Unifying prompt templates and decoding hyperparameters into agent configurations enables systematic population-based search over the joint space, which is more efficient than manual trial-and-error or single-agent search.
- **Natural correspondence with GWO hierarchy**: The α/β/δ leaders naturally correspond to the best, second-best, and third-best configurations; followers converge toward the leaders via weighted updates while maintaining diversity—an elegant design.
- **Training-free inference-time adaptation**: The entire optimization process is conducted on a frozen model, and the resulting configuration can be applied directly at inference time, making it highly practical.

## Limitations & Future Work

- The optimization process requires repeated evaluation of all agents on the validation set; computational cost scales proportionally with $n \times K$.
- The quality of LLM-driven prompt editing depends on system instruction design, and the editing scope is constrained to "lightweight" modifications.
- Validation is limited to reasoning tasks (mathematical/logical); defining fitness for open-ended generation tasks is considerably more challenging.
- The default configuration of 5 agents and 10 iterations may not be optimal across all scenarios; adaptive adjustment strategies remain to be explored.

## Related Work & Insights

- **vs. AFlow**: AFlow also performs automatic workflow optimization, but its search space is restricted to workflow structure. Agent-GWO simultaneously searches over prompts and decoding parameters, covering a substantially larger space.
- **vs. CoT-SC**: Self-consistency improves robustness via majority voting over multiple samples but does not optimize the prompt itself. Agent-GWO directly optimizes generation configurations, addressing reasoning quality more fundamentally.
- **vs. AoT (Atom-of-Thought)**: AoT is Agent-GWO's strongest competitor, improving performance through atomized reasoning. Agent-GWO slightly surpasses AoT on most tasks, and the two approaches are orthogonal and potentially complementary.

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing GWO into prompt optimization is a novel cross-domain combination, though the pairing of metaheuristic optimization with LLMs has prior precedent.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on 11 benchmarks across 3 backbones against 7 baselines—highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ The framework is described clearly, though theoretical analysis could be deepened.
- Value: ⭐⭐⭐⭐ Provides a practical inference-time optimization solution that is particularly valuable in resource-constrained settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination](atlas_adaptive_trading_with_llm_agents_through_dynamic_prompt_optimization_and_m.md)
- [\[ACL 2026\] ImplicitMemBench: Measuring Unconscious Behavioral Adaptation in Large Language Models](implicitmembench_measuring_unconscious_behavioral_adaptation_in_large_language_m.md)
- [\[ICLR 2026\] ToolWeaver: Weaving Collaborative Semantics for Scalable Tool Use in Large Language Models](../../ICLR2026/llm_agent/toolweaver_weaving_collaborative_semantics_for_scalable_tool_use_in_large_langua.md)
- [\[ACL 2026\] AnchorMem: Anchored Facts with Associative Contexts for Building Memory in Large Language Models](anchormem_anchored_facts_with_associative_contexts_for_building_memory_in_large_.md)
- [\[ACL 2026\] Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments](feedback-driven_tool-use_improvements_in_large_language_models_via_automated_bui.md)

</div>

<!-- RELATED:END -->

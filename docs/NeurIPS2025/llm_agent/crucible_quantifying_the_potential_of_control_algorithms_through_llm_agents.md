---
title: >-
  [Paper Note] Crucible: Quantifying the Potential of Control Algorithms through LLM Agents
description: >-
  [NeurIPS 2025][LLM Agent][tuning potential] This paper is the first to formalize the concept of **Tuning Potential**, using LLM agents to simulate multi-level developers performing dual-layer (parameter + logic) optimiza…
tags:
  - "NeurIPS 2025"
  - "LLM Agent"
  - "tuning potential"
  - "control algorithm"
  - "parameter optimization"
  - "Bayesian optimization"
date: 2026-05-08
content_hash: ffb5d02709452be3
---

# Crucible: Quantifying the Potential of Control Algorithms through LLM Agents

**Conference**: NeurIPS 2025
**arXiv**: [2510.18491](https://arxiv.org/abs/2510.18491)  
**Code**: [https://github.com/thu-media/Crucible](https://github.com/thu-media/Crucible)  
**Area**: LLM Agent / Control Algorithm Evaluation
**Keywords**: tuning potential, LLM agent, control algorithm, parameter optimization, Bayesian optimization

## TL;DR

This paper is the first to formalize the concept of **Tuning Potential**, using LLM agents to simulate multi-level developers performing dual-layer (parameter + logic) optimization of control algorithms. On CartPole, Bang-bang improves from 34→500, reaching DQN-level performance; on ABR tasks, Crucible achieves up to 44.1% improvement over Bayesian optimization.

## Background & Motivation

**Blind spot in control algorithm evaluation.** Existing studies typically evaluate algorithms under default parameters or ideal conditions, whereas in production environments algorithms are always tuned and logically restructured by domain experts for specific scenarios. An algorithm's practical value depends not only on its default design, but also on its intrinsic tunability—i.e., its *Tuning Potential*. No systematic method currently exists to measure this property, causing it to be overlooked in algorithm selection and design.

**Evaluation challenges beyond traditional parameter sensitivity analysis.** Assessing tuning potential cannot be limited to hyperparameter search; it must also cover deeper logic-level modifications—such as adding control branches or integrating new components. These structural changes heavily depend on developers' subjective understanding of the algorithm, rendering traditional evaluation methods ineffective. The central tension lies in simultaneously capturing the interaction between objective performance metrics and subjective comprehension factors.

**Motivating case studies.** Preliminary experiments by the authors show that: in the ABR scenario, the simple HYB algorithm after tuning improves QoE from worst to first and reduces video stall time by 92%; in the scheduling scenario, FIFO after tuning achieves the optimal cumulative waiting time. These results demonstrate that "simple algorithm + thorough tuning" can outperform "complex algorithm + default configuration." The core idea of Crucible is to use LLMs to simulate developers of varying skill levels performing algorithm optimization, and to establish a formal potential measurement framework.

## Method

### Overall Architecture

Crucible consists of two core components: (1) a multi-level expert simulation agent driven by LLMs, simulating developers of different capability levels performing parameter tuning and logic restructuring of control algorithms; and (2) a unified tuning potential metric based on environment performance feature vectors. The system workflow is: execute the LLM optimization loop for each test environment → collect cases with the largest performance gaps → LLM proposes optimization suggestions → apply modifications with optional Bayesian optimization → after traversing all environments, enter the evaluation phase to compute potential.

### Key Designs

1. **Multi-dimensional domain knowledge injection**:

    - Function: Constructs a complete task-understanding context for the LLM.
    - Mechanism: Injects three-dimensional knowledge via system prompt—task description (input state + output action space), optimization objective (improvement direction + evaluation criteria), and environment overview (scenario characteristics + constraints).
    - Design Motivation: The LLM requires full contextual understanding of the control task to make effective logic-level modifications; the three-dimensional design ensures coverage of "what to do, what to optimize, and where to operate."

2. **Dual-layer parameter-logic optimization agent**:

    - Function: Simultaneously explores the hyperparameter space and the algorithm logic space to fully mine the algorithm's tuning capacity.
    - Mechanism: Bayesian optimization is encapsulated as a tool interface (evaluating performance upper bounds within the parameter space), while the LLM performs logic-level modifications (adding control branches, restructuring algorithm logic). Each modification is stored as a triplet (modification rationale, specific operation, observed result) to serve as empirical grounding for subsequent optimization.
    - Design Motivation: Parameter optimization has a ceiling (constrained by the representational capacity of the algorithm logic), while pure LLM logic modification is unstable (no improvement in 60% of scenarios). The two must collaborate—Bayesian optimization performs fine-grained search within the new solution spaces opened by LLM modifications.

3. **Differentiated developer capability simulation**:

    - Function: Simulates developers with varying skill levels and resource budgets.
    - Mechanism: Capability differences are modeled by adjusting computational budgets rather than designing distinct prompts—specifically, by limiting the number of Bayesian optimization calls (0/10/20) and reflection iteration steps (1/2/3).
    - Design Motivation: Based on a key insight—the core difference between experts and novices is not "knowing more" but "being able to invest more resources in trial-and-error and fine-grained tuning." Resource budget is a more realistic proxy for capability.

### Loss & Training

Crucible does not involve conventional model training. The formal definition of tuning potential is as follows: probe algorithms are first run across all evaluation environments and normalized to obtain a performance feature vector per environment; the distance between two environments is defined as the RMSE of their feature vectors, and similarity is defined as $\text{sim}(E_i, E_t) = \max(0, 1 - \text{dis}(E_i, E_t))$; the final potential is $\mathcal{P} = \frac{1}{|\mathcal{T}|} \sum_{E_t} [(S_{t,c} - S_{t,o}) \times \text{sim}(E_i, E_t)]$, i.e., the similarity-weighted mean performance gain across all test environments. This design down-weights gains achieved on environments that differ substantially from the ideal environment, ensuring robustness and fairness of the metric.

## Key Experimental Results

### Main Results

| Dataset/Task | Metric | Ours (Crucible) | Prev. SOTA (Bayes/Default) | Gain |
|---|---|---|---|---|
| CartPole Bang-bang | Score | 500 | 34 (default) / 56 (Bayes) | 34→500, one LLM logic modification |
| CartPole PID | Score | 500 | 34 (default) / 77 (Bayes) | Reaches DQN optimum in two iterations |
| ABR Puffer | QoE improvement | +44.1% vs. Bayes | Bayesian baseline | Highest gain |
| ABR Real deployment (Dash.js) | QoE | HYB/BBA = 1.72 | Pensieve (RL) = 1.66 | Surpasses RL baseline after tuning |

### Ablation Study

| Configuration | Key Metric | Note |
|---|---|---|
| Bayes=0, LLM iteration | 60% of scenarios show no improvement | Pure LLM logic modification is unstable |
| Bayes=20, LLM iteration | 20% of scenarios show no improvement | Bayes+LLM synergy is significant |
| Claude 3.7 (HYB) | QoE = 1.12 | Stronger model unlocks greater potential |
| Claude 3.5 (HYB) | QoE = 1.03 | Consistent conclusions across models |
| GPT-4o-mini (HYB) | QoE = 1.04 | Robust across model variants |

### Key Findings
- An algorithm's **representational capacity** is the central factor of potential: HYB (dual-state input) potential 0.068 >> BBA (single-state) 0.018.
- An algorithm's **interpretability** is equally critical: Pitree (decision tree distilled from RL) has low initial performance and low potential (0.033), as complex logic impedes LLM optimization.
- ABR improvement margins far exceed those of scheduling algorithms, because scheduling DAG input states are more complex and harder for LLMs to comprehend.
- Potential evaluation guides algorithm redesign: BBA augmented with bandwidth input (BBA_C) differs only 0.5% in initial performance, but achieves 4% improvement after tuning; SJF after tuning surpasses a multi-level feedback algorithm that was initially superior.

## Highlights & Insights
- Pioneers **Tuning Potential** as a new dimension for algorithm evaluation—assessing not only "how good it is now" but "how good it can be optimized to become."
- The collaborative design of parameter-level exploitation (Bayesian) + logic-level exploration (LLM) is broadly applicable to other LLM-as-optimizer scenarios.
- Implications for algorithm design: favor algorithms that are simple, interpretable, and operate over wide state spaces, rather than complex black boxes.
- The approach of modeling developer capability through resource budgets (rather than prompt engineering) offers a novel and realistic modeling perspective.

## Limitations & Future Work
- LLM version affects results, though the authors interpret this as simulating developers of different skill levels—an argument that is somewhat circular.
- The framework cannot directly modify the internal logic of black-box algorithms; it is limited to interpretable models such as decision trees.
- The success rate of pure LLM modification is only 40%, reflecting heavy reliance on Bayesian optimization.
- The standardized interface constrains the LLM to modifying algorithm code only, precluding modifications to environment settings or evaluation procedures.

## Related Work & Insights
- Connection to LLM-based human simulation (e.g., Generative Agents): both use LLMs to simulate human behavior, but Crucible focuses on algorithm tuning rather than social behavior.
- Complementarity with Bayesian optimization: conventional BO searches only the parameter space, while Crucible simultaneously searches the logic space.
- Implications for AutoML/NAS: algorithm potential can serve as a novel objective function for search and evaluation.

## Rating
- Novelty: ⭐⭐⭐⭐ The concept of "Tuning Potential" is original and formally complete, opening a new dimension for algorithm evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three-layer validation across classic control, computer systems, and real-world deployment, with thorough cross-LLM robustness testing.
- Writing Quality: ⭐⭐⭐⭐ Motivating experiments are convincing, problem formulation is clear, and the logic from motivation to validation flows smoothly.
- Value: ⭐⭐⭐⭐ Introduces a new dimension to algorithm design and evaluation, offering meaningful reference for the LLM-as-optimizer paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Reflection-Driven Control for Trustworthy Code Agents](../../AAAI2026/llm_agent/reflection-driven_control_for_trustworthy_code_agents.md)
- [\[ICML 2026\] EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle](../../ICML2026/llm_agent/evolver_self-evolving_llm_agents_through_an_experience-driven_lifecycle.md)
- [\[NeurIPS 2025\] A-MEM: Agentic Memory for LLM Agents](a-mem_agentic_memory_for_llm_agents.md)
- [\[ACL 2026\] When Agents Look the Same: Quantifying Distillation-Induced Similarity in Tool-Use Behaviors](../../ACL2026/llm_agent/when_agents_look_the_same_quantifying_distillation-induced_similarity_in_tool-us.md)
- [\[ICML 2026\] Scaling Small Agents Through Strategy Auctions](../../ICML2026/llm_agent/scaling_small_agents_through_strategy_auctions.md)

</div>

<!-- RELATED:END -->

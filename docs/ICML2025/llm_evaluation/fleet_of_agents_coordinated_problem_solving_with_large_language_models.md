---
title: >-
  [Paper Note] Fleet of Agents: Coordinated Problem Solving with Large Language Models
description: >-
  [ICML2025][LLM Evaluation][LLM Reasoning] Proposes Fleet of Agents (FoA), which coordinates LLM reasoning across multiple agents based on genetic particle filtering: independent exploration by multiple agents $\rightarrow$ resampling based on heuristic value functions $\rightarrow$ dynamic branching to adapt to discovered solutions. It improves quality by 5% on average compared to SOTA methods while requiring only 40% of the cost.
tags:
  - "ICML2025"
  - "LLM Evaluation"
  - "LLM Reasoning"
  - "Genetic Particle Filtering"
  - "Tree Search"
  - "Cost-efficiency"
  - "Exploration-exploitation"
date: 2026-05-08
content_hash: 9d54ecb962f0b48a
---

# Fleet of Agents: Coordinated Problem Solving with Large Language Models

**Conference**: ICML2025  
**arXiv**: [2405.06691](https://arxiv.org/abs/2405.06691)  
**Code**: [GitHub - FoA](https://github.com/au-clan/FoA)  
**Area**: LLM Evaluation  
**Keywords**: LLM Reasoning, Genetic Particle Filtering, Tree Search, Cost-efficiency, Exploration-exploitation

## TL;DR
Proposes Fleet of Agents (FoA), which coordinates LLM reasoning across multiple agents based on genetic particle filtering: independent exploration by multiple agents $\rightarrow$ resampling based on heuristic value functions $\rightarrow$ dynamic branching to adapt to discovered solutions. It improves quality by 5% on average compared to SOTA methods while requiring only 40% of the cost.

## Background & Motivation

### Efficiency Issues of Multi-Query Reasoning
Tree search methods such as ToT, GoT, and LATS provide high quality but are costly; the search tree can grow exponentially, making costs unpredictable.

### Limitations of Single-Query Methods
Methods like CoT have low costs but are unsuitable for sequential decision-making tasks requiring environmental interaction.

### Goal of FoA
To find a better balance between quality and cost by precisely controlling the tree width ($n$ agents) and depth ($t$ steps).

## Method

### Genetic Particle Filtering Framework
1. **Generation**: $n$ agents independently generate candidate solutions (particles).
2. **Resampling**: Select the optimal particles based on a heuristic value function.
3. **Mutation**: Continue exploration starting from the selected particles.
4. Periodic selection $\rightarrow$ retain promising exploration directions $\rightarrow$ eliminate poorly performing ones.

### Key Designs
- Fixed $n$ agents $\times$ $t$ steps = predictable cost.
- Dynamic branching: promising directions automatically receive more agents.
- No backtracking required—always moving forward.

### Loss & Training
The model is trained end-to-end, with an optimization objective that integrates task loss and regularization terms.

## Key Experimental Results

### Game of 24

### Main Results

| Method | Quality | Cost | Cost-to-Quality Ratio |
|------|------|------|-----------|
| ToT | High | High | Medium |
| GoT | High | Very High | Low |
| **FoA** | **Higher (+5%)** | **Low (40%)** | **Optimal** |

### Performance across Different LLMs

### Ablation Study

| FoA + Model | Game of 24 |
|-----------|-----------|
| GPT-3.5 | High |
| GPT-4 | Higher |
| LLaMA-11B | High |
| LLaMA-90B | Higher |

### Generalization Across Tasks

| Task | FoA vs SOTA |
|------|-----------|
| Game of 24 | +5% Quality / 40% Cost |
| Mini-Crosswords | Consistent Advantage |
| WebShop | Consistent Advantage |

### Key Findings
1. **FoA + LLaMA-11B outperforms LLaMA-90B**—methodology is more crucial than model scale.
2. Consistently achieves the best cost-quality trade-off across all 3 tasks $\times$ 4 LLMs.
3. A particle count of $n = 5\text{--}10$ is optimal.
4. Resampling frequency affects the exploration-exploitation balance.
5. Advantages are more pronounced in tasks involving environmental interactions (e.g., WebShop).

## Highlights & Insights

1. The fusion of genetic particle filtering and LLM agents is highly intuitive.
2. Cost predictability is a key advantage for engineering deployment.
3. An 11B model outperforming a 90B model demonstrates the value of the reasoning framework.
4. Consistently effective across diverse tasks such as Game of 24 and WebShop.
5. The lack of backtracking simplifies implementation.

## Limitations & Future Work

1. Designing the heuristic value function requires task-specific domain knowledge.
2. A fixed budget of $n \times t$ may lack flexibility.
3. Comparison with o1/reasoning models is currently missing.
4. The effectiveness on tasks with longer horizons has not been verified.
5. Information sharing among particles is limited (independent exploration).

## Related Work & Insights

- Difference from ToT/GoT: FoA employs particle filtering instead of backtracking search.
- Difference from LATS: LATS utilizes MCTS, whereas FoA uses genetic resampling.
- Insight: The concept of particle filtering can be extended to other multi-agent collaboration scenarios.

## Rating
- Novelty: 4.5/5 — Genetic Particle Filtering + LLM Agent
- Experimental Thoroughness: 4.5/5 — 3 Tasks $\times$ 4 LLMs
- Writing Quality: 4.5/5
- Value: 5.0/5 — Best Cost-Quality Trade-off

## Supplementary

### Insights from 11B Outperforming 90B
The quality of a reasoning framework can compensate for deficiencies in model scale. This suggests that a smaller model paired with an effective framework can be more cost-effective than a larger model using basic reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Creativity or Brute Force? Using Brainteasers as a Window into the Problem-Solving Abilities of Large Language Models](../../NeurIPS2025/llm_evaluation/creativity_or_brute_force_using_brainteasers_as_a_window_into_the_problem-solvin.md)
- [\[ACL 2026\] EngiBench: A Benchmark for Evaluating Large Language Models on Engineering Problem Solving](../../ACL2026/llm_evaluation/engibench_a_benchmark_for_evaluating_large_language_models_on_engineering_proble.md)
- [\[ICML 2025\] Communicating Activations Between Language Model Agents](communicating_activations_between_language_model_agents.md)
- [\[NeurIPS 2025\] EvaLearn: Quantifying the Learning Capability and Efficiency of LLMs via Sequential Problem Solving](../../NeurIPS2025/llm_evaluation/evalearn_quantifying_the_learning_capability_and_efficiency_of_llms_via_sequenti.md)
- [\[ICML 2025\] Correlated Errors in Large Language Models](correlated_errors_in_large_language_models.md)

</div>

<!-- RELATED:END -->

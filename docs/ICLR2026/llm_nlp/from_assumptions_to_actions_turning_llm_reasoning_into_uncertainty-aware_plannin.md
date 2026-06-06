---
title: >-
  [Paper Note] From Assumptions to Actions: Turning LLM Reasoning into Uncertainty-Aware Planning
description: >-
  [ICLR 2026][LLM/NLP][uncertainty-aware planning] This paper proposes PCE (Planner-Composer-Evaluator), a framework that explicitly extracts and organizes implicit environmental assumptions from LLM reasoning chains into…
tags:
  - "ICLR 2026"
  - "LLM/NLP"
  - "uncertainty-aware planning"
  - "LLM multi-agent collaboration"
  - "decision tree"
  - "partially observable environments"
  - "communication optimization"
date: 2026-05-08
content_hash: 78bacf944272195a
---

# From Assumptions to Actions: Turning LLM Reasoning into Uncertainty-Aware Planning

**Conference**: ICLR 2026
**arXiv**: [2602.04326](https://arxiv.org/abs/2602.04326)  
**Code**: Available (anonymous supplementary material)  
**Area**: LLM/NLP
**Keywords**: uncertainty-aware planning, LLM multi-agent collaboration, decision tree, partially observable environments, communication optimization

## TL;DR

This paper proposes PCE (Planner-Composer-Evaluator), a framework that explicitly extracts and organizes implicit environmental assumptions from LLM reasoning chains into decision trees, enabling uncertainty-aware action selection via a likelihood-gain-cost scoring function, thereby substantially reducing communication overhead in multi-agent collaboration.

## Background & Motivation

In decentralized, partially observable multi-agent collaboration scenarios (e.g., two robots cooperating to prepare a meal), each agent perceives only a fraction of the environment and faces pervasive uncertainty regarding hidden objects and collaborator intentions.

Existing LLM-driven multi-agent systems exhibit fundamental shortcomings:

**Over-reliance on communication**: Methods such as CoELA, REVECA, CaPo, and CoTS repeatedly engage in natural language dialogue to verify plans, exchange information, and iteratively refine decisions, incurring substantial token and time costs.

**Disruption of human workflows**: When collaborators are humans, frequent queries and status updates interrupt established work routines.

**Ineffectiveness of naive scaling**: Increasing model capacity or deepening reasoning chains does not fundamentally resolve uncertainty—without an explicit mechanism to identify and evaluate assumptions, even large models cannot adjudicate among competing environmental hypotheses.

Two key **empirical observations** motivate the design:
- LLMs **implicitly generate assumptions about uncertain environments** during zero-shot CoT reasoning (e.g., "there might be food in the cabinet").
- These assumptions are **referenced locally and implicitly**, never explicitly aggregated for global decision-making, precluding systematic reconciliation of multiple hypotheses.

## Method

### Overall Architecture

PCE redesigns the planning module as a three-stage pipeline:

```
Observation Module → Memory Module → [Planner → Composer → Evaluator] → Communication/Execution Module
```

Core Idea: Elevate **implicit assumptions** within LLM reasoning chains to **first-class decision variables**, reasoning over assumptions before committing to action.

### Key Designs

**Planner**: Receives goal $G$, current progress, message logs, and the list of available actions; leverages LLM reasoning to produce candidate actions along with their reasoning chains. Critically, the reasoning chains contain isolated assumption–action associations (e.g., "the bathroom cabinet might have something useful" → "go check the bathroom cabinet"), but the relationships among assumptions remain unestablished.

**Composer**: The central component, which organizes assumptions from the reasoning chain into a **decision tree**:

- **Internal nodes**: Represent environmental assumptions, each with True/False branches.
- **Leaf nodes**: Optimal actions (physical or communicative) under a specific assumption path.
- **Construction strategy**: Top-down expansion using a local ranking strategy that prioritizes assumption branches that **maximally reduce uncertainty** and **most strongly influence action selection**.
- **New assumption generation**: When existing assumptions are insufficient, the Composer proposes new atomic assumptions grounded in entities present in context.
- **Depth constraint**: Tree depth is limited to $D=3$.

Example: Goal is to find food → root assumption "living room has food" → True branch leads to "explore the living room"; False branch → new assumption "collaborator Bob might know the cupcake location" → True branch leads to "send a message to Bob."

**Evaluator**: Scores each root-to-leaf path in the decision tree along three dimensions:

1. **Scenario likelihood** $\mathcal{L}(\mathcal{S})$: Estimated probability that the assumption path holds, evaluated by the LLM based on observations and message history.
2. **Conditional gain** $\mathcal{G}(a)$: Given the assumption is true, the degree to which action $a$ advances goal completion.
3. **Execution cost** $C(a) = \alpha \cdot d(a) \cdot \mathbf{1}\{\text{move}\} + \beta \cdot \ell(a) \cdot \mathbf{1}\{\text{comm}\}$

**Final scoring function**:

$$U(\mathcal{S}, a) = \mathcal{L}(\mathcal{S}) \cdot \mathcal{G}(a) - \lambda \cdot C(a)$$

Ranking leaf nodes by $U$ yields the optimal action. Communication is treated as an **atomic option** within the action space, selected only when its utility exceeds that of physical actions—fundamentally distinguishing PCE from methods that treat communication as a search mechanism.

### Loss & Training

PCE is a pure inference-time framework requiring no training. Default hyperparameters: $D=3, \alpha=1, \beta=1, \lambda=1, K_{\text{action}}=10, K_{\text{message}}=3$. The same configuration is applied across three LLM backbones (GPT-4o mini, GPT-OSS:20B, Gemma3:4B).

## Key Experimental Results

### Main Results

**C-WAH environment (total steps ↓ lower is better)**:

| Method | GPT-4o mini | GPT-OSS:20B | Gemma3:4B |
|--------|-------------|-------------|-----------|
| **PCE** | **42.76** | **49.60** | **59.20** |
| CoELA | 60.40 | 72.72 | 77.20 |
| REVECA | 46.80 | 53.86 | 62.56 |
| CaPo | 60.82 | 68.34 | 75.88 |
| CoTS | 64.00 | 65.26 | 72.32 |

**TDW-MAT environment (transport success rate ↑ higher is better)**:

| Method | GPT-4o mini Total | GPT-OSS:20B Total | Gemma3:4B Total |
|--------|-------------------|-------------------|-----------------|
| **PCE** | **87.50%** | **81.25%** | **70.83%** |
| CoELA | 62.50% | 55.00% | 45.84% |
| REVECA | 81.25% | 73.33% | 52.09% |
| CaPo | 73.33% | 65.41% | 67.50% |
| CoTS | 75.00% | 59.17% | 63.33% |

**Communication count comparison (PCE vs. baselines, GPT-4o mini)**:
- C-WAH: PCE 1.70 vs. CoELA 9.88 / CaPo 8.72 / CoTS 10.24
- TDW-MAT: PCE 3.58 vs. CoELA 13.33 / CaPo 70.79 / CoTS 108.92

### Ablation Study

**Component ablation (C-WAH, GPT-4o mini)**:

| Variant | Total Steps ↓ | Token Cost ↓ |
|---------|--------------|-------------|
| **PCE (full)** | **42.76** | 44353 |
| w/o Planner | 56.46 | 139918 |
| w/o Composer | 46.82 | 33347 |
| w/o Evaluator | 47.34 | 44720 |

**LLM capacity scaling experiment**: Scaling Gemma3 from 4B → 12B → 27B with only the Planner (no Composer + Evaluator) yields limited improvement, whereas PCE consistently accelerates task completion across all capacity levels.

### Key Findings

1. **80%+ reduction in communication**: PCE's communication count is only 10–20% of baselines, yet task performance is comprehensively superior.
2. **Controlled token usage**: Although PCE's three-module architecture incurs higher per-step inference cost, the substantially shorter episode length keeps total token consumption comparable to baselines.
3. **Scaling cannot substitute structure**: Simply increasing model size (4B → 27B) or deepening reasoning (low → high reasoning budget) yields limited gains; PCE's structured uncertainty handling is complementary to, rather than a substitute for, scaling.
4. **User study validation**: Across 12 participants, PCE received the highest ratings on both efficiency and trust; selective communication was preferred over both "always communicate" and "never communicate" strategies.

## Highlights & Insights

1. **Paradigm shift**: From "communication-driven coordination" to "structured assumption reasoning," demoting communication from a search mechanism to an ordinary option within the action space.
2. **Assumptions as first-class citizens**: For the first time, implicit assumptions in LLM reasoning are explicitly modeled as decision variables—a concise yet powerful elevation of abstraction.
3. **Essential distinction from ToT/CoTS**: ToT searches over reasoning-step space; CoTS searches over a joint reasoning-action space using communication; PCE searches over **assumption space**—the trees represent fundamentally different things.
4. **Consistent three-way validation**: Quantitative results (two benchmarks), qualitative analysis (case studies), and user study all support the core claims.

## Limitations & Future Work

1. **LLM-generated assumptions**: The quality and coverage of assumptions depend on the LLM's commonsense reasoning capability, which may omit critical hypotheses.
2. **LLM-estimated scoring**: Both likelihood and gain are estimated by the LLM rather than derived from true probabilities, potentially introducing systematic bias.
3. **Validation limited to simulated household environments**: C-WAH and TDW-MAT are challenging but represent a narrow range of scenario types.
4. **Fixed tree depth**: $D=3$ may be insufficient for complex, long-horizon tasks; adaptive depth strategies warrant exploration.
5. **Two-agent constraint**: Large-scale validation with more than two agents has not yet been conducted.

## Related Work & Insights

- **Relationship to CoELA/REVECA**: These methods exchange state and plan information through dialogue; PCE replaces the majority of such communication with internal structured assumption reasoning.
- **Distinction from Tree of Thoughts**: Nodes in ToT's tree represent reasoning steps (cognitive space); nodes in PCE's tree represent environmental assumptions (probabilistic state space).
- **Relationship to DEC-POMDP**: PCE can be viewed as a practical approach to approximating Bayesian inference under the DEC-POMDP framework using LLMs.
- **Broader insight**: Structuring LLM-generated free text into evaluable formal representations is a general direction for enhancing LLM decision-making, applicable beyond multi-agent settings.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The paradigm shift of "assumptions as decision variables" carries far-reaching implications.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Full coverage across two benchmarks, three backbones, component ablations, scaling analysis, and user study.
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear, method motivation is well-grounded, and differentiation from related work is precise.
- Value: ⭐⭐⭐⭐⭐ — High practical utility (80%+ communication reduction), strong generality (consistent gains across multiple backbones), and deep insights.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Predicting LLM Reasoning Performance with Small Proxy Models](predicting_llm_reasoning_performance_with_small_proxy_models.md)
- [\[AAAI 2026\] Uncertainty Under the Curve: A Sequence-Level Entropy Area Metric for Reasoning LLMs](../../AAAI2026/llm_nlp/uncertainty_under_the_curve_a_sequence-level_entropy_area_metric_for_reasoning_l.md)
- [\[ICLR 2026\] The Path of Least Resistance: Guiding LLM Reasoning Trajectories for Efficient Consistency](the_path_of_least_resistance_guiding_llm_reasoning_trajectories_for_efficient_co.md)
- [\[ACL 2026\] Iterative Formalization and Planning in Partially Observable Environments](../../ACL2026/llm_nlp/iterative_formalization_and_planning_in_partially_observable_environments.md)
- [\[AAAI 2026\] Scalable and Accurate Graph Reasoning with LLM-Based Multi-Agents](../../AAAI2026/llm_nlp/scalable_and_accurate_graph_reasoning_with_llm-based_multi-agents.md)

</div>

<!-- RELATED:END -->

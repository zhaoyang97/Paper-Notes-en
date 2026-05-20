---
title: >-
  [Paper Note] Iterative Formalization and Planning in Partially Observable Environments
description: >-
  [ACL 2026][LLM/NLP][partially observable environments] This paper proposes PDDLego+, a framework that enables LLMs to iteratively generate and refine PDDL (Planning Domain Definition Language) representations in partiall…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "partially observable environments"
  - "PDDL formalization"
  - "iterative planning"
  - "LLM-as-Formalizer"
  - "error refinement"
date: 2026-05-08
content_hash: 9147d32f1e615bd2
---

# Iterative Formalization and Planning in Partially Observable Environments

**Conference**: ACL 2026
**arXiv**: [2505.13126](https://arxiv.org/abs/2505.13126)  
**Code**: [GitHub](https://github.com/zharry29/pddlego-plus)  
**Area**: LLM NLP / AI Planning
**Keywords**: partially observable environments, PDDL formalization, iterative planning, LLM-as-Formalizer, error refinement

## TL;DR

This paper proposes PDDLego+, a framework that enables LLMs to iteratively generate and refine PDDL (Planning Domain Definition Language) representations in partially observable environments. Through a two-phase error refinement loop (solver error + simulation error), the framework achieves effective planning without fine-tuning or in-context demonstrations.

## Background & Motivation

**Background**: Leveraging large language models for planning is a prominent direction in AI planning research. Existing approaches fall into two categories: LLM-as-planner (directly generating action plans) and LLM-as-formalizer (formalizing environments into PDDL and delegating planning to classical solvers). The latter is favored for its interpretability and controllability, yet the vast majority of prior work focuses exclusively on fully observable environments.

**Limitations of Prior Work**: Real-world planning scenarios—such as robotic exploration of unknown rooms or web agent navigation—are typically partially observable: agents perceive only local observations and cannot generate complete plans in a single pass. The few works that address partially observable settings suffer from three deficiencies: (1) they assume partial planning representations are given (e.g., predefined predicates or domain files); (2) they rely on one-shot formalization rather than iterative refinement; and (3) they depend on existing trajectories as in-context demonstrations.

**Key Challenge**: Planning languages such as PDDL are grounded in the closed-world assumption, which requires complete definitions of the initial state and goal. This stands in direct contradiction to the nature of partially observable environments, where information is revealed incrementally.

**Goal**: To design a framework that requires no fine-tuning, no demonstrations, and no pre-specified domain files, enabling LLMs to iteratively construct complete PDDL representations and accomplish planning tasks in partially observable environments through exploration and error-driven refinement.

**Core Idea**: Decompose the partially observable problem into a series of fully observable subproblems. At each step, the agent generates a local PDDL representation from current observations, plans using a classical solver, executes the plan, and iteratively updates the formalization based on new observations and error feedback.

## Method

### Overall Architecture

The core of PDDLego+ is an iterative loop of generate→solve→execute→update: (1) the LLM generates a Domain File ($\mathbb{DF}$, defining types, predicates, and actions) and a Problem File ($\mathbb{PF}$, defining objects, initial state, and goal) from current observations; (2) a formal solver (Fast Downward) searches for an action plan; (3) the plan is executed in a simulated environment; (4) PDDL is updated based on new observations or repaired based on errors. Unlike PDDLego, PDDLego+ jointly infers both DF and PF without assuming the domain file is given.

### Key Designs

1. **Two-Phase Error Refinement**

    - Function: Handles two categories of errors arising during PDDL generation.
    - Mechanism: The inner loop addresses solver errors (solver failures caused by syntactic or semantic errors in PDDL), while the outer loop addresses simulation errors (plan execution failures in the simulator). Formally, solver error refinement is expressed as $\mathrm{df}_i^{j,k+1}, \mathrm{pf}_i^{j,k+1} = \text{LLM}(\mathrm{err}_{\text{sol}}, \mathrm{df}_i^{j,k}, \mathrm{pf}_i^{j,k})$; simulation error refinement as $\mathrm{df}_i^{j+1}, \mathrm{pf}_i^{j+1} = \text{LLM}(\mathrm{err}_{\text{sim}}, \mathrm{df}_i^j, \mathrm{pf}_i^j)$.
    - Design Motivation: The two error types are qualitatively different—solver errors are immediate syntactic or logical issues, while simulation errors reflect deeper semantic problems (e.g., missing preconditions) that require hierarchical treatment.

2. **Goal Decomposition and Subgoal Prediction**

    - Function: Decomposes an unreachable global goal into locally achievable subgoals.
    - Mechanism: Two prompting templates are provided—a simple prompt (coarse goal decomposition guidance) and a detailed prompt (PDDL goal templates such as `(:goal (at ?location))` with placeholders for the LLM to fill). At each timestep, the LLM predicts a locally reachable subgoal.
    - Design Motivation: In partially observable environments, the global goal is typically not directly achievable and must be approached incrementally through exploration.

3. **Full Domain + Problem Inference**

    - Function: Jointly infers both DF and PF from natural language observations, without assuming DF is given.
    - Mechanism: The LLM receives textual observations (e.g., "You are in the kitchen; there is a closed door to the east") and generates complete PDDL type definitions, predicates, and action semantics (DF), as well as object instances, initial states, and goals (PF).
    - Design Motivation: DF inference is a substantially harder task (analogous to synthesizing classes and functions rather than merely function calls), yet in realistic scenarios a domain file cannot be assumed to be provided in advance.

### Domain Knowledge Reuse

Domain files produced from successful trials can be reused as learned domain knowledge for future tasks. Experiments employ RAG to retrieve DFs from prior successful trials; fixing the DF and having the LLM predict only the PF yields notable success rate improvements on certain models.

## Key Experimental Results

### Main Results

Evaluation is conducted on two text-based simulated environments—CoinCollector (navigation task) and ALFWorld (object manipulation task):

| Method | CoinCollector (o3-mini) | ALFWorld (o3-mini) |
|--------|------------------------|-------------------|
| PlanGen (LLM-as-planner) | 52% | 5% |
| PDDLego (no refinement) | 49% | 3% |
| PDDLego+ (Ours) | **86%** | **38%** |

| Model | CoinCollector PlanGen / PDDLego+ | ALFWorld PlanGen / PDDLego+ |
|-------|----------------------------------|------------------------------|
| DeepSeek-R1 | ~55% / ~75% | ~8% / ~25% |
| GPT-4.1 | ~60% / ~55% | ~3% / ~20% |
| o3-mini | 52% / 86% | 5% / 38% |
| o4-mini | ~65% / ~80% | ~10% / ~30% |

### Ablation Study

- **Complexity Robustness**: As the number of rooms in CoinCollector increases from 3 to 11, PDDLego+ maintains stable success rates while PlanGen and PDDLego degrade progressively.
- **Goal Prompt Ablation**: The detailed prompt outperforms the simple prompt; however, PDDLego+ with the simple prompt still substantially outperforms baselines.
- **Domain Knowledge Reuse**: Retrieving DFs via RAG improves success rates for DeepSeek-R1 and GPT-4.1, while o3-mini shows a slight decline (its DF generation capability is already sufficiently strong).

### Key Findings

- PDDLego+ outperforms PlanGen on all models in the more complex ALFWorld environment, demonstrating the advantage of formalization-based methods in complex planning tasks.
- The majority of errors are solver errors (PDDL syntactic issues) rather than simulation errors; o3-mini achieves the highest error recovery rate.
- Error analysis reveals that the primary bottleneck lies in semantic errors within PF: hallucinated facts, unreachable goals, and forgetting previously observed information.

## Highlights & Insights

- **Formalization is viable in partially observable environments**: This work provides the first systematic demonstration that LLM-as-formalizer is effective in partially observable settings, challenging the assumption that PDDL is applicable only to fully observable environments.
- **Interpretability advantage**: Unlike LLM-as-planner, every failure in PDDLego+ can be attributed to a specific PDDL error, enabling causal error analysis.
- **Transferable domain knowledge**: DFs generated from successful trials are reusable, highlighting a unique advantage of formalization-based methods for knowledge accumulation.
- **Advantage of reasoning models**: Reasoning-oriented models such as o3-mini substantially outperform standard models in PDDL generation, consistent with findings reported by Huang & Zhang (2025).

## Limitations & Future Work

- The framework relies on environments providing informative error messages; it may fail in settings where error feedback is ambiguous.
- Prompts must be tailored to specific environments, limiting generalization to unseen domains.
- High-capability LLMs (e.g., o3-mini / DeepSeek-R1) are required, and multiple inference calls lead to substantial computational cost.
- The highest success rate on ALFWorld is only 38%, leaving considerable room for improvement.
- Hallucinated facts and information forgetting in PF remain the primary bottleneck, necessitating better world-state tracking mechanisms.

## Related Work & Insights

- **vs. PDDLego (Zhang et al. 2024)**: PDDLego assumes DF is given and includes no error refinement; PDDLego+ infers the full DF+PF and introduces the two-phase refinement loop.
- **vs. PlanGen (LLM-as-planner)**: PlanGen is occasionally superior on simpler tasks (direct action generation without formalization overhead), but PDDLego+ comprehensively outperforms it on complex tasks (ALFWorld).
- **vs. ReAct**: PDDLego+ can be viewed as a formalized extension of ReAct—replacing natural language reasoning with PDDL to obtain formal guarantees.

## Rating

- Novelty: ⭐⭐⭐⭐ First work to achieve complete iterative PDDL formalization in partially observable environments; the two-phase refinement loop is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers two environments, four models, multi-dimensional analysis, and detailed error dissection, though the success rate on ALFWorld remains low.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated, method formalization is complete, and error analysis is thorough.
- Value: ⭐⭐⭐⭐ Provides a viable path toward applying LLM-driven formal planning in realistic scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] From Assumptions to Actions: Turning LLM Reasoning into Uncertainty-Aware Planning](../../ICLR2026/llm_nlp/from_assumptions_to_actions_turning_llm_reasoning_into_uncertainty-aware_plannin.md)
- [\[NeurIPS 2025\] SYMPHONY: Synergistic Multi-agent Planning with Heterogeneous Language Model Assemblies](../../NeurIPS2025/llm_nlp/symphony_synergistic_multi-agent_planning_with_heterogeneous_language_model_asse.md)
- [\[NeurIPS 2025\] Planning without Search: Refining Frontier LLMs with Offline Goal-Conditioned RL](../../NeurIPS2025/llm_nlp/planning_without_search_refining_frontier_llms_with_offline_goal-conditioned_rl.md)
- [\[ACL 2026\] Lost in the Prompt Order: Revealing the Limitations of Causal Attention in Language Models](lost_in_the_prompt_order_revealing_the_limitations_of_causal_attention_in_langua.md)
- [\[ACL 2026\] Route to Rome Attack: Directing LLM Routers to Expensive Models via Adversarial Suffixes](route_to_rome_attack_directing_llm_routers_to_expensive_models_via_adversarial_s.md)

</div>

<!-- RELATED:END -->

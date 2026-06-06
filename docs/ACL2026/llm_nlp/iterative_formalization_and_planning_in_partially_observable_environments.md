---
title: >-
  [Paper Note] Iterative Formalization and Planning in Partially Observable Environments
description: >-
  [ACL 2026][LLM/NLP][Partially Observable Environments] The proposed PDDLego+ framework enables LLMs to iteratively generate and refine PDDL (Planning Domain Definition Language) representations in partially observable en…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "Partially Observable Environments"
  - "PDDL Formalization"
  - "Iterative Planning"
  - "LLM-as-Formalizer"
  - "Error Correction"
date: 2026-05-08
content_hash: 6fca22a46bcd0477
---

# Iterative Formalization and Planning in Partially Observable Environments

**Conference**: ACL 2026  
**arXiv**: [2505.13126](https://arxiv.org/abs/2505.13126)  
**Code**: [GitHub](https://github.com/zharry29/pddlego-plus)  
**Area**: LLM NLP / AI Planning  
**Keywords**: Partially Observable Environments, PDDL Formalization, Iterative Planning, LLM-as-Formalizer, Error Correction

## TL;DR

The proposed PDDLego+ framework enables LLMs to iteratively generate and refine PDDL (Planning Domain Definition Language) representations in partially observable environments. Through a two-phase error refinement loop (solver error + simulation error), it achieves effective planning without fine-tuning or exemplars.

## Background & Motivation

**Background**: Utilizing Large Language Models (LLMs) for planning is a prominent direction in AI planning. Current methods are primarily divided into two categories: LLM-as-planner (direct generation of action plans) and LLM-as-formalizer (formalizing the environment into PDDL for traditional solvers). The latter is favored for its interpretability and controllability, yet most existing work focuses solely on fully observable environments.

**Limitations of Prior Work**: Real-world planning scenarios (e.g., robots exploring unknown rooms, web agent operations) are typically partially observable—the agent only sees local observations and cannot generate a complete plan at once. Few works addressing partially observable environments suffer from three drawbacks: (1) assuming partial planning representations are known (e.g., predefined predicates or domain files); (2) using one-shot formalization rather than iterative refinement; (3) relying on existing trajectories as in-context examples.

**Key Challenge**: Planning languages like PDDL are based on the closed-world assumption—requiring complete definitions of initial states and goals. This directly contradicts the nature of partially observable environments where information is revealed incrementally.

**Goal**: To design a framework that requires no fine-tuning, no exemplars, and no preset domain files, allowing LLMs to incrementally build complete PDDL representations and complete planning tasks through iterative exploration and error refinement in partially observable environments.

**Core Idea**: Decompose the partially observable problem into a sequence of fully observable subproblems. For each step, a local PDDL is generated based on current observations, planned and executed via a solver, and iteratively updated based on new observations and error feedback.

## Method

### Overall Architecture

The core process of PDDLego+ is a "Generate → Solve → Execute → Update" iterative loop: (1) The LLM generates the Domain File ($\mathbb{DF}$, defining types, predicates, and actions) and the Problem File ($\mathbb{PF}$, defining objects, initial states, and goals) based on current observations; (2) A symbolic solver (Fast Downward) searches for an action plan; (3) The plan is executed in a simulated environment; (4) The PDDL is updated based on new observations or refined based on errors. Unlike PDDLego, PDDLego+ infers both DF and PF simultaneously without assuming the domain file is known.

### Key Designs

1. **Two-Phase Error Refinement**

    - **Function**: Handles two types of errors during PDDL generation.
    - **Mechanism**: The inner loop handles solver errors (solver failure due to PDDL syntax/semantic errors), while the outer loop handles simulation errors (plan execution failure in the simulator). Formalized as: solver error refinement $\mathrm{df}_i^{j,k+1}, \mathrm{pf}_i^{j,k+1} = \text{LLM}(\mathrm{err}_{\text{sol}}, \mathrm{df}_i^{j,k}, \mathrm{pf}_i^{j,k})$; simulation error refinement $\mathrm{df}_i^{j+1}, \mathrm{pf}_i^{j+1} = \text{LLM}(\mathrm{err}_{\text{sim}}, \mathrm{df}_i^j, \mathrm{pf}_i^j)$.
    - **Design Motivation**: The two error types differ in nature—solver errors are immediate syntax/logic issues, while simulation errors are deeper semantic issues (e.g., missing preconditions), requiring layered processing.

2. **Goal Decomposition**

    - **Function**: Decomposes unreachable global goals into currently reachable subgoals.
    - **Mechanism**: Provides two prompt templates—a simple prompt (coarse goal decomposition guidance) and a detailed prompt (providing PDDL goal templates like `(:goal (at ?location))` for the LLM to fill). The LLM predicts a locally reachable subgoal at each timestep.
    - **Design Motivation**: Global goals in partially observable environments are usually not directly achievable and must be approached incrementally through exploration.

3. **Full Domain+Problem Inference**

    - **Function**: Infers both DF and PF simultaneously from natural language observations instead of assuming DF is known.
    - **Mechanism**: The LLM receives text observations (e.g., "You are in the kitchen, there is a closed door to the east") and generates complete PDDL type definitions, predicates, action semantics (DF), object instances, initial states, and goals (PF).
    - **Design Motivation**: DF inference is a more difficult task (analogous to synthesizing classes and functions vs. synthesizing function calls), but DF cannot be pre-given in real-world scenarios.

### Domain Knowledge Reuse

DFs produced after successful trials can be reused as "learned domain knowledge" for future tasks. The experiment utilizes RAG to retrieve DFs from historically successful trials. Fixing the DF and only letting the LLM predict the PF significantly improves the success rate for certain models.

## Key Experimental Results

### Main Results

Evaluated on two text simulation environments: CoinCollector (navigation) and ALFWorld (object manipulation):

| Method | CoinCollector (o3-mini) | ALFWorld (o3-mini) |
|------|------------------------|-------------------|
| PlanGen (LLM-as-planner) | 52% | 5% |
| PDDLego (No refinement) | 49% | 3% |
| PDDLego+ (Ours) | **86%** | **38%** |

| Model | CoinCollector PlanGen/PDDLego+ | ALFWorld PlanGen/PDDLego+ |
|------|-------------------------------|--------------------------|
| DeepSeek-R1 | ~55% / ~75% | ~8% / ~25% |
| GPT-4.1 | ~60% / ~55% | ~3% / ~20% |
| o3-mini | 52% / 86% | 5% / 38% |
| o4-mini | ~65% / ~80% | ~10% / ~30% |

### Ablation Study

- **Complexity Robustness**: As the number of rooms in CoinCollector increases from 3 to 11, the success rate of PDDLego+ remains stable, while PlanGen and PDDLego gradually decline.
- **Goal Prompt Ablation**: Detailed prompts outperform simple prompts, but PDDLego+ still significantly outperforms baselines under simple prompts.
- **Domain Knowledge Reuse**: Using RAG-retrieved DFs improves success rates for DeepSeek-R1 and GPT-4.1, while o3-mini shows a slight decrease (indicating already strong DF generation capabilities).

### Key Findings

- PDDLego+ outperforms PlanGen across all models on the more complex ALFWorld, demonstrating the advantage of formal methods in complex planning tasks.
- Most errors are solver errors (PDDL syntax issues) rather than simulation errors; o3-mini demonstrates the highest error refinement rate.
- Error analysis shows the main bottleneck lies in semantic errors of the PF: hallucinating facts, unreachable goals, and forgetting observed information.

## Highlights & Insights

- **Feasibility of Formal Methods in Partial Observability**: Systematically proves for the first time the effectiveness of LLM-as-formalizer in partially observable environments, challenging the notion that "PDDL is only for fully observable environments."
- **Interpretability Advantage**: Unlike LLM-as-planner, every failure in PDDLego+ can be traced back to specific PDDL errors, supporting causal error analysis.
- **Transferable Domain Knowledge**: DFs generated from successful trials can be reused, showcasing the unique advantage of formal methods in knowledge accumulation.
- **Advantage of Reasoning Models**: Reasoning models like o3-mini significantly outperform standard models in PDDL generation, aligning with findings by Huang & Zhang (2025).

## Limitations & Future Work

- Relies on environments providing information-rich error messages; may fail in environments with vague error feedback.
- Requires environment-specific prompt engineering, limiting generalization to unknown domains.
- Requires high-capability LLMs (e.g., o3-mini/DeepSeek-R1) and multiple calls, resulting in high computational costs.
- Highest success rate on ALFWorld is only 38%, leaving significant room for improvement.
- Hallucinated facts and forgetting in the PF remain major bottlenecks, requiring better world-state maintenance mechanisms.

## Related Work & Insights

- **vs PDDLego (Zhang et al. 2024)**: PDDLego assumes a known DF and lacks error refinement; PDDLego+ infers the complete DF+PF and introduces a two-phase refinement loop.
- **vs PlanGen (LLM-as-planner)**: PlanGen is sometimes better for simple tasks (direct action generation without formalization), but PDDLego+ leads decisively in complex tasks (ALFWorld).
- **vs ReAct**: PDDLego+ can be viewed as a formalized upgrade of ReAct—replacing natural language reasoning with PDDL to obtain formal guarantees.

## Rating

- Novelty: ⭐⭐⭐⭐ First to achieve complete iterative PDDL formalization in partially observable environments; clever two-phase refinement design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two environments, four models, multi-dimensional analysis, and error dissection, though ALFWorld success rates remain low.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, complete formalization of methods, and detailed error analysis.
- Value: ⭐⭐⭐⭐ Provides a feasible path for applying LLM-driven formal planning in real-world scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] From Assumptions to Actions: Turning LLM Reasoning into Uncertainty-Aware Planning](../../ICLR2026/llm_nlp/from_assumptions_to_actions_turning_llm_reasoning_into_uncertainty-aware_plannin.md)
- [\[ICML 2026\] SAC-Opt: Semantic Anchors for Iterative Correction in Optimization Modeling](../../ICML2026/llm_nlp/sac-opt_semantic_anchors_for_iterative_correction_in_optimization_modeling.md)
- [\[NeurIPS 2025\] SYMPHONY: Synergistic Multi-agent Planning with Heterogeneous Language Model Assemblies](../../NeurIPS2025/llm_nlp/symphony_synergistic_multi-agent_planning_with_heterogeneous_language_model_asse.md)
- [\[ACL 2026\] Wait, There's a Way Out: A Decision Mechanism for Conversational Derailment Prediction](wait_theres_a_way_out_a_decision_mechanism_for_forecasting_conversational_derail.md)
- [\[ACL 2026\] Can AI Be a Good Peer Reviewer? A Survey of Peer Review Process, Evaluation, and the Future](can_ai_be_a_good_peer_reviewer_a_survey_of_peer_review_process_evaluation_and_th.md)

</div>

<!-- RELATED:END -->

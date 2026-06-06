---
title: >-
  [Paper Note] Lifting Traces to Logic: Programmatic Skill Induction with Neuro-Symbolic Learning for Long-Horizon Agentic Tasks
description: >-
  [ICML 2026][LLM Reasoning][Skill Induction] NSI "lifts" LLM agent interaction traces into neuro-symbolic workflow graphs with explicit conditional branches and dynamic variable binding…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Skill Induction"
  - "First-Order Logic"
  - "Workflow Graph"
  - "Reflective Planning"
  - "ALFWorld"
date: 2026-05-08
content_hash: 459d7c38f4456f6f
---

# Lifting Traces to Logic: Programmatic Skill Induction with Neuro-Symbolic Learning for Long-Horizon Agentic Tasks

**Conference**: ICML 2026  
**arXiv**: [2605.01293](https://arxiv.org/abs/2605.01293)  
**Code**: None  
**Area**: LLM Agent / Neuro-Symbolic / Long-Horizon Planning  
**Keywords**: Skill Induction, First-Order Logic, Workflow Graph, Reflective Planning, ALFWorld

## TL;DR
NSI "lifts" LLM agent interaction traces into neuro-symbolic workflow graphs with explicit conditional branches and dynamic variable binding, evolving skills from stateless scripts into state-aware logical programs. Achieves 98.0 / 76.5 / 95.2 success rates on ALFWorld / WebShop / TextCraft, comprehensively outperforming programmatic skill baselines like ASI and AWM.

## Background & Motivation

**Background**: Foundation model-driven agents increasingly rely on "skill induction" for long-horizon tasks—distilling past successful traces into reusable Python functions (e.g., ASI, AWM), thus expanding the action space and avoiding repeated reasoning. This essentially solidifies System-2 reasoning into System-1 muscle memory.

**Limitations of Prior Work**: Current skills are either textual workflows (AWM, non-executable) or stateless parameterized scripts (ASI, e.g., `Open(Receptacle) → Pick(Object)`). These scripts fail outright when the environment deviates slightly—for example, if there is no "apple" in the fridge, the script still mechanically executes Pick, without querying the state or making decisions.

**Key Challenge**: Programmatic skills do not match the "conditionality" of real environments—LLMs, when synthesizing code, only see a linear trace and thus hardcode all actions into a sequential structure, unable to express branch logic like "if apple exists after opening the fridge, pick it; otherwise, search elsewhere." This lack of expressiveness causes ASI to score only 7.7 on WebShop (far below AWM's 49.2).

**Goal**: Upgrade skills from linear scripts to graph programs with explicit control flow and dynamic variable binding; enable agents to induce highly generalizable logic from few demonstrations (single trace), and continuously repair skills during deployment via reflection.

**Key Insight**: The authors adopt a neuro-symbolic perspective—LLMs excel at mapping perception to semantic predicates (System-1 like), while symbolic interpreters excel at executing precise if/loop logic (System-2 like). Decoupling the two preserves the LLM's flexible perception and the program's verifiability.

**Core Idea**: Use a "trace-to-logic" lifting mechanism to abstract demonstrations into first-order logic and workflow graphs; employ a two-stage algorithm (intra-trajectory consistency + inter-trajectory merging) to induce global skills; at runtime, use reflective planning to graft failed subgraphs onto failure nodes, enabling skill self-evolution.

## Method

### Overall Architecture
A skill is defined as $\pi_\omega = (\theta_\omega, \phi_\omega, G_\omega)$: invocation parameters $\theta_\omega$, neuro-perception module $\phi_\omega$ (LLM as semantic parser, mapping raw observations to symbolic state $Z_t$), and symbolic execution graph $G_\omega$ (executed node-by-node by an interpreter). The pipeline has three stages: (1) NeSy Grounding maps environment perception to FOL predicate space; (2) Offline Induction abstracts successful traces into modular skills and stores them; (3) Online Evolution uses reflective planning to repair skill feasibility domains and logic branches based on runtime feedback.

### Key Designs

1. **Neuro-symbolic Workflow Representation with Four Node Types**:

    - **Function**: Extends skills from linear scripts to graph programs, with nodes categorized as DataOp (dynamic variable binding), CheckOp/LoopOp (control flow), PrimitiveOp (atomic actions), TerminalOp (success/failure termination).
    - **Mechanism**: DataOp synthesizes a program $f_v: \mathcal{C} \times \mathcal{Z} \to \mathcal{C}$, e.g., `target = select_one(x, is_type(x, 'apple') ∧ contains(loc, x))`; CheckOp synthesizes Boolean predicates like `is_closed(y) ∧ locates('agent', y)`; PrimitiveOp references variables bound by upstream DataOps as parameters; TerminalOp auto-generates diagnostic info like "$\nexists x, \text{is\_type}(x, \text{apple})$" on failure.
    - **Design Motivation**: Modularity enables local repair—agents can rewrite a specific CheckOp without regenerating the entire skill; explicit logic nodes force the LLM to formalize "why" and "when," preventing unconditional skipping of checks.

2. **Empirical Programmatic Consistency as Induction Objective**:

    - **Function**: Uses historical trace consistency to replace online verification in partially observable environments.
    - **Mechanism**: A skill $\pi_\omega$ is "consistent" on trajectory $\tau$ at state $s_h$ iff all non-empty actions $\hat{a}_k$ it produces from $s_h$ match expert actions $a^\ast_{h+m(k)}$. The objective $\max_{\pi_\omega} \sum_\tau |\widehat{\mathcal{R}}_{\pi_\omega}^\tau| - \lambda |\pi_\omega|$ maximizes empirical coverage while minimizing program complexity (MDL principle).
    - **Design Motivation**: Embodied/web environments often cannot be perfectly reset, making online verification infeasible; trace consistency avoids environment resets while enforcing strong "faithful replay" constraints, and the MDL penalty suppresses overfitting.

3. **Two-Stage Progressive Induction + Four Structural Operators**:

    - **Function**: Decomposes skill synthesis into "local expert fitting + global merging," using Branching / Crossover / Lifting / LoopFold operators to search the program space.
    - **Mechanism**: Stage 1 synthesizes local $\pi_\tau$ for each trace, inserting CheckOp branches at each counterexample $s_{\text{err}}$ via the Branching operator; Stage 2 uses a greedy algorithm to iteratively merge—finds the hardest-to-cover trajectory's $\pi_{\text{hard}}$, consolidates with the global skill via $\mathtt{Consolidate}(\pi_{\text{glb}}, \pi_{\text{hard}})$, accepting only if the feasible region strictly expands. Crossover grafts subgraphs; Lifting promotes constants to parameters for cross-instance generalization; LoopFold abstracts repeated structures into LoopOps.
    - **Design Motivation**: Directly optimizing the global objective leads to combinatorial explosion in program space; specializing first and then generalizing allows the LLM to resolve local conflicts one at a time ("divide and conquer"), improving efficiency and interpretability.

### Loss & Training

NSI does not update LLM parameters; all "training" occurs in program space. Stage 1 uses iterative consistency checking and LLM-based program synthesis; Stage 2 uses greedy feasibility dominance for validation and updates. In the online phase, Reflective Planning detects failures, invokes the LLM to generate corrective trajectories, and merges them into the skill graph using the same structural operators; new branches are tentative and only solidify after repeated success. GPT-4o is used as the backbone with temperature set to 0 for reproducibility.

## Key Experimental Results

### Main Results

| Method | ALFWorld SR (%) | WebShop Score | WebShop SR (%) | TextCraft SR (%) |
|--------|-----------------|--------------|----------------|------------------|
| ReAct | 85.8 | 44.0 | 20.0 | 62.0 |
| Reflexion | 84.3 | 40.8 | 23.0 | 59.0 |
| AWM | 91.3 | 49.2 | 30.0 | 92.5 |
| ASI | 70.6 | 7.7 | 7.5 | 77.8 |
| **NSI w/o online honing** | **93.5** | **58.8** | **30.5** | 78.5 |
| **NSI (Ours)** | **98.0** | **76.5** | **44.5** | **95.2** |

### Ablation Study

| Configuration | Phenomenon | Interpretation |
|---------------|------------|----------------|
| ASI (no logic branch) | WebShop Score only 7.7 | Linear scripts cannot express conditional logic at all |
| NSI offline only | Already surpasses all baselines | Offline-induced logic representation is already strong |
| NSI full (with online honing) | SOTA across all three benchmarks | Reflective planning turns runtime failures into permanent capabilities |
| Avg. atomic steps / skill | NSI $\approx 7.4$ vs ASI lower | NSI compresses 7+ steps of logic into a single skill |

### Key Findings
- Formalizing experience as scripts (ASI) is actually worse than AWM's pure textual workflows—demonstrating that "under-expressive programs" are inferior to "non-executable text," validating the necessity of logic branches.
- ALFWorld "long-horizon collapse": baselines' success rates drop to 0 for $>22$ steps, while NSI maintains performance at 53+ steps by compressing 7.4 atomic actions into a single skill, thus "compressing" the planning horizon.
- Reflexion's textual memory gains over ReAct are negligible, further indicating that the bottleneck in long-horizon tasks is not "recall" but "robust execution."

## Highlights & Insights
- The "trace-to-logic lifting" concept is highly general—any LLM agent can upgrade demonstrations into verifiable programs this way, enabling cross-task transfer to more complex scenarios like SWE-bench and robotic manipulation.
- Reflective Planning converts failure signals into "local subgraph grafting," essentially a continual learning version of program synthesis, avoiding catastrophic forgetting (the skill graph grows monotonically).
- The combination of MDL penalty and four structural operators gives LLMs a clear bias when searching program space (coverage and simplicity), serving as a reusable template for future "program synthesis + LLM" methods.

## Limitations & Future Work
- All experiments assume the environment provides an enumerable predicate vocabulary (ALFWorld / WebShop have structured feedback); in real open worlds, predicate discovery itself is a challenge.
- GPT-4o as the synthesizer is costly; the authors do not discuss whether smaller models can drive this synthesizer.
- Online honing relies on the LLM to propose corrective trajectories; if the LLM suggests incorrect recovery plans, grafting them may pollute the skill graph. The paper mitigates this with a "tentative → solidify" two-stage acceptance but does not quantify failure rates.
- The improvement on TextCraft is relatively marginal (95.2 vs AWM's 92.5), suggesting that for tasks where "recursive decomposition" suffices, the marginal value of logic branching is limited.

## Related Work & Insights
- **vs ASI**: ASI synthesizes skills as parameterized scripts, lacking explicit control flow nodes like CheckOp/LoopOp; NSI invents predicates to synthesize branch discriminators, boosting WebShop score from 7.7 (ASI) to 76.5.
- **vs AWM**: AWM skills are textual templates, non-executable; NSI skills are symbolic graph programs, verifiable and precisely executable by interpreters.
- **vs Agentic Workflow Generation (AFlow, GPTSwarm)**: These assemble predefined nodes (Debate / Voting); NSI's nodes are "invented" internal logic, with finer granularity and stronger generalization.
- **vs Classic RL Options (Sutton 1999)**: Traditional options are black-box neural policies requiring extensive parameter optimization; NSI skills are readable Python code, naturally aligned with LLM generation capabilities.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Implements the neuro-symbolic idea of "lifting traces to logical programs" with LLMs and deploys it in agent frameworks
- Experimental Thoroughness: ⭐⭐⭐⭐ Three mainstream agent benchmarks + thorough ablation + long-horizon analysis
- Writing Quality: ⭐⭐⭐⭐ Clear explanation of algorithms and node definitions, though some sections are highly formalized
- Value: ⭐⭐⭐⭐⭐ Provides a new paradigm for expressive skill learning in LLM agents

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Segment-Level Attribution for Selective Learning of Long Reasoning Traces](../../ICLR2026/llm_reasoning/segment-level_attribution_for_selective_learning_of_long_reasoning_traces.md)
- [\[ACL 2026\] FS-Researcher: Test-Time Scaling for Long-Horizon Research Tasks with File-System-Based Agents](../../ACL2026/llm_reasoning/fs-researcher_test-time_scaling_for_long-horizon_research_tasks_with_file-system.md)
- [\[ICML 2026\] ToolMATH: A Math Tool Benchmark for Realistic Long-Horizon Multi-Tool Reasoning](toolmath_a_math_tool_benchmark_for_realistic_long-horizon_multi-tool_reasoning.md)
- [\[ICLR 2026\] The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs](../../ICLR2026/llm_reasoning/the_illusion_of_diminishing_returns_measuring_long_horizon_execution_in_llms.md)
- [\[ICML 2026\] Many-Shot CoT-ICL: Making In-Context Learning Truly Learn](many-shot_cot-icl_making_in-context_learning_truly_learn.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Lifting Traces to Logic: Programmatic Skill Induction with Neuro-Symbolic Learning for Long-Horizon Agentic Tasks
description: >-
  [ICML 2026][LLM Agent][Skill Induction] NSI "lifts" LLM agent interaction traces into neuro-symbolic workflow graphs with explicit conditional branches and dynamic variable binding. This evolves skills from stateless scr…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "Skill Induction"
  - "First-Order Logic"
  - "Workflow Graph"
  - "Reflective Planning"
  - "ALFWorld"
date: 2026-05-08
content_hash: fadd8f67e7b3d980
---

# Lifting Traces to Logic: Programmatic Skill Induction with Neuro-Symbolic Learning for Long-Horizon Agentic Tasks

**Conference**: ICML 2026  
**arXiv**: [2605.01293](https://arxiv.org/abs/2605.01293)  
**Code**: None  
**Area**: LLM Agent / Neuro-Symbolic / Long-Horizon Planning  
**Keywords**: Skill Induction, First-Order Logic, Workflow Graph, Reflective Planning, ALFWorld

## TL;DR
NSI "lifts" LLM agent interaction traces into neuro-symbolic workflow graphs with explicit conditional branches and dynamic variable binding. This evolves skills from stateless scripts into state-aware logical programs, achieving success rates of 98.0 / 76.5 / 95.2 on ALFWorld / WebShop / TextCraft respectively, significantly outperforming programmatic skill baselines such as ASI and AWM.

## Background & Motivation

**Background**: Foundation model-driven agents in long-horizon tasks increasingly rely on "skill induction"—distilling past successful trajectories into reusable Python functions (e.g., ASI, AWM) to expand the action space and avoid redundant reasoning. This is equivalent to fossilizing System-2 thinking into System-1 muscle memory.

**Limitations of Prior Work**: Current skills are either textual workflows (AWM, non-executable) or stateless parameterized scripts (ASI, e.g., `Open(Receptacle) → Pick(Object)`). These scripts fail when minor environment deviations occur—for instance, if an "apple" is missing from the refrigerator, the script blindly executes Pick without querying the state to re-evaluate.

**Key Challenge**: There is a mismatch between programmatic skills and the "conditionality" of real environments. During code synthesis, LLMs only observe a linear trajectory, leading them to hardcode all actions into sequential structures. This lacks the expressivity for branching logic such as "if the apple exists after opening the fridge, take it; otherwise, search other locations." This lack of expressivity results in ASI scoring only 7.7 on WebShop (well below AWM's 49.2).

**Goal**: To upgrade skills from linear scripts to graph-based programs with explicit control flow and dynamic variable binding; to enable agents to induce highly generalized logic from minimal demonstrations (a single trajectory) and continuously patch them through reflection during deployment.

**Key Insight**: The authors adopt a neuro-symbolic perspective—LLMs excel at mapping perception to semantic predicates (System-1 like), while symbolic interpreters excel at executing precise if/loop logic (System-2 like). Decoupling these allows for the flexibility of LLM perception alongside the verifiability of programs.

**Core Idea**: A "trace-to-logic" lifting mechanism abstracts demonstrations into first-order logic + workflow graphs. A two-stage algorithm involving intra-trajectory consistency and inter-trajectory merging induces global skills. At runtime, reflective planning grafts failed subgraphs onto failure nodes to allow for skill self-evolution.

## Method

### Overall Architecture
A skill is defined as $\pi_\omega = (\theta_\omega, \phi_\omega, G_\omega)$, comprising call parameters $\theta_\omega$, a neuro-perception module $\phi_\omega$ (where the LLM acts as a semantic parser to convert raw observations into symbolic states $Z_t$), and a symbolic execution graph $G_\omega$ (executed node-by-node by an interpreter). The pipeline follows three phases: (1) NeSy Grounding maps environment perception to FOL predicate space; (2) Offline Induction induces modular skills from successful trajectories and populates a library; (3) Online Evolution uses reflective planning to patch the feasibility regions and logical branches of skills via runtime feedback.

### Key Designs

1.  **Neuro-Symbolic Workflow Representation with Four Node Types**:
    - Function: Expands skills from linear scripts to graph programs. Nodes are categorized as DataOp (dynamic variable binding), CheckOp/LoopOp (control flow), PrimitiveOp (atomic actions), and TerminalOp (success/failure termination).
    - Mechanism: DataOp synthesizes a program $f_v: \mathcal{C} \times \mathcal{Z} \to \mathcal{C}$, such as `target = select_one(x, is_type(x, 'apple') ∧ contains(loc, x))`; CheckOp synthesizes a Boolean discriminant like `is_closed(y) ∧ locates('agent', y)`; PrimitiveOp references variables bound by upstream DataOps as arguments; TerminalOp automatically generates diagnostic info like "$\nexists x, \text{is\_type}(x, \text{apple})$" upon failure.
    - Design Motivation: Modularity enables local patching—agents can rewrite a specific CheckOp without regenerating the entire skill. Explicit logical nodes force the LLM to formalize "why" and "when," preventing the unconditional skipping of state checks.

2.  **Induction Objective Driven by Empirical Programmatic Consistency**:
    - Function: Replaces online verification with historical trajectory consistency in partially observable environments.
    - Mechanism: A skill $\pi_\omega$ is "consistent" with a trajectory $\tau$ at state $s_h$ if and only if all non-empty actions $\hat{a}_k$ produced starting from $s_h$ match the expert actions $a^\ast_{h+m(k)}$. The objective $\max_{\pi_\omega} \sum_\tau |\widehat{\mathcal{R}}_{\pi_\omega}^\tau| - \lambda |\pi_\omega|$ simultaneously maximizes the empirical coverage area and minimizes program complexity (MDL principle).
    - Design Motivation: Embodied/web environments often cannot be perfectly reset, making online verification infeasible. Trajectory consistency provides a strong constraint of "faithful reproduction" without environment restarts, while the MDL penalty suppresses overfitting.

3.  **Two-Stage Progressive Induction + Four Structural Operators**:
    - Function: Splits skill synthesis into "local expert fitting" followed by "global merging," searching the program space using Branching, Crossover, Lifting, and LoopFold operators.
    - Mechanism: Stage 1 synthesizes a local $\pi_\tau$ for each trajectory, inserting CheckOp branches at each counterexample $s_{\text{err}}$ via the Branching operator. Stage 2 uses a greedy algorithm for iterative merging—finding $\pi_{\text{hard}}$ for the currently worst-covered trajectory and performing $\mathtt{Consolidate}(\pi_{\text{glb}}, \pi_{\text{hard}})$, accepting the merge only if it strictly expands the feasibility region. Crossover grafts subgraphs; Lifting upgrades constants to parameters for cross-instance generalization; LoopFold abstracts repetitive structures into LoopOps.
    - Design Motivation: Direct optimization of the global objective leads to combinatorial explosion in the program space. Specializing before generalizing allows the LLM to resolve one local conflict at a time, ensuring efficiency and interpretability.

### Loss & Training
NSI does not update LLM parameters; all "training" occurs in the program space. Stage 1 employs iterative consistency detection + LLM-based program synthesis. Stage 2 uses greedy feasibility dominance verification for updates. During the online phase, Reflective Planning detects failures and calls the LLM to generate a corrective trajectory, which is then merged into the skill graph using the same structural operators. New branches are initially tentative and solidify only after repeated successes. GPT-4o is used as the backbone with temperature set to 0 to ensure reproducibility.

## Key Experimental Results

### Main Results

| Method | ALFWorld SR (%) | WebShop Score | WebShop SR (%) | TextCraft SR (%) |
| :--- | :--- | :--- | :--- | :--- |
| ReAct | 85.8 | 44.0 | 20.0 | 62.0 |
| Reflexion | 84.3 | 40.8 | 23.0 | 59.0 |
| AWM | 91.3 | 49.2 | 30.0 | 92.5 |
| ASI | 70.6 | 7.7 | 7.5 | 77.8 |
| **NSI w/o online honing** | **93.5** | **58.8** | **30.5** | 78.5 |
| **NSI (Ours)** | **98.0** | **76.5** | **44.5** | **95.2** |

### Ablation Study

| Configuration | Observation | Interpretation |
| :--- | :--- | :--- |
| ASI (No logic branches) | WebShop Score only 7.7 | Linear scripts are completely unable to express conditional logic. |
| NSI offline only | Already exceeds all baselines | The logical representation induced offline is inherently powerful. |
| NSI full (inc. online honing) | SOTA across all three benchmarks | Reflective planning converts runtime failures into permanent capabilities. |
| Avg. atomic steps / skill | NSI $\approx 7.4$ vs lower for ASI | NSI compresses 7+ steps of logic into a single skill. |

### Key Findings
- ASI's formalization of experience into scripts actually performed worse than AWM's pure text workflows, suggesting that "expressively restricted programs" are worse than "non-executable text," thus validating the necessity of logical branching.
- "Long-horizon collapse" in ALFWorld: Baselines see success rates drop to 0 beyond 22 steps, whereas NSI maintains performance at 53+ steps because it compresses 7.4 atomic actions into a single skill, thereby "compressing" the planning horizon.
- The performance gain of Reflexion's textual memory over ReAct is nearly negligible, further indicating that the bottleneck for long-horizon tasks is "stable execution" rather than "memory retrieval."

## Highlights & Insights
- The concept of "trace-to-logic lifting" is highly generalizable—any LLM agent can use this method to upgrade demonstrations into verifiable programs, transferable to more complex scenarios such as SWE-bench or robotic manipulation.
- Reflective Planning converts failure signals into "local subgraph grafting," serving as a programmatic version of continual learning that avoids catastrophic forgetting (the skill graph grows monotonically).
- The combination of the MDL penalty and four structural operators provides the LLM with clear preferences (coverage vs. simplicity) when searching the program space, providing a reusable template for future "program synthesis + LLM" methodologies.

## Limitations & Future Work
- All experiments rely on the assumption that the "environment provides an enumerable predicate vocabulary" (both ALFWorld and WebShop provide structured feedback); for real-world open environments, predicate discovery itself remains a challenge.
- Using GPT-4o as a synthesizer is costly; the authors did not explore whether smaller models could drive this synthesis framework.
- Online honing depends on the LLM proposing its own corrective trajectory. If the LLM provides an incorrect recovery plan, grafting it into the graph might pollute the skill; the paper uses a "tentative → solidify" cycle to mitigate this but does not quantify failure rates.
- The improvement in TextCraft was relatively minimal (95.2 vs. AWM's 92.5), suggesting that for tasks where "recursive decomposition" is sufficient, the marginal value of logical branching is limited.

## Related Work & Insights
- **vs ASI**: ASI synthesizes skills into parameterized scripts without explicit control flows like CheckOp/LoopOp; NSI actively synthesizes branch discriminants through predicate invention, raising WebShop scores from 7.7 to 76.5.
- **vs AWM**: AWM's skills are textual templates and are non-executable; NSI's skills are symbolic graph programs that can be verified and precisely executed by an interpreter.
- **vs Agentic Workflow Generation (AFlow, GPTSwarm)**: These assemble predefined nodes (e.g., Debate, Voting); NSI's nodes are "invented" internal logic, providing finer granularity and stronger generalization.
- **vs Classic RL Options (Sutton 1999)**: Traditional options are black-box neural policies requiring extensive parameter optimization; NSI skills are readable Python-like code, naturally aligned with LLM generation capabilities.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Implements the neuro-symbolic idea of "lifting traces to logical programs" using LLMs within an agent framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three major agent benchmarks + comprehensive ablations + long-horizon analysis.
- Writing Quality: ⭐⭐⭐⭐ Algorithms and node definitions are clear, though some formalization is dense.
- Value: ⭐⭐⭐⭐⭐ Provides a new paradigm for increasing the expressivity of skill learning for LLM agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Skill-Pro: Learning Reusable Skills from Experience via Non-Parametric PPO for LLM Agents](skill-pro_learning_reusable_skills_from_experience_via_non-parametric_ppo_for_ll.md)
- [\[ACL 2026\] SOLAR-RL: Semi-Online Long-horizon Assignment Reinforcement Learning](../../ACL2026/llm_agent/solar-rl_semi-online_long-horizon_assignment_reinforcement_learning.md)
- [\[ICLR 2026\] Solving the Granularity Mismatch: Hierarchical Preference Learning for Long-Horizon LLM Agents](../../ICLR2026/llm_agent/solving_the_granularity_mismatch_hierarchical_preference_learning_for_long-horiz.md)
- [\[ICML 2026\] ACON: Optimizing Context Compression for Long-horizon LLM Agents](acon_optimizing_context_compression_for_long-horizon_llm_agents.md)
- [\[ACL 2026\] FregeLogic at SemEval 2026 Task 11: A Hybrid Neuro-Symbolic Architecture for Content-Robust Syllogistic Validity Prediction](../../ACL2026/llm_agent/fregelogic_at_semeval_2026_task_11_a_hybrid_neuro-symbolic_architecture_for_cont.md)

</div>

<!-- RELATED:END -->

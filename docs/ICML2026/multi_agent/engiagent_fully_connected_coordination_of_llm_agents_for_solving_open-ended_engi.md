---
title: >-
  [Paper Note] EngiAgent: Fully Connected Coordination of LLM Agents for Solving Open-ended Engineering Problems with Feasible Solutions
description: >-
  [ICML 2026][Multi-Agent][Fully Connected Coordinator] EngiAgent decomposes engineering problem solving into five expert agents: Analyzer, Modeler, Verifier, Solver…
tags:
  - "ICML 2026"
  - "Multi-Agent"
  - "Fully Connected Coordinator"
  - "Feasibility"
  - "Engineering Modeling"
  - "Feedback Routing"
date: 2026-05-08
content_hash: dd08675e2418bfce
---

# EngiAgent: Fully Connected Coordination of LLM Agents for Solving Open-ended Engineering Problems with Feasible Solutions

**Conference**: ICML 2026  
**arXiv**: [2605.02289](https://arxiv.org/abs/2605.02289)  
**Code**: https://github.com/AI4Engi/EngiAgent (Available)  
**Area**: LLM Agent / Engineering Problem Solving / Multi-Agent Systems  
**Keywords**: Fully Connected Coordinator, Feasibility, Engineering Modeling, Multi-Agent, Feedback Routing

## TL;DR
EngiAgent decomposes engineering problem solving into five expert agents: Analyzer, Modeler, Verifier, Solver, and Evaluator. By employing a **fully connected coordinator** to dynamically route feedback (rather than following a fixed pipeline), the feasible solution rate for GPT-4o on engineering tasks increases from 5.66% (zero-shot) / 7.55% (MM-Agent) to 64.15%, representing an approximately 7-fold improvement over previous SOTA.

## Background & Motivation

**Background**: LLMs have reached near-saturation in mathematical reasoning (e.g., GSM8K) and code generation (e.g., HumanEval). The community naturally seeks to extend these capabilities to engineering problem solving, such as traffic scheduling, power dispatch, and mechanical system coordination. Existing approaches roughly fall into three categories: (1) ResearchAgent-like agents for open exploration; (2) MM-Agent-like agents for mathematical modeling; (3) DS-Agent-like agents for data-driven code generation.

**Limitations of Prior Work**: The authors empirically present three striking figures: (a) Autonomous research directions produce < 10% feasible solutions because the goal is "novelty" rather than "executability"; (b) MM-Agent styles achieve 70% on math benchmarks but only about 13% provide numerical solutions in engineering problems; (c) DS-Agent styles achieve a 62.26% numerical output rate but only 5.66% are feasible under physical/safety constraints. Specifically, four types of errors are common in engineering problems: fancy but vague modeling, tampering with original data, violating physical laws, and over-constraining leading to insolvability.

**Key Challenge**: Existing agent systems generally adopt **fixed pipelines** (a linear DAG like Analyzer → Modeler → Solver). If an error occurs at a certain stage, feedback can only return to the previous step. It cannot perform cross-stage directional repairs (e.g., when a Verifier detects data inconsistency, it should jump back to the Analyzer rather than the Modeler). This rigid structure vs. the multi-source failure modes of engineering problems represents a structural mismatch.

**Goal**: (1) Elevate "feasibility" to a metric equivalent to correctness; (2) Design a coordination mechanism capable of arbitrary feedback routing, fault tolerance, and active switching among five roles; (3) Construct a feasibility benchmark covering various fields such as power, transportation, manufacturing, and structures.

**Key Insight**: Change the topology of multi-agent collaboration from a "chain" to a "fully connected graph + state-aware coordinator." This allows the coordinator (itself an LLM) to dynamically decide which agent takes control next based on error history in a shared Memory.

**Core Idea**: Use a hybrid strategy of "LLM autonomous decision-making + structured engineering protocols as context," paired with a mandatory agent switching mechanism (forcing a role change if the same agent fails repeatedly) to prevent infinite debugging loops.

## Method

### Overall Architecture
The system consists of 5 functional agents (Analyzer, Modeler, Verifier, Solver, Evaluator), 1 fully connected coordinator, and 1 shared Memory. The baseline pipeline (blue arrows) follows Analyzer → Modeler → Verifier → Solver → Evaluator. The coordinator layer (red arrows) allows arbitrary jumps—for instance, if the Verifier detects "data inconsistency," it can route directly back to the Analyzer for re-extraction, while if the Solver reports "no solution," it can route to the Modeler to relax over-constraints.

### Key Designs

1. **Fully Connected Coordinator**:
    - **Function**: Acts as the control hub, determining the next activated agent based on real-time feedback rather than a fixed sequence.
    - **Mechanism**: The coordinator receives engineering protocols (e.g., "data consistency priority > formal correctness," "relax over-constraints rather than deleting them") as context. Combined with error history in the shared Memory, this constrains LLM decision-making within reasonable engineering boundaries. Implementation includes: (a) State-aware Memory, recording the last $k$ outputs and failure causes for each agent; (b) **Mandatory Agent Switching Mechanism**—if an agent fails on the same error mode beyond a threshold, control is forcibly handed to another agent, preventing dead loops such as "the Modeler repeatedly trying to change formulas when the root cause is a data extraction error in the Analyzer."
    - **Design Motivation**: Fixed pipelines cannot precisely repair multi-source errors across stages; the fully connected structure allows feedback paths to match root causes one-on-one, while mandatory switching prevents local oscillations outside the root cause.

2. **Verifier: Hierarchical Asymmetric Check**:
    - **Function**: Strictly verifies if the "model remains faithful to the original problem" without bottlenecking the system.
    - **Mechanism**: A two-level determination—Level 1 involves mandatory, non-negotiable semantic checks (objective direction, core physical laws, data consistency); any violation leads to immediate rejection with diagnostic output and routing suggestions. Level 2 tolerates **functionally equivalent representational differences** (e.g., $a + b$ vs. $b + a$, or converting $\leq$ to $\geq$ by taking negatives), avoiding meaningless reruns due to "formatting differences." It also explicitly prohibits "deleting core hard constraints to enable a solution."
    - **Design Motivation**: Pure correctness checks cause the system to idle on formatting differences; pure tolerance allows hard constraints to be quietly removed while still passing. The hierarchical design suppresses failure modes at both ends.

3. **Role-based Agent Division + Shared Memory**:
    - **Function**: Explicitly maps the engineering expert workflow (Analysis → Modeling → Verification → Solving → Evaluation) to 5 LLM agents, each customized with prompts and domain knowledge bases.
    - **Mechanism**: The Analyzer transforms natural language problems into structured representations (decision variables, parameters, constraints, objectives) while identifying **implicit** engineering rules. The Modeler uses templated code generation to turn structured representations into executable models. The Verifier performs three types of checks (semantic consistency, constraint integrity, data consistency). The Solver selects backends, sets resource limits, and avoids deadlocks. The Evaluator performs a four-dimensional assessment of the complete solution (feasibility, model-problem alignment, engineering effectiveness, overall quality). Shared Memory allows all agents to see each other's outputs and history, serving as the basis for the coordinator's routing decisions.
    - **Design Motivation**: Bringing the clear division of labor from an engineering team into a multi-agent system ensures each agent has boundaries while gaining a global view via shared Memory, avoiding common failure modes where a "single agent takes all," leading to over-long prompts and role confusion.

### Loss & Training
This work is an inference-time agent framework and does not involve training. All effects derive from the design of the prompts, coordinator, and Memory. The benchmark compares three LLM backends (GPT-4o, Gemini-1.5 Flash, DeepSeek-V3-671B); the self-constructed dataset covers 53 high-quality problems across four engineering domains, focusing on feasibility rather than just numerical output rates.

## Key Experimental Results

### Main Results

| Backend / Method | Numerical Rate (Num.) ↑ | **Feasibility Rate (Feas.) ↑** | IE | DR | MO | UH | Avg. |
|------|------|------|------|------|------|------|------|
| GPT-4o Zero-shot | 22.64% | 5.66% | 5.66 | 5.42 | 4.47 | 3.33 | 4.72 |
| GPT-4o MM-Agent | 13.21% | 7.55% | 6.89 | 7.21 | 6.48 | 7.98 | 7.14 |
| GPT-4o EngiAgent (Fixed) | 47.17% | 47.17% | 8.30 | 7.22 | 6.67 | 7.14 | 7.33 |
| **GPT-4o EngiAgent (Coord.)** | **66.04%** | **64.15%** | **8.67** | **7.74** | **7.05** | **7.41** | **7.72** |
| Gemini-2.5 Flash EngiAgent (Coord.) | 52.83% | 50.94% | 8.30 | 6.89 | 6.30 | 6.06 | 6.89 |
| DeepSeek-V3-671B EngiAgent (Coord.) | — | 75.47% | — | — | — | — | — |

### Ablation Study

| Configuration | Feasibility Rate | Description |
|------|--------|------|
| Full EngiAgent (Coord.) | 64.15% (GPT-4o) | Complete system |
| EngiAgent (Fixed pipeline) | 47.17% | Coordinator removed, using fixed DAG, drop of ~17 pp |
| w/o Verifier | Significant drop | Lack of strict verification leads to physical/data violation solutions being misjudged as passing |
| w/o Mandatory Switching | Debugging cycles increase | Same agent oscillates repeatedly outside the root cause |

### Key Findings
- **The Coordinator is the key differentiator**: Across three backends, "Coord. vs. Fixed" improves the average feasibility rate by over 10 pp; on GPT-4o, this single component contributes +16.98 pp.
- **Numerical Output $\neq$ Feasibility**: DS-Agent on DeepSeek can produce numerical solutions for 77.36% of cases, but only 28.30% are feasible. EngiAgent's numerical output and feasibility rates almost overlap, proving that feasibility cannot be solved by "generating first and repairing later" but must be enforced within the generation path.
- **Backend Robustness**: The feasibility rates for all three LLM backends rank first in their respective groups, implying that the gains come primarily from the "collaboration structure" rather than a specific model—this is significant lateral evidence for agent frameworks.
- **High Quality of the Feasible Subset**: The average score within the feasible solutions is 8.12, indicating that EngiAgent does not achieve feasibility by "lowering the bar"; the feasible solutions themselves are of high quality.

## Highlights & Insights
- **"Topology Revolution" Narrative**: Attributes the failure of multi-agent systems to the "fixed pipeline" structure rather than LLM capabilities themselves, providing a minimum-cost alternative (fully connected + state-aware coordination). This insight is valuable for any multi-agent system design.
- **Mandatory Switching against Dead Loops**: Forcing a role change when a single agent fails repeatedly is a highly engineering-oriented design that solves the real problem of LLMs fixating on incorrect root causes.
- **Feasibility as a First-Class Citizen**: Explicitly separating feasibility rates from numerical output rates and categorizing failures into four types (vague modeling, altering data, physical violation, over-constraining) provides the community with a reusable failure mode taxonomy.

## Limitations & Future Work
- Evaluation is limited to 53 problems across 4 domains, which is relatively small; larger-scale, cross-domain engineering benchmarks are still needed.
- The coordinator itself is an LLM call; the upper bound of decision quality is limited by the backend model. Effectiveness on small models (< 7B) has not been verified.
- The five agent roles and engineering protocols are manually designed, lacking an automated generation or adaptive adjustment mechanism; adapting to new domains still requires significant prompt engineering.
- Inference costs are significantly higher than single-agent solutions (a task may involve dozens of LLM calls + multiple Solver calls). The paper does not report token or wall-clock costs, which is an unquantified variable for practical deployment.
- Some metrics (Avg. of IE/DR/MO/UH) are LLM-scored, carrying risks of self-evaluation bias; more human comparative verification is needed.

## Related Work & Insights
- **vs. ResearchAgent / AI Scientist**: These pursue open exploration and novelty, with feasibility < 10%. EngiAgent targets "feasible + practical," representing a different engineering-focused route.
- **vs. MM-Agent**: MM-Agent focuses on mathematical modeling quality. EngiAgent integrates "modeling + solving + feasibility checks," leading significantly in engineering settings.
- **vs. DS-Agent**: DS-Agent is a typical example of high numerical output but low feasibility—proving that "producing numbers" $\neq$ "usable," which indirectly argues for the necessity of Verifiers and Evaluators.
- **vs. HuggingGPT / AutoGen-like General Agent Frameworks**: These are more general, but their routing strategies are relatively fixed. EngiAgent takes routing flexibility further, with engineering protocols explicitly constraining the boundaries of that flexibility.

## Rating
- Novelty: ⭐⭐⭐⭐ While multi-agent frameworks are not new, the combination of "fully connected coordination + mandatory switching + engineering protocols" is pioneering in the engineering domain.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comparisons across three backends, four domains, and multiple baselines plus thorough ablation. Sample size of 53 is slightly small.
- Writing Quality: ⭐⭐⭐⭐ The illustration of the four failure modes clearly explains the need for this architecture; the narrative is clear.
- Value: ⭐⭐⭐⭐ Provides a high-quality reference implementation for those aiming to push LLMs into "real engineering closed loops."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] RoadMapper: A Multi-Agent System for Roadmap Generation of Solving Complex Research Problems](../../ACL2026/multi_agent/roadmapper_a_multi-agent_system_for_roadmap_generation_of_solving_complex_resear.md)
- [\[ACL 2026\] Diversity Collapse in Multi-Agent LLM Systems: Structural Coupling and Collective Failure in Open-Ended Idea Generation](../../ACL2026/multi_agent/diversity_collapse_in_multi-agent_llm_systems_structural_coupling_and_collective.md)
- [\[ACL 2026\] ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination](../../ACL2026/multi_agent/atlas_adaptive_trading_with_llm_agents_through_dynamic_prompt_optimization_and_m.md)
- [\[ICML 2026\] CoOT: Learning to Coordinate In-Context with Coordination Transformers](coot_learning_to_coordinate_in-context_with_coordination_transformers.md)
- [\[ICML 2026\] Sheaf-ADMM: Learning Multi-Agent Coordination via Sheaf-ADMM](learning_multi-agent_coordination_via_sheaf-admm.md)

</div>

<!-- RELATED:END -->

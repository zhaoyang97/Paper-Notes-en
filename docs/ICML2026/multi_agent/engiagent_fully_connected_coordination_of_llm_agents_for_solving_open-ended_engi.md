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
content_hash: 06f646e37394f3ca
---

# EngiAgent: Fully Connected Coordination of LLM Agents for Solving Open-ended Engineering Problems with Feasible Solutions

**Conference**: ICML 2026  
**arXiv**: [2605.02289](https://arxiv.org/abs/2605.02289)  
**Code**: https://github.com/AI4Engi/EngiAgent (available)  
**Area**: LLM Agent / Engineering Problem Solving / Multi-Agent Systems  
**Keywords**: Fully Connected Coordinator, Feasibility, Engineering Modeling, Multi-Agent, Feedback Routing

## TL;DR
EngiAgent decomposes engineering problem solving into five expert agents: Analyzer, Modeler, Verifier, Solver, and Evaluator. A **fully connected coordinator** dynamically routes feedback (instead of following a fixed pipeline), boosting the feasible solution rate on engineering tasks with GPT-4o from 5.66% (zero-shot) / 7.55% (MM-Agent) to 64.15%—an average improvement of about 7x over previous SOTA.

## Background & Motivation

**Background**: LLMs have nearly saturated benchmarks in mathematical reasoning (e.g., GSM8K) and code generation (e.g., HumanEval), prompting the community to extend them to engineering problem solving—such as traffic scheduling, power dispatch, and mechanical system coordination. Existing approaches fall into three categories: (1) ResearchAgent for open-ended exploration; (2) MM-Agent for mathematical modeling; (3) DS-Agent for data-driven code generation.

**Limitations of Prior Work**: The authors empirically highlight three striking statistics: (a) Autonomous research approaches yield <10% feasible solutions, as they prioritize "novelty" over "executability"; (b) MM-Agent achieves 70% on math benchmarks but only about 13% numerical solutions for engineering problems; (c) DS-Agent produces 62.26% numerical outputs but only 5.66% are feasible under physical/safety constraints. Common engineering errors include: vague but flashy modeling, altering raw data, violating physical laws, and over-constraining to the point of infeasibility.

**Key Challenge**: Most agent systems use a **fixed pipeline** (e.g., Analyzer → Modeler → Solver in a linear DAG). If an error occurs mid-process, feedback can only return to the previous step, making it impossible to directly fix cross-stage errors (e.g., when the Verifier detects data inconsistency, it should jump back to the Analyzer, not the Modeler). This rigid structure is structurally mismatched with the multi-source failure modes of engineering problems.

**Goal**: (1) Elevate "feasibility" to a metric on par with correctness; (2) Design a coordination mechanism that can route feedback, tolerate errors, and actively switch among five roles; (3) Build a feasibility benchmark covering power, transportation, manufacturing, and structural domains.

**Key Insight**: Transform the multi-agent collaboration topology from a "chain" to a "fully connected graph + state-aware coordinator," allowing the coordinator (itself an LLM) to dynamically assign control based on error history in shared memory.

**Core Idea**: Employ a hybrid strategy of "LLM autonomous decision-making + structured engineering protocols as context," with a forced agent-switching mechanism (if an agent repeatedly fails, force a switch) to prevent infinite debugging loops.

## Method

### Overall Architecture
The system consists of five functional agents (Analyzer, Modeler, Verifier, Solver, Evaluator), one fully connected coordinator, and a shared memory. The baseline pipeline (blue arrows) follows Analyzer → Modeler → Verifier → Solver → Evaluator; the coordinator layer (red arrows) enables arbitrary cross-stage routing—for example, if the Verifier detects "data inconsistency," it can route directly to the Analyzer for re-extraction, while if the Solver reports "no solution," it can route to the Modeler to relax over-constraints.

### Key Designs

1. **Fully Connected Coordinator**:

    - **Function**: Acts as the control center, deciding which agent to activate next based on real-time feedback, rather than following a fixed order.
    - **Mechanism**: The coordinator receives engineering protocols (e.g., "data consistency > formal correctness," "over-constraints should be relaxed, not deleted") as context, and, together with error history in shared memory, constrains LLM decision-making within engineering-reasonable boundaries. Implementation includes: (a) State-aware memory, recording each agent's recent $k$ outputs and failure reasons; (b) **Forced agent-switching mechanism**—if the same agent fails repeatedly on the same error pattern beyond a threshold, control is forcibly handed to another agent, preventing deadlocks such as "Modeler repeatedly tweaking formulas when the root cause is in Analyzer's data extraction."
    - **Design Motivation**: Fixed pipelines cannot precisely repair multi-source errors across stages; the fully connected structure allows feedback paths to match error root causes one-to-one, while forced switching prevents LLMs from oscillating locally outside the true root cause.

2. **Verifier: Hierarchical Asymmetric Checking**:

    - **Function**: Strictly checks whether the model remains faithful to the original problem without blocking the system.
    - **Mechanism**: Two-level judgment—first, enforce non-negotiable semantic checks (goal direction, core physical laws, data consistency); any violation triggers immediate rejection with error diagnosis and routing suggestions. Second, tolerate **functionally equivalent expression differences** (e.g., $a + b$ vs $b + a$, $\leq$ rewritten as $\geq$ with negation), avoiding meaningless reruns due to format differences. Explicitly prohibits "removing core hard constraints for the sake of solvability."
    - **Design Motivation**: Pure correctness checks cause the system to stall on format differences; pure tolerance allows hard constraints to be quietly deleted and still pass—hierarchical design suppresses both failure modes.

3. **Role-based Agent Division + Shared Memory**:

    - **Function**: Explicitly maps the engineering expert workflow (analysis → modeling → verification → solving → evaluation) to five LLM agents, each customized with prompts and domain knowledge bases.
    - **Mechanism**: Analyzer converts natural language problems into structured representations ("decision variables + parameters + constraints + objectives"), also identifying **implicit** engineering rules; Modeler uses templated code generation to turn structured representations into executable models; Verifier performs three checks (semantic consistency, constraint completeness, data consistency); Solver selects backend, sets resource limits, and avoids deadlocks; Evaluator assesses the complete solution on four dimensions (feasibility, alignment with problem, engineering validity, overall quality). Memory allows all agents to access each other's outputs and history, serving as the basis for coordinator routing decisions.
    - **Design Motivation**: Translates the clear division of labor in engineering teams into a multi-agent system, where each agent has boundaries but can access a global view via shared memory, avoiding the common failure mode of "single agent does everything, leading to overly long prompts and role confusion."

### Loss & Training
This is an inference-time agent framework with no training involved; all performance derives from prompt, coordinator, and memory design. Benchmarks use three LLM backends (GPT-4o, Gemini-2.5 Flash, DeepSeek-V3-671B) for comparison; the custom dataset covers 53 high-quality problems across four engineering domains, focusing on feasibility rather than just numerical output rate.

## Key Experimental Results

### Main Results

| Backend / Method | Numerical Output Rate (Num.) ↑ | **Feasibility Rate (Feas.) ↑** | IE | DR | MO | UH | Avg. |
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
| EngiAgent (Fixed pipeline) | 47.17% | Coordinator removed, fixed DAG, ~17 pp drop |
| w/o Verifier | Significant drop (see §6.5) | Lack of strict checking leads to solutions violating physical/data constraints being misjudged as feasible |
| w/o Forced Switching | Debugging loop failure rate increases | Same agent oscillates outside root cause |

### Key Findings
- **Coordinator is the key differentiator**: Across three backends, "Coord. vs Fixed" yields an average feasibility rate improvement of over 10 pp; on GPT-4o, this alone contributes +16.98 pp.
- **Numerical output ≠ Feasibility**: DS-Agent on DeepSeek produces 77.36% numerical solutions but only 28.30% are feasible; EngiAgent's numerical and feasible rates nearly coincide, proving feasibility cannot be fixed post hoc but must be enforced during generation.
- **Backend robustness**: All three LLM backends achieve the highest feasibility rates in their respective groups, indicating that gains stem mainly from the collaborative structure, not any particular model—strong horizontal evidence for the agent framework.
- **High quality of feasible subset**: Average score within feasible solutions is 8.12, indicating EngiAgent's high feasibility rate is not due to "relaxed feasibility criteria," but that feasible solutions are genuinely high quality.

## Highlights & Insights
- **"Topological revolution" narrative**: Attributes multi-agent system failures to "fixed pipeline" structure rather than LLM capability, and proposes a minimally invasive alternative (fully connected + state-aware coordinator), offering design inspiration for any multi-agent system.
- **Forced switching prevents deadlocks**: Forcing a switch when an agent repeatedly fails is a highly practical design that directly addresses the typical issue of "LLMs obsessing outside the root cause."
- **Feasibility as a first-class citizen**: Clearly separates feasibility rate from numerical output rate, and provides a reusable taxonomy of four error types (vague modeling, altering data, physical violation, over-constraining) for the community.

## Limitations & Future Work
- Evaluation is limited to 53 problems across four domains; scale is small. Larger, cross-domain engineering benchmarks are needed (as acknowledged in the paper).
- The coordinator itself is an LLM call, so decision quality is bounded by the backend model; effectiveness on small models (<7B) remains untested.
- The five agent roles and engineering protocols are manually designed, lacking automated generation or adaptive adjustment; onboarding new domains still requires substantial prompt engineering.
- Inference cost is significantly higher than single-agent solutions (each task may involve dozens of LLM calls + multiple Solver invocations); the paper does not report token/wall-clock costs, leaving deployment feasibility unquantified.
- Some evaluation metrics (IE/DR/MO/UH "Avg.") are LLM-scored, posing a risk of self-assessment bias; more human comparative validation is needed.

## Related Work & Insights
- **vs ResearchAgent / AI Scientist**: These pursue open-ended exploration and novelty, with feasibility rates <10%; EngiAgent directly targets "feasible + practical" solutions, representing a distinct engineering-oriented path.
- **vs MM-Agent**: MM-Agent focuses on mathematical modeling quality; EngiAgent integrates "modeling + solving + feasibility checking," thus achieving significant gains in engineering settings.
- **vs DS-Agent**: DS-Agent typifies high numerical output but low feasibility—demonstrating that "producing numbers" ≠ "usable solutions," indirectly justifying the necessity of Verifier + Evaluator.
- **vs HuggingGPT / AutoGen general agent frameworks**: These are more general but have relatively fixed routing strategies; EngiAgent advances routing flexibility, with engineering protocols explicitly constraining the boundaries of flexibility.

## Rating
- Novelty: ⭐⭐⭐⭐ The multi-agent framework itself is not new, but the combination of "fully connected coordination + forced switching + engineering protocols" is a first in the engineering domain.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three backends × four domains × multiple baselines + thorough ablation; sample size of 53 is somewhat small.
- Writing Quality: ⭐⭐⭐⭐ The illustration of four failure modes clearly motivates the architecture; narrative is clear.
- Value: ⭐⭐⭐⭐ Provides a high-quality reference implementation for anyone seeking to bring LLMs into "real engineering closed loops."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Diversity Collapse in Multi-Agent LLM Systems: Structural Coupling and Collective Failure in Open-Ended Idea Generation](../../ACL2026/multi_agent/diversity_collapse_in_multi-agent_llm_systems_structural_coupling_and_collective.md)
- [\[ACL 2026\] SILO-BENCH: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems](../../ACL2026/multi_agent/silo-bench_a_scalable_environment_for_evaluating_distributed_coordination_in_mul.md)
- [\[ICML 2026\] MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems](maspo_joint_prompt_optimization_for_llm-based_multi-agent_systems.md)
- [\[ICML 2026\] OMAC: A Holistic Optimization Framework for LLM-Based Multi-Agent Collaboration](omac_a_holistic_optimization_framework_for_llm-based_multi-agent_collaboration.md)
- [\[ICML 2026\] E-mem: Multi-Agent Based Episodic Context Reconstruction for LLM Agent Memory](e-mem_multi-agent_based_episodic_context_reconstruction_for_llm_agent_memory.md)

</div>

<!-- RELATED:END -->

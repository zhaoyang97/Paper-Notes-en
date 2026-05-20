---
title: >-
  [Paper Note] Towards Reliable Code-as-Policies: A Neuro-Symbolic Framework for Embodied Task Planning
description: >-
  [NeurIPS 2025 (Spotlight)][Robotics][Code-as-Policies] This paper proposes a neuro-symbolic framework for embodied task planning that augments LLM-based code generation with explicit symbolic verification (checking wheth…
tags:
  - "NeurIPS 2025 (Spotlight)"
  - "Robotics"
  - "Code-as-Policies"
  - "neuro-symbolic framework"
  - "task planning"
  - "LLM code generation"
  - "symbolic verification"
date: 2026-05-08
content_hash: 66464a73e6d603cc
---

# Towards Reliable Code-as-Policies: A Neuro-Symbolic Framework for Embodied Task Planning

**Conference**: NeurIPS 2025 (Spotlight)
**arXiv**: [2510.21302](https://arxiv.org/abs/2510.21302)  
**Code**: None  
**Area**: Robotics / Embodied Intelligence
**Keywords**: Code-as-Policies, neuro-symbolic framework, task planning, LLM code generation, symbolic verification

## TL;DR

This paper proposes a neuro-symbolic framework for embodied task planning that augments LLM-based code generation with explicit symbolic verification (checking whether preconditions are satisfied) and interactive verification (active exploration to acquire missing information), enabling more reliable code execution in dynamic and partially observable environments. On RLBench, task success rate improves from a baseline of 38.5% to 84.7%, with executability reaching 86.8%.

## Background & Motivation

**Background**: Automatic executable code generation via LLMs (Code-as-Policies, CaP) has become a dominant paradigm for robot task planning. Given natural language task descriptions and environment APIs, LLMs directly generate Python code to control robots, leveraging their strong reasoning and code generation capabilities without requiring manually crafted policies for each task.

**Limitations of Prior Work**: CaP methods suffer from severe **environment grounding failures**, particularly in dynamic or partially observable settings. Three failure modes are observed: (1) LLMs generate code based on incomplete environmental observations, potentially referencing non-existent objects or incorrect locations; (2) generated code may contain logical errors—e.g., placing a lid before pouring liquid; (3) action sequences lack awareness of environmental state changes, as objects may be moved or occluded during execution.

**Key Challenge**: LLM "world knowledge" is general-purpose, whereas robot execution demands precise environment-specific information. Standard CaP methods generate all code in a single pass, with no mechanism to verify whether the generated plan is consistent with the current environment, nor any capability to actively acquire missing observations when information is insufficient. More critically, unconstrained exploratory actions may destroy existing task progress (e.g., knocking over already-placed objects).

**Goal**: How can the flexibility of LLM code generation be preserved while ensuring environmental consistency and execution reliability? This decomposes into: (1) How to detect which assumptions in the code are unsatisfied? (2) How to safely acquire missing information? (3) How to explore the environment without disrupting task state?

**Key Insight**: Symbolic reasoning is introduced as a "guardrail" for LLM code generation. Symbolic methods formally check preconditions and postconditions before code execution; when conditions are not met, interactive exploration is triggered rather than direct execution. The intervention occurs at the generation stage rather than the execution stage, following a "verify before act" principle.

**Core Idea**: A symbolic "look before you act" mechanism is incorporated into LLM code generation—formal precondition checking exposes information gaps, and exploratory code subject to state-preservation constraints safely fills those gaps.

## Method

### Overall Architecture

The framework transforms the LLM code generation pipeline from "one-shot generation → execution" into an iterative loop of "generation → symbolic verification → exploration → correction → execution." The input consists of a natural language task description and environment APIs. The LLM first generates an initial code plan, which then enters a verification loop: a symbolic verifier checks whether preconditions for each action are satisfied; if not, interactive verification is triggered—exploratory code is generated to interact with the environment and acquire missing observations while preserving task-relevant state. After new information is obtained, the LLM revises the code and re-verifies, repeating until all preconditions are satisfied.

### Key Designs

1. **Symbolic Precondition Verifier**:

    - **Function**: Formally checks the preconditions of each action in the LLM-generated code.
    - **Mechanism**: Preconditions for robot actions (e.g., pick, place, pour) are represented as logical predicates—for example, $\text{pick}(obj)$ requires $\text{is\_reachable}(obj) \wedge \text{is\_graspable}(obj) \wedge \neg\text{is\_held}(\text{any})$. The verifier formalizes current environmental observations as a logical state and checks whether each action's preconditions are satisfied. Unsatisfied preconditions are flagged as information gaps to be resolved through exploration.
    - **Design Motivation**: LLMs excel at high-level reasoning but are insensitive to low-level physical constraints. Symbolic verification provides a deterministic "safety net"—even when the LLM errs (e.g., ignoring occlusion), the symbolic checker catches the error before execution. This is more reliable than relying on LLM self-reflection (e.g., Reflexion).

2. **Interactive Exploratory Code Generation**:

    - **Function**: Generates safe exploratory code to acquire missing information when preconditions are not satisfied.
    - **Mechanism**: Unsatisfied preconditions are fed back to the LLM as "information requirements," prompting the generation of exploratory code—e.g., "move to the right side of the table to observe occluded objects" or "open the drawer to inspect its contents." A critical constraint is that exploratory code must pass state-preservation verification before execution: exploration must not alter the state of task-relevant objects (e.g., must not knock over an already-placed cup).
    - **Design Motivation**: Unlike simple retries or reflection, interactive exploration actively engages with the environment to acquire information, thereby addressing partial observability. This introduces the concept of active perception into the CaP framework.

3. **State Preservation Constraint**:

    - **Function**: Ensures that exploratory actions do not destroy existing task progress.
    - **Mechanism**: A set of invariants is defined to describe intermediate task states already achieved—e.g., "the cup is on the table and upright." Symbolic reasoning verifies whether all invariants are maintained after the exploratory code executes. Exploratory actions that violate invariants are rejected, and the LLM is required to generate a safer alternative.
    - **Design Motivation**: Unconstrained exploration can be catastrophic—a robot may inadvertently knock over other objects while trying to observe a target. In real-robot settings, such failures are irreversible. The state preservation constraint is the core safeguard for safe exploration.

### Loss & Training

The framework is entirely based on prompt engineering and requires no additional training. The core pipeline is realized through carefully designed prompt templates: (1) task description + environment API → LLM outputs initial code; (2) symbolic verification feedback → LLM outputs corrected or exploratory code; (3) exploration results → LLM outputs final code. The entire process is transparent to the LLM backend and is compatible with different models such as GPT-4 and GPT-3.5.

## Key Experimental Results

### Main Results (RLBench Simulation Environment)

Evaluation is conducted on RLBench tasks involving dynamic and partially observable scenarios.

| Method | Task Success Rate (%) ↑ | Executability (%) ↑ |
|--------|------------------------|---------------------|
| Code-as-Policies | 38.5 | 52.3 |
| SayCan | 42.1 | 61.8 |
| ProgPrompt | 45.3 | 63.5 |
| Inner Monologue | 51.2 | 68.4 |
| VoxPoser | 48.7 | 65.2 |
| **Ours** | **84.7** | **86.8** |

Compared to the CaP baseline, the proposed framework achieves an absolute improvement of 46.2% in success rate and 34.5% in executability. Against the strongest baseline, Inner Monologue, it achieves an absolute gain of 33.5%.

### Ablation Study

| Component | Success Rate (%) | ΔSR | Note |
|-----------|-----------------|------|------|
| Full framework | 84.7 | — | All components enabled |
| w/o symbolic verification | 62.3 | −22.4 | Cannot detect precondition violations |
| w/o interactive verification | 68.1 | −16.6 | Cannot acquire missing observations |
| w/o state preservation constraint | 73.5 | −11.2 | Exploration may disrupt task state |
| Single-pass generation (no iteration) | 45.8 | −38.9 | Degenerates to vanilla CaP |
| GPT-4 → GPT-3.5 | 71.2 | −13.5 | LLM capability affects performance ceiling |

### Key Findings

- **All three components are indispensable**: Symbolic verification (−22.4%) and interactive verification (−16.6%) contribute most individually. Removing either causes the framework to degrade toward a standard CaP variant. The −11.2% from removing state preservation constraints confirms that safe exploration is not optional.
- **Most significant gains in partially observable scenarios**: In dynamic settings, success rate improves from approximately 25% (baseline) to 78%, demonstrating the decisive value of active exploration in information-incomplete environments.
- **LLM backend matters but is not the determining factor**: GPT-3.5 lags GPT-4 by 13.5%, yet still substantially outperforms all baselines (71.2% vs. 51.2% for the strongest baseline), indicating that the structural design of the framework contributes more than raw LLM capability.
- **Real-robot validation**: The framework is validated on a real robot platform, confirming that the gains are not artifacts of the simulation environment.

## Highlights & Insights

- **A paradigmatic neuro-symbolic integration**: LLMs provide flexible reasoning and code generation, while symbolic verification provides deterministic correctness guarantees. The combination avoids the respective weaknesses of "unreliable pure neural methods" and "inflexible pure symbolic methods." The general design pattern of "LLM generation + symbolic verification" is transferable to domains such as code security auditing and automated testing.
- **Formalization of "look before you act"**: While "observe before acting" is intuitively natural, this paper formalizes it into a complete pipeline—precondition checking → information gap identification → safe exploration → code revision—with state preservation constraints ensuring safety. This formalization transforms intuition into an engineerable methodology.
- **Training-free, plug-and-play framework**: Built entirely on prompt engineering, the framework requires no additional training data or fine-tuning, making it highly amenable to rapid deployment in new scenarios. It also benefits automatically from improvements in LLM capabilities.
- **Importance of the executability metric**: The paper introduces executability as an evaluation dimension complementary to success rate—even when a task is not fully completed, generated code should at least be executable. An executability of 86.8% means the vast majority of generated code does not crash at runtime, which is critical for real-world deployment.

## Limitations & Future Work

- **Exploration introduces latency**: Interactive exploration requires additional observation and code generation rounds, increasing task completion time. This may be unsuitable for scenarios with strict real-time requirements (e.g., rapid grasping, emergency obstacle avoidance). Selective exploration—triggering exploration only for high-uncertainty actions—is a promising direction.
- **Dependence on predefined environment APIs**: The symbolic verifier requires manually defined preconditions and postconditions for each action, as well as a description of the environment's state space. Adapting to new environments requires human effort to specify these definitions. Automatically inferring preconditions from API documentation using LLMs is a worthwhile direction.
- **Completeness of invariant definitions**: State preservation constraints rely on manually defined invariants, which may omit critical constraints in complex scenarios—for instance, non-intuitive physical interactions (e.g., bumping a table corner causing a distant object to slide). Automatically learning invariants from physical simulation is a valuable research direction.
- **Strong dependence on LLM code quality**: While symbolic verification can detect precondition errors, it cannot identify logical errors within the code (e.g., incorrect loop control, erroneous arithmetic). Combining program analysis techniques (e.g., abstract interpretation) with symbolic verification could provide more comprehensive coverage.
- **Preconditions only**: The current framework verifies only pre-execution conditions and does not check postconditions. Incorporating postcondition verification would enable runtime monitoring and failure recovery.

## Related Work & Insights

- **vs. Code-as-Policies (Liang et al., 2023)**: The direct baseline for this work. CaP generates and executes code in a single pass with no verification mechanism. The proposed framework inserts a verify-explore-correct loop between code generation and execution; the key distinction is shifting from "blindly trusting the LLM" to "trust after verification."
- **vs. Inner Monologue (Huang et al., 2022)**: Inner Monologue iteratively revises generation using environmental feedback, but the feedback is derived from execution outcomes (post-hoc correction). Symbolic verification in this work constitutes pre-execution checking—problems are identified before execution, avoiding irreversible failures.
- **vs. SayCan (Ahn et al., 2022)**: SayCan uses affordance scoring to constrain LLM outputs—essentially a learned "soft" constraint. This work employs symbolic verification as a "hard" constraint—actions whose preconditions are unsatisfied are not executed. This yields higher reliability at the cost of some flexibility.
- **vs. classical STRIPS planning**: Traditional task planning relies entirely on symbolic methods with limited flexibility. The proposed framework preserves the flexible code generation capability of LLMs while introducing symbolic verification only at critical checkpoints, embodying a philosophy of "minimally necessary symbolization."

## Rating

- Novelty: ⭐⭐⭐⭐ — The application of neuro-symbolic integration to CaP is novel, particularly the safe exploration mechanism under state preservation constraints.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — RLBench + real-robot evaluation, comprehensive ablations, and multi-LLM-backend comparisons provide thorough coverage.
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear and the methodological narrative is logically coherent, consistent with Spotlight-level quality.
- Value: ⭐⭐⭐⭐⭐ — Directly advances the reliability of LLM-driven robotic systems; the "verify before act" design pattern has broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] UniDomain: Pretraining a Unified PDDL Domain from Real-World Demonstrations for Generalizable Task Planning](pretraining_a_unified_pddl_domain_from_real-world_demonstrations_for_generalizab.md)
- [\[ICLR 2026\] REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning?](../../ICLR2026/robotics/rei-bench_can_embodied_agents_understand_vague_human_instructions_in_task_planni.md)
- [\[NeurIPS 2025\] LatentGuard: Controllable Latent Steering for Robust Refusal of Attacks and Reliable Response Generation](latentguard_controllable_latent_steering_for_robust_refusal_of_attacks_and_relia.md)
- [\[CVPR 2026\] Cross-Domain Demo-to-Code via Neurosymbolic Counterfactual Reasoning](../../CVPR2026/robotics/cross-domain_demo-to-code_via_neurosymbolic_counterfactual_reasoning.md)
- [\[NeurIPS 2025\] NeSyPr: Neurosymbolic Proceduralization For Efficient Embodied Reasoning](nesypr_neurosymbolic_proceduralization_for_efficient_embodied_reasoning.md)

</div>

<!-- RELATED:END -->

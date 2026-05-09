---
title: >-
  [Paper Note] One Demo Is All It Takes: Planning Domain Derivation with LLMs from A Single Demonstration
description: >-
  [ICLR 2026][Robotics][PDDL] This paper proposes the PDDLLM framework, which derives a complete PDDL planning domain (predicates + actions) automatically from **a single demonstration trajectory**. It generates interpretable symbolic representations through cross-validation between LLM reasoning and physical simulation, and employs a Logical Constraint Adapter (LoCA) to automatically interface with motion planners. The method achieves at least 20% higher success rates than 6 LLM baselines across 1,200+ tasks in 9 environments, and is successfully deployed on 3 physical robot platforms.
tags:
  - ICLR 2026
  - Robotics
  - PDDL
  - Task and Motion Planning
  - LLM Reasoning
  - Physical Simulation
  - Predicate Generation
  - Motion Planning Interface
date: 2026-05-08
content_hash: 38ec15d39a3193e7
---

# One Demo Is All It Takes: Planning Domain Derivation with LLMs from A Single Demonstration

**Conference**: ICLR 2026
**arXiv**: [2505.18382](https://arxiv.org/abs/2505.18382)
**Code**: None
**Area**: Robot Planning / TAMP / LLM
**Keywords**: PDDL, Task and Motion Planning, LLM Reasoning, Physical Simulation, Predicate Generation, Motion Planning Interface

## TL;DR

This paper proposes the PDDLLM framework, which derives a complete PDDL planning domain (predicates + actions) automatically from **a single demonstration trajectory**. It generates interpretable symbolic representations through cross-validation between LLM reasoning and physical simulation, and employs a Logical Constraint Adapter (LoCA) to automatically interface with motion planners. The method achieves at least 20% higher success rates than 6 LLM baselines across 1,200+ tasks in 9 environments, and is successfully deployed on 3 physical robot platforms.

## Background & Motivation

**Background**: Task and Motion Planning (TAMP) combines high-level symbolic reasoning with low-level motion planning and represents the dominant paradigm for long-horizon robot manipulation tasks. It fundamentally relies on planning domains defined in PDDL, $\mathcal{D} = (\mathcal{P}, \mathcal{A})$, comprising a predicate set $\mathcal{P}$ and an action set $\mathcal{A}$.

**Limitations of Prior Work**: Constructing PDDL planning domains is highly labor-intensive: defining predicates (e.g., `(on ?o1 ?o2)`), action preconditions $\mathcal{P}_{pre}$, and effects $\mathcal{P}_{eff}$ all require careful design by domain experts, incurring substantial engineering effort and poor transferability to new environments.

**Limitations of LLMs**: Although LLMs exhibit strong generalization in task planning, they remain unreliable for long-horizon reasoning—prone to errors when temporal dependencies become complex (Huang et al., 2022a).

**Limitations of Prior Work on Domain Generation**: (1) Methods requiring predefined predicates or actions as prior knowledge (Silver et al., 2023; Kumar et al., 2023) still demand significant human involvement; (2) LLM-based approaches require detailed natural-language domain descriptions and careful prompt engineering (Guan et al., 2023); (3) many methods assume that symbolic actions already have corresponding motion skills, leaving the action-to-motion-planner interface to be completed manually.

**Key Insight**: The paper combines LLMs' semantic understanding with the verification capability of physical simulation—simulation provides physical feasibility checks (which LLMs alone cannot guarantee), while LLMs contribute semantic abstraction and pattern recognition. Only a single demonstration is required to fully automate planning domain construction.

**Core Contributions**: (1) The first fully automated pipeline from a one-shot demonstration to a complete PDDL domain; (2) LoCA, which automatically interfaces symbolic actions with motion planners; (3) validation of the approach across 1,200+ tasks with successful deployment on real robot platforms.

## Method

### Overall Architecture

PDDLLM takes as input a demonstration trajectory $\tau_{demo}$ (a sequence of continuous environment states) and its task description $T_{demo}$ (a brief natural-language description), and outputs an executable PDDL planning domain. For new tasks, a symbolic solver generates a task plan, which LoCA then automatically interfaces with a motion planner to produce executable trajectories:

$$\tilde{a}^{(0)}, \tilde{a}^{(1)}, \ldots = \text{MotionPlanner}(\text{PDDLLM}(\mathcal{S}_{new}^{(init)}, \mathcal{S}_{new}^{(goal)}, T_{demo}, \tau_{demo}))$$

### Key Design 1: Predicate Imagination

- **Function**: Automatically generates semantically meaningful logical predicates from physical simulation roll-outs, proceeding in two stages: first-order predicates and higher-order predicates.
- **Mechanism**:
    - **Stage 1 — First-Order Predicates**: Object states are uniformly sampled in the feature space (position $(x,y,z)$, color $(r,g,b)$, etc.) → the physical simulator verifies feasibility (filtering out penetrations, floating states, etc.) → the continuous feature space is discretized into sub-intervals → an LLM summarizes semantically meaningful predicates (e.g., `(is_on ?o1 ?o2)`, `(smaller ?o1 ?o2)`) from the feasible subspaces and annotates their relevance to the current task. Each predicate is associated with the physical constraints of its corresponding subspace boundaries, which are used to evaluate predicate truth values during motion planning.
    - **Stage 2 — Higher-Order Predicates**: More complex relational expressions are systematically derived by composing first-order predicates using logical operators (negation $\neg$) and quantifiers (universal $\forall$, existential $\exists$). For example, `(is_on ?o1 ?o2)` → `(not_is_on ?o1 ?o2)` → `(∀_o1_not_is_on ?o1 ?o2)` (o2 is on top of everything).
- **Design Motivation**: Predicates imagined by LLMs alone may be physically implausible (e.g., a floating "is_on" relation). Simulation-based verification ensures every predicate has a physically feasible grounding. Discretization enables the LLM to extract stable relational patterns from the scattered continuous space and provides computable constraint boundaries for motion planning.

### Key Design 2: Action Invention

- **Function**: Automatically extracts PDDL actions (with preconditions and effects) from demonstration trajectories.
- **Mechanism**: The continuous states in $\tau_{demo}$ are projected into logical space $\tau_{demo}^{logic}$ using the predicate library, compressing trajectories exceeding 1,000 steps into only a few logical state transitions. Pre- and post-state pairs of each logical transition are extracted and provided as prompts to the LLM, which summarizes PDDL action definitions (preconditions $\mathcal{P}_{pre}$ and effects $\mathcal{P}_{eff}$).
- **Design Motivation**: Recognizing patterns in logical space is far simpler than in continuous space—long trajectories are dramatically compressed and the LLM only needs to process key state changes. This design transforms continuous manipulation into a discrete pattern recognition problem, leveraging LLMs' summarization strengths.

### Key Design 3: Logical Constraint Adapter (LoCA)

- **Function**: Fully automatically interfaces the generated PDDL planning domain with the underlying motion planner.
- **Mechanism**: For each logical action in the task plan, LoCA automatically extracts the physical constraints (expressed as mathematical inequalities) of all first-order predicates in the effect set $\mathcal{P}_{eff}$, applies these constraints sequentially to the motion planner, and converts each logical action into a standard **constrained motion planning problem**. The resulting motion trajectories are guaranteed to satisfy the semantic meaning of each action.
- **Design Motivation**: In traditional TAMP, manually encoding mathematical constraints to bridge symbolic actions and motion planning (Toussaint, 2015) is one of the most cumbersome engineering steps. LoCA exploits physical constraint information already generated during predicate imagination, requiring no additional manual mapping. The framework is also compatible with VLA models—logical actions can serve directly as VLA prompt conditions.

### Key Design 4: Parallel Prompting with Feedback

- **Function**: Improves the robustness of PDDL domain generation.
- **Mechanism**: Multiple candidate PDDL domains are generated in parallel from the same demonstration → runtime validation (syntax, completeness, reachability checks) → failed candidates are discarded → the best candidate is selected by majority vote. Experiments show that correctness stabilizes when the number of parallel prompts exceeds 5.
- **Design Motivation**: LLM outputs are stochastic and single-pass generation may fail. Parallel generation combined with feedback substantially improves reliability at low additional cost.

## Key Experimental Results

### Table 1: Planning Success Rate (%) Across 9 Tasks (Time Limit = 50s)

| Method | Stack | Unstack | Color Class. | Alignment | Parts Assem. | Rearrange | Burger Cook | Bridge Build | Tower Hanoi | **Overall** |
|--------|-------|---------|-------------|-----------|-------------|-----------|-------------|-------------|-------------|------------|
| Expert | 98.5 | 100 | 100 | 100 | 98.9 | 73.3 | 100 | 100 | 100 | 95.7 |
| LLMTAMP | 41.7 | 89.4 | 18.1 | 31.1 | 33.3 | 5.6 | 27.8 | 43.3 | 14.3 | 35.7 |
| LLMTAMP-FF | 70.8 | 94.6 | 36.4 | 52.0 | 53.9 | 17.4 | 50.0 | 53.3 | 14.3 | 52.5 |
| LLMTAMP-FR | 64.2 | 92.1 | 49.0 | 40.0 | 41.3 | 11.8 | 48.6 | 51.7 | 14.3 | 48.6 |
| RuleAsMem | 85.5 | 88.4 | 88.7 | 96.0 | 95.0 | 1.1 | 27.8 | 20.0 | 14.3 | 69.9 |
| **PDDLLM** | **97.5** | **97.7** | **100** | **100** | **100** | **64.3** | **91.7** | **87.2** | **100** | **93.3** |

### Table 2: PDDLLM vs. Reasoning LLMs — Success Rate and Token Cost (Time Limit = 500s)

| Task | PDDLLM | LLMTAMP | o1-TAMP | R1-TAMP | PDDLLM(k) | LLMTAMP(k) | o1-TAMP(k) | R1-TAMP(k) |
|------|--------|---------|---------|---------|-----------|------------|------------|------------|
| Rearrangement | **73.8** | 5.6 | 70.8 | 40.0 | 334 | 212 | 1200 | 1460 |
| Tower of Hanoi | **100** | 14.3 | 33.3 | 14.3 | 535 | 36 | 529 | 353 |
| Bridge Building | **87.2** | 44.3 | 51.7 | 40.0 | 375 | 50 | 270 | 363 |
| **Overall** | **80.5** | 13.9 | 61.5 | 35.9 | 415 | 99 | 666 | 725 |

### Table 3: Domain Quality — Missing/Redundant Predicate Rates

| Task | Missing Predicates | Redundant Predicates |
|------|-------------------|---------------------|
| Stack | 4.2% | 8.3% |
| Burger Cooking | 22.2% | 3.7% |
| Bridge Building | 22.2% | 3.7% |
| Tower of Hanoi | 0.0% | 14.3% |

## Key Findings

1. **Large Margin over LLM Baselines**: PDDLLM achieves an overall success rate of 93.3%, surpassing the best LLM baseline LLMTAMP-FF (52.5%) by more than 40 percentage points. Improvements are especially pronounced on complex tasks (Tower of Hanoi, Color Classification)—from 14.3%/36.4% to 100%.
2. **Outperforms Reasoning LLMs**: Even under the extended 500s time limit, PDDLLM (80.5% overall) outperforms o1-TAMP (61.5%) with lower token cost (415k vs. 666k).
3. **Near Expert-Level Performance**: PDDLLM's 93.3% overall success rate is only 2.4 percentage points below the expert-designed domain (95.7%), and fully matches the expert's 100% success rate on four tasks: Color Classification, Alignment, Parts Assembly, and Tower of Hanoi.
4. **Cross-Task Knowledge Transfer**: PDDLLM can modularly combine actions learned from different demonstrations—solving rearrangement using demonstrations of stacking and unstacking, and solving bridge building using demonstrations of stacking and alignment.
5. **Indispensability of Simulation Verification**: Ablation studies confirm that predicate quality degrades significantly when simulation verification is removed (more physically implausible predicates are generated). Discretizing the continuous feature space is also critical—it enables the LLM to extract stable patterns from scattered data.
6. **Token Efficiency**: PDDLLM's token cost is concentrated in the one-time domain derivation phase; subsequent planning is handled entirely by the symbolic solver without additional token consumption, making it particularly suitable for long-term deployment involving repeated similar tasks.

## Highlights & Insights

- **Complementary Simulation + LLM Architecture**: Simulation provides physical feasibility verification (which LLMs cannot guarantee alone), while LLMs provide semantic abstraction and pattern recognition—neither is sufficient independently. This represents an excellent paradigm for applying LLMs to physical reasoning.
- **One-Shot Demo → Generalizable Domain**: The extremely low data requirement (a single demonstration) yields a planning domain that generalizes to new tasks, significantly lowering the barrier to entry for TAMP.
- **LoCA Eliminates the Largest Manual Bottleneck**: The hand-coded interface between symbolic actions and motion planning is traditionally the most cumbersome engineering step in TAMP. LoCA automates this entirely using physical constraint information already produced during predicate imagination.
- **Modular Action Composition**: The modular nature of PDDL action syntax enables natural cross-task knowledge transfer—actions learned from different demonstrations can be directly composed into new domains.
- **Symbolic Solver vs. LLM Planner**: Comparison with RuleAsMem reveals an important insight: LLMs have limited capacity to interpret complex planning domains, making it more reliable to delegate domain rules to a symbolic solver than to an LLM planner.

## Limitations & Future Work

1. **Missing Complex Predicates**: In complex tasks (e.g., Bridge Building, Burger Cooking), higher-order predicates (e.g., `(all_base_finished)`) are occasionally omitted, requiring additional feasibility checks and backtracking from the planner, which reduces success rates under fixed time budgets.
2. **Perceptual Limitations**: The framework relies on simple perception schemes such as ArUco markers and cannot directly derive planning domains from raw visual input. Complex dynamics and geometry (deformable objects, fluids) also remain challenging.
3. **Simulation Dependency**: The predicate imagination stage requires a physical simulator to perform extensive parallel roll-outs, making it difficult to apply in settings where accurate simulation models are unavailable.
4. **Sensitivity to Initial Discretization**: Although the LLM can subsequently refine the discretization granularity, the initial choice of the hyperparameter $u_f$ still affects predicate generation quality.

## Related Work & Insights

### vs. Direct LLM Planning (LLMTAMP, SayCan-style)

| Dimension | PDDLLM | Direct LLM Planning |
|-----------|--------|---------------------|
| Planning Approach | LLM derives domain → symbolic solver plans | LLM directly outputs action sequences |
| Long-Horizon Reasoning | Strong (guaranteed by symbolic solver) | Weak (LLMs prone to errors on temporal dependencies) |
| Token Cost | One-time domain derivation; no subsequent cost | Token consumption per planning query |
| Verifiability | PDDL domain is human-readable and correctable | LLM output is unverifiable |
| Data Requirement | 1 demonstration | Requires task descriptions / historical context |

### vs. Classical Domain Learning (Predicate Invention, VisualPredicator)

| Dimension | PDDLLM | Classical Domain Learning |
|-----------|--------|--------------------------|
| Prior Knowledge | No predefined predicates or actions required | Requires partially predefined symbols |
| Training Data | Single demonstration | Typically requires large numbers of demonstrations |
| Motion Interface | Automatic via LoCA | Usually assumes predefined motion skills |
| Interpretability | Generates human-readable PDDL domains | Some methods produce uninterpretable representations |

### vs. Hand-Crafted TAMP Domains (Expert Design)

| Dimension | PDDLLM | Expert Design |
|-----------|--------|--------------|
| Construction Cost | Fully automated; no domain expertise needed | Requires expert manual authoring and debugging |
| Planning Performance | 93.3% overall success rate | 95.7% overall success rate (only 2.4% higher) |
| Transferability | New environment requires only a new demonstration | New environment requires full domain redesign |
| Domain Completeness | Occasional missing predicates | Complete but engineering-intensive |

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First fully automated pipeline from a one-shot demonstration to a complete PDDL domain, requiring no predefined predicates or actions
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 9 environments / 1,200+ tasks / 6 baselines / 3 physical platforms / ablation studies / domain quality evaluation / token analysis
- **Writing Quality**: ⭐⭐⭐⭐ — Framework diagrams are clear and intuitive; motivation and design rationale for each component are well articulated, with occasional inconsistencies in notation
- **Value**: ⭐⭐⭐⭐⭐ — Substantially lowers the barrier to long-horizon robot task planning; LoCA addresses the largest engineering bottleneck in TAMP

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] VLBiMan: Vision-Language Anchored One-Shot Demonstration Enables Generalizable Bimanual Robotic Manipulation](vlbiman_vision-language_anchored_one-shot_demonstration_enables_generalizable_bi.md)
- [\[CVPR 2026\] Cross-Domain Demo-to-Code via Neurosymbolic Counterfactual Reasoning](../../CVPR2026/robotics/cross-domain_demo-to-code_via_neurosymbolic_counterfactual_reasoning.md)
- [\[ICLR 2026\] What's the Plan? Metrics for Implicit Planning in LLMs and Their Application to Rhyme Generation and Question Answering](whats_the_plan_metrics_for_implicit_planning_in_llms_and_their_application_to_rh.md)
- [\[ICLR 2026\] String Seed of Thought: Prompting LLMs for Distribution-Faithful and Diverse Generation](string_seed_of_thought_prompting_llms_for_distribution-faithful_and_diverse_gene.md)
- [\[ICLR 2026\] Tracing and Reversing Edits in LLMs](tracing_and_reversing_edits_in_llms.md)

<!-- RELATED:END -->

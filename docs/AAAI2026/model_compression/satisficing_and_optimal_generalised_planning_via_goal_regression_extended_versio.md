---
title: >-
  [Paper Note] Satisficing and Optimal Generalised Planning via Goal Regression (Extended Version)
description: >-
  [AAAI 2026][Model Compression][generalised planning] This paper presents the Moose planner, which synthesises generalised planning programs from training problems via goal regression. It decomposes multi-goal problems in…
tags:
  - "AAAI 2026"
  - "Model Compression"
  - "generalised planning"
  - "goal regression"
  - "policy synthesis"
  - "search pruning"
  - "planning domains"
date: 2026-05-08
content_hash: 88d8f7c512ef0823
---

# Satisficing and Optimal Generalised Planning via Goal Regression (Extended Version)

**Conference**: AAAI 2026
**arXiv**: [2511.11095](https://arxiv.org/abs/2511.11095)
**Code**: [https://github.com/dillonzchen/moose](https://github.com/dillonzchen/moose)
**Area**: Model Compression
**Keywords**: generalised planning, goal regression, policy synthesis, search pruning, planning domains

## TL;DR

This paper presents the Moose planner, which synthesises generalised planning programs from training problems via goal regression. It decomposes multi-goal problems into single-goal subproblems, solves each optimally, and applies regression followed by lifting to produce a set of first-order condition-action rules. These rules support either satisficing planning (direct rule execution) or optimal planning (encoded as axioms to prune the search space).

## Background & Motivation

Generalised Planning (GP) aims to synthesise programs that solve a family of related planning problems. The core objective is to **amortise synthesis cost**: solving a family of problems faster than invoking a general-purpose planner on each instance individually.

Many real-world domains are computationally easy to solve yet difficult to scale with existing general-purpose planners. For instance, UPS delivers over 20 million parcels daily across 200+ countries, yet state-of-the-art general planners struggle with simplified delivery problems involving as few as 100 parcels.

Existing GP methods (e.g., sketch learner, PG3, Loom) each have notable limitations:

- High synthesis cost, with some domains failing to complete within 12 hours
- Excessive memory consumption
- Lack of theoretical guarantees for optimal planning

Moose is motivated by leveraging **goal regression**—a technique with a long history in the knowledge representation community—combined with **problem relaxation** (decomposing multi-goal problems into single-goal subproblems), enabling efficient generalised plan synthesis and instantiation in a three-step pipeline.

## Method

### Overall Architecture

Moose operates in two phases:

**Synthesis Phase** (Algorithm 1):
1. For each training problem, enumerate goal-atom orderings and decompose each into single-goal subproblems.
2. Solve each subgoal optimally while advancing the current state along the way.
3. Apply **goal regression** to each sub-plan (backpropagating from the goal state to the required preconditions).
4. **Lift** the resulting state–goal–action sequences by replacing concrete objects with free variables, yielding first-order rules.

**Instantiation Phase**:
- **Satisficing planning** (Algorithm 3): On a new problem, attempt to ground rules in priority order and execute matching macro-action sequences until the goal is reached.
- **Optimal planning** (Section 3.3): Encode learned rules as PDDL axioms to constrain the set of applicable actions at each search state, thereby pruning the search space.

### Key Designs

1. **Moose Rule**:

   - *Function*: Compactly represents a generalised policy as first-order condition–action pairs.
   - *Mechanism*: Each rule contains a set of free variables, state conditions (atoms the current state must satisfy), goal conditions (unsatisfied goal atoms), and an action sequence (including macro-actions).
   - *Design Motivation*: Rule preconditions compactly capture families of states; lifted rules generalise across varying numbers of objects.
   - Each rule carries a priority (cost-to-go) that determines execution order.

2. **Goal Regression + Lifting**:

   - *Function*: Extracts generalised condition–action rules from concrete plans.
   - *Mechanism*: Regression computes the preimage of an action—the minimal set of atoms that must hold before execution—after which concrete objects are replaced by free variables.
   - *Key Property*: Regression naturally eliminates atoms irrelevant to the goal (e.g., the location of a dog is ignored in a plan for transporting a cake), endowing rules with strong generalisation.

3. **Goal Independence Classification (TGI / SGI / OGI)**:

   - *Function*: Formally classifies the degree to which goals in a planning domain can be solved independently.
   - TGI (True Goal Independence): A greedy algorithm is effective under any goal ordering.
   - SGI (Serialisable Goal Independence): Some goal ordering exists under which the greedy algorithm is effective.
   - OGI (Optimal Goal Independence): Some ordering exists under which the greedy algorithm yields an optimal solution.
   - *Computational Complexity*: pTGI is in P; pSGI and pOGI are NP-complete.

### Loss & Training

This work does not involve neural network training. Rule synthesis is based on:
- Solving each subproblem optimally using A* with the LM-cut heuristic.
- A default of $n_p = 3$ goal orderings per training problem.
- Rule grounding implemented via SQLite database queries.
- An optional extension: regressing over goal subsets (rather than single goals) to generate a richer set of high-quality rules.

## Key Experimental Results

### TGI Effectiveness on IPC Benchmarks

| Statistic | Value |
|-----------|-------|
| Domains with all outputs valid (potential TGI) | 13/38 (34.2%) |
| Domains with at least half of outputs valid | 19/38 (50.0%) |
| Domains with at least one valid output | 27/38 (71.1%) |
| Valid outputs across all problems | 590/1184 (49.8%) |

### Main Results: Satisficing Planning Coverage

| Domain | SLearn-0 | SLearn-1 | SLearn-2 | Lama | Moose |
|--------|---------|---------|---------|------|-------|
| Barman | 0.0 | 0.0 | 0.0 | 49 | **90.0** |
| Ferry | 15.0 | 67.0 | 60.0 | 69 | **90.0** |
| Gripper | 59.6 | 50.8 | 33.0 | 65 | **90.0** |
| Logistics | 0.0 | 0.0 | 0.0 | 77 | **89.6** |
| Miconic | 68.8 | 72.6 | 67.8 | 77 | **90.0** |
| Rovers | 0.0 | 0.0 | 0.0 | 66 | **90.0** |
| Satellite | 0.0 | 29.2 | 34.6 | 89 | **90.0** |
| Transport | 0.0 | 63.0 | 46.8 | 66 | **90.0** |
| **Total (720)** | 143.4 | 282.6 | 242.2 | 558 | **719.6** |

### Optimal Planning Coverage

| Domain | Blind | LMcut | Scorpion | SymK | Moose |
|--------|-------|-------|----------|------|-------|
| Barman | 0 | 0 | 0 | 12 | **24.6** |
| Ferry | 10 | 18 | 17 | 18 | **30.0** |
| Gripper | 9 | 8 | 7 | **30** | 27.0 |
| Miconic | 30 | 30 | 30 | 30 | **30.0** |
| **Total (240)** | 93 | 119 | 140 | 154 | **183.0** |

### Ablation Study: Effect of Joint Goal Regression

| Configuration | Description |
|---------------|-------------|
| $n_r = 1$ (single-goal regression) | Base configuration; fast synthesis. |
| $n_r = 2$ (2-subset goal regression) | Improved solution quality in 7/8 domains (1.1%–25.0%), at higher synthesis and instantiation cost. |
| $n_r = 3$ (3-subset goal regression) | Marginal improvement over $n_r = 2$. |

### Key Findings

- **Moose solves nearly all classical and numeric planning problems** (719.6/720), substantially outperforming Lama (558) and the best SLearn configuration (282.6).
- **Synthesis efficiency**: Moose is faster than the best SLearn configuration in 5/8 domains and consistently uses less memory (<1 GB vs. up to 7.6 GB).
- **Empirical optimality**: Although no theoretical guarantee of optimality is given, Moose empirically produces optimal plans in all observed cases.
- **Encoding rules as axioms generally outperforms vanilla SymK**: the benefit of search space reduction exceeds the overhead of axiom evaluation.
- **Solution quality**: Moose outperforms Lama and ENHSP in 5/8 classical domains and 3/4 numeric domains.

## Highlights & Insights

- **Elegant yet effective pipeline**: Decompose goals → solve optimally → regress → lift. This four-step process synthesises generalised policies with conceptual clarity.
- **Theoretical guarantees**: Completeness for satisficing planning is proved under the TGIC condition; completeness for optimal planning is proved under TGIC + OGI.
- **Planning states as databases, rules as queries**: SQLite is leveraged for efficient rule grounding, elegantly reusing mature database optimisation infrastructure.
- **Identification of ESHO domains (Easy-to-Solve, Hard-to-Optimise)**: Domains that are P-time solvable but NP-hard to optimise are precisely where Moose excels.
- **Generalisation to numeric planning**: The same framework extends naturally to numeric planning domains (NFerry, NMiconic, NMinecraft, NTransport).

## Limitations & Future Work

- The TGI assumption does not hold in many planning domains (e.g., the Sussman Anomaly in Blocksworld), limiting the method's applicability.
- The training set size required for theoretical guarantees is exponential in domain size in the worst case.
- The greedy decomposition strategy may fail when goals exhibit complex interdependencies (e.g., mutually exclusive goals).
- The number of rules grows rapidly with $n_r$, causing instantiation cost to increase by an order of magnitude.
- Domains requiring transitive closure computation are not handled.
- Synthesis relies on an optimal planner (A* + LM-cut), limiting efficiency when subproblems are themselves hard to solve optimally.

## Related Work & Insights

- **Loom** (Illanes & McIlraith, 2019): Generates quantified problems via equivalent-object bagging and synthesises policies with a FOND planner.
- **PG3** (Yang et al., 2022): Performs heuristic search over the space of generalised policies, also employing goal regression to handle "missing" states.
- **Sketch Learner** (Drexler et al., 2022): Learns sketches (abstract descriptions) at higher synthesis cost.
- **Long history of goal regression**: From STRIPS (Fikes et al., 1972) to Waldinger (1977) to Reiter (2001), Moose revitalises this classical technique in a bottom-up manner.
- *Insight*: Combining classical formal methods from planning with modern data-driven policy learning can yield significant advances on practical benchmarks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Combining goal regression with goal decomposition for generalised plan synthesis is novel; the theoretical analysis (complexity classification of TGI/SGI/OGI) is rigorous.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers 8 classical and 4 numeric domains, evaluates both satisficing and optimal planning, and assesses synthesis cost, instantiation cost, and solution quality.
- **Writing Quality**: ⭐⭐⭐⭐ — The theoretical sections are rigorous and the algorithmic pseudocode is clear, though the paper is lengthy and requires careful reading.
- **Value**: ⭐⭐⭐⭐ — Substantially advances the state of the art on ESHO domains and addresses a practical scalability challenge for general-purpose planners.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Lightweight Optimal-Transport Harmonization on Edge Devices](lightweight_optimal-transport_harmonization_on_edge_devices.md)
- [\[ICLR 2026\] ACPBench Hard: Unrestrained Reasoning about Action, Change, and Planning](../../ICLR2026/model_compression/acpbench_hard_unrestrained_reasoning_about_action_change_and_planning.md)
- [\[CVPR 2026\] Planning in 8 Tokens: A Compact Discrete Tokenizer for Latent World Model](../../CVPR2026/model_compression/planning_in_8_tokens_a_compact_discrete_tokenizer_for_latent_world_model.md)
- [\[ICLR 2026\] Compute-Optimal Quantization-Aware Training](../../ICLR2026/model_compression/compute-optimal_quantization-aware_training.md)
- [\[ICLR 2026\] Dataset Distillation as Pushforward Optimal Quantization](../../ICLR2026/model_compression/dataset_distillation_as_pushforward_optimal_quantization.md)

</div>

<!-- RELATED:END -->

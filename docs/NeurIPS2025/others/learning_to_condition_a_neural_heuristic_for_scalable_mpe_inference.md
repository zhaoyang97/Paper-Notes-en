---
title: >-
  [Paper Note] Learning to Condition: A Neural Heuristic for Scalable MPE Inference
description: >-
  [NeurIPS 2025][MPE inference] This paper proposes Learning to Condition (L2C), which trains an attention network to learn dual scores — *optimality* and *simplification* — for variable-value pairs from solver search trajectories, guiding conditioning decisions in MPE inference over probabilistic graphical models (PGMs). L2C substantially reduces the search space on high-treewidth models while maintaining or improving solution quality.
tags:
  - NeurIPS 2025
  - MPE inference
  - probabilistic graphical models
  - neural heuristic
  - conditioning
  - branch-and-bound
date: 2026-05-08
content_hash: 9b61c4638a403cc4
---

# Learning to Condition: A Neural Heuristic for Scalable MPE Inference

**Conference**: NeurIPS 2025
**arXiv**: [2509.25217](https://arxiv.org/abs/2509.25217)
**Code**: None
**Area**: Other
**Keywords**: MPE inference, probabilistic graphical models, neural heuristic, conditioning, branch-and-bound

## TL;DR

This paper proposes Learning to Condition (L2C), which trains an attention network to learn dual scores — *optimality* and *simplification* — for variable-value pairs from solver search trajectories, guiding conditioning decisions in MPE inference over probabilistic graphical models (PGMs). L2C substantially reduces the search space on high-treewidth models while maintaining or improving solution quality.

## Background & Motivation

**State of the Field**: Most Probable Explanation (MPE) inference is a core task in PGMs, aiming to find the most probable assignment of unobserved variables given evidence. Classical exact methods such as AND/OR search and integer linear programming (ILP) guarantee optimality but are computationally prohibitive for high-treewidth models. Approximate methods sacrifice solution quality.

**Limitations of Prior Work**: Conditioning — fixing the values of a subset of variables to reduce inference complexity — is a classical acceleration strategy (e.g., cutset conditioning, recursive conditioning), but its effectiveness critically depends on variable selection and assignment ordering. Existing approaches either rely on hand-crafted heuristics (e.g., maximum degree) or computationally expensive full strong branching for step-by-step lookahead.

**Root Cause**: Which variables should be fixed, to what values, and in what order? Fixing the wrong variable can irreversibly exclude the optimal solution, while not fixing it forfeits computational speedup. An ideal strategy must simultaneously satisfy *safety* (preserving the optimal solution) and *efficiency* (substantially reducing solving cost), yet these two objectives are often in tension.

**Paper Goals**: (1) How to automatically learn which variable-value pairs jointly satisfy optimality preservation and inference simplification? (2) How to integrate such a learned strategy into existing exact solvers?

**Starting Point**: L2C observes that *solution ambiguity* modulates the risk of conditioning — if a variable takes the same value across all optimal solutions, fixing it is safe but also most dangerous (a wrong estimate is fatal); if the variable's value distribution is more uniform, the conditioning risk is lower. The model adaptively learns this trade-off through extensive training.

**Core Idea**: Train a neural network to learn dual scores (optimality + simplification) for variable-value pairs from solver search trajectories, replacing hand-crafted heuristics for conditioning decisions.

## Method

### Overall Architecture

L2C operates in two phases: (1) *Offline data generation and training* — an oracle solver is invoked to obtain optimal solutions and solving statistics, constructing a supervised dataset to train an attention network; (2) *Online inference* — the trained network scores all variable-value pairs, selects high-scoring assignments via greedy search or beam search for conditioning, and passes the simplified problem to an exact solver.

### Key Designs

1. **Scalable Data Generation Pipeline**:

    - Function: Extract training signals from solver search trajectories.
    - Mechanism: For each PGM instance, random full assignments are sampled and partitioned into query and evidence sets. An oracle solver is called to obtain the MPE optimal solution. For a randomly selected subset of $c_{max}$ variables from the query set, each variable is fixed one at a time and the problem is re-solved, recording statistics such as runtime and search node count. The optimal solution provides *optimality* labels (whether a variable-value pair appears in the solution), and solving statistics, normalized via softmax, provide *simplification* ranking labels.
    - Design Motivation: Enumerating all MPE solutions is infeasible; sampling $c_{max}$ variables controls computational overhead while collecting sufficient supervision signal.

2. **Attention Dual-Head Architecture**:

    - Function: Output optimality and simplification scores for each variable-value pair.
    - Mechanism: Each variable-value pair is represented via an embedding table. Embeddings of unobserved variables serve as queries and interact with evidence variable embeddings through multi-head attention, capturing inter-variable dependencies. Contextualized embeddings pass through a shared encoder and are then fed into two separate MLP heads — an optimality head (sigmoid) estimating the probability that an assignment appears in the optimal solution, and a simplification head (softmax) estimating the relative utility of an assignment for reducing inference cost.
    - Design Motivation: The dual-head design decouples the objectives of *preserving the optimal solution* and *reducing solving overhead*; attention achieves permutation invariance and generalizes to arbitrary evidence-query partitions.

3. **Multi-Task Loss and Inference Strategies**:

    - Function: Jointly optimize both objectives and provide flexible inference-time integration options.
    - Mechanism: Total loss is $\mathcal{L} = \lambda_{opt} \cdot \mathcal{L}_{opt} + \lambda_{rank} \cdot \mathcal{L}_{rank}$, where optimality uses binary cross-entropy and simplification uses list-ranking cross-entropy. Three inference strategies are provided: greedy conditioning (iteratively selecting the top-scored assignment), beam search (maintaining $W$ candidate sequences), and NN-guided B&B (directly used as branching and node selection heuristics).
    - Design Motivation: Ranking loss is more robust than absolute value prediction; multiple strategies accommodate varying latency/quality requirements.

## Key Experimental Results

### Main Results

Evaluated on 14 high-treewidth binary PGMs (90–1444 variables) across 12 configurations (4 conditioning depths × 3 time budgets).

| Method | Configs Outperforming Unconditioned Oracle | Notes |
|--------|------------------------------------------|-------|
| L2C-Rank | Nearly all 12 configs | Most consistent |
| L2C-Opt | Majority of configs | Optimality head only |
| Full Strong Branching | Few configs | Occasional improvement |
| Graph heuristic | Very few configs | Weakest |

### Ablation Study (AOBB oracle: node reduction vs. solution quality)

| Conditioning Depth | L2C-Rank LL Gap | L2C-Rank Node Reduction | Baseline LL Gap |
|-------------------|----------------|------------------------|----------------|
| 5% | ≈0 | 40–60% | Noticeable deviation |
| 15% | ≈0 | 60–80% | Larger deviation |
| 25% | ≈0 | 70–90% | Significant degradation |

### Key Findings

- L2C-Rank helps SCIP find better solutions in almost all configurations, with advantages growing as conditioning depth increases.
- When combined with AOBB, L2C substantially reduces search nodes while maintaining near-optimal solution quality, whereas baseline methods degrade rapidly.
- As a B&B branching/node selection heuristic, L2C also significantly outperforms SCIP's default strategy (heatmap predominantly green).

## Highlights & Insights

- **The dual-score design is elegant**: Decomposing conditioning decisions into orthogonal *optimality preservation* and *inference simplification* dimensions enables a more fine-grained safety-efficiency trade-off than any single heuristic. This idea is transferable to any search problem requiring a quality-efficiency trade-off.
- **The learn-from-solver-traces paradigm is reusable**: The strategy of using a solver as an oracle to generate training data is generalizable to combinatorial optimization problems such as SAT and scheduling.
- **Plug-and-play integration**: L2C can serve both as a preprocessing step to simplify the problem and as an internal heuristic within B&B to guide search, offering strong flexibility.

## Limitations & Future Work

- The data generation phase requires multiple oracle solver calls, which may be infeasible for models with millions of variables.
- Only binary-variable PGMs are evaluated; generalization to multi-valued variables remains untested.
- Only runtime and node count are used as solver signals; richer signals (e.g., LP bound changes, branch-and-cut decisions) may further improve performance.
- A separate model is trained per PGM; transfer learning across PGM families has not been explored.

## Related Work & Insights

- **vs. Full Strong Branching**: Strong branching evaluates pruning effectiveness via one-step lookahead, which is computationally expensive and does not generalize; L2C internalizes this knowledge through offline training at negligible inference cost.
- **vs. Neural Branching (Gasse et al.)**: Prior work uses GNNs to imitate strong branching decisions; L2C additionally introduces simplification scoring and is specifically tailored to MPE inference in PGMs.
- **vs. Relaxation-based methods (CPN, VMP-NN)**: Optimizing relaxed likelihood objectives offers no optimality guarantees; L2C maintains compatibility with exact solvers.

## Rating

- Novelty: ⭐⭐⭐⭐ The dual-head scoring and learn-from-solver-traces paradigm are original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematic evaluation across 14 PGMs, two oracles, and multiple integration strategies.
- Writing Quality: ⭐⭐⭐⭐ Technical descriptions are clear and algorithmic pseudocode is complete.
- Value: ⭐⭐⭐⭐ Practically valuable for high-treewidth PGM inference; the learn-from-solver paradigm is broadly applicable.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Scalable Inference of Functional Neural Connectivity at Submillisecond Timescales](scalable_inference_of_functional_neural_connectivity_at_submillisecond_timescale.md)
- [\[NeurIPS 2025\] Statistical Inference Under Performativity](statistical_inference_under_performativity.md)
- [\[NeurIPS 2025\] Statistical Inference for Gradient Boosting Regression](statistical_inference_for_gradient_boosting_regression.md)
- [\[NeurIPS 2025\] Robust Sampling for Active Statistical Inference](robust_sampling_for_active_statistical_inference.md)
- [\[NeurIPS 2025\] FlowMoE: A Scalable Pipeline Scheduling Framework for Distributed MoE Training](flowmoe_a_scalable_pipeline_scheduling_framework_for_distributed_mixture-of-expe.md)

<!-- RELATED:END -->

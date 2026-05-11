---
title: >-
  [Paper Note] Boolean Satisfiability via Imitation Learning
description: >-
  [ICLR 2026][Reinforcement Learning][SAT] This paper proposes ImitSAT, the first imitation learning-based branching heuristic for CDCL solvers. By compressing solver runs into conflict-free KeyTrace expert sequences and f…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "SAT"
  - "imitation learning"
  - "CDCL"
  - "branching heuristic"
  - "autoregressive"
  - "Transformer"
date: 2026-05-08
content_hash: 1790be0b4d89f764
---

# Boolean Satisfiability via Imitation Learning

**Conference**: ICLR 2026
**arXiv**: [2509.25411](https://arxiv.org/abs/2509.25411)
**Code**: [https://github.com/zewei-Zhang/ImitSAT](https://github.com/zewei-Zhang/ImitSAT)
**Area**: Reinforcement Learning
**Keywords**: SAT, imitation learning, CDCL, branching heuristic, autoregressive, Transformer

## TL;DR
This paper proposes ImitSAT, the first imitation learning-based branching heuristic for CDCL solvers. By compressing solver runs into conflict-free KeyTrace expert sequences and framing branching decisions as an autoregressive prediction task conditioned on the decision prefix, ImitSAT significantly reduces propagation counts and solving time under a small query budget, and demonstrates strong generalization to structured SAT benchmarks.

## Background & Motivation

**Background**: Boolean Satisfiability (SAT) is a cornerstone of theoretical CS and AI. Modern solvers are dominated by the CDCL framework, in which branching decisions determine the search trajectory, while unit propagation accounts for 80%–90% of runtime.

**Limitations of Prior Work**:
- Classical branching heuristics (e.g., VSIDS) are manually designed and offer limited adaptability.
- SATformer adjusts variable activity only at initialization and cannot influence branching once search begins.
- Graph-Q-SAT employs an online RL agent, but RL requires extensive exploration, suffers from sparse and unstable rewards, and relies solely on the current state snapshot without exploiting the full decision history.

**Key Challenge**: Propagation accounts for approximately 88.9% of CDCL runtime (measured on MiniSAT); reducing propagation count is therefore the primary path to accelerating solving, and high-quality branching decisions directly reduce propagations.

**Key Insight**: Replace reinforcement learning with imitation learning—learn directly from expert trajectories to obtain dense, decision-level supervision signals while avoiding the cost of exploration.

## Core Problem
How can high-quality expert signals be extracted from CDCL solver execution traces, and how can a plug-and-play branching policy be trained to reduce propagation counts?

## Method

### Overall Architecture
ImitSAT consists of two core components: (1) **KeyTrace construction**—compressing the full solver run into a compact sequence retaining only surviving decisions; and (2) **an autoregressive imitation learner**—predicting the next branching decision conditioned on the decision prefix.

### Key Designs

1. **KeyTrace Expert Trajectory Extraction**:

    - **Function**: Extract a compact sequence $\mathcal{K}_t$ from the full CDCL execution trace $\mathcal{T}_t$, retaining only surviving decisions and propagations.
    - **Mechanism**: Scan the trace left to right while maintaining a working sequence $\mathcal{K}$. Decision and propagation events are appended directly; upon a backtrack event, $\text{trim}_{\leq h}$ removes all suffix events above the backtrack level before appending the new decision. Restarts are treated as trimming to level 0.
    - **Key Effect**: Replaying the KeyTrace produces virtually no conflicts—requiring only 0.2% of the conflicts, 19.6% of the decisions, and 4.3% of the propagations of the original MiniSAT run.

2. **Serialization and Autoregressive Learner**:

    - **Function**: Serialize the CNF formula and KeyTrace prefix into a unified input and train an autoregressive model to predict the next signed variable.
    - **Input Format**: `[CNF] || F_DIMACS || [SEP] || enc(K_t) || [D]`, where `[D]` is a decision probe token.
    - **Training Objective**: Behavioral cloning, minimizing the negative log-likelihood (cross-entropy) of expert decisions.
    - **Architecture**: Perceiver AR (Hawthorne et al., 2022) with a latent array of length 1, achieving $O(N)$ per-query complexity rather than $O(N^2)$; 16 attention heads and 12 Transformer blocks.

3. **Online Integration into CDCL**:

    - **Function**: Query the learner at each decision point within a small budget, falling back to VSIDS under uncertainty.
    - **Core Strategy**: Front-loaded query scheduling—early decisions have the greatest impact on search, so the query budget is concentrated at the beginning of solving.
    - **Completeness Guarantee**: All other CDCL components (propagation, conflict analysis, clause learning) remain unchanged.

### Training Techniques
- **Variable Permutation Augmentation**: Random permutation of variable IDs to construct training samples, effectively mitigating overfitting (without augmentation, validation loss first decreases then increases).
- **Staged Curriculum Learning**: Gradually scaling from small to large instances to accelerate convergence on easier cases.

## Key Experimental Results

### Main Results: Random 3-SAT Test Set (MRPP $\tilde{r}$↓, lower is better)

| Method | 5–15 | 16–30 | 31–60 | 61–100 | 50 | 100 |
|--------|------|-------|-------|--------|----|-----|
| Graph-Q-SAT (3 calls) | 1.00 | 0.94 | 0.89 | 1.15 | 0.71 | 0.85 |
| SATformer | 1.00 | 0.89 | 0.84 | 0.78 | 0.88 | 0.81 |
| **ImitSAT (3 calls)** | **0.75** | **0.83** | **0.75** | **0.78** | **0.74** | **0.76** |

ImitSAT achieves the lowest MRPP across nearly all variable ranges, and also attains the highest 1% win rate $W_{1\%}$ (best across all ranges under 3 calls).

### Generalization: Structured SAT Families (MRPP $\tilde{r}$↓)

| Method | JNH | AIM | PARITY | PHOLE | PRET |
|--------|-----|-----|--------|-------|------|
| Graph-Q-SAT (5 calls) | 1.11 | 1.18 | 0.56 | 0.82 | 0.92 |
| SATformer | 1.36 | 1.01 | 0.73 | 1.00 | 1.00 |
| **ImitSAT (5 calls)** | **0.85** | **0.81** | **0.30** | **0.82** | **0.42** |

Trained exclusively on random 3-SAT, ImitSAT generalizes to structured problems without fine-tuning, with particularly pronounced advantages on PARITY and PRET.

### Wall-Clock Time
- On harder instances (61–100 variables), ImitSAT achieves the lowest wall-clock time, as propagation savings outweigh query overhead.
- On easier instances (16–30, 31–60 variables), vanilla MiniSAT remains faster; however, ImitSAT is the strongest learning-based method and the gap is small.

## Highlights & Insights
- **First imitation learning-based CDCL branching heuristic**: Avoids the exploration instability of RL and provides dense, decision-level supervision signals.
- **Elegant KeyTrace design**: Compresses verbose solver traces to the extreme (4.3% of propagations), with virtually no conflicts upon replay—an ideal expert signal.
- **Natural fit between prefix-conditioned modeling and sequence prediction**: Branching decisions are inherently conditioned on the prefix history, and autoregressive models are a perfect structural match.
- **Plug-and-play**: Requires no modification to other CDCL components, preserves completeness, and uses only a small number of queries (3–5 per instance).

## Limitations & Future Work
- Training and evaluation are restricted to instances with ≤100 variables due to context window constraints; scaling to industrial-scale instances has not been explored.
- On easy instances (16–30 variables), model inference overhead cannot be offset by propagation savings, resulting in worse wall-clock time than vanilla MiniSAT.
- The current implementation uses a Python-based MiniSAT; a C++ implementation may further amplify efficiency gains.
- Hybrid imitation–reinforcement learning schemes (noted by the authors as future work) remain unexplored.
- Optimal strategies for curriculum learning and query budget scheduling require more systematic investigation.

## Related Work & Insights
- **vs. SATformer**: SATformer influences VSIDS scores only at initialization and does not intervene during search; ImitSAT directly guides branching throughout the search process, offering a more direct mechanism.
- **vs. Graph-Q-SAT**: GQSAT uses RL with current-state snapshots, suffering from sparse rewards and no use of history; ImitSAT uses IL with the full decision prefix, yielding more stable training.
- **vs. IL methods for MIP branch-and-bound**: B&B in MIP is a monotone tree search, whereas CDCL involves non-monotone search, restarts, and clause learning, making the dynamics considerably more complex. ImitSAT trains a full-sequence autoregressive policy rather than a local ranking model.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First application of imitation learning to CDCL branching; the KeyTrace expert construction method is original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation across multiple SAT families, ablations, and wall-clock time; limited by variable scale.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation-to-method derivation is coherent and clear; KeyTrace visualization is intuitive.
- **Value**: ⭐⭐⭐⭐ Opens a new direction for learning-augmented SAT solving, though scaling to industrial applications remains a bottleneck.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Latent Wasserstein Adversarial Imitation Learning](latent_wasserstein_adversarial_imitation_learning.md)
- [\[ICLR 2026\] On Discovering Algorithms for Adversarial Imitation Learning](on_discovering_algorithms_for_adversarial_imitation_learning.md)
- [\[ICLR 2026\] Model Predictive Adversarial Imitation Learning for Planning from Observation](model_predictive_adversarial_imitation_learning_for_planning_from_observation.md)
- [\[ICLR 2026\] Near-Optimal Second-Order Guarantees for Model-Based Adversarial Imitation Learning](near-optimal_second-order_guarantees_for_model-based_adversarial_imitation_learn.md)
- [\[NeurIPS 2025\] Quantifying Generalisation in Imitation Learning](../../NeurIPS2025/reinforcement_learning/quantifying_generalisation_in_imitation_learning.md)

</div>

<!-- RELATED:END -->

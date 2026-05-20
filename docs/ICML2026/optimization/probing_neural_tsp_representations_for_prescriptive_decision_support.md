---
title: >-
  [Paper Note] Probing Neural TSP Representations for Prescriptive Decision Support
description: >-
  [ICML 2026][Optimization][TSP] The authors treat a trained TSP neural solver as a "transferable encoder," using frozen representations and lightweight probes to predict two types of expensive operations research sensitiv…
tags:
  - "ICML 2026"
  - "Optimization"
  - "TSP"
  - "neural CO"
  - "probing"
  - "sensitivity analysis"
  - "transfer learning"
date: 2026-05-08
content_hash: 08cfd30ffd9783bf
---

# Probing Neural TSP Representations for Prescriptive Decision Support

**Conference**: ICML 2026  
**arXiv**: [2602.07216](https://arxiv.org/abs/2602.07216)  
**Code**: https://github.com/ReubenNarad/tsp_prescriptive_probe  
**Area**: Neural Combinatorial Optimization / Representation Probing / Decision Support  
**Keywords**: TSP, neural CO, probing, sensitivity analysis, transfer learning

## TL;DR
The authors treat a trained TSP neural solver as a "transferable encoder," using frozen representations and lightweight probes to predict two types of expensive operations research sensitivity queries (node removal and edge forbidding). They systematically demonstrate that probe accuracy improves monotonically with solver quality and that ensembling with traditional heuristics achieves SOTA.

## Background & Motivation

**Background**: Neural combinatorial optimization (NCO) has enabled end-to-end solvers for TSP/VRP using attention-based policies and reinforcement learning (e.g., Pointer Network, Kool 2018, POMO), offering fast and flexible inference. However, compared to classical exact/heuristic solvers like Concorde/LKH, NCO methods remain less robust and are thus positioned as "alternative solvers."

**Limitations of Prior Work**: Nearly all NCO evaluations focus solely on tour cost or optimality gap, discarding model representations as "byproducts." This means that even if the solver encodes valuable logistics structures (e.g., bottleneck nodes, irreplaceable edges), these are not exploited.

**Key Challenge**: Real-world logistics decisions involve more than constructing a good tour; they require "what-if" queries—e.g., which warehouse removal most impacts total length? Which road closure is most critical? These queries are prohibitively expensive to answer via repeated re-solving, yet NCO solvers may encode the answers in a single forward pass.

**Goal**: To formalize two "operations research canonical" downstream tasks (node-removal sensitivity and edge-forbid sensitivity) and systematically test: (1) Can frozen NCO encoders predict these sensitivities? (2) Does encoder utility improve with training? (3) Can simple probe-heuristic ensembles outperform strong baselines?

**Key Insight**: Adapting the "probing" paradigm from NLP—fixing pretrained representations and training only a lightweight classifier/regressor to recover target properties. This approach isolates whether information is explicitly encoded, decoupling "representation quality" from "probe capacity."

**Core Idea**: Treat the TSP solver as a foundation encoder; train DeepSets/Set Transformer probes on node embeddings to directly predict sensitivity scores for each candidate node/edge. Use ensembling to convexly combine probe scores with geometric heuristics, achieving both speed and strength.

## Method

### Overall Architecture
The pipeline consists of three steps: (i) Train the NCO solver—an attention model based on Kool 2018 with REINFORCE rollout baseline, sweeping three residual dimensions (64/128/256), saving checkpoints every 2000 steps; (ii) Offline label generation—use Concorde to find the optimum for each 100-node instance, then enumerate each candidate (node or tour edge), re-solve, and record the optimal length change $\Delta_i$ or $\Delta_e$; (iii) Probe training—freeze the encoder, extract the final layer node embedding $h_i$, use $h_i$ directly for node tasks, and symmetric features $[h_u, h_v, |h_u-h_v|]$ for edge tasks, inputting into linear / DeepSets / Set Transformer heads to predict top-k sensitivities.

### Key Designs

1. **Two Prescriptive Tasks Aligned with Query Timing**:

    - *Function*: Ground vague "what-if" decisions into quantifiable, labelable, and query-aligned supervised tasks.
    - *Mechanism*: Node-removal is *pre-solve advisory*—before route selection, ask "which customer removal saves most," so only instance geometry is allowed, not the tour. Edge-forbid is *post-solve contingency*—the route is fixed but a segment is blocked, so candidates are limited to the $n$ tour edges. Labels are defined as $\Delta_i^{(\%)}=100\cdot(L^\star(X)-L^\star(X\setminus\{i\}))/L^\star(X)$ and $\Delta_e^{(\%)}=100\cdot(L^\star(X|\text{forbid }e)-L^\star(X))/L^\star(X)$, both obtained via repeated Concorde solving.
    - *Design Motivation*: Explicitly incorporate "what information is available at query time" into task definitions, avoiding incomparable oracle baselines; node task candidates are $O(n)$, edge tasks limited to $O(n)$ on the tour, keeping probe training tractable.

2. **Frozen Encoder + Multi-Capacity Probe Family**:

    - *Function*: Reveal "how much sensitivity information is encoded in the representation" under controlled variables.
    - *Mechanism*: For each checkpoint, extract the encoder's final per-node representation $h_i \in \mathbb R^d$, using only forward passes (no autoregressive rollout), so all representations are cached in one inference. The probe family spans a capacity spectrum: linear readout, DeepSets (MLP over sets), Set Transformer (permutation-invariant attention). Training objectives include regression, hard CE, and soft listwise CE, selected automatically by validation; evaluation metrics are top-1/top-5 accuracy and Spearman $\rho$.
    - *Design Motivation*: Use "geometric features + same probe family" as a representation-free control, and "randomly initialized encoder + same probe family" as a representation-quality control, isolating the contributions of "probe capacity" and "solver training" to final performance.

3. **Probe × Heuristic Ensemble + Solver Quality–Representation Quality Correlation**:

    - *Function*: Achieve the strongest predictor in practice and reveal the "better solver ⇒ better probe" pattern.
    - *Mechanism*: For node-removal, per-instance z-score the Set Transformer probe and geometry-only Set Transformer scores, then convexly combine; for edge-forbid, ensemble with 2-opt repair proxy. Coefficients are tuned on a small validation set. To verify scaling laws, probe training is repeated across 3 model sizes (0.44M/1.10M/3.36M) and every 2000-step checkpoint, tracking Spearman $\rho$ between "probe accuracy" and "negative suboptimality."
    - *Design Motivation*: Probe and heuristic signals have complementary errors across instances; convex combination automatically learns when to trust each. The solver quality–representation quality curve directly answers whether "better-trained NCOs provide better downstream representations," the paper's core scientific contribution.

### Loss & Training

Solver: REINFORCE + rollout baseline, Adam lr $10^{-4}$, exponential decay $\gamma=0.998$, batch 512, 600k steps, temperature 0.5 sampling, greedy evaluation. Probe: encoder is frozen, trained only on cached representations; data split 2500/250/250 (node) and 800/100/100 (edge), with training set standardization.

## Key Experimental Results

### Main Results

On TSP100, node-removal and edge-forbid top-1 / top-5 accuracy and Spearman $\rho$ (excerpt from Table 1):

| Method | Node Top-1 | Node Top-5 | Node $\rho$ | Edge Top-1 | Edge Top-5 | Edge $\rho$ |
|---|---|---|---|---|---|---|
| Nearest-neighbor heuristic | 0.440 | 0.857 | 0.613 | – | – | – |
| Detour heuristic | – | – | – | 0.540 | 0.940 | 0.668 |
| Geometry-only Set Transformer | 0.577 | 0.873 | 0.675 | 0.140 | 0.490 | 0.276 |
| Linear probe | 0.413 | 0.769 | 0.405 | 0.410 | 0.720 | 0.468 |
| DeepSets probe | 0.497 | 0.880 | 0.693 | 0.510 | 0.840 | 0.631 |
| Transformer probe | **0.615** | 0.902 | 0.736 | 0.462 | 0.818 | 0.626 |
| Probe + geometry / 2-opt ensemble | **0.653** | **0.933** | **0.739** | **0.730** | **0.980** | **0.763** |

### Ablation Study

| Configuration | Edge Top-1 | Notes |
|---|---|---|
| Linear probe, untrained policy | 0.130 | Probe capacity only, no representation signal |
| Transformer probe, untrained policy | 0.220 | Large probe on random representations |
| Transformer probe, trained policy | 0.462 | Full model; representation brings 24+ point gain |
| 2-opt repair (oracle) | 0.670 | Assumes optimal tour known |
| Ensemble (probe + 2-opt) | 0.730 | Probe complements oracle heuristic's weaknesses |

### Key Findings
- The edge-forbid task, which is "globally structure-sensitive," best demonstrates representation value: geometry-only Set Transformer achieves only 0.14 top-1, but adding solver representations jumps to 0.462, a 3× improvement.
- "Better solver ⇒ better probe" holds monotonically for most model sizes: on the 1.10M model, linear/transformer probe accuracy and solver suboptimality Spearman $\rho$ are 0.71/0.45 (node) and 0.65/0.40 (edge), i.e., representations improve steadily with training.
- Probe accuracy continues to rise even after tour cost plateaus, indicating that traditional NCO metrics (tour length) severely underestimate progress in representation learning.

## Highlights & Insights
- Distinguishing pre-solve/post-solve tasks by query timing and aligning allowed baseline information is an effective way to avoid oracle loopholes; this meta-method can generalize to any prescriptive analytics.
- Viewing NCO solvers as foundation encoders is novel, effectively telling the operations research community that "training NCOs not only yields good solutions but also transferable features," potentially opening the "NCO foundation model" research direction.
- The ensemble scheme is extremely simple (per-instance z-score + convex combination) yet consistently surpasses strong baselines, reminding us that "representation learning" and "traditional heuristics" are complementary, not substitutes.

## Limitations & Future Work
- Evaluation is limited to Euclidean TSP100; it is unknown whether results hold for larger $n$, non-uniform distributions, or constrained VRP.
- Label generation depends on repeated Concorde solving; for edge-forbid, each instance averages 49.6s, making larger probe training sets costly.
- Currently, only two sensitivities are examined; real logistics scenarios involve more complex what-if queries (e.g., dynamic node addition, vehicle capacity adjustment), motivating the design of a unified multi-task probing framework.

## Related Work & Insights
- **vs Zhang 2025 (CS-Probing)**: They use probes to describe whether NCO representations encode certain structures; this work shifts to "using representations to predict economically meaningful decision support metrics," closer to practical deployment.
- **vs Narad 2025 (sparse autoencoders for TSP)**: SAE extracts human-interpretable features, while this work directly supervises probe training on task-relevant signals; the two can be combined—first use SAE to find units, then probes to assess which units contribute to which sensitivities.
- **vs Lozano 2017 (TSP interdiction)**: Classical OR uses integer programming for interdiction; this work replaces exact solving with learned ranking, enabling millisecond-level decision support.

## Rating
- Novelty: ⭐⭐⭐⭐ First to evaluate NCO solvers as transferable encoders for prescriptive downstream tasks.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple probe families, model sizes, training dynamics, and two controls—very systematic.
- Writing Quality: ⭐⭐⭐⭐ Clear task definitions, rigorous heuristic and control experiment design.
- Value: ⭐⭐⭐⭐ Provides new perspectives for both OR and ML communities; code is open-sourced and easily reproducible.

## Related Papers

- [\[NeurIPS 2025\] Probing Neural Combinatorial Optimization Models](../../NeurIPS2025/optimization/probing_neural_combinatorial_optimization_models.md)
- [\[NeurIPS 2025\] Contribution of Task-Irrelevant Stimuli to Drift of Neural Representations](../../NeurIPS2025/optimization/contribution_of_task-irrelevant_stimuli_to_drift_of_neural_representations.md)
- [\[ICML 2026\] Learning to Approximate Uniform Facility Location via Graph Neural Networks](learning_to_approximate_uniform_facility_location_via_graph_neural_networks.md)
- [\[ICML 2026\] Neural QAOA$^2$: Differentiable Joint Graph Partitioning and Parameter Initialization for Quantum Combinatorial Optimization](neural_qaoa2_differentiable_joint_graph_partitioning_and_parameter_initializatio.md)
- [\[ICML 2025\] Adjustment for Confounding using Pre-Trained Representations](../../ICML2025/optimization/adjustment_for_confounding_using_pre-trained_representations.md)

</div>

<!-- RELATED:END -->

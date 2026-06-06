---
title: >-
  [Paper Note] Probing Neural TSP Representations for Prescriptive Decision Support
description: >-
  [ICML 2026][Optimization][TSP] The authors treat trained TSP neural solvers as "transferable encoders," utilizing frozen representations with lightweight probes to predict two types of expensive operations research sensi…
tags:
  - "ICML 2026"
  - "Optimization"
  - "TSP"
  - "neural CO"
  - "probing"
  - "sensitivity analysis"
  - "transfer learning"
date: 2026-05-08
content_hash: 5ff0d2a06176bcec
---

# Probing Neural TSP Representations for Prescriptive Decision Support

**Conference**: ICML 2026  
**arXiv**: [2602.07216](https://arxiv.org/abs/2602.07216)  
**Code**: https://github.com/ReubenNarad/tsp_prescriptive_probe  
**Area**: Neural Combinatorial Optimization / Representation Probing / Decision Support  
**Keywords**: TSP, neural CO, probing, sensitivity analysis, transfer learning

## TL;DR
The authors treat trained TSP neural solvers as "transferable encoders," utilizing frozen representations with lightweight probes to predict two types of expensive operations research sensitivity queries (node removal and edge forbidding). They systematically demonstrate that probe accuracy improves monotonically with solver quality and achieves SOTA through integration with traditional heuristics.

## Background & Motivation

**Background**: Neural Combinatorial Optimization (NCO) has enabled the training of end-to-end solvers for problems like TSP/VRP using attention strategies and Reinforcement Learning (Pointer Network, Kool 2018, POMO, etc.). While fast and flexible, they remain less robust than classical exact/heuristic solvers like Concorde/LKH and have been primarily positioned as "alternative solvers."

**Limitations of Prior Work**: Almost all NCO evaluations focus solely on tour cost or optimality gaps, discarding model representations as "by-products." This means that even if the solver learns structures valuable for logistics (e.g., node bottlenecks or critical edges), these insights are never extracted.

**Key Challenge**: Practical logistics decisions involve more than constructing a good tour; they require "what-if" queries—such as which warehouse removal impacts the total length most, or which road closure is most critical. Answering these through repeated re-solving is computationally prohibitive, yet NCO solvers may already encode these answers within a single forward pass.

**Goal**: To formalize two "prescriptive operations" downstream tasks (node-removal sensitivity and edge-forbid sensitivity) and systematically examine: (1) whether frozen NCO encoders can predict these sensitivities; (2) whether encoders become more useful as training progresses; and (3) whether simple probe-heuristic ensembles can beat strong baselines.

**Key Insight**: Borrowing the "probing" paradigm from NLP—fixing pre-trained representations and training only a lightweight classifier/regressor to recover target attributes. This determines whether "information is explicitly encoded" while controlling for probe capacity, naturally decoupling representation quality from probe architecture.

**Core Idea**: Treat the TSP solver as a foundation encoder. Train DeepSets/Set Transformer probes on node embeddings to directly predict sensitivity scores for each candidate node/edge. Use ensembling to combine probe scores with geometric heuristics via convex combination for speed and performance.

## Method

### Overall Architecture
The pipeline consists of three steps: (i) Train NCO solver—based on the Attention Model (Kool 2018) + REINFORCE rollout baseline, scanning three residual dimensions (64/128/256) and saving checkpoints every 2000 steps; (ii) Offline label generation—for each 100-node instance, find the optimum using Concorde, then enumerate each candidate (node or tour edge) for a re-solve to record the optimal length change $\Delta_i$ or $\Delta_e$; (iii) Train probes—freeze the encoder, extract the final layer node embeddings $h_i$. Use $h_i$ directly for node tasks, and symmetric features $[h_u, h_v, |h_u-h_v|]$ for edge tasks, feeding into Linear / DeepSets / Set Transformer heads to predict top-k sensitivity.

### Key Designs

1. **Two prescriptive tasks and temporal alignment**:

    - **Function**: Grounding vague "what-if" decisions into quantifiable, labeled supervised tasks consistent with actual query scenarios.
    - **Mechanism**: Node-removal is a *pre-solve advisory*—asking "which customer removal saves most" before a route is decided, thus only allowing instance geometry. Edge-forbid is a *post-solve contingency*—asking about road closures after a route is fixed, restricting candidates to $n$ edges on the tour. Labels are defined as $\Delta_i^{(\%)}=100\cdot(L^\star(X)-L^\star(X\setminus\{i\}))/L^\star(X)$ and $\Delta_e^{(\%)}=100\cdot(L^\star(X|\text{forbid }e)-L^\star(X))/L^\star(X)$, obtained via repeated Concorde calls.
    - **Design Motivation**: Explicitly incorporating "what information is available at query time" into the task definition to avoid incomparable oracle baselines; keeping candidate sets at $O(n)$ makes probe training scalable.

2. **Frozen encoder + multi-capacity probe families**:

    - **Function**: Investigating the amount of sensitivity information encoded in representations while controlling variables.
    - **Mechanism**: Extract per-node representations $h_i \in \mathbb R^d$ from the final encoder layer at each checkpoint. Only forward passes are used without autoregressive rollout, allowing one-time caching. Probe families span the capacity spectrum: Linear readout, DeepSets (MLP over sets), and Set Transformer (permutation-invariant attention). Training targets include regression, hard CE, and soft listwise CE; evaluation uses top-1/top-5 accuracy and Spearman $\rho$.
    - **Design Motivation**: Using "geometry features + same probe" as a representation-free control and "randomly initialized encoder + same probe" as a representation-quality control isolates the contributions of probe capacity versus solver training.

3. **Probe × Heuristic Ensemble + Solver-Representation Correlation**:

    - **Function**: Achieving high-performance predictors while revealing the relationship: "Better Solver ⇒ Better Probes."
    - **Mechanism**: For node-removal, Set Transformer probe scores and geometry-only scores are combined via per-instance z-score convex combination. For edge-forbid, combination is done with a 2-opt repair proxy. For scaling laws, probe training is repeated for 3 model sizes (0.44M/1.10M/3.36M) and every 2000 steps to measure the Spearman $\rho$ between probe accuracy and negative gap.
    - **Design Motivation**: Errors of probes and heuristics are complementary across instances; the quality curve answers whether better NCO training serves downstream representations, a core scientific contribution.

### Loss & Training
Solver: REINFORCE + rollout baseline, Adam lr $10^{-4}$, exponential decay $\gamma=0.998$, batch 512, 600k steps, temperature 0.5 sampling, greedy evaluation. Probes: Encoder remains frozen; trained on cached representations with data splits of 2500/250/250 (node) and 800/100/100 (edge).

## Key Experimental Results

### Main Results

Top-1 / Top-5 accuracy and Spearman $\rho$ for node-removal and edge-forbid on TSP100 (selected from Table 1):

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

| Configuration | Edge Top-1 | Description |
|---|---|---|
| Linear probe, untrained policy | 0.130 | Probe capacity only, no representation signal |
| Transformer probe, untrained policy | 0.220 | High-capacity probe on random representations |
| Transformer probe, trained policy | 0.462 | Full model; representations yield 24+ point gain |
| 2-opt repair (oracle) | 0.670 | Assuming optimal tour is known |
| Ensemble (probe + 2-opt) | 0.730 | Probe compensates for oracle heuristic weaknesses |

### Key Findings
- Tasks involving "global structural sensitivity" like Edge-forbid best demonstrate representation value: geometry-only models reach only 0.14 top-1, while adding solver representations jumps to 0.462 (3× improvement).
- "Better Solver ⇒ Better Probes" holds monotonically across most sizes. In the 1.10M model, Spearman $\rho$ between probe accuracy and solver status was 0.71/0.45 (node) and 0.65/0.40 (edge) for linear/transformer probes, indicating representations improve throughout training.
- Probe accuracy continues to rise even after tour cost has plateaued, suggesting traditional NCO metrics underestimate the progress of representation learning.

## Highlights & Insights
- Using "query timing" to distinguish pre-solve/post-solve tasks and aligning them with baseline information is an excellent way to avoid oracle loopholes; this meta-method generalizes to any prescriptive analytics.
- Viewing the NCO solver as a "foundation encoder" for the first time implies that training NCO yields transferable features, potentially opening the "NCO foundation model" research direction.
- The ensemble scheme is simple (z-score + convex combination) yet consistently outperforms strong baselines, highlighting that "learned representations" and "traditional heuristics" are complementary rather than mutually exclusive.

## Limitations & Future Work
- Evaluations are limited to Euclidean TSP100; results for larger $n$, non-uniform distributions, or constrained VRP remain unknown.
- Label generation relies on repeated Concorde solves, averaging 49.6s per edge-forbid instance, impacting the cost of larger probe training sets.
- Currently only two sensitivities are examined; actual logistics involve complex "what-ifs" like dynamic point addition or capacity adjustments requiring a unified multi-task probing framework.

## Related Work & Insights
- **vs Zhang 2025 (CS-Probing)**: While they use probes to check for structural encoding, this paper focuses on "predicting economically meaningful decision support metrics," which is closer to real-world application.
- **vs Narad 2025 (sparse autoencoders for TSP)**: SAE extracts human-interpretable features; this paper supervises task-relevant probes. These could be combined to identify which interpretability units contribute to specific sensitivities.
- **vs Lozano 2017 (TSP interdiction)**: Classical OR solves interdiction with integer programming; this work replaces exact solving with learned ranking, returning decision suggestions in milliseconds.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First to evaluate NCO solvers as transferable encoders for prescriptive downstream tasks.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Systematic approach involving multiple probe families, model sizes, training dynamics, and controls.
- **Writing Quality**: ⭐⭐⭐⭐ Clear task definitions and rigorous design of heuristics and control experiments.
- **Value**: ⭐⭐⭐⭐ Provides new perspectives for both OR and ML communities; code is open-source and reproducible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Probing Neural Combinatorial Optimization Models](../../NeurIPS2025/optimization/probing_neural_combinatorial_optimization_models.md)
- [\[ICML 2026\] Support-Proximity Augmented Diffusion Estimation for Offline Black-Box Optimization](support-proximity_augmented_diffusion_estimation_for_offline_black-box_optimizat.md)
- [\[NeurIPS 2025\] Contribution of Task-Irrelevant Stimuli to Drift of Neural Representations](../../NeurIPS2025/optimization/contribution_of_task-irrelevant_stimuli_to_drift_of_neural_representations.md)
- [\[ICML 2026\] URS: A Unified Neural Routing Solver](urs_a_unified_neural_routing_solver_for_cross-problem_zero-shot_generalization.md)
- [\[ICML 2026\] Learning to Approximate Uniform Facility Location via Graph Neural Networks](learning_to_approximate_uniform_facility_location_via_graph_neural_networks.md)

</div>

<!-- RELATED:END -->

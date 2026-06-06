---
title: >-
  [Paper Note] LoRe: Adaptive Interaction-Evaluation Routing with Per-Step Interaction Budgets for Iterative Graph Solvers
description: >-
  [ICML 2026][Optimization][Per-step Interaction Budgets] LoRe ports the "Cluster + Bath" decomposition from condensed matter physics to diffusion-based graph combinatorial optimization (CO) solvers as a training-free infe…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Per-step Interaction Budgets"
  - "Dynamic Routing"
  - "Cluster-Bath Decomposition"
  - "Training-free"
  - "MIS/TSP"
date: 2026-05-08
content_hash: ec4e5cc83216a69f
---

# LoRe: Adaptive Interaction-Evaluation Routing with Per-Step Interaction Budgets for Iterative Graph Solvers

**Conference**: ICML 2026  
**arXiv**: [2605.29005](https://arxiv.org/abs/2605.29005)  
**Code**: To be confirmed  
**Area**: Combinatorial Optimization / Diffusion-based Neural Solvers / Inference Acceleration  
**Keywords**: Per-step Interaction Budgets, Dynamic Routing, Cluster-Bath Decomposition, Training-free, MIS/TSP

## TL;DR
LoRe ports the "Cluster + Bath" decomposition from condensed matter physics to diffusion-based graph combinatorial optimization (CO) solvers as a training-free inference-time wrapper. By evaluating only a fixed proportion of high-conflict edges at each step and compensating for the discarded parts with an $\mathcal{O}(N)$ global recall term, it enables Maximum Independent Set (MIS) solvers to exceed the baseline OOM limit by $3\times$ (solving $n=50\mathrm{k}$ instances on a single card), while achieving $\sim 15\times$ speedup and $44\times$ memory compression on TSP $n=1000$.

## Background & Motivation

**Background**: Diffusion-based and GNN-based neural solvers like DIFUSCO, DiffUCO, T2TCO, and COExpander treat CO problems (MIS, TSP, etc.) as iterative denoising processes on graphs. They repeatedly perform message passing on a dense interaction set $\mathcal{A}$ (all edges in MIS, candidate moves in TSP) to resolve conflicts, which is the current mainstream approach for learnable CO solvers.

**Limitations of Prior Work**: The cost of these solvers is $\mathcal{O}(T|\mathcal{A}|)$, and the **peak memory of each step is linearly related to $|\mathcal{A}|$**. On industrial-scale instances ($n \ge 20\mathrm{k}$ ER graphs or $n \ge 500$ dense TSPs), single-step dense message passing hits GPU limits, causing either OOM or unacceptable latency. "Anytime" scenarios like scheduling and network allocation require feasible solutions within specific latency and memory budgets.

**Key Challenge**: Reducing the number of steps (via distillation or Fast-T2T) only affects $T$ and **does not lower the single-step peak memory**. Static spatial sparsification (fixed kNN candidate graphs, fixed masks) can reduce single-step overhead, but "conflict hotspots" in CO solvers drift along the trajectory. If a critical edge for the current round is permanently pruned, truncation errors accumulate, causing the trajectory to deviate completely. In other words, **single-step budget constraints** and **support set drift over time** must be addressed simultaneously.

**Goal**: Incorporate the hard constraint of "evaluating only a fixed proportion $\rho$ of $|\mathcal{A}|$ at each step" into the solver loop, while ensuring: (a) no retraining of the backbone, (b) no loss in solution quality, and (c) end-to-end wall-clock auditability.

**Key Insight**: The authors noted that this dilemma is structurally isomorphic to multi-body problems in strongly correlated condensed matter physics. Cluster Dynamical Mean-Field Theory (C-DMFT) decomposes interactions of infinite lattices into an "exactly solved local cluster" and an "approximately compensated mean-field bath." Natural clusters (high-conflict neighborhoods) and baths (stable background relationships) also exist in CO solving, allowing for the adoption of this algorithmic blueprint.

**Core Idea**: Use a time-varying subset $M_t \subseteq \mathcal{A}$ as the cluster for precise edge message passing, and use an $\mathcal{O}(N)$ coverage-weighted global signal as the bath to compensate for discarded edges. **Hotspots that drift are tracked dynamically using a proxy score refreshed every $R$ steps.**

## Method

### Overall Architecture
The authors formalize the iterative solver as a discrete dynamical system $x^{t+1} = \Pi_t\big(\mathcal{T}_t(x^t; \mathcal{A})\big)$, where $x^t \in \mathbb{R}^{n \times d}$ is the hidden state of $n$ variables, $\Pi_t$ is a lightweight projection/repair/decoding operator, and $\mathcal{T}_t$ is the main message passing operator. The latter is split into a node term $\mathcal{B}_t(x)$ and an edge interaction term $\sum_{a \in \mathcal{A}} \Delta_{t,a}(x)$. LoRe does not change the backbone parameters or total steps $T$, but replaces the second term with a budget-constrained version $\tilde{\mathcal{T}}_t(x; M_t, g_t) = \mathcal{B}_t(x) + \sum_{a \in M_t} \Delta_{t,a}(x) + \mathcal{R}_t(x; g_t)$, with the constraint $|M_t| \le B = \lfloor \rho |\mathcal{A}| \rfloor$. The pipeline consists of three components: dynamic routing to select $M_t$, optional global recall $\mathcal{R}_t$, and shared projection/greedy decoding $\Pi_t$. All variants use the same DIFUSCO checkpoint and $\Pi_t$, so reported end-to-end speedups stem entirely from the compression of the interaction operator.

### Key Designs

1.  **Dynamic Routing (Cluster Selection)**:
    *   **Function**: Select the $B$ most valuable interactions for precise evaluation from $\mathcal{A}$ at each refresh step.
    *   **Mechanism**: $M_t$ consists of two parts: a fixed small skeleton $E_{\mathrm{skel}}$ (the top $\lfloor \gamma B\rfloor$ edges ranked by degree $\deg(i)+\deg(j)$) to ensure structural key edges are always present, and the remaining budget selected from $E \setminus E_{\mathrm{skel}}$ based on a top-$(B-|E_{\mathrm{skel}}|)$ proxy score $S_t$. For MIS, the score combines endpoint uncertainty and temporal instability: $S_t((i,j); x^t, x_{\text{prev}}) = u_i u_j + \lambda_{\mathrm{stab}}(|x^t_i - x_{\text{prev},i}| + |x^t_j - x_{\text{prev},j}|)$, where node uncertainty $u_i = 1 - |2 x^t_i - 1|$ peaks at $x^t_i = 1/2$ and approaches 0 for decided nodes. To reduce overhead, $M_t$ is reselected only every $R$ steps.
    *   **Design Motivation**: High-conflict points drift along the diffusion trajectory; static kNN/masks cannot track them. By focusing on uncertain and unstable edges, the system locks onto "unconverged trouble zones" and skips already stabilized edges, utilizing the hard budget where it matters most.

2.  **Cluster-Bath Global Recall (Optional Stability Term)**:
    *   **Function**: Use an $\mathcal{O}(N)$ global signal to approximate interactions in the discarded set $\mathcal{A} \setminus M_t$, preventing the cluster subgraph from becoming isolated.
    *   **Mechanism**: Aggregate signals from discarded interactions as $g_t = \text{Pool}_t(x^t; \mathcal{A} \setminus M_t)$, then inject $g_t$ back into nodes via a parameterized coverage interpolation $U_t([x^t, g_t])_i = \alpha_i x^t_i + (1-\alpha_i) g_{t,i}$, where $\alpha_i = d_i(M_t) / d_i(\mathcal{A})$ is the ratio of precise neighbor evaluations for node $i$. Higher precise coverage leads to higher self-reliance, while lower coverage shifts trust to the background field. This step requires no training and only caches one tensor; it can be toggled on or off.
    *   **Design Motivation**: Pure routing under ultra-low budgets may "lose context," leading to trajectory drift. A cheap mean-field-like background correction can suppress the bath residual $\|r_t\|$ in the error bound when $\rho$ is small without introducing additional operator evaluations. The main table disables this to isolate the routing contribution, but it is recommended for ultra-low budgets.

3.  **Auditable End-to-End Accounting Protocol**:
    *   **Function**: Compare all variants (baseline, static sparse, LoRe) under a comprehensive wall-clock and peak memory accounting system to avoid common overestimations like "calculating GNN but ignoring decoding."
    *   **Mechanism**: All methods share the same DIFUSCO implementation, greedy decoding, and task-level feasibility repair (including standard 2-opt for TSP). LoRe only modifies the single-step active interaction set $M_t$. Thus, the wall-clock/memory ratio directly reflects gains from interaction operator compression. The paper also provides an informal error bound $e_{t+1} \le L_t e_t + \|\delta_t\|$, where $\|\delta_t\| \le \epsilon_t(\rho) + \|r_t\|$. Routing ensures $\epsilon_t(\rho)$ remains small by placing high-influence edges in the cluster.
    *   **Design Motivation**: Many CO papers inflate speedup ratios by compressing operators while allowing post-processing to expand. Unified accounting and shared $\Pi_t$ ensure apples-to-apples comparisons and make the claim "speedup comes entirely from interaction budgets" falsifiable.

### Loss & Training
LoRe is an inference-time wrapper and **does not change any training procedures**. It uses pre-trained weights from original DIFUSCO/COExpander models. Hyperparameters include budget ratio $\rho$, skeleton proportion $\gamma$, refresh interval $R$, and stability coefficient $\lambda_{\mathrm{stab}}$. A single set of default un-tuned configurations was used for all experiments.

## Key Experimental Results

### Main Results
Hardware: NVIDIA RTX PRO 6000 (96 GB). All timings include decoding and repair. MIS on ER graphs ($p=0.05$) scaling is shown below (base = DIFUSCO; LoRe/base: lower is better; Retention: quality ratio, $\ge 1$ means no loss vs. baseline):

| Task | Scale $n$ | Time LoRe/base (s) | Memory LoRe/base (GB) | Mem. Compression | Speedup $\times$ | Quality Retention |
|------|-----------|--------------------|-----------------------|------------------|-----------------|-------------------|
| MIS  | 1k        | 7.9 / 17.3         | 0.07 / 0.42           | 5.7$\times$      | 2.19±0.03       | 0.815±0.048       |
| MIS  | 3k        | 18.6 / 149         | 0.35 / 3.51           | 10.0$\times$     | 8.03±0.03       | 0.835±0.017       |
| MIS  | 8k        | 124 / 1030         | 2.15 / 24.7           | 11.5$\times$     | 8.28±0.12       | 1.019±0.014       |
| MIS  | 15k       | 442 / 3604         | 7.32 / 86.7           | 11.9$\times$     | 8.16±0.04       | 1.010±0.013       |
| MIS  | 20k       | 767 / **OOM**      | 12.9 / **OOM**        | –                | –               | –                 |
| MIS  | 50k       | 4949 / **OOM**     | 79.5 / **OOM**        | –                | –               | –                 |
| TSP  | 500       | 0.72 / 3.61        | 0.05 / 1.23           | 24.6$\times$     | 5.10±0.39       | 0.953±0.014       |

The baseline hits OOM at $n=20\mathrm{k}$, whereas LoRe handles $n=50\mathrm{k}$ with only 79.5 GB peak memory, pushing the feasible inference boundary by $\ge 3\times$. For $n \ge 5\mathrm{k}$, quality retention exceeds 1, suggesting budget constraints stabilize large-scale trajectories.

### Ablation Study

| Configuration | Key Findings | Explanation |
|---------------|--------------|-------------|
| LoRe vs. static kNN (same budget $\rho$) | LoRe is strictly better across all $n$ | Static supports miss drifting hotspots, leading to error accumulation. |
| LoRe vs. static + greedy refresh | LoRe still dominates | Greedy refresh without uncertainty/instability scoring is insufficient. |
| Without global recall (default) | Main table performance is stable | Pure routing is often sufficient; recall is insurance for ultra-low $\rho$. |
| TSP topology transfer (zero-shot) | Tour quality comparable to baseline | Routing selects edges based on state, naturally robust to train-test drift. |

### Key Findings
*   **Dynamic > Static**: Strictly verified under matching budgets, explaining why neural CO cannot simply copy fixed candidate edge strategies from LK.
*   **Quality Ratio > 1 for large $n$**: Dense evaluation on large graphs likely overfits to noise; LoRe's budget constraint acts as an implicit regularizer.
*   **Hyperparameter Insensitivity**: A single un-tuned configuration worked for both MIS and TSP, suggesting the approach is deployment-friendly.

## Highlights & Insights
*   **Physics Analogy as Engineering Blueprint**: The authors borrow the "local precise + global approximate" mode from C-DMFT. Unlike common "physics-as-decoration" papers, the analogy maps directly to the three core algorithmic components.
*   **Auditable End-to-End Accounting**: This is the most reproducible aspect. All CO acceleration works should use this protocol to ensure comparable speedup ratios.
*   **Engineering Value of Drop-in Wrappers**: Pushing the OOM boundary by 3x without changing checkpoints or horizons provides a nearly free benefit for deployment-based CO services.

## Limitations & Future Work
*   Acceleration conclusions are primarily validated on DIFUSCO. While transfers to T2TCO and COExpander are shown, more evidence is needed for RL-based or non-diffusion GNN solvers.
*   The scoring function $S_t$ is task-specific (hand-designed for MIS/TSP); adapting to MaxCut or SAT might require re-tuning.
*   The error bound is based on an informal local Lipschitz assumption without a formal theorem on graph families where $\epsilon_t(\rho)$ is guaranteed to be controllable.
*   The gain of the global recall term $\mathcal{R}_t$ at ultra-low $\rho$ is only briefly mentioned in the appendix.

## Related Work & Insights
*   **vs. DIFUSCO / DiffUCO / Sanokowski et al.**: This work serves as a runtime routing layer rather than a replacement, making it more deployment-friendly.
*   **vs. Fast-T2T / Distillation**: These reduce total steps $T$ while LoRe reduces single-step operators. The two paths are orthogonal and combinable.
*   **vs. Static Spatial Sparsification (kNN / Fixed mask / LK edges)**: The key difference is **time-varying vs. permanent**. Static pruning loses drifting hotspots; LoRe delays decisions to runtime via budget constraints and dynamic scoring.

## Rating
*   Novelty: ⭐⭐⭐⭐ Incorporating C-DMFT into neural CO solvers is a first. While the technical pieces (top-k + interpolation) are not radical, the combination and perspective are novel.
*   Experimental Thoroughness: ⭐⭐⭐⭐ MIS scaling reached OOM limits; TSP scaling and topology transfer are included. Lacks transfer evidence for non-diffusion backbones.
*   Writing Quality: ⭐⭐⭐⭐ Clear distinction between claims and intuitions regarding error bounds and physics analogies.
*   Value: ⭐⭐⭐⭐ High engineering utility for neural CO deployment; formalizes budget constraints at the solver operator level.

## Related Papers

- [\[ICML 2026\] On the Interaction of Batch Noise, Adaptivity, and Compression, under $(L_0,L_1)$-Smoothness: An SDE Approach](on_the_interaction_of_batch_noise_adaptivity_and_compression_under_l_0l_1-smooth.md)
- [\[ICML 2026\] Adaptive Sharpness-Aware Minimization with a Polyak-type Step size: A Theory-Grounded Scheduler](adaptive_sharpness-aware_minimization_with_a_polyak-type_step_size_a_theory-grou.md)
- [\[ICLR 2026\] FrontierCO: Real-World and Large-Scale Evaluation of Machine Learning Solvers for Combinatorial Optimization](../../ICLR2026/optimization/frontierco_real-world_and_large-scale_evaluation_of_machine_learning_solvers_for.md)
- [\[ICML 2026\] Distribution-Free Uncertainty Quantification for Continuous AI Agent Evaluation](distribution-free_uncertainty_quantification_for_continuous_ai_agent_evaluation.md)
- [\[ICML 2026\] SPSsafe: Safeguarded Stochastic Polyak Step Sizes for Non-smooth Optimization](safeguarded_stochastic_polyak_step_sizes_for_non-smooth_optimization_robust_perf.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] On the Interaction of Batch Noise, Adaptivity, and Compression, under $(L_0,L_1)$-Smoothness: An SDE Approach](on_the_interaction_of_batch_noise_adaptivity_and_compression_under_l_0l_1-smooth.md)
- [\[ICML 2026\] Adaptive Sharpness-Aware Minimization with a Polyak-type Step size: A Theory-Grounded Scheduler](adaptive_sharpness-aware_minimization_with_a_polyak-type_step_size_a_theory-grou.md)
- [\[ICLR 2026\] FrontierCO: Real-World and Large-Scale Evaluation of Machine Learning Solvers for Combinatorial Optimization](../../ICLR2026/optimization/frontierco_real-world_and_large-scale_evaluation_of_machine_learning_solvers_for.md)
- [\[ICML 2026\] URS: A Unified Neural Routing Solver](urs_a_unified_neural_routing_solver_for_cross-problem_zero-shot_generalization.md)
- [\[ICML 2026\] Distribution-Free Uncertainty Quantification for Continuous AI Agent Evaluation](distribution-free_uncertainty_quantification_for_continuous_ai_agent_evaluation.md)

</div>

<!-- RELATED:END -->

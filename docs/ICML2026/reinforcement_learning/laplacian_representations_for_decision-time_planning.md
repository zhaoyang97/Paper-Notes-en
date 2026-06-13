---
title: >-
  [Paper Note] Laplacian Representations for Decision-Time Planning
description: >-
  [ICML 2026][Reinforcement Learning][Laplacian representations] This paper proposes ALPS, which utilizes the eigenvector space of the graph Laplacian (scaled to approximate commute-time distance) as a latent space for hie…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Laplacian representations"
  - "hierarchical planning"
  - "decision-time planning"
  - "CEM"
  - "offline goal-conditioned RL"
date: 2026-05-08
content_hash: 5d6b0425863a8373
---

# Laplacian Representations for Decision-Time Planning

**Conference**: ICML 2026  
**arXiv**: [2602.05031](https://arxiv.org/abs/2602.05031)  
**Code**: https://github.com/machado-research/ALPS  
**Area**: Reinforcement Learning / Representation Learning / Model-based RL  
**Keywords**: Laplacian representations, hierarchical planning, decision-time planning, CEM, offline goal-conditioned RL  

## TL;DR
This paper proposes ALPS, which utilizes the eigenvector space of the graph Laplacian (scaled to approximate commute-time distance) as a latent space for hierarchical decision-time planning. It first discovers subgoals via k-means in this space and runs Dijkstra to generate high-level paths, followed by short-range low-level planning in the original state space using CEM with a behavioral prior. On OGBench offline goal-conditioned RL, this model-based planning method systematically outperforms model-free SOTA for the first time.

## Background & Motivation

**Background**: Model-based reinforcement learning has theoretical advantages over model-free RL in sample efficiency, generalization, and adaptation speed. Decision-time planning (e.g., MPC, MCTS) is a standard approach to transform learned models into behaviors. However, difficult offline goal-conditioned RL benchmarks like OGBench have long been dominated by model-free methods such as HIQL, CRL, and QRL, while model-based planning methods are rarely seen in long-horizon tasks.

**Limitations of Prior Work**: The core challenge of decision-time planning is **compounding error**—learned one-step models deviate rapidly from real dynamics after repeated rollouts over long horizons, causing optimizers like CEM to make incorrect decisions based on "hallucinated trajectories." Hierarchical planning is a recognized solution (where the high-level picks subgoals and the low-level executes short-range paths), but it requires a latent space that simultaneously satisfies two contradictory needs: proximity for nearby states (supporting local cost calculation) and preservation of long-range reachability (supporting high-level path search).

**Key Challenge**: Common contrastive learning latent spaces (e.g., the random walk contrastive objective in PcLast) perform well in low-level distance estimation but **lack explicit encoding of global connectivity**. This leads to subgoals generated via k-means that often cross obstacles or produce unreachable high-level paths. Conversely, using raw Euclidean distance ignores environmental dynamics (e.g., two points in a maze being close in Euclidean distance but requiring a long detour).

**Goal**: To find a latent space such that: (1) distance approximates commute-time distance (CTD), supporting both high- and low-level planning; (2) it is naturally suited for spectral clustering for subgoal discovery; (3) it can be learned from samples without relying on the $O(|\mathcal{S}|^3)$ cost of exact eigendecomposition.

**Key Insight**: The authors observe that the eigenvectors of the graph Laplacian are specifically designed to express "multi-scale graph structure"—the first few eigenvectors encode global structures (rooms, regions), while subsequent ones encode local details. Furthermore, spectral clustering theoretically guarantees partitioning the graph along "bottlenecks," naturally corresponding to corridors connecting rooms in navigation tasks. Crucially, the scaled Laplacian representation $\psi_i(s) = \phi_i(s)/\sqrt{\lambda_i}$ is equivalent to CTD under Euclidean distance, allowing it to serve concurrently as a low-level cost and a high-level distance measure.

**Core Idea**: Use the learned scaled Laplacian representation $\psi$ as a unified latent space. Perform k-means clustering in the $\psi$ space to obtain subgoals and run Dijkstra to find high-level paths. At the low-level, utilize CEM with a behavioral prior for short-range optimization in the original state space, successfully returning model-based planning to the OGBench leaderboard.

## Method

### Overall Architecture
ALPS consists of a two-stage algorithm: pre-training and decision-time planning. In the **pre-training** phase, three components are learned from the offline dataset $\mathcal{D}$: (1) Laplacian representation $\phi$ (rescaled to $\psi$), (2) a one-step forward model $f$ on the raw state space, and (3) a goal-conditioned behavior prior $\pi_{\text{prior}}$. Subsequently, k-means is performed in the $\psi$ space to obtain $C$ clusters, and a cluster graph $G_c$ is constructed using cluster centroids as vertices and observed transitions as edges. At **decision-time**, given $(s_{\text{start}}, s_{\text{goal}})$, both states are mapped to $\psi$ space to determine their clusters $(c_s, c_g)$. Dijkstra is run on $G_c$ to calculate the shortest cluster path $\mathcal{P}_G$ as the high-level plan. At each step, CEM performs short-range optimization toward the $\psi$ representation of the current target cluster centroid. As the agent enters the next cluster, the high-level pointer advances; if it deviates from $\mathcal{P}_G$, re-planning is triggered.

### Key Designs

1. **Scaled Laplacian Representation $\psi$ as a Unified Latent Space**:

    - **Function**: Maps any state $s$ to a $D$-dimensional vector $\psi(s) = \phi(s) \oslash \sqrt{\lambda}$, where $\phi$ are the first $D$ non-zero eigenvectors of the graph Laplacian and $\lambda$ are the corresponding eigenvalues. The Euclidean distance satisfies $c(u,v) \approx \|\psi(u) - \psi(v)\|^2$, approximating the commute-time distance.
    - **Mechanism**: Use ALLO (Augmented Lagrangian Laplacian Objective) to learn $\phi$ from samples. The objective is $\max_\beta \min_u \sum_i \langle u_i, L u_i \rangle + \sum_{j,k} \beta_{jk}(\langle u_j, [[u_k]]\rangle - \delta_{jk}) + B \cdot (\cdot)^2$, where the stop-gradient $[[\cdot]]$ and Lagrange multipliers $\beta$ enforce ortho-normalization constraints. Eigenvalues are extracted directly from dual variables $\lambda_i = -\beta_{ii}/2$, and scaled as $\psi_i = \sqrt{2}\phi_i/\sqrt{-\beta_{ii}}$. Training pairs $(S_t, S_{t+\Delta})$ are sampled with $\Delta \sim \text{Geom}(1-\gamma_s)$ from a geometric distribution to avoid the $O(|\mathcal{S}|^3)$ cost of exact eigendecomposition.
    - **Design Motivation**: CTD simultaneously encodes local "one-step reachability" and global "detour distance." Consequently, the **same latent space** can serve as the cost function for low-level CEM (distance to subgoal) and the implicit foundation for high-level Dijkstra edge weights (k-means clustering in this space automatically adheres to environmental bottlenecks), unlike PcLast which maintains two distinct latent spaces.

2. **CEM Low-level Planning Accelerated by Behavioral Prior**:

    - **Function**: For each high-level subgoal $z_{\text{sub}}$, CEM optimizes the cost $J^m = \sum_{t=1}^H (\|\psi(\hat{S}_t^m) - z_{\text{sub}}\|_2^2 + \lambda \|A_t^m\|_2^2)$ within an action window of length $H$ to select the initial action.
    - **Mechanism**: Standard CEM samples action sequences from uninformative Gaussian distributions, which converges slowly in high-dimensional action spaces. ALPS learns a deterministic prior $\pi_{\text{prior}}(S_t, \psi(S_t), \psi(S_{t+k}))$ via goal-conditioned behavior cloning, targeting the regression of $A_t$ with $k \sim U(1, K_{\max})$. During planning, $\pi_{\text{prior}}$ is combined with a learned multi-step autoregressive forward model $f$ (trained with $\frac{1}{H_f}\sum_{\tau=1}^{H_f} \|\hat{S}_{t+\tau} - S_{t+\tau}\|_2^2$, backpropagating errors through time) to roll out a mean action sequence $\mathbf{a}_{t:t+H-1}$. Then, $N_s$ candidates are generated by adding time-correlated Gaussian noise around this mean, ranked by cost to update the distribution via top-$N_e$ elites over $N_{\text{iter}}$ iterations.
    - **Design Motivation**: The behavioral prior biases the initial CEM search toward actions that "resemble goal-oriented trajectories in the dataset," avoiding random search. The multi-step autoregressive forward model mitigates compounding errors during rollouts, enabling CEM to operate reliably within the model's trust window.

3. **High-level Dijkstra Planning and Drift Re-planning via Cluster Graph**:

    - **Function**: Partition the $\psi$ space into $C$ clusters using k-means. Cluster centroids serve as vertices of $G_c$, and edges are defined by observed "cluster $i \to$ cluster $j$" transitions. Nucleus sampling retains only the top-$p\%$ frequent neighbors to avoid unreachable edges. Dijkstra calculates the shortest path $\mathcal{P}_G$. The agent validates its current cluster $c_{\text{curr}}$ at each step; if $c_{\text{curr}} \notin \mathcal{P}_G$, Dijkstra is re-executed from the current cluster.
    - **Mechanism**: Spectral clustering (equivalent to k-means in $\psi$ space) partitions the state space along environmental bottlenecks—e.g., maze rooms separated by corridors. State pairs with high CTD are naturally assigned to different clusters. This decomposes long-horizon problems into discrete subtasks of "traversing rooms," where each segment length fits within the forward model's trust window.
    - **Design Motivation**: Decomposing long-horizon problems into short-horizon subtasks is the core mechanism for mitigating compounding error. Cluster graph discretization reduces planning complexity from continuous search to graph search on $|C|$ vertices, which is near-instantaneous. Drift re-planning ensures that even if low-level CEM enters an off-plan cluster, the high-level can immediately provide a correction.

### Loss & Training
ALLO uses $\gamma_s$ to control time scales (geometric distribution parameter) and $B$ as a barrier coefficient (reported as insensitive). The forward model is trained with $H_f$-step autoregressive MSE. The behavioral prior uses MSE behavior cloning. Key CEM hyperparameters include planner horizon $H$, number of samples $N_s$, elite count $N_e$, iterations $N_{\text{iter}}$, action penalty $\lambda$, and subgoal reach threshold $\epsilon$.

## Key Experimental Results

### Main Results

| Dataset | Metric | ALPS | Prev. SOTA (Model-free) | Gain |
|--------|------|------|---------|------|
| pointmaze-large-stitch-v0 | Success Rate % | 96 ±2 | QRL 84 ±15 | +12 |
| pointmaze-giant-stitch-v0 | Success Rate % | 98 ±1 | QRL 50 ±8 | +48 |
| antmaze-large-navigate-v0 | Success Rate % | 93 ±5 | HIQL 91 ±2 | +2 |
| antmaze-giant-navigate-v0 | Success Rate % | 69 ±9 | HIQL 65 ±5 | +4 |
| pointmaze-giant-navigate-v0 | Success Rate % | 67 ±11 | QRL 68 ±7 | -1 (Tied) |

On OGBench, ALPS is significantly superior to all model-free baselines (GCBC/GCIVL/GCIQL/QRL/CRL/HIQL) according to the Wilcoxon test with Holm-Bonferroni correction at $p<0.001$. The most dramatic performance gains occur in "stitch" datasets—where methods must learn to concatenate short trajectories. Model-free methods largely fail here (e.g., HIQL scores 0 on pointmaze-giant-stitch), while ALPS achieves 98%.

### Ablation Study

| Configuration | Hallway | Rooms | Spiral | Description |
|------|---------|-------|--------|------|
| PcLast (1 cluster, low-level only) | 51 ±4 | 30 ±3 | 35 ±4 | Contrastive latent space |
| PcLast (16 clusters) | 62 ±4 | 57 ±10 | 60 ±6 | + high-level planning |
| ALPS† (1 cluster, low-level only) | 94 ±3 | 92 ±3 | 91 ±4 | $\psi$ space, no behavior prior |
| ALPS† (16 clusters) | 97 ±2 | 96 ±2 | 94 ±2 | + high-level planning |

### Key Findings
- The $\psi$ space represents a critical contribution: simply replacing the latent space (ALPS† 1 cluster vs PcLast 1 cluster) increases success rates from 51% to 94% in Hallway and 30% to 92% in Rooms, indicating that Laplacian/CTD geometry is significantly more reliable for low-level cost estimation than contrastive objectives.
- High-level planning is essential for PcLast (performance drops by 11–27 percentage points without it) but is merely an enhancement for ALPS (1 cluster vs 16 clusters differs by only 2–5 points). This suggests that distance in $\psi$ space already implicitly encodes global topology, allowing low-level CEM to avoid most obstacles by simply traversing toward $\psi(g)$.
- "Stitch" datasets demonstrate the natural advantage of model-based planning—as long as the forward model learns local transitions, the planner can discover new paths by concatenating sub-trajectories, whereas model-free value functions are limited by the data distribution.
- "Teleport" tasks (where teleportation mechanisms violate CTD assumptions) are a weakness for ALPS, with only 13% on pointmaze-teleport-stitch. This is because scaled Laplacian distance assumes locally smooth dynamics, which are broken by state jumps.

## Highlights & Insights
- Using a single latent space for "high-level subgoal discovery + high-level distance + low-level cost" avoids inconsistencies associated with maintaining multiple representations. This unity originates from the mathematical relationship between CTD and spectral clustering—the former provides a distance metric while the latter provides partitioning, satisfying both low-level and high-level requirements.
- Reintroducing classic geometric concepts like commute-time distance from graph theory into deep RL, and making them learnable via the differentiable ALLO objective, represents an elegant synthesis of representation learning and planning. This concept of "latent spaces explicitly encoding dynamics" is transferable to various domains requiring long-range reasoning (e.g., embodied navigation, molecular design).
- Enhancing CEM from a "black-box optimizer" to a "data-aware planner" using a behavioral prior is an efficient way to leverage implicit knowledge from offline data. The computational cost of the prior is minimal (a BC network), yet it allows CEM to perform far fewer iterations in high-dimensional action spaces.

## Limitations & Future Work
- Teleport-stitch tasks are a notable weakness because the scaled Laplacian assumes locally smooth dynamics; any environment featuring portals or state jumps distorts the CTD metric.
- ALLO learns a Laplacian based on the behavioral policy $\pi$, meaning representation quality depends heavily on dataset coverage. The optimal representations for "explore" datasets (high coverage) versus "navigate/stitch" datasets may differ; the paper lacks a systematic ablation of the impact of dataset types on $\psi$ quality.
- Future improvements could involve merging multiple dataset policies to train $\psi$ or performing online fine-tuning. Additionally, the deterministic high-level Dijkstra could be replaced with belief-MDP planning, and the CEM cost function could incorporate uncertainty penalties (e.g., ensemble variance from the forward model).

## Related Work & Insights
- **vs PcLast**: While the hierarchical framework is similar (k-means for subgoals, CEM for low-level), ALPS replaces the contrastive latent space with the scaled Laplacian and introduces a behavioral prior. Table 1 shows that the $\psi$ space improves performance by over 30 percentage points on average in Maze2D tasks, proving that latent geometry is more influential than the framework architecture.
- **vs HIQL/QRL**: HIQL uses hierarchical model-free learning (high-level predicts subgoal representations, low-level uses IQL), and QRL learns a quasimetric. ALPS excels at stitching trajectories beyond the training distribution—a core advantage of model-based planning—but struggles in environments with non-smooth dynamics like teleportation.
- **vs MuZero/Dreamer**: These model-based RL methods use learned latent one-step predictions for MCTS or imagination rollouts but do not explicitly model commute-time geometry. ALPS emphasizes that the latent space should match the cost semantics of the planner, offering a new perspective on representation design.

## Rating
- Novelty: ⭐⭐⭐⭐ Elegantly integrates classic graph-theoretic tools (CTD + spectral clustering) into deep RL via the ALLO objective and uses a unified latent space for multiple roles.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation on OGBench (locomotion + manipulation, state-based + pixel-based), 8 seeds, Wilcoxon significance testing, direct PcLast comparison, and extensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Section 3 clearly elucidates the relationship between CTD, spectral clustering, and the Laplacian; motivation and methodology are well-separated; Figure 1 visualization is highly informative.
- Value: ⭐⭐⭐⭐ Effectively breaks the model-free dominance on OGBench, providing a concrete roadmap for model-based planning in offline RL. The unified latent space design is highly relevant for other long-range planning tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Learning to Search and Searching to Learn for Generalization in Planning](learning_to_search_and_searching_to_learn_for_generalization_in_planning.md)
- [\[ICML 2026\] Quantifying and Optimizing Simplicity via Polynomial Representations](quantifying_and_optimizing_simplicity_via_polynomial_representations.md)
- [\[ICML 2026\] DR.Q: Debiased Model-based Representations for Sample-efficient Continuous Control](debiased_model-based_representations_for_sample-efficient_continuous_control.md)
- [\[ICLR 2026\] Dual Goal Representations](../../ICLR2026/reinforcement_learning/dual_goal_representations.md)
- [\[ICML 2026\] Multi-Agent Decision-Focused Learning via Value-Aware Sequential Communication](multi-agent_decision-focused_learning_via_value-aware_sequential_communication.md)

</div>

<!-- RELATED:END -->

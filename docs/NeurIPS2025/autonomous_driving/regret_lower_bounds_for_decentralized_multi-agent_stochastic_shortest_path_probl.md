---
title: >-
  [Paper Note] Regret Lower Bounds for Decentralized Multi-Agent Stochastic Shortest Path Problems
description: >-
  [NeurIPS 2025][Autonomous Driving][Multi-Agent Reinforcement Learning] This paper establishes the first $\Omega(\sqrt{K})$ regret lower bound for the Decentralized Multi-Agent Stochastic Shortest Path (Dec-MASSP) problem under linear function approximation. By constructing a family of hard-to-learn instances and employing a symmetry argument to identify the structure of optimal policies, the paper demonstrates that this lower bound matches existing upper bounds in terms of the number of episodes $K$.
tags:
  - NeurIPS 2025
  - Autonomous Driving
  - Multi-Agent Reinforcement Learning
  - Stochastic Shortest Path
  - Regret Lower Bounds
  - Decentralized Learning
  - Linear Function Approximation
date: 2026-05-08
content_hash: 756728e7c18a531a
---

# Regret Lower Bounds for Decentralized Multi-Agent Stochastic Shortest Path Problems

**Conference**: NeurIPS 2025
**arXiv**: [2511.04594](https://arxiv.org/abs/2511.04594)
**Code**: None
**Area**: Autonomous Driving
**Keywords**: Multi-Agent Reinforcement Learning, Stochastic Shortest Path, Regret Lower Bounds, Decentralized Learning, Linear Function Approximation

## TL;DR
This paper establishes the first $\Omega(\sqrt{K})$ regret lower bound for the Decentralized Multi-Agent Stochastic Shortest Path (Dec-MASSP) problem under linear function approximation. By constructing a family of hard-to-learn instances and employing a symmetry argument to identify the structure of optimal policies, the paper demonstrates that this lower bound matches existing upper bounds in terms of the number of episodes $K$.

## Background & Motivation
**State of the Field**: The Stochastic Shortest Path (SSP) problem is a foundational model for goal-oriented decision-making. Single-agent SSP learning has been thoroughly studied in both tabular and linear function approximation settings, with matching upper and lower bounds (tabular: $\Omega(B^*\sqrt{SAK})$; linear: $\Omega(dB^*\sqrt{K})$).

**Limitations of Prior Work**: Multi-agent systems (robot swarms, traffic routing, distributed control) inherently require decentralized coordination, yet the learning theory for Dec-MASSP remains nearly absent. The only prior work (trivedi2023massp) provides an upper bound of $\widetilde{O}(B^{*1.5}d\sqrt{nK/c_{\min}})$ without establishing any lower bound.

**Root Cause**: The absence of regret lower bounds makes it impossible to assess the optimality of existing algorithms or to quantify the intrinsic difficulty of decentralized multi-agent learning.

**Paper Goals**: To answer the core open question: "What are the fundamental limits of decentralized MASSP learning?"

**Starting Point**: Constructing a provably hard family of LM-MASSP (Linearly-Mixed MASSP) instances and deriving regret lower bounds via information-theoretic tools.

**Core Idea**: Through a carefully designed two-node $n$-agent instance and a novel symmetry argument, the paper establishes for the first time that $\Omega(\sqrt{K})$ regret is unavoidable in Dec-MASSP.

## Method

### Overall Architecture
The analysis proceeds in three steps: (1) construct a hard-to-learn family of MASSP instances; (2) analyze the structure of optimal policies and value functions; (3) derive regret lower bounds via information theory.

### Key Designs

1. **LM-MASSP Instance Construction**:

    - Minimal yet expressive network: only two nodes $\mathcal{V}=\{s,g\}$ with $n$ agents
    - Global state space $\mathcal{S}=\{s,g\}^n$ of size $2^n$ (exponential)
    - Per-agent action space $\mathcal{A}_i=\{-1,1\}^{d-1}$, global action space $|\mathcal{A}|=2^{n(d-1)}$
    - Uniform cost structure: $c(\mathbf{s},\mathbf{a})=1$ for all non-goal states, reducing optimal policy to minimizing expected time to reach the goal
    - Parameterized instances: each instance is determined by $(n, \delta, \Delta, \theta)$, with $\theta$ varying over exponentially many candidates

2. **Novel Feature Design**:

    - Linear features $\phi(\mathbf{s}'|\mathbf{s},\mathbf{a}) \in \mathbb{R}^{nd}$ for transition probabilities are designed to satisfy validity conditions
    - Core innovation: the feature construction enables transition probabilities to decompose into a structured form depending on each agent's local information
    - Transition probabilities depend only on the number of agents currently at node $s$ (denoted $r$) and the number remaining at $s$ after the transition ($r'$)
    - This feature design offers independent reference value for the broader MARL community

3. **State Space Partitioning and Symmetry Exploitation**:

    - The exponential state space $\mathcal{S}$ is partitioned as $\mathcal{S}_0, \mathcal{S}_1, \ldots, \mathcal{S}_n$ according to the number of agents at node $s$
    - It is shown that the optimal value function takes the same value for all states of the same type (same $r$): $V^*(\mathbf{s})=V^*_r$
    - Strict monotonicity is established: $0=V^*_0 < V^*_1 < \cdots < V^*_n = B^*$
    - This monotonicity avoids the need for closed-form expressions of the value function

4. **KL Divergence Bound Derivation**:

    - In standard single-agent approaches, the KL divergence contains only two terms and can be handled analytically
    - In the multi-agent setting, the KL divergence contains exponentially many terms, precluding direct analysis
    - By leveraging the non-negativity of KL divergence and instance symmetry, the following upper bound is obtained: $\text{KL}(\mathbb{P}^\pi_\theta \| \mathbb{P}^\pi_{\theta^j}) \leq 3 \cdot 2^{2n} \cdot \frac{\Delta^2}{\delta(d-1)^2} \cdot \mathbb{E}_{\theta,\pi}[N^-]$

### Proof Strategy
- **Theorem 1 (Optimal Policy Structure)**: Via induction on $r$, it is shown that the policy selecting $\mathbf{a}_\theta$ (i.e., $a_{i,j}=\text{sgn}(\theta_{i,j})$) is optimal in all states
- **Theorem 2 (Regret Lower Bound)**: Average regret over all $\theta \in \Theta$ → decompose into truncated count terms → apply Pinsker's inequality + KL divergence upper bound → optimize $\Delta$ to obtain the tightest bound

## Key Experimental Results

### Main Results

This paper is a purely theoretical contribution with no experimental data. The main theoretical results are:

| Result | Bound | Matching Status |
|--------|-------|-----------------|
| Regret lower bound (Ours) | $\Omega\left(\frac{d\sqrt{KB^*/n}}{2^n}\right)$ | Matches upper bound $\widetilde{O}(B^{*1.5}d\sqrt{nK/c_{\min}})$ in $K$ |
| Single-agent case ($n=1$) | $\Omega(dB^*\sqrt{K})$ | Recovers the known lower bound from min2022learning |
| Validity condition | $K > \frac{n(d-1)^2 \cdot \delta}{2^{10} B^* (\frac{1-2\delta}{1+n+n^2})^2}$ | Ensures the chosen parameter $\Delta^*$ is valid |

### Ablation Study (Theoretical Parameter Analysis)

| Dimension | Conclusion |
|-----------|------------|
| Effect of $n$ | The lower bound contains a $2^{-n}$ factor, but this should not be interpreted as "more agents are easier"—the constraint $\Delta < 2^{-n}(\frac{1-2\delta}{1+n+n^2})$ limits the regime of $n$ |
| Number of near-optimal policies | Exponentially many near-optimal policies exist (differing from the optimal in only a few state-action components), making learning extremely difficult |
| Range of $\delta$ | $\delta \in (2/5, 1/2)$, controlling the base transition probability from $s$ to $g$ |
| Dimension $d$ | The lower bound scales linearly with feature dimension $d$ |

### Key Findings
1. **$\Omega(\sqrt{K})$ is unavoidable**: No decentralized learning algorithm—including those with shared parameter estimates—can avoid $\sqrt{K}$-order regret
2. **Single-agent recovery**: At $n=1$, the lower bound reduces to the known single-agent result $\Omega(dB^*\sqrt{K})$
3. **Elegant structure of optimal policies**: Despite the exponential state space, the optimal policy depends only on the sign of parameter $\theta$, and the value function depends only on the number of agents at the non-goal node

## Highlights & Insights
- **First lower bound for Dec-MASSP**: Fills a critical gap in the learning theory of decentralized multi-agent SSP
- **Analysis of exponential state spaces**: First lower bound analysis to handle the exponential state-action space arising in the MASSP setting
- **Generality**: Results are not restricted to any specific communication protocol, and apply across settings ranging from full information sharing to zero communication
- **Independent value of feature design**: The proposed linear feature construction method offers a reusable technique for broader MARL settings
- **Elegance of the symmetry argument**: State space partitioning and monotonicity avoid the need to derive closed-form expressions for the value function

## Limitations & Future Work
1. **Two-node network**: The instance construction is restricted to two nodes; while sufficient for the lower bound, it may not reflect the complexity of practical networks
2. **$2^{-n}$ decay factor**: The exponentially decaying factor weakens the bound for large $n$; tightness in the $n$ dimension remains an open question
3. **Linear function approximation**: Results apply only to the linear setting; lower bounds under nonlinear function approximation remain open
4. **Uniform cost assumption**: The uniform cost structure simplifies the analysis; non-uniform costs are not covered
5. **No empirical validation**: As a purely theoretical work, the nature of lower bound results makes experimental verification inherently difficult
6. **Communication effects**: The impact of different communication levels (e.g., communication graph topology) on regret bounds is not analyzed
7. **Future directions**: Regret bounds under model misspecification, nonlinear function approximation, and the influence of communication protocols

## Related Work & Insights
- **Single-agent SSP**: tarbouriech2020no proposes UC-SSP; rosenberg2020near gives near-optimal bounds; cohen2021minimax achieves minimax optimality; min2022learning provides matching upper and lower bounds in the linear setting—this paper extends this line of work to the multi-agent dimension
- **Dec-MASSP upper bounds**: trivedi2023massp first defines Dec-MASSP and establishes an $\widetilde{O}(B^{*1.5}d\sqrt{nK/c_{\min}})$ upper bound—this paper completes the theoretical picture with a matching lower bound
- **Decentralized MARL**: The linear cost approximation and communication graph formulations of zhang2018fully and trivedi2022multi provide the formal framework adopted in this paper
- **Insights**: The state space partitioning and symmetry argument techniques introduced here can be applied to lower bound analyses in other multi-agent learning settings

## Rating ⭐4
This paper establishes the first regret lower bound for Dec-MASSP, offering a solid theoretical contribution with novel proof techniques (feature design + symmetry argument). The $2^{-n}$ decay factor, however, limits the completeness of the result.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] F3DGS: Federated 3D Gaussian Splatting for Decentralized Multi-Agent World Modeling](../../CVPR2026/autonomous_driving/f3dgs_federated_3d_gaussian_splatting_for_decentralized_multi-agent_world_modeli.md)
- [\[NeurIPS 2025\] BayesG: Bayesian Ego-Graph Inference for Networked Multi-Agent Reinforcement Learning](bayesian_ego-graph_inference_for_networked_multi-agent_reinforcement_learning.md)
- [\[ICCV 2025\] SRefiner: Soft-Braid Attention for Multi-Agent Trajectory Refinement](../../ICCV2025/autonomous_driving/srefiner_soft-braid_attention_for_multi-agent_trajectory_refinement.md)
- [\[NeurIPS 2025\] UrbanIng-V2X: A Large-Scale Multi-Vehicle Multi-Infrastructure Dataset Across Multiple Intersections for Cooperative Perception](urbaning-v2x_a_large-scale_multi-vehicle_multi-infrastructure_dataset_across_mul.md)
- [\[NeurIPS 2025\] V2X-Radar: A Multi-Modal Dataset with 4D Radar for Cooperative Perception](v2x-radar_a_multi-modal_dataset_with_4d_radar_for_cooperative_perception.md)

<!-- RELATED:END -->

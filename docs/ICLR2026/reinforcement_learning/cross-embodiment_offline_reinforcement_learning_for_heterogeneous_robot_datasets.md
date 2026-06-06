---
title: >-
  [Paper Note] Cross-Embodiment Offline Reinforcement Learning for Heterogeneous Robot Datasets
description: >-
  [ICLR 2026][Reinforcement Learning][cross-embodiment learning] This paper systematically investigates cross-embodiment offline RL pretraining…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "cross-embodiment learning"
  - "offline RL"
  - "gradient conflict"
  - "robot foundation model"
  - "morphology grouping"
date: 2026-05-08
content_hash: e5be14d95b45ca49
---

# Cross-Embodiment Offline Reinforcement Learning for Heterogeneous Robot Datasets

**Conference**: ICLR 2026
**arXiv**: [2602.18025](https://arxiv.org/abs/2602.18025)  
**Code**: To be confirmed  
**Area**: Reinforcement Learning / Robotics
**Keywords**: cross-embodiment learning, offline RL, gradient conflict, robot foundation model, morphology grouping

## TL;DR
This paper systematically investigates cross-embodiment offline RL pretraining, identifies gradient conflicts leading to negative transfer under increasing suboptimal data ratios and robot diversity, and proposes Embodiment Grouping (EG)—a strategy that clusters robots by morphological graph distance and updates the actor group-wise. On a locomotion benchmark spanning 16 robot platforms, EG substantially mitigates negative transfer (IQL+EG improves over IQL by 34% on the 70% suboptimal dataset).

## Background & Motivation

**Background**: Robot foundation models (e.g., RT-2, Octo, π0) learn generalizable control priors from multi-embodiment data via cross-embodiment learning. However, these models rely almost exclusively on imitation learning, requiring high-quality expert demonstrations that are costly to collect.

**Limitations of Prior Work**: (a) Imitation learning can only replicate behaviors present in the dataset and cannot surpass the quality ceiling of the data. (b) Offline RL can exploit suboptimal data through trajectory stitching to learn superior policies, yet has not been systematically combined with cross-embodiment learning. (c) Naïvely performing joint training on heterogeneous robot data can cause gradient conflicts across morphologies, leading to performance degradation for certain robots.

**Key Challenge**: Cross-embodiment learning increases data scale—beneficial. However, policy gradients from morphologically dissimilar robots may conflict—harmful. This conflict intensifies when the proportion of suboptimal trajectories in the data is high.

**Goal**: Systematically analyze the benefits and failure modes of cross-embodiment offline RL, and design a method to mitigate gradient conflicts among heterogeneous morphologies.

**Key Insight**: Each robot is represented as a morphological graph (joints/end-effectors as nodes), and the Fused Gromov-Wasserstein (FGW) distance is used to compute inter-robot similarity. The paper finds that morphological similarity is strongly correlated with gradient cosine similarity (Pearson $r = 0.63$), motivating actor updates that are grouped according to morphological clustering.

**Core Idea**: Morphologically similar robots exhibit aligned policy gradient directions; grouping robots by morphological clustering and updating the actor group-wise effectively mitigates gradient conflicts in cross-embodiment offline RL.

## Method

### Overall Architecture
Offline datasets from 16 robots → URMA architecture (unifying heterogeneous state/action spaces) → global critic update (using all data) → grouped actor update (robots clustered by morphology; each group updates the actor using only its own data) → the learned cross-embodiment policy can be efficiently fine-tuned to new robots downstream.

### Key Designs

1. **Systematic Analysis of Cross-Embodiment Offline RL**:

    - Function: Compares BC and IQL across different data quality levels and analyzes positive/negative transfer.
    - Key Findings: (a) On expert data, BC ≈ IQL; on suboptimal data, IQL significantly outperforms BC, consistent with the trajectory stitching advantage of offline RL. (b) Cross-embodiment pretraining accelerates downstream fine-tuning convergence. (c) On the 70% suboptimal dataset, bipedal robots suffer severe negative transfer (Unitree H1: 54.47→6.00; G1: 78.93→0.86).
    - Design Motivation: Reveals that cross-embodiment offline RL is not unconditionally beneficial—gradient conflicts must be explicitly addressed.

2. **Gradient Conflict Analysis**:

    - Function: Quantifies the directional conflict of policy gradients across different morphologies.
    - Mechanism: Computes the actor gradient cosine similarity for each pair of robots $C[\tau_i, \tau_j] = \frac{\langle g_{\tau_i}, g_{\tau_j} \rangle}{\|g_{\tau_i}\| \|g_{\tau_j}\|}$ and records the proportion of pairs with $C < 0$.
    - Key Findings: Higher suboptimal data ratio → higher negative cosine proportion; more robot types → higher negative cosine proportion; transfer gain is strongly correlated with mean gradient cosine ($r = 0.815$).
    - Design Motivation: When robots with large morphological differences have opposing gradient directions, joint updates cancel or corrupt useful gradient information.

3. **Correlation Between Morphological Graph Distance and Gradient Alignment**:

    - Function: Represents robots as graphs, computes FGW distances, and validates their correlation with gradient directions.
    - Mechanism: Nodes represent torso/joints/feet; edges represent mechanical connections; node features encode relative positions and control parameters. FGW distance jointly accounts for graph structure and node features.
    - Key Findings: Pearson correlation between morphological similarity and gradient cosine similarity is $r = 0.63$ ($p = 1.26 \times 10^{-14}$)—robots that are morphologically alike also have aligned gradient directions.

4. **Embodiment Grouping (EG)**:

    - Function: Hierarchically clusters 16 robots into $M$ groups by morphological distance; during training, the critic is updated globally while the actor is updated sequentially per group.
    - Mechanism: Algorithm 1 — (1) Sample a global mini-batch; (2) Update the global critic/value network; (3) Randomly permute group order; (4) For each group $\mathcal{G}_m$, extract samples $\mathcal{B}_m$ belonging to that group, compute the actor loss, and update $\theta_\pi$.
    - Design Motivation: Robots within the same group have consistent gradient directions, so intra-group updates do not conflict; sequential inter-group updates avoid gradient cancellation.
    - Distinction from PCGrad: PCGrad dynamically projects conflicting gradients, incurring high computational cost with limited benefit; EG employs static grouping, which is simpler and more effective.

### Loss & Training
Based on the IQL framework:
- Critic: expectile regression to fit $V_\psi(s)$, followed by TD updates for $Q_\theta(s, a)$.
- Actor: advantage-weighted regression $\mathcal{L}_\tau^\pi(\theta) = -\mathbb{E}_{(s,a) \sim \mathcal{D}_\tau}[w(s,a) \log \pi_\theta(a|s)]$, where $w(s,a) = \exp(\beta(Q(s,a) - V(s)))$.
- EG modifies only the actor update procedure (grouping) without altering the loss functions.

## Key Experimental Results

### Main Results

| Method | Expert Forward | 70% Suboptimal Forward | 70% Suboptimal Backward | Mean |
|--------|---------------|------------------------|--------------------------|------|
| BC | 63.31 | 30.52 | 41.42 | 49.17 |
| IQL | 63.39 | 36.62 | 38.69 | 52.05 |
| IQL+PCGrad | 63.37 | 39.63 | 41.04 | 53.48 |
| IQL+SEL | 63.37 | 44.59 | 44.45 | 55.07 |
| **IQL+EG** | **63.52** | **51.19** | **49.60** | **57.29** |

Under the 70% suboptimal data setting, IQL+EG improves over IQL by 34%, over PCGrad by 16%, and over SEL by 16%.

### Ablation Study

| Grouping Strategy | 70% Suboptimal Forward | Gain over IQL |
|-------------------|------------------------|---------------|
| IQL (baseline) | 37.57 | 0% |
| Random grouping | 38.73 | +3.08% |
| Heuristic (bipeds/quads) | 34.45 | −8.31% |
| **EG (ours)** | **51.98** | **+38.34%** |

### Key Findings
- **EG's advantage is not solely attributable to additional actor update steps**: In compute-normalized experiments (matched optimizer steps and data volume), EG still outperforms IQL by 7.78 points.
- **Intuitive grouping fails**: Heuristic grouping by leg count (biped/quadruped/hexapod) actually degrades performance—coarse morphological categories fail to capture factors that govern gradient directions (actuator placement, link lengths, mass distribution, etc.).
- **Random grouping is nearly ineffective** (+3.08%), confirming that a principled grouping strategy is essential.
- **$M = 2$–$4$ groups suffice**: Additional groups yield marginal performance gains at significantly increased training time.
- **Algorithm-agnostic**: EG is effective across BC, TD3+BC, and IQL.

## Highlights & Insights
- **Morphological distance predicts gradient conflict**: This is a profound finding—the physical structural similarity of robots directly correlates with the alignment of gradient directions during policy learning. This implies that which robot datasets can be safely co-trained is predictable prior to training.
- **Simple grouping outperforms complex conflict resolution**: Static morphological clustering (EG) surpasses PCGrad's dynamic gradient projection and SEL's dynamic task grouping—leveraging domain knowledge (morphological structure) a priori is more reliable than inferring task relationships at runtime.
- **Strong complementarity between cross-embodiment learning and offline RL**: Cross-embodiment learning provides data diversity; offline RL exploits suboptimal data—their combination enables robot foundation models to reduce dependence on large quantities of high-quality expert demonstrations.

## Limitations & Future Work
- Validation is limited to MuJoCo locomotion tasks in simulation; extension to real robots and manipulation tasks remains unexplored.
- Computing FGW graph distances requires pre-defined robot graph structures, which may necessitate manual modeling for robots with unknown morphologies.
- Grouping is static, whereas gradient conflict patterns may evolve dynamically during training—adaptive grouping strategies could yield further improvements.
- Dataset scale is relatively small (1M steps per robot); scalability at larger data regimes has not been verified.
- The critic is still updated globally—if critic learning also suffers from cross-embodiment conflicts, grouped critic updates may offer additional gains.

## Related Work & Insights
- **vs. Open X-Embodiment**: OXE employs cross-embodiment imitation learning; this paper presents the first systematic study of cross-embodiment offline RL.
- **vs. Q-Transformer**: Q-Transformer applies offline RL to large-scale robot data but on a single platform; this paper extends the setting to 16 morphologies.
- **vs. PCGrad**: PCGrad resolves multi-task conflicts via gradient projection but offers limited benefit in the cross-embodiment setting; EG's static morphological clustering is more effective.
- **vs. SEL**: SEL groups tasks by dynamic affinity, incurring additional computation and exhibiting less stability than grouping based on morphological priors.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic study of cross-embodiment offline RL; the correlation between morphological distance and gradient conflict is a novel finding.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 16 robots, 6 dataset configurations, 8 method comparisons, comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from problem identification → root cause analysis → hypothesis validation → solution design is clear and complete.
- Value: ⭐⭐⭐⭐ Opens a new direction for scaling robot foundation model data; the EG strategy is simple and practically applicable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Dual-Robust Cross-Domain Offline Reinforcement Learning Against Dynamics Shifts](dual-robust_cross-domain_offline_reinforcement_learning_against_dynamics_shifts.md)
- [\[ICLR 2026\] Less is More: Clustered Cross-Covariance Control for Offline RL](less_is_more_clustered_cross-covariance_control_for_offline_rl.md)
- [\[ICLR 2026\] Flow Actor-Critic for Offline Reinforcement Learning (FAC)](flow_actor-critic_for_offline_reinforcement_learning.md)
- [\[ICML 2026\] Unified Value Alignment and Assignment in Cross-Domain Offline Reinforcement Learning](../../ICML2026/reinforcement_learning/unifying_value_alignment_and_assignment_in_cross-domain_offline_reinforcement_le.md)
- [\[ICLR 2026\] ReFORM: Reflected Flows for On-support Offline RL via Noise Manipulation](reform_reflected_flows_for_on-support_offline_rl_via_noise_manipulation.md)

</div>

<!-- RELATED:END -->

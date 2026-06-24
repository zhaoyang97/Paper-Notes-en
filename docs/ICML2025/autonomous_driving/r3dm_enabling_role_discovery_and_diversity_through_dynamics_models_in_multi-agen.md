---
title: >-
  [Paper Note] R3DM: Enabling Role Discovery and Diversity Through Dynamics Models in Multi-agent Reinforcement Learning
description: >-
  [ICML 2025][Autonomous Driving][Multi-Agent Reinforcement Learning] Proposes the R3DM framework, which balances role diversity and coordination by maximizing the mutual information between agent roles, historical trajectories, and future expected behaviors, leveraging intrinsic rewards driven by dynamics models. It improves the win rate by up to 20% in SMAC/SMACv2 environments.
tags:
  - "ICML 2025"
  - "Autonomous Driving"
  - "Multi-Agent Reinforcement Learning"
  - "Role Discovery"
  - "Dynamics Models"
  - "Contrastive Learning"
  - "Intrinsic Reward"
date: 2026-05-08
content_hash: 2b7a893c878a6290
---

# R3DM: Enabling Role Discovery and Diversity Through Dynamics Models in Multi-agent Reinforcement Learning

**Conference**: ICML 2025  
**arXiv**: [2505.24265](https://arxiv.org/abs/2505.24265)  
**Code**: [Yes](https://github.com/UTAustin-SwarmLab/R3DM)  
**Area**: Autonomous Driving  
**Keywords**: Multi-Agent Reinforcement Learning, Role Discovery, Dynamics Models, Contrastive Learning, Intrinsic Reward

## TL;DR

Proposes the R3DM framework, which balances role diversity and coordination by maximizing the mutual information between agent roles, historical trajectories, and future expected behaviors, leveraging intrinsic rewards driven by dynamics models. It improves the win rate by up to 20% in SMAC/SMACv2 environments.

## Background & Motivation

Multi-agent reinforcement learning (MARL) has made significant progress in fields such as traffic control, autonomous driving, and collaborative robotics. Existing methods mainly face the following challenges:

**Parameter Sharing vs. Behavioral Diversity**: The CTDE paradigm (e.g., QMIX, MAPPO) improves sample efficiency by sharing policy parameters, but prevents individual agents from learning differentiated behaviors.

**Diversity vs. Coordination**: Diversity-driven methods (e.g., CDS), while encouraging individual differences, often sacrifice team coordination.

**Limitations of Prior Work on Roles**: Role-based methods such as ROMA, RODE, and ACORM derive roles solely from an agent's **past experiences**, ignoring the impact of roles on **future behaviors**.

**Key Challenge Example**: In a firefighting drone scenario, if two drones begin with similar initial observations, history-based role inference would assign them the exact same role, causing both to fly towards the same fire, failing to achieve an effective division of labor.

**Core Idea**: An agent's role should shape its future behavior—agents adopting different roles should naturally exhibit distinct trajectories at any subsequent moment. Therefore, roles must be linked to future expected behaviors through dynamics models.

## Method

### Overall Architecture

R3DM proposes an-information theoretic objective function under the CTDE framework to maximize the mutual information between the agent's role $m_i^t$, observation-action history $\tau_i^t$, and future trajectory $\tau_i^{t+1:t+k}$. Theorem 4.1 decomposes this intractable objective into two optimizable sub-objectives:

$$I(\tau_i^{t+k}; m_i^t) \geq \mathbb{E}_{e_i^t, z_i^t, m_i^t}\left[\log\frac{p(z_i^t \mid e_i^t)}{p(z_i^t)}\right] + I(\tau_i^{t+1:t+k}; z_i^t)$$

- **First Term**: Learns intermediate role embeddings from history $\rightarrow$ Optimized via contrastive learning.
- **Second Term**: Ensures that role embeddings guide future behavioral diversity $\rightarrow$ Optimized via intrinsic rewards.

### Key Designs

1. **Role Embedding via Contrastive Learning (Optimizing the First Term)**: A trajectory encoder $f_{\theta_e}$ encodes the observation-action history into an embedding $e_i^t$, which is then mapped to a role embedding $z_i^t$ through a role encoder $f_{\theta_r}$. K-means clusters agent embeddings into $|M|$ role groups. Embeddings from the same group are treated as positive pairs, and those from different groups as negative pairs. A bilinear scoring function $g(z_i^t, e_i^t)$ calculates similarity. Core formula (Theorem 4.2):

    $\mathbb{E}\left[\log\frac{p(z_i^t | e_i^t)}{p(z_i^t)}\right] \geq \log|M| + \mathbb{E}\left[\log\frac{g(z_i^t, e_i^t)}{g(z_i^t, e_i^t) + \sum_{m_i^{t*}} g(z_i^t, e_i^{t*})}\right]$

   Design Motivation: Reuses the mature contrastive learning framework of ACORM to obtain intermediate role representations, serving as the foundation for subsequent intrinsic rewards.

2. **Policy Intrinsic Reward**: Theorem 4.3 decomposes the mutual information between future trajectories and roles into a policy term and a dynamics term. The policy intrinsic reward measures the influence of the role on action selection:

    $r_{i,\text{pol}}^t = \sum_{l=t}^{t+k-1} \mathbb{D}_{KL}\left(\text{SoftMax}(Q_i(\cdot|\tau_i^l, z_i^t; \phi_Q)) \| p(\cdot|\tau_i^l)\right)$

   where $p(\cdot|\tau_i^l) = \mathbb{E}_{z_i^t}[\text{SoftMax}(Q_i(\cdot|\tau_i^l, z_i^t; \phi_Q))]$ is the average action probability across all roles. This KL divergence encourages different roles to generate differentiated policy distributions.

3. **Dynamics Intrinsic Reward**: Learns two DreamerV3-style RSSM world models—the role-conditional model $q_\psi(o_i^{l+1}|\tau_i^l, z_i^t, a_i^l)$ and the role-agnostic model $p(o_i^{l+1}|\tau_i^l, a_i^l)$. RSSM consists of four components: a sequence model, an observation encoder, a dynamics predictor, and an observation decoder. The dynamics intrinsic reward is the log-likelihood difference between the two models:

    $r_{i,\text{dyn}}^t = \sum_{l=t}^{t+k-1}\left(\beta_1[\log q_{\psi_{\text{dec}}}(\cdot) + \beta_2 \log q_{\psi_{\text{dyn}}}(\cdot)] - [\text{role-agnostic terms}]\right)$

   Design Motivation: When the predictions of the role-conditional model are significantly better than those of the role-agnostic model, it implies that the role embedding indeed possesses predictive power over future trajectories. $\beta_1$ balances cross-role-trajectory diversity and role-trajectory consistency.

### Loss & Training

Total intrinsic reward: $r_{\text{int}}^t = \sum_{i \in I} \beta_3 r_{i,\text{pol}}^t + r_{i,\text{dyn}}^t$

Final TD learning objective:

$$\mathcal{L}_{TD}(\theta) = \left[r^t + \alpha r_{\text{int}}^t + \gamma \max_{a^{t+1}} Q_{\text{tot}}(s^{t+1}, a^{t+1}; \phi^{-}) - Q_{\text{tot}}(s^t, a^t; \phi)\right]^2$$

where $\alpha$ balances the task reward and the intrinsic reward, and $\phi^{-}$ represents the frozen target network parameters. The default imagination horizon is $k=1$, and $\epsilon$-greedy exploration linearly decays from 1.0 to 0.02.

## Key Experimental Results

### Main Results

Evaluated on SMAC (6 hard/super-hard maps) and SMACv2 (6 environments), comparing against QMIX, CDS, EMC, CIA, GoMARL, and ACORM.

| Scenario | Metric | Ours (R3DM) | ACORM (SOTA) | Gain |
|--------|------|------|----------|------|
| 3s5z_vs_3s6z (SMAC) | Test Win Rate | ~55% | ~35% | +20% |
| Corridor (SMAC) | Test Win Rate | ~90% | ~80% | +10% |
| 6h_vs_8z (SMAC) | Test Win Rate | ~30% | ~20% | +10% |
| protoss_10_vs_11 (SMACv2) | Test Win Rate | Best | Second Best | Marginal |
| protoss_5_vs_5 (SMACv2) | Cumulative Reward | Best | Close | More Efficient Policy |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| k=1 (Default) | Best Win Rate | Single-step imagination is sufficient |
| k=10 | Significant performance degradation | Cumulative error of partially observable world model |
| \|M\|=3 | Fastest convergence | Balances coordination and specialization |
| \|M\|=8 | Similar final performance but slower convergence | Over-specialization |
| W/o Contrastive Learning | Lower than full version but superior to ACORM | Intrinsic reward is the core contribution |
| W/o Intrinsic Reward (=ACORM) | Baseline level | Confirms the effectiveness of the dynamics reward |

### Key Findings

1. Intrinsic reward is core: Removing contrastive learning still outperforms ACORM, while removing the intrinsic reward degrades the method to ACORM.
2. Shorter-horizon predictions work better: World models based on partial observations suffer from severe error accumulation during multi-step prediction.
3. Qualitative Analysis (3s_vs_5z): One stalker in R3DM learns an "enemy-luring" role, baiting 3 zealots away while the main force splits into two teams to annihilate the weakened enemy; in contrast, all agents in ACORM charge forward and ultimately lose.
4. In SMACv2, R3DM shows a clear advantage in cumulative rewards, learning a more efficient winning strategy even when win rates are similar.

## Highlights & Insights

- The core insight is simple yet powerful: Roles should shape future behavior, rather than solely being inferred from the past—directly addressing a fundamental limitation of existing methods.
- Introducing the DreamerV3 world model into MARL role learning is a novel cross-domain combination.
- Rigorous information-theoretic derivation: Each step is supported by theorems, from the MI objective to the tractable lower bounds and the specific reward designs.
- Highly convincing qualitative analysis (tactical visualization) that intuitively demonstrates the tactical coordination advantages brought by role differentiation.

## Limitations & Future Work

1. **Role Number Needs to Be Predefined**: $|M|$ is a hyperparameter; future work can explore dynamically deriving it from the replay buffer.
2. **Partially Observable World Model**: Modeling is based only on the ego agent's observations, without considering the actions/roles of other agents.
3. **Only Validated on SMAC-style Environments**: Not yet tested in continuous action spaces or real-world autonomous driving scenarios.
4. **Computational Overhead**: Requires training two RSSM models (role-conditional + role-agnostic), increasing both computational and memory burdens.

## Related Work & Insights

- **vs. ACORM**: R3DM builds on ACORM by adding a dynamics-based intrinsic reward, improving the win rate on 3s5z_vs_3s6z from 35% to 55%.
- **vs. CDS**: CDS overemphasizes individual diversity to the detriment of coordination, whereas R3DM avoids this issue through role-constrained diversity.
- **vs. MAVEN**: MAVEN uses latent variables to promote exploration without learning explicit roles, whereas R3DM is more interpretable.
- **Insights**: The combination of world models and intrinsic rewards can be extended to more MARL scenarios. Future work can leverage a global world model to improve prediction accuracy.

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing world models to MARL role learning is an innovative combination, backed by rigorous MI decomposition.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation on SMAC/SMACv2 including ablation and qualitative analysis, though the environment types are somewhat single.
- Writing Quality: ⭐⭐⭐⭐ Clear theorem derivations and highly intuitive throughout, using the firefighting drone example.
- Value: ⭐⭐⭐⭐ Provides a new paradigm connecting history and future behaviors for MARL role learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] BayesG: Bayesian Ego-Graph Inference for Networked Multi-Agent Reinforcement Learning](../../NeurIPS2025/autonomous_driving/bayesian_ego-graph_inference_for_networked_multi-agent_reinforcement_learning.md)
- [\[ICML 2025\] Hybrid Quantum-Classical Multi-Agent Pathfinding](hybrid_quantum-classical_multi-agent_pathfinding.md)
- [\[CVPR 2026\] RLFTSim: Realistic and Controllable Multi-Agent Traffic Simulation via Reinforcement Learning Fine-Tuning](../../CVPR2026/autonomous_driving/rlftsim_realistic_and_controllable_multi-agent_traffic_simulation_via_reinforcem.md)
- [\[ICML 2025\] GoIRL: Graph-Oriented Inverse Reinforcement Learning for Multimodal Trajectory Prediction](goirl_graph-oriented_inverse_reinforcement_learning_for_multimodal_trajectory_pr.md)
- [\[NeurIPS 2025\] RAW2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving](../../NeurIPS2025/autonomous_driving/raw2drive_reinforcement_learning_with_aligned_world_models_for_end-to-end_autono.md)

</div>

<!-- RELATED:END -->

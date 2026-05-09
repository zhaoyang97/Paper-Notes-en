---
title: >-
  [Paper Note] BayesG: Bayesian Ego-Graph Inference for Networked Multi-Agent Reinforcement Learning
description: >-
  [NeurIPS 2025][Autonomous Driving][Bayesian inference] BayesG enables each agent in networked MARL to learn the dynamic structure of its local communication graph via Bayesian variational inference — sampling edge masks with Gumbel-Softmax and jointly optimizing policy and graph structure under an ELBO objective — achieving 50%+ reward improvement over the best baseline in a 167-agent New York traffic scenario.
tags:
  - NeurIPS 2025
  - Autonomous Driving
  - Bayesian inference
  - ego-graph
  - networked MARL
  - dynamic communication graph
  - decentralization
date: 2026-05-08
content_hash: 295088048ce4d398
---

# BayesG: Bayesian Ego-Graph Inference for Networked Multi-Agent Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2509.16606](https://arxiv.org/abs/2509.16606)
**Code**: [https://github.com/Wei9711/BayesG](https://github.com/Wei9711/BayesG)
**Area**: Autonomous Driving
**Keywords**: Bayesian inference, ego-graph, networked MARL, dynamic communication graph, decentralization

## TL;DR
BayesG enables each agent in networked MARL to learn the dynamic structure of its local communication graph via Bayesian variational inference — sampling edge masks with Gumbel-Softmax and jointly optimizing policy and graph structure under an ELBO objective — achieving 50%+ reward improvement over the best baseline in a 167-agent New York traffic scenario.

## Background & Motivation

**State of the Field**: In networked MARL, agents exchange information through communication graphs. Existing methods rely on fixed communication graphs or require global state to learn dynamic graphs.

**Limitations of Prior Work**: Fixed neighbor sets are suboptimal in dynamic environments, as the informational value of different neighbors varies over time. Centralized graph learning (requiring global observability) is impractical in decentralized systems.

**Root Cause**: Agents possess only local observations yet must determine "which neighbors provide the most useful information" — an inherently uncertainty-laden problem.

**Paper Goals**: Enable each agent to learn task-adaptive local communication graph structures in a fully decentralized manner.

**Starting Point**: Model edge existence/absence as Bernoulli random variables and estimate the posterior from local data via variational Bayesian inference.

**Core Idea**: Each agent performs Bayesian variational inference over the edges of its ego-graph (Bernoulli + Gumbel-Softmax); an ELBO objective jointly optimizes policy and graph structure, enabling decentralized dynamic communication.

## Method

### Overall Architecture
Agent $i$'s policy is conditioned on a sampled subgraph: $\pi_i(u_i, G_{\mathcal{V}_i} | s_{\mathcal{V}_i}) = \rho(G | s) \cdot \tilde{\pi}_i(u_i | \tilde{f}_i(s, G))$. The edge mask $Z_i$ is sampled from the variational distribution $q(Z_i; \phi_i) = \prod \text{Bern}(z_{ij}; \sigma(\phi_{ij}))$ and made differentiable via Gumbel-Softmax.

### Key Designs

1. **Bayesian Edge Inference**: The variational approximation $q(Z_{ij})$ is Bernoulli, and the prior $p(Z_{ij})$ incorporates a retention bias $\lambda$. ELBO: $\mathcal{L} = E_q[-\mathcal{L}_{\theta,\varphi}] - \sum_{j} \text{KL}(q \| p)$
2. **GNC Message Passing**: Graph neural communication is performed over the masked adjacency matrix $A_i^* = Z_i \odot A_i$
3. **Multi-Feature Input**: Three categories of information — neighbor states, trajectories, and policy features

### Loss & Training
- Actor-Critic jointly optimized with ELBO
- KL regularization encourages sparse graphs (retaining only informative edges)

## Key Experimental Results

### Main Results (Adaptive Traffic Signal Control, ATSC)

| Environment | BayesG | NeurComm | CommNet | Gain |
|---|---|---|---|---|
| Grid 5×5 | ~-15 | ~-20 | ~-30 | +25% |
| NewYork 167 agents | **~-30** | ~-45 | ~-60 | **+50%** |

### Ablation Study

| Configuration | Performance |
|---|---|
| No mask | Baseline performance |
| Random mask | Severe degradation |
| Learned mask | **Optimal** |
| Trajectory + State + Policy | Best feature combination |

### Key Findings
- Learned graph structure significantly outperforms fixed graphs — especially in large-scale scenarios (167 agents)
- Random masking is harmful, demonstrating the necessity of structure learning
- Faster convergence (substantial lead observed in early training stages)

## Highlights & Insights
- **Bayesian treatment of uncertainty is natural**: When it is unclear which neighbors are informative, probabilistic sampling is more robust than hard selection
- **KL regularization induces sparsity automatically**: No manual communication budget specification is required

## Limitations & Future Work
- The temporal evolution of learned graph structures is not analyzed
- Evaluation is limited to 167 agents
- Fixed communication intervals are assumed

## Related Work & Insights
- **vs. CommNet**: Uses fixed fully-connected graphs; BayesG learns sparse dynamic graphs
- **vs. NeurComm**: Employs centralized graph learning; BayesG is fully decentralized

## Rating
- Novelty: ⭐⭐⭐⭐ Natural integration of Bayesian graph inference with MARL
- Experimental Thoroughness: ⭐⭐⭐⭐ 5 environments + ablation study
- Writing Quality: ⭐⭐⭐⭐ Method is clearly presented
- Value: ⭐⭐⭐⭐ Practical solution for distributed multi-agent systems
- Interaction structure should be dynamic rather than predefined — Bayesian inference enables agents to adaptively select interaction partners
- Outperforms fully-connected and fixed-graph methods in 167-agent traffic control; the learned sparse graph is more efficient
- The core contribution lies in the simplicity and effectiveness of the design
- Experimental results thoroughly validate the central hypothesis

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Causality Meets Locality: Provably Generalizable and Scalable Policy Learning for Networked Systems](causality_meets_locality_provably_generalizable_and_scalable_policy_learning_for.md)
- [\[NeurIPS 2025\] Self-Supervised Learning of Graph Representations for Network Intrusion Detection](self-supervised_learning_of_graph_representations_for_network_intrusion_detectio.md)
- [\[NeurIPS 2025\] Regret Lower Bounds for Decentralized Multi-Agent Stochastic Shortest Path Problems](regret_lower_bounds_for_decentralized_multi-agent_stochastic_shortest_path_probl.md)
- [\[NeurIPS 2025\] RAW2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving](raw2drive_reinforcement_learning_with_aligned_world_models_for_end-to-end_autono.md)
- [\[ICLR 2026\] SMART-R1: Advancing Multi-agent Traffic Simulation via R1-Style Reinforcement Fine-Tuning](../../ICLR2026/autonomous_driving/advancing_multi-agent_traffic_simulation_via_r1-style_reinforcement_fine-tuning.md)

<!-- RELATED:END -->

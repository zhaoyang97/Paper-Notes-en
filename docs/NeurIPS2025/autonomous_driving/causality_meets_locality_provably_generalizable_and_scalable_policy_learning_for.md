---
title: >-
  [Paper Note] Causality Meets Locality: Provably Generalizable and Scalable Policy Learning for Networked Systems
description: >-
  [NeurIPS 2025][Autonomous Driving][Networked MARL] This paper proposes the GSAC framework, which integrates causal representation learning with meta Actor-Critic. By learning sparse causal masks from networked MARL to co…
tags:
  - "NeurIPS 2025"
  - "Autonomous Driving"
  - "Networked MARL"
  - "Causal Mask"
  - "Approximate Compact Representation ACR"
  - "Domain Generalization"
  - "Meta Actor-Critic"
date: 2026-05-08
content_hash: 3bf4e9340270bf64
---

# Causality Meets Locality: Provably Generalizable and Scalable Policy Learning for Networked Systems

**Conference**: NeurIPS 2025
**arXiv**: [2510.21427](https://arxiv.org/abs/2510.21427)
**Code**: To be confirmed
**Area**: Autonomous Driving
**Keywords**: Networked MARL, Causal Mask, Approximate Compact Representation ACR, Domain Generalization, Meta Actor-Critic

## TL;DR
This paper proposes the GSAC framework, which integrates causal representation learning with meta Actor-Critic. By learning sparse causal masks from networked MARL to construct Approximate Compact Representations (ACR), GSAC achieves scalability; by conditioning policies on domain factors, it achieves cross-domain generalization. Finite-sample guarantees are provided for causal recovery, convergence, and adaptation gap.

## Background & Motivation

**Background**: Large-scale networked systems (traffic networks, power grids, wireless communication) face two fundamental challenges in MARL: scalability (joint state-action space grows exponentially with the number of agents) and generalizability (training and deployment environments differ). Existing networked MARL work (Qu 2022, etc.) exploits local interactions for scalability but assumes fixed environments.

**Limitations of Prior Work**:
   - Scalability: Even with $\kappa$-hop truncation, input dimensionality remains high when node degree or $\kappa$ is large.
   - Generalizability: While single-agent domain generalization RL has been studied, simultaneously achieving scalability and generalizability in multi-agent networked systems remains an open problem.
   - No prior work provides sample complexity guarantees for structural identifiability in networked MARL.

**Key Challenge**: Networked systems must simultaneously address scale ($n$ agents, exponential state space) and generalization (environment parameters $\omega$ shift between training and testing) — no existing framework provides provable guarantees for both.

**Goal**: Design the first networked MARL algorithm that is provably both scalable and generalizable.

**Key Insight**: Causal structure is invariant across domains; only the domain factor $\omega$ varies. By identifying the minimal set of variables that each agent's state transition depends on via causal masks, compact representations are constructed that simultaneously reduce dimensionality (scalability) and isolate domain factors (generalizability).

**Core Idea**: Use causal masks to identify minimal neighborhood dependencies, construct Approximate Compact Representations (ACR), and build upon this foundation a meta Actor-Critic that trains policies across domains and rapidly adapts to new environments.

## Method

### Overall Architecture
GSAC follows a four-stage pipeline: (1) causal discovery + domain factor estimation → (2) ACR construction → (3) meta-learning Actor-Critic training across source domains → (4) rapid adaptation in the target domain.

### Key Designs

1. **Approximate Compact Representation (ACR)**:

    - Function: Identifies the minimal subset of variables within the $\kappa$-hop neighborhood state that genuinely influences the value function.
    - Mechanism: Employs a causal mask $\mathbf{c}$ via recursive traceback — starting from agent $i$'s reward $r_i$, identifies state variables that directly affect $r_i$; then traces one step back to find variables influencing the next-step states; recursing $\kappa$ steps yields $\mathbf{s}_{\mathcal{N}_i^\kappa}^\circ \subset \mathbf{s}_{\mathcal{N}_i^\kappa}$.
    - Approximation error: $|\tilde{Q}_i^{\tilde{\pi}} - Q_i^\pi| \leq \frac{3\bar{r}}{1-\gamma}\gamma^{\kappa+1}$, i.e., the error still decays exponentially in $\kappa$.
    - Design Motivation: Standard truncation reduces global state to the $\kappa$-hop neighborhood; ACR further exploits causal sparsity to reduce dimensionality within the neighborhood, yielding $|\mathbf{s}^\circ| \ll |\mathbf{s}_{\mathcal{N}_i^\kappa}|$.

2. **Domain Factor ACR**:

    - Function: Constructs a compact representation $\omega^\circ$ of the domain factor $\omega$.
    - Mechanism: Analogous to state ACR, traces dependency paths from $\omega$ to rewards via the causal mask.
    - Key Corollary: Domain generalization requires estimating only the compact domain factor $\omega^\circ$ rather than the full $\omega$.

3. **Meta Actor-Critic Learning**:

    - Function: Trains a shared policy $\pi_i^{\theta_i}(\cdot | \mathbf{s}_{\mathcal{N}_i}^\circ, \omega_{\mathcal{N}_i}^\circ)$ across $M$ source domains.
    - Critic update: Performs TD learning on each source domain to estimate $\hat{Q}_i$ over the ACR input space.
    - Actor update: Aggregates Q-values from all agents within the $\kappa$-hop neighborhood and updates parameters via policy gradient.

4. **Rapid Adaptation (Phase 4)**:

    - Function: Collects a small number of trajectories in a new domain, estimates the domain factor $\hat{\omega}^{M+1}$, and directly deploys the meta-policy.
    - Key Theorem (Thm 4): The adaptation gap decays at $O(1/\sqrt{T_a})$.

### Theoretical Guarantees

| Theorem | Content | Rate |
|---------|---------|------|
| Thm 1 | Structural identifiability of causal mask | — |
| Prop 4 | Sample complexity of causal recovery | $O(d \cdot d_{\max} \log(dn/\delta) / \lambda^2)$ |
| Prop 5 | Domain factor estimation error | $O(\sqrt{D_\Omega \log(nT_e/\delta)/T_e})$ |
| Thm 2 | Critic error bound | $O(1/\sqrt{T} + \rho^{\kappa+1} + 1/\sqrt{T_e})$ |
| Thm 3 | Policy gradient convergence | $O(1/\sqrt{K} + \rho^{\kappa+1} + 1/\sqrt{T_e} + 1/\sqrt{M})$ |
| Thm 4 | Adaptation gap | $O(1/\sqrt{T_a})$ |

## Key Experimental Results

### Main Results: Wireless Communication Network

| Method | Grid 3×3 | Grid 4×4 | Grid 5×5 |
|--------|----------|----------|----------|
| GSAC (ours) | Highest return + fastest adaptation | Highest | Highest |
| SAC-MTL | Moderate, slow adaptation | Moderate | Moderate |
| SAC-FT | Poor initially, requires fine-tuning | Poor | Poor |
| SAC-LFS | Slowest convergence | Worst | Worst |

### Key Findings
- **GSAC adapts rapidly within 1–30 episodes**: Only a small number of target-domain trajectories are needed for deployment, far outpacing fine-tuning and training from scratch.
- **Scalability**: From 16 to 36 agents, GSAC maintains consistently high performance.
- **ACR substantially reduces dimensionality**: The effective input dimensionality is far smaller than the full dimensionality of the $\kappa$-hop neighborhood.
- **Efficient domain factor estimation**: As few as $T_e = 20$ trajectories suffice for accurate domain factor estimation.

## Highlights & Insights
- **First provably scalable and generalizable networked MARL algorithm**: Fills a significant theoretical gap in the field.
- **Dual value of ACR**: The same causal mask identification mechanism simultaneously serves scalability (dimensionality reduction) and generalizability (domain factor isolation), yielding a remarkably unified and elegant design.
- **Complete theoretical chain**: From causal identification → ACR approximation error → Critic convergence → Actor convergence → adaptation gap, every step is supported by finite-sample guarantees.
- **Causal structure as a cross-domain invariant**: This insight elevates causal representation learning beyond an interpretability tool to a core mechanism for generalization.

## Limitations & Future Work
- **Restricted to tabular, fully observable settings**: Current experiments and theory assume discrete finite state spaces and full observability.
- **Strength of causal discovery assumptions**: The faithfulness assumption and minimum mutual information assumption may not fully hold in practical systems.
- **Moderate experimental scale**: Validation is limited to at most 36 agents; verification at larger scales (hundreds of agents) is absent.
- **Independent domain factor estimation required**: The framework assumes domain factors are exogenously given or estimable from limited data; online updates may be needed for continuously shifting non-stationary environments.

## Related Work & Insights
- **vs. Qu et al. (2022)**: That work establishes the theoretical foundation for scalability in networked MARL but does not address domain generalization; this paper extends it by incorporating causal ACR and domain conditioning.
- **vs. single-agent domain generalization RL**: Single-agent approaches (Bisimulation, CaReL, etc.) do not account for the exponential complexity introduced by multi-agent network structures.
- **vs. causal RL**: Existing causal RL work focuses on eliminating redundant dependencies or goal conditioning, but does not address scalability in large-scale networked systems.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to unify causal representation learning and domain generalization in networked MARL, with significant theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐ Theory is rigorous, but experiments are limited to two tabular benchmarks; empirical validation for practical applications is insufficient.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] BayesG: Bayesian Ego-Graph Inference for Networked Multi-Agent Reinforcement Learning](bayesian_ego-graph_inference_for_networked_multi-agent_reinforcement_learning.md)
- [\[NeurIPS 2025\] DriveDPO: Policy Learning via Safety DPO For End-to-End Autonomous Driving](drivedpo_policy_learning_via_safety_dpo_for_end-to-end_autonomous_driving.md)
- [\[ICCV 2025\] Wavelet Policy: Lifting Scheme for Policy Learning in Long-Horizon Tasks](../../ICCV2025/autonomous_driving/wavelet_policy_lifting_scheme_for_policy_learning_in_long-horizon_tasks.md)
- [\[ICCV 2025\] DriveX: Omni Scene Modeling for Learning Generalizable World Knowledge in Autonomous Driving](../../ICCV2025/autonomous_driving/drivex_omni_scene_modeling_for_learning_generalizable_world_knowledge_in_autonom.md)
- [\[NeurIPS 2025\] Model-Based Policy Adaptation for Closed-Loop End-to-End Autonomous Driving](model-based_policy_adaptation_for_closed-loop_end-to-end_autonomous_driving.md)

</div>

<!-- RELATED:END -->

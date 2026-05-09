---
title: >-
  [Paper Note] Latent Wasserstein Adversarial Imitation Learning
description: >-
  [ICLR 2026][Reinforcement Learning][Wasserstein distance] LWAIL leverages ICVF to learn a dynamics-aware latent representation from a small amount of random data, replacing the Euclidean ground metric in Wasserstein-based imitation learning with a latent-space distance. The method achieves expert-level imitation performance using only a single state-only expert trajectory.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - Wasserstein distance
  - ICVF
  - dynamics-aware embedding
  - state-only imitation
  - few-shot
date: 2026-05-08
content_hash: 9543bd4920fc32d3
---

# Latent Wasserstein Adversarial Imitation Learning

**Conference**: ICLR 2026
**arXiv**: [2603.05440](https://arxiv.org/abs/2603.05440)
**Code**: [GitHub](https://github.com/JackyYang258/LWAIL)
**Area**: Imitation Learning / Reinforcement Learning
**Keywords**: Wasserstein distance, ICVF, dynamics-aware embedding, state-only imitation, few-shot

## TL;DR
LWAIL leverages ICVF to learn a dynamics-aware latent representation from a small amount of random data, replacing the Euclidean ground metric in Wasserstein-based imitation learning with a latent-space distance. The method achieves expert-level imitation performance using only a single state-only expert trajectory.

## Background & Motivation

**Background**: Imitation learning aims to learn a policy from expert demonstrations. Adversarial imitation learning (AIL) achieves this by matching the state distributions of the agent and the expert, with $f$-divergences and the Wasserstein distance being the two predominant distribution metrics.

**Limitations of Prior Work**: (1) $f$-divergence-based methods require overlapping distribution supports, which is difficult to satisfy when low-quality, non-expert data is used. (2) Wasserstein methods based on the Kantorovich–Rubinstein (KR) dual are more robust, but the 1-Lipschitz constraint implicitly assumes Euclidean distance as the ground metric—an assumption that fails to capture environment dynamics. For instance, although state $B$ may be closer to expert state $C$ in Euclidean space, if $B$ cannot reach $C$, it is less valuable than a more distant state $A$ that can.

**Key Challenge**: The Wasserstein distance requires a meaningful ground metric over states, yet Euclidean distance ignores transition dynamics—states that are geometrically close may be entirely unreachable from one another in the environment.

**Key Insight**: ICVF (Intent-Conditioned Value Function) is used to learn a dynamics-aware embedding $\phi(s)$ from a small amount of random state data, such that Euclidean distances in the embedding space naturally reflect reachability relationships.

**Core Idea**: Replace the Euclidean space in Wasserstein AIL with a dynamics-aware latent space learned via ICVF.

## Method

### Overall Architecture
Two-stage pipeline: (1) **Pre-training** — train ICVF on 1% of randomly collected state data to obtain $\phi(s)$; (2) **Online imitation** — perform Wasserstein adversarial imitation learning in the $\phi$ space.

### Key Designs

1. **ICVF Pre-training**:

   - **Function**: Learn a state embedding $\phi_\theta(s)$ from random state transition data.
   - **Mechanism**: $V_\theta(s, s_+, z) = \phi_\theta(s)^T T_\theta(z) \psi_\theta(s_+)$, trained with IQL offline RL. The embedding $\phi(s)$ encodes the reachability structure of the state space.
   - **Design Motivation**: Theorem 3.1 proves that the state-pair occupancy measure $d_{ss}^{\pi_z}(s,s')$ is approximately a linear combination of $\phi(s)$, implying that the $\phi$ space is naturally suited for Wasserstein-based state distribution matching.

2. **Latent-Space Wasserstein AIL**:

   - **Function**: Match the state-pair distributions of the agent and the expert in $\phi$ space.
   - **Mechanism**: $\min_\pi \max_{\|f\|_L \leq 1} \left(\mathbb{E}_{d_{ss}^\pi}[f(\phi(s), \phi(s'))] - \mathbb{E}_{d_{ss}^E}[f(\phi(s), \phi(s'))]\right)$
   - **Design Motivation**: The 1-Lipschitz constraint in $\phi$ space corresponds to a Euclidean distance that is now dynamics-aware.

3. **Reward Design**:

   - **Function**: Construct an RL reward from the discriminator output: $r(s,s') = \sigma(-f(\phi(s), \phi(s')))$.
   - **Design Motivation**: Sigmoid normalization to $[0,1]$ stabilizes downstream TD3 training.

## Key Experimental Results

### Main Results
MuJoCo environments, single state-only expert trajectory (no actions):

| Environment | LWAIL | WDAIL | GAIfO | IQlearn | Expert Score |
|-------------|-------|-------|-------|---------|--------------|
| Hopper | ~Expert | Low | Medium | Medium | 113.23 |
| HalfCheetah | ~Expert | Low | Low | Medium | 88.42 |
| Walker2D | ~Expert | Low | Medium | Low | 106.84 |
| Ant | ~Expert | Low | Low | Low | 116.97 |

### Ablation Study

| Configuration | Performance | Notes |
|---------------|-------------|-------|
| LWAIL (full) | Best | ICVF embedding + Wasserstein |
| w/o ICVF (Euclidean distance) | Significant drop | Validates the importance of the embedding |
| Varying random data volume | 1% is sufficient | Extremely high data efficiency |

### Key Findings
- ICVF requires only 1% of the online interaction data (as random transitions) to learn an effective embedding.
- t-SNE visualizations clearly show that states in the ICVF embedding space are organized according to dynamic relationships (high-reward states cluster together), a property absent in the raw state space.
- On Maze2D, LWAIL even surpasses TD3 trained with true sparse rewards, as ICVF provides a denser reward signal.

## Highlights & Insights
- **Importance of the ground metric**: The paper identifies a widely overlooked issue in KR-dual Wasserstein methods—the 1-Lipschitz constraint locks the metric to Euclidean distance. This insight is broadly relevant to the Wasserstein IL community.
- **Surprising value of random data**: State transitions collected by a random policy, constituting only 1% of online data, suffice to learn a dynamics-aware embedding of high quality. This demonstrates that "low-quality" data can be highly valuable when properly utilized.
- **Elegant theory–practice integration**: Theorem 3.1 provides a theoretical justification for performing Wasserstein matching in the $\phi$ space.

## Limitations & Future Work
- Whether the multiplicative factorization $V = \phi^T T \psi$ in ICVF limits representational capacity remains an open question.
- Experiments are conducted only on continuous-control MuJoCo tasks; high-dimensional observation settings (e.g., images) are unexplored.
- The single expert trajectory setting is extreme; performance under 5–10 trajectories is not reported.
- The method requires environment interaction to collect random data; its effectiveness in purely offline settings has yet to be validated.

## Related Work & Insights
- **vs. WDAIL / IQlearn**: Both share the KR-dual framework but overlook the ground metric issue; LWAIL directly addresses this via ICVF.
- **vs. SMODICE**: $f$-divergence-based methods require distribution coverage assumptions; LWAIL's Wasserstein formulation is more robust.
- **vs. primal Wasserstein methods**: The primal formulation circumvents the metric issue but introduces other complications.

## Rating
- Novelty: ⭐⭐⭐⭐ — The dynamics-aware ground metric idea is concise and insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multiple environments, multiple baselines, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ — Problem motivation is clearly articulated; theory and experiments are well integrated.
- Value: ⭐⭐⭐⭐ — Offers direct practical improvements to Wasserstein IL methods.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] On Discovering Algorithms for Adversarial Imitation Learning](on_discovering_algorithms_for_adversarial_imitation_learning.md)
- [\[ICLR 2026\] Model Predictive Adversarial Imitation Learning for Planning from Observation](model_predictive_adversarial_imitation_learning_for_planning_from_observation.md)
- [\[ICLR 2026\] Near-Optimal Second-Order Guarantees for Model-Based Adversarial Imitation Learning](near-optimal_second-order_guarantees_for_model-based_adversarial_imitation_learn.md)
- [\[ICLR 2026\] Learning to Generate Unit Test via Adversarial Reinforcement Learning](learning_to_generate_unit_test_via_adversarial_reinforcement_learning.md)
- [\[CVPR 2026\] Lifelong Imitation Learning with Multimodal Latent Replay and Incremental Adjustment](../../CVPR2026/reinforcement_learning/lifelong_imitation_learning_multimodal_latent_rep.md)

<!-- RELATED:END -->

---
title: >-
  [Paper Note] SUSD: Structured Unsupervised Skill Discovery through State Factorization
description: >-
  [ICLR 2026][Reinforcement Learning][unsupervised skill discovery] This paper proposes SUSD (Structured Unsupervised Skill Discovery), which factorizes the state space into independent factors and assigns dedicated skill variables to each factor. Combined with a curiosity-driven factor-weighting mechanism, SUSD discovers diverse skills that cover all controllable factors in complex multi-object and multi-agent environments.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - unsupervised skill discovery
  - state factorization
  - distance maximization
  - curiosity-driven
  - hierarchical reinforcement learning
date: 2026-05-08
content_hash: ba85b08544e2a196
---

# SUSD: Structured Unsupervised Skill Discovery through State Factorization

**Conference**: ICLR 2026
**arXiv**: [2602.01619](https://arxiv.org/abs/2602.01619)
**Code**: [https://github.com/hadi-hosseini/SUSD](https://github.com/hadi-hosseini/SUSD)
**Area**: Unsupervised Skill Discovery / Reinforcement Learning
**Keywords**: unsupervised skill discovery, state factorization, distance maximization, curiosity-driven, hierarchical reinforcement learning

## TL;DR

This paper proposes SUSD (Structured Unsupervised Skill Discovery), which factorizes the state space into independent factors and assigns dedicated skill variables to each factor. Combined with a curiosity-driven factor-weighting mechanism, SUSD discovers diverse skills that cover all controllable factors in complex multi-object and multi-agent environments.

## Background & Motivation

- **Goal of Unsupervised Skill Discovery (USD)**: autonomously learn diverse skills without external rewards for use in downstream tasks.
- **Two main technical paradigms**:
    - **Mutual Information (MI) methods** (e.g., DIAYN): maximize mutual information between skill variables and states, but tend to learn simple, static behaviors due to transformation invariance.
    - **Distance-maximizing Skill Discovery (DSD) methods** (e.g., CSD, METRA): encourage dynamic behaviors by maximizing distances in state space, but focus only on the most easily controllable factor in complex multi-object environments.
- **Key limitation of DSD**: lack of mechanisms to ensure skill diversity covers all controllable factors. These methods perform well in simple environments (Ant, HalfCheetah) but degrade in multi-object settings (multi-agent, Kitchen).
- **Core solution**: exploit the compositional structure of environments as an inductive bias by factorizing the state space and learning dedicated skills for each factor.

## Method

### Overall Architecture

SUSD builds on the DSD framework and consists of three core components: factorized embeddings, curiosity-driven factor weighting, and dual gradient descent training.

### 1. State Space Factorization

The state space is decomposed into $N$ factors: $\mathcal{S} := \mathcal{S}^1 \times \cdots \times \mathcal{S}^N$, and the skill space is decomposed accordingly as $\mathcal{Z} := \mathcal{Z}^1 \times \cdots \times \mathcal{Z}^N$.

The mapping function $\phi$ is split into $N$ independent networks $\phi_i(s^i)$, each processing only its corresponding factor. The optimization objective becomes:

$$\sup_{\pi, \{\phi_i\}_{i=1}^N} \mathbb{E}_{p(\tau, z)} \sum_{i=1}^N \sum_{t=0}^{T-1} (\phi_i(s_{t+1}^i) - \phi_i(s_t^i))^\top z^i$$

$$\text{s.t.} \sum_{i=1}^N \|\phi_i(s'^i) - \phi_i(s^i)\|_2 \leq 1, \quad \forall (s, s') \in \mathcal{S}_{\text{adj}}$$

### 2. Curiosity-Driven Factor Weighting

A density model $q_\theta(s'|s) = \mathcal{N}(\mu_\theta(s), \Sigma_\theta(s))$ is trained, and its factor-wise marginals are used to estimate a "curiosity" weight for each factor:

$$-\log q_\theta(s_{t+1}^i | s_t) \propto (s_{t+1}^i - \mu_\theta^i(s_t))^\top \Sigma_\theta^i(s_t)^{-1} (s_{t+1}^i - \mu_\theta^i(s_t))$$

A high curiosity value corresponds to a low-probability transition, indicating that the factor warrants more attention. $\sqrt{-\log q_\theta(s_{t+1}^i|s_t)}$ serves as a valid distance metric and is incorporated into the objective.

### 3. Final Objective and Intrinsic Reward

The skill reward for each factor:

$$r_i^{\text{SUSD}} := (\phi_i(s_{t+1}^i) - \phi_i(s_t^i))^\top z^i$$

Total intrinsic reward (weighted sum):

$$R := \sum_{i=1}^N \sqrt{-\log q_\theta(s_{t+1}^i | s_t)} \cdot r_i^{\text{SUSD}}$$

The mapping functions and Lagrange multipliers are updated via dual gradient descent, and the policy is trained with SAC.

## Key Experimental Results

### Downstream Task Performance (Multi-Particle and Kitchen Environments)

| Method | MP Avg. Return | Kitchen Avg. Return |
|--------|---------------|---------------------|
| DIAYN | Low | Low |
| LSD | Low | Low |
| CSD | Medium | Low |
| METRA | Medium | Low–Medium |
| DUSDi | Medium | Low |
| **SUSD** | **High** | **High** |

SUSD significantly outperforms all baselines in complex factorized environments, with particularly large margins in the Kitchen environment.

### Incidental Task Completion During Skill Learning

| Task | SUSD | CSD | METRA | LSD | DUSDi |
|------|------|-----|-------|-----|-------|
| BiP (Butter in Pan) | **39.9±18.5** | 0.0 | 0.0 | 0.0 | 0.0 |
| MiP (Meatball in Pan) | **58.9±25.8** | 0.0 | 0.0 | 0.0 | 2.5 |
| PoS (Pan on Stove) | **20.5±18.0** | 0.0 | 0.0 | 0.0 | 1.3 |

SUSD incidentally completes downstream tasks during the skill learning phase, while all other methods fail entirely.

### Factor Decoding Error

| Method | Multi-Particle | Kitchen | 2D-Gunner |
|--------|---------------|---------|-----------|
| **SUSD** | **0.060** | **0.014** | **0.080** |
| METRA | 0.147 | 0.028 | 0.186 |
| CSD | 0.313 | 0.049 | 0.404 |
| LSD | 0.308 | 0.038 | 0.224 |

SUSD's latent skill embeddings contain the richest factor information.

### Key Findings

- SUSD achieves substantially better state coverage, especially for the worst-performing agent.
- SUSD remains competitive in non-factorized environments (Ant, HalfCheetah).
- The curiosity-weighting mechanism effectively redirects attention to under-explored factors.

## Highlights & Insights

1. **First factorized DSD approach**: introduces state factorization as an inductive bias into the DSD framework for the first time.
2. **Fine-grained curiosity weighting**: unlike CSD's coarse weighting (one weight per full state transition), SUSD computes independent weights for each factor.
3. **Composable skills**: the factorized skill representation naturally supports skill composition and chaining.
4. **Theoretical grounding**: Lemma 4.1 rigorously proves that the distance term can serve as a coefficient for the intrinsic reward.

## Limitations & Future Work

- Requires prior knowledge of the state factorization structure (i.e., which dimensions belong to which factor).
- Pixel-based settings require additional disentangled representation learning.
- The skill space dimensionality grows linearly with the number of factors.
- Advantages diminish in non-factorized environments.

## Related Work & Insights

- **MI-based USD**: DIAYN, DADS, DUSDi (factorized MI).
- **DSD-based USD**: LSD, CSD, METRA.
- **State factorization in RL**: FMDP, causal decomposition, DUSDi.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of factorized DSD and curiosity-driven factor weighting is both novel and effective.
- **Technical Depth**: ⭐⭐⭐⭐ — Solid theoretical derivation (Lemma 4.1) with a complete optimization framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-environment evaluation with extensive ablation and qualitative analysis.
- **Value**: ⭐⭐⭐ — Relies on the state factorization assumption, limiting applicability, but delivers strong performance within its scope.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Periodic Skill Discovery](../../NeurIPS2025/reinforcement_learning/periodic_skill_discovery.md)
- [\[ICLR 2026\] Self-Improving Skill Learning for Robust Skill-based Meta-Reinforcement Learning](self-improving_skill_learning_for_robust_skill-based_meta-reinforcement_learning.md)
- [\[ICLR 2026\] How Far Can Unsupervised RLVR Scale LLM Training?](how_far_can_unsupervised_rlvr_scale_llm_training.md)
- [\[ICLR 2026\] AutoQD: Automatic Discovery of Diverse Behaviors with Quality-Diversity Optimization](autoqd_diverse_behaviors.md)
- [\[ICLR 2026\] unsupervised learning of efficient exploration pre-training adaptive policies vi](unsupervised_learning_of_efficient_exploration_pre-training_adaptive_policies_vi.md)

</div>

<!-- RELATED:END -->

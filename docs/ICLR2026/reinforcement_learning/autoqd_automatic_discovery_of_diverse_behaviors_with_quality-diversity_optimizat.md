---
title: >-
  [Paper Note] AutoQD: Automatic Discovery of Diverse Behaviors with Quality-Diversity Optimization
description: >-
  [ICLR 2026][Reinforcement Learning][quality-diversity] AutoQD is proposed to embed policy occupancy measures into a finite-dimensional space via random Fourier features (RFF)…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "quality-diversity"
  - "occupancy measure"
  - "random Fourier features"
  - "behavior descriptor"
  - "CMA-MAE"
date: 2026-05-08
content_hash: cf0ad1392256c32c
---

# AutoQD: Automatic Discovery of Diverse Behaviors with Quality-Diversity Optimization

**Conference**: ICLR 2026
**arXiv**: [2506.05634](https://arxiv.org/abs/2506.05634)  
**Code**: [conflictednerd/autoqd-code](https://github.com/conflictednerd/autoqd-code)  
**Area**: Reinforcement Learning / Quality-Diversity Optimization
**Keywords**: quality-diversity, occupancy measure, random Fourier features, behavior descriptor, CMA-MAE

## TL;DR

AutoQD is proposed to embed policy occupancy measures into a finite-dimensional space via random Fourier features (RFF), followed by weighted PCA for dimensionality reduction to obtain behavior descriptors, enabling QD optimization without manually designed BDs. It comprehensively outperforms hand-crafted BDs and existing unsupervised QD methods across 6 continuous control tasks.

## Background & Motivation

**Background**: Quality-Diversity (QD) algorithms aim to discover a collection of policies that are both high-quality and behaviorally diverse, achieving success in robotic locomotion, game level generation, and protein design. QD-RL integrates QD principles into sequential decision-making, maintaining an archive where each cell stores the highest-return policy within a specific behavioral region.

**Limitations of Prior Work**: QD algorithms are highly dependent on **hand-crafted behavior descriptors (BDs)**—functions mapping policies to low-dimensional vectors (e.g., foot contact patterns of a bipedal robot). Designing BDs manually requires extensive domain knowledge and restricts diversity search to predefined dimensions, potentially missing interesting behavioral variants. Existing unsupervised methods (e.g., AURORA using autoencoders for BD learning) lack theoretical guarantees, while skill discovery methods such as DIAYN/SMERL require pre-specifying the number of skills and scale poorly.

**Key Challenge**: The occupancy measure $\rho^\pi(s,a) = (1-\gamma)\sum_{t=0}^{\infty}\gamma^t P(S_t=s, A_t=a|\pi)$ is the discounted visitation frequency distribution of a policy over the state-action space. Under standard assumptions, there is a **bijection** between Markov policies and their occupancy measures, making the occupancy measure a complete characterization of policy behavior. Can distances between occupancy measures be leveraged to automatically construct BDs?

## Method

### Core Idea: From Occupancy Measures to Behavior Descriptors

AutoQD proceeds in three steps: (1) embed policies into a Euclidean space where distances approximate the MMD between occupancy measures using RFF; (2) apply weighted PCA to reduce dimensionality to a low-dimensional BD; (3) perform QD optimization with CMA-MAE. The overall pipeline alternates between BD updates and policy search throughout optimization.

### Policy Embedding: RFF Approximation of MMD

Given state $s$ and action $a$, define a $D$-dimensional random feature map:

$$\phi(s,a) = \sqrt{\frac{2}{D}}\left[\cos(w_1^T[s;a]+b_1), \ldots, \cos(w_D^T[s;a]+b_D)\right]$$

where $w_i \sim \mathcal{N}(0, \sigma^{-2}I)$ and $b_i \sim \mathcal{U}(0, 2\pi)$. These random features approximate the Gaussian kernel $k(x,y) = \exp(-\|x-y\|^2/(2\sigma^2))$.

The embedding of policy $\pi$ is defined as the empirical mean of RFF under its occupancy measure. In practice, to fully exploit trajectory data, a discount-weighted formulation is adopted:

$$\psi^\pi = \frac{1}{n}\sum_{j=1}^{n}(1-\gamma)\sum_{t=0}^{T}\gamma^t \phi(s_t^j, a_t^j)$$

where $n$ is the number of trajectories. The Euclidean distance between two policy embeddings approximates the MMD between their occupancy measures: $\|\psi^{\pi_1} - \psi^{\pi_2}\| \approx \text{MMD}(\rho^{\pi_1}, \rho^{\pi_2})$.

**Theorem 1 (MMD Approximation Guarantee)**: For any two policies $\pi_1, \pi_2$, the error between the distance of their embeddings and the true MMD converges exponentially:

$$\Pr\left[\left|\|\phi_1 - \phi_2\|_2 - \text{MMD}(\rho_1, \rho_2)\right| \geqslant \frac{3}{4}\varepsilon\right] \leqslant 2e^{-nc\varepsilon^2} + \mathcal{O}\left(\frac{1}{\varepsilon^2}\exp\left(\frac{-D\varepsilon^2}{64(d+2)}\right)\right) + 6e^{-\frac{n\varepsilon^2}{8}}$$

Key implication: the embedding dimension $D$ need only grow **linearly** with the state-action dimension $d$ to control the error.

### Low-Dimensional Behavior Descriptors: cwPCA Projection

The high-dimensional embedding $\psi^\pi \in \mathbb{R}^D$ cannot be used directly as a BD (QD archive size grows exponentially with dimension), requiring reduction to $k \ll D$ dimensions. AutoQD employs **Calibrated Weighted PCA (cwPCA)**:

1. **Weighted PCA**: PCA is applied with embeddings weighted by policy fitness (return), giving higher-quality policies greater influence over the principal directions, encouraging exploration near high-quality behaviors.
2. **Calibration step**: Each output axis is scaled so that projected values lie within $[-1, 1]$, ensuring stable archive boundaries.

The final BD is an affine transformation $\text{desc}(\pi) = A\psi^\pi + b$, where $A \in \mathbb{R}^{k \times D}$ and $b \in \mathbb{R}^k$.

### AutoQD Complete Algorithm

The algorithm alternates between two phases:

1. **QD Optimization Phase**: Using the current BD and CMA-MAE to search for diverse policies; CMA-ES maintains a Gaussian distribution over policy parameters, performing sample → evaluate → rank by archive improvement → update distribution.
2. **BD Update Phase**: According to a predefined schedule, cwPCA projection matrices are recomputed from the embeddings of all policies in the archive, refreshing the BD definition.

Throughout this process, the random Fourier features $\{w_i, b_i\}$ are fixed after initialization; only the projection matrix $A, b$ is updated as the archive evolves.

## Key Experimental Results

**Environments**: 6 continuous control tasks—Ant, HalfCheetah, Hopper, Swimmer, Walker2d (MuJoCo) + BipedalWalker (Gymnasium).

**Baselines**: 5 comparison methods covering three categories—hand-crafted BD, unsupervised QD, and diversity RL:

| Baseline | Type | BD Source |
|:---------|:-----|:----------|
| RegularQD | Hand-crafted BD + CMA-MAE | Environment-specific manually designed BD |
| AURORA | Unsupervised QD | Autoencoder-reconstructed final-state latent codes |
| LSTM-AURORA | Unsupervised QD | LSTM-encoded hidden states over full trajectories |
| DvD-ES | Diversity Evolution | Policy action distributions over random states |
| SMERL | Diversity RL | Skill-conditioned policies + discriminator reward |

**Evaluation Metrics**:

| Metric | Description | Measures |
|:-------|:------------|:---------|
| GT QD Score | QD score computed using hand-crafted BD archive | Quality + human-defined diversity |
| Vendi Score (VS) | Effective population size based on occupancy embedding similarity | Pure diversity |
| qVS | Quality-weighted Vendi Score | Quality × diversity |

### Main Results: Comprehensive Comparison across 6 Environments

| Environment | Metric | AutoQD | RegularQD | AURORA | LSTM-AURORA | DvD-ES | SMERL |
|:------------|:-------|:-------|:----------|:-------|:------------|:-------|:------|
| Ant | QD (×10⁴) | **361.4** | 182.6 | 5.6 | 19.2 | 0.3 | 1.0 |
| Ant | VS | **72.4** | 39.5 | 1.1 | 1.9 | 1.0 | 1.3 |
| HalfCheetah | QD (×10⁴) | **30.8** | 24.9 | 11.4 | 11.4 | 0.9 | 1.6 |
| Hopper | QD (×10⁴) | **1.84** | 1.20 | 1.06 | 1.36 | 0.56 | 0.97 |
| Hopper | qVS | **1.94** | 1.35 | 0.66 | 0.36 | 0.90 | 1.81 |
| Swimmer | QD (×10⁴) | **21.3** | 11.1 | 8.1 | 10.3 | 0.2 | 0.02 |
| Walker2d | QD (×10⁴) | **18.4** | 11.4 | 7.7 | 13.0 | 0.6 | 1.2 |
| BipedalWalker | QD (×10⁴) | **6.09** | 1.81 | 3.00 | 3.36 | 0.09 | 0.14 |
| BipedalWalker | VS | **12.2** | 1.6 | 2.9 | 3.4 | 1.1 | 5.5 |

⭐ **Key Findings**: AutoQD achieves the **best GT QD Score in all 6 environments**, and ranks best in qVS and VS in 4/6 environments. The only exceptions are HalfCheetah (high VS but low qVS, discovering diverse but low-return "sliding" behaviors) and Walker2d (qVS/VS slightly below RegularQD).

### Adaptability Experiment: Robustness under Environmental Dynamics Changes

Adaptability under friction coefficient/mass variations tested on BipedalWalker:

| Variation | AutoQD | RegularQD | AURORA | LSTM-AURORA | DvD-ES | SMERL |
|:----------|:-------|:----------|:-------|:------------|:-------|:------|
| Friction AUC | **1429.7** | 30.3 | 1309.4 | 1226.3 | 1204.0 | 496.2 |
| Mass AUC | **295.7** | 12.8 | 260.6 | 271.8 | 113.7 | 71.4 |

⭐ AutoQD's diverse policy repertoire demonstrates the strongest adaptability under dynamic changes: both the best single-policy performance and the number of successfully adapting policies at a strict threshold ($p=0.9$) are highest.

## Highlights & Insights

**Strengths** ⭐⭐⭐⭐

- Solid theoretical foundation: grounded in the bijection between occupancy measures and MMD approximation theorems, providing probabilistic bounds on error convergence.
- Fully automated: no domain knowledge required for BD design; embedding dimension needs only linear growth with $d$.
- Comprehensive experiments: 6 environments × 5 baselines × 3 metrics, 3 random seeds, covering hand-crafted BD, unsupervised QD, and diversity RL.
- Convincing adaptability validation: systematic evaluation under friction and mass variations.

**Weaknesses** ⭐⭐⭐

- qVS is not optimal on HalfCheetah/Walker2d, suggesting that automatic BDs may over-emphasize certain behavioral dimensions while neglecting human-relevant variants.
- Large numbers of trajectories are needed to estimate embeddings in highly stochastic environments, leading to low sample efficiency.
- The kernel bandwidth $\sigma$ is fixed and does not adapt to different stages of learning.
- Only combined with CMA-MAE; compatibility with gradient-based QD methods (e.g., PGA-ME, PPGA) is not validated.
- Experiments are limited to state-vector observation spaces and not extended to image observations.

## Key Findings

1. **Universality of occupancy measure embeddings**: The RFF embedding framework extends beyond QD and can be directly applied to policy clustering, policy matching in imitation learning, and behavior comparison in inverse reinforcement learning. Transforming policy space into a metrizable Euclidean space is an elegant and general tool.

2. **Limitations of cwPCA**: Weighted PCA is fundamentally a linear dimensionality reduction method. If the behavior space exhibits a nonlinear manifold structure, PCA may discard important information. Kernel PCA or nonlinear methods such as UMAP may further improve BD quality.

3. **Challenges of integration with gradient-based QD methods**: The paper notes that BD updates destabilize gradient-based QD methods—changes in BD imply changes in the objective function, invalidating the direction of policy gradients. Smooth BD updates (e.g., EMA) or freezing the BD for multiple gradient steps before switching may be necessary.

4. **Practical value**: The approach is particularly valuable for building robotic behavior repertoires—no manual definition of "what constitutes a different behavior" is needed, as the algorithm automatically discovers the principal dimensions of behavioral diversity. In sim-to-real scenarios, diverse policy repertoires provide a natural complement to domain randomization.

## Related Work & Insights

- **MAP-Elites** (Cully et al., 2015): Foundational QD work maintaining a BD-organized archive.
- **CMA-MAE** (Fontaine & Nikolaidis, 2023): Reformulates QD as single-objective optimization with a soft archive mechanism.
- **AURORA** (Grillotti & Cully, 2022): Learns BDs with autoencoders; lacks theoretical guarantees.
- **DIAYN** (Eysenbach et al., 2019): Discovers diverse skills by maximizing skill-state mutual information.
- **SMERL** (Kumar et al., 2020): Extends DIAYN with task reward.
- **DvD-ES** (Parker-Holder et al., 2020): Characterizes behavioral differences via policy action distributions over random states.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Post-training Large Language Models for Diverse High-Quality Responses](post-training_large_language_models_for_diverse_high-quality_responses.md)
- [\[ICLR 2026\] SUSD: Structured Unsupervised Skill Discovery through State Factorization](susd_structured_unsupervised_skill_discovery_through_state_factorization.md)
- [\[ICLR 2026\] AutoTool: Automatic Scaling of Tool-Use Capabilities in RL via Decoupled Entropy Constraints](autotool_automatic_scaling_of_tool-use_capabilities_in_rl_via_decoupled_entropy_.md)
- [\[NeurIPS 2025\] Inner Speech as Behavior Guides: Steerable Imitation of Diverse Behaviors for Human-AI Coordination](../../NeurIPS2025/reinforcement_learning/inner_speech_as_behavior_guides_steerable_imitation_of_diverse_behaviors_for_hum.md)
- [\[ICLR 2026\] Rethinking Policy Diversity in Ensemble Policy Gradient in Large-Scale Reinforcement Learning](rethinking_policy_diversity_in_ensemble_policy_gradient_in_large-scale_reinforce.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Adaptively Coordinating with Novel Partners via Learned Latent Strategies
description: >-
  [NeurIPS 2025][Reinforcement Learning][zero-shot coordination] This paper proposes the TALENTS framework, which learns a latent strategy space via a VAE, discovers strategy types through K-Means clustering…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "zero-shot coordination"
  - "ad hoc teamwork"
  - "latent strategy"
  - "VAE"
  - "regret minimization"
  - "human-agent collaboration"
date: 2026-05-08
content_hash: 4d9d6e740686f595
---

# Adaptively Coordinating with Novel Partners via Learned Latent Strategies

**Conference**: NeurIPS 2025
**arXiv**: [2511.12754](https://arxiv.org/abs/2511.12754)  
**Code**: To be confirmed  
**Area**: Reinforcement Learning
**Keywords**: zero-shot coordination, ad hoc teamwork, latent strategy, VAE, regret minimization, human-agent collaboration

## TL;DR

This paper proposes the TALENTS framework, which learns a latent strategy space via a VAE, discovers strategy types through K-Means clustering, and performs online teammate-type inference using the Fixed-Share regret minimization algorithm, enabling zero-shot real-time adaptive coordination with unknown human or agent teammates.

## Background & Motivation

**Core Challenge in Ad Hoc Teamwork**: An agent must collaborate in real time with unknown teammates (human or AI) whose behavioral patterns, preferences, and skill levels vary substantially and may shift dynamically during interaction.

**Limitations of Prior Work**:
- **Self-Play**: Agents trained via self-play tend to develop rigid conventions that generalize poorly to diverse teammates.
- **Population-Based Training (PBT)**: Methods such as FCP and MEP expand strategy coverage by training over diverse populations, yet remain constrained by discrete, finite population sizes that cannot cover the continuous distribution of human behavior.
- **Existing Strategy Inference Methods**: Directly encoding a new teammate's trajectory into a latent space suffers from distribution shift between training-time and test-time trajectory distributions, making such approaches brittle when interacting with humans.

**Key Insight**: Unify PBT-style strategy diversity generation with online strategy inference—using a single latent space both to generate training partners and to infer teammate types at test time.

**Non-Stationarity Problem**: Teammates may switch strategies multiple times within a single episode (due to partner influence or ongoing learning), which standard static regret minimization cannot handle.

## Method

### Overall Architecture

TALENTS (Team Adaptation via LatEnt No-regreT Strategies) proceeds in three stages:

```
Offline trajectory data → VAE strategy space learning → K-Means clustering
                                                                ↓
                    Strategy-conditioned Cooperator training ← Sample partners from clusters
                                                                ↓
                    Online deployment: Fixed-Share inference of teammate type → Conditioned execution
```

### Key Designs

**Design 1: VAE Strategy Space Learning**

- **Input**: Offline trajectory data $\mathcal{D}_{traj} = \{\tau_i\}_{i=1}^N$ collected from joint rollouts of population agents (FCP/MEP/BP).
- **Encoder** $q_\phi(z|\tau)$: Maps a trajectory window of length $h$ to a multivariate Gaussian $\mathcal{N}(\mu_\phi(\tau), \Sigma_\phi(\tau))$ with latent dimension 8.
- **Decoder** $p_\theta(a_{t:t+H}|z, o_t)$: Given latent variable $z$ and current observation $o_t$, predicts an action sequence of $H=50$ future steps to capture long-horizon intent.
- **Clustering**: K-Means is applied in the latent space with Silhouette Analysis to automatically determine the optimal number of clusters $K$ and their centroids.

**Design 2: Strategy-Conditioned Cooperator Training**

- At each episode, a strategy cluster $c$ is sampled, $z$ is drawn from its latent mean $\mu_c$, and partner actions are generated via the decoder.
- **Key Mechanism — Action Bias**: An embedding matrix $E$ learns a bias vector $b_c = E[c]$ per cluster, which is added to the actor network's logits: $\tilde{l}_t = l_t + b_c$.
- This design explicitly encourages or suppresses specific actions of the cooperator under different teammate types, and is more effective than naively concatenating cluster embeddings to the observation.
- Priority-based Sampling adjusts cluster sampling weights based on historical returns; Independent PPO is used for training.

**Design 3: Fixed-Share Online Adaptation**

- Each strategy cluster serves as an "expert"; weights are initialized uniformly as $w^1 = (1/K, \ldots, 1/K)$.
- At each step, each expert decodes a predicted teammate action $\hat{a}_t^c$ from its latent variable.
- Upon observing the teammate's actual action $a_t^p$, the loss for each expert is computed as $\ell_c^t = -\log p_\theta(a_t^p | z_c, o_t)$.
- Exponential weight update with **weight sharing**: $w_c^{t+1} = (1-\alpha)\tilde{w}_c^{t+1} + \alpha \sum_j \tilde{w}_j^{t+1} / K$.
- The sharing parameter $\alpha$ enables tracking of non-stationary teammates (tolerating up to $m-1$ strategy switches), with a regret bound of $O(\sqrt{T(m\ln N + m\ln(T/m))})$.

### Loss & Training

The VAE is trained by optimizing the ELBO:

$$\mathcal{L}(\theta, \phi; \tau) = \mathbb{E}_{z \sim q_\phi}[\log p_\theta(a_{t:t+H}|z, o_t)] - \beta D_{KL}(q_\phi(z|\tau) \| p(z))$$

where $\beta$ is linearly annealed during training (KL annealing) to balance reconstruction fidelity and latent space regularization.

## Key Experimental Results

### Experimental Setup

An enhanced version of Overcooked-ai is used, incorporating order timers and fast-delivery bonuses, three cooking stations, two recipes, and four map layouts (Open, Hallway, Forced-Coord, Ring).

### Main Results

**Agent–Agent Zero-Shot Coordination (Table 1)**

| Population | Method | Open | Hallway | Forced-Coord | Ring |
|------|------|------|---------|-------------|------|
| FCP | **TALENTS** | **710.36±88.75** | **635.59±107.54** | 34.38±6.59 | **596.19±33.34** |
| FCP | GAMMA | 616.67±14.99 | 537.60±26.75 | 38.36±6.65 | 395.03±10.09 |
| FCP | BR | 427.07±14.17 | 366.14±70.50 | **56.09±12.33** | 288.61±31.85 |
| BP | **TALENTS** | **842.39±36.47** | **642.94±81.00** | 56.55±9.09 | **647.62±16.96** |
| BP | GAMMA | 573.93±52.69 | 513.47±34.06 | 47.52±2.61 | 387.58±17.41 |

**Human–Agent Zero-Shot Coordination (N = 119 participants)**

- **Team Score**: TALENTS significantly outperforms both baselines (ANOVA: $F(2,166)=5.76, p=.003$).
- **Subjective Ratings**: Team fluency ($F(2,122)=4.31, p=.02$) and trust ($F(2,122)=3.23, p=.04$) are both significantly higher than the BR baseline.
- Positive trends are also observed for workload (NASA-TLX), coordination, and satisfaction.

### Ablation Study

1. **TALENTS surpasses all baselines on 3 out of 4 maps**; performance is weaker on Forced-Coord, where explicit role division leads to sparse reward signals that penalize strategy-exploration approaches.
2. **Fixed-Share vs. Static Regret**: When teammate strategy switches mid-episode, the static regret variant cannot update its belief and suffers degraded rewards in the second half; Fixed-Share tracks the switch and recovers performance (Fig. 4).
3. **Cross-Population Consistency**: TALENTS achieves the highest or near-highest performance regardless of whether FCP, MEP, or BP populations are used for training.
4. **Action Bias > Observation Concatenation**: The action bias mechanism more effectively learns strategy-specific behaviors than concatenating cluster embeddings to the observation.

## Highlights & Insights

- **Unified Framework**: The same VAE latent space serves both partner generation during training and strategy inference at test time—an elegant design.
- **Theoretical Guarantee**: Fixed-Share provides a formal tracking regret bound, offering mathematical guarantees for non-stationary teammates.
- **Human Adaptation without Human Data**: The model is trained solely on agent data yet significantly outperforms baselines in a 119-participant user study.
- **Action Bias Mechanism**: A simple embedding matrix achieves strategy conditioning without expanding the observation space.

## Limitations & Future Work

1. **Weak Performance on Forced-Coord**: When tasks have clearly defined role divisions, diversity-driven strategy exploration introduces noise, and priority sampling biases toward low-skill partners.
2. **Generalization Bounded by VAE Interpolation**: The approach can only generalize to new teammates whose behaviors are similar to those in the training data, and remains fundamentally limited by the strategy coverage of the training population.
3. **Artificially Reduced Action Frequency**: Humans act at approximately 3–4 actions per second, while agents act at every step (10 steps/s); the experiments artificially throttle the agent's action rate, which may impair long-horizon planning.
4. **Evaluation Limited to Overcooked**: Despite the increased task complexity, transfer to more realistic domains (e.g., robotic collaboration) has not been validated.
5. **Fixed Number of Clusters $K$**: Although K-Means with Silhouette Analysis is automated, it may fail to capture hierarchical structures or continuous gradations in the strategy space.

## Related Work & Insights

- **Zero-Shot Coordination**: Self-Play → FCP (checkpoints simulate varying skill levels) → MEP (maximum-entropy population) → E3T (mixing self-play and random strategies) → GAMMA (generative model for broader strategy coverage) → **TALENTS** (clustering + conditioning + online inference).
- **Strategy Inference**: Latent embedding methods (LIAM, CoDAG), Bayesian inference (MeLIBA), apprenticeship learning with mixture of experts (zhao2022), and cross-entropy policy selection (9540646). TALENTS distinguishes itself by using a single latent space for both generation and inference.
- **Theory of Mind**: Methods such as ProAgent and ToMnet predict partner intent and beliefs; these are complementary to the present work but have a different emphasis.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of VAE clustering and Fixed-Share online adaptation is original; unifying PBT and strategy inference within a shared latent space constitutes a clear contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers 3 populations (FCP/MEP/BP) × 4 maps, a 119-participant user study, and ablation experiments; the human study follows rigorous methodology (mixed design, subjective questionnaires).
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, complete mathematical formalization (HiP-MDP, tracking regret bound), and detailed algorithmic pseudocode.
- **Value**: ⭐⭐⭐⭐ — A practical approach to human–agent collaboration with statistically significant human study results; the framework is generalizable to other cooperative settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Hybrid Latent Reasoning via Reinforcement Learning](hybrid_latent_reasoning_via_reinforcement_learning.md)
- [\[NeurIPS 2025\] Inverse Optimization Latent Variable Models for Learning Costs Applied to Route Problems](inverse_optimization_latent_variable_models_for_learning_costs_applied_to_route_.md)
- [\[NeurIPS 2025\] Dynamics-Aligned Latent Imagination in Contextual World Models for Zero-Shot Generalization](dynamics-aligned_latent_imagination_in_contextual_world_models_for_zero-shot_gen.md)
- [\[ACL 2026\] SpiralThinker: Latent Reasoning through an Iterative Process with Text-Latent Interleaving](../../ACL2026/reinforcement_learning/spiralthinker_latent_reasoning_through_an_iterative_process_with_text-latent_int.md)
- [\[ACL 2026\] Glance-or-Gaze: Incentivizing LMMs to Adaptively Focus Search via Reinforcement Learning](../../ACL2026/reinforcement_learning/glance-or-gaze_incentivizing_lmms_to_adaptively_focus_search_via_reinforcement_l.md)

</div>

<!-- RELATED:END -->

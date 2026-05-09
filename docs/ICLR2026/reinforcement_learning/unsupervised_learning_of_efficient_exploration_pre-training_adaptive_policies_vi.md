---
title: >-
  [Paper Note] Unsupervised Learning of Efficient Exploration: Pre-training Adaptive Policies via Self-Imposed Goals
description: >-
  Proposes ULEE, a method that meta-learns exploration-efficient and fast-adaptation pretraining strategies in unsupervised settings via adversarial goal generation and post-adaptation difficulty-based curriculum learning.
date: 2026-05-08
tags:
  - ICLR 2026
  - Reinforcement Learning
  - unsupervised RL
  - meta-learning
  - goal generation
  - curriculum learning
  - exploration
content_hash: 5bae29ef6a850147
---
## Basic Information

- **Conference**: ICLR 2026
- **arXiv**: [2601.19810](https://arxiv.org/abs/2601.19810)
- **Code**: Open-sourced (stated in the paper: all code is open-sourced)
- **Area**: Reinforcement Learning / Unsupervised Pre-training / Meta-Learning
- **Keywords**: unsupervised RL, meta-learning, goal generation, curriculum learning, exploration

## TL;DR

This paper proposes ULEE, an unsupervised meta-learning method that trains adaptive policies via adversarially self-generated goal curricula, achieving efficient exploration and few-shot adaptation on the XLand-MiniGrid benchmark.

## Background & Motivation

Large-scale pre-training has achieved remarkable success in CV and NLP, yet RL remains dominated by single-task training from scratch. Unsupervised RL aims to acquire transferable foundation policies without external rewards, with core challenges including:

**Data collection**: What data should the agent collect? The collected data is directly determined by its own behavior.

**Goal generation**: How should an agent that autonomously generates goals effectively create, select, and exploit those goals?

**Downstream generalization**: When the target task distribution is broad and out-of-distribution, zero-shot solution of all tasks is infeasible, necessitating efficient adaptation.

Existing methods exhibit the following limitations:
- Methods such as GoalGAN evaluate goal difficulty based on immediate performance, without considering post-adaptation performance.
- Goal-conditioned policies fail when goals are unknown or their representations are uninterpretable.
- The unsupervised meta-learning setting remains largely underexplored.

## Method

### Overall Architecture

ULEE (Unsupervised Learning of Efficient Exploration) comprises four core components:

1. **Pre-trained Policy $\pi$**: A non-goal-conditioned in-context learner that adapts through multi-episode interaction histories.
2. **Goal-search Policy $\pi_{gs}$**: An adversarially trained goal-search policy that seeks difficult goals.
3. **Difficulty Predictor**: Predicts the difficulty of a goal after adaptation.
4. **Goal Selection**: Samples goals within a moderate difficulty range.

### Key Design 1: Post-Adaptation Difficulty Metric

Unlike prior work that evaluates difficulty based on immediate performance, ULEE defines difficulty in terms of **post-adaptation** performance:

$$d(g; \pi, M) = 1 - \mathbb{E}_{\rho_M, P_M, \pi}\left[\frac{1}{K}\sum_{j=H-K+1}^{H} \mathbf{1}\left\{\exists t: f(s_{t+1}^{(j)}) = g\right\}\right]$$

where $H$ is the total number of episodes; only the success rate of the last $K$ episodes is evaluated, while the preceding $H-K$ episodes serve for exploration and adaptation. This better matches the evaluation scenario.

### Key Design 2: Adversarial Goal Generation

The goal-search policy is trained to maximize the difficulty of discovered goals:

$$r_t^{gs} = r^{gs}(s_t; \pi, M) = d(f(s_t); \pi, M)$$

In each environment, $\pi_{gs}$ is first executed to collect a candidate goal set $GC_M$, from which goals are sampled according to a difficulty range $[LB, UB]$.

### Key Design 3: In-Context Meta-Learning

Policy $\pi$ is trained to maximize the cumulative discounted return over a multi-episode lifetime:

$$\mathcal{J}(\pi) = \mathbb{E}_{M \sim \mu^{\text{unsup}}, g \sim p(g|M)}\left[\mathbb{E}_{\rho_M, P_M, \pi}\left[\sum_{j=1}^{H}\sum_{t=0}^{T-1} \gamma^{(j-1)T+t} r_t^{(j)}\right]\right]$$

A Transformer-XL backbone is employed, enabling in-context learning via historical context across multiple episodes.

### Loss & Training

- **Difficulty Predictor**: Supervised L2 regression loss

$$\mathcal{L}_{DP}(\phi) = \frac{1}{|B_g|}\sum_{(g,\xi,\tilde{d}) \in B_g} \left(\hat{d}_\phi(g, \xi) - \tilde{d}(g)\right)^2$$

- **Policy Optimization**: PPO with a Transformer-XL backbone.

## Key Experimental Results

### Main Results

Evaluation on three XLand-MiniGrid benchmarks (4Rooms-Trivial, 4Rooms-Small, 6Rooms-Small):

| Evaluation Dimension | ULEE vs. Random | ULEE vs. DIAYN |
|---|---|---|
| Exploration (20 ep) | 2× goal discovery rate | 2×+ |
| Fast adaptation (30 ep) | 3× mean return improvement | Significantly superior |
| Fine-tuning (1B steps) | Consistently superior | DIAYN advantage is brief |
| Meta-RL initialization | Consistently superior | — |

### Ablation Study

| Variant | Description | Result |
|---|---|---|
| adversarial + bounded | Full ULEE | **Best** |
| random + bounded | Random search + moderate difficulty sampling | Plateaus after initial adaptation |
| adversarial + uniform | Adversarial search + uniform sampling | Below full model |
| ULEE (SED) | Immediate rather than post-adaptation difficulty | Performance gap grows with difficulty |

### Generalization Experiments

After pre-training on 4Rooms-Small, evaluation is conducted across various MiniGrid tasks:
- ULEE ($f_{\text{counts}}$) achieves non-zero returns on all 14 test environments.
- Substantially outperforms baselines on tasks such as Unlock and UnlockPickUp.

### Key Findings

1. The post-adaptation difficulty metric is more effective than the immediate difficulty metric; the gap widens as benchmark difficulty increases.
2. The combination of adversarial goal search and bounded sampling yields the best results.
3. The goal mapping $f$ serves as an inductive bias with a significant impact on outcomes.
4. The pre-trained policy continues to improve with increasing environment steps.

## Highlights & Insights

- **Post-Adaptation Difficulty Metric**: The first introduction of a difficulty metric based on post-adaptation performance in unsupervised RL.
- **Unified Framework**: Integrates unsupervised goal generation, automatic curriculum learning, and meta-learning into a single system.
- **Multi-Scale Evaluation**: Comprehensive validation spanning zero-shot to long-term fine-tuning.
- **No Goal Conditioning Required**: The policy is deployed directly without goal encoding.

## Limitations & Future Work

- Experiments are limited to discrete-action grid-world environments; applicability to continuous control tasks remains unvalidated.
- Performance on finer-grained task hierarchies (e.g., rule trees with depth > 1) still has substantial room for improvement.
- More than 60% of test tasks yield zero return under few-shot settings, indicating that difficult tasks remain a challenge.
- Computational budget is large (up to 5B steps).

## Related Work & Insights

- **Intrinsic reward methods**: RND (Burda et al., 2018), DIAYN (Eysenbach et al., 2018)
- **Automatic curriculum learning**: GoalGAN (Florensa et al., 2018), AMIGo (Campero et al., 2020)
- **Unsupervised meta-learning**: Gupta et al. (2018), Jabri et al. (2019)
- **In-context RL**: RL² (Duan et al., 2016), Ada (Team et al., 2023)

## Rating

- **Novelty**: 8/10 — The combination of post-adaptation difficulty metric and adversarial goal generation is novel.
- **Technical Depth**: 8/10 — The co-design of four components is well-considered.
- **Experimental Thoroughness**: 7/10 — Evaluation is comprehensive but environmental complexity is limited.
- **Writing Quality**: 8/10 — Well-organized with rigorous mathematical descriptions.
- **Overall**: 7.5/10

---

## Basic Information

- **Conference**: ICLR 2026
- **arXiv**: [2601.19810](https://arxiv.org/abs/2601.19810)
- **Code**: Open-sourced
- **Area**: Reinforcement Learning / Unsupervised Pre-training / Meta-Learning
- **Keywords**: unsupervised RL, automatic curriculum learning, meta-learning, goal generation, exploration policy

## TL;DR

This paper proposes ULEE, a method that meta-learns pre-trained policies with efficient exploration and rapid adaptation capabilities in unsupervised environments, via adversarial goal generation and a post-adaptation difficulty-based curriculum.

## Background & Motivation

### Core Problem

Large-scale pre-training has achieved tremendous success in vision and language, yet reinforcement learning is still dominated by training from scratch. The central question is: how can general RL policies (foundation policies) be pre-trained without external rewards, endowing them with transferable exploration and adaptation capabilities?

### Limitations of Prior Work

1. **Intrinsic reward methods** (e.g., DIAYN): The diversity of learned skills is limited; performance tends to plateau or even degrade as training progresses.
2. **Goal-conditioned policies**: Perform poorly when goals are unknown or cannot be encoded.
3. **Fixed goal space assumption**: Most curriculum learning methods assume the goal space is consistent between training and evaluation.
4. **Immediate performance-based difficulty estimation**: Does not account for the adaptation budget and is unsuitable for evaluation scenarios requiring multiple rounds of adaptation.

### Key Motivation

Humans develop capabilities by autonomously setting and pursuing goals. The paper addresses three core questions: **how goals are generated**, **how they are selected**, and **how to learn from them**. When the downstream task distribution is broad and unknown, zero-shot solution of all tasks is infeasible; therefore, optimizing multi-round exploration and adaptation efficiency is essential.

## Method

### Overall Architecture: ULEE (Unsupervised Learning of Efficient Exploration)

ULEE consists of four core components:
1. **Pre-trained policy** $\pi$ (in-context learner)
2. **Goal-search policy** $\pi_{gs}$ (adversarial goal generation)
3. **Difficulty prediction network** (estimates post-adaptation performance)
4. **Goal sampling strategy** (selects from a moderate difficulty range)

### Pre-trained Policy

A black-box meta-learning approach is adopted; the policy selects actions based on the complete interaction history (including past observations, actions, and rewards), enabling in-context adaptation. The training objective is to maximize the expected discounted return over the entire lifetime:

$$\mathcal{J}(\pi) = \mathbb{E}_{M \sim \mu^{\text{unsup}}, g \sim p(g|M)} \left[ \mathbb{E}_{\rho_M, P_M, \pi} \left[ \sum_{j=1}^{H} \sum_{t=0}^{T-1} \gamma^{(j-1)T+t} r_t^{(j)} \right] \right]$$

where $j$ indexes the episodes within the lifetime and $H$ is the total number of episodes.

### Post-Adaptation Goal Difficulty Metric

**Core innovation**: Difficulty is defined as the complement of the policy's performance after the adaptation budget, rather than as the immediate success rate:

$$d(g; \pi, M) = 1 - \mathbb{E}_{\rho_M, P_M, \pi} \left[ \frac{1}{K} \sum_{j=H-K+1}^{H} \mathbf{1}\{ \exists t: f(s_{t+1}^{(j)}) = g \} \right]$$

Only the success rate of the last $K$ episodes is counted; the exploration and adaptation process of the preceding $H-K$ episodes is excluded.

### Adversarial Goal Search

The goal-search policy $\pi_{gs}$ is trained to maximize the difficulty of discovered goals:

$$r_t^{gs} = r^{gs}(s_t; \pi, M) = d(f(s_t); \pi, M)$$

In each environment, $\pi_{gs}$ is executed for several episodes prior to the pre-trained policy, collecting a candidate goal set.

### Goal Selection and Sampling

Goals of moderate difficulty are selected from the candidate set:

$$g_M \sim \text{Unif}(S), \quad S = \{g \in GC_M : LB \leq d(g; \pi, M) \leq UB \}$$

where $LB = 0.1$ and $UB = 0.9$, avoiding uninformative goals that are too easy or too hard.

### Difficulty Prediction Network

A supervised difficulty predictor is introduced using an L2 regression loss:

$$\mathcal{L}_{DP}(\phi) = \frac{1}{|B_g|} \sum_{(g, \xi, \tilde{d}) \in B_g} (\hat{d}_\phi(g, \xi) - \tilde{d}(g))^2$$

This provides an approximate immediate difficulty estimate, avoiding additional environment interactions.

## Key Experimental Results

### Experimental Setup

- **Environment**: XLand-MiniGrid, a JAX-based procedurally generated partially observable grid environment.
- **Three benchmarks**: 4Rooms-Trivial, 4Rooms-Small, 6Rooms-Small.
- **Baselines**: DIAYN, PPO (from scratch), RND (online exploration), RL² (meta-learning).

### Main Results

| Metric | ULEE | DIAYN | Random |
|---|---|---|---|
| 20-episode exploration coverage | **Highest (2×+)** | Moderate | Low |
| Few-shot adaptation (30 episodes) | **3× improvement** | Step-wise improvement | — |
| Fine-tuning (1B steps) | **Consistently superior** | Brief advantage | — |
| Meta-RL initialization | **Universally superior** | — | Baseline |

### Ablation Study

| Variant | Goal Search | Sampling Strategy | Relative Performance |
|---|---|---|---|
| ULEE (adversarial + bounded) | Adversarial | Moderate difficulty | **Optimal** |
| ULEE (random + bounded) | Random | Moderate difficulty | Second best |
| ULEE (adversarial + uniform) | Adversarial | Uniform | Slightly below full model |
| ULEE (SED) | Adversarial | Immediate difficulty | Degrades as difficulty increases |

### Key Findings

1. Adversarial goal search combined with moderate difficulty sampling achieves the best performance.
2. The post-adaptation difficulty metric yields greater advantages in more challenging environments.
3. The goal mapping $f_{\text{counts}}$ serves as a more effective inductive bias than $f_{\text{grid}}$.
4. Performance continues to improve as the pre-training budget is scaled up to 5 billion steps.
5. ULEE generalizes to MiniGrid tasks with varying grid sizes and room configurations.

## Highlights & Insights

1. **Post-Adaptation Difficulty Metric**: Extending curriculum learning from immediate evaluation to a meta-learning setting that accounts for the adaptation budget represents a significant conceptual contribution.
2. **Unconditional Policy Pre-training**: No goal conditioning is required; the policy is deployed directly, broadening its applicability.
3. **Multi-Level Evaluation**: Covers four dimensions—zero-shot exploration, few-shot adaptation, long-term fine-tuning, and meta-RL initialization.
4. **Systematic Ablation**: Clearly demonstrates the contribution of each component.

## Limitations & Future Work

1. Validation is limited to 2D grid worlds; scalability to high-dimensional continuous control tasks is unknown.
2. The design of the goal mapping $f$ still requires manual specification; automatically discovering appropriate goal spaces remains an open problem.
3. Adversarial training introduces an additional ~25% computational overhead.
4. On the most difficult out-of-distribution tasks, more than 60% of tasks still yield zero return.

## Related Work & Insights

- **Unsupervised RL**: DIAYN, RND, and other intrinsic reward methods.
- **Automatic curriculum learning**: GoalGAN, AMIGo, etc.—none of which consider post-adaptation metrics.
- **Unsupervised meta-learning**: Gupta et al. (2018) first explored this direction; ULEE extends it with adversarial curriculum learning.
- **Ada (DeepMind)**: Large-scale meta-learning in procedurally generated environments, but relies on external task distributions rather than self-generated goals.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of post-adaptation difficulty metric and unconditional policy meta-learning is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-dimensional evaluation, though environmental complexity is limited.
- **Writing Quality**: ⭐⭐⭐⭐ — Methods and experiments are clearly organized.
- **Value**: ⭐⭐⭐ — Breakthrough applications require further extension to more complex environments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] How Far Can Unsupervised RLVR Scale LLM Training?](how_far_can_unsupervised_rlvr_scale_llm_training.md)
- [\[ICLR 2026\] Stop Unnecessary Reflection: Training LRMs for Efficient Reasoning with Adaptive Reflection and Length Coordinated Penalty](stop_unnecessary_reflection_training_lrms_for_efficient_reasoning_with_adaptive_.md)
- [\[ICLR 2026\] AMPED: Adaptive Multi-objective Projection for balancing Exploration and skill Diversification](amped_adaptive_multi-objective_projection_for_balancing_exploration_and_skill_di.md)
- [\[ICLR 2026\] Sample-efficient and Scalable Exploration in Continuous-Time RL](sample-efficient_and_scalable_exploration_in_continuous-time_rl.md)
- [\[ACL 2026\] Data Mixing Agent: Learning to Re-weight Domains for Continual Pre-training](../../ACL2026/reinforcement_learning/data_mixing_agent_learning_to_re-weight_domains_for_continual_pre-training.md)

</div>

<!-- RELATED:END -->

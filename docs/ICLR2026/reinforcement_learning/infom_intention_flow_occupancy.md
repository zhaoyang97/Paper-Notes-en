---
title: >-
  [Paper Note] InFOM: Intention-Conditioned Flow Occupancy Models
description: >-
  [ICLR 2026][Reinforcement Learning][Occupancy measures] InFOM learns a latent intention encoder via variational inference and models intention-conditioned discounted state occupancy measures using flow matching…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Occupancy measures"
  - "flow matching"
  - "intention inference"
  - "pre-training & fine-tuning"
  - "generalized policy improvement"
date: 2026-05-08
content_hash: 2a41e5f9a102e956
---

# InFOM: Intention-Conditioned Flow Occupancy Models

**Conference**: ICLR 2026
**arXiv**: [2506.08902](https://arxiv.org/abs/2506.08902)  
**Code**: [https://github.com/chongyi-zheng/infom](https://github.com/chongyi-zheng/infom)  
**Area**: Reinforcement Learning
**Keywords**: Occupancy measures, flow matching, intention inference, pre-training & fine-tuning, generalized policy improvement

## TL;DR

InFOM learns a latent intention encoder via variational inference and models intention-conditioned discounted state occupancy measures using flow matching, enabling efficient pre-training and fine-tuning in RL. It achieves 1.8× median return and 36% higher success rate over baselines across 36 state-based tasks and 4 image-based tasks.

## Background & Motivation

**Background**: The pre-train–fine-tune paradigm of foundation models has achieved remarkable success in NLP and CV, but remains an open problem in reinforcement learning. The core challenges lie in reasoning across time (actions have long-horizon dependencies) and identifying the diverse intentions of different users in offline datasets.

**Limitations of Prior Work**: Most existing RL pre-training methods overlook both the temporal and intentional dimensions. Behavioral cloning predicts actions without reasoning about long-term consequences; world models suffer from compounding errors that hinder long-horizon prediction; occupancy models (successor representations) can predict future state distributions but are difficult to train and ignore user intentions.

**Key Challenge**: Large-scale offline datasets are typically collected by multiple users performing different tasks, yet existing pre-training methods either ignore intentions (leading to mode averaging) or rely on discrete skills (limiting expressiveness), failing to exploit the heterogeneous structure in the data.

**Goal**: To construct a probabilistic model that simultaneously captures (1) temporal information (long-horizon state visitation distributions) and (2) user intentions, enabling efficient RL pre-training and downstream fine-tuning.

**Key Insight**: Combining variational inference for learning latent intentions, an advanced generative model (flow matching) for modeling occupancy measures, and generalized policy improvement (GPI) to aggregate Q-functions across intentions for policy extraction.

**Core Idea**: A latent variable model encodes user intentions, while flow matching models intention-conditioned discounted state occupancy measures, enabling intention-aware long-horizon prediction and efficient policy extraction.

## Method

### Overall Architecture

**Pre-training phase**: Reward-free offline dataset $D = \{(s,a,s',a')\}$ → variational intention encoder $p_e(z|s',a')$ infers the latent intention of each transition → SARSA flow loss trains an intention-conditioned flow occupancy model $q_d(s_f|s,a,z)$ to predict discounted future state distributions.

**Fine-tuning phase**: Given a reward-labeled dataset → sample future states from the occupancy model to compute Monte Carlo Q-values → distill implicit GPI via expectile loss → policy optimization with behavioral cloning regularization.

### Key Designs

1. **Variational Intention Inference**:

    - **Function**: Infers the latent intention of the data-collecting policy from consecutive transition pairs.
    - **Mechanism**: Maximizes the evidence lower bound (ELBO) of the likelihood of observing future state $s_f$ given $(s,a)$. The encoder $p_e(z|s',a')$ infers intentions from the next transition (exploiting the consistency assumption—consecutive transitions share the same intention); the decoder $q_d(s_f|s,a,z)$ predicts future states conditioned on the intention. KL divergence regularization $D_{KL}(p_e(z|s',a') \| \mathcal{N}(0,I))$ enforces an information bottleneck. Inferring intentions from the next transition rather than the current one prevents overfitting.
    - **Design Motivation**: Intentions serve as an information bottleneck that captures the heterogeneous behavioral structure in the data while preventing the encoder from degenerating into an identity mapping.

2. **SARSA Flow Occupancy Model**:

    - **Function**: Models intention-conditioned discounted state occupancy measures using flow matching.
    - **Mechanism**: The occupancy measure satisfies the Bellman equation $p_\gamma^\pi(s_f|s,a) = (1-\gamma)\delta_s(s_f) + \gamma \mathbb{E}[p_\gamma^\pi(s_f|s',a')]$. This is embedded into the flow matching framework by training a vector field $v_d(t, s^t, s, a, z)$. The SARSA flow loss comprises two parts: a current flow loss (using $s$ itself as the target) and a future flow loss that applies TD-bootstrapping over $(s',a')$ recursively. SARSA (rather than Q-learning) bootstrapping is chosen to avoid counterfactual errors introduced by intention conditioning.
    - **Design Motivation**: Flow matching is more stable to train and faster to evaluate than diffusion models (deterministic ODE vs. stochastic SDE); TD-based training supports dynamic programming and trajectory stitching compared to Monte Carlo estimation.

3. **Implicit Generalized Policy Improvement (Implicit GPI)**:

    - **Function**: Efficiently extracts a policy from multiple intention-conditioned Q-functions.
    - **Mechanism**: Intention-conditioned Q-functions are first estimated via Monte Carlo: $Q_z(s,a) = \frac{1}{(1-\gamma)N}\sum_i r(s_f^{(i)})$, where $s_f^{(i)} \sim q_d(s_f|s,a,z)$. Standard GPI takes a max over a finite set of intentions, which can get stuck in local optima and requires backpropagating gradients through the ODE solver. Instead, the authors use the expectile loss $L_2^\mu$ to distill into a scalar Q-function, effectively performing a "soft max" over intentions. A BC-regularized actor then maximizes the distilled Q.
    - **Design Motivation**: The expectile replaces the hard max to avoid local optima and instability from ODE gradient backpropagation; BC regularization prevents out-of-distribution actions and the propagation of Q overestimation.

### Loss & Training

**Pre-training**: SARSA flow loss (Eq. 5) + KL divergence regularization (Eq. 4) jointly train the encoder and the flow model.

**Fine-tuning**: A reward predictor is trained with simple regression; the critic uses an expectile distillation loss (Eq. 7, $\mu \in [0.5, 1)$); the actor is trained with Q-maximization + BC regularization (Eq. 8).

## Key Experimental Results

### Main Results

| Domain | InFOM | Prev. SOTA | Baseline Name | Gain |
|--------|-------|------------|---------------|------|
| ExORL Jaco (4 tasks) | Significantly higher | ~0 return | All baselines | ~20× |
| OGBench Manipulation (20 tasks) | Highest success rate | Second best | FB Rep. | +36% |
| OGBench Visual (4 tasks) | Highest success rate | Second best | HILP | +31% |
| Real Robot | Outperforms baselines | — | Multiple | +34% |

### Ablation Study

| Configuration | Result |
|---------------|--------|
| InFOM (full) | Highest return + lowest variance |
| InFOM + standard GPI | 44% lower than implicit GPI, 8× larger variance |
| FOM + one-step PI (no intention) | Significant drop in both return and success rate |
| Discrete latent variables (VQ) | Continuous latent space generally superior |
| N=16 sampled future states | Good trade-off point for Q estimation |

### Key Findings

- On the most challenging OGBench manipulation tasks, InFOM achieves 36% higher success rate than the strongest baseline—primarily because diverse intentions allow exploration of different state regions, alleviating the sparse reward problem.
- t-SNE visualizations of intention encodings show that InFOM clearly separates "grasping" and "placing" behaviors, while FB+FOM and HILP+FOM produce entangled intention representations.
- Implicit GPI outperforms standard GPI by 44% while reducing variance by 8×, demonstrating the stability advantage of expectile distillation.

## Highlights & Insights

- **The unified intention–occupancy framework** is the paper's primary contribution—jointly learning user intentions and long-horizon state predictions under an elegant probabilistic framework, addressing the two core challenges in RL pre-training.
- **The choice of SARSA flow over Q-learning flow** is insightful: once intention conditioning is introduced, off-policy correction is no longer necessary, avoiding counterfactual errors and training instability.
- **The implicit GPI design** (expectile distillation in place of hard max) is a general policy aggregation technique transferable to any setting requiring GPI over a continuous conditioning space.

## Limitations & Future Work

- Intentions are inferred from consecutive $(s',a')$ pairs and may not fully capture trajectory-level intent.
- The variance of Monte Carlo Q estimates is sensitive to the number of samples $N$, leading to relatively high inter-seed variance in some domains (e.g., cheetah, puzzle).
- The consistency assumption (consecutive transitions share the same intention) may not hold in highly dynamic environments.
- Orthogonal combinations with behavioral cloning pre-training methods remain unexplored.

## Related Work & Insights

- **vs. TD Flows (Farebrother et al., 2025)**: TD Flows also uses flow matching to model occupancy measures but encodes intentions via forward-backward representations and performs GPI over a finite intention set. InFOM uses variational inference to learn a continuous intention space with implicit GPI, yielding superior performance.
- **vs. HILP (Park et al., 2024)**: HILP learns Hilbert representations as skills during pre-training. InFOM does not learn skills; instead it directly uses intention-conditioned occupancy models.
- **vs. Behavioral Cloning (BC) pre-training**: BC only imitates actions without reasoning about long-term consequences, whereas InFOM enables long-horizon reasoning through occupancy measures.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Elegantly unifies variational intention inference, flow matching occupancy models, and implicit GPI.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 40 tasks, 8 baselines, comprehensive ablations, and real-robot validation.
- Writing Quality: ⭐⭐⭐⭐ Well-structured but technically dense; requires a strong RL background.
- Value: ⭐⭐⭐⭐⭐ Provides a powerful and general framework for RL pre-training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Flow Actor-Critic for Offline Reinforcement Learning (FAC)](flow_actor-critic_for_offline_reinforcement_learning.md)
- [\[ICLR 2026\] Divide, Harmonize, Then Conquer It: Shooting Multi-Commodity Flow Problems with Multimodal Language Models](divide_harmonize_then_conquer_it_shooting_multi-commodity_flow_problems_with_mul.md)
- [\[ICLR 2026\] PolicyFlow: Policy Optimization with Continuous Normalizing Flow in Reinforcement Learning](policyflow_policy_optimization_with_continuous_normalizing_flow_in_reinforcement.md)
- [\[AAAI 2026\] First-Order Representation Languages for Goal-Conditioned RL](../../AAAI2026/reinforcement_learning/first-order_representation_languages_for_goal-conditioned_rl.md)
- [\[ICLR 2026\] Scalable Exploration for High-Dimensional Continuous Control via Value-Guided Flow](scalable_exploration_for_high-dimensional_continuous_control_via_value-guided_fl.md)

</div>

<!-- RELATED:END -->

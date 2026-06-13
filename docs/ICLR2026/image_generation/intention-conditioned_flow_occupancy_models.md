---
title: >-
  [Paper Note] Intention-Conditioned Flow Occupancy Models
description: >-
  [Image Generation] This paper proposes InFOM, which leverages flow matching to construct an intention-conditioned occupancy model. By applying variational inference to infer latent intentions from unannotated data…
tags:
  - "Image Generation"
date: 2026-05-08
content_hash: 1e98cbf1bfbd1d18
---

# Intention-Conditioned Flow Occupancy Models

## TL;DR

This paper proposes InFOM, which leverages flow matching to construct an intention-conditioned occupancy model. By applying variational inference to infer latent intentions from unannotated data, InFOM enables RL pre-training without labeled datasets, achieving a 1.8× improvement in median return and a 36% gain in success rate across 36 state-based tasks and 4 visual tasks.

## Background & Motivation

The large-scale pre-training–fine-tuning paradigm has achieved remarkable success in NLP and CV, yet remains an open problem in reinforcement learning (RL). The core challenges in RL include:

**Temporal Reasoning**: Agents must reason about the long-term consequences of current actions, while world models are limited by compounding errors that restrict long-horizon reasoning.

**Intention Reasoning**: Large-scale offline datasets are typically collected by multiple users performing different tasks, and these implicit "intentions" are not explicitly annotated.

**Limitations of Prior Work**: Behavioral cloning (BC) imitates actions without capturing intentions; discriminative occupancy models are difficult to train; successor feature methods generally ignore user intentions.

This paper proposes InFOM (Intention-conditioned Flow Occupancy Models), which jointly learns a probabilistic model to capture both temporal and intention information, enabling the pre-trained model to be aware of the behavioral goals of different users and thereby facilitating more efficient policy learning during downstream fine-tuning.

## Method

InFOM consists of two stages: pre-training and fine-tuning.

### Pre-training Stage

**1. Variational Intention Inference**

- Given an unannotated dataset $D=\{(s,a,s',a')\}$, latent intentions $z$ are inferred via variational inference.
- An intention encoder $p_e(z|s',a')$ infers the intention from the next transition $(s',a')$, based on a consistency assumption that consecutive transitions share the same intention.
- The ELBO is maximized: $\mathbb{E}[\log q_d(s_f|s,a,z)] - \lambda D_{KL}(p_e(z|s',a') \| p(z))$
- The prior is $p(z) = \mathcal{N}(0,I)$, and $\lambda$ controls the KL regularization strength.

**2. SARSA Flow Occupancy Models**

- Flow matching is used to learn a generative occupancy model $q_d(s_f|s,a,z)$ that predicts discounted state occupancy measures.
- Temporal difference (TD) reasoning is incorporated into the flow matching loss to enable dynamic programming and compositional generalization.
- The loss is decomposed into two components: a current flow loss $(1-\gamma)\mathcal{L}_{\text{current}}$ and a future flow loss $\gamma \mathcal{L}_{\text{future}}$.
- The SARSA variant is simpler and more stable than the Q-learning variant, exhibiting better performance on large datasets.

### Fine-tuning Stage

**3. Generative Value Estimation**

- The pre-trained occupancy model is frozen; $N=16$ future states $s_f^{(i)} \sim q_d(s_f|s,a,z)$ are sampled.
- The intention-conditioned Q-function is estimated via Monte Carlo: $Q_z(s,a) = \frac{1}{(1-\gamma)N}\sum_i r(s_f^{(i)})$
- Intentions $z$ are sampled from the prior $p(z)$ rather than the posterior.

**4. Implicit Generalized Policy Improvement (GPI)**

- The explicit max over a finite set of intentions in naive GPI is replaced by an upper expectile loss.
- Multiple $Q_z$ estimates are distilled into a single scalar Q-function: $\mathcal{L}(Q) = \mathbb{E}[L_2^\mu(Q_z(s,a) - Q(s,a))]$
- This avoids backpropagating gradients through the ODE solver, yielding more stable training.
- A behavioral cloning regularizer is appended to suppress out-of-distribution actions.

## Key Experimental Results

### Experiment 1: ExORL and OGBench Benchmarks

Comparison against 8 baseline methods across 36 state-based tasks and 4 visual tasks:

| Domain | InFOM | Best Baseline | Gain |
|--------|-------|---------------|------|
| walker (4 tasks avg) | **380.9** | 327.6 (MBPO+ReBRAC) | ~16% |
| jaco (4 tasks avg) | **727.4** | 67.7 (IQL) | ~20× |
| cube single (5 tasks) | **92.5** | 77.8 (MBPO+ReBRAC) | ~19% |
| visual tasks (4 tasks) | — | — | +31% over best |

- Matches or surpasses all baselines on 7 out of 9 domains.
- The most substantial improvement occurs in the jaco domain (~20×), attributed to the high-dimensional state space and sparse rewards.
- Outperforms the strongest baseline by 31% on image-based tasks.
- Overall median return improves by 1.8× and success rate by 36%.

### Experiment 2: Ablation Study on Implicit GPI

| Policy Extraction Method | quadruped jump Return | scene task 1 Success Rate |
|--------------------------|----------------------|--------------------------|
| InFOM (implicit GPI) | **Highest** | **Highest** |
| InFOM + GPI (naive max) | 44% lower | Lower, 8× variance |
| FOM + one-step PI | Significantly lower | Significantly lower |

- Implicit GPI outperforms naive GPI by 44% and reduces variance by 8×.
- Removing the intention encoder (FOM + one-step PI) causes a substantial performance drop, validating the importance of intention inference.

## Highlights & Insights

- **Unified Framework**: InFOM is the first to integrate intention inference with flow matching occupancy models, capturing both temporal and intention information within a single framework.
- **Implicit GPI**: Replacing the explicit max operation with an expectile loss avoids instability from ODE backpropagation and the limitations of finite intention sets.
- **Strong Empirical Performance**: Comprehensively outperforms 8 baselines across 36+4 tasks, with a ~20× improvement in the jaco domain.
- **Intention Visualization**: t-SNE visualizations demonstrate that InFOM discovers clustering structures aligned with ground-truth intentions, whereas representations from FB and HILP are entangled.

## Limitations & Future Work

1. Inferring intentions from consecutive state-action pairs is a simplification that may fail to accurately capture the original intention at the full trajectory level.
2. Monte Carlo Q-estimation introduces variance, as evidenced by relatively large cross-seed standard deviations on some tasks.
3. Joint pre-training of the encoder and flow model incurs higher computational cost compared to pure BC methods.
4. The consistency assumption—that consecutive transitions share the same intention—may not hold in complex real-world scenarios.

## Related Work & Insights

- **Offline Unsupervised RL**: FB (Touati & Ollivier, 2021) and HILP (Park et al., 2024) learn skills or representations but do not simultaneously model occupancy measures.
- **Occupancy Models / Successor Representations**: Dayan (1993), Janner et al. (2020), and TD flows (Farebrother et al., 2025) use flow matching to model occupancy measures but do not model intentions.
- **Generative RL**: Decision Transformer, Diffuser, and related methods use generative models to model trajectories or policies but do not explicitly predict long-horizon state distributions.
- **Representation Learning**: Contrastive learning, MAE, and similar approaches learn general-purpose representations but do not guarantee transferability to policy adaptation.
- **InFOM's Novelty**: Relative to the most closely related work, TD flows, InFOM introduces variational latent variables for intention modeling and replaces explicit GPI over finite sets with implicit GPI.

## Rating

⭐⭐⭐⭐ (4/5)

- Theoretical motivation is clear, with variational inference and flow matching occupancy models organically integrated.
- Experimental coverage is broad and baselines are comprehensive: 36+4 tasks × 8 baselines × 8 seeds.
- Implicit GPI constitutes an elegant engineering and theoretical contribution.
- Deductions: the intention consistency assumption is strong, and the variance issue in Monte Carlo estimation is not fully resolved.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SCFlow: Implicitly Learning Style and Content Disentanglement with Flow Models](../../ICCV2025/image_generation/scflow_implicitly_learning_style_and_content_disentanglement_with_flow_models.md)
- [\[ICCV 2025\] MAVFlow: Preserving Paralinguistic Elements with Conditional Flow Matching for Zero-Shot AV2AV Multilingual Translation](../../ICCV2025/image_generation/mavflow_preserving_paralinguistic_elements_with_conditional_flow_matching_for_ze.md)
- [\[ICLR 2026\] ToProVAR: Efficient Visual Autoregressive Modeling via Tri-Dimensional Entropy-Aware Semantic Analysis and Sparsity Optimization](toprovar_efficient_visual_autoregressive_modeling_via_tri-dimensional_entropy-aw.md)
- [\[ICCV 2025\] Joint Diffusion Models in Continual Learning](../../ICCV2025/image_generation/joint_diffusion_models_in_continual_learning.md)
- [\[NeurIPS 2025\] Emergence and Evolution of Interpretable Concepts in Diffusion Models](../../NeurIPS2025/image_generation/emergence_and_evolution_of_interpretable_concepts_in_diffusi.md)

</div>

<!-- RELATED:END -->

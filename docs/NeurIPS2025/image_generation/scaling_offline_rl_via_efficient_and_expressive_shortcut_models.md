---
title: >-
  [Paper Note] Scaling Offline RL via Efficient and Expressive Shortcut Models
description: >-
  [NeurIPS 2025][Image Generation][Offline RL] This paper proposes SORL, which leverages the self-consistency property of shortcut models to enable efficient single-stage training with variable inference steps for policy optimization in offline RL, while supporting both sequential and parallel test-time scaling.
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Offline RL"
  - "Shortcut Models"
  - "Flow Matching"
  - "Self-Consistency"
  - "Test-time Scaling"
date: 2026-05-08
content_hash: a6c19513904e862e
---

# Scaling Offline RL via Efficient and Expressive Shortcut Models

**Conference**: NeurIPS 2025
**arXiv**: [2505.22866](https://arxiv.org/abs/2505.22866)  
**Code**: [nico-espinosadice.github.io/projects/sorl](https://nico-espinosadice.github.io/projects/sorl)  
**Area**: Image Generation
**Keywords**: Offline RL, Shortcut Models, Flow Matching, Self-Consistency, Test-time Scaling

## TL;DR

This paper proposes SORL, which leverages the self-consistency property of shortcut models to enable efficient single-stage training with variable inference steps for policy optimization in offline RL, while supporting both sequential and parallel test-time scaling.

## Background & Motivation

**Background**: Offline reinforcement learning (Offline RL) trains agents from fixed datasets without online exploration. Diffusion models and Flow Matching, as powerful generative models, can capture multimodal behavior distributions.

**Limitations of Prior Work**: Applying diffusion/flow models to offline RL faces two key challenges: (a) the iterative noise sampling process complicates policy optimization, requiring backpropagation through multiple timesteps; and (b) inference efficiency is poor due to slow multi-step generation.

**Key Challenge**: Training efficiency demands fewer denoising steps (to avoid multi-step backpropagation), whereas modeling complex distributions requires more discretization steps for expressiveness. At inference time, some applications demand fast generation (e.g., autonomous driving) while others require precise actions (e.g., surgical robots).

**Goal**: How to maintain expressiveness while achieving efficient training, and support on-demand computational scaling at inference time?

**Key Insight**: The paper introduces shortcut models—a novel class of generative models capable of producing high-quality samples under arbitrary inference budgets. The key insight is to unify denoising processes of varying step counts into a single model via self-consistency.

**Core Idea**: By exploiting the self-consistency of shortcut models, SORL decouples the policy optimization steps, regularization steps, and inference steps within a single-stage training framework, enabling efficient training and flexible inference scaling.

## Method

### Overall Architecture

SORL is built on a behavior-regularized actor-critic architecture, modeling the policy as a shortcut function $s_\theta(z_t, t, h \mid x)$ conditioned on both timestep $t$ and step size $h$. Actions are sampled via the Euler method (Algorithm 2), supporting an arbitrary number of inference steps $M^{\text{inf}}$.

### Key Designs

**1. Shortcut Model Policy Class**

- **Function**: Extends standard flow matching models to condition on step size
- **Mechanism**: The model $s_\theta(z_t, t, h \mid x)$ predicts the normalized direction from $z_t$ to $z_{t+h}$, such that $z_t + s(z_t, t, h) \cdot h \approx z_{t+h}$
- **Design Motivation**: Standard flow matching requires small step sizes for accurate inference; shortcut models learn large-step jumps to enable efficient inference

**2. Three-Component Actor Loss**

The training objective of SORL consists of three terms:

$$\mathcal{L}_\pi(\theta) = \mathcal{L}_{\text{QL}}(\theta) + \mathcal{L}_{\text{FM}}(\theta) + \mathcal{L}_{\text{SC}}(\theta)$$

**(a) Q Loss** — Policy optimization:

$$\mathcal{L}_{\text{QL}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{a^\pi \sim \pi_\theta(\cdot|x)} [-Q_\phi(x, a^\pi)]$$

Actions are sampled using at most $M^{\text{BTT}}$ inference steps (i.e., backpropagation is conducted through $M^{\text{BTT}}$ timesteps). $M^{\text{BTT}}$ is typically small (1, 2, 4, 8), keeping backpropagation efficient.

**(b) Flow Matching Loss** — Offline data regularization:

$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}[\|s_\theta(a^t, t, 1/M^{\text{disc}} \mid x) - (a^1 - a^0)\|^2]$$

This ensures that at the minimum step size, the model recovers the true drift direction $a^1 - a^0$, where $a^0 \sim \mathcal{N}(0, I)$ and $a^1 \sim \mathcal{D}$.

**(c) Self-Consistency Loss** — Step-size consistency:

$$\mathcal{L}_{\text{SC}}(\theta) = \mathbb{E}[\|s_\theta(a^t, t, 2h \mid x) - s_{\text{target}}\|^2]$$

where $s_{\text{target}} = \frac{1}{2}[s_\theta(a^t, t, h \mid x) + s_\theta(a^{t+h}, t+h, h \mid x)]$, enforcing that one large step $2h$ is equivalent to two small steps $h$.

**3. Test-Time Scaling**

- **Sequential Scaling**: Increasing the number of inference steps $M^{\text{inf}}$ (up to the training-time $M^{\text{disc}}$)
- **Parallel Scaling**: Best-of-$N$ sampling — independently sampling $N$ actions from the policy and selecting the optimal one using the $Q$ function as a verifier: $\arg\max_{a \in \{a_1, ..., a_N\}} Q(x, a)$

### Loss & Training

- The critic loss minimizes the standard Bellman error: $\mathcal{L}_Q(\phi) = (Q_\phi(x, a^1) - r - \gamma Q_{\phi}^{\text{target}}(x', a_{x'}^\pi))^2$
- Training is single-stage, requiring no distillation or two-stage pipelines
- Step size $h$ is sampled uniformly from powers of 2, and $t$ is sampled uniformly over $[0,1]$

## Key Experimental Results

### Main Results

SORL is evaluated against 10 baselines (3 Gaussian policies, 3 diffusion policies, 4 flow policies) on the OGBench task suite, covering 40 tasks and 8 seeds.

| Environment | BC | IQL | ReBRAC | IDQL | FQL | **SORL** |
|---|---|---|---|---|---|---|
| antmaze-large (5 tasks) | 11 | 53 | 81 | 21 | 79 | **89±2** |
| antmaze-giant (5 tasks) | 0 | 4 | **26** | 0 | 9 | 9±6 |
| humanoidmaze-medium (5 tasks) | 2 | 33 | 22 | 1 | 58 | **64±4** |
| humanoidmaze-large (5 tasks) | 1 | 2 | 2 | 1 | 4 | 5±2 |
| antsoccer-arena (5 tasks) | 1 | 8 | 0 | 12 | 60 | **69±2** |
| cube-single (5 tasks) | 5 | 83 | 91 | 95 | 96 | **97±1** |
| cube-double (5 tasks) | 2 | 7 | 12 | 15 | **29** | 25±3 |
| scene (5 tasks) | 5 | 28 | 41 | 46 | 56 | **57±2** |

SORL achieves the best performance in 5 out of 8 environments.

### Ablation Study

| Inference Setting | Effect |
|---|---|
| $M^{\text{inf}}=1$ (1-step inference) | Feasible but lower performance |
| $M^{\text{inf}}=2,4,8$ | Performance improves with more steps |
| $M^{\text{BTT}}=1$ + parallel scaling | Recovers optimal performance of $M^{\text{BTT}}=8$ |
| Best-of-8 + more inference steps | Generalizes beyond training-time step counts |

### Key Findings

1. Under a fixed training budget, increasing inference steps consistently improves performance (sequential scaling).
2. Reducing training compute ($M^{\text{BTT}}=1,2,4$) can be compensated by test-time scaling to recover optimal performance.
3. The number of inference steps generalizes beyond the backpropagation steps used during training.
4. SORL outperforms FQL by approximately 10 percentage points on antmaze-large (89 vs. 79) and by 9 points on antsoccer-arena (69 vs. 60).

## Highlights & Insights

1. **Elegant Decoupling Design**: By exploiting the self-consistency of shortcut models, SORL fully decouples the policy optimization steps $M^{\text{BTT}}$, discretization steps $M^{\text{disc}}$, and inference steps $M^{\text{inf}}$.
2. **Theoretical Guarantees**: Theorem 2 proves that the distribution generated by the shortcut model at any step size is close to the target distribution under the 2-Wasserstein distance, with an upper bound decomposed into discretization error + FM error + SC error.
3. **Interchangeability of Training and Inference Compute**: Less training with more inference can recover the performance of more training with less inference, offering a flexible solution for resource-constrained scenarios.
4. **Single-Stage Training**: Avoids the two-stage complexity and error accumulation associated with distillation-based approaches.

## Limitations & Future Work

1. SORL achieves only moderate performance on antmaze-giant and humanoidmaze-large; complex long-horizon planning remains a challenge.
2. The gains from parallel scaling (Best-of-N) lack theoretical guarantees, as the learned $Q$ function rather than the true reward is used as the verifier.
3. Self-consistency training introduces additional computational overhead compared to standard flow matching.
4. Adaptive inference step selection (e.g., dynamically adjusting steps based on $Q$ gradients) remains unexplored.
5. Restricting the discretization step count to powers of 2 limits the design space.

## Related Work & Insights

- **FQL (Flow Q-Learning)**: The closest baseline, which applies flow models to offline RL but requires distillation to obtain single-step inference capability; SORL unifies multi-step inference within a single framework.
- **Shortcut Models (Frans et al., 2024)**: SORL is the first work to introduce shortcut models into offline RL, representing the RL counterpart of test-time compute scaling in LLM reasoning.
- **Insights**: The self-consistency idea may generalize to online deployment of robotic policies — executing multi-step precise actions when inference resources are abundant, and single-step fast responses when resources are constrained.

## Rating

⭐⭐⭐⭐ (4/5)

- Novelty ⭐⭐⭐⭐: The combination of shortcut models and offline RL is novel, and the test-time scaling perspective is inspiring.
- Theory ⭐⭐⭐⭐: The $W_2$-distance regularization guarantee is original and rigorous.
- Experimental Thoroughness ⭐⭐⭐⭐: Comprehensive evaluation across 40 tasks, though improvements are limited in some environments.
- Writing Quality ⭐⭐⭐⭐: Single-stage training with flexible inference is practically valuable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Improved Training Technique for Shortcut Models (iSM)](improved_training_technique_for_shortcut_models.md)
- [\[ICLR 2026\] SSCP: Flow-Based Single-Step Completion for Efficient and Expressive Policy Learning](../../ICLR2026/image_generation/flow-based_single-step_completion_for_efficient_and_expressive_policy_learning.md)
- [\[ACL 2025\] R-VC: Rhythm Controllable and Efficient Zero-Shot Voice Conversion via Shortcut Flow Matching](../../ACL2025/image_generation/rvc_rhythm_voice_conversion.md)
- [\[NeurIPS 2025\] Scaling Diffusion Transformers Efficiently via μP](scaling_diffusion_transformers_efficiently_via_μp.md)
- [\[CVPR 2025\] Finite Difference Flow Optimization for RL Post-Training of Text-to-Image Models](../../CVPR2025/image_generation/finite_difference_flow_optimization_for_rl_post-training_of_text-to-image_models.md)

</div>

<!-- RELATED:END -->

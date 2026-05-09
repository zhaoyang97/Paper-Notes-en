---
title: >-
  [Paper Note] Self-Improving Embodied Foundation Models
description: >-
  [NeurIPS 2025][Reinforcement Learning][Embodied foundation models] This paper proposes a two-stage post-training framework for embodied foundation models: Stage 1 performs supervised fine-tuning via behavior cloning and steps-to-go prediction; Stage 2 leverages the resulting self-reward function and success detector for online RL self-improvement. Using only 1–3% additional data, the method achieves over 1.5× improvement in success rate and, for the first time, demonstrates a robot autonomously acquiring novel skills beyond the distribution of imitation data.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - Embodied foundation models
  - self-improvement
  - RL post-training
  - steps-to-go prediction
  - robot manipulation
date: 2026-05-08
content_hash: 8019b99f80928bb6
---

# Self-Improving Embodied Foundation Models

**Conference**: NeurIPS 2025
**arXiv**: [2509.15155](https://arxiv.org/abs/2509.15155)
**Code**: Not available
**Area**: Reinforcement Learning
**Keywords**: Embodied foundation models, self-improvement, RL post-training, steps-to-go prediction, robot manipulation

## TL;DR
This paper proposes a two-stage post-training framework for embodied foundation models: Stage 1 performs supervised fine-tuning via behavior cloning and steps-to-go prediction; Stage 2 leverages the resulting self-reward function and success detector for online RL self-improvement. Using only 1–3% additional data, the method achieves over 1.5× improvement in success rate and, for the first time, demonstrates a robot autonomously acquiring novel skills beyond the distribution of imitation data.

## Background & Motivation
Foundation models pretrained on web-scale data can be fine-tuned into low-level robot control policies (e.g., RT-2, Octo, π0), inheriting strong generalization from pretraining. However, training of embodied foundation models (EFMs) has been limited to behavior cloning (supervised learning). In contrast, the standard post-training pipeline for LLMs follows SFT → RL, and RL post-training has been shown to rapidly and substantially improve downstream performance, becoming a critical component of foundation model development.

Applying RL post-training to real-world robots introduces a unique challenge: **reward engineering** — manually designing reward functions for each manipulation task requires extensive trial and error, and measuring rewards in the physical world demands significant engineering effort. As task diversity grows, manual reward design becomes unsustainable.

**Core Idea**: Steps-to-go prediction — estimating the temporal distance remaining to task completion — serves as a bridge that naturally yields a well-shaped **data-driven reward function** and a robust **success detector**. Both inherit the generalization capability of the underlying foundation model acquired during web-scale pretraining, eliminating task-specific reward engineering and enabling a single human operator to supervise multiple autonomously training robots.

## Method

### Overall Architecture
A two-stage post-training framework:
- **Stage 1 (SFT)**: Starting from pretrained PaLI (a 3B-parameter vision-language model), jointly trains behavior cloning and steps-to-go prediction on robot imitation datasets.
- **Stage 2 (Self-Improvement)**: A frozen Stage 1 model is used to compute rewards and detect success, while a second Stage 1 model is initialized as the policy and improved via online RL (REINFORCE) through autonomous practice.

### Key Designs
1. **Steps-to-Go Prediction and Data-Driven Reward Function**:

    - Stage 1 trains two objectives: the behavior cloning loss $\mathcal{L}_\text{BC} = -\mathbb{E}[\log p_\text{action}^\text{EFM}(a_t | o_t, g_{t'})]$ and the steps-to-go loss $\mathcal{L}_\text{steps-to-go} = -\mathbb{E}[\log p_\text{steps-to-go}^\text{EFM}(t'-t | o_t, g_{t'})]$.
    - The temporal distance is defined as $d(o, g) := \mathbb{E}_{p_\text{steps-to-go}}[\text{steps-to-go}]$.
    - The reward function is the **temporal distance difference**: $r(o_t, a_t, o_{t+1}, g) := d(o_t, g) - d(o_{t+1}, g)$.
    - Mathematical derivation shows this reward implicitly implements potential-based reward shaping:
    $$r = \underbrace{(1-\gamma) \cdot V^\mu(o_{t+1}, g)}_{\text{core reward}} + \underbrace{[\gamma \cdot V^\mu(o_{t+1}, g) - V^\mu(o_t, g)]}_{\text{reward shaping}}$$
    - Here $V^\mu$ is the value function of the dataset policy $\mu$. The core reward term provides higher rewards in regions where $\mu$ performs well; the shaping term serves as a baseline to reduce variance.
    - **Design Motivation**: Entirely data-driven, requiring no manual design, and automatically inherits the generalization capability of the pretrained model.

2. **Steps-to-Go Success Detector**:

    - Success is determined by $\text{success}(o, g) := \mathbb{1}[d(o, g) \leq s]$, where $s$ is a small step-count threshold.
    - More robust than explicitly training a binary success classifier, remaining reliable even under low data regimes.
    - **Design Motivation**: Terminating successful episodes prevents the collection of redundant data where the robot remains stationary in a success state.

3. **On-Policy Self-Improvement Loop**:

    - Uses the REINFORCE policy gradient: $-c \cdot R_t \cdot \log p_\text{action}^\text{EFM}(a_t | o_t, g)$.
    - Monte Carlo returns $R_t = \sum_{i=t}^T \gamma^{i-t} \cdot r(o_i, a_i, o_{i+1}, g)$ with discount factor $\gamma = 0.9$.
    - No experience replay, no value function training — eliminating two vertices of the deadly triad (off-policy + bootstrapping).
    - After collecting sufficient data per iteration, $N$ policy updates are performed and the buffer is cleared.
    - A single human operator can supervise multiple robot stations simultaneously, intervening manually only in anomalous cases.
    - **Design Motivation**: Maximize training stability and reliability, laying the groundwork for real-world deployment.

### Loss & Training
- Stage 1: Joint optimization of $\mathcal{L}_\text{BC} + \mathcal{L}_\text{steps-to-go}$ (with optional auxiliary tasks such as instruction prediction).
- Stage 2: REINFORCE loss with weight coefficient $c = 5\times 10^{-2}$.
- A separate frozen Stage 1 model is used in Stage 2 for reward computation to prevent reward signal drift during training.
- The PaLI model follows the RT-2 parameterization, tokenizing continuous actions as language tokens.

## Key Experimental Results

### Main Results

| Domain | Data | BC (Stage 1) | Self-Improvement (Stage 2) | Extra Data |
|------|--------|-------------|---------------------------|------------|
| sim LanguageTable 10% | 10% imitation | 25% | 60% | +1% episodes |
| sim LanguageTable 20% | 20% imitation | 35% | 70% | +1.5% episodes |
| sim LanguageTable 80% | 80% imitation | 45% | **75%** | +2% episodes |
| real LanguageTable 20% | 20% imitation | ~62% | **~88%** | +3% episodes |
| real LanguageTable 80% | 80% imitation | ~63% | **~87%** | +3% episodes |
| sim Aloha 5K | 5K imitation | ~40% | ~65% | +2.5K episodes |
| sim Aloha 10K | 10K imitation | ~55% | **~75%** | +2K episodes |

### Ablation Study

| Reward Model | 10% Data | 20% Data | 80% Data | Notes |
|---------|---------|---------|---------|------|
| PaLI (multimodal pretraining) | **60%** | **70%** | **75%** | Best |
| Uni-PaLI (unimodal pretraining) | ~40% | ~50% | ~65% | Significantly below PaLI |
| Scratch (random init) | High variance | High variance | ~55% | Complete failure at low data |

| BananaTable Generalization | Success Rate | Notes |
|---------------------|--------|------|
| Before Self-Improvement | ~63% | Policy has never seen bananas |
| After Self-Improvement | **~85%** | 8 hours of autonomous practice |

### Key Findings
- Self-improvement is far more sample-efficient than scaling imitation data: 10% data + 1% autonomous practice surpasses 20% imitation-only, which in turn surpasses 80% imitation-only in multiple settings.
- Multimodal pretraining is critical for self-improvement: PaLI with 20% data achieves better self-improvement than Uni-PaLI with 80% data.
- The BananaTable experiment demonstrates **behavioral generalization** (not merely semantic generalization): the policy learns novel manipulation strategies for pushing bananas (from the middle or tip), exceeding behavioral patterns present in the imitation data.
- In Real2Sim transfer, only 3% additional data raises target-domain performance from 22% to 59%.

## Highlights & Insights
- **First demonstration of RL post-training for robot foundation models**: Drawing on the LLM SFT→RL paradigm, the method creatively resolves the reward engineering challenge via steps-to-go. The elegance lies in the reward function naturally inheriting the foundation model's generalization — the same model serves as both policy and reward source.
- **Significance of the BananaTable experiment**: Unlike prior semantic generalization (e.g., RT-2 executing familiar actions in novel contexts), BananaTable demonstrates **behavioral generalization** — the policy acquires entirely new manipulation skills. This suggests that web-scale pretraining combined with online self-improvement can unlock behavioral repertoires that imitation learning alone can never cover.

## Limitations & Future Work
- Performance degrades after the self-improvement peak, and effective early stopping or adaptive regularization mechanisms are absent.
- Only on-policy REINFORCE is used without data reuse; off-policy methods could potentially reduce the required robot-hours further.
- Steps-to-go estimation is inadequate for out-of-distribution failure states, as imitation data contains no failure recovery trajectories.
- The success detector, while robust, relies on a fixed threshold and cannot support fine-grained evaluation of partial success.
- Validation is limited to bimanual manipulation and tabletop pushing; extension to legged locomotion or more complex long-horizon multi-step tasks remains unexplored.

## Related Work & Insights
- **vs. RT-2**: RT-2 is the direct equivalent of this paper's Stage 1 (BC fine-tuning of a VLM); this work adds steps-to-go prediction and RL post-training on top, yielding substantial performance gains.
- **vs. RoboCat**: RoboCat uses hindsight relabeling with iterative BC to improve policies, but hindsight-relabeled supervised learning has known failure modes. This work instead applies explicit RL optimization.
- **vs. Code-as-Rewards**: LLM-generated reward code requires iterative refinement, is difficult to measure in physical environments, and requires a separate success detector, making it unsuitable for general-purpose robot learning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First validation of foundation model RL post-training on real robots; the use of steps-to-go as a bridge is an elegant design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across simulation and real environments, two platforms, multiple data scales, pretraining ablations, domain transfer, and behavioral generalization.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, with mathematical and visual intuition presented in tandem; experimental narrative builds progressively.
- Value: ⭐⭐⭐⭐⭐ Establishes a systematic SFT→RL post-training pathway for robot foundation models; the behavioral generalization shown in BananaTable carries paradigm-level significance.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Foundation Models as World Models: A Foundational Study in Text-Based GridWorlds](foundation_models_as_world_models_a_foundational_study_in_text-based_gridworlds.md)
- [\[NeurIPS 2025\] Exploration with Foundation Models: Capabilities, Limitations, and Hybrid Approaches](exploration_with_foundation_models_capabilities_limitations_and_hybrid_approache.md)
- [\[NeurIPS 2025\] FedRAIN-Lite: Federated Reinforcement Algorithms for Improving Idealised Numerical Weather and Climate Models](fedrain-lite_federated_reinforcement_algorithms_for_improving_idealised_numerica.md)
- [\[NeurIPS 2025\] Communicating Plans, Not Percepts: Scalable Multi-Agent Coordination with Embodied World Models](communicating_plans_not_percepts_scalable_multi-agent_coordination_with_embodied.md)
- [\[ICLR 2026\] Self-Improving Skill Learning for Robust Skill-based Meta-Reinforcement Learning](../../ICLR2026/reinforcement_learning/self-improving_skill_learning_for_robust_skill-based_meta-reinforcement_learning.md)

<!-- RELATED:END -->

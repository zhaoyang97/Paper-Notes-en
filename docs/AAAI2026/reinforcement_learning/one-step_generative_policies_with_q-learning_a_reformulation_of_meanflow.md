---
title: >-
  [Paper Note] One-Step Generative Policies with Q-Learning: A Reformulation of MeanFlow
description: >-
  [AAAI 2026][Reinforcement Learning][Offline reinforcement learning] This paper reformulates MeanFlow from visual generation into a generative policy for offline RL. It proposes a residual-form direct noise-to-action mapping that achieves expressive one-step sampling and enables stable joint optimization with a Q-function in a single training stage, achieving strong performance across 73 tasks on OGBench and D4RL.
tags:
  - AAAI 2026
  - Reinforcement Learning
  - Offline reinforcement learning
  - generative policy
  - MeanFlow
  - one-step sampling
  - Q-learning
date: 2026-05-08
content_hash: 4554eed98679a622
---

# One-Step Generative Policies with Q-Learning: A Reformulation of MeanFlow

**Conference**: AAAI 2026
**arXiv**: [2511.13035](https://arxiv.org/abs/2511.13035)
**Code**: [https://github.com/HiccupRL/MeanFlowQL](https://github.com/HiccupRL/MeanFlowQL)
**Area**: Reinforcement Learning / Offline RL
**Keywords**: Offline reinforcement learning, generative policy, MeanFlow, one-step sampling, Q-learning

## TL;DR
This paper reformulates MeanFlow from visual generation into a generative policy for offline RL. It proposes a residual-form direct noise-to-action mapping that achieves expressive one-step sampling and enables stable joint optimization with a Q-function in a single training stage, achieving strong performance across 73 tasks on OGBench and D4RL.

## Background & Motivation

**Background**: Offline RL learns policies from fixed datasets and faces a trade-off between expressiveness and efficiency. Gaussian policies are fast but cannot model multimodal action distributions; flow/diffusion policies are expressive but require multi-step iterative sampling, and combining them with Q-learning via backpropagation through time (BPTT) leads to training instability.

**Limitations of Prior Work**: Existing solutions adopt a two-stage distillation pipeline—first training a multi-step generative policy via behavior cloning, then distilling it into a one-step policy for joint optimization with Q-values. However, distillation introduces an expressiveness bottleneck and increases training complexity. Directly applying MeanFlow to RL causes out-of-bound actions during early training that require clipping, leading to inconsistency between policy outputs and Bellman targets and unstable training.

**Key Challenge**: A policy is needed that simultaneously possesses the strong multimodal modeling capacity of flow models and the one-step sampling and stable Q-learning compatibility of Gaussian policies—a contradiction within prior frameworks.

**Goal**: Design a generative policy supporting one-step noise-to-action generation that can be jointly trained directly with a Q-function in a single stage without distillation.

**Key Insight**: MeanFlow achieves one-step sampling by modeling a mean velocity field, but its two-step inference (velocity estimation → velocity integration) causes out-of-bound actions in RL. Reformulating it in residual form as $g(a_t,b,t) = a_t - u(a_t,b,t)$ merges velocity estimation and action generation into a single network forward pass.

**Core Idea**: Merge MeanFlow's two-step process (estimate velocity → integrate to obtain action) into a one-step residual mapping $g_\theta$, combined with appropriate initialization strategies (zero initialization / small-variance Kaiming initialization) to ensure outputs remain within valid bounds during early training, while preserving expressiveness via the Universal Approximation Theorem (UAT).

## Method

### Overall Architecture
Given state $s$ and Gaussian noise $e \sim \mathcal{N}(0,I)$, a single forward pass generates action $\hat{a} = g_\theta(e, b=0, t=1) = e - u_\theta(e, b=0, t=1)$. The training objective combines a MeanFlow Identity loss (behavior cloning) and Q-value maximization (policy improvement).

### Key Designs

1. **Residual MeanFlow Policy Reformulation**:

    - Function: Enables a differentiable one-step noise-to-action mapping.
    - Mechanism: Define $g(a_t,b,t) = a_t - u(a_t,b,t)$, where $u$ is the MeanFlow mean velocity field. At $b=0, t=1$ this reduces to $g(e,0,1) = e - u(e,0,1)$—i.e., one-step generation. The key distinction is using $a_t$ (a data-noise interpolation) rather than pure noise $\epsilon$ as input; UAT guarantees that a sufficiently large MLP allows $g_\theta$ to approximate any continuous mapping.
    - Design Motivation: The naive form $a = \epsilon - u(\epsilon, b, t)$ fails to fit multimodal distributions in toy experiments. Using the interpolated $a_t$ as input preserves the conditional probability path structure of flow matching.

2. **MeanFlow Identity Training Loss**:

    - Function: Trains the mean velocity field without explicit velocity integration.
    - Mechanism: $\mathcal{L}_{MFI}(\theta) = \mathbb{E}||g_\theta(a_t,b,t) - \text{sg}(g_{tgt})||_2^2$, where the target $g_{tgt}$ is derived from the MeanFlow Identity. A stop-gradient operator prevents mode collapse. During training, $(s,a)$ is sampled from data, $e$ from a Gaussian, $a_t = (1-t)a + te$ is constructed, and $g_\theta$ is optimized to satisfy the MeanFlow identity.
    - Design Motivation: Directly leverages the theoretical framework of MeanFlow, avoiding the instability of ODE solvers.

3. **Joint Optimization with Q-Learning and Practical Enhancements**:

    - Function: Simultaneously performs behavior cloning and policy improvement in a single training stage.
    - Mechanism: Total objective = MFI loss (BC regularization) + Q-value maximization + adaptive BC regularization weight. Value-guided rejection sampling is additionally introduced to improve inference quality—multiple noise samples are drawn and the action with the highest Q-value is selected.
    - Design Motivation: One-step mapping allows Q-value gradients to backpropagate directly to policy parameters (no BPTT), yielding stable and efficient training.

### Loss & Training
$\mathcal{L}_\pi = -Q_\phi(s, g_\theta(e,0,1)) + \alpha \cdot \mathcal{L}_{MFI}$. The critic is trained with standard Bellman error. $\alpha$ is adjusted adaptively.

## Key Experimental Results

### Main Results

| Method | OGBench (73 tasks avg) | D4RL avg | Inference Steps | Training Stages |
|---|---|---|---|---|
| Gaussian (SAC-style) | Moderate | Moderate | 1 | Single |
| Diffusion Policy | Competitive | Competitive | Multi | Two |
| Flow Policy + Distillation | Competitive | Competitive | 1 | Two |
| **MeanFlowQL** | **Strong** | **Strong** | **1** | **Single** |

### Ablation Study

| Configuration | Performance | Notes |
|---|---|---|
| Original MeanFlow (two-step inference) | Unstable training | Out-of-bound actions + clipping issues |
| Naive residual form | Underfitting | Cannot model multimodal distributions |
| Corrected residual form (Ours) | Best | Maintains expressiveness + training stability |
| Without rejection sampling | Slightly lower | Sampling quality affects performance |
| Without adaptive BC regularization | Slightly lower | BC-Q balance is important |

### Key Findings
- The choice of residual form is critical—the naive form completely fails to fit multimodal distributions in toy experiments.
- Single-stage training is simpler than two-stage distillation and yields a more expressive final policy.
- Value-guided rejection sampling provides high returns at low inference cost.
- Performance is stable across 73 tasks and remains competitive in the offline-to-online setting.

## Highlights & Insights
- **Elegant Transfer of MeanFlow from Generation to RL**: A one-step method originally designed for image generation is reformulated as an RL policy, resolving the compatibility issue between flow policies and Q-learning.
- **In-Depth Analysis of the Residual Form**: Rather than simply proposing one solution, the paper systematically analyzes multiple reformulation variants and rigorously explains why only a specific form is effective.
- **Elimination of Two-Stage Training Complexity**: Single-stage end-to-end training is cleaner than distillation and avoids the expressiveness loss it introduces.

## Limitations & Future Work
- The approach relies on theoretical assumptions of MeanFlow (e.g., smoothness of the velocity field); applicability to very high-dimensional action spaces remains to be verified.
- Value-guided rejection sampling increases inference cost, albeit only linearly.
- Only offline RL is validated; applicability to purely online RL settings is unexplored.
- Integration with world models could further improve learning from offline data alone.

## Related Work & Insights
- **vs. IDQL/SfBC (two-stage distillation)**: Those methods require BC pre-training followed by distillation; this work completes training in a single stage with better expressiveness.
- **vs. Diffusion Policy (e.g., DDPO)**: Multi-step inference with BPTT is unstable; this work uses one-step inference with full differentiability.
- **vs. Gaussian Policy**: One-step but unable to model multimodal distributions; this work retains one-step sampling while achieving flow-level expressiveness.
- The approach of reformulating MeanFlow as an RL policy generalizes to other scenarios requiring one-step sampling (e.g., real-time control).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Introduces MeanFlow into RL and resolves compatibility issues; residual reformulation analysis is thorough.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 73 tasks spanning OGBench and D4RL, covering both offline and offline-to-online settings.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear and theoretical derivations are complete.
- Value: ⭐⭐⭐⭐⭐ Addresses a core bottleneck in combining generative policies with Q-learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Offline Reinforcement Learning with Generative Trajectory Policies](../../ICLR2026/reinforcement_learning/offline_reinforcement_learning_with_generative_trajectory_policies.md)
- [\[AAAI 2026\] Explaining Decentralized Multi-Agent Reinforcement Learning Policies](explaining_decentralized_multi-agent_reinforcement_learning_policies.md)
- [\[AAAI 2026\] ManiLong-Shot: Interaction-Aware One-Shot Imitation Learning for Long-Horizon Manipulation](manilong-shot_interaction-aware_one-shot_imitation_learning_for_long-horizon_man.md)
- [\[AAAI 2026\] CHDP: Cooperative Hybrid Diffusion Policies for RL in Parametric Environments](chdp_cooperative_hybrid_diffusion_policies_for_reinforcement_learning_in_paramet.md)
- [\[NeurIPS 2025\] CORE: Constraint-Aware One-Step Reinforcement Learning for Simulation-Guided Neural Network Accelerator Design](../../NeurIPS2025/reinforcement_learning/core_constraint-aware_one-step_reinforcement_learning_for_simulation-guided_neur.md)

</div>

<!-- RELATED:END -->

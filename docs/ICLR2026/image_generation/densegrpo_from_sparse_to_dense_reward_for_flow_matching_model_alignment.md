---
title: >-
  [Paper Note] DenseGRPO: From Sparse to Dense Reward for Flow Matching Model Alignment
description: >-
  [ICLR 2026][Image Generation][GRPO] This paper addresses the sparse reward problem in Flow Matching + GRPO alignment by estimating step-wise reward gains as dense rewards via ODE denoising rollouts of intermediate latent…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "GRPO"
  - "dense reward"
  - "flow matching"
  - "human preference alignment"
  - "exploration calibration"
date: 2026-05-08
content_hash: 9b6c2957d01518fd
---

# DenseGRPO: From Sparse to Dense Reward for Flow Matching Model Alignment

**Conference**: ICLR 2026
**arXiv**: [2601.20218](https://arxiv.org/abs/2601.20218)  
**Code**: None  
**Area**: Diffusion Models / RLHF Alignment
**Keywords**: GRPO, dense reward, flow matching, human preference alignment, exploration calibration

## TL;DR
This paper addresses the sparse reward problem in Flow Matching + GRPO alignment by estimating step-wise reward gains as dense rewards via ODE denoising rollouts of intermediate latents, and adaptively calibrating per-timestep noise injection in the SDE sampler based on dense rewards to regulate exploration. The method outperforms Flow-GRPO on three tasks: human preference alignment, compositional generation, and text rendering.

## Background & Motivation

**Background**: GRPO has become the dominant method for human preference alignment of flow matching models (Flow-GRPO, DanceGRPO), converting deterministic ODE samplers into SDE samplers to enable stochastic exploration and optimizing the policy with group-normalized reward signals.

**Limitations of Prior Work**: Existing GRPO methods compute a single sparse reward $R^i$ only at the end of the trajectory (the final generated image) and apply it uniformly to all intermediate denoising steps. Since $R^i$ reflects the cumulative contribution of all $T$ denoising steps, assigning it to individual steps introduces a feedback–contribution mismatch.

**Key Challenge**: Mismatch between global trajectory-level rewards and local step-level contributions. Additionally, the uniform noise level in the SDE sampler cannot adapt to the time-varying nature of the denoising process — some timesteps may suffer from excessive exploration (all samples receiving negative rewards), while others are insufficiently explored.

**Goal**: (a) Estimate dense rewards (step-wise contributions) for each denoising step; (b) Calibrate the exploration space at each timestep based on dense rewards.

**Key Insight**: Leveraging the determinism of ODEs — given an intermediate latent $x_t$, the ODE denoising trajectory is uniquely determined, enabling ODE rollout from intermediate latents to obtain clean images that can be scored by a reward model, yielding per-step reward gains.

**Core Idea**: The contribution of each step = expected reward after that step − expected reward before that step (reward gain), estimated via ODE rollout.

## Method

### Overall Architecture

DenseGRPO introduces two improvements over Flow-GRPO:
- **Input**: Flow matching model + reward model + text prompts
- **SDE Sampling**: Sample $G$ trajectories using timestep-adaptive noise levels $\psi(t)$
- **Dense Reward Estimation**: Perform ODE rollout from each intermediate latent along every trajectory to obtain a clean image, scored by the reward model as $R_t^i$
- **Reward Gain**: $\Delta R_t^i = R_{t-1}^i - R_t^i$ serves as the dense reward for each step
- **GRPO Training**: Replace sparse $R^i$ with $\Delta R_t^i$ for group normalization and policy optimization

### Key Designs

1. **Step-wise Dense Reward Estimation**:

    - **Function**: Evaluate the contribution of each denoising step.
    - **Mechanism**: Apply $n$-step ODE denoising to intermediate latent $x_t^i$ to obtain a clean latent $\hat{x}_{t,0}^i = \text{ODE}_n(x_t^i, c)$, decode it into an image, and score it with the reward model $R_t^i = \mathcal{R}(\hat{x}_{t,0}^i, c)$. The step-level dense reward is $\Delta R_t^i = R_{t-1}^i - R_t^i$.
    - **Design Motivation**: The determinism of ODEs guarantees a one-to-one mapping from latent to clean image, eliminating the need to train a separate critic. Existing reward models are designed for clean images and can be used directly without adaptation. Experiments show that multi-step ODE rollout ($n=t$) yields more accurate estimates than single-step rollout.

2. **Exploration Space Calibration**:

    - **Function**: Adaptively adjust the SDE noise level for each timestep.
    - **Mechanism**: Replace the uniform noise parameter $a$ in the SDE sampler with timestep-specific $\psi(t)$. The calibration objective is to balance positive and negative rewards at each timestep (approximately equal counts), achieved through simple incremental adjustments: increase $\psi(t)$ if rewards are balanced (expanding exploration), otherwise decrease it (contracting exploration).
    - **Design Motivation**: A uniform noise level $a=0.7$ causes all samples near terminal timesteps (e.g., timestep=2) to receive negative rewards — the exploration space is too large, producing only poor samples from which no effective policy can be learned. Adaptive adjustment ensures meaningful positive and negative signals at every timestep.

### Loss & Training

- Standard GRPO loss (Eq. 4), with advantages computed from dense reward gains (Eq. 10)
- $T=10$ sampling steps, $G=24$ group size, 512 resolution
- KL coefficient $\beta$: 0.04 for compositional generation and text rendering, 0.01 for human preference alignment
- ODE rollout steps $n=t$ (full rollout at each step)

## Key Experimental Results

### Main Results (Based on SD3.5-M)

| Task | Metric | Flow-GRPO | Flow-GRPO+CoCA | **DenseGRPO** |
|------|--------|-----------|----------------|---------------|
| Compositional Generation | GenEval ↑ | 0.95 | 0.96 | **0.97** |
| Text Rendering | OCR Acc. ↑ | 0.92 | 0.93 | **0.95** |
| Human Preference | PickScore ↑ | 23.31 | 23.63 | **24.64** |
| Human Preference | Aesthetic ↑ | 5.92 | 6.22 | **6.35** |
| Human Preference | ImageReward ↑ | 1.28 | 1.32 | **1.41** |

### Ablation Study

| Configuration | PickScore | Note |
|---------------|-----------|------|
| DenseGRPO (full, n=t) | **Best** | Multi-step ODE rollout |
| n=2 ODE steps | Second best | 2-step approximation |
| n=1 ODE step | Worse than Flow-GRPO | Inaccurate single-step estimation |
| Without exploration calibration | Second best | Uniform noise |
| Flow-GRPO (baseline) | Baseline | Sparse reward |

### Key Findings
- **Human preference alignment shows the largest gains**: PickScore improves from 23.31 to 24.64 (+1.33), indicating that dense rewards are most beneficial for tasks requiring fine-grained adjustment.
- ODE rollout depth is critical: single-step ODE ($n=1$) performs worse than Flow-GRPO, as single-step denoising drifts from the clean image domain, leading to inaccurate reward model evaluations.
- Exploration space calibration further improves performance, especially at timesteps near the terminal end.
- Generalizability is confirmed on FLUX.1-dev and 1024 resolution.
- Reward hacking risk exists: dense rewards optimize more precisely toward the reward model but are also more prone to overfitting it.

## Highlights & Insights
- **Elegant exploitation of ODE determinism**: No critic network needs to be trained; ODE rollout combined with existing reward models suffices to estimate intermediate step rewards. This approach is simple and practical — any reward model capable of evaluating clean images can be seamlessly integrated.
- **Dense rewards expose exploration space issues**: Without dense rewards, the pathological behavior of uniform noise settings would remain undetected. Dense rewards thus serve not only as better training signals but also as diagnostic tools.
- **The reward gain formulation for step-level credit assignment** is transferable to settings with longer trajectories, such as video generation.

## Limitations & Future Work
- ODE rollout incurs significant additional computational cost (performing $t$-step ODE denoising + VAE decoding + reward model forward pass at each step), potentially making training several times more expensive than Flow-GRPO.
- Dense rewards increase the risk of reward hacking (acknowledged by the authors in the supplementary material).
- The exploration calibration update rule is overly simplistic ($\varepsilon$-level adjustments) and may lack robustness across different tasks and models.
- Validation is limited to flow matching models; transferability to traditional diffusion models such as DDPM remains unexplored.

## Related Work & Insights
- **vs. Flow-GRPO**: DenseGRPO is a direct improvement within the same framework, replacing sparse rewards with dense rewards. PickScore improves by 1.33 on the human preference task.
- **vs. CoCA**: CoCA also attempts step-level signal assignment but still allocates trajectory-level rewards by latent similarity, leaving the optimization mismatch unresolved. DenseGRPO's reward gain formulation provides a more principled solution.
- **vs. TempFlow-GRPO**: TempFlow-GRPO provides step-level rewards via trajectory branching but still optimizes steps with trajectory-level signals. DenseGRPO's ODE rollout approach yields more accurate step-level estimates.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — ODE rollout-based dense reward estimation is a novel and practical idea; exploration calibration demonstrates clear insight.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers three tasks, ablation studies, FLUX.1-dev validation, and high-resolution scaling.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem motivation is clearly articulated; Fig. 3 intuitively visualizes the exploration space issue.
- **Value**: ⭐⭐⭐⭐⭐ — A significant improvement to the GRPO alignment paradigm; dense reward is the right direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GLASS Flows: Efficient Inference for Reward Alignment of Flow and Diffusion Models](glass_flows_reward_alignment_diffusion.md)
- [\[NeurIPS 2025\] Value Gradient Guidance for Flow Matching Alignment](../../NeurIPS2025/image_generation/value_gradient_guidance_for_flow_matching_alignment.md)
- [\[ICLR 2026\] Multi-agent Coordination via Flow Matching](multi-agent_coordination_via_flow_matching.md)
- [\[ICLR 2026\] Laplacian Multi-scale Flow Matching for Generative Modeling](laplacian_multi-scale_flow_matching_for_generative_modeling.md)
- [\[ICLR 2026\] Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning](flow_matching_with_injected_noise_for_offline-to-online_reinforcement_learning.md)

</div>

<!-- RELATED:END -->

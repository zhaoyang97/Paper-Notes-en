---
title: >-
  [Paper Note] Task-Agnostic Pre-training and Task-Guided Fine-tuning for Versatile Diffusion Planner
description: >-
  [ICML2025][Image Generation][Diffusion Models] Proposes the SODP framework: first pre-trains a diffusion planner with a large dataset of sub-optimal multi-task trajectories without reward labels, then quickly adapts to downstream tasks using policy-gradient-based RL fine-tuning, and introduces BC regularization to prevent performance collapse, achieving a 60.56% success rate (SOTA) on Meta-World 50 tasks.
tags:
  - "ICML2025"
  - "Image Generation"
  - "Diffusion Models"
  - "Multi-Task RL"
  - "Pre-training and Fine-tuning"
  - "Sub-optimal Data"
  - "Policy Gradient"
  - "Behavior Cloning Regularization"
date: 2026-05-08
content_hash: 5dbf49cac65c8b79
---

# Task-Agnostic Pre-training and Task-Guided Fine-tuning for Versatile Diffusion Planner

**Conference**: ICML2025  
**arXiv**: [2409.19949](https://arxiv.org/abs/2409.19949)  
**Code**: To be confirmed  
**Area**: Diffusion Models / Reinforcement Learning / Multi-Task Planning  
**Keywords**: Diffusion Models, Multi-Task RL, Pre-training and Fine-tuning, Sub-optimal Data, Policy Gradient, Behavior Cloning Regularization

## TL;DR
Proposes the SODP framework: first pre-trains a diffusion planner with a large dataset of sub-optimal multi-task trajectories without reward labels, then quickly adapts to downstream tasks using policy-gradient-based RL fine-tuning, and introduces BC regularization to prevent performance collapse, achieving a 60.56% success rate (SOTA) on Meta-World 50 tasks.

## Background & Motivation
- **Difficulty in Multi-Task RL**: Traditional methods assume a shared latent structure across tasks, making it difficult to capture multimodal optimal behavior distributions. Although diffusion models excel at modeling complex distributions, existing approaches either rely on costly expert demonstration data or require reward labels for each task.
- **Core Problem**: Can a versatile diffusion planner be learned from a large volume of low-quality, sub-optimal trajectories without reward annotations, enabling rapid adaptation to various downstream tasks?
- **Motivation**: Drawing an analogy to the pre-training + RLHF paradigm of LLMs, this work introduces the concept of "pre-training on a large amount of sub-optimal data followed by fine-tuning on a small amount of task-specific rewards" into the diffusion planner for multi-task RL.

## Method

### Overall Architecture
SODP = **Sub-Optimal data → Diffusion Planner**, which consists of pre-training and fine-tuning stages.

### Stage One: Unconditional Pre-training
An unconditional diffusion model is trained on a multi-task mixed dataset $\mathcal{D} = \cup_{i=1}^{N} \mathcal{D}_i$ to predict future action sequences based on historical states:

$$\max_{\theta} \mathbb{E}_{(\mathbf{s}_t, \mathbf{a}_t) \sim \cup_i \mathcal{D}_i} \left[ \log p_\theta(\mathbf{a}_t^0 | \mathbf{s}_t) \right]$$

where $\mathbf{a}_t^0 = (a_t, a_{t+1}, \ldots, a_{t+H-1})$ is an action sequence of length $H$, and $\mathbf{s}_t$ is a historical state sequence of length $T_o$.

**Pre-training Loss** (standard denoising loss):

$$\mathcal{L}_{\text{pre-train}}(\theta) = \mathbb{E}_{k, \epsilon, (\mathbf{s}_t, \mathbf{a}_t^0) \sim D} \left[ \| \epsilon - \epsilon_\theta(\mathbf{a}_t^k, \mathbf{s}_t, k) \|^2 \right]$$

**Key Designs**: Focuses on shared action spaces (e.g., robot end-effector poses) to promote cross-task generalization, eliminating the need for reward labels or task descriptions.

### Stage Two: Reward-Based Policy Gradient Fine-Tuning
The denoising process is modeled as a $K$-step MDP, optimized using PPO-style importance sampling:

**Fine-Tuning Goal**: Maximize the expected cumulative reward of the downstream task $\mathcal{T}$:
$$J^{\mathcal{T}}(\theta) = \sum_t \mathbb{E}_{p_\theta(\mathbf{a}_t^0 | \mathbf{s}_t)} [r^{\mathcal{T}}(\mathbf{a}_t^0)]$$

**Policy Gradient Estimation** (with PPO clipping):

$$\mathcal{L}_{\text{Imp}}^{\mathcal{T}}(\theta) = \sum_t \mathbb{E}_{p_{\theta_{\text{old}}}} \left[ \sum_{k=1}^{K} -r^{\mathcal{T}}(\mathbf{a}_t^0) \cdot \max\left(\rho_k, \text{clip}(\rho_k, 1+\epsilon, 1-\epsilon)\right) \right]$$

where $\rho_k(\theta, \theta_{\text{old}}) = \frac{p_\theta(\mathbf{a}_t^{k-1} | \mathbf{a}_t^k, \mathbf{s}_t)}{p_{\theta_{\text{old}}}(\mathbf{a}_t^{k-1} | \mathbf{a}_t^k, \mathbf{s}_t)}$ is the importance ratio.

### BC Regularization
A behavior cloning regularization term is introduced to prevent performance collapse during fine-tuning, where the target policy $\mu$ is approximated by recent optimal experiences:

$$\mathcal{L}_{\text{BC}}(\theta) = \mathbb{E}_{k, \mathbf{a}_\mu^k \sim p_\mu} \left[ \| \epsilon(\mathbf{a}_\mu^k, k) - \epsilon_\theta(\mathbf{a}_\mu^k, k) \|^2 \right]$$

**Final Fine-Tuning Loss**: $\mathcal{L}_{\text{fine-tuning}}^{\mathcal{T}}(\theta) = \mathcal{L}_{\text{Imp}}^{\mathcal{T}}(\theta) + \lambda \mathcal{L}_{\text{BC}}(\theta)$

## Key Experimental Results

### Main Results: Meta-World 50 Tasks (MT50-rand, sub-optimal data)

| Method | Average Success Rate |
|------|-----------|
| RLPD | 10.16% |
| IBRL | 25.29% |
| Cal-QL | 35.09% |
| MTBC | 34.53% |
| MTDQL | 17.33% |
| MTDT | 42.33% |
| MTIQL | 43.28% |
| Prompt-DT | 48.40% |
| MTDIFF-P | 48.67% |
| HarmoDT-F | 57.20% |
| **SODP (Ours)** | **60.56%** |

### Efficient Fine-Tuning Verification (Only 100k steps of online fine-tuning)

| Method | Average Success Rate |
|------|-----------|
| RLPD | 7.62% |
| Cal-QL | 24.60% |
| HarmoDT-F | 57.37% |
| **SODP**| **59.26%** |

### Regularization Ablation
- No regularization: Performance rapidly degrades, and the model loses its pre-trained capability.
- KL regularization: The model gets stuck oscillating near sub-optimal regions.
- PL (Pre-training Loss) regularization: Exploration lacks direction, potentially leading to worse regions.
- **BC Regularization (Ours)**: Retains pre-trained knowledge while effectively exploring high-reward regions.

### Multi-Task Pre-training Gains
Pre-training on MT-10 and transferring to unseen tasks leads to success rates such as: drawer-open 34.7%, plate-slide-side 55.3%, and handle-pull-side 71.3%, whereas pre-training on a single task fails completely (0%).

## Highlights & Insights
1. **Paradigm Innovation**: Successfully transfers the "pre-training + RLHF" paradigm from the LLM domain to multi-task RL diffusion planners, extracting value even from low-quality data.
2. **Ingenious BC Regularization Design**: Uses recent optimal experiences to approximate the target policy, which simultaneously prevents forgetting and guides exploration, outperforming KL/PL regularization.
3. **High Data Efficiency**: Achieves 59.26% success with only 100k steps of fine-tuning, with virtually no performance drop.
4. **Pre-training without Reward Labels**: Lowers the barrier and cost of multi-task data collection.

## Limitations & Future Work
1. **Validation Only in Simulation**: Both Meta-World and Adroit are simulated environments; the method has not been tested on physical robots.
2. **Online Fine-Tuning Still Requires Environment Interaction**: Each downstream task requires an online RL environment, and the sim-to-real gap for physical deployment is not discussed.
3. **Questionable Classification**: The paper actually focuses on RL planning rather than image generation; the current categorization under `image_generation` might be inaccurate.
4. **Scalability to be Verified**: The scale of 50 tasks is limited; performance on larger-scale task sets (e.g., hundreds of tasks) remains unknown.
5. **Computational Cost of Denoising Steps $K$**: PPO fine-tuning requires computing the importance ratio for each denoising step, resulting in significant computational overhead when $K$ is large.

## Rating
- Novelty: ⭐⭐⭐⭐ — The first systematic exploration of the pre-training + RL fine-tuning paradigm on diffusion planners; the BC regularization design is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive evaluations on 50 tasks with various baselines and thorough ablation studies, though it lacks physical robot verification.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, solid mathematical derivations, and a standardized notation system.
- Value: ⭐⭐⭐⭐ — Provides a viable pathway for training multi-task agents driven by low-quality data.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] LIFT: Latent Implicit Functions for Task- and Data-Agnostic Encoding](../../ICCV2025/image_generation/lift_latent_implicit_functions_for_task-_and_data-agnostic_encoding.md)
- [\[NeurIPS 2025\] Tree-Guided Diffusion Planner](../../NeurIPS2025/image_generation/tree-guided_diffusion_planner.md)
- [\[ICML 2025\] Zero-Shot Adaptation of Parameter-Efficient Fine-Tuning in Diffusion Models](zero-shot_adaptation_of_parameter-efficient_fine-tuning_in_diffusion_models.md)
- [\[NeurIPS 2025\] UtilGen: Utility-Centric Generative Data Augmentation with Dual-Level Task Adaptation](../../NeurIPS2025/image_generation/utilgen_utility-centric_generative_data_augmentation_with_dual-level_task_adapta.md)
- [\[ICML 2025\] Revisiting Diffusion Models: From Generative Pre-training to One-Step Generation](revisiting_diffusion_models_from_generative_pre-training_to_one-step_generation.md)

</div>

<!-- RELATED:END -->

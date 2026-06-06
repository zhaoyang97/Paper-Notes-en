---
title: >-
  [Paper Note] Human-assisted Robotic Policy Refinement via Action Preference Optimization
description: >-
  [NeurIPS 2025][LLM Alignment][VLA models] This paper proposes Action Preference Optimization (APO), a human-robot collaboration framework that collects interactive trajectories and applies preference alignment to VLA mod…
tags:
  - "NeurIPS 2025"
  - "LLM Alignment"
  - "VLA models"
  - "preference alignment"
  - "human-robot collaboration"
  - "robotic manipulation"
  - "adaptive reweighting"
date: 2026-05-08
content_hash: f060638606b06700
---

# Human-assisted Robotic Policy Refinement via Action Preference Optimization

**Conference**: NeurIPS 2025
**arXiv**: [2506.07127](https://arxiv.org/abs/2506.07127)  
**Code**: [GitHub](https://github.com/GeWu-Lab/Action-Preference-Optimization)  
**Area**: LLM Alignment
**Keywords**: VLA models, preference alignment, human-robot collaboration, robotic manipulation, adaptive reweighting

## TL;DR

This paper proposes Action Preference Optimization (APO), a human-robot collaboration framework that collects interactive trajectories and applies preference alignment to VLA models using binary desirability signals grounded in prospect theory and an adaptive reweighting scheme, enabling the model to learn from failures and improve iteratively.

## Background & Motivation

Vision-Language-Action (VLA) models, as foundation models for robotic deployment, rely heavily on offline expert demonstration data and lack the capacity for continuous improvement after deployment. Existing approaches face two fundamental challenges:

**Behavioral Cloning (BC)**: Cannot effectively exploit informative signals in failure trajectories, and suffers from distribution shift between expert data and interactive data when fine-tuning large-scale VLA models.

**Reinforcement Learning (RL)**: Encounters gradient instability and difficulty in generalizing value functions when training large-scale VLA models.

Furthermore, transferring mature preference alignment methods from LLMs to VLA models introduces two core technical obstacles:
- **Irreversible interactions**: The irreversibility of robotic manipulation makes it extremely difficult to collect paired positive and negative samples under identical conditions.
- **Token probability mismatch**: Autoregressive VLA models discretize continuous actions into tokens, causing a fundamental mismatch between token probabilities and true action regression loss.

## Method

### Overall Architecture

APO consists of two core components:

1. **Human-robot collaborative deployment framework**: A human operator monitors robot execution in real time, intervenes to correct difficult scenarios, and simultaneously collects interactive trajectories.
2. **Action preference optimization process**: An adaptive reweighting-based preference alignment algorithm that refines the VLA model.

The overall pipeline follows an iterative deploy-optimize loop: an initial policy $\pi_\theta^0$ is obtained via behavioral cloning on expert data, followed by repeated cycles of deployment for interactive data collection and preference optimization for model updates.

### Key Designs

**Interactive trajectory annotation**: Each action in a collected trajectory is annotated with label $c_t$:
- $c_t = 1$: action autonomously executed by the policy
- $c_t = 2$: action corrected by human intervention
- $c_t = 0$: actions within $K$ steps prior to human intervention (marked as undesirable)

**Prospect theory-based preference alignment**: Drawing on Kahneman & Tversky's prospect theory, preference learning is performed using binary desirability signals rather than paired preference data. The reward function is estimated as:

$$r_\theta(o, \hat{a}) = \log \frac{\pi_\theta(\hat{a}|o)}{\pi_{ref}(\hat{a}|o)}$$

The utility function is then defined as:

$$v(o, \hat{a}) = \begin{cases} \lambda_D \sigma(r_\theta(o,\hat{a}) - z_0) & \text{if } \hat{a} \sim \hat{a}_{desirable} \\ \lambda_U \sigma(z_0 - r_\theta(o,\hat{a})) & \text{if } \hat{a} \sim \hat{a}_{undesirable} \end{cases}$$

where $z_0 = KL(\pi_\theta || \pi_{ref})$ serves as a penalty term to prevent excessive deviation from the reference model.

**Adaptive reweighting**: To address the mismatch between token probabilities and continuous action regression, the L1 continuous action loss is computed for each sample and normalized at the batch level:

$$w_i = \frac{l_i}{\sum_{i=1}^{B} l_i}$$

The normalized weights adaptively modulate $\lambda_D$ and $\lambda_U$:
- Desirable data: $\lambda_D = 1 - e^{-\beta_D \cdot w}$ (samples with larger action prediction error receive higher weight)
- Undesirable data: $\lambda_U = e^{-\beta_U \cdot w}$ (samples closer to failure actions receive higher weight)

### Loss & Training

The final loss function is:

$$L(\pi_\theta, \pi_{ref}) = \mathbb{E}_{x,y \sim D^h}[-v(x,y)]$$

Training employs **balanced sampling**: each batch contains 50% expert actions, 25% human-intervention actions, and 25% failure actions, ensuring balanced representation across data types.

Implementation details: OpenVLA is fine-tuned using LoRA (rank=32) with a learning rate of 5e-5, batch size of 8, on 4 A100 GPUs; $K=10$ is used for labeling undesirable behaviors.

## Key Experimental Results

### Main Results

**RoboMimic simulation environment** (4 long-horizon manipulation tasks, 50 trajectories/task, 50 evaluation episodes):

| Method | Coffee | StackThree | ThreePiece | Square | Avg. |
|--------|--------|------------|------------|--------|------|
| Base policy | 44% | 46% | 44% | 28% | 40.5% |
| Dagger | 42% | 50% | 36% | 28% | 39.0% |
| DPO | 52% | 46% | 28% | 22% | 37.0% |
| KTO | 48% | 52% | 46% | 32% | 43.5% |
| **APO** | **60%** | **54%** | **46%** | **32%** | **48.0%** |

**Perturbation generalization** (only 20 interactive trajectories + 20 expert demonstrations from the original task):

| Method | Position Perturb. | Background Perturb. | Texture Perturb. | Avg. |
|--------|-------------------|---------------------|------------------|------|
| Base policy | 12% | 42% | 10% | 21.3% |
| KTO | 20% | 46% | 6% | 24.0% |
| **APO** | **26%** | **46%** | **12%** | **28.0%** |

**Real-world experiment** (block insertion task):

| Method | In-distribution | Position Perturb. | Background Perturb. | Texture Perturb. |
|--------|-----------------|-------------------|---------------------|------------------|
| Base policy | 65% | 25% | 10% | 25% |
| TPO | 75% | 40% | 20% | 45% |
| **APO** | **85%** | **55%** | **30%** | **55%** |

### Ablation Study

**Generalization across VLA architectures** (π0-FAST model):

| Method | Coffee | StackThree | Insert Square |
|--------|--------|------------|---------------|
| Base policy | 68% | 64% | 85% |
| Dagger | 64% | 66% | 85% |
| **APO** | **76%** | **74%** | **95%** |

**Lifelong learning experiment**: The model is updated every 20 interactions. APO continues to improve performance even after gains from expert data saturate, while the human intervention rate gradually decreases.

### Key Findings

1. Behavioral cloning methods fail to surpass the base policy on VLA models, primarily due to distribution shift between expert and interactive trajectories.
2. DPO, which relies on synthetic negative samples rather than real failure interactions, yields the worst performance.
3. TPO uses randomly sampled negative examples, leading to training instability.
4. Under perturbation scenarios, APO not only adapts to new settings but also improves performance on the original task, whereas other methods exhibit catastrophic forgetting.
5. APO learns to autonomously recover from failures (e.g., re-grasping, iteratively adjusting positions).

## Highlights & Insights

1. **Elegant problem decomposition**: The two core obstacles to transferring LLM preference learning to VLA models—irreversible interactions and token probability mismatch—are addressed separately via prospect theory and adaptive reweighting, respectively.
2. **Practical human-robot collaboration paradigm**: Human intervention simultaneously ensures deployment reliability and naturally generates the data required for preference learning.
3. **No paired preference data required**: Following the spirit of KTO, only binary signals (desirable/undesirable) are needed rather than pairwise comparisons, substantially reducing annotation cost.
4. **Learning from failure**: APO not only avoids undesirable actions but also actively self-corrects after encountering failures.

## Limitations & Future Work

1. Experiments are limited to autoregressive VLA models (OpenVLA and π0-FAST); generalization to regression-based or diffusion policy models remains unverified.
2. Human intervention still requires real-time monitoring; reducing the cost of human involvement is an open problem.
3. The undesirable behavior annotation window $K=10$ is fixed and may need to be dynamically adjusted per task.
4. Gains under perturbation scenarios are modest (e.g., texture perturbation improves from 10% to 12%), indicating substantial room for further robustness improvement.

## Related Work & Insights

- **KTO**: The core preference alignment idea in this work derives from KTO's prospect-theory-based optimization, augmented with adaptive reweighting.
- **DPO/RLHF**: Classic preference alignment methods face the challenge of obtaining paired data in robotic settings.
- **Grape (TPO)**: A trajectory-level preference alignment method that requires paired trajectories under identical conditions, which is infeasible in real-world deployments.
- This work originates from NLP preference alignment and provides a transferable continual learning paradigm for VLA models in embodied intelligence.

## Rating

- Novelty: ⭐⭐⭐⭐ (transfers preference alignment to VLA while resolving key technical barriers)
- Experimental Thoroughness: ⭐⭐⭐⭐ (simulation + real world, multi-task multi-perturbation, lifelong learning, cross-model validation)
- Writing Quality: ⭐⭐⭐⭐ (clear problem formulation, well-motivated)
- Value: ⭐⭐⭐⭐ (provides a practical solution for continuous post-deployment improvement of VLA models)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ORPO-Distill: Mixed-Policy Preference Optimization for Cross-Architecture LLM Distillation](orpo-distill_mixed-policy_preference_optimization_for_cross-architecture_llm_dis.md)
- [\[NeurIPS 2025\] Self Iterative Label Refinement via Robust Unlabeled Learning](self_iterative_label_refinement_via_robust_unlabeled_learning.md)
- [\[NeurIPS 2025\] SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Constrained Learning](safevla_towards_safety_alignment_of_vision-language-action_model_via_constrained.md)
- [\[NeurIPS 2025\] Strategyproof Reinforcement Learning from Human Feedback](strategyproof_reinforcement_learning_from_human_feedback.md)
- [\[NeurIPS 2025\] GVPO: Group Variance Policy Optimization for Large Language Model Post-Training](gvpo_group_variance_policy_optimization_for_large_language_model_post-training.md)

</div>

<!-- RELATED:END -->

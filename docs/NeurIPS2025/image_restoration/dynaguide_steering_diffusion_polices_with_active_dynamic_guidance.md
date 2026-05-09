---
title: >-
  [Paper Note] DynaGuide: Steering Diffusion Policies with Active Dynamic Guidance
description: >-
  [NeurIPS 2025][Image Restoration][Diffusion Policy] This paper proposes DynaGuide, which applies classifier guidance to a frozen pretrained diffusion policy at inference time via an external latent dynamics model, steering the robot toward arbitrary positive/negative goals without modifying policy weights. It achieves an average success rate of 70% on CALVIN simulation and 80% on a real robot.
tags:
  - NeurIPS 2025
  - Image Restoration
  - Diffusion Policy
  - Classifier Guidance
  - Latent Dynamics Model
  - DinoV2
  - Robot Manipulation
date: 2026-05-08
content_hash: c9152560c4bce1b6
---

# DynaGuide: Steering Diffusion Policies with Active Dynamic Guidance

**Conference**: NeurIPS 2025
**arXiv**: [2506.13922](https://arxiv.org/abs/2506.13922)
**Code**: [dynaguide.github.io](https://dynaguide.github.io)
**Area**: Image Restoration
**Keywords**: Diffusion Policy, Classifier Guidance, Latent Dynamics Model, DinoV2, Robot Manipulation

## TL;DR

This paper proposes DynaGuide, which applies classifier guidance to a frozen pretrained diffusion policy at inference time via an external latent dynamics model, steering the robot toward arbitrary positive/negative goals without modifying policy weights. It achieves an average success rate of 70% on CALVIN simulation and 80% on a real robot.

## Background & Motivation

**Background**: Diffusion Policy has become a dominant paradigm for robot manipulation, capable of learning complex multimodal behaviors. However, how to flexibly adapt behavior at deployment time for specific scenarios—i.e., "policy steering"—remains an open problem.

**Limitations of Prior Work**:
- Goal-conditioned policies require anticipating all possible guidance distributions at training time and degrade severely when encountering out-of-distribution goals at inference time;
- Sampling-based methods (e.g., GPC-Rank) draw multiple samples from the policy and select the best, but rely on the policy itself to generate goal-satisfying actions—making them ineffective for low-probability behaviors;
- Fine-tuning the policy is costly and may destroy previously learned skills.

**Key Challenge**: How can a pretrained diffusion policy be flexibly steered toward arbitrary goals (including multi-objectives and negative goals) **without modifying** its weights?

**Key Insight**: Drawing inspiration from classifier guidance in image generation—training an external dynamics model as a "classifier" that predicts the future visual outcome of an action sequence and uses gradient signals to directly modify actions during denoising.

**Core Idea**: The external dynamics model answers "what will be seen after executing this action sequence," and gradients are used to pull the predicted future closer to desired goals and push it away from negative goals. The entire process only modifies the denoising direction at inference time; policy weights remain completely unchanged.

## Method

### Overall Architecture

DynaGuide consists of two independent modules that collaborate at inference time:

- **Base Diffusion Policy** $\pi_\theta(\mathbf{a}|o_t)$: A pretrained Diffusion Policy that generates action sequences from Gaussian noise via DDIM denoising, with weights frozen;
- **Guidance Module**: Comprising a latent dynamics model $h_\theta$ and a guidance metric $\mathbf{d}$, which computes the gradient $\nabla_{\mathbf{a}^k}\mathbf{d}$ at each denoising step and adds it to the denoising signal.

The two components are fully decoupled; the guidance module can be swapped at any time without affecting the base policy.

### Key Designs

**1. Latent Dynamics Model**

- **Function**: Given the current observation $o_t$ and action sequence $\mathbf{a}$, predict the visual state $\hat{z}_{t+H}$ after $H$ steps.
- **Encoder**: A frozen DinoV2 extracts patch embeddings as visual latent representations $z_t = \phi(o_t)$, providing semantically rich and training-stable features.
- **Predictor**: A Transformer architecture that takes $(z_t, \mathbf{a})$ as input and outputs $\hat{z}_{t+H}$.
- **Training Objective**: Simple MSE regression $\mathcal{L} = \|\phi(o_{t+H}) - h_\theta(\phi(o_t), \mathbf{a})\|_2^2$.
- **Data Augmentation**: Gaussian noise following the same scheduler as inference is added to training actions, making the model robust to noisy actions during denoising.
- **Training Data**: CALVIN play data for simulation experiments; UMI open-source data plus a small number of in-environment demonstrations for real-robot experiments.

**2. Guidance Metric Design (Positive/Negative Multi-Objectives)**

The guidance condition $\mathcal{G} = \mathbf{g}^+ \cup \mathbf{g}^-$, where $\mathbf{g}^+$ is a set of desired outcome images and $\mathbf{g}^-$ is a set of outcomes to avoid. All guidance conditions are projected into the same DinoV2 space for distance computation:

$$\mathbf{d} = \log\!\left[\sum_i \exp\frac{-\|\phi(g_i^+) - \hat{z}_{t+H}\|_2^2}{\sigma}\right] - \log\!\left[\sum_j \exp\frac{-\|\phi(g_j^-) - \hat{z}_{t+H}\|_2^2}{\sigma}\right]$$

- **Log-Sum-Exp Aggregation**: Acts as a soft maximum, computing a smooth maximum over multiple guidance conditions. When some guidance images are of low quality (e.g., mismatched robot pose or scene), useful signals are not overwhelmed.
- **Positive/Negative Separation**: The first term attracts toward the goal; the second term repels from negative goals, naturally supporting multi-objective and avoidance behaviors.
- **Hyperparameter** $\sigma$: Controls aggregation sharpness; smaller $\sigma$ focuses more on the nearest guidance condition.

**3. Injecting Classifier Guidance into the Denoising Process**

At each DDIM denoising step $k$, the guidance gradient is added to the noise prediction:

$$\hat{\epsilon}(\mathbf{a}^k, o_t) = \epsilon(\mathbf{a}^k, o_t) - s\sqrt{1-\bar{\alpha}_k}\,\nabla_{\mathbf{a}^k}\mathbf{d}$$

- $s$ is the guidance strength—larger values enforce stricter adherence to the goal but may cause unsmooth trajectories.
- **Stochastic Sampling Stabilization**: Each denoising step is repeated $M$ times via MCMC sampling, preventing guidance gradients from pushing actions out of the valid distribution and enabling the use of higher $s$ values.
- Gradients are backpropagated through the dynamics model $h_\theta$ to the action space; the entire process is differentiable.

### Loss & Training

- Dynamics model: MSE regression with action noise augmentation, trained on unstructured robot interaction data.
- Guidance process: Computed purely at inference time; no additional training required.
- Base policy: Pretrained and frozen; DynaGuide is plug-and-play with any DDIM-based diffusion policy.

## Key Experimental Results

### Experimental Setup

Four scenario types are tested in the CALVIN simulation environment, along with real-robot experiments. Baselines include:

| Method | Description |
|--------|-------------|
| Base Policy | Unguided diffusion policy |
| Goal Conditioning (GC) | Policy trained with goal image conditioning |
| DynaGuide-Sampling (GPC) | Selects optimal action by sampling using the same dynamics model |
| Position Guidance (ITPS) | Steers diffusion policy using 3D coordinates |

### Main Results

| Experiment | DynaGuide | GC | GPC | Notes |
|------------|-----------|-----|-----|-------|
| ArticulatedParts (fixed objects) | **70%** | ~95% | Lower | GC performs best in-distribution; DynaGuide achieves 8.7× improvement over base policy |
| MovableObjects (randomized objects) | **Significantly outperforms GPC** | Large drop | ≈ base policy | Object randomization causes GC to fail out-of-distribution; GPC suffers from high sampling variance |
| UnderspecifiedObjectives (low-quality guidance) | **5.4× higher than GC** | <10% | Moderate | DynaGuide is most robust under random robot states and scene mismatches |
| MultiObjectives (multi-goal + negative goals) | **80%** success on positive goals | N/A | Lower | GPC has higher failure rate when avoiding negative goals |
| UnderrepresentedBehaviors (1% data) | **40%** | — | Lower | Behaviors with only 1% training data can still be activated through guidance |

### Real-Robot Experiments

Using a publicly available pretrained UMI policy (cup placement task) without modifying policy weights:

| Scenario | Success Rate | Notes |
|----------|-------------|-------|
| CupPreference (color preference) | **72.5%** | Guides selection of a specific colored cup |
| HiddenCup (occluded object) | **80%** | Guides the robot to find a partially occluded red cup |
| NovelBehavior (novel behavior) | 2× interactions | Guides touching a mouse—a behavior absent from training data |

### Key Findings

- **Plug-and-play validation**: Applied directly to an off-the-shelf real-world policy without any fine-tuning.
- **Highly robust to low-quality guidance**: LSE aggregation still extracts valid signals when robot poses or irrelevant objects in guidance conditions are mismatched.
- **Active guidance vs. passive sampling**: DynaGuide directly modifies the denoising direction, enabling activation of low-probability modes in the policy; GPC can only select from samples already generated by the policy.
- **Data efficiency**: With only 1% of target-behavior training data, DynaGuide still achieves 40% success rate.

## Highlights & Insights

- **Modular decoupling is the core advantage**: The policy and guidance module are fully independent; the same policy can be paired with different guidance modules for different tasks, greatly improving practical deployment efficiency.
- **DinoV2 as a universal state space**: The frozen visual backbone provides a stable semantic comparison space, avoiding representation drift encountered in end-to-end training.
- **Transferring classifier guidance from image generation to robotics**: This work demonstrates that guidance theory for diffusion models is equally effective in action space, opening a new path for inference-time customization of robot policies.
- **Practical value of negative goal guidance**: In real deployment, knowing "what not to do" is as important as knowing "what to do"; DynaGuide supports this natively.

## Limitations & Future Work

- Training an additional dynamics model increases system complexity.
- Guidance conditions currently only support visual observation images; richer modalities such as language or kinematic demonstrations are not supported.
- Stochastic Sampling (repeating each denoising step $M$ times) increases inference latency.
- The method can only specify "desired/undesired outcomes" and cannot finely control the process of "how" to reach the goal.
- The upper bound of guidance effectiveness is determined by the predictive quality of the dynamics model.

## Rating

- Novelty: ⭐⭐⭐⭐ Transfers classifier guidance from image generation to diffusion policies for robotics; the design of using an external dynamics model as a classifier is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five simulation experiments and three real-robot experiments with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Method derivation is clear; figures and tables are highly informative.
- Value: ⭐⭐⭐⭐ The plug-and-play property has direct practical significance for real-world robot deployment.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Activation Steering for Masked Diffusion Language Models](../../ICLR2026/image_restoration/activation_steering_for_masked_diffusion_language_models.md)
- [\[ICCV 2025\] Blind Noisy Image Deblurring Using Residual Guidance Strategy](../../ICCV2025/image_restoration/blind_noisy_image_deblurring_using_residual_guidance_strateg.md)
- [\[NeurIPS 2025\] MRO: Enhancing Reasoning in Diffusion Language Models via Multi-Reward Optimization](mro_enhancing_reasoning_in_diffusion_language_models_via_multi-reward_optimizati.md)
- [\[ICCV 2025\] Decouple to Reconstruct: High Quality UHD Restoration via Active Feature Disentanglement and Reversible Fusion](../../ICCV2025/image_restoration/decouple_to_reconstruct_high_quality_uhd_restoration_via_active_feature_disentan.md)
- [\[NeurIPS 2025\] Improving Diffusion-based Inverse Algorithms under Few-Step Constraint via Learnable Linear Extrapolation](improving_diffusion-based_inverse_algorithms_under_few-step_constraint_via_learn.md)

<!-- RELATED:END -->

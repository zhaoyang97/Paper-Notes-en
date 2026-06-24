---
title: >-
  [Paper Note] Optimizing Diffusion Models for Joint Trajectory Prediction and Controllable Generation
description: >-
  [ECCV 2024][Autonomous Driving][Diffusion Models] This paper proposes two techniques, Optimal Gaussian Diffusion (OGD) and Estimated Clean Manifold (ECM) Guidance. By optimizing the diffusion prior distribution and directly injecting guidance gradients onto the clean manifold respectively, they reduce the diffusion steps for joint trajectory prediction to 1/12 and the guided sampling steps to 1/5 of the baseline, while achieving superior performance on Argoverse 2.
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "Diffusion Models"
  - "Trajectory Prediction"
  - "Controllable Generation"
  - "Computational Efficiency"
  - "Joint Prediction"
date: 2026-05-08
content_hash: 0d42a23a29aad073
---

# Optimizing Diffusion Models for Joint Trajectory Prediction and Controllable Generation

**Conference**: ECCV 2024  
**arXiv**: [2408.00766](https://arxiv.org/abs/2408.00766)  
**Code**: [https://yixiaowang7.github.io/OptTrajDiff_Page/](https://yixiaowang7.github.io/OptTrajDiff_Page/) (with project page)  
**Area**: Autonomous Driving  
**Keywords**: Diffusion Models, Trajectory Prediction, Controllable Generation, Computational Efficiency, Joint Prediction

## TL;DR

This paper proposes two techniques, Optimal Gaussian Diffusion (OGD) and Estimated Clean Manifold (ECM) Guidance. By optimizing the diffusion prior distribution and directly injecting guidance gradients onto the clean manifold respectively, they reduce the diffusion steps for joint trajectory prediction to 1/12 and the guided sampling steps to 1/5 of the baseline, while achieving superior performance on Argoverse 2.

## Background & Motivation

Diffusion models have shown strong capabilities in autonomous driving trajectory prediction and controllable generation—they can satisfy additional constraints (such as collision avoidance and reaching specified target points) during inference through gradient guidance without retraining. However, diffusion models face two major efficiency bottlenecks:

**Limitations of Prior Work 1: Large number of reverse diffusion steps and slow inference.** Traditional diffusion models start denoising from a standard Gaussian distribution. Since the gap between the standard Gaussian and the actual data distribution is huge, a large number of steps is required to obtain high-quality samples. Autonomous driving requires real-time inference with limited onboard computational resources, and high step counts severely restrict deployment. Although fast samplers (DDIM, DPM-Solver) and distillation methods exist, the fixed standard Gaussian prior remains the fundamental bottleneck for acceleration.

**Limitations of Prior Work 2: High computational overhead of guided sampling.** Controllable generation guides the denoising process by biasing it with the gradient of a cost function. The cost function is typically defined on clean data, but the guidance is applied to noisy intermediate data. This requires an initial forward pass to estimate the clean data, followed by backpropagation through the entire network to compute gradients, consuming massive computation and GPU memory.

**Key Challenge:** While the generation quality and controllability of diffusion models are powerful, a severe conflict exists between their computational efficiency and real-time requirements.

**Key Insight:** Since the standard Gaussian prior is the efficiency bottleneck, can an "optimal" Gaussian prior that incorporates data information be used to reduce the number of diffusion steps? Since guidance gradients require backpropagation through the entire network, can gradients be injected directly on the estimated clean manifold to bypass the backpropagation?

**Core Idea:** Utilize the trajectory statistics (mean and variance) provided by a pre-trained marginal trajectory predictor to analytically compute the optimal Gaussian prior and optimal perturbation kernel, enabling high-quality diffusion in fewer steps; reformulate the guidance as a multi-objective optimization problem on the clean manifold and solve it hierarchically to avoid network backpropagation.

## Method

### Overall Architecture

The method consists of two independent but combinable modules: OGD is responsible for accelerating the reverse diffusion process, and ECM(R) is responsible for accelerating the guided sampling. Both share a diffusion model architecture based on the QCNet scenario encoder. During inference, a pre-trained QCNet is first used to generate marginal trajectory samples and likelihoods for each vehicle. Based on these, the optimal prior and reference trajectories are calculated, followed by short-step diffusion and efficient guidance.

### Key Designs

1. **Optimal Gaussian Diffusion (OGD)**:

    - **Function**: Replace the standard Gaussian distribution with an information-rich optimal Gaussian distribution as the starting point of diffusion, significantly reducing the number of reverse diffusion steps.
    - **Mechanism**: According to the KL divergence upper bound analysis, the KL divergence between the distribution learned by the diffusion model and the true distribution is bounded by two terms: the cumulative score matching error $\mathcal{G}(\mathbf{x}_\theta, T)$ and the KL divergence between the prior and the diffusion endpoint distribution $\text{KL}[p_T \| q_\phi]$. The former increases as $T$ increases, while the latter can be minimized by optimizing the prior parameters. The key conclusion is that the mean and variance of the optimal prior can be analytically expressed using the statistics of the data distribution:
      $$\boldsymbol{\mu}^* \approx \sqrt{\bar{\alpha}_T}\boldsymbol{\mu}_d, \quad \boldsymbol{\Sigma}^* \approx \bar{\alpha}_T \boldsymbol{\Sigma}_d + (1-\bar{\alpha}_T)^2 \boldsymbol{\Sigma}_p^*$$
      where $\boldsymbol{\mu}_d, \boldsymbol{\Sigma}_d$ are the mean and variance of the data distribution. For joint trajectory prediction, assuming the prior covariance is block-diagonal (vehicles are independent), only the marginal statistics of each vehicle are required.
    - **Design Motivation**: Unlike previous methods that require training an additional GAN to learn the prior, the prior of OGD can be analytically calculated at inference time for any arbitrary $T$ without extra training, and the number of steps can be flexibly adjusted. In practice, the multi-modal marginal trajectory sets and their likelihoods predicted by the pre-trained QCNet are utilized to estimate $\boldsymbol{\mu}_d$ and $\boldsymbol{\Sigma}_d$.

2. **Estimated Clean Manifold (ECM) Guidance**:

    - **Function**: Directly inject guidance gradients on the estimated clean data manifold to avoid backpropagation through the entire diffusion network.
    - **Mechanism**: Reformulate controllable generation as a multi-objective optimization problem on the clean data $\mathbf{x}_0$—the first objective (most important) is to maximize the likelihood $-\log q_\theta(\mathbf{x}_0)$, and the second objective is to minimize the guidance cost $\mathcal{J}(\mathbf{x}_0)$. A hierarchical optimization strategy is adopted over $K$ iterations: in each iteration, the current sample is first perturbed with noise to $t_k$, and then denoised in a single step using the diffusion model to obtain a high-likelihood estimate $\hat{\mathbf{x}}_0$ (optimizing the first objective). Afterwards, a gradient descent step is performed on $\hat{\mathbf{x}}_0$ to minimize the guidance cost (optimizing the second objective):
      $$\mathbf{x}_0(k-1) \leftarrow \hat{\mathbf{x}}_0(k) - \zeta \nabla_{\hat{\mathbf{x}}_0(k)} \mathcal{J}(\hat{\mathbf{x}}_0(k))$$
      Since the gradient is solved directly with respect to the clean data, network backpropagation is bypassed completely.
    - **Design Motivation**: Traditional methods computing guidance gradients for noisy intermediate data $\mathbf{x}_t$ require $\nabla_{\mathbf{x}_t} f_\theta(\mathbf{x}_t)$, which involves computing the Jacobian of the entire network, leading to huge computational load and GPU memory. The key insight of ECM is that guidance is essentially searching for samples that are simultaneously high-likelihood and low-cost on the clean manifold, which can be operated on directly.

3. **Reference Joint Trajectories (ECMR)**:

    - **Function**: Construct reference joint trajectories using combinations of marginal trajectories to warm-start the guidance process, resolving the local minima issues caused by multi-modal distributions.
    - **Mechanism**: The joint trajectory distribution is multi-modal (due to road topology and decision variables leading to multiple peaks); directing guidance directly can easily get trapped in local optima near the initial peak. ECMR constructs a candidate set $\mathcal{R} \bigotimes \hat{\mathbf{x}}_0(k)$, evaluates the guidance cost of all marginal trajectory combinations, and selects the combination with the minimum cost as the reference point for the current iteration before performing gradient updates.
    - **Design Motivation**: In a multi-modal distribution, migrating from one peak to another requires passing through low-likelihood regions, which causes numerical instability. Directly "jumping" to the vicinity of a better peak through reference trajectories avoids lengthy cross-peak trajectories.

### Loss & Training

- The diffusion model utilizes the standard DDPM noise prediction loss $\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|_2^2$, where the perturbation kernel uses the optimal $\boldsymbol{\Sigma}_p^*$ derived by OGD.
- During training, $T_{train}=100$, and during inference, $T$ can be flexibly adjusted (e.g., 40 or 70).
- DDIM is used to accelerate sampling with a step size of 10.
- Trajectories are represented using a 10-dimensional latent space and mapped to 120-dimensional real trajectories.

## Key Experimental Results

### Main Results

**Joint Trajectory Prediction (Argoverse 2):**

| Model | Diffusion Steps T | avgMinFDE₆* | avgMinADE₆* | avgMinFDE₁₂₈ | avgMinADE₁₂₈ |
|------|----------|-------------|-------------|--------------|--------------|
| VD₅₀₀ | 500 | 0.61 | 1.36 | 0.48 | 0.91 |
| VD₁₀₀ | 100 | 0.62 | 1.38 | 0.49 | 0.97 |
| **OGD** | **70** | **0.59** | **1.31** | **0.42** | **0.75** |
| OGD | 40 | 0.61 | 1.34 | 0.47 | 0.77 |

OGD surpasses the 500 steps of VD₅₀₀ in only 40 steps, and reduces avgMinFDE₁₂₈ by 12.5% at the optimal 70 steps.

**Controllable Generation (U+Deceleration Task):**

| Model | Sampling Method | DDIM Steps | minJFDE↓ | meanJFDE↓ | minJRDE↓ | meanJRDE↓ |
|------|---------|--------|---------|----------|---------|----------|
| VD₅₀₀ | SF | 50 | 0.538 | 2.339 | 0.158 | 0.500 |
| VD₅₀₀ | NNM | 50 | 0.778 | 2.913 | 0.130 | 0.309 |
| **OGD** | **ECMR** | **10** | **0.053** | **0.146** | **0.110** | **0.154** |

ECMR + OGD reduces minJFDE by 90% while using only 1/5 of the steps!

### Ablation Study

| Configuration | minJFDE | minJRDE | Step Time (ms) | GPU Memory Increment (GB) |
|------|---------|---------|------------|-------------|
| NNM | 0.724 | 0.119 | 113 | 3.21 |
| SF | 0.155 | 0.126 | 247 | 7.96 |
| ECM (Ours) | 0.075 | 0.116 | 111 | 3.21 |
| ECMR (Ours) | 0.053 | 0.110 | 116 | 3.22 |

The single-step inference time of ECM is comparable to NNM (111ms vs 113ms) but achieves far superior results; it is 2.2 times faster than SF while using only 40% of the GPU memory.

### Key Findings

- OGD performance first increases and then decreases as T decreases, with the optimal point at T=70, indicating that the benefit of reducing cumulative errors outweighs the prior approximation error.
- The reference trajectories (R) yield significant improvements on the "mean" metrics, verifying the effectiveness of resolving multi-modal local optima.
- OGD ranks 4th on the Argoverse 2 Multi-world leaderboard in terms of avgBrierMinFDE.

## Highlights & Insights

1. **Analytical Prior** is the most elegant innovation of this paper—both the optimal Gaussian prior and the optimal perturbation kernel have closed-form solutions, requiring no additional network training, with steps flexibly adjustable during inference.
2. ECM conceptualizes guidance as a multi-objective optimization problem on the clean manifold, providing clear theoretical motivation and significant practical results.
3. The clever reuse of the existing pre-trained marginal predictor (QCNet) obtains prior information and reference trajectories without increasing training costs.
4. OGD and ECM are two orthogonal acceleration modules that can be used either independently or in combination.

## Limitations & Future Work

- The performance of OGD relies on the accuracy of the marginal trajectory predictor; inaccurate marginal predictions lead to inaccurate optimized priors.
- The prior covariance is currently assumed to be block-diagonal (independent among vehicles), ignoring cross-vehicle correlations in the joint distribution.
- Direct learning of the mean and variance for joint prediction, rather than approximating from the marginals, could be explored.
- The guidance cost function is currently hand-designed (destination point distance), and the effectiveness of more complex costs (e.g., safety scores) remains to be verified.

## Related Work & Insights

- Compared to works such as DiffScene and TRACE that use traditional guided sampling, ECM achieves qualitative efficiency improvements by avoiding network backpropagation.
- Compared to the method in [51] that uses GANs to learn priors, the analytical prior of OGD is simpler, more flexible, and requires no extra training.
- Insight: For diffusion tasks with domain prior knowledge (such as marginal statistics in trajectory prediction), leveraging prior information to initialize the diffusion process could be a general acceleration strategy.

## Rating

- Novelty: ⭐⭐⭐⭐ Both the analytical prior and clean manifold guidance present theoretical innovations, but the overall framework remains within the diffusion model paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Large-scale validation on Argoverse 2, leaderboard performance, and multi-metric ablation.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivations and rigorous formulas, though with slightly heavy notation.
- Value: ⭐⭐⭐⭐ Significant efficiency improvements with practical implications for deploying diffusion models in autonomous driving.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Safe-Sim: Safety-Critical Closed-Loop Traffic Simulation with Diffusion-Controllable Adversaries](safe-sim_safety-critical_closed-loop_traffic_simulation_with_diffusion-cont.md)
- [\[ECCV 2024\] UniTraj: A Unified Framework for Scalable Vehicle Trajectory Prediction](unitraj_a_unified_framework_for_scalable_vehicle_trajectory_prediction.md)
- [\[ECCV 2024\] VisionTrap: Vision-Augmented Trajectory Prediction Guided by Textual Descriptions](visiontrap_vision-augmented_trajectory_prediction_guided_by_textual_descriptions.md)
- [\[ECCV 2024\] DySeT: A Dynamic Masked Self-distillation Approach for Robust Trajectory Prediction](dyset_a_dynamic_masked_self-distillation_approach_for_robust_trajectory_predicti.md)
- [\[ICCV 2025\] Generative Active Learning for Long-tail Trajectory Prediction via Controllable Diffusion Model](../../ICCV2025/autonomous_driving/generative_active_learning_for_long-tail_trajectory_prediction_via_controllable_.md)

</div>

<!-- RELATED:END -->

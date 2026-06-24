---
title: >-
  [Paper Note] Don't be so Negative! Score-based Generative Modeling with Oracle-assisted Guidance
description: >-
  [ICML 2025][Autonomous Driving][diffusion model] Proposes the Gen-neG method, which redirects the generative distribution from constraint-violating regions to the positive support region by iteratively training a Bayes-optimal classifier on synthetic data from diffusion models and using it to guide the sampling process. The key innovation lies in correctly handling the importance sampling of class prior probabilities, reducing the collision and out-of-boundary rate from 29.3%…
tags:
  - "ICML 2025"
  - "Autonomous Driving"
  - "diffusion model"
  - "classifier guidance"
  - "constrained generation"
  - "oracle"
  - "collision avoidance"
  - "negative examples"
date: 2026-05-08
content_hash: f66beedd445cca75
---

# Don't be so Negative! Score-based Generative Modeling with Oracle-assisted Guidance

**Conference**: ICML 2025  
**arXiv**: [2307.16463](https://arxiv.org/abs/2307.16463)  
**Code**: [GitHub](https://github.com/plai-group/gen-neg)  
**Area**: Autonomous Driving  
**Keywords**: diffusion model, classifier guidance, constrained generation, oracle, collision avoidance, negative examples

## TL;DR

Proposes the Gen-neG method, which redirects the generative distribution from constraint-violating regions to the positive support region by iteratively training a Bayes-optimal classifier on synthetic data from diffusion models and using it to guide the sampling process. The key innovation lies in correctly handling the importance sampling of class prior probabilities, reducing the collision and out-of-boundary rate from 29.3% to 5.6% in traffic scene generation.

## Background & Motivation

**Background**: Score-based diffusion models have achieved great success in tasks such as image generation, motion synthesis, and scene generation. However, in constrained domain applications, a certain proportion of generated samples violate physical or safety constraints. For example, in autonomous driving scene generation, vehicles may overlap (collision) or depart from the road; in human motion generation, joints may penetrate the ground.

**Limitations of Prior Work**: The simplest solution is rejection sampling—using an oracle (such as a collision detector) to filter the generation results. However, in real-time systems (such as autonomous driving path planning), when the violation rate is $\epsilon$, obtaining at least one valid sample with $1-\delta$ probability requires $\log\delta / \log\epsilon$ parallel samplers (e.g., approximately 30 samplers are needed when $\epsilon=0.5, \delta=10^{-9}$), making the computational cost potentially unaffordable.

**Key Challenge**: Diffusion models achieve generalization through regularization (such as finite network capacity and limited diffusion steps). However, the cost of generalization is that probability mass is also allocated to constraint-violating regions. The challenge is to control the "direction" of generalization—generalizing within the valid region but not into the invalid region.

**Goal**: Given a training dataset (all valid samples) + an oracle function $\mathcal{O}: \mathcal{X} \to \{0, 1\}$ (labeling whether a sample is valid), train a diffusion model to: (1) maximize the likelihood of the training data; (2) allocate zero probability mass in the invalid regions.

**Key Insight**: The authors identify a previously overlooked necessary condition for classifier guidance—the classifier must be a Bayes-optimal classifier with respect to the current diffusion model distribution, and the class prior probabilities $\alpha$ and $1-\alpha$ of positive and negative samples must be handled correctly during training. If trained on a balanced dataset without importance sampling correction, the decision boundary of the classifier shifts, leading to incorrect guidance directions.

**Core Idea**: Generate synthetic data from the diffusion model, label positive and negative samples using the oracle, train a Bayes-optimal classifier via importance sampling for classifier guidance, and iteratively stack classifiers to progressively eliminate violations.

## Method

### Overall Architecture

Gen-neG consists of two stages. Stage 1: Train a baseline diffusion model on the original training data in a standard manner. Stage 2 (Iterative): Sample from the current model $\to$ label with oracle $\to$ train a classifier on a balanced dataset with importance sampling $\to$ perform classifier guidance to obtain a new model $\to$ repeat. Optionally, multiple stacked classifiers can be distilled into a single model to reduce inference costs.

### Key Designs

1. **Bayes-optimal Classifier and Importance Sampling Correction**:

    - **Function**: Train a time-dependent binary classifier $C_\phi(\mathbf{x}_t; t)$ with respect to the current model distribution $p_\theta$, to be used for classifier guidance.
    - **Mechanism**: The objective of the Bayes-optimal classifier is the cross-entropy loss $\mathcal{L}_\phi^{\text{CE}} = -\mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_t}[\mathcal{O}(\mathbf{x}_0) \log C_\phi(\mathbf{x}_t; t) + (1-\mathcal{O}(\mathbf{x}_0)) \log(1 - C_\phi(\mathbf{x}_t; t))]$. Key challenge: When the baseline model is already relatively good, negative samples (violations) are far fewer than positive samples, leading to severe class imbalance. Gen-neG corrects this by sampling a balanced dataset and using the class prior $\alpha = p_\theta(y=1)$ for importance weighting in the loss: positive sample weights are multiplied by $\alpha$, and negative samples by $(1-\alpha)$. The final loss is $\hat{\mathcal{L}}_\phi^{\text{cls}} = \frac{1}{N}\sum_{\mathbf{x}_0 \in \hat{\mathcal{D}}^+} \alpha \cdot [-\log C_\phi] + \frac{1}{N}\sum_{\mathbf{x}_0 \in \hat{\mathcal{D}}^-} (1-\alpha) \cdot [-\log(1-C_\phi)]$.
    - **Design Motivation**: Without the importance sampling correction, the classifier converges on the balanced dataset to $C^*(x) = p(y=1|x; \text{balanced})$ instead of the true $p_\theta(y=1|x)$—the decision boundaries of these two are different, and only the latter can correctly guide the diffusion model. Experiments prove that failing to correct $\alpha$ leads to a significant deterioration in ELBO.

2. **Classifier Guidance and Iterative Stacking**:

    - **Function**: Modify the score function of the diffusion process using classifier gradients to guide sampling away from violating regions.
    - **Mechanism**: The modified score is $s_{\theta,\phi}(\mathbf{x}_t; t) = s_\theta(\mathbf{x}_t; t) + \nabla_{\mathbf{x}_t} \log C_\phi(\mathbf{x}_t; t)$. Theoretical guarantee (Theorem 3.1): When $C_\phi$ is a Bayes-optimal classifier, $s_{\theta,\phi}$ is exactly equal to the score of the positive class-conditional distribution $p_\theta(\mathbf{x}_t | y=1)$. Corollary (Corollary 3.2): The guided model has zero probability mass in the violating region, and its likelihood on any valid dataset is no lower than that of the original model. In practice, the classifier is imperfect, so some violations may remain—Gen-neG resolves this through iterative stacking: treating the guided model as the new baseline, regenerating $\to$ labeling $\to$ training classifiers $\to$ guiding.
    - **Design Motivation**: A single round of guidance may not completely eliminate violations (due to approximation errors of the classifier). Iterative stacking trains a new classifier at each round targeting the current residual violations, progressively tightening the boundary of the positive support region.

3. **Model Distillation**:

    - **Function**: Distill multiple stacked classifiers into a single diffusion model to reduce the computational overhead during inference.
    - **Mechanism**: Train a student model $s_\psi^{\text{dtl}}$ to minimize the score difference from the teacher (baseline model + all stacked classifiers): $\mathcal{L}_\psi^{\text{dtl}} = \mathbb{E}_{\mathbf{x}_0, t}[\gamma_t \|s_{\theta, \mathbf{\Phi}}(\mathbf{x}_t; t) - s_\psi^{\text{dtl}}(\mathbf{x}_t; t)\|^2]$. After distillation, only one forward pass is required per step instead of 1+K passes.
    - **Design Motivation**: The inference cost increases linearly to K+1 network evaluations after K rounds of stacking; distillation eliminates this overhead. Experiments show that the distilled model sometimes even outperforms the teacher (knowledge distillation effect).

### Loss & Training

Gen-neG involves two optimization objectives: (1) the standard score matching loss of the baseline diffusion model $\mathcal{L}_\theta^{\text{DM}} = \mathbb{E}[\gamma_t \|s_\theta(\mathbf{x}_t; t) - \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t | \mathbf{x}_0)\|^2]$; (2) the importance-weighted cross-entropy loss of the classifier $\hat{\mathcal{L}}_\phi^{\text{cls}}$. These are optimized alternately—first training the baseline model to convergence, and then training the classifier in each round of stacking.

## Key Experimental Results

### Traffic Scene Generation (Joint Sampling of 12 Vehicles)

| Method | Collision Rate (%) ↓ | Out-of-boundary Rate (%) ↓ | Total Violation Rate (%) ↓ | r-ELBO (×10⁻²) ↑ |
|------|----------|----------|------------|----------------|
| Baseline DM | 28.3±0.70 | 1.3±0.14 | 29.3±0.64 | -27.5±0.01 |
| Normalizing Flow | 91.2±0.27 | 13.1±0.48 | 91.9±0.25 | — |
| Time-indep. classifier | 20.7±0.59 | 0.9±0.09 | 21.4±0.63 | -244±30.4 |
| w/o IS (Ablation) | 14.6±0.49 | 0.8±0.13 | 15.2±0.50 | -28.0±0.01 |
| Gen-neG (iter 1) | 16.4±0.50 | 0.9±0.12 | 17.2±0.44 | -27.7±0.01 |
| Gen-neG (iter 2) | 11.6±0.65 | 0.6±0.10 | 12.2±0.60 | -28.0±0.01 |
| **Gen-neG (distill iter2)** | **5.1±0.24** | **0.5±0.09** | **5.6±0.20** | -27.0±0.01 |

### Human Motion Generation (Ground Penetration Avoidance)

| Method | Violation Rate (%) ↓ | Violation Rate per Step (%) ↓ | r-ELBO (×10⁻²) ↑ | FID ↓ | KID (×10⁻³) ↓ |
|------|----------|-------------|----------------|------|------------|
| MDM (baseline) | 27.66±0.77 | 7.84±0.27 | -1.06±0.02 | 0.445±0.040 | 8.27±2.14 |
| **Gen-neG** | 24.25±0.35 | 6.12±0.19 | **-1.01±0.03** | **0.414±0.030** | **6.99±0.78** |
| w/o IS (Ablation) | **22.85±0.18** | **5.47±0.18** | -1.13±0.06 | 0.415±0.030 | 8.40±1.83 |

### Key Findings

1. The distilled version of Gen-neG reduces the total violation rate in traffic scenes from 29.3% to 5.6% (an 81% reduction), while improving the r-ELBO from -27.5 to -27.0, meaning distribution quality increases rather than decreases.
2. Although not using importance sampling correction (w/o IS) can yield a lower violation rate, the ELBO deteriorates significantly—indicating that the distribution is improperly distorted and the density estimation of the positive region worsens.
3. While the time-independent classifier reduces the collision rate to 20.7%, the ELBO catastrophically deteriorates to -244—proving that time dependency is necessary.
4. The distilled version occasionally outperforms the teacher (from 12.2% at iter 2 to 5.6% after distillation), which is likely due to the regularization effect of knowledge distillation.
5. Utilizing Gen-neG reduces the average GPU cost of autonomous driving simulation by 57% (due to fewer rejection sampling loops).

## Highlights & Insights

1. **Reveals a neglected necessary condition for classifier guidance**: The class prior probabilities must correctly match the positive-to-negative ratio of the current model. This seemingly simple finding has profound implications—failing to correct it leads to a systematic bias in the guidance direction.
2. **Elegant theoretical guarantees**: Guidance from a Bayes-optimal classifier yields sampling exactly equivalent to the positive conditional distribution—achieving zero violations with no drop in likelihood ($Theorem\ 3.1 + Corollary\ 3.2$).
3. **Oracle as a free constraint signal**: Check functions for physical constraints such as collision detection and ground penetration are virtually free in practice.
4. **Distillation compresses multiple rounds of classifiers into a single model**: This resolves the linear inference overhead of stacking, making the method deployable in real-time systems.

## Limitations & Future Work

- Iterative stacking to train multiple classifiers increases the overall training time.
- The current method only trains subsequent classifiers using synthetic data without referencing the real training set—long-term iteration may lead to distribution shift.
- The improvement in motion generation experiments is relatively small (27.66% $\to$ 24.25%), which may be related to the MDM architecture predicting $x_0$ instead of the score.
- The oracle function needs to be differentiable or approximable—non-differentiable constraints require additional processing.

## Related Work & Insights

- Unlike standard classifier guidance by Dhariwal & Nichol (2021), Gen-neG trains classifiers on synthetic data and uses IS correction—suitable for scenarios where negative samples are absent in the training set.
- Diffusion bridges (Liu et al., 2023) can theoretically achieve zero violations but require analytically formatted constraints—Gen-neG operates via a black-box oracle, adapting to a wider range of constraint types.
- Inspiration: The oracle-guided framework can be generalized to areas like molecular design (meeting physical property constraints) and protein folding (satisfying structural constraints).

## Rating

⭐⭐⭐⭐ The method is concise with clear theory, revealing crucial details overlooked in classifier guidance. The traffic scene experiments are highly impressive (29.3% $\to$ 5.6%). The limited improvement in motion generation experiments is a minor drawback. Overall, this is a significant contribution to the field of constrained generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Geometry-to-Image Synthesis-Driven Generative Point Cloud Registration](geometry-to-image_synthesis-driven_generative_point_cloud_registration.md)
- [\[ICCV 2025\] SA-Occ: Satellite-Assisted 3D Occupancy Prediction in Real World](../../ICCV2025/autonomous_driving/sa-occ_satellite-assisted_3d_occupancy_prediction_in_real_world.md)
- [\[CVPR 2025\] Generative Gaussian Splatting for Unbounded 3D City Generation](../../CVPR2025/autonomous_driving/generative_gaussian_splatting_for_unbounded_3d_city_generation.md)
- [\[ICCV 2025\] Leveraging 2D Priors and SDF Guidance for Dynamic Urban Scene Rendering](../../ICCV2025/autonomous_driving/leveraging_2d_priors_and_sdf_guidance_for_urban_scene_rendering.md)
- [\[ICCV 2025\] ReconDreamer++: Harmonizing Generative and Reconstructive Models for Driving Scene Representation](../../ICCV2025/autonomous_driving/recondreamer_harmonizing_generative_and_reconstructive_models_for_driving_scene_.md)

</div>

<!-- RELATED:END -->

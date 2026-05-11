---
title: >-
  [Paper Note] RNE: plug-and-play diffusion inference-time control and energy-based training
description: >-
  [ICLR 2026][Image Generation][diffusion models] This paper proposes the Radon-Nikodym Estimator (RNE), which exploits density ratios between path distributions to reveal the fundamental relationship between marginal dens…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "diffusion models"
  - "density ratio estimation"
  - "inference-time control"
  - "energy-based model training"
  - "Radon-Nikodym derivative"
date: 2026-05-08
content_hash: 2913abfb1cb104cc
---

# RNE: plug-and-play diffusion inference-time control and energy-based training

**Conference**: ICLR 2026
**arXiv**: [2506.05668](https://arxiv.org/abs/2506.05668)
**Code**: N/A
**Area**: Image Generation / Diffusion Models
**Keywords**: diffusion models, density ratio estimation, inference-time control, energy-based model training, Radon-Nikodym derivative

## TL;DR

This paper proposes the Radon-Nikodym Estimator (RNE), which exploits density ratios between path distributions to reveal the fundamental relationship between marginal densities and transition kernels, providing a unified plug-and-play framework that simultaneously enables diffusion density estimation, inference-time control, and energy-based diffusion training.

## Background & Motivation

Diffusion models generate data by iteratively denoising, corresponding to the time-reversal of a forward noising process. In many applications, access to denoising kernels alone is insufficient; knowledge of **marginal densities** along the generative trajectory is required. Such knowledge supports:

**Density estimation**: evaluating the probability density of a generative model at arbitrary points

**Inference-time control**: dynamically guiding outputs during generation, e.g., conditional generation and composition of multiple models

**Energy-based diffusion training**: training energy functions to parameterize diffusion models

However, obtaining marginal densities of diffusion models has long been a challenging problem:
- Direct computation requires integrating over all possible forward paths, which is computationally intractable
- Existing methods (e.g., likelihood estimation via the probability-flow ODE) are either computationally expensive or insufficiently accurate
- Inference-time control methods typically rely on specific assumptions (e.g., approximations via the Tweedie formula), limiting their applicability

**Core Insight**: By invoking the concept of the Radon-Nikodym derivative (density ratio), a fundamental mathematical relationship between marginal densities and transition kernels can be established—without training additional models and without depending on any specific diffusion model architecture.

## Method

### Overall Architecture

The core idea of RNE is to operate at the level of distributions over generative paths:

1. Define two path distributions: the forward (noising) path distribution $\mathbb{P}$ and the backward (denoising) path distribution $\mathbb{Q}$
2. By the Radon-Nikodym theorem, the density ratio $\frac{d\mathbb{Q}}{d\mathbb{P}}$ of $\mathbb{Q}$ with respect to $\mathbb{P}$ can be expressed in terms of transition kernels
3. By manipulating this density ratio, marginal densities can be estimated and controlled

### Key Designs

1. **Path distribution density ratio**: the central mathematical tool

    - Consider the forward process $q(x_0, x_1, ..., x_T)$ and the backward process $p(x_T, x_{T-1}, ..., x_0)$
    - The Radon-Nikodym derivative links the two path distributions: $\frac{d\mathbb{Q}}{d\mathbb{P}}(x_{0:T}) = \frac{p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1}|x_t)}{q(x_0) \prod_{t=1}^{T} q(x_t|x_{t-1})}$
    - This ratio decomposes into a product of stepwise local ratios, each involving only known transition kernels
    - Design Motivation: reformulate the global density problem as local (stepwise) density ratio estimation

2. **Diffusion density estimation**: obtaining marginal densities via density ratios

    - At any intermediate timestep $t$, the marginal density of state $x_t$ can be estimated using partial products of path density ratios
    - No additional density model needs to be trained; the transition kernels of the diffusion model itself are directly exploited
    - The density ratio in path space can be estimated via Monte Carlo sampling

3. **Inference-time control**: plug-and-play conditional generation

    - **Annealing**: the estimated marginal density is used as an annealing temperature schedule for the energy function, enabling more accurate conditional sampling
    - **Model Composition**: multiplying density ratios from multiple diffusion models achieves multi-condition compositional control
    - RNE operates as a plug-and-play module without modifying pretrained diffusion model weights
    - Supports **inference-time scaling**: larger computational budgets translate directly into improved control quality

4. **Energy-based diffusion training**: RNE as a training regularizer

    - Training traditional energy-based diffusion models requires estimating the partition function, which is computationally difficult
    - RNE provides a straightforward regularization approach: leveraging density ratios to constrain energy function training
    - This avoids explicit estimation of the partition function and simplifies the training pipeline

5. **Modality agnosticism**: not restricted to continuous diffusion

    - The theoretical framework of RNE is grounded in the general notion of path distributions
    - It applies not only to continuous-state diffusion models but also to **discrete diffusion models** (e.g., discrete denoising for text generation)
    - This makes RNE a modality-agnostic general-purpose tool

### Loss & Training

RNE requires **no additional training** in the inference-time control setting (plug-and-play), and serves as an auxiliary regularization loss in the energy-based diffusion training setting:

- Inference-time control: the pretrained model is frozen; sampling trajectories are adjusted solely via density ratios
- Energy training regularization: an RNE-based regularization term is added to the standard denoising loss to enforce consistency between the learned energy function and the true density ratio

## Key Experimental Results

### Main Results

| Task | Method | Key Metric | Notes |
|------|--------|------------|-------|
| Annealed sampling | RNE | Outperforms standard methods | More accurate conditional sampling |
| Model composition | RNE | High-quality multi-condition generation | Combines multiple pretrained models |
| Inference-time scaling | RNE | Performance improves with compute | Validates scaling property |
| Energy-based diffusion training | RNE regularization | Simple and effective | No partition function estimation needed |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|------------|-------|
| Without RNE density estimation | Inaccurate density estimation | Lacks path-level density ratio information |
| With RNE | Improved density estimation accuracy | Fully exploits transition kernel information |
| Continuous diffusion | Verified effective | Standard setting |
| Discrete diffusion | Equally effective | Validates modality agnosticism |

### Key Findings

1. **Unified framework for inference-time control**: RNE unifies seemingly disparate inference-time control methods—such as annealing and model composition—under a density-ratio perspective
2. **Inference-time scaling**: allocating more computation (additional sampled paths) consistently improves control accuracy, consistent with the trend of inference-time compute scaling
3. **Simplified energy training**: RNE regularization circumvents the difficulty of partition function estimation in conventional energy-based model training
4. **Cross-modal generality**: the effectiveness of RNE is verified on both continuous and discrete diffusion models

## Highlights & Insights

1. **Theoretical elegance**: by employing the Radon-Nikodym derivative—a fundamental tool from measure theory—the paper establishes a unified connection among three seemingly independent problems in diffusion models: density estimation, inference-time control, and energy-based training
2. **Plug-and-play design**: requires neither modification of pretrained models nor training of additional control networks (e.g., ControlNet), substantially lowering the barrier to adoption
3. **Innovation of the path-distribution perspective**: rather than operating at the level of single-step transitions, the framework establishes connections at the level of distributions over complete trajectories—a higher level of abstraction
4. **Inference-time scaling property**: resonates with the broader AI community's interest in test-time compute and inference-time scaling
5. **Applicability to discrete diffusion**: extends the framework's scope, with potential value for diffusion-based generation of discrete sequences such as text and proteins

## Limitations & Future Work

1. **Variance of Monte Carlo estimation**: density ratio estimation in path space may exhibit high variance, particularly along long diffusion trajectories
2. **Computational cost**: although no additional training is required, inference-time estimation of density ratios demands multiple sampled paths, increasing inference latency
3. **Insufficient validation at large scale**: validation on large-scale models such as Stable Diffusion and DALL-E is needed
4. **Lack of systematic comparison with existing inference-time control methods**: detailed comparisons with Classifier Guidance, Classifier-Free Guidance, DPS, and related methods are warranted
5. **Gap between theory and practice**: the theoretical framework assumes exact forward/backward kernels, whereas learned approximate models are used in practice; the impact of approximation error requires deeper analysis

## Related Work & Insights

- **Density estimation for diffusion models**: compared to the continuous normalizing flow (CNF) approach of Song et al., RNE does not require solving an ODE but instead operates directly at the level of path distributions
- **Inference-time control**: complementary to Classifier Guidance (Dhariwal & Nichol, 2021), DPS (Chung et al., 2022), FreeDoM (Yu et al., 2023), and related methods, while providing a more unified theoretical perspective
- **Energy-based models**: complements energy-based diffusion training methods by simplifying the partition function estimation problem
- **Insights**: RNE demonstrates the power of reasoning about generative models at the distributional rather than pointwise level, a perspective that may inspire further unified frameworks

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Unifying three independent problems via the Radon-Nikodym derivative represents a strongly original theoretical contribution
- Experimental Thoroughness: ⭐⭐⭐ — Proof-of-concept experiments are sufficient, but large-scale validation is lacking
- Writing Quality: ⭐⭐⭐⭐ — Theory is clearly presented with a coherent unified framework
- Value: ⭐⭐⭐⭐ — The plug-and-play nature and theoretical unification carry significant practical and academic value

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Taming Score-Based Denoisers in ADMM: A Convergent Plug-and-Play Framework](../../CVPR2026/image_generation/taming_score-based_denoisers_in_admm_a_convergent_plug-and-play_framework.md)
- [\[ICLR 2026\] Diffusion Blend: Inference-Time Multi-Preference Alignment for Diffusion Models](diffusion_blend_inference-time_multi-preference_alignment_for_diffusion_models.md)
- [\[ICLR 2026\] Unsupervised Conformal Inference: Bootstrapping and Alignment to Control LLM Uncertainty](unsupervised_conformal_inference_bootstrapping_and_alignment_to_control_llm_unce.md)
- [\[ICCV 2025\] Trans-Adapter: A Plug-and-Play Framework for Transparent Image Inpainting](../../ICCV2025/image_generation/trans-adapter_a_plug-and-play_framework_for_transparent_image_inpainting.md)
- [\[ICLR 2026\] Training-Free Reward-Guided Image Editing via Trajectory Optimal Control](training-free_reward-guided_image_editing_via_trajectory_optimal_control.md)

</div>

<!-- RELATED:END -->

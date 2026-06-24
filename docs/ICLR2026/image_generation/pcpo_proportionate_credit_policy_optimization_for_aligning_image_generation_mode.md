---
title: >-
  [Paper Note] PCPO: Proportionate Credit Policy Optimization for Aligning Image Generation Models
description: >-
  [ICLR 2026][Image Generation][Policy Gradient] Proposes PCPO, which fixes the disproportionate credit assignment inherent in the policy gradients of diffusion/flow models through stable objective reconstruction and principled timestep re-weighting, significantly accelerating convergence and mitigating model collapse.
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Policy Gradient"
  - "Credit Assignment"
  - "Diffusion Models"
  - "Flow Matching"
  - "Model Collapse"
date: 2026-05-08
content_hash: 54ee9d0529831db2
---

# PCPO: Proportionate Credit Policy Optimization for Aligning Image Generation Models

**Conference**: ICLR 2026  
**arXiv**: [2509.25774](https://arxiv.org/abs/2509.25774)  
**Code**: [GitHub](https://github.com/jaylee2000/pcpo/)  
**Area**: Image Generation  
**Keywords**: Policy Gradient, Credit Assignment, Diffusion Models, Flow Matching, Model Collapse

## TL;DR

Proposes PCPO, which fixes the disproportionate credit assignment inherent in the policy gradients of diffusion/flow models through stable objective reconstruction and principled timestep re-weighting, significantly accelerating convergence and mitigating model collapse.

## Background & Motivation

- GRPO has become the SOTA framework for T2I model alignment, but training is unstable and prone to model collapse.
- **Root Cause Analysis**:
  1. Standard objectives are susceptible to numerical precision errors.
  2. The mathematical structure of samplers leads to **disproportionate credit assignment**—gradient contributions from different timesteps are arbitrarily scaled.

## Method

### Overall Architecture

PCPO treats GRPO-style T2I alignment as a policy gradient problem over a denoising trajectory. It first diagnoses two major pathologies in the standard objective—numerically explosive policy ratios and non-uniform timestep weights imposed by the sampler's mathematical structure—and then specifically replaces the objective function and calibrates the credit weights for each step. This correction does not modify the model architecture or sampling process; it only changes the training objective and the variance/re-weighting schedule. Consequently, it can be directly applied to existing frameworks such as DDPO, DanceGRPO, and Flow-GRPO.

### Key Designs

**1. Problem Diagnosis: Identifying non-uniform native weights $w(t)$**

To identify the source of instability, the authors provide an exact decomposition of the log policy ratio under DDIM sampling via Proposition 1:

$$\log \rho_t = -\left[w(t)(\hat{\boldsymbol{\varepsilon}}_\theta^{(t)} - \hat{\boldsymbol{\varepsilon}}_{\text{old}}^{(t)}) \cdot \boldsymbol{\epsilon}_{\text{old}}^{(t)} + \frac{1}{2}\|w(t)(\hat{\boldsymbol{\varepsilon}}_\theta^{(t)} - \hat{\boldsymbol{\varepsilon}}_{\text{old}}^{(t)})\|^2\right]$$

The key insight is that the native weight $w(t) = C(t)/\sigma_t$ spans several orders of magnitude across different timesteps, causing gradients for certain steps within the same trajectory to be arbitrarily amplified while others are suppressed. This becomes clearer when compared to standard REINFORCE: in REINFORCE, the contributions of different actions are scaled uniformly, whereas the diffusion sampler gradient formula, though similar in form, introduces this non-uniform $w(t)$—a byproduct of sampler mathematics rather than an intentional credit assignment strategy. Identifying this provides a clear target for correction.

**2. Stable log-hinge objective: Replacing explosive $\rho_t-1$ with $\log\rho_t$**

The $\rho_t - 1$ term in standard objectives is easily dominated by errors when numerical precision is limited. Validating that the difference is minimal under small updates (Taylor approximation error < 1.2%), the authors switch to a smoother $\log\rho_t$, which significantly reduces the clipping ratio. The reconstructed base objective is:

$$\mathcal{L}_{\text{PCPO-base}}(\theta) = \mathbb{E}\left[\sum_{t=1}^T \max\{0, \xi|A| - A\log\rho_t\}\right]$$

where $A$ is the advantage and $\xi$ controls the hinge boundary. This step only changes the form of the objective without introducing extra overhead, serving as a prerequisite for the stable application of proportionate credit assignment.

**3. Proportionate Credit Assignment: Enforcing per-step weight alignment**

Having diagnosed non-uniform $w(t)$ as the root cause, PCPO flattens it directly. For diffusion models, it redesigns the DDIM variance schedule $\tilde{\sigma}_t$ such that $w(t) = w^*$ becomes a constant, and scales $w^*$ to match the mean of the original weights, thereby eliminating non-uniformity without altering the overall gradient magnitude. For flow models, Proposition 2 provides a more direct approach—re-weighting the training objective so that each step’s credit is proportional to its integration interval, $w(t_i) = \zeta \Delta t_i$. Both corrections restore proper credit assignment, which explains why the acceleration is most pronounced on models like FLUX, where timestep shifts are more aggressive and native weights are more non-uniform.

## Key Experimental Results

### Training Efficiency

| Baseline | Reward | Target Level | Baseline Epochs | PCPO Epochs | Gain |
|------|------|---------|------------|------------|------|
| DDPO | Aesthetics | 6.90 | 147 | 118 | **24.6%** |
| DDPO | BERTScore | 0.52 | 191 | 146 | **30.8%** |
| DanceGRPO (SD1.4) | HPS | 0.370 | 236 | 188 | **25.5%** |
| DanceGRPO (FLUX) | HPS | 0.360 | 209 | 148 | **41.2%** |

### Main Results (At matching reward levels)

| Model | Method | FID(↓) | FD_DINO(↓) | LPIPS(↑) |
|------|------|--------|-----------|----------|
| SD1.5 (batch=512) | Baseline | 24.09 | 451.19 | 0.6321 |
| SD1.5 (batch=512) | **Ours** | **22.06** | **391.30** | **0.6525** |
| FLUX | Baseline | 46.23 | 539.83 | 0.5736 |
| FLUX | **Ours** | **40.38** | **438.88** | 0.5708 |

### Key Findings

1. PCPO significantly reduces the clipping fraction under all settings, directly improving convergence.
2. The acceleration is most significant on FLUX (41.2%) because timestep shifts exacerbate the non-uniformity of native weights.
3. LMM statistical analysis confirms that the improvement in FID by PCPO is statistically significant (p=0.047).
4. Good generalization is observed on unseen prompts from MSCOCO-2017 and MJHQ-30K.

## Highlights & Insights

1. **Precise Root Cause Analysis**: Attributes training instability to the disproportionate credit assignment caused by the mathematical structure of the sampler.
2. **Unified Treatment**: Provides principled correction schemes for both diffusion and flow models.
3. **Simple yet Effective**: Requires only modifications to the variance schedule or re-weighting objective, making it plug-and-play.
4. **Mitigates Model Collapse**: Beyond accelerating convergence, it maintains image diversity and fidelity.

## Limitations & Future Work

- The Taylor approximation $\log \rho_t \approx \rho_t - 1$ assumes small policy updates and may fail during large-step updates.
- Modifying the variance schedule may alter the characteristics of the sampling trajectory (though experiments suggest the impact is negligible).
- Limited in-depth comparison with concurrent works such as TempFlow-GRPO or MixGRPO.
- Validation is restricted to three rewards: Aesthetics, BERTScore, and HPSv2.1.

## Related Work

- **T2I Alignment**: DDPO, DPO-Diffusion, DanceGRPO, Flow-GRPO
- **LLM Alignment**: PPO, GRPO, DPO
- **Model Collapse**: Shumailov et al. (2024), Reward Hacking

## Rating

- Novelty: ⭐⭐⭐⭐ — Precise diagnosis, though the correction method is relatively straightforward.
- Technical Depth: ⭐⭐⭐⭐ — Profound analysis from the REINFORCE perspective with complete mathematical derivations.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Verified across multiple frameworks and rewards with LMM statistical validation.
- Value: ⭐⭐⭐⭐⭐ — Plug-and-play, providing direct improvements to existing alignment pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] PCPO: Proportionate Credit Policy Optimization for Preference Alignment of Image Generation Models](pcpo_proportionate_credit_policy_optimization_for_preference_alignment_of_image_.md)
- [\[ICLR 2026\] LayerSync: Self-aligning Intermediate Layers](layersync_self-aligning_intermediate_layers.md)
- [\[ICLR 2026\] STORK: Accelerating Diffusion and Flow Matching Sampling by Simultaneously Solving Stiffness and Structural Dependency](stork_faster_diffusion_and_flow_matching_sampling_by_resolving_both_stiffness_an.md)
- [\[ICLR 2026\] Diverse Text-to-Image Generation via Contrastive Noise Optimization](diverse_text-to-image_generation_via_contrastive_noise_optimization.md)
- [\[ICLR 2026\] Mitigating Noise Shift in Denoising Generative Models with Noise Awareness Guidance](mitigating_noise_shift_in_denoising_generative_models_with_noise_awareness_guida.md)

</div>

<!-- RELATED:END -->

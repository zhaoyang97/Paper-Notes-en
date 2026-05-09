---
title: >-
  [Paper Note] PCPO: Proportionate Credit Policy Optimization for Aligning Image Generation Models
description: >-
  [ICLR 2026][Image Generation][Policy Gradient] This paper proposes PCPO, which corrects the disproportionate credit assignment inherent in policy gradient methods for diffusion/flow models via a stabilized objective reformulation and principled timestep reweighting, significantly accelerating convergence and mitigating model collapse.
tags:
  - ICLR 2026
  - Image Generation
  - Policy Gradient
  - Credit Assignment
  - Diffusion Models
  - Flow Matching
  - Model Collapse
date: 2026-05-08
content_hash: c124ba1541c9304d
---

# PCPO: Proportionate Credit Policy Optimization for Aligning Image Generation Models

**Conference**: ICLR 2026
**arXiv**: [2509.25774](https://arxiv.org/abs/2509.25774)
**Code**: [GitHub](https://github.com/jaylee2000/pcpo/)
**Area**: Image Generation
**Keywords**: Policy Gradient, Credit Assignment, Diffusion Models, Flow Matching, Model Collapse

## TL;DR

This paper proposes PCPO, which corrects the disproportionate credit assignment inherent in policy gradient methods for diffusion/flow models via a stabilized objective reformulation and principled timestep reweighting, significantly accelerating convergence and mitigating model collapse.

## Background & Motivation

- GRPO has become the SOTA framework for T2I model alignment, but suffers from training instability and model collapse.
- **Root cause analysis**:
  1. Standard objectives are susceptible to numerical precision errors.
  2. The mathematical structure of samplers induces **disproportionate credit assignment**—gradient contributions at different timesteps are arbitrarily scaled.

## Method

### Problem Diagnosis

**Proposition 1**: For DDIM sampling, the log policy ratio decomposes as:

$$\log \rho_t = -[w(t)(\hat{\boldsymbol{\varepsilon}}_\theta^{(t)} - \hat{\boldsymbol{\varepsilon}}_{\text{old}}^{(t)}) \cdot \boldsymbol{\epsilon}_{\text{old}}^{(t)} + \frac{1}{2}\|w(t)(\hat{\boldsymbol{\varepsilon}}_\theta^{(t)} - \hat{\boldsymbol{\varepsilon}}_{\text{old}}^{(t)})\|^2]$$

where the native weight $w(t) = C(t)/\sigma_t$ is highly non-uniform (spanning several orders of magnitude), constituting the primary source of training instability.

### Key Designs

**Step 1: Stabilized log-hinge objective**

Replace the unstable $\rho_t - 1$ with $\log \rho_t$ (Taylor approximation error < 1.2%):

$$\mathcal{L}_{\text{PCPO-base}}(\theta) = \mathbb{E}\left[\sum_{t=1}^T \max\{0, \xi|A| - A\log\rho_t\}\right]$$

**Step 2: Proportionate credit assignment**

- **Diffusion models**: Redesign the DDIM variance schedule $\tilde{\sigma}_t$ such that $w(t) = w^*$ (constant), with $w^*$ scaled to match the mean of the original weights.
- **Flow models** (Proposition 2): Directly reweight the training objective so that credit is proportional to the integration interval: $w(t_i) = \zeta \Delta t_i$.

### Analogy to REINFORCE

In standard REINFORCE, each action contributes with uniform scaling. The gradient formulation for diffusion samplers is analogous, but introduces non-uniform weights $w(t)$—an artifact of the sampler's mathematics rather than a deliberate credit assignment strategy. PCPO restores correct credit assignment by enforcing uniform weights.

## Key Experimental Results

### Training Efficiency

| Baseline | Reward | Target Level | Baseline Epochs | PCPO Epochs | Speedup |
|----------|--------|-------------|-----------------|-------------|---------|
| DDPO | Aesthetics | 6.90 | 147 | 118 | **24.6%** |
| DDPO | BERTScore | 0.52 | 191 | 146 | **30.8%** |
| DanceGRPO (SD1.4) | HPS | 0.370 | 236 | 188 | **25.5%** |
| DanceGRPO (FLUX) | HPS | 0.360 | 209 | 148 | **41.2%** |

### Image Quality (at Matched Reward Levels)

| Model | Method | FID(↓) | FD_DINO(↓) | LPIPS(↑) |
|-------|--------|--------|-----------|----------|
| SD1.5 (batch=512) | Baseline | 24.09 | 451.19 | 0.6321 |
| SD1.5 (batch=512) | **PCPO** | **22.06** | **391.30** | **0.6525** |
| FLUX | Baseline | 46.23 | 539.83 | 0.5736 |
| FLUX | **PCPO** | **40.38** | **438.88** | 0.5708 |

### Key Findings

1. PCPO substantially reduces the clipping fraction across all settings, directly improving convergence.
2. The speedup is most pronounced on FLUX (41.2%), as timestep shifting renders the native weights more non-uniform.
3. LMM-based statistical analysis confirms that PCPO's improvement in FID is statistically significant (p=0.047).
4. The method generalizes well to unseen prompts on MSCOCO-2017 and MJHQ-30K.

## Highlights & Insights

1. **Precise root cause identification**: Training instability is attributed to disproportionate credit assignment arising from the mathematical structure of the sampler.
2. **Unified treatment of diffusion and flow models**: Principled corrections are provided for distinct sampler types.
3. **Simple implementation, significant effect**: Only variance schedule modification or objective reweighting is required; the method is plug-and-play.
4. **Mitigates model collapse**: Beyond accelerating convergence, PCPO preserves image diversity and fidelity.

## Limitations & Future Work

- The Taylor approximation $\log \rho_t \approx \rho_t - 1$ assumes small policy updates and may break down under large update steps.
- Variance schedule modification may alter sampling trajectory characteristics, though experiments suggest the effect is negligible.
- In-depth comparison with concurrent work such as TempFlow-GRPO and MixGRPO is limited.
- Validation is restricted to three reward functions: Aesthetics, BERTScore, and HPSv2.1.

## Related Work & Insights

- **T2I alignment**: DDPO, DPO-Diffusion, DanceGRPO, Flow-GRPO
- **LLM alignment**: PPO, GRPO, DPO
- **Model collapse**: Shumailov et al. (2024), reward hacking

## Rating

- Novelty: ⭐⭐⭐⭐ — Precise diagnosis, though the correction itself is relatively straightforward
- Technical Depth: ⭐⭐⭐⭐ — Analysis from the REINFORCE perspective is insightful; mathematical derivations are complete
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Multi-framework, multi-reward validation with LMM-based statistical verification
- Value: ⭐⭐⭐⭐⭐ — Plug-and-play; directly improves existing alignment pipelines

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Asynchronous Denoising Diffusion Models for Aligning Text-to-Image Generation](asynchronous_denoising_diffusion_models_for_aligning_text-to-image_generation.md)
- [\[ICLR 2026\] Diverse Text-to-Image Generation via Contrastive Noise Optimization](diverse_text-to-image_generation_via_contrastive_noise_optimization.md)
- [\[CVPR 2026\] Neighbor GRPO: Contrastive ODE Policy Optimization Aligns Flow Models](../../CVPR2026/image_generation/neighbor_grpo_contrastive_ode_policy_optimization_aligns_flow_models.md)
- [\[ICLR 2026\] AlignTok: Aligning Visual Foundation Encoders to Tokenizers for Diffusion Models](aligntok_aligning_visual_foundation_encoders_to_tokenizers_for_diffusion_models.md)
- [\[ICLR 2026\] Pareto-Conditioned Diffusion Models for Offline Multi-Objective Optimization](pareto-conditioned_diffusion_models_for_offline_multi-objective_optimization.md)

</div>

<!-- RELATED:END -->

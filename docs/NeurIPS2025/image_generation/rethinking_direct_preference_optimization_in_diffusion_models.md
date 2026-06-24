---
title: >-
  [Paper Note] Rethinking Direct Preference Optimization in Diffusion Models
description: >-
  [SPIGM@NeurIPS 2025 / AAAI 2026 (Oral)][Image Generation][Diffusion Models] To address two core issues in DPO for diffusion models — limited exploration and reward scale imbalance — this paper proposes a stable reference model update strategy and a timestep-aware training strategy, both of which can be integrated into various preference optimization algorithms.
tags:
  - "SPIGM@NeurIPS 2025 / AAAI 2026 (Oral)"
  - "Image Generation"
  - "Diffusion Models"
  - "DPO"
  - "Preference Optimization"
  - "Reference Model Update"
  - "Timestep-Aware"
date: 2026-05-08
content_hash: 2e056c77fe98278a
---

# Rethinking Direct Preference Optimization in Diffusion Models

**Conference**: SPIGM@NeurIPS 2025 / AAAI 2026 (Oral)

**arXiv**: [2505.18736](https://arxiv.org/abs/2505.18736)

**Code**: [GitHub](https://github.com/kaist-cvml/RethinkingDPO_Diffusion_Models)

**Area**: LLM Alignment / Diffusion Models

**Keywords**: Diffusion Models, DPO, Preference Optimization, Reference Model Update, Timestep-Aware

## TL;DR

To address two core issues in DPO for diffusion models — limited exploration and reward scale imbalance — this paper proposes a stable reference model update strategy and a timestep-aware training strategy, both of which can be integrated into various preference optimization algorithms.

## Background & Motivation

Aligning text-to-image (T2I) diffusion models with human preferences is an active research direction. While preference optimization techniques such as DPO have been extended from LLMs to diffusion models, several domain-specific challenges remain:

**Limited Exploration**: A frozen reference model constrains the policy's exploration space, resulting in insufficient generation diversity.

**Reward Scale Imbalance**: The magnitude of reward signals varies drastically across denoising timesteps, undermining training stability.

**Diffusion-Specific Difficulty**: Unlike LLMs, the generation process in diffusion models involves multi-step denoising, with different optimization objectives at each step.

## Method

### Overall Architecture

Two orthogonal improvement strategies are proposed, which can be combined with existing methods such as Diffusion-DPO, DPOK, and D3PO.

### Key Designs

**1. Stable Reference Model Update**

In standard DPO, the reference model $\pi_{\text{ref}}$ remains fixed, limiting exploration:
- Proposes **gradual updates** to the reference model: $\pi_{\text{ref}}^{(t+1)} = (1-\alpha) \pi_{\text{ref}}^{(t)} + \alpha \pi_\theta^{(t)}$
- Reference model regularization: constrains the updated reference model from deviating too far from the initial model
- Balances exploration and stability: small $\alpha$ yields conservative updates; large $\alpha$ yields aggressive updates

$$\mathcal{L}_{\text{reg}} = \text{KL}(\pi_{\text{ref}}^{(t)} \| \pi_{\text{ref}}^{(0)})$$

**2. Timestep-Aware Training**

The signal-to-noise ratio varies substantially across timesteps in diffusion models:
- Observation: reward signal magnitudes at large timesteps (high noise) are far greater than at small timesteps (low noise)
- This causes training to be dominated by large timesteps, leaving small timesteps under-optimized
- Solution: apply **normalized weighting** to losses across different timesteps

$$\mathcal{L}_t = \frac{w(t)}{\sum_t w(t)} \mathcal{L}_{\text{DPO}}^{(t)}$$

where $w(t)$ is a normalized weight based on the variance of the reward signal at timestep $t$.

### Loss & Training

Full training objective:
$$\mathcal{L} = \mathcal{L}_{\text{DPO}} + \lambda_1 \mathcal{L}_{\text{reg}} + \lambda_2 \mathcal{L}_{\text{timestep-norm}}$$

Training pipeline: reference model updates and timestep normalization are incorporated into the standard DPO training loop.

## Key Experimental Results

### Main Results

Human preference evaluation on SDXL (Pick-a-Pic, HPSv2):

| Method | HPSv2 ↑ | Pick Score ↑ | Aesthetic ↑ | CLIP Score |
|--------|---------|-------------|------------|-----------|
| SDXL (baseline) | 27.5 | 21.8 | 5.82 | 0.315 |
| Diffusion-DPO | 28.2 | 22.3 | 5.95 | 0.318 |
| Diffusion-DPO + Ours | **28.8** | **22.9** | **6.12** | **0.322** |
| D3PO | 28.0 | 22.1 | 5.90 | 0.316 |
| D3PO + Ours | **28.5** | **22.6** | **6.05** | **0.320** |

Results on SD1.5:

| Method | HPSv2 ↑ | Aesthetic ↑ | Diversity (FID) |
|--------|---------|------------|----------------|
| SD1.5 | 25.8 | 5.45 | 15.2 |
| Diffusion-DPO | 26.5 | 5.68 | 18.5 |
| Diffusion-DPO + Ours | **27.1** | **5.85** | **16.8** |

### Ablation Study

Individual contributions of each strategy (SDXL, HPSv2):

| Configuration | HPSv2 | Gain |
|---------------|-------|------|
| Diffusion-DPO (baseline) | 28.2 | - |
| + Reference Model Update | 28.5 | +0.3 |
| + Timestep-Aware | 28.5 | +0.3 |
| + Both Combined | **28.8** | **+0.6** |

### Key Findings

1. The two strategies offer complementary contributions, each providing approximately +0.3 HPSv2 improvement.
2. Reference model updates become more effective in later training stages, as the policy drifts further from the initial model.
3. Timestep-aware training yields the greatest improvement at low-noise steps, which govern fine-grained detail quality.
4. The method integrates seamlessly into multiple algorithms, including Diffusion-DPO and D3PO.

## Highlights & Insights

- **Orthogonal Improvements**: The two strategies are mutually independent and can be applied separately or jointly.
- **Generality**: Both components serve as plug-and-play modules compatible with any diffusion-based preference optimization method.
- **Practical Insight**: Reward imbalance across timesteps is an overlooked yet important issue in diffusion DPO.

## Limitations & Future Work

1. The optimal value of $\alpha$ is task- and model-dependent.
2. Reference model updates increase memory requirements, as additional model parameters must be stored.
3. Validation is limited to image generation; video generation settings remain unexplored.
4. The design of timestep normalization weights lacks adaptivity.

## Related Work & Insights

- **Diffusion-DPO** (Wallace et al.): extends DPO to diffusion models
- **D3PO**: an alternative diffusion preference optimization method
- **Online DPO**: related work on reference model updates in LLMs

## Rating

- ⭐ Novelty: 7/10 — Both improvements are effective but conceptually straightforward
- ⭐ Value: 8/10 — Plug-and-play design with open-source code offers high practical utility
- ⭐ Writing Quality: 8/10 — Ablation analysis is clear and experimental design is well-structured

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Curriculum Direct Preference Optimization for Diffusion and Consistency Models](../../CVPR2025/image_generation/curriculum_direct_preference_optimization_for_diffusion_and_consistency_models.md)
- [\[ICLR 2026\] Reinforcing Diffusion Models by Direct Group Preference Optimization](../../ICLR2026/image_generation/reinforcing_diffusion_models_by_direct_group_preference_optimization.md)
- [\[CVPR 2025\] InPO: Inversion Preference Optimization with Reparametrized DDIM for Efficient Diffusion Model Alignment](../../CVPR2025/image_generation/inpo_inversion_preference_optimization_diffusion_alignment.md)
- [\[NeurIPS 2025\] Diffusion Model as a Noise-Aware Latent Reward Model for Step-Level Preference Optimization](diffusion_model_as_a_noiseaware_latent_reward_model_for_step.md)
- [\[ICML 2025\] Smoothed Preference Optimization via ReNoise Inversion for Aligning Diffusion Models with Varied Human Preferences](../../ICML2025/image_generation/smoothed_preference_optimization_via_renoise_inversion_for_aligning_diffusion_mo.md)

</div>

<!-- RELATED:END -->

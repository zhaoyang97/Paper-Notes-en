---
title: >-
  [Paper Note] Remasking Discrete Diffusion Models with Inference-Time Scaling
description: >-
  [NeurIPS 2025][Image Generation][discrete diffusion models] This paper proposes the ReMDM sampler, which enables iterative error correction in discrete mask diffusion models by allowing already-decoded tokens to be remasked during generation. This mechanism supports inference-time compute scaling and yields substantial quality improvements on text, image, and molecular design tasks.
tags:
  - NeurIPS 2025
  - Image Generation
  - discrete diffusion models
  - remasking sampling
  - inference-time compute scaling
  - iterative refinement
  - controllable generation
date: 2026-05-08
content_hash: 3579b310f3ccdc80
---

# Remasking Discrete Diffusion Models with Inference-Time Scaling

**Conference**: NeurIPS 2025
**arXiv**: [2503.00307](https://arxiv.org/abs/2503.00307)
**Code**: [https://github.com/guanghanwang/remdm](https://github.com/guanghanwang/remdm)
**Area**: Image Generation / Discrete Diffusion
**Keywords**: discrete diffusion models, remasking sampling, inference-time compute scaling, iterative refinement, controllable generation

## TL;DR
This paper proposes the ReMDM sampler, which enables iterative error correction in discrete mask diffusion models by allowing already-decoded tokens to be remasked during generation. This mechanism supports inference-time compute scaling and yields substantial quality improvements on text, image, and molecular design tasks.

## Background & Motivation

Diffusion models have achieved remarkable success in image and video generation, with **iterative refinement**—repeatedly correcting outputs and fixing errors across multiple generation steps—being one of their core advantages. However, state-of-the-art discrete diffusion models, particularly masked/absorbing-state models (MDLM), suffer from a fundamental limitation: **once a token is decoded, it can never be updated** (the failure-to-remask property). Decoding errors become permanently "locked in," analogous to the sequential generation constraint of autoregressive models.

This limitation gives rise to three problems: (1) independent token decoding during parallel generation leads to inconsistency errors (e.g., "She sell" instead of "She sells" or "They sell"); (2) increasing the number of sampling steps fails to effectively improve quality (inference-time compute scaling is bottlenecked); (3) controllable generation capability is constrained.

The core idea of this paper is to design a new posterior distribution that allows already-decoded tokens to be remasked with some probability, thereby enabling iterative error correction. Crucially, this new posterior preserves the same marginal distribution as MDLM, so pretrained MDLM weights can be directly reused without retraining.

## Method

### Overall Architecture

Building upon MDLM, ReMDM introduces a parameter $\sigma_t$ to control the remasking probability. When $\sigma_t = 0$, the method reduces to standard MDLM; when $\sigma_t > 0$, decoded tokens can be remasked with positive probability, enabling iterative refinement. The entire method can be applied directly as a sampler on top of pretrained MDLM models.

### Key Designs

1. **Remasking Posterior**: For an already-decoded token $z_t \neq m$, the posterior is $q(z_s \mid z_t = x, x) = (1 - \sigma_t)x + \sigma_t \cdot m$, i.e., the token is remasked with probability $\sigma_t$. For tokens that remain masked, the posterior is carefully designed to preserve the same marginal distribution $q(z_t \mid x)$ as MDLM. This is established as a core theorem (Theorem 3.1), guaranteeing compatibility with pretrained weights. ReMDM is a non-Markovian process, analogous to the relationship between DDIM and DDPM.

2. **Design Strategies for $\sigma_t$**: Several strategies are proposed:

    - **Max-Capped (ReMDM-cap)**: $\sigma_t$ is clipped at a constant upper bound $\eta_\text{cap}$.
    - **Rescaled (ReMDM-rescale)**: $\sigma_t = \eta_\text{rescale} \cdot \sigma_t^{\max}$, controlling remasking strength via a scaling factor.
    - **Confidence-Based (ReMDM-conf)**: Remasking probability is allocated according to the model's per-token prediction confidence—tokens with lower confidence are more likely to be remasked, an intuitively motivated design.

3. **Turn On/Off Strategies**:

    - **Switch**: Remasking is activated only after timestep $t_\text{switch}$, with standard MDLM used beforehand to generate an initial draft.
    - **Loop**: A three-phase procedure—(1) standard MDLM decoding; (2) holding $\alpha$ fixed and iteratively remasking and re-predicting in a loop (error correction phase); (3) decoding remaining tokens with standard MDLM. This is the strongest strategy, intuitively performing draft-then-revise generation.

4. **Relationship to Predictor–Corrector Methods**: The authors prove that ReMDM is strictly more general—the FB corrector and DFM corrector are both special cases of ReMDM (Propositions 4.2 and 4.3), and ReMDM can handle the case where $\alpha_t$ is constant (Proposition 4.4), which DFM cannot.

### Loss & Training

The NELBO of ReMDM is a reweighted version of the MDLM objective, differing only by a factor of $(1 - \sigma_t)$. Since setting $\sigma_t = 0$ recovers MDLM, and experiments confirm comparable performance when trained with the ReMDM objective, the **recommended approach is to reuse pretrained MDLM weights directly** and apply the ReMDM sampler at inference time without retraining.

## Key Experimental Results

### Main Results

| Task / Dataset | Metric | ReMDM | Prev. SOTA | Gain |
|---|---|---|---|---|
| OWT text generation ($T=4096$) | MAUVE ↑ | 0.656 | 0.269 (DFM) | 2.4× |
| OWT text generation ($T=1024$) | MAUVE ↑ | 0.403 | 0.254 (DFM) | 1.6× |
| OWT fast sampling ($T=512$) | MAUVE ↑ | 0.350 | 0.211 (DFM) | 1.7× |
| ImageNet generation ($T=64$) | FID ↓ | 4.45 | 4.69 (MDLM) | Better |
| ImageNet generation ($T=64$) | IS ↑ | 209.45 | 196.38 (MaskGiT) | +6.7% |
| LLaDA Countdown | pass@1% | 46.1 | 45.2 (LLaDA) | +0.9 |

### Ablation Study

| Configuration | MAUVE (OWT $T=4096$) | Notes |
|---|---|---|
| MDLM (baseline) | 0.035 | No remasking |
| + 64-bit precision | Improved | Avoids diversity collapse |
| + Nucleus sampling | Improved | Top-$p=0.9$ is a key quality factor |
| + ReMDM remasking | Large improvement | Largest single contributor |
| + Loop strategy | 0.656 | Full ReMDM |

### Key Findings
- ReMDM excels at inference-time compute scaling: increasing the number of steps $T$ yields consistent quality improvement, whereas MDLM and corrector-based methods saturate.
- On molecular design tasks, ReMDM pushes the novelty–property Pareto frontier of controllable generation toward more favorable regions.
- Applying ReMDM to large discrete language models such as LLaDA 8B also yields substantial improvements on downstream tasks.

## Highlights & Insights
- **Zero additional training cost**: Directly reusing pretrained MDLM weights and modifying only the sampling strategy yields significant gains, greatly lowering the barrier to adoption.
- **Theoretical elegance**: By preserving the marginal distribution, ReMDM constructs a non-Markovian process compatible with MDLM while unifying the FB and DFM correctors as special cases.
- **Discrete analog of DDIM**: ReMDM stands in relation to MDLM as DDIM does to DDPM, providing a more flexible sampling space for discrete diffusion.

## Limitations & Future Work
- The hyperparameter search space for remasking strategies ($\sigma_t$ schedule, loop parameters) is large and requires task-specific tuning.
- While remasking enables more effective use of additional sampling steps, it correspondingly demands more compute to reach optimal performance.
- Validation is currently limited to text, discretized images, and molecules; scaling to larger language models and continuous signals remains to be explored.

## Related Work & Insights
- **vs. DDIM**: DDIM achieves flexible sampling in the continuous domain via non-Markovian processes; ReMDM transfers this idea to absorbing-state diffusion in the discrete domain.
- **vs. FB/DFM correctors**: ReMDM is strictly more general, handling the case where $\alpha_t$ is constant (via the loop strategy), which these correctors cannot accommodate.
- **vs. MaskGiT**: MaskGiT decodes based on model confidence but does not permit remasking; ReMDM's confidence-based schedule integrates the advantages of both approaches.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Elegantly addresses a core limitation of discrete diffusion from a probabilistic modeling perspective, with complete theoretical derivations.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers text, image, and molecular design, with additional validation on large-scale dLLMs such as LLaDA.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clear, theoretical derivations are rigorous, and experimental presentation is thorough.
- Value: ⭐⭐⭐⭐⭐ Improves discrete diffusion model performance at zero training cost, with direct impact on the dLLM community.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Inference-Time Scaling for Flow Models via Stochastic Generation and Rollover Budget Forcing](inference-time_scaling_for_flow_models_via_stochastic_generation_and_rollover_bu.md)
- [\[NeurIPS 2025\] Learnable Sampler Distillation for Discrete Diffusion Models](learnable_sampler_distillation_for_discrete_diffusion_models.md)
- [\[NeurIPS 2025\] Non-Markovian Discrete Diffusion with Causal Language Models](non-markovian_discrete_diffusion_with_causal_language_models.md)
- [\[NeurIPS 2025\] Progressive Inference-Time Annealing of Diffusion Models for Sampling from Boltzmann Densities](progressive_inference-time_annealing_of_diffusion_models_for_sampling_from_boltz.md)
- [\[NeurIPS 2025\] Constrained Discrete Diffusion](constrained_discrete_diffusion.md)

<!-- RELATED:END -->

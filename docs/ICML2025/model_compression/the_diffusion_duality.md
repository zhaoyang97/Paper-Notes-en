---
title: >-
  [Paper Note] The Diffusion Duality
description: >-
  [ICML 2025][Model Compression][Discrete Diffusion Duality] Reveals that Uniform-state discrete diffusion processes inherently emerge from underlying Gaussian diffusion (via the argmax mapping). Leveraging this duality, curriculum learning strategies and consistency distillation from Gaussian diffusion are transferred to the discrete setting, achieving a 2x training speedup and acceleration of sampling by two orders of magnitude (from 1024 to 8 steps)…
tags:
  - "ICML 2025"
  - "Model Compression"
  - "Discrete Diffusion Duality"
  - "Uniform-state Diffusion"
  - "Curriculum Learning"
  - "Discrete Consistency Distillation"
  - "Few-step Text Generation"
date: 2026-05-08
content_hash: e44d44735c21a7f7
---

# The Diffusion Duality

**Conference**: ICML 2025  
**arXiv**: [2506.10892](https://arxiv.org/abs/2506.10892)  
**Code**: [https://s-sahoo.com/duo](https://s-sahoo.com/duo) (Yes, with checkpoints)  
**Area**: Discrete Diffusion Models / Language Modeling  
**Keywords**: Discrete Diffusion Duality, Uniform-state Diffusion, Curriculum Learning, Discrete Consistency Distillation, Few-step Text Generation

## TL;DR
Reveals that Uniform-state discrete diffusion processes inherently emerge from underlying Gaussian diffusion (via the argmax mapping). Leveraging this duality, curriculum learning strategies and consistency distillation from Gaussian diffusion are transferred to the discrete setting, achieving a 2x training speedup and acceleration of sampling by two orders of magnitude (from 1024 to 8 steps), outperforming autoregressive models on 3 out of 7 datasets in zero-shot perplexity.

## Background & Motivation

**Background**: Discrete diffusion models (e.g., MDLM, SEDD) have demonstrated potential in text generation, but their performance consistently lags behind autoregressive (AR) models. Currently, there are two main categories of discrete diffusion: Masked Diffusion Models (MDM, using a [MASK] token) and Uniform-state Diffusion Models (USDM, where tokens can transition to any token in the vocabulary).

**Limitations of Prior Work**:
   - **Poor performance of USDMs**: Although USDMs inherently possess self-correction capabilities (tokens can be modified at any step), they have historically underperformed compared to MDMs.
   - **High training variance**: The training variance of the ELBO in discrete diffusion is much larger than that in Gaussian diffusion, leading to slow convergence.
   - **Inability to perform few-step generation**: MDMs lack a Probability Flow ODE (due to the deterministic mask prior) and cannot use techniques like consistency distillation; such techniques have not been developed for USDMs either.
   - **Technical gap with Gaussian diffusion**: Gaussian diffusion enjoys a rich set of acceleration techniques (efficient parameterizations, fast samplers, distillation), but these techniques cannot be directly transferred to discrete diffusion.

**Key Challenge**: The design space of discrete diffusion models remains primitive—still using mean parameterization and slow ancestral sampling—whereas Gaussian diffusion has accumulated over 15 years of methodologies.

**Goal**: (1) Establish a theoretical connection between discrete and Gaussian diffusion, (2) thereby transfer the efficient training and sampling techniques of Gaussian diffusion to the discrete setting.

**Key Insight**: Mathematically, it is discovered that the argmax operation transforms a Gaussian diffusion process into a Uniform-state discrete diffusion process—this is not an approximation, but an exact mathematical relationship.

**Core Idea**: Discrete diffusion is an "emergent phenomenon" of Gaussian diffusion (Diffusion Duality). Utilizing this duality allows for the free adoption of the Gaussian diffusion toolbox.

## Method

### Overall Architecture

The Duo framework is established on the following theoretical discovery:

Given the Gaussian diffusion latent variable $\mathbf{w}_t \sim \mathcal{N}(\tilde{\alpha}_t \mathbf{x}, (1-\tilde{\alpha}_t^2)\mathbf{I}_K)$, defining $\mathbf{z}_t = \arg\max(\mathbf{w}_t)$, then $\mathbf{z}_t$ follows a Uniform-state discrete diffusion:

$$\mathbf{z}_t \sim \text{Cat}(\cdot; \mathcal{T}(\tilde{\alpha}_t)\mathbf{x} + (1-\mathcal{T}(\tilde{\alpha}_t))\mathbf{1}/K)$$

where $\mathcal{T}$ is the **diffusion transformation operator** that maps the Gaussian diffusion parameter $\tilde{\alpha}_t$ to the discrete diffusion parameter $\alpha_t$.

### Key Designs

1. **Mathematical Establishment of Diffusion Duality**:

    - **Marginal Distribution Correspondence**: The argmax maps Gaussian marginals to categorical distributions, with parameters linked via the $\mathcal{T}$ operator:
    $\mathcal{T}(\tilde{\alpha}_t) = \frac{K}{K-1}\left[\int_{-\infty}^{\infty} \phi\left(\frac{z-\tilde{\alpha}_t}{\sqrt{1-\tilde{\alpha}_t^2}}\right)\Phi^{K-1}(z)dz - \frac{1}{K}\right]$
    - **Transition Dynamics Correspondence**: The temporal evolution of discrete marginals satisfies the pairwise transition matrix $Q_t$ of Uniform-state diffusion.
    - **ELBO Relationship** (Theorem 3.1): The ELBO of discrete diffusion is **strictly tighter** than the ELBO of the underlying Gaussian diffusion.
    - **Why it matters**: This implies that discrete spaces are more suitable for modeling than continuous spaces—training in the discrete space yields a tighter likelihood lower bound.

2. **Curriculum Learning to Accelerate Training**:

    - Core Idea: Use a tempered softmax ($\tau > 0$) instead of argmax ($\tau \to 0$) to relax discretization.
    - Training Loss:
    $\mathcal{L}_{train} = \mathbb{E}_{t, \tilde{q}_t} \sum_{\ell} f_{Duo}(\mathbf{z}_t^\ell := \arg\max(\mathbf{w}_t^\ell), \mathbf{x}_\theta([\text{softmax}(\mathbf{w}_t^{\ell'}/\tau)]^L_{\ell'=1}, t), \alpha_t; \mathbf{x}^\ell)$
    - Curriculum Strategy: $\tau = 0.001$ (first 500K steps) $\to \tau = 0$ (subsequent 500K steps).
    - Simultaneously, training is restricted to the time window $t \in [\beta, \gamma]$ to avoid regions with extremely weak gradient signals.
    - **Why it works**: argmax is extremely sensitive to small perturbations—tiny Gaussian noise causes drastic token flips. Tempered softmax preserves more continuous signals, reducing the difficulty of denoising $\to$ reducing gradient variance (observed to be reduced by an order of magnitude) $\to$ resulting in 2× faster training convergence.
    - Additional Optimization: **Rao-Blackwellized ELBO** avoids materializing one-hot vectors, reducing memory overhead and further lowering variance.

3. **Discrete Consistency Distillation (DCD)**:

    - Challenge: The discrete space lacks a Probability Flow ODE, making direct application of consistency distillation impossible.
    - Solution: Leverage duality to construct Deterministic Discrete Trajectories (DDT) using the optimal denoiser in the **Gaussian space**:
    $\mathcal{P}_{DDT}(\mathbf{x}^{1:L}, \epsilon^{1:L}) = \{[\arg\max(\tilde{\alpha}_t \mathbf{x}^\ell + \sqrt{1-\tilde{\alpha}_t^2}\epsilon^\ell)]^L_{\ell=1}\}_{t \in [0,1]}$
    - Distillation Loss: The student model $\mathbf{x}_\theta$ matches the prediction of the teacher model $\mathbf{x}_{\theta^-}$ on adjacent points along a DDT:
    $\mathcal{L}_{DCD} = D_{KL}(\mathbf{x}_\theta(\mathbf{z}_t^{1:L}, t) \| \mathbf{x}_{\theta^-}(\mathbf{z}_s^{1:L}, s))$
    - Distillation Process: $N=5$ rounds, with $M=10K$ steps per round, doubling the step size $\delta$ each round.
    - **Greedy-Tail Sampler**: Uses greedy decoding instead of sampling at the final step, further reducing the number of function evaluations (NFE) (1024 $\to$ 8 steps).
    - **Why DDT works**: Although there is no deterministic ODE trajectory in the discrete space, a "pseudo-deterministic" trajectory can be constructed by projecting the Gaussian space ODE via argmax, where tokens generally flip only once, behaving similarly to the carry-over behavior of MDMs.

### Loss & Training

- Base Model: Improved DiT with 170M parameters, RoPE positional encodings, AdaLN time-conditioning.
- Training: 8× H100 with bfloat16, batch size 512, learning rate 3e-4, 1M steps.
- Data: LM1B (context 128) and OpenWebText (context 1024).
- Distillation: Sampling in float64 precision (to avoid artificially low Gen PPL caused by low precision numerical errors).

## Key Experimental Results

### Main Results: Language Modeling Perplexity (LM1B + OWT)

| Method | Type | LM1B PPL ↓ | OWT PPL ↓ |
|------|------|-----------|-----------|
| Transformer | Autoregressive | 22.3 | 17.5 |
| MDLM | Absorbing Diffusion | 27.0 | 23.2 |
| SEDD Absorb | Absorbing Diffusion | 32.7 | 24.1 |
| SEDD Uniform | Uniform Diffusion | 40.3 | 29.7 |
| UDLM | Uniform Diffusion | 31.3 | 27.4 |
| **Duo (Ours)** | **Uniform Diffusion** | **29.9** | **25.2** |

Duo significantly narrows the gap between USDM and MDM/AR.

### Ablation Study

| Configuration | OWT PPL ↓ | Description |
|------|----------|------|
| Duo (Full) | 33.7 (LM1B w/ packing) | Combined effect of both optimizations |
| W/o Curriculum Learning | 35.0 | +1.3 PPL, curriculum learning contributes ~1.3 |
| W/o Rao-Blackwell | 36.7 | +1.7 PPL, improved training loss contributes ~1.7 |

### Zero-Shot Perplexity (OWT training $\to$ 7 datasets)

| Method | PTB | Wiki | LM1B | Lambada | AG News | Pubmed | Arxiv |
|------|-----|------|------|---------|---------|--------|-------|
| AR Transformer | 82.1 | 25.8 | 51.3 | 51.3 | 52.1 | 49.0 | 41.7 |
| MDLM | 95.3 | 32.8 | 67.0 | 47.5 | 61.2 | 41.9 | 37.4 |
| **Duo** | 89.4 | 33.6 | 73.9 | **49.8** | 67.8 | **44.5** | **40.4** |

Duo **outperforms the AR model on three datasets**: Lambada, Pubmed, and Arxiv (indicated in bold).

### Sample Quality after Distillation (Gen PPL ↓)

| Sampling Steps | MDLM+SDTT | Duo+DCD (Ancestral) | Duo+DCD (Greedy-Tail) |
|---------|-----------|--------------------|-----------------------|
| 1024 | 36.9 | 50.6 | **36.5** |
| 128 | 42.0 | 54.2 | **40.1** |
| 32 | 62.3 | 61.3 | **46.3** |
| 16 | 89.2 | 75.2 | **54.1** |
| 8 | 193.1 | 111.9 | **69.6** |

Duo significantly outperforms MDLM in the low-NFE regime—because USDM's self-correction property is highly valuable with fewer steps.

### Key Findings
- **Curriculum learning reduces gradient variance by an order of magnitude**: The gradient variance of the top-100 weights decreases from ~55 to ~0.86 (at 100K steps).
- **USDM is naturally superior in few-step generation**: Once MDM unmasks a token, it cannot modify it, meaning errors cannot be corrected with few steps. USDMs can update any token at every step.
- **DDT trajectories resemble MDM**: Tokens basically flip only once (starting as random tokens and then becoming the correct ones), yet maintain the flexibility of USDM.
- **Directly using denoised weights (non-EMA) as the teacher yields better results**: This differs from the standard practice in consistency models.

## Highlights & Insights
- **Profound Theory**: "The eternal theme in mathematics—discreteness emerges from the underlying continuity". This discovery elegantly unifies continuous and discrete diffusion.
- **High Practical Value**: 2× training speedup and 128× sampling acceleration (1024 $\to$ 8 steps) with no degradation in performance, but rather an improvement.
- **Rediscovering the Potential of USDM**: Previously considered inferior to MDM, USDM is proved to be stronger in few-step sampling scenarios.
- **Rao-Blackwellized ELBO**: Analytically eliminates the materialization of one-hot vectors, saving memory and decreasing variance, representing a highly practical engineering optimization.

## Limitations & Future Work
- The model scale is limited to 170M parameters and has not yet been validated at the 1B+ scale.
- The generated text quality (Gen PPL ~70) still lags behind AR models (~22).
- DCD distillation requires multiple rounds of training (5 rounds × 10K steps), introducing non-negligible training costs.
- The Greedy-Tail sampler reduces diversity (entropy drops from 5.55 to 5.19–5.30).
- The $\mathcal{T}$ operator requires precomputing 100K $(\tilde{\alpha}_t, \mathcal{T}(\tilde{\alpha}_t))$ pairs, increasing implementation complexity.

## Related Work & Insights
- **MDLM/SEDD/UDLM**: Primary baselines. Duo theoretically unifies and outperforms them.
- **Consistency Models**: Distillation techniques for Gaussian diffusion. DCD is their generalization to the discrete space.
- **Block Diffusion**: Concurrent work that interpolates between AR and diffusion.
- **Insights**: The duality framework could potentially apply to other discrete generative tasks (graphs, molecules, proteins); curriculum learning strategies are applicable to any discrete diffusion model.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Duality is a profound theoretical find; DCD bridges the gap in few-step generation for discrete diffusion.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive likelihood evaluation, zero-shot evaluation, generation quality evaluation, ablation, and distillation round analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous and complete theoretical derivations, in-depth experimental analysis, and exhaustive appendices.
- Value: ⭐⭐⭐⭐⭐ Significantly advances the field of discrete diffusion, opening the door to transferring techniques from Gaussian diffusion.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Diffusion Sampling Correction via Approximately 10 Parameters](diffusion_sampling_correction_via_approximately_10_parameters.md)
- [\[CVPR 2025\] EfficientViM: Efficient Vision Mamba with Hidden State Mixer based State Space Duality](../../CVPR2025/model_compression/efficientvim_efficient_vision_mamba_with_hidden_state_mixer_based_state_space_du.md)
- [\[ICLR 2026\] SSDi8: Accurate and Efficient 8-bit Quantization for State Space Duality](../../ICLR2026/model_compression/ssdi8_accurate_and_efficient_8-bit_quantization_for_state_space_duality.md)
- [\[CVPR 2025\] Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers](../../CVPR2025/model_compression/q-dit_accurate_post-training_quantization_for_diffusion_transformers.md)
- [\[NeurIPS 2025\] One-Step Diffusion-Based Image Compression with Semantic Distillation](../../NeurIPS2025/model_compression/one-step_diffusion-based_image_compression_with_semantic_distillation.md)

</div>

<!-- RELATED:END -->

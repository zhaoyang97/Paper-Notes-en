---
title: >-
  [Paper Note] Unified Masked Diffusion Models with Diverse Generation Orders
description: >-
  [ICML 2026][Image Generation][Masked Diffusion Models] This paper proposes a unified framework, OeMDM, and its learnable counterpart, LoMDM, which unify random masking, autoregressive…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Masked Diffusion Models"
  - "Generation Order"
  - "Velocity Field"
  - "Joint Learning"
date: 2026-05-08
content_hash: e6f1e3734871d22a
---

# Unified Masked Diffusion Models with Diverse Generation Orders

**Conference**: ICML 2026  
**arXiv**: [2602.02112](https://arxiv.org/abs/2602.02112)  
**Code**: TBD  
**Area**: Diffusion Models / Text Generation / Language Modeling  
**Keywords**: Masked Diffusion Models, Generation Order, Velocity Field, Joint Learning

## TL;DR
This paper proposes a unified framework, OeMDM, and its learnable counterpart, LoMDM, which unify random masking, autoregressive, and block diffusion models under a single NELBO by explicitly modeling "velocity" (generation priority), enabling joint learning of generation orders and the diffusion backbone from scratch.

## Background & Motivation

**Background**: Masked Diffusion Models (MDMs) are potential alternatives to Autoregressive Models (ARMs), but their generation quality depends heavily on the generation order.

**Limitations of Prior Work**: Existing solutions either use hard-coded orders (e.g., block-wise L2R) or learn order policies for pre-trained MDMs—the latter requires additional computation and leads to sub-optimal solutions due to two-stage optimization.

**Key Challenge**: MDMs are inherently order-agnostic; uniform noise scheduling ensures all positions denoise at the same rate, resulting in entirely random generation orders. Existing ordered approaches operate independently without a unified perspective.

**Goal**: (1) Unify MDMs, ARMs, and block diffusion under a single framework; (2) Enable joint end-to-end learning of generation orders and diffusion models from scratch.

**Key Insight**: Explicitly represent the generation rate implicit in the NELBO as a "velocity" function and design position-dependent adaptive noise scheduling.

**Core Idea**: Replace global fixed scheduling with position-dependent schedulers so the diffusion process "knows" which positions to generate first. Optimize the backbone and generation strategy simultaneously via a velocity matching loss.

## Method

### Overall Architecture
OeMDM introduces a free-form scheduler $\alpha_F: I \times [0,1] \to [0,1]^L$, allowing different positions to receive varying amounts of noise during the forward process. The NELBO is decomposed into a reconstruction loss and a velocity mismatch loss. LoMDM parameterizes the forward and backward velocities using neural networks $\phi$ and $\psi$ to achieve end-to-end joint learning.

### Key Designs

1.  **Velocity Field Explicitization**:
    - **Function**: Materializes the implicit generation order into an optimizable function $A(u,t) = -\partial_t\alpha_F(u,t) \oslash (1-\alpha_F(u,t))$, representing the denoising speed of position $i$ at time $t$.
    - **Mechanism**: The reverse posterior and denoising process can be reformulated as $\text{Cat}((1-A^{(i)}dt)m + A^{(i)}dt \cdot x^{(i)})$, where positions with higher velocity are restored earlier.
    - **Design Motivation**: To solve the order-agnosticism of MDMs—explicit velocity focuses training signals on high-priority tokens and allows for a principled NELBO decomposition.

2.  **Generalized NELBO Decomposition**:
    - **Function**: Decomposes the OeMDM objective function into $L_{\text{main}} + L_{\text{velocity}}$, where $L_{\text{main}}$ is the reconstruction loss weighted by velocity, and $L_{\text{velocity}} = A(i)(\log A(i) - \log \hat{A}(i)) - (A(i) - \hat{A}(i)) \geq 0$.
    - **Mechanism**: $L_{\text{velocity}}$ becomes zero when the forward velocity $A_\phi$ and backward velocity $\hat{A}_\psi$ are aligned, forcing both to learn the same generation order.
    - **Design Motivation**: To unify training and inference—the order learned during training is directly applied during generation, avoiding two-stage optimization.

3.  **Parameter-Efficient Joint Learning**:
    - **Function**: Reuses the Transformer feature extractor of the diffusion backbone $\theta$ and parameterizes $\alpha_\phi(x,t)$ and $\hat{\alpha}_\psi(z_t,t)$ using lightweight MLP+Transformer layers.
    - **Mechanism**: $\alpha^{(i)}_\phi(x,t) := 1 - t^{c_1 + c_2 \cdot [\text{NormSig}(g_\phi(f(x)))]_i}$, using a normalized Sigmoid output to modulate relative priority.
    - **Design Motivation**: To prevent optimization instability from excessive parameters; employs stop-gradient techniques to ensure the scheduler is optimized independently.

## Key Experimental Results

### Main Results

| Dataset | MDLM | BD3LM(L'=4) | GenMD4 | LoMDM | Gain |
|---------|------|-------------|--------|-------|------|
| LM1B    | 27.0 | -           | 26.9   | 25.4  | -1.5 vs MDLM |
| LM1B+packed | 31.8 | 28.2 | 30.0 | 27.2 | -4.6 vs MDLM |
| OWT     | 23.2 | 20.7        | 21.8   | 20.4  | -2.8 vs MDLM |

### Zero-shot Generalization

| Dataset | MDLM | BD3LM | LoMDM | vs MDLM |
|---------|------|-------|-------|---------|
| PTB     | 95.26 | 96.81 | 80.40 | ↓14.86 |
| WikiText | 32.83 | 31.31 | 27.82 | ↓5.01 |
| Lambada | 47.52 | 50.03 | 36.32 | ↓11.20 |

### Key Findings
- LoMDM outperforms MDLM on 7/7 zero-shot datasets and lead all diffusion models on 6/7; it beats autoregressive Transformers on 4/7 datasets.
- Generation PPL (NFE=256): LoMDM achieves 73.98 vs MDLM's 79.43.
- Ablation Study: Disabling inference scheduling ($c_2=0$) results in a generation PPL increase from 48.29 to 59.34.

## Highlights & Insights
- **Unified Perspective**: Interprets ARMs, MDMs, and block diffusion as special cases of OeMDM under different schedules, all explained within a single NELBO framework.
- **Velocity Matching**: The convex form of $L_{\text{velocity}} \geq 0$ ensures optimization stability while strictly enforcing consistency between training and inference.
- **End-to-End Joint Learning**: Unlike GenMD4, which uses a frozen backbone with a learned scheduler, LoMDM performs synchronous optimization, allowing the backbone to receive order-aware training signals.

## Limitations & Future Work
- Training Cost: Each iteration requires three forward passes, leading to slightly lower absolute throughput.
- Scheduler Design: The parameterization of $\alpha_\phi(x,t)$ still relies on manually designed functional forms.
- Scalability: Experiments were limited to LM1B/OWT scales; performance on larger models remains to be verified.

## Related Work & Insights
- **vs MDLM/SEDD**: Both utilize random masking but lack order optimization; LoMDM achieves context-aware generation paths via explicit schedulers.
- **vs BD3LM**: Employs a hard-coded L2R block structure; LoMDM learns more flexible and adaptive orders.
- **vs GenMD4**: Both learn schedulers, but GenMD4 is two-stage; LoMDM provides end-to-end optimization from scratch.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to unify discrete diffusion and autoregression via velocity fields.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 3 datasets, 3 evaluation metrics, and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Derivations are complete and clear.
- Value: ⭐⭐⭐⭐⭐ Provides a principled framework for discrete diffusion within text generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Unlearning in Diffusion Models: A Unified Framework Based on KL Divergence and Likelihood Constraints](unlearning_in_diffusion_models_a_unified_framework_with_kl_divergence_and_likeli.md)
- [\[ICML 2026\] A Unified Framework for Diffusion Model Unlearning with f-Divergence](a_unified_framework_for_diffusion_model_unlearning_with_f-divergence.md)
- [\[ICCV 2025\] LoRAverse: A Submodular Framework to Retrieve Diverse Adapters for Diffusion Models](../../ICCV2025/image_generation/loraverse_a_submodular_framework_to_retrieve_diverse_adapters_for_diffusion_mode.md)
- [\[ICCV 2025\] Latent Diffusion Models with Masked AutoEncoders](../../ICCV2025/image_generation/latent_diffusion_models_with_masked_autoencoders.md)
- [\[ICML 2026\] Initialization is Half the Battle: Generating Diverse Images from a Guidance Potential Posterior](initialization_is_half_the_battle_generating_diverse_images_from_a_guidance_pote.md)

</div>

<!-- RELATED:END -->

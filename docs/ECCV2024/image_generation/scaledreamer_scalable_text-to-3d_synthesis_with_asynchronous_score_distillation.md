---
title: >-
  [Paper Note] ScaleDreamer: Scalable Text-to-3D Synthesis with Asynchronous Score Distillation
description: >-
  [ECCV 2024][Image Generation][Text-to-3D] This paper proposes Asynchronous Score Distillation (ASD), which reduces noise prediction error and aligns the distribution of rendered images by shifting diffusion timesteps forward (rather than fine-tuning the diffusion model). This addresses the issue of VSD fine-tuning destroying text comprehension capabilities, thereby achieving stable training and prompt-amortized 3D generator training scalable to 100,000 text prompts.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Text-to-3D"
  - "Score Distillation"
  - "Diffusion Models"
  - "Asynchronous Timesteps"
  - "Large-Scale 3D Generation"
date: 2026-05-08
content_hash: 029728298ae6a943
---

# ScaleDreamer: Scalable Text-to-3D Synthesis with Asynchronous Score Distillation

**Conference**: ECCV 2024  
**arXiv**: [2407.02040](https://arxiv.org/abs/2407.02040)  
**Code**: [https://github.com/theEricMa/ScaleDreamer](https://github.com/theEricMa/ScaleDreamer)  
**Area**: Image Generation  
**Keywords**: Text-to-3D, Score Distillation, Diffusion Models, Asynchronous Timesteps, Large-Scale 3D Generation

## TL;DR
This paper proposes Asynchronous Score Distillation (ASD), which reduces noise prediction error and aligns the distribution of rendered images by shifting diffusion timesteps forward (rather than fine-tuning the diffusion model). This addresses the issue of VSD fine-tuning destroying text comprehension capabilities, thereby achieving stable training and prompt-amortized 3D generator training scalable to 100,000 text prompts.

## Background & Motivation

**Background**: Text-to-3D methods leverage the prior knowledge of pretrained 2D diffusion models to optimize 3D representations through Score Distillation. Prompt-specific methods (SDS, VSD) require separate optimization for hours per prompt; prompt-amortized methods train a text-to-3D generation network to enable inference in seconds, but rely heavily on efficient score distillation.

**Limitations of Prior Work**:
   - **SDS**: Assumes the rendered images follow a Dirac distribution, leading to extremely large gradients (requiring CFG=100) and numerical instability during training, causing deep networks like 3DConv-Net to collapse within a few thousand steps.
   - **CSD**: Replaces the term with an unconditional diffusion term, but since the unconditional term does not depend on text, it fails to provide effective gradients for different prompts.
   - **VSD**: Fine-tunes the diffusion model to align the distribution of rendered images, but this fine-tuning **destroys the pretrained model's text comprehension capability**, leading to mode collapse under large-scale prompts.

**Key Challenge**: The core objective of VSD is to reduce noise prediction errors to align distributions, but it achieves this by fine-tuning the diffusion model. This forms a bi-level optimization problem (resembling the alternating training of GANs), which is unstable under large numbers of prompts and compromises text comprehension.

**Goal**: Can the noise prediction error be reduced **without changing the weights of the diffusion model**?

**Key Insight**: The authors discovered that diffusion models naturally have lower noise prediction errors at **earlier timesteps** (larger $t$, closer to pure noise). This is because when $t \to T_{max}$, $\mathbf{x}_t \to \boldsymbol{\epsilon}$, allowing the model to accurately predict the noise by "copying the input as the output".

**Core Idea**: Shifting the diffusion timestep forward by $\Delta t$ approximates the effect of VSD fine-tuning. This achieves distribution alignment without changing the weights of the diffusion model, thus preserving its powerful text comprehension capabilities.

## Method

### Overall Architecture
ASD shares the same text-to-3D pipeline with SDS/VSD: text $\to$ 3D generator $\to$ rendered images $\to$ noise addition $\to$ diffusion model noise prediction $\to$ gradient computation $\to$ 3D generator update. The key difference lies in the gradient computation. ASD adds noise to and predicts noise from the rendered images at two asynchronous timesteps ($t$ and $t + \Delta t$) separately, using the difference between the two predictions as the gradient.

### Key Designs

1. **Asynchronous Score Distillation Objective**:

    - **Function**: Replaces the diffusion model fine-tuning in VSD with timestep forward shifting.
    - **Mechanism**: $\nabla_\theta \mathcal{L}_{ASD} = \mathbb{E}_{t, \boldsymbol{\epsilon}} [\omega(t)(\boldsymbol{\epsilon}_\phi(\mathbf{x}_t; t, y) - \boldsymbol{\epsilon}_\phi(\mathbf{x}_{t+\Delta t}; t+\Delta t, y)) \frac{\partial \mathbf{x}}{\partial \theta}]$
    - **Comparison**: SDS uses $\boldsymbol{\epsilon}$ (ground-truth noise), CSD uses $\boldsymbol{\epsilon}_\phi(\mathbf{x}_t; t)$ (unconditional prediction), VSD uses $\boldsymbol{\epsilon}_{\phi'}$ (fine-tuned model prediction), and ASD uses $\boldsymbol{\epsilon}_\phi(\mathbf{x}_{t+\Delta t}; t+\Delta t, y)$ (prediction at the forward-shifted timestep).
    - **Advantages**: Freezes the weights of the diffusion model, eliminates bi-level optimization, and preserves text comprehension.

2. **Timestep Offset $\Delta t$ Configuration**:

    - **Function**: Dynamically sets the shift amount for different timesteps.
    - **Mechanism**: $\Delta t \sim \mathcal{U}[0, \eta(t - T_{min})]$, uniform random sampling.
    - **Design Motivation**: (a) $\Delta t$ increases as $t$ increases—because closer to $T_{max}$, the error curve becomes flatter, requiring a larger offset to match the error of the fine-tuned model; (b) random sampling rather than deterministic sampling—since the optimal offset varies across different training iterations, images, and prompts.
    - **Hyperparameter $\eta = 0.1$**: An excessively large $\eta$ degenerates into SDS, while an excessively small one fails to fully exploit the noise reduction effect brought by shifting forward.

3. **Compatibility with Multiple 3D Generators**:

    - **Function**: Validates that ASD, as a general score distillation method, can work with different 3D architectures.
    - **Three Generators**: Hyper-iNGP (Hypernetwork + Hash Encoding), 3DConv-Net (3D Convolutional Voxel), Triplane-Transformer (Triplane + Transformer).
    - **Two Diffusion Models**: Stable Diffusion and MVDream (Multi-view Diffusion).
    - **Design Motivation**: Proves that ASD does not rely on a specific architecture and is a truly general score distillation method.

### Loss & Training
No extra losses are introduced; the core is the ASD gradient itself. The CFG scale is set to 7.5 (whereas SDS requires 100), and the gradient magnitude is comparable to VSD.

## Key Experimental Results

### Prompt-Specific vs Prompt-Amortized (Hyper-iNGP, MG15)

| Setting | Method | CLIP Sim↑ | R@1↑ |
|------|------|-----------|------|
| Prompt-Specific (iNGP) | SDS | 0.288 | 1.000 |
| Prompt-Specific (iNGP) | VSD | 0.276 | 0.932 |
| Prompt-Specific (iNGP) | **ASD** | **0.289** | **1.000** |
| Prompt-Amortized (Hyper-iNGP) | ATT3D(SDS) | 0.195 | 0.468 |
| Prompt-Amortized (Hyper-iNGP) | SDS | 0.257 | 0.918 |
| Prompt-Amortized (Hyper-iNGP) | VSD | 0.259 | 0.987 |
| Prompt-Amortized (Hyper-iNGP) | **ASD** | **0.284** | **1.000** |

ASD achieves optimal performance on both prompt-specific and prompt-amortized settings, showing almost no performance degradation from specific to amortized.

### Large-Scale Scalability (3DConv-Net)

| Prompt Set | Method | Sim↑ | R@1↑ |
|--------|------|------|------|
| DF415 | SDS | × (Collapsed) | × (Collapsed) |
| DF415 | CSD | 0.176 | 0.062 |
| DF415 | VSD | 0.158 | 0.002 |
| DF415 | **ASD** | **0.237** | **0.276** |
| CP100k | CSD | 0.195 | 0.108 |
| CP100k | VSD | 0.103 | 0.000 |
| CP100k | **ASD** | **0.199** | **0.117** |

VSD completely collapses under 100k prompts (R@1=0.000), while ASD remains effective. SDS collapses immediately upon training on 3DConv-Net.

### Ablation Study ($\Delta t$ Configuration)

| Setting | Sim↑ | R@1↑ | Description |
|------|------|------|------|
| $\eta=0$ (No forward shifting) | 0.235 | 0.267 | Severe Janus problem |
| Deterministic $\Delta t = 0.1(t-T_{min})$ | 0.214 | 0.178 | Inaccurate geometry and color |
| Deterministic $\Delta t = 0.2(t-T_{min})$ | 0.214 | 0.180 | Similar to the row above |
| Random $\Delta t \sim \mathcal{U}[0, 0.1(t-T_{min})]$ | **0.237** | **0.276** | **Optimal** |
| Random $\Delta t \sim \mathcal{U}[0, 0.2(t-T_{min})]$ | 0.229 | 0.237 | Oversized and overly rounded shapes |

### Key Findings
- **Timestep offset is necessary**: Severe Janus problems are observed when $\eta=0$ (e.g., frogs with multiple eyes, peacocks with tails on both front and back). This occurs because the unaligned diffusion model tends to generate content that looks like a front view from every perspective.
- **Random sampling is far superior to deterministic sampling**: Deterministic offset yields an R@1 of only 0.178, whereas random sampling reaches 0.276.
- **$\eta$ cannot be too large**: When $\eta=0.2$, shapes skew towards being oversized and overly rounded, as an excessively large offset degenerates into SDS.
- **VSD suffers from mode collapse at scale**: At 100k prompts, R@1 drops to 0.000, and almost all outputs are identical, validating the hypothesis that fine-tuning destroys text comprehension.
- **SDS is fatal to deep networks**: Under 3DConv-Net, SDS consistently collapses, confirming the destructive nature of large gradients on deep networks.
- ASD combined with MVDream also outperforms SDS*, producing more natural geometry and textures.

## Highlights & Insights
- **The insight of forward shifting timesteps is highly inspiring**: The observation that diffusion models have lower noise prediction errors at earlier timesteps is simple yet profound. Utilizing this property to elegantly replace the fine-tuning process of VSD is highly appealing both theoretically and practically.
- **Freezing the diffusion model preserves text comprehension**: This design choice is crucial in large-scale training. While the bi-level optimization of VSD works at a small scale, it completely fails when scaled to 100k prompts—providing a valuable lesson for the scalable design of score distillation.
- **ASD is a truly general score distillation method**: It is compatible with three 3D generators and two diffusion models, requiring minimal code modifications (only altering the sampling timesteps), which offers high practical value.
- **CP100k Prompt Set**: Evaluation of score distillation at the 100,000 prompt scale is performed for the first time, providing a benchmark for future research.

## Limitations & Future Work
- For man-made objects with regular shapes (e.g., chairs, airplanes), performance still lags behind data-driven methods due to the lack of 3D training data.
- The absolute generation quality under 100k prompts still has room for improvement (R@1 is only 0.117), which remains far from practical applications.
- $\eta$ requires manual tuning, and its optimal value may vary across different diffusion models or generators.
- Comparison with other recent score distillation methods, such as ISM (Interval Score Matching), is lacking.
- The rendering resolution for 3DConv-Net and Triplane-Transformer is restricted to 64x64, which limits their practical visual performance.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The concept of shifting timesteps forward is novel and concise, with a clear and elegant theoretical motivation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluations across three generators, two diffusion models, and prompt scales ranging from 15 to 100,000.
- Writing Quality: ⭐⭐⭐⭐⭐ The analysis of the problem and the derivation of the method are logically rigorous, complemented by intuitive error curve designs.
- Value: ⭐⭐⭐⭐⭐ Successfully addresses the core bottleneck of score distillation in large-scale training, providing a substantial push forward for the text-to-3D field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Score Distillation Beyond Acceleration: Generative Modeling from Corrupted Data](../../ICLR2026/image_generation/score_distillation_beyond_acceleration_generative_modeling_from_corrupted_data.md)
- [\[ICCV 2025\] LUSD: Localized Update Score Distillation for Text-Guided Image Editing](../../ICCV2025/image_generation/lusd_localized_update_score_distillation_for_text-guided_image_editing.md)
- [\[ECCV 2024\] Editable Image Elements for Controllable Synthesis](editable_image_elements_for_controllable_synthesis.md)
- [\[ECCV 2024\] Learning Differentially Private Diffusion Models via Stochastic Adversarial Distillation](learning_differentially_private_diffusion_models_via_stochastic_adversarial_dist.md)
- [\[ECCV 2024\] RodinHD: High-Fidelity 3D Avatar Generation with Diffusion Models](rodinhd_high-fidelity_3d_avatar_generation_with_diffusion_models.md)

</div>

<!-- RELATED:END -->

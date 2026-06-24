---
title: >-
  [Paper Note] Visual Generation Without Guidance
description: >-
  [ICML 2025][Image Generation][Classifier-Free Guidance] This work proposes Guidance-Free Training (GFT), which parameterizes the conditional model as a linear interpolation between a sampling network and an unconditional network. This enables direct training of guidance-free visual generative models from data, halving the sampling computation while matching the performance of CFG on five models (DiT, VAR, LlamaGen, MAR, and LDM).
tags:
  - "ICML 2025"
  - "Image Generation"
  - "Classifier-Free Guidance"
  - "guidance-free generation"
  - "sampling efficiency"
  - "pseudo-temperature parameterization"
  - "distillation alternative"
date: 2026-05-08
content_hash: 19073161d6a2b29d
---

# Visual Generation Without Guidance

**Conference**: ICML 2025  
**arXiv**: [2501.15420](https://arxiv.org/abs/2501.15420)  
**Code**: [https://github.com/thu-ml/GFT](https://github.com/thu-ml/GFT)  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Classifier-Free Guidance, guidance-free generation, sampling efficiency, pseudo-temperature parameterization, distillation alternative

## TL;DR
This work proposes Guidance-Free Training (GFT), which parameterizes the conditional model as a linear interpolation between a sampling network and an unconditional network. This enables direct training of guidance-free visual generative models from data, halving the sampling computation while matching the performance of CFG on five models (DiT, VAR, LlamaGen, MAR, and LDM).

## Background & Motivation
**Background**: CFG is a standard technology in visual generation, which improves generation quality by running both conditional and unconditional models simultaneously during sampling, which directly doubles the inference computation.

**Limitations of Prior Work**: (a) Doubled inference costs; (b) complicated post-training processes (distillation and RLHF require special handling of unconditional models); (c) inconsistency with the simple temperature sampling used in LLMs.

**Key Challenge**: The sampling distribution of CFG, $p^s(\boldsymbol{x}|c) \propto p(\boldsymbol{x}|c)[p(\boldsymbol{x}|c)/p(\boldsymbol{x})]^s$, does not have a corresponding real-world dataset, precluding direct maximum likelihood training.

**Goal**: Can a single model achieve the quality-diversity trade-off of CFG?

**Key Insight**: Realigning the CFG formulation to express the conditional model as a weighted combination of a sampling model and an unconditional model, allowing direct learning of the sampling model.

**Core Idea**: CFG formulated as $\epsilon^c = \frac{1}{1+s}\epsilon^s + \frac{s}{1+s}\epsilon^u$, which directly optimizes $\epsilon^s$ enabling sampling without guidance.

## Method

### Overall Architecture
GFT maintains the same maximum likelihood training objective as CFG but parameterizes the conditional model differently: the conditional model is defined as an implicit linear combination of the sampling network $\epsilon_\theta^s$ and the unconditional network $\epsilon_\theta^u$. A pseudo-temperature $\beta = 1/(1+s)$ is introduced as an additional input to the model, allowing flexible adjustment during inference.

### Key Designs

1. **Implicit Conditional Parameterization**:

    - **Function**: Enables the training to directly optimize the sampling model $\epsilon_\theta^s$ instead of the conditional model $\epsilon_\theta^c$
    - **Mechanism**: $\epsilon_\theta^c(\boldsymbol{x}_t|\boldsymbol{c},\beta) = \beta \epsilon_\theta^s(\boldsymbol{x}_t|\boldsymbol{c},\beta) + (1-\beta) \epsilon_\theta^u(\boldsymbol{x}_t)$, where this implicit representation is trained using standard conditional loss
    - **Design Motivation**: Although $\epsilon^s$ cannot be learned directly without a dataset for $p^s$, $\epsilon^c$ is learnable, which allows $\epsilon^s$ to be optimized indirectly through it

2. **Stop-Gradient Trick**:

    - **Function**: Improves training efficiency and stability
    - **Mechanism**: The unconditional model $\epsilon_\theta^u$ is executed in eval mode with stop-gradient, propagating gradients only to $\epsilon_\theta^s$
    - **Design Motivation**: (a) Highly aligned with CFG training, differing only by one unconditional inference; (b) introduces negligible GPU memory overhead; (c) increases training time by only 19%

3. **Pseudo-Temperature $\beta$ Input**:

    - **Function**: Allows a single model to support sampling at different temperatures
    - **Mechanism**: $\beta \sim U(0,1)$ is randomly sampled and added to the time/class embeddings after being processed by a Fourier embedding and MLP
    - **Design Motivation**: $\beta=1$ is equivalent to standard conditional generation, while $\beta \to 0$ approaches low-temperature high-quality sampling

### Loss & Training
- Diffusion version: $\mathcal{L} = \|\beta\epsilon_\theta^s(\boldsymbol{x}_t|\boldsymbol{c}_\varnothing,\beta) + (1-\beta)\mathbf{sg}[\epsilon_\theta^u(\boldsymbol{x}_t|\varnothing,1)] - \boldsymbol{\epsilon}\|_2^2$
- AR/Masked version: Conditional logits $\ell_\theta^c = \beta \ell_\theta^s + (1-\beta)\mathbf{sg}[\ell_\theta^u]$, followed by computing the standard cross-entropy
- Fine-tuning pretrained CFG models requires only 1-5% of pretraining epochs, utilizing zero-initialization on the final MLP for $\beta$ to avoid impacting initial outputs

## Key Experimental Results

### Main Results

| Model | CFG FID ↓ | GFT FID ↓ | GFT Fine-tuning/Scratch |
|------|-----------|-----------|--------------|
| DiT-XL/2 | 2.11 | **1.99** | Fine-tune 2% epochs |
| DiT-XL/2 (Distillation) | 2.11 | - | - |
| VAR-d30 | - | Matches CFG | Train from scratch |
| LlamaGen | - | Matches CFG | Train from scratch |
| MAR | - | Matches CFG | Train from scratch |
| LDM (T2I) | - | Matches CFG | Fine-tune |

### Ablation Study

| Method | Domain | Extra Training Time | VRAM Increase | Trainable from Scratch |
|------|--------|------------|---------|-----------|
| Guidance Distillation | Diffusion only | ×1.19 | ×1.15 | ✗ |
| Contrastive Alignment | AR/Masked only | ×1.69 | ×1.39 | ✗ |
| **GFT (Ours)** | **All** | **×1.00** | **×1.00** | **✓** |

### Key Findings
- GFT fine-tuning on DiT-XL reaches an FID of 1.99 with only 2% epochs, outperforming the CFG baseline of 2.11.
- GFT is the only method that supports training from scratch and is compatible across all three model types: diffusion, AR, and masked models.
- Adjusting $\beta$ enables a diversity-fidelity trade-off comparable to CFG.

## Highlights & Insights
- **Extremely Simple Implementation**: Requires modifying only a few lines of existing CFG code, directly inheriting most hyperparameters.
- **Unity**: Integrates guidance-free training methods for diffusion, AR, and masked models for the first time.
- **Theoretical Guarantee**: Theorem 1 proves that the optimal solution of GFT is consistent with the target CFG sampling distribution.

## Limitations & Future Work
- The authors acknowledge that the sampling distribution of $\beta$ (uniform distribution) may not be optimal.
- Validation on large-scale T2I generation (such as SDXL level) remains insufficient.
- The integration with acceleration methods like consistency models has not been explored.

## Rating
- Novelty: ⭐⭐⭐⭐ The parameterization perspective is ingenious, though the core is mathematically a reformulation of CFG.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Full coverage across five models, class-conditional/text-conditional settings.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear derivation and fair experimental comparisons.
- Value: ⭐⭐⭐⭐⭐ Highly practical, with strong potential to become a standard method replacing CFG.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Learning Visual Generative Priors without Text](../../CVPR2025/image_generation/learning_visual_generative_priors_without_text.md)
- [\[ICML 2025\] Continuous Visual Autoregressive Generation via Score Maximization](continuous_visual_autoregressive_generation_via_score_maximization.md)
- [\[ICCV 2025\] StyleKeeper: Prevent Content Leakage using Negative Visual Query Guidance](../../ICCV2025/image_generation/stylekeeper_prevent_content_leakage_using_negative_visual_query_guidance.md)
- [\[CVPR 2025\] Rectified Diffusion Guidance for Conditional Generation](../../CVPR2025/image_generation/rectified_diffusion_guidance_for_conditional_generation.md)
- [\[ICML 2025\] Angle Domain Guidance: Latent Diffusion Requires Rotation Rather Than Extrapolation](angle_domain_guidance_latent_diffusion_requires_rotation_rather_than_extrapolati.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation
description: >-
  [Image Generation] SANA-Sprint proposes a hybrid distillation framework combining continuous-time consistency models (sCM) and latent adversarial diffusion distillation (LADD). It converts pretrained Flow Matching models to TrigFlow in a lossless manner and jointly trains with sCM+LADD, achieving unified adaptive high-quality text-to-image generation in 1–4 steps, with a single-step latency of only 0.1 seconds on H100.
tags:
  - Image Generation
date: 2026-05-08
content_hash: 99dd1ffbd6017781
---

# SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation

## Metadata
- **Conference**: ICCV 2025
- **arXiv**: [2503.09641](https://arxiv.org/abs/2503.09641)
- **Code**: [GitHub](https://github.com/NVlabs/Sana) | [HuggingFace Models](https://huggingface.co/Efficient-Large-Model/SANA-Sprint)
- **Institution**: NVIDIA, MIT, Tsinghua University, HuggingFace
- **Area**: Image Generation / Diffusion Model Distillation
- **Keywords**: One-step generation, continuous-time consistency models (sCM), latent adversarial diffusion distillation (LADD), Flow Matching → TrigFlow conversion, real-time interactive generation

## TL;DR

SANA-Sprint proposes a hybrid distillation framework combining continuous-time consistency models (sCM) and latent adversarial diffusion distillation (LADD). It converts pretrained Flow Matching models to TrigFlow in a lossless manner and jointly trains with sCM+LADD, achieving unified adaptive high-quality text-to-image generation in 1–4 steps, with a single-step latency of only 0.1 seconds on H100.

## Background & Motivation

Diffusion models typically require 50–100 iterative denoising steps, incurring substantial computational cost. Existing step-distillation methods exhibit notable limitations:

**GAN-based methods** (e.g., ADD, LADD): Training instability with mode collapse and adversarial dynamic oscillation; hyperparameter tuning is difficult.

**VSD-based methods**: Require jointly training an additional diffusion model, placing heavy demands on GPU memory.

**Consistency models (CM)**: Quality degrades significantly at very low step counts (<4 steps), particularly in text-to-image tasks where truncation errors cause semantic alignment deterioration.

These challenges motivate the design of a unified distillation framework that balances efficiency, flexibility, and quality. SANA-Sprint builds upon the pretrained SANA model and incorporates recent advances in continuous-time consistency models (sCM) to eliminate the discretization errors inherent in discrete-time consistency models.

## Method

### Overall Architecture

SANA-Sprint adopts a three-stage design:
1. **Training-free conversion**: Mathematically equivalent conversion of a Flow Matching model to a TrigFlow model.
2. **Stability optimization**: Addressing training instability via QK normalization and dense time embeddings.
3. **Hybrid distillation**: The sCM loss provides teacher alignment and diversity preservation, while the LADD adversarial loss enhances single-step fidelity.

### Key Design 1: Lossless Flow Matching → TrigFlow Conversion

Flow Matching and TrigFlow are misaligned in three aspects: time domain ([0,1] vs. [0,π/2]), noise schedule (inconsistent data scaling), and prediction target (static vs. time-varying coefficients).

The authors derive a lossless mathematical conversion:

$$t_{\text{FM}} = \frac{\sin(t_{\text{Trig}})}{\sin(t_{\text{Trig}}) + \cos(t_{\text{Trig}})}$$

$$x_{t,\text{FM}} = \frac{x_{t,\text{Trig}}}{\sigma_d} \cdot \sqrt{t_{\text{FM}}^2 + (1-t_{\text{FM}})^2}$$

The TrigFlow model output is obtained via linear combination:

$$\hat{F_\theta} = \frac{1}{\sqrt{t_{\text{FM}}^2 + (1-t_{\text{FM}})^2}} \left[ (1-2t_{\text{FM}}) x_{t,\text{FM}} + (1-2t_{\text{FM}}+2t_{\text{FM}}^2) v_\theta \right]$$

Experiments confirm that FID remains nearly unchanged before and after conversion (5.81 vs. 5.73), demonstrating losslessness in both theory and practice.

### Key Design 2: Stable Continuous-Time Distillation

- **Dense time embeddings**: The noise coefficient $c_{\text{noise}}(t)$ is changed from $1000t$ to $t$, preventing the time derivative $\partial_t F_{\theta^{-}}$ from being amplified by a factor of 1000, which would otherwise cause gradient explosion. PCA visualization shows that embeddings in the 0–1 range are more densely similar.
- **QK normalization**: RMS Normalization is applied to the Query and Key in both self-attention and cross-attention, resolving training collapse caused by gradient norms exceeding $10^3$ when scaling the model from 0.6B to 1.6B parameters. Only 5,000 iterations of teacher model fine-tuning are required.

### Key Design 3: sCM + LADD Hybrid Loss

sCM distills teacher knowledge in a local manner (learning consistency between adjacent timesteps) and converges slowly. LADD adversarial loss is introduced to provide global supervision across timesteps:

$$\mathcal{L} = \mathcal{L}_{\text{sCM}} + \lambda \mathcal{L}_{\text{adv}}, \quad \lambda = 0.5$$

- LADD uses the frozen teacher model as a feature extractor and trains multiple discriminator heads in the latent space.
- Discriminators employ hinge loss to distinguish real noisy samples from generated noisy samples.

### Additional Maximum Timestep Weighting

With probability $p$, the training timestep is set to $t = \pi/2$ (pure noise), reinforcing single-step generation capability. Experiments find that 50% probability is optimal.

### Loss & Training

sCM loss (continuous-time consistency):
$$\mathcal{L}_{\text{sCM}} = \mathbb{E}_{x_t, t} \left[ \frac{e^{w_\phi(t)}}{D} \left\| \hat{F_\theta} - \hat{F_{\theta^{-}}} - \cos(t) \frac{d\hat{f_{\theta^{-}}}}{dt} \right\|_2^2 - w_\phi(t) \right]$$

where $w_\phi(t)$ is an adaptive weighting function that minimizes variance across different timesteps.

## Key Experimental Results

### Main Results: Efficiency and Performance Comparison with SOTA Methods

| Method | Steps | Throughput (samples/s) | Latency (s) | Params (B) | FID ↓ | CLIP ↑ | GenEval ↑ |
|---|---|---|---|---|---|---|---|
| FLUX-schnell | 4 | 0.5 | 2.10 | 12.0 | 7.94 | 28.14 | 0.71 |
| SDXL-DMD2† | 4 | 2.27 | 0.54 | 0.9 | 6.82 | 28.84 | 0.60 |
| SD3.5-Turbo | 4 | 0.94 | 1.15 | 8.0 | 11.97 | 27.35 | 0.72 |
| **SANA-Sprint 0.6B** | **4** | **5.34** | **0.32** | **0.6** | **6.48** | **28.45** | **0.76** |
| **SANA-Sprint 1.6B** | **4** | **5.20** | **0.31** | **1.6** | **6.54** | **28.45** | **0.77** |
| FLUX-schnell | 1 | 1.58 | 0.68 | 12.0 | 7.26 | 28.49 | 0.69 |
| SDXL-DMD2† | 1 | 3.36 | 0.32 | 0.9 | 7.10 | 28.93 | 0.59 |
| **SANA-Sprint 0.6B** | **1** | **7.22** | **0.21** | **0.6** | **7.04** | **28.04** | **0.72** |
| **SANA-Sprint 1.6B** | **1** | **6.71** | **0.21** | **1.6** | **7.69** | **28.27** | **0.76** |

Key findings:
- 4-step SANA-Sprint 0.6B achieves **10.7×** the throughput of FLUX-schnell with better FID (6.48 vs. 7.94).
- Single-step SANA-Sprint 0.6B achieves a latency of only 0.21s and a GenEval score of 0.72, surpassing FLUX-schnell at the same step count.

### Ablation Study: Loss Combinations and Training Strategies

| Configuration | FID ↓ | CLIP ↑ |
|---|---|---|
| sCM only | 8.93 | 27.51 |
| LADD only | 12.20 | 27.00 |
| **sCM + LADD** | **8.11** | **28.02** |
| w/o CFG Embed | 9.23 | 27.15 |
| w/ CFG Embed | 8.72 | 28.09 |
| 0% maxT | 9.44 | 27.65 |
| 50% maxT | 8.32 | 27.94 |
| 70% maxT | 8.11 | 28.02 |
| sCM:LADD = 1.0:1.0 | 8.81 | 27.93 |
| sCM:LADD = 1.0:0.5 | 8.43 | 27.85 |
| sCM:LADD = 1.0:0.1 | 8.90 | 27.76 |

Key findings:
- The complementary effect of sCM + LADD is significant: FID drops from 12.20 (LADD only) to 8.11.
- CFG embedding improves CLIP score by 0.94.
- Maximum timestep weighting from 0% to 50% reduces FID from 9.44 to 8.32.

### Real-Time Interactive Generation

- With ControlNet integration, 1024×1024 image generation achieves a latency of 250ms on H100.
- Single-step generation takes 0.31s on a consumer-grade RTX 4090 GPU.

## Highlights & Insights

1. **Elegant training-free conversion**: The mathematically equivalent Flow Matching → TrigFlow transformation avoids the enormous cost of repretraining the teacher model and is compatible with automatic differentiation via differentiable mappings.
2. **Unified step-adaptive model**: A single model supports 1–4 steps without requiring step-specific training, unlike many competitors (e.g., SDXL-DMD2, PCM) that train separate models per step count.
3. **Complementarity of hybrid distillation**: sCM ensures alignment with the teacher distribution and diversity preservation, while LADD enhances single-step fidelity; the two objectives are orthogonally complementary.
4. **Engineering insights**: Dense time embeddings ($c_{\text{noise}}=t$ rather than $1000t$) and QK-Norm are identified as critical stabilization techniques for training large-scale continuous-time consistency models.

## Limitations & Future Work

1. Validation is primarily conducted on the SANA architecture; although generalizability to Flow Matching models such as FLUX and SD3 is claimed, no experimental evidence is provided.
2. Generation quality has not fully reached the level of the 20-step teacher model, particularly in complex semantic scenes.
3. The LADD discriminator increases training complexity; although lighter than VSD, there remains room for further optimization.
4. ControlNet is validated only under HED edge conditioning; the effectiveness of other condition types (depth, segmentation, etc.) remains unknown.

## Related Work & Insights

- **sCM** [Lu et al., 2024]: Theoretical foundation for continuous-time consistency models; SANA-Sprint directly adopts its training framework.
- **LADD** [Sauer et al., 2024]: Key reference for latent adversarial diffusion distillation, using teacher feature space to train discriminators.
- **SANA** [Xie et al., 2024]: Base model architecture enabling efficient image generation via linear attention transformers.
- **Inspiration**: For other generative tasks requiring few-step inference (video, 3D), the paradigm of Flow → TrigFlow conversion + hybrid distillation holds broad general value.

## Rating ⭐⭐⭐⭐⭐

This work delivers exceptional engineering value. The lossless Flow → TrigFlow conversion eliminates sCM's dependency on dedicated pretraining, significantly lowering the barrier to continuous-time consistency distillation. The innovative combination of hybrid distillation achieves a notable breakthrough on the speed–quality Pareto frontier. The results of 1-step FID 7.04 and GenEval 0.72 with only 0.6B parameters are impressive and have direct practical implications for deployment on consumer-grade GPUs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Joint Diffusion Models in Continual Learning](joint_diffusion_models_in_continual_learning.md)
- [\[ICCV 2025\] OminiControl: Minimal and Universal Control for Diffusion Transformer](ominicontrol_minimal_and_universal_control_for_diffusion_transformer.md)
- [\[ICCV 2025\] EDiT: Efficient Diffusion Transformers with Linear Compressed Attention](edit_efficient_diffusion_transformers_with_linear_compressed_attention.md)
- [\[ICCV 2025\] Timestep-Aware Diffusion Model for Extreme Image Rescaling](timestep-aware_diffusion_model_for_extreme_image_rescaling.md)
- [\[ICCV 2025\] FreeMorph: Tuning-Free Generalized Image Morphing with Diffusion Model](freemorph_tuning-free_generalized_image_morphing_with_diffusion_model.md)

</div>

<!-- RELATED:END -->

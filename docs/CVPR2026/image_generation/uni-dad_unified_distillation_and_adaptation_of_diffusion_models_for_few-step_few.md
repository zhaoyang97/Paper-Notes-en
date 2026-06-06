---
title: >-
  [Paper Note] Uni-DAD: Unified Distillation and Adaptation of Diffusion Models for Few-step Few-shot Image Generation
description: >-
  [CVPR 2026][Image Generation][Diffusion model distillation] This paper proposes Uni-DAD, the first method to unify diffusion model distillation and domain adaptation into a single-stage pipeline. Through a dual-domain DM…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Diffusion model distillation"
  - "few-shot image generation"
  - "domain adaptation"
  - "GAN"
  - "distribution matching distillation"
date: 2026-05-08
content_hash: bea5a55f57abe521
---

# Uni-DAD: Unified Distillation and Adaptation of Diffusion Models for Few-step Few-shot Image Generation

**Conference**: CVPR 2026
**arXiv**: [2511.18281](https://arxiv.org/abs/2511.18281)  
**Code**: [GitHub](https://github.com/yaramohamadi/uni-DAD)  
**Area**: Image Generation
**Keywords**: Diffusion model distillation, few-shot image generation, domain adaptation, GAN, distribution matching distillation

## TL;DR

This paper proposes Uni-DAD, the first method to unify diffusion model distillation and domain adaptation into a single-stage pipeline. Through a dual-domain DMD loss and a multi-head GAN loss, Uni-DAD achieves high-quality and diverse generation in few-shot target domains using only 1–4 sampling steps.

## Background & Motivation

Diffusion models (DMs) excel at image generation but suffer from two major bottlenecks: (1) sampling requires hundreds to thousands of iterative denoising steps, making inference extremely slow; and (2) even after adapting pretrained models to new domains (e.g., few-shot scenarios), the slow sampling problem persists.

Existing solutions follow two-stage pipelines:

- **Distill-then-Adapt**: The teacher is first distilled into a few-step student, which is then fine-tuned to the target domain. This is computationally friendly but saturates the student's adaptation capacity, producing overly smooth outputs.
- **Adapt-then-Distill**: The teacher is first fine-tuned to the target domain and then distilled. This yields better quality but binds the student to the adapted teacher's performance and is prone to overfitting in few-shot settings.

Both approaches are non-end-to-end and tend to lose source-domain diversity information during training. The authors argue that **distillation and adaptation need not be separated** and can be accomplished simultaneously in a single stage.

## Method

### Overall Architecture

Uni-DAD compresses a frozen source-domain teacher $\epsilon^{\text{src}}$ (pretrained on large-scale source data, $T \sim 1000$ steps) into a fast student generator $G$ ($1 \leq \text{NFE} \leq 4$), while simultaneously adapting to a target distribution $p^{\text{trg}}(y)$ represented by a small set of samples $Y$ ($|Y| \leq 10$).

Training involves alternating optimization of three model groups:

1. **Student $G$**: updated with dual-domain DMD and multi-head GAN generator losses.
2. **Fake teacher $\epsilon^{\text{fk}}$ + multi-head discriminator $D$**: the fake teacher tracks the student's distribution; the discriminator distinguishes real target samples from student generations.
3. **Target teacher $\epsilon^{\text{trg}}$ (optional)**: fine-tuned on target samples to provide target-domain score guidance.

### Dual-Domain Distribution Matching Distillation (Dual-domain DMD)

The core idea of DMD is to minimize the KL divergence between the student distribution $p^{\text{fk}}$ and the teacher distribution $p^{\text{src}}$. Its gradient can be approximated in noise estimation form as:

$$\nabla_{\theta} \mathcal{L}_{\text{DMD}^{\text{src}}} \approx \mathbb{E}_{t,z}\left[\omega_t \left(\epsilon^{\text{fk}}(x_t) - \epsilon^{\text{src}}(x_t)\right) \frac{dG_\theta}{d\theta}\right]$$

where $x = G(z),\; z \sim \mathcal{N}(0,I)$ and $t \sim \mathcal{U}\{0.02T, 0.98T\}$. $\epsilon^{\text{fk}}$ is an online fake teacher that tracks the student's output distribution, and $\epsilon^{\text{src}}$ is the frozen source-domain teacher.

The authors extend this to **dual-domain DMD**, simultaneously aligning the student to both the source and target domain distributions:

$$\nabla_{\theta} \mathcal{L}_{\text{DMD}}^{\text{trg}+\text{src}} = (1-a)\nabla_{\theta}\mathcal{L}_{\text{DMD}^{\text{src}}} + a\nabla_{\theta}\mathcal{L}_{\text{DMD}^{\text{trg}}}$$

The weighting factor $a \in [0,1]$ controls the relative influence of the source and target domains. The source term preserves diversity (pose, background, expression, etc.), while the target term guides structural adaptation. A small $a$ is used when the target domain is structurally close to the source (e.g., $a=0.25$ for Babies), and a larger $a$ for greater structural divergence (e.g., $a=0.75$ for MetFaces).

Weight normalization is defined as:

$$\omega_t = \frac{\sigma_t \cdot H \cdot S}{\|\epsilon - \epsilon^{\text{fk}}(x_t)\|_1}$$

where $H$ is the number of channels and $S$ is the number of spatial locations, ensuring balanced contributions across different timesteps.

### Fake Teacher and Target Teacher

**Fake teacher $\epsilon^{\text{fk}}$**: Initialized from $\epsilon^{\text{src}}$ weights and continuously trained on student-generated samples to track the student's evolving distribution:

$$\mathcal{L}_{\text{fk}}(\phi) = \mathbb{E}_{t,z}\left[\|\epsilon^{\text{fk}}_\phi(x_t) - \epsilon\|_2^2\right]$$

Gradients are not backpropagated through $G$ during this update; $x$ is treated as fixed.

**Target teacher $\epsilon^{\text{trg}}$ (optional)**: Initialized from $\epsilon^{\text{src}}$ and fine-tuned on target samples $Y$:

$$\mathcal{L}_{\text{trg}}(\eta) = \mathbb{E}_{t,\epsilon,y}\left[\|\epsilon^{\text{trg}}_\eta(y_t) - \epsilon\|_2^2\right]$$

When the target domain is structurally distant from the source, incorporating the target teacher helps adapt structural information. A pre-adapted checkpoint may be used directly as a frozen target teacher.

### Multi-Head GAN Loss

To enforce visual fidelity of student outputs to the target domain $Y$, a multi-head GAN is introduced. The discriminator reuses the encoder and intermediate blocks of the fake teacher $\epsilon^{\text{fk}}$ as a feature extractor, with a linear classification head appended after each encoder block $b \in \mathcal{B}$:

$$D^b(\cdot) = \sigma\left(h^b(f^b(\cdot))\right)$$

The multi-head design enables the discriminator to compare real and generated samples at multiple feature scales, mitigating overfitting and mode collapse in the few-shot regime ($|Y| \leq 10$):

$$\mathcal{L}_{\text{GAN}}^G(\theta) = -\mathbb{E}_{t,z}\sum_{b \in \mathcal{B}} \log\left(D^b_\theta(x_t)\right)$$

$$\mathcal{L}_{\text{GAN}}^D(\psi,\phi) = -\mathbb{E}_{t,y}\sum_{b \in \mathcal{B}} \log\left(D^b(y_t)\right) - \mathbb{E}_{t,z}\sum_{b \in \mathcal{B}} \log\left(1 - D^b(x_t)\right)$$

### Overall Training Objective

Student total loss:

$$\mathcal{L}_G(\theta) = \mathcal{L}_{\text{DMD}}^{\text{trg}+\text{src}}(\theta) + \lambda_{\text{GAN}}^G \mathcal{L}_{\text{GAN}}^G(\theta)$$

Fake teacher + discriminator total loss:

$$\mathcal{L}_{\text{fk}+D}(\phi,\psi) = \mathcal{L}_{\text{fk}}(\phi) + \lambda_{\text{GAN}}^D \mathcal{L}_{\text{GAN}}^D(\psi,\phi)$$

Per iteration, $\epsilon^{\text{fk}} + D$ is updated 5–10 times while $G$ and $\epsilon^{\text{trg}}$ are each updated once, ensuring the fake teacher keeps pace with the student's continuously evolving output distribution. Hyperparameters: $\lambda_{\text{GAN}}^G = 0.01$, $\lambda_{\text{GAN}}^D = 0.03$, learning rate $2 \times 10^{-6}$, batch size 1, bf16 mixed precision.

## Key Experimental Results

### Main Results: Few-Shot Image Generation (FSIG)

The source model is a guided-DDPM pretrained on FFHQ (70K diverse face images), adapted to 10-shot target domains at 256×256 resolution.

| Method | NFE↓ | Single-stage | Babies FID↓ | Sunglasses FID↓ | MetFaces FID↓ | Cats FID↓ | Babies LPIPS↑ | Sunglasses LPIPS↑ |
|------|------|--------|-------------|-----------------|---------------|-----------|---------------|-------------------|
| DDPM-PA | 1000 | ✓ | 48.92 | 34.75 | — | — | 0.59 | 0.60 |
| CRDI | 25 | ✓ | 48.52 | 24.62 | 121.36 | 220.95 | 0.52 | 0.50 |
| FT | 25 | ✓ | 57.06 | 37.86 | 72.99 | 61.62 | 0.32 | 0.48 |
| DMD2-FT | 3 | ✗ | 140.27 | 77.49 | 129.26 | 89.32 | 0.08 | 0.20 |
| FT-DMD2 | 3 | ✗ | 57.11 | 41.85 | 63.25 | 51.85 | 0.42 | 0.42 |
| **Uni-DAD (no $\epsilon^{\text{trg}}$)** | **3** | **✓** | **47.38** | **22.57** | 72.18 | 199.91 | 0.45 | 0.51 |
| **Uni-DAD** | **3** | **✓** | **45.09** | **24.45** | **58.13** | **55.32** | 0.46 | 0.54 |

### Main Results: Subject-Driven Personalization (SDP)

The source model is SDv1.5, evaluated on the DreamBooth benchmark (30 subjects, 25 prompts) at 512×512 resolution.

| Method | NFE↓ | DINO↑ | CLIP-I↑ | CLIP-T↑ | Intra-LPIPS↑ | Inter-LPIPS↑ |
|------|------|-------|---------|---------|-------------|-------------|
| FT (DreamBooth) | 2×50 | 0.58 | 0.77 | 0.32 | 0.67 | 0.73 |
| Turbo-PSO (SDXL) | 4 | 0.50 | 0.70 | 0.30 | 0.42 | 0.60 |
| DMD2-FT | 1 | 0.20 | 0.61 | 0.26 | 0.58 | 0.70 |
| FT-DMD2 | 1 | 0.57 | 0.75 | 0.25 | 0.22 | 0.25 |
| **Uni-DAD** | **1** | **0.47** | **0.73** | **0.29** | **0.51** | **0.59** |

### Ablation Study

**NFE and target set size ablation (FID↓, Babies / MetFaces):**

| Method | NFE | 1-shot B | 1-shot M | 5-shot B | 5-shot M | 10-shot B | 10-shot M |
|------|-----|----------|----------|----------|----------|-----------|-----------|
| CRDI | 25 | 105.51 | 145.10 | 51.71 | 126.34 | 48.52 | 121.36 |
| Uni-DAD | 4 | 72.38 | 95.44 | 45.86 | 81.85 | 41.39 | 59.49 |
| Uni-DAD | 3 | 90.33 | 90.29 | 52.73 | 83.69 | 45.09 | 58.13 |
| Uni-DAD | 1 | 109.55 | 132.79 | 93.52 | 103.84 | 98.52 | 89.03 |

**Component ablation (FID↓):**

| Configuration | DMD$^{\text{src}}$ | DMD$^{\text{trg}}$ | GAN$^{\text{Mh}}$ | Babies | MetFaces |
|------|----|----|----|----|-----|
| GAN-only | — | — | ✓ | 56.90 | 80.14 |
| DMD-only (src) | ✓ | — | — | 110.39 | 68.05 |
| DMD$^{\text{src}}$ + GAN$^{\text{Mh}}$ | ✓ | — | ✓ | 47.38 | 64.13 |
| All components | ✓ | ✓ | ✓ | **45.09** | **58.13** |

### Key Findings

1. **Single-stage outperforms two-stage**: Uni-DAD achieves lower FID at 3 steps than non-distillation methods requiring 25–1000 steps, with comparable diversity (Intra-LPIPS).
2. **DMD2-FT fails severely**: Distilling before fine-tuning negates the distillation benefit, causing FID to spike above 140 and Intra-LPIPS to drop to 0.08, producing severely over-smoothed outputs.
3. **Multi-head GAN is critical**: Multi-head design is substantially more stable than single-head in the few-shot regime, yielding significantly lower FID (Babies: 56.90 vs. 130.34).
4. **Role of target teacher**: For near-domain adaptation (Babies), omitting the target teacher already achieves competitive FID (47.38); for far-domain adaptation (MetFaces), incorporating it yields a large improvement (72.18→58.13).
5. **100× inference speedup**: In the SDP setting, NFE is reduced from 100 to 1 while quality remains comparable.
6. **Lower training cost**: Single-stage requires 2.2–2.8 GPU·h vs. 3.0 GPU·h for two-stage; inference of 5K images takes only 4.2 minutes vs. 35–63 minutes.

## Highlights & Insights

- **First single-stage distillation + adaptation framework**: Conceptually clean, eliminating the design complexity and information loss of two-stage pipelines.
- **Elegant dual-domain DMD design**: The source term preserves diversity while the target term promotes adaptation; the weighting factor $a$ provides flexible control.
- **Multi-head GAN reuses fake teacher features**: No additional feature extraction network is introduced; multi-scale discrimination effectively suppresses few-shot overfitting.
- **Checkpoint-agnostic**: Pre-distilled student checkpoints or pre-adapted teacher checkpoints can be used directly for initialization, providing strong flexibility.
- **Thorough cross-benchmark and cross-backbone validation**: FSIG (guided-DDPM) and SDP (SDv1.5) settings cover near-domain to far-domain scenarios.

## Limitations & Future Work

- Hyperparameter sensitivity and few-shot overfitting risk inherent to GAN training remain.
- The weighting factor $a$ must be manually set based on source–target domain distance; no adaptive scheduling mechanism is provided.
- Training cost remains higher than adaptation-only methods, though lower than two-stage pipelines.
- Validation is limited to 256×256 (FSIG) and 512×512 (SDP) resolutions; scaling to larger backbones (SDXL/DiT) has not been explored.
- Extension to other modalities such as video and audio diffusion models is not addressed.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |
| Overall | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] V-Bridge: Bridging Video Generative Priors to Versatile Few-shot Image Restoration](v-bridge_bridging_video_generative_priors_to_versatile_few-shot_image_restoratio.md)
- [\[CVPR 2026\] DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution](duo-vsr_dual-stream_distillation_for_one-step_video_super-resolution.md)
- [\[ICML 2026\] Envisioning Beyond the Few: Disentangled Semantics and Primitives for Few-Shot Atypical Layout-to-Image Generation](../../ICML2026/image_generation/envisioning_beyond_the_few_disentangled_semantics_and_primitives_for_few-shot_at.md)
- [\[ICLR 2026\] SeMoBridge: Semantic Modality Bridge for Efficient Few-Shot Adaptation of CLIP](../../ICLR2026/image_generation/semobridge_semantic_modality_bridge_for_efficient_few-shot_adaptation_of_clip.md)
- [\[CVPR 2026\] Few-shot Acoustic Synthesis with Multimodal Flow Matching](few-shot_acoustic_synthesis_with_multimodal_flow_matching.md)

</div>

<!-- RELATED:END -->

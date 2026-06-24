---
title: >-
  [Paper Note] Revisiting Diffusion Models: From Generative Pre-training to One-Step Generation
description: >-
  [ICML 2025][Image Generation][Diffusion Distillation] This paper proposes a novel perspective that views diffusion training as "generative pre-training", revealing the fundamental limitation in distillation where the teacher and student models converge to different local optima. The authors demonstrate that pre-trained diffusion models can be efficiently converted into one-step generators (D2O) using only GAN objectives (without distillation loss). Furthermore…
tags:
  - "ICML 2025"
  - "Image Generation"
  - "Diffusion Distillation"
  - "One-Step Generation"
  - "GAN Fine-Tuning"
  - "Generative Pre-training"
  - "Parameter Freezing"
date: 2026-05-08
content_hash: b64442937cc72b3b
---

# Revisiting Diffusion Models: From Generative Pre-training to One-Step Generation

**Conference**: ICML 2025  
**arXiv**: [2506.09376](https://arxiv.org/abs/2506.09376)  
**Code**: None  
**Area**: Diffusion Models  
**Keywords**: Diffusion Distillation, One-Step Generation, GAN Fine-Tuning, Generative Pre-training, Parameter Freezing

## TL;DR

This paper proposes a novel perspective that views diffusion training as "generative pre-training", revealing the fundamental limitation in distillation where the teacher and student models converge to different local optima. The authors demonstrate that pre-trained diffusion models can be efficiently converted into one-step generators (D2O) using only GAN objectives (without distillation loss). Furthermore, a fine-tuned variant with 85% of its parameters frozen (D2O-F) achieves highly competitive results using only 0.2M images.

## Background & Motivation

Although diffusion models have demonstrated image/video generation quality superior to GANs, their inference requires multi-step iterative sampling, resulting in high computational overhead. To accelerate sampling, diffusion distillation is widely adopted to compress the multi-step process into few-step or one-step models. However, existing distillation methods face two major issues:

**High Training Cost**: Generates a massive demand for computational resources and data during distillation training.

**Performance Degradation**: The student model often struggles to match the generation quality of the teacher model.

Recent studies found that incorporating GAN loss during distillation significantly improves results, but the underlying mechanism remains unclear. This paper aims to theoretically and empirically reveal the fundamental limitations of distillation and propose a more elegant alternative.

**Key Insight**: In distillation methods, the multi-step inference of the teacher model and the one-step inference of the student model cause them to converge to different local optima, making instance-level imitation inherently sub-optimal. The GAN objective naturally bypasses this limitation by aligning distributions rather than individual samples.

## Method

### Overall Architecture

This paper proposes the **D2O (Diffusion to One-Step)** method, with the mechanism divided into three steps:

1. **Theoretical Analysis**: Unveiling the fundamental issue of "local optima inconsistency" between the teacher and student models in distillation.
2. **D2O Baseline**: Initializing the generator with a pre-trained diffusion model and fine-tuning it solely with a GAN objective (without distillation loss) to convert the multi-step diffusion model into a one-step generator.
3. **D2O-F (Frozen)**: Freezing 85% of the parameters (all convolutional layers) and fine-tuning only the normalization layers and skip connections to validate the "generative pre-training" hypothesis.

### Key Designs

#### 1. Theoretical Analysis of Distillation Limitations

Starting from the EDM framework, the authors define the ODE solver as:

$$\mathbf{S}_\phi(x_{t_i}, t_i, t_{i-1}) = \frac{t_i - t_{i-1}}{t_i}(\mathbf{g}_\phi(x_{t_i}, t_i) - x_{t_i}) + x_{t_i}$$

In Progressive Distillation, the teacher model requires multiple passes through the neural network, whereas the student model passes through only once. This introduces a critical inductive bias:

- **Teacher Model**: Can transform between pixel and latent spaces multiple times, effectively having a larger parameter capacity.
- **Student Model**: Can only perform a single transformation, restricting its parameter space.

The authors empirically validate this hypothesis by using FID to measure the differences between the teacher models with various step sizes (2/4/6/8/10 steps) and the one-step student model:
- The FIDs to the training set are close (both can generate high-quality images).
- However, the FID between the teacher and the student increases significantly with the number of teacher steps.
- Even for the closest 2-step teacher, there is still an FID gap of 1.78 with the student.

This demonstrates that the teacher and the student achieve similar performance in "different ways", making instance-level imitation sub-optimal.

#### 2. D2O Model with Pure GAN Objectives

Based on the above analysis, D2O entirely discards the distillation loss and only utilizes non-saturating GAN objectives:

- **Discriminator Objective**: $\max_{\mathbf{D}} \mathbb{E}_\mathbf{x}[\log(\mathbf{D}(\mathbf{x}))] + \mathbb{E}_\mathbf{z}[\log(1 - \mathbf{D}(G(\mathbf{z})))]$
- **Generator Objective**: $\min_{G} -\mathbb{E}_\mathbf{z}[\log(\mathbf{D}(G(\mathbf{z})))]$

Key design details:
- The generator is directly initialized with the U-Net from the pre-trained diffusion model.
- The input is pure Gaussian noise (corresponding to the maximum noise level $t_{max}$), and the output is the image generated by one-step denoising.
- Real images (rather than teacher outputs) are used as positive samples for the discriminator, directly pushing the student to approximate the real data distribution.
- There is no need for online inference of the teacher model, avoiding the computational overhead of multi-step forward passes in distillation.

#### 3. D2O-F Parameter Freezing Strategy

To validate the hypothesis of "diffusion training = generative pre-training", D2O-F freezes most parameters during fine-tuning:

- **Frozen Layers (blue, ~85%)**: Parameters of all convolutional layers remain unchanged.
- **Trainable Layers (red, ~15%)**: Only fine-tune normalization layers (GroupNorm) and skip connections.
- The theoretical rationale for this design is that the convolutional layers have already learned generalized generative feature representation capabilities during diffusion pre-training, so fine-tuning only needs to adjust feature scaling/shifting and cross-layer information flow.

### Loss & Training

- **Initialization**: The generator is initialized with EDM pre-trained weights; the discriminator is randomly initialized.
- **Extremely Low Data Demand**: D2O achieves near-SOTA performance with 5M images, and D2O-F exhibits competitive performance with only 0.2M images.
- **Training Efficiency**: Due to freezing most parameters, D2O-F has very few trainable parameters, leading to fast convergence and low GPU memory footprint.
- **Frequency Domain Analysis**: The authors further explain from a frequency domain perspective why diffusion pre-training enables the model's one-step generation capability. During diffusion training, the model progressively learns to generate different frequency components—low-noise steps learn low frequencies, while high-noise steps learn high frequencies. After pre-training, the model already possesses the potential to reconstruct the full spectrum from noise in a single step.

## Key Experimental Results

### Main Results: ImageNet 64×64 One-Step Generation FID Comparison

| Method | Type | Training Images | FID ↓ |
|------|------|-----------|-------|
| EDM (63 steps) | Multi-step Diffusion | — | 2.44 |
| Progressive Distillation | Distillation | Large | ~3.0+ |
| Consistency Distillation | Distillation | Large | ~3.5+ |
| Various GAN + Distillation Methods | Hybrid | Large | ~2.0-2.5 |
| **D2O (Ours)** | **Pure GAN** | **5M** | **≤2.2** |
| **D2O-F (Ours)** | **Pure GAN + Frozen** | **0.2M** | **Competitive** |

### Ablation Study: Validation of Local Optima Inconsistency between Teacher and Student

| Teacher Steps | vs Training Set FID | vs One-step Student FID |
|---------|-------------|---------------|
| 2 steps | ~2.0 | 1.78 |
| 4 steps | ~2.0 | ~3.5 |
| 6 steps | ~2.1 | ~5.0 |
| 8 steps | ~2.1 | ~6.5 |
| 10 steps | ~2.2 | ~8.0 |

> Key Findings: The more steps the teacher takes, the larger the distribution discrepancy with the student, despite both having similar FID to the training set. This directly supports the "local optima inconsistency" hypothesis.

### Key Findings: CIFAR-10 Baseline Ablation

| Setup | FID ↓ | Description |
|------|-------|------|
| Pure Distillation (No GAN) | Higher | Still difficult to match even with massive data |
| Distillation + GAN | Medium | GAN loss brings significant gains |
| **Pure GAN (D2O)** | **Optimal** | Achieves the best performance without distillation loss |
| Random Init + GAN | Poor | Requires tens of millions of images, proving the value of pre-training |

## Highlights & Insights

1. **Novel Perspective**: Viewing "diffusion training as generative pre-training" is an elegant and explanatory new perspective. Methodologically matching the pre-train → fine-tune paradigm in NLP, diffusion training teaches the model generalized denoising/generative capabilities, while downstream tasks (such as one-step generation) only require lightweight fine-tuning.
2. **Minimalist Design**: D2O eliminates all complex components from distillation methods (online inference of teacher models, progressive step reduction, specialized loss functions), keeping only the simplest GAN objective, yet achieving superior results. This is a classic application of Occam's razor.
3. **Astonishing Data Efficiency**: D2O-F operates with only 0.2M images, reducing the data requirement by 1–2 orders of magnitude compared to traditional distillation methods. This strongly demonstrates that rich generative capabilities are already encoded within the pre-trained weights.
4. **Compelling Freezing Experiment**: Freezing 85% of the parameters maintains or even improves performance. This not only validates the hypothesis but also suggests that the convolutional features of the diffusion model act as a highly reusable, general-purpose generative foundation.
5. **Depth in Frequency Domain Interpretation**: Explaining how diffusion pre-training endows the model with full-spectrum generation capabilities from a frequency-decomposition perspective bridges the gap between theory and practice.

## Limitations & Future Work

1. **Evaluation Limited to Pixel Space**: Experiments are mainly conducted on ImageNet 64×64 and CIFAR-10. Exploration on higher resolutions (such as 256×256 or 512×512) or latent space diffusion models (LDM/SDXL) is lacking, leaving generalization to be further validated.
2. **Instability of GAN Training**: Although pre-trained initialization mitigates the instability of GAN training, its stability on larger-scale or more complex datasets remains to be observed.
3. **Architectural Limitations**: The method is only validated on the U-Net architecture. Newer architectures like the Diffusion Transformer (DiT), which have become mainstream, are not explored.
4. **Conditional Generation**: The study mainly demonstrates class-conditional generation, without addressing more practical scenarios such as text-to-image generation.
5. **Theoretical Depth Can Be Enhanced**: The analysis of local optima inconsistency is mainly based on FID empirical evidence, lacking more rigorous theoretical proofs (e.g., optimization landscape analysis or convergence proofs).
6. **Lack of Perceptual Quality Evaluation**: Complementary metrics like IS (Inception Score) or CLIP-FID are not reported, and human perceptual evaluations are missing.

## Related Work & Insights

- **Diffusion Distillation**: Approaches like Progressive Distillation (Salimans & Ho, 2022), Consistency Models (Song et al., 2023), and Guided Distillation (Meng et al., 2023) all require distillation loss. This paper demonstrates that it can be completely discarded.
- **GAN + Diffusion Hybrids**: SDXL-Turbo (Sauer et al., 2024), UFOGen (Xu et al., 2023), and DMD (Yin et al., 2024) overlay GAN loss on top of distillation; this work further simplifies the framework to a pure GAN.
- **Diffusion Models as Pre-training**: This shares similar motivations with diffusion fine-tuning works such as DreamBooth and ControlNet. However, this paper is the first to explicitly propose the unified perspective of "generative pre-training" and provides direct empirical evidence through parameter-freezing experiments.
- **Insights for Future Research**: This perspective could promote (1) a two-stage paradigm of large-scale diffusion pre-training followed by GAN fine-tuning; (2) probing to explore which layers encode specific generative capabilities (similar to probing in NLP); (3) extending the frozen fine-tuning strategy to video or 3D generation.

## Rating

- Novelty: ⭐⭐⭐⭐ The perspective of "diffusion training as pre-training" is novel and the insight that pure GAN target is sufficient is powerful, though the combination of GAN and diffusion is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐ The core argument is clear but the scale is limited (64x64), lacking high-resolution and LDM/DiT experiments.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, with a smooth narrative flowing from problem identification → mechanistic analysis → methodology proposal → empirical validation.
- Value: ⭐⭐⭐⭐ The low data demand and parameter freezing strategies hold practical significance, and the novel perspective is inspiring for future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Riemannian MeanFlow for One-Step Generation on Manifolds](../../ICML2026/image_generation/riemannian_meanflow_for_one-step_generation_on_manifolds.md)
- [\[ICML 2025\] Task-Agnostic Pre-training and Task-Guided Fine-tuning for Versatile Diffusion Planner](task-agnostic_pre-training_and_task-guided_fine-tuning_for_versatile_diffusion_p.md)
- [\[ICLR 2026\] TwinFlow: Realizing One-step Generation on Large Models with Self-adversarial Flows](../../ICLR2026/image_generation/twinflow_realizing_one-step_generation_on_large_models_with_self-adversarial_flo.md)
- [\[CVPR 2025\] Aesthetic Post-Training Diffusion Models from Generic Preferences with Step-by-step Preference Optimization](../../CVPR2025/image_generation/aesthetic_post-training_diffusion_models_from_generic_preferences_with_step-by-s.md)
- [\[CVPR 2026\] PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion](../../CVPR2026/image_generation/pixelrush_ultrafast_trainingfree_highresolution_im.md)

</div>

<!-- RELATED:END -->

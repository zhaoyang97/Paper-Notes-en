---
title: >-
  [Paper Note] SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation
description: >-
  [ICCV 2025][Image Generation][one-step diffusion] This work converts a pretrained SANA flow matching model into TrigFlow via a lossless mathematical transformation, and combines continuous-time consistency distillation (sCM) with latent adversarial diffusion distillation (LADD) in a hybrid training strategy, achieving unified 1–4 step adaptive high-quality image generation. One-step generation of 1024×1024 images requires only 0.1s on an H100, surpassing FLUX-schnell with an FID of 7.59 and GenEval of 0.74 while being 10× faster.
tags:
  - ICCV 2025
  - Image Generation
  - one-step diffusion
  - consistency distillation
  - flow matching
  - adversarial distillation
  - real-time generation
date: 2026-05-08
content_hash: 23c3b4eaa2f28d17
---

# SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation

**Conference**: ICCV 2025
**arXiv**: [2503.09641](https://arxiv.org/abs/2503.09641)
**Code**: [https://github.com/NVlabs/Sana](https://github.com/NVlabs/Sana)
**Area**: Image Generation / Diffusion Model Acceleration
**Keywords**: one-step diffusion, consistency distillation, flow matching, adversarial distillation, real-time generation

## TL;DR
This work converts a pretrained SANA flow matching model into TrigFlow via a lossless mathematical transformation, and combines continuous-time consistency distillation (sCM) with latent adversarial diffusion distillation (LADD) in a hybrid training strategy, achieving unified 1–4 step adaptive high-quality image generation. One-step generation of 1024×1024 images requires only 0.1s on an H100, surpassing FLUX-schnell with an FID of 7.59 and GenEval of 0.74 while being 10× faster.

## Background & Motivation
Diffusion models typically require 20–100 sampling steps, severely limiting real-time applications. Existing acceleration methods fall into two categories: trajectory distillation (e.g., consistency models, CM) and distribution distillation (e.g., GAN/VSD), each with distinct drawbacks: GAN training is unstable and prone to mode collapse; VSD requires an additional diffusion model, increasing memory overhead; discrete-time CMs suffer quality degradation at very few steps (<4). More critically, continuous-time consistency models (sCM) require models in TrigFlow format, whereas mainstream models (SANA/FLUX/SD3) use flow matching, making training a TrigFlow model from scratch prohibitively expensive.

## Core Problem
How to efficiently convert an existing flow matching model into a consistency model capable of one-step generation—without training from scratch—while preserving generation quality and multi-step flexibility.

## Method

### Overall Architecture
SANA-Sprint builds upon the pretrained SANA model in three stages: (1) a lossless mathematical transformation converts the flow matching model to TrigFlow format; (2) sCM-based continuous-time consistency distillation maintains alignment with the teacher, while LADD adversarial distillation enhances single-step fidelity; (3) unified training produces a step-adaptive model shared across 1–4 steps.

### Key Designs
1. **Training-Free Flow→TrigFlow Conversion**: The core contribution—via rigorous mathematical derivation (Proposition 3.1), the inputs and outputs of a flow matching model are converted to TrigFlow format through a differentiable transformation. The conversion involves time remapping ($t_{FM} = \sin(t_{Trig})/(\sin(t_{Trig})+\cos(t_{Trig}))$), input rescaling, and linear combination of outputs. Both theoretical analysis and experiments confirm the transformation is completely lossless (FID 5.81→5.73 at 50 steps). This eliminates the need to pretrain TrigFlow models from scratch and enables the sCM framework to be directly applied to any flow matching model.

2. **Hybrid sCM + LADD Distillation**: sCM learns self-consistent mappings along ODE trajectories via a continuous-time consistency loss, maintaining distributional alignment and diversity with the teacher; LADD performs adversarial learning in latent space using teacher model features as the discriminator, enhancing single-step fidelity. The two are complementary: sCM alone achieves FID=8.93, LADD alone achieves FID=12.20, and their combination yields FID=8.11.

3. **Training Stabilization Techniques**: (a) *Dense Time Embedding*—replacing the noise coefficient from $1000t$ to $t$ stabilizes time derivatives and accelerates convergence; (b) *QK-Normalization*—applying RMS norm in self/cross-attention stabilizes gradients for large models (1.6B) and prevents training collapse; (c) *Max-Time Weighting*—setting the training timestep to $\pi/2$ (i.e., pure noise) with 50% probability in LADD, significantly improving one-step generation quality.

### Loss & Training
- Total loss: $\mathcal{L} = \mathcal{L}_{sCM} + 0.5 \cdot \mathcal{L}_{adv}$
- The teacher is obtained by pruning and fine-tuning SANA-1.5 4.8B; the teacher is first fine-tuned for 5K steps to adapt to dense time embedding and QK-Norm, followed by 20K steps of student distillation.
- Training uses 32× A100 GPUs with a global batch size of 512.

## Key Experimental Results

| Method | Steps | Params | FID↓ | GenEval↑ | Latency (A100) |
|--------|-------|--------|------|----------|----------------|
| FLUX-dev | 50 | 12B | 10.15 | 0.67 | 23.0s |
| SANA-1.6B | 20 | 1.6B | 5.76 | 0.66 | 1.2s |
| FLUX-schnell | 4 | 12B | 7.94 | 0.71 | 2.10s |
| SD3.5-Turbo | 4 | 8B | 11.97 | 0.72 | 1.15s |
| **SANA-Sprint 0.6B** | **4** | **0.6B** | **6.48** | **0.76** | **0.32s** |
| **SANA-Sprint 1.6B** | **4** | **1.6B** | **6.54** | **0.77** | **0.31s** |
| FLUX-schnell | 1 | 12B | 7.26 | 0.69 | 0.68s |
| SDXL-DMD2 | 1 | 0.9B | 7.10 | 0.59 | 0.32s |
| **SANA-Sprint 0.6B** | **1** | **0.6B** | **7.04** | **0.72** | **0.21s** |
| **SANA-Sprint 1.6B** | **1** | **1.6B** | **7.69** | **0.76** | **0.21s** |

- One-step generation: FID=7.04 and GenEval=0.72, surpassing FLUX-schnell (7.26/0.69) at 3.2× the speed.
- Generates 1024×1024 images in 0.31s on RTX 4090 and 0.1s on H100—genuinely AIPC-class real-time performance.
- ControlNet integration: 1024×1024 in 0.25s (H100), enabling real-time sketch-to-image interaction.
- Unified step-adaptive design: a single model performs well across 1–4 steps without step-specific training.
- 8.4× faster than the teacher SANA and 64.7× faster than FLUX-schnell (Transformer computation only).

### Ablation Study
- sCM+LADD ≫ sCM alone ≫ LADD alone.
- Dense time embedding ($t$ vs. $1000t$): eliminates 1000× gradient amplification, significantly improving stability.
- QK-Norm is critical for the 1.6B model (training collapses without it).
- Max-time weighting at 50% is optimal (0%→FID=9.44, 50%→FID=8.32).
- CFG embedding improves CLIP score by +0.94.
- The Flow→TrigFlow transformation is fully lossless (FID difference <0.1).

## Highlights & Insights
- **Lossless Flow→TrigFlow transformation** is a breakthrough contribution: it enables any flow matching model (FLUX/SD3, etc.) to directly adopt the sCM distillation framework without repretraining.
- **Hybrid distillation (sCM+LADD)** offers strong complementarity: sCM preserves alignment and diversity, while LADD ensures single-step fidelity.
- **Genuine AIPC-class performance**: 0.31s on RTX 4090 and 0.1s on H100—a milestone for real-time text-to-image generation on consumer GPUs.
- **Unified step-adaptive design**: a single model delivers high quality across 1–4 steps, greatly simplifying deployment.
- **General-purpose stabilization techniques**: dense time embedding and QK-Norm are transferable to other distillation methods.

## Limitations & Future Work
- SANA-Sprint 1.6B achieves slightly worse FID than 0.6B at one step (7.69 vs. 7.04), suggesting larger models may require more distillation iterations for single-step generation.
- The JVP computation in sCM currently does not support Flash Attention, requiring Linear Attention as a substitute.
- Validation is limited to the SANA architecture; applicability to FLUX/SD3 is claimed but not empirically demonstrated.
- ControlNet is only evaluated with HED edge conditioning; other condition types remain untested.

## Related Work & Insights
- **vs. FLUX-schnell**: Both are distilled models; SANA-Sprint with 0.6B parameters surpasses the 12B FLUX-schnell on both FID and GenEval at over 10× the speed.
- **vs. LCM/PCM**: Discrete-time CMs suffer severe quality degradation below 4 steps (SDXL-LCM FID=50.51 at 1 step); SANA-Sprint eliminates discretization error via sCM.
- **vs. DMD2**: DMD2 is pure distribution distillation (VSD) requiring step-specific models; SANA-Sprint uses hybrid distillation with a step-adaptive model.
- **vs. Dense2MoE**: Dense2MoE accelerates DiT inference via MoE; SANA-Sprint reduces step count via distillation—the two acceleration paradigms are orthogonal.

The Flow→TrigFlow transformation elevates sCM from a theoretical tool to a practical one, potentially catalyzing a wave of work accelerating existing flow matching models to one-step generation. The sCM+GAN hybrid strategy is also transferable to video diffusion model acceleration, where real-time performance is even more critical. This work is orthogonal to MoE-based methods such as Dynamic-DINO—combining MoE parameter reduction with sCM step reduction could yield even more extreme acceleration.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The lossless Flow→TrigFlow transformation is a theoretical innovation; the sCM+LADD hybrid framework is elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive comparison of 6 methods across 1/2/4 steps; per-component ablation of stabilization techniques; detailed timestep scheduling analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are rigorous (with complete proofs); Figures 1 and 2 are highly persuasive.
- **Value**: ⭐⭐⭐⭐⭐ A milestone for real-time text-to-image generation on consumer GPUs; open-sourced code and models with significant community impact.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Inference-Time Diffusion Model Distillation](inference-time_diffusion_model_distillation.md)
- [\[ICCV 2025\] StreamDiffusion: A Pipeline-level Solution for Real-time Interactive Generation](streamdiffusion_a_pipeline-level_solution_for_real-time_interactive_generation.md)
- [\[ICCV 2025\] Compression-Aware One-Step Diffusion Model for JPEG Artifact Removal](compression-aware_one-step_diffusion_model_for_jpeg_artifact_removal.md)
- [\[ICCV 2025\] Contrastive Flow Matching (ΔFM)](contrastive_flow_matching.md)
- [\[ICCV 2025\] MMAIF: Multi-task and Multi-degradation All-in-One for Image Fusion with Language Guidance](mmaif_multi-task_and_multi-degradation_all-in-one_for_image_fusion_with_language.md)

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Realism Control One-step Diffusion for Real-World Image Super-Resolution
description: >-
  [AAAI 2026][Image Generation][Image Super-Resolution] This paper proposes the RCOD framework, which endows one-step diffusion (OSD) super-resolution methods with the ability to flexibly control the fidelity–realism trade…
tags:
  - "AAAI 2026"
  - "Image Generation"
  - "Image Super-Resolution"
  - "One-Step Diffusion"
  - "Realism Control"
  - "Degradation-Aware"
  - "Latent Domain Grouping"
date: 2026-05-08
content_hash: 36e0ba592812135c
---

# Realism Control One-step Diffusion for Real-World Image Super-Resolution

**Conference**: AAAI 2026
**arXiv**: [2509.10122](https://arxiv.org/abs/2509.10122)
**Code**: [https://zongliang-wu.github.io/RCOD-SR](https://zongliang-wu.github.io/RCOD-SR)
**Area**: Image Generation
**Keywords**: Image Super-Resolution, One-Step Diffusion, Realism Control, Degradation-Aware, Latent Domain Grouping

## TL;DR

This paper proposes the RCOD framework, which endows one-step diffusion (OSD) super-resolution methods with the ability to flexibly control the fidelity–realism trade-off at inference time via a latent domain grouping strategy and degradation-aware sampling. A visual prompt injection module is also introduced to replace text prompts, improving restoration accuracy.

## Background & Motivation

### State of the Field

Real-world image super-resolution (Real-ISR) aims to recover high-resolution images from low-resolution inputs with unknown degradations. Methods based on pretrained Stable Diffusion (SD)—such as DiffBIR, StableSR, and SeeSR—achieve excellent perceptual quality through iterative latent-space optimization, but their multi-step sampling latency renders them impractical for real-time applications.

### Limitations of Prior Work

To address efficiency, one-step diffusion methods (OSEDiff, S3Diff) compress multi-step diffusion priors into single-step inference via knowledge distillation, achieving 10–100× speedups. However, **OSD methods face a core tension**:
- **Multi-step diffusion** can flexibly balance fidelity and realism by adjusting the number of sampling steps.
- **One-step diffusion**, trained with a fixed timestep $T$, can only learn to restore a degradation "average," lacking adaptability to scene-specific requirements.
- Existing OSD methods produce a single output and cannot accommodate varying fidelity/realism demands across different scenarios.

### Root Cause

OSD methods train with a single fixed timestep across LR inputs of varying degradation levels, causing the model to converge to a confined domain that generates a fixed degree of detail, forfeiting the flexibility of multi-step methods.

### Starting Point

Since the timestep conditioning is an irreducible component of diffusion denoising networks and governs the mean and variance of noisy latent features, assigning different timesteps according to degradation severity can grant one-step diffusion controllable generation capability.

## Method

### Overall Architecture

The RCOD (Realism Controlled One-step Diffusion) framework comprises three core components:
1. **Latent Domain Grouping (LDG)**: Groups training data by degradation severity and assigns corresponding timesteps.
2. **Degradation-Aware Sampling distillation (DAS)**: Aligns the timestep sampling of the regularization network with LDG groupings during distillation.
3. **Visual Prompt Injection Module (VPIM)**: Replaces text prompts with degradation-aware visual tokens.

The framework can be readily integrated into existing OSD methods (e.g., OSEDiff and S3Diff).

### Key Designs

#### 1. **Latent Domain Grouping (LDG)**

**Mechanism**: Rather than using a fixed timestep $T$, the timestep $t$ is adaptively selected based on a latent metric $M_L$:

$$\hat{z}_H = F_\theta(z_L; t, c_y)$$

$$t = k \cdot \left(n - \left\lfloor \frac{n \cdot (M_L - M_{L\text{-min}})}{M_{L\text{-max}} - M_{L\text{-min}}} \right\rfloor \right), \quad k \in \mathbb{Z}^+$$

where $M_L$ measures the degree of degradation in the LR latent features, $n$ is the number of groups (set to $\leq 4$), and $k=250$ is the timestep interval.

**Design Motivation**: A larger timestep corresponds to stronger generative capacity in the denoising process (i.e., larger $t$ produces more detail in SD-Turbo). Through the grouping strategy, severely degraded samples are assigned higher timesteps (stronger generation), while mildly degraded samples receive lower timesteps (preserving fidelity). At inference, users can directly select a timestep to monotonically control the realism level.

**Latent Metric $M_L$**: Cosine similarity is adopted as the metric:

$$M_L = \frac{z_L \cdot z_H}{\|z_L\| \|z_H\|}$$

Experiments show that cosine similarity achieves higher Spearman correlation with objective (SSIM), perceptual (DISTS), and semantic (CLIPIQA) metrics than L1/MSE distances (Spearman coefficients 0.78/0.42/0.27 vs. L1's 0.60/0.15/0.06, respectively).

#### 2. **Degradation-Aware Sampling (DAS)**

**Mechanism**: During VSD distillation, the timestep sampling of the regularization network is aligned with LDG:

$$t_r = S(\max(20, t-k), \min(980, t+k))$$

where $t_r$ is the timestep sampled by the regularization network, $t$ is the OSD timestep selected by LDG, and $S(t_{min}, t_{max})$ denotes uniform random sampling.

**Design Motivation**: The original VSD method samples timesteps over a broad range (20–980) for regularization without considering degradation information. DAS constrains the sampling range to a neighborhood of the LDG-assigned timestep, aligning the distillation regularization strength with the degree of degradation.

#### 3. **Visual Prompt Injection Module (VPIM)**

**Mechanism**: The text encoder (CLIP text model) is replaced by a CLIP visual model combined with an MLP, which directly extracts degradation-aware visual tokens from the LR image to serve as input to the U-Net cross-attention layers.

**Design Motivation**:
- Text prompts (especially those relying on VLMs) introduce additional computational overhead, and textual descriptions may not fully align with image content.
- Visual prompts are directly bound to image pixel features, simultaneously improving both fidelity and realism.
- Dependency on VLMs is eliminated, reducing inference latency.

#### 4. **Metric Estimation Module (MEM)**

Used for adaptive timestep selection at inference. Intermediate layer features from a pretrained model and a simple MLP are employed to estimate $M_L$, operating independently of OSD training.

### Loss & Training

The total loss function (using RCOD_O as an example):

$$\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda_2 \mathcal{L}_{reg} + \lambda_3 \mathcal{L}_{diff} + \lambda_4 \mathcal{L}_{cls}$$

where:
- $\mathcal{L}_{data} = \mathcal{L}_{MSE}(\hat{x}_H, x_H) + \lambda_1 \mathcal{L}_{LPIPS}(\hat{x}_H, x_H)$: data consistency loss
- $\mathcal{L}_{reg}$: VSD regularization loss (used with the DAS strategy)
- $\mathcal{L}_{diff}$: regularizer fine-tuning loss
- LoRA is used for parameter-efficient fine-tuning

Training configuration: 30K+ iterations, batch size 4, learning rate $2 \times 10^{-5}$. At inference, $n=3$ corresponds to three levels: Fidelity ($t=250$), Neutral ($t=500$), and Realism ($t=750$).

## Key Experimental Results

### Main Results

Performance on DRealSR (real-world data):

| Method | PSNR↑ | SSIM↑ | MANIQA↑ | CLIPIQA↑ | Notes |
|--------|-------|-------|---------|----------|-------|
| OSEDiff | 27.92 | 0.7835 | 0.5899 | 0.6963 | Original baseline |
| RCOD_O-Fid. | **28.90** | **0.7906** | 0.6275 | 0.7023 | Fidelity mode outperforms baseline |
| RCOD_O-Neu. | 28.30 | 0.7775 | 0.6385 | 0.7179 | Neutral mode |
| RCOD_O-Real. | 27.59 | 0.7600 | **0.6295** | **0.7325** | Realism mode |
| S3Diff | 27.54 | 0.7491 | 0.6134 | 0.7130 | Original baseline |
| RCOD_S-Adap. | 27.83 | 0.7661 | 0.6223 | 0.7110 | Adaptive mode |

On RealSR, RCOD_O-Fid. achieves PSNR 26.01 (vs. OSEDiff's 25.15) and MANIQA 0.6647 (vs. OSEDiff's 0.6326).

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Cosine similarity vs. L1 | Spearman(SSIM): 0.78 vs. 0.60 | CS shows strongest correlation with image quality metrics |
| Cosine similarity vs. MSE | Spearman(DISTS): 0.42 vs. 0.11 | CS advantage more pronounced on perceptual metrics |
| Different degradation pipelines | Orig: PSNR 25.15, New Deg: 24.59 | Stronger degradation → lower PSNR but higher MANIQA |

### Efficiency Analysis

| Method | Inference Time (s) | Trainable Params (M) | PSNR | MANIQA |
|--------|-------------------|---------------------|------|--------|
| PiSA-SR | 0.13 | 8.1 | 28.31 | 0.6156 |
| OSEDiff | 0.11 | 8.5 | 27.92 | 0.5899 |
| RCOD_O-Fid. | **0.09** | 9.5 | **28.90** | **0.6275** |
| S3Diff | 0.28 | 34.5 | 27.54 | 0.6134 |

### Key Findings

- The realism level increases **monotonically** with timestep, validating the effectiveness of the LDG strategy.
- Even in fidelity mode, RCOD outperforms the original methods on NR metrics, indicating that the grouping strategy itself improves overall performance.
- Replacing text prompts with VPIM yields faster inference (no text encoder/VLM required).
- In adaptive mode, MEM-estimated $M_L$ values are mostly close to 1, consistent with the training data distribution.

## Highlights & Insights

1. **Simple yet effective core idea**: The essential contribution is "assigning timesteps according to degradation severity"—achieving controllable generation through the most fundamental diffusion condition (the timestep) without additional trainable parameters.
2. **Strong generalizability**: A framework-level approach, verified to be integrable into two architecturally distinct OSD methods, OSEDiff and S3Diff.
3. **Superiority of cosine similarity**: Spearman correlation analysis demonstrates that cosine similarity captures degradation information in high-dimensional latent features more effectively than L1/MSE.
4. **Visual prompts outperform text prompts**: Dependency on VLMs is eliminated while simultaneously improving both fidelity and realism.

## Limitations & Future Work

- The number of groups $n$ is fixed at 3–4; whether finer-grained grouping yields better control remains to be explored.
- The accuracy of the MEM module directly affects adaptive mode performance; a simple MLP is currently used, and a more powerful estimation network could be considered.
- Validation is limited to SD-series models; applicability to DiT architectures has not been verified.
- At inference, users must manually select a mode or rely on MEM; a mechanism for automatic content-driven decision-making is lacking.

## Related Work & Insights

- Compared to PiSA-SR (two LoRAs + two-step diffusion for fidelity/realism control), RCOD is simpler and more efficient (single-step, no additional parameters).
- Compared to OFTSR (trajectory alignment distillation), RCOD incorporates degradation-awareness.
- Insight: In other conditional generation tasks, controllable generation may similarly be achieved by grouping conditions (e.g., noise levels, guidance strength).

## Rating

- Novelty: ⭐⭐⭐⭐ (Simple but effective idea; core concept is clear)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Multiple datasets, baselines, metrics, and comprehensive ablations)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure; motivation and method are explained in detail)
- Value: ⭐⭐⭐⭐ (Addresses a practical limitation of OSD methods with strong generalizability)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Mixture of Ranks with Degradation-Aware Routing for One-Step Real-World Image Super-Resolution](mixture_of_ranks_with_degradation-aware_routing_for_one-step_real-world_image_su.md)
- [\[NeurIPS 2025\] DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution](../../NeurIPS2025/image_generation/dove_efficient_one-step_diffusion_model_for_real-world_video_super-resolution.md)
- [\[AAAI 2026\] Continuous Degradation Modeling via Latent Flow Matching for Real-World Super-Resolution](continuous_degradation_modeling_via_latent_flow_matching_for_real-world_super-re.md)
- [\[AAAI 2026\] QuantVSR: Low-Bit Post-Training Quantization for Real-World Video Super-Resolution](quantvsr_low-bit_post-training_quantization_for_real-world_video_super-resolutio.md)
- [\[CVPR 2026\] OARS: Process-Aware Online Alignment for Generative Real-World Image Super-Resolution](../../CVPR2026/image_generation/oars_process-aware_online_alignment_for_generative_real-world_image_super-resolu.md)

</div>

<!-- RELATED:END -->

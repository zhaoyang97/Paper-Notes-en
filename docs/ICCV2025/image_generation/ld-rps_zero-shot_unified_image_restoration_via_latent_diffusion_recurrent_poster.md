---
title: >-
  [Paper Note] LD-RPS: Zero-Shot Unified Image Restoration via Latent Diffusion Recurrent Posterior Sampling
description: >-
  [ICCV 2025][Image Generation][Zero-shot image restoration] LD-RPS proposes a zero-shot, dataset-free unified image restoration method that performs recurrent posterior sampling via a pretrained latent diffusion model. It leverages multimodal large language models for semantic priors and a learnable F-PAM module to align the degradation domain, achieving high-quality blind restoration across diverse degradation types.
tags:
  - ICCV 2025
  - Image Generation
  - Zero-shot image restoration
  - posterior sampling
  - latent diffusion
  - recurrent refinement
  - multimodal prior
date: 2026-05-08
content_hash: ed825afbae2a242c
---

# LD-RPS: Zero-Shot Unified Image Restoration via Latent Diffusion Recurrent Posterior Sampling

**Conference**: ICCV 2025
**arXiv**: [2507.00790](https://arxiv.org/abs/2507.00790)
**Code**: [https://github.com/AMAP-ML/LD-RPS](https://github.com/AMAP-ML/LD-RPS)
**Area**: Image Generation
**Keywords**: Zero-shot image restoration, posterior sampling, latent diffusion, recurrent refinement, multimodal prior

## TL;DR

LD-RPS proposes a zero-shot, dataset-free unified image restoration method that performs recurrent posterior sampling via a pretrained latent diffusion model. It leverages multimodal large language models for semantic priors and a learnable F-PAM module to align the degradation domain, achieving high-quality blind restoration across diverse degradation types.

## Background & Motivation

Unified Image Restoration (UIR) aims to handle multiple degradation types (noise, low-light, haze, colorization, etc.) within a single model and represents an important direction in low-level vision. Existing methods face three major challenges:

**Task-specific methods lack generalization**: Traditional methods (e.g., ZDCE++, AOD-Net) design networks for specific degradations and cannot generalize to other types.

**Supervised unified methods are constrained to closed sets**: Methods such as AirNet, PromptIR, and DiffUIR are trained on specific datasets and exhibit significant performance degradation when encountering unseen degradation types.

**Existing posterior sampling methods are unstable**: Methods like GDP rely on pixel-level diffusion and explicit degradation modeling ($y = Ax + B$), which is unsuitable for complex real-world degradations.

An ideal unified restoration solution should simultaneously satisfy: (1) unsupervised — no reliance on labeled data; (2) dataset-free — no need for training data collection; and (3) generalizable — capable of handling unseen degradation types.

The authors' core insight is twofold: **latent space is more suitable for posterior sampling than pixel space**, as latent representations filter out redundant pixel information and degradation noise; and **recurrent sampling is more stable than single-pass sampling**, as using the previous iteration's result as the next initialization progressively improves quality.

## Method

### Overall Architecture

The inference pipeline of LD-RPS comprises three core components:

1. **MLLM Semantic Prior Generation**: A multimodal large language model (e.g., GPT-4V) generates textual descriptions from low-quality images, which serve as text embeddings to guide the diffusion model.
2. **F-PAM (Feature and Pixel Alignment Module)**: A lightweight learnable network that bridges the degraded image domain and the diffusion model's generation domain.
3. **Recurrent Posterior Sampling**: Extends single-pass posterior sampling into a multi-round recurrent refinement process.

### Key Designs

**1. Task-Blind Semantic Prior Generation**

The method exploits the image understanding capability of MLLMs to extract semantic information from degraded images: a low-quality image and a hand-crafted prompt are fed to the MLLM → the MLLM generates a content description → the description is encoded as a text embedding $c$ → the embedding guides the diffusion model toward the target content. This eliminates the need for manually specifying the degradation type.

**2. F-PAM: Feature and Pixel Alignment Module**

This is the central design for addressing the unique challenges of LD-RPS. Two gaps must be bridged:
- **Spatial gap**: dimensional mismatch between latent space $z$ and image space $x$
- **Domain gap**: distributional discrepancy between the clean image domain and the degraded image domain

F-PAM structure: $\psi[\tilde{z}_0, \tilde{z}_0'] = h_2(h_1(f[\tilde{z}_0, \tilde{z}_0'])) + p \odot h_1(f[\tilde{z}_0, \tilde{z}_0'])$

where $f$ is a frozen VAE decoder, $h_1/h_2$ are learnable convolutional networks, and $p$ is a learnable channel attention factor. F-PAM is jointly optimized with the reverse diffusion process using L2 loss + perceptual loss + GAN loss.

**3. Two-Stage Posterior Sampling**

The reverse diffusion process is divided into two stages:
- **$T \to t_1$ (early stage)**: Only F-PAM is trained; $g = 0$, with no intervention in the diffusion direction.
- **$t_1 \to 0$ (late stage)**: F-PAM and the posterior direction are jointly optimized, correcting the sampling trajectory via gradient $g = \nabla_{z_t} \log p(y|\hat{z}_0)$.

The posterior loss includes:
- **Distance loss $L$**: L2 + perceptual loss + GAN loss (degraded-to-degraded domain alignment)
- **Quality loss $Q$**: brightness constraint + chrominance consistency constraint

**4. Recurrent Refinement**

The core idea is to re-encode and re-noise the restoration result $x_0^{(i)}$ from iteration $i$ to noise level $\gamma T$, using it as the initialization for iteration $i+1$. Each round starts from a lower noise level, yielding greater stability. The recurrence factor $\gamma \in (0,1)$ controls the degree of re-noising.

### Loss & Training

LD-RPS is a **purely inference-time method** requiring no pretraining. However, online optimization occurs during inference:

- **F-PAM training loss $S_\psi$**: L2 reconstruction + VGG perceptual + GAN adversarial
- **Posterior guidance loss $L_\text{total}$**: distance loss (L2 + perceptual + GAN) + quality loss (brightness + chrominance)
- **Type discriminator $D_2$**: distinguishes residuals between "clean–degraded" and "generated–degraded version" pairs

All experiments are conducted on NVIDIA H20 GPUs, and results are averaged over 3 random seeds.

## Key Experimental Results

### Main Results

**Low-light Enhancement (LOLv1 dataset)**:

| Method | Definition (B/D/U) | PSNR↑ | SSIM↑ | LPIPS↓ | PI↓ | NIQE↓ |
|------|:-:|:-:|:-:|:-:|:-:|:-:|
| DiffUIR | ✓/✗/✗ | 21.36 | 0.907 | 0.125 | 4.68 | 5.95 |
| ZERO-IG | ✗/✓/✓ | 17.22 | 0.794 | 0.184 | 4.92 | 6.22 |
| GDP | ✗/✓/✓ | 16.52 | 0.690 | 0.261 | 4.16 | 5.73 |
| TAO | ✓/✓/✓ | 15.84 | 0.757 | 0.363 | 6.34 | 8.79 |
| **LD-RPS** | ✓/✓/✓ | **17.45** | **0.804** | 0.277 | 4.79 | **5.52** |

**Dehazing (RESIDE-HSTS dataset)**:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|:-:|:-:|:-:|
| YOLY | 20.49 | 0.794 | 0.108 |
| GDP | 13.15 | 0.757 | 0.144 |
| TAO | 18.38 | 0.823 | 0.147 |
| **LD-RPS** | **21.45** | 0.813 | 0.177 |

### Ablation Study

**Effect of recurrence count** (LOLv1 / RESIDE / Kodak24):

| Recurrence Count | LOLv1 PSNR↑ | RESIDE PSNR↑ | Kodak24 PSNR↑ |
|:-:|:-:|:-:|:-:|
| 0 | 16.78 | 19.35 | 27.75 |
| 1 | 17.21 | 20.38 | 28.60 |
| 2 | **17.73** | 20.83 | 28.26 |
| 3 | 17.10 | **21.60** | 28.49 |

The optimal recurrence count correlates with the degree of coupling between degradation and semantics: stronger coupling (e.g., dehazing) requires more iterations.

**Ablation of text guidance**:

| Setting | LOLv1 PSNR | RESIDE PSNR | Kodak24 PSNR |
|------|:-:|:-:|:-:|
| w/o Text | 16.03 | 19.63 | 28.13 |
| **Full (w/ Text)** | **17.73 (+1.70)** | **21.60 (+1.97)** | **28.60 (+0.47)** |

Textual priors yield significant improvements across all tasks, most notably for dehazing (+1.97 PSNR).

### Key Findings

1. **LD-RPS surpasses all posterior sampling baselines in the zero-shot setting**: It outperforms GDP and TAO on low-light enhancement, dehazing, and denoising.
2. **Recurrent refinement is effective but not monotonically beneficial**: An optimal recurrence count exists; excessive iterations may degrade quality.
3. **Textual priors are a critical performance factor**: Semantic descriptions generated by the MLLM provide essential directional guidance for the diffusion model.
4. **F-PAM addresses implicit degradation modeling**: Compared to GDP's explicit modeling ($y = Ax + B$), F-PAM adapts to complex nonlinear degradations.

## Highlights & Insights

1. **Posterior sampling in latent space is a forward-looking idea**: Compared to pixel space, latent space inherently suppresses degradation noise, making it naturally advantageous for restoration.
2. **MLLMs provide zero-shot semantic priors**: The approach cleverly leverages large models' image understanding capabilities to compensate for the absence of degradation-type priors.
3. **Recurrent refinement is simple yet effective**: Inspired by bootstrap thinking, it converts the instability of single-pass sampling into the stability of iterative refinement.
4. **Genuinely unified and zero-shot**: Simultaneously satisfies task-blind, dataset-free, and unsupervised conditions.

## Limitations & Future Work

1. **Slow inference speed**: Recurrent sampling combined with online F-PAM training results in long processing times per image.
2. **Color bias**: Color shifts still occur in certain scenarios, requiring quality loss $Q$ as a constraint.
3. **Dependence on MLLM quality**: The quality of textual priors depends on the MLLM's ability to understand degraded images, which may fail under severe degradation.
4. **Unstable GAN discriminator training**: Online discriminator training may introduce instability.
5. **Lack of evaluation on super-resolution and deblurring**: Validation is limited to enhancement, dehazing, denoising, and colorization; spatial degradation types are not covered.

## Related Work & Insights

- **GDP**: A pixel-space diffusion posterior sampling method; the direct improvement target of LD-RPS.
- **TAO**: A test-time adaptive diffusion method and another posterior sampling baseline.
- **DiffUIR / DA-CLIP**: Supervised unified restoration methods, constrained to closed sets.
- **AirNet / PromptIR**: Degradation-aware unified restoration methods requiring paired training data.
- **Insights**: The combination of latent space + learnable degradation mapping + recurrent refinement constitutes a powerful paradigm for zero-shot restoration; MLLMs can serve as general-purpose semantic prior providers.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] FlowDPS: Flow-Driven Posterior Sampling for Inverse Problems](flowdps_flow-driven_posterior_sampling_for_inverse_problems.md)
- [\[NeurIPS 2025\] Split Gibbs Discrete Diffusion Posterior Sampling](../../NeurIPS2025/image_generation/split_gibbs_discrete_diffusion_posterior_sampling.md)
- [\[ICCV 2025\] AnyPortal: Zero-Shot Consistent Video Background Replacement](anyportal_zero-shot_consistent_video_background_replacement.md)
- [\[ICCV 2025\] Early Timestep Zero-Shot Candidate Selection for Instruction-Guided Image Editing](early_timestep_zero-shot_candidate_selection_for_instruction-guided_image_editin.md)
- [\[ICCV 2025\] DynamicID: Zero-Shot Multi-ID Image Personalization with Flexible Facial Editability](dynamicid_zero-shot_multi-id_image_personalization_with_flexible_facial_editabil.md)

<!-- RELATED:END -->

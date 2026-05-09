---
title: >-
  [Paper Note] InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models
description: >-
  [CVPR 2026][Medical Imaging][anomaly detection] This paper proposes a "detection via noising" paradigm to replace the conventional "detection via denoising" approach. By mapping images to the latent noise space via DDIM inversion, the method measures deviation from the prior distribution as an anomaly score using only 3 inference steps—without any reconstruction—achieving state-of-the-art accuracy at 88 FPS (more than 2× faster than OmiAD).
tags:
  - CVPR 2026
  - Medical Imaging
  - anomaly detection
  - diffusion model
  - DDIM inversion
  - reconstruction-free
  - industrial inspection
date: 2026-05-08
content_hash: 2ce40f84422d5692
---

# InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models

**Conference**: CVPR 2026
**arXiv**: [2504.05662](https://arxiv.org/abs/2504.05662)
**Code**: [https://github.com/SkyShunsuke/InversionAD](https://github.com/SkyShunsuke/InversionAD)
**Area**: Medical Imaging
**Keywords**: anomaly detection, diffusion model, DDIM inversion, reconstruction-free, industrial inspection

## TL;DR
This paper proposes a "detection via noising" paradigm to replace the conventional "detection via denoising" approach. By mapping images to the latent noise space via DDIM inversion, the method measures deviation from the prior distribution as an anomaly score using only 3 inference steps—without any reconstruction—achieving state-of-the-art accuracy at 88 FPS (more than 2× faster than OmiAD).

## Background & Motivation
The dominant paradigm for diffusion model-based anomaly detection (AD) is "detection via denoising in RGB space"—adding noise to the input image and then reconstructing it with a diffusion model trained on normal data, using reconstruction error as the anomaly measure. This paradigm has two fundamental limitations: (1) **Noise level sensitivity**: excessive noise corrupts normal regions (increased false positives), while insufficient noise allows anomalous regions to be perfectly restored (increased false negatives); (2) **Expensive multi-step denoising**: most methods operate at only 1–2 FPS, precluding real-time deployment.

## Core Problem
How to eliminate the necessity of reconstruction in diffusion-based AD, achieving efficient anomaly detection without tuning noise-level hyperparameters or performing iterative multi-step inference?

## Method

### Overall Architecture
"Detection via noising in latent space": (1) extract features $z = g_\phi(x)$ using a pretrained backbone (EfficientNet-B4); (2) apply DDIM inversion in feature space (3 steps, $\tau_3 = [333, 666, 999]$) to map features to noise space $z_T$; (3) compute anomaly scores based on the degree of deviation from the prior distribution.

### Key Designs
1. **DDIM Inversion Instead of Reconstruction**: Conventional methods follow the path $x_0 \to x_t \to \hat{x}_0$; InvAD follows $x_0 \to x_T$. Since the diffusion model is trained solely on normal data, normal images are mapped to high-density regions of the Gaussian prior along the PF-ODE trajectory, while anomalous images are mapped to low-density regions. This establishes a deterministic bijective mapping from data space to the noise prior—enabling direct discrimination at the noise end without any reconstruction.

2. **Minimal Inversion Steps Suffice**: A key insight is that AD does not require precise reconstruction, and therefore does not require accurate ODE solving. Using only 3 Euler integration steps ($S=3$), despite low integration accuracy, anomalous pixels are still reliably mapped to low-density regions. This improves inference speed from 1 FPS to 88 FPS.

3. **Feature-Space Diffusion**: The diffusion process operates in the feature space of a pretrained backbone rather than in RGB space, leveraging high-level semantic information to improve detection performance while further reducing inference cost (feature resolution 16×16 vs. input resolution 256×256).

4. **NLL+Diff Anomaly Scoring**: This addresses the reverse-scoring problem, where normal data in high-dimensional spaces may obtain lower NLL. Two complementary scores are used: (a) NLL measures typicality under the noise prior; (b) Diff computes the max-min difference of $z_T^{normed}$, exploiting the local sparsity of anomalies. Their combination is robust to the choice of step count $S$.

### Loss & Training
- An unconditional DiT diffusion model is trained (no class conditioning, no pseudo-anomalies) using the standard DDPM $\epsilon$-prediction loss.
- 300 epochs, AdamW optimizer, warmup + cosine schedule, batch size = 8.
- Inference is plug-and-play: the method can directly replace the inference stage of any diffusion-based AD approach.

## Key Experimental Results

| MVTecAD (multi-class) | Image AU-ROC↑ | Image AP↑ | mAD↑ | FPS↑ |
|---|---|---|---|---|
| DiAD (AAAI'24) | 97.2 | 99.0 | 84.0 | 0.1 |
| HVQ-Trans (NeurIPS'23) | 98.0 | 99.5 | 83.6 | 5.6 |
| OmiAD (ICML'25) | 98.8 | 99.7 | 85.3 | 39.4 |
| **InvAD** | **99.0** | 99.6 | 83.7 | **88.1** |

| VisA | Image AU-ROC↑ | mAD↑ | FPS↑ |
|---|---|---|---|
| OmiAD | 95.3 | 79.3 | 35.3 |
| **InvAD** | **96.9** | **80.3** | **74.1** |

| MPDD | Image AU-ROC↑ | mAD↑ | FPS↑ |
|---|---|---|---|
| OmiAD | 93.7 | 78.9 | 49.8 |
| **InvAD** | **96.5** | **80.1** | **120** |

- Plug-and-play results: DiAD+InvAD → AU-ROC 97.2→98.2, FPS 0.1→88.1; MDM+InvAD → 91.9→98.2, FPS 2.2→63.
- BMAD medical dataset: mAD 87.2, FPS 88, comprehensively outperforming PatchCore, RD4AD, and others.

### Ablation Study
- Inversion vs. reconstruction (Table 4): reconstruction at $S=3$/$r=40\%$ yields only 89.4 AU-ROC, while InvAD at $S=3$ achieves 99.0—completely eliminating noise-level tuning.
- Feature-space diffusion (FDM): single-step inversion in pixel space yields only 44.9 AU-ROC; multi-step inversion in feature space reaches 83.7.
- Scoring scheme (Table 9): NLL alone degrades at large $S$ due to reverse-scoring; Diff alone is also unstable; NLL+Diff remains stable across all $S$ values (99.0→95.4).
- Backbone selection: EfficientNet-B4 > DINO-B > ViT-B; diffusion architecture has minor impact (MLP/UNet/DiT differences are small).
- Inversion steps: $S=3$ is optimal; uniform timestep schedule outperforms quadratic/cubic/exponential schedules.

## Highlights & Insights
- **Paradigm-level innovation**: "detection via noising" replaces "detection via denoising," elegantly circumventing the two fundamental issues of noise-level tuning and multi-step denoising.
- **3-step inference at 88 FPS**: more than 2× faster than OmiAD, the previously fastest diffusion-based AD method, without requiring adversarial distillation.
- **Plug-and-play design**: only the inference stage is modified, enabling direct integration into any existing diffusion-based AD framework.
- **Theoretical depth**: the paper analyzes why low-precision inversion remains effective for anomaly detection and how the reverse-scoring problem is mitigated.

## Limitations & Future Work
- The method still requires 3 NFEs (neural function evaluations); compressing to 1 step via distillation is a promising future direction.
- Pixel-level localization performance is slightly below SOTA (pixel AP/F1 lags behind OmiAD), due to the limited resolution of the 16×16 feature space.
- Detection quality depends on the pretrained backbone; backbone compression of anomaly-relevant information may cause small anomalies to be missed.
- The max-min difference in the NLL+Diff scoring scheme is relatively sensitive to noise.

## Related Work & Insights
- **vs. DiAD/GLAD/TransFusion (reconstruction paradigm)**: these methods require multi-step denoising and noise-level tuning; InvAD eliminates reconstruction and tuning, running tens to hundreds of times faster.
- **vs. OmiAD (distillation-based)**: OmiAD compresses to 1 NFE via adversarial distillation but requires a heavier backbone, achieving 39 FPS; InvAD uses 3 NFEs with a lightweight DiT, achieving 88 FPS.
- **vs. DeCo-Diff (CVPR'25)**: operates at 17 FPS and requires more complex training; InvAD uses standard unconditional training and runs at 88 FPS.

The "inversion instead of reconstruction" concept may generalize to other tasks employing diffusion models for conditional generation or contrastive inference. The NLL+Diff combination strategy for anomaly scoring offers reference value for any high-dimensional density estimation task.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — "Detection via noising" represents a paradigm shift in diffusion-based AD; the idea is elegant and concise.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 4 benchmarks, 7 ablation dimensions, plug-and-play validation, visual analysis, and comprehensive supplementary material.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Motivation is clearly articulated, the comparison table (Table 1) is intuitive, and theoretical derivations are complete.
- **Value**: ⭐⭐⭐⭐⭐ — Achieves dual SOTA in speed and accuracy; plug-and-play design enables direct integration into existing systems.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer](visualad_language-free_zero-shot_anomaly_detection_via_vision_transformer.md)
- [\[CVPR 2026\] Unleashing Video Language Models for Fine-grained HRCT Report Generation](unleashing_video_language_models_for_fine-grained_hrct_report_generation.md)
- [\[CVPR 2026\] MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection](moeclip_patch-specialized_experts_for_zero-shot_anomaly_detection.md)
- [\[CVPR 2026\] Bidirectional Multimodal Prompt Learning with Scale-Aware Training for Few-Shot Multi-Class Anomaly Detection](bidirectional_multimodal_prompt_learning_with_scale-aware_training_for_few-shot_.md)
- [\[CVPR 2026\] Elucidating the Design Space of Arbitrary-Noise-Based Diffusion Models](eda_arbitrary_noise_diffusion_design_space.md)

<!-- RELATED:END -->

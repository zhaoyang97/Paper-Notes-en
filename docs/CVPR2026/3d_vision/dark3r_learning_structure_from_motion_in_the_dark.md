---
title: >-
  [Paper Note] Dark3R: Learning Structure from Motion in the Dark
description: >-
  [CVPR 2026][3D Vision][Low-light 3D reconstruction] Dark3R is a teacher-student distillation framework that transfers the 3D priors of MASt3R to extremely low-light (SNR < −4 dB) raw images, enabling Structure from Motion (SfM) and novel view synthesis in dark environments where traditional methods fail entirely.
tags:
  - CVPR 2026
  - 3D Vision
  - Low-light 3D reconstruction
  - Structure from Motion
  - knowledge distillation
  - feature matching
  - novel view synthesis
  - NeRF
date: 2026-05-08
content_hash: e6989a6018c7cad0
---

# Dark3R: Learning Structure from Motion in the Dark

**Conference**: CVPR 2026
**arXiv**: [2603.05330](https://arxiv.org/abs/2603.05330)
**Code**: [Project Page](https://andrewguo.com/pub/dark3r)
**Area**: 3D Vision
**Keywords**: Low-light 3D reconstruction, Structure from Motion, knowledge distillation, feature matching, novel view synthesis, NeRF

## TL;DR

Dark3R is a teacher-student distillation framework that transfers the 3D priors of MASt3R to extremely low-light (SNR < −4 dB) raw images, enabling Structure from Motion (SfM) and novel view synthesis in dark environments where traditional methods fail entirely.

## Background & Motivation

**Traditional SfM collapses under low light**: Existing SfM pipelines (e.g., COLMAP) rely on feature detection and matching. When image SNR falls below 0 dB, noise dominates the signal, causing feature extraction to fail completely, rendering pose estimation and triangulation infeasible.

**Learning-based methods also fail**: 3D foundation models such as MASt3R and VGGT are pretrained on large-scale datasets, but their training distributions do not include low-SNR raw images, resulting in severely degraded generalization under extreme noise.

**Single-frame denoising breaks multi-view consistency**: Applying a denoiser (e.g., BM3D or neural denoisers) independently to each frame may improve single-image quality, but disrupts cross-view feature consistency, causing downstream matching and pose estimation to fail.

**Burst denoising assumptions are violated**: Burst denoising methods assume minimal inter-frame motion, which is incompatible with the large baselines and significant camera motion inherent in 3D reconstruction scenarios.

**Existing low-light NeRF methods require external poses**: Methods such as RawNeRF can reconstruct radiance fields from raw images, but depend on COLMAP-provided camera poses, creating a deadlock: pose estimation requires clean images, which in turn require the poses.

**Lack of suitable datasets**: No large-scale, multi-view low-light raw image dataset with accurate 3D annotations previously existed, hindering research and evaluation in this direction.

## Method

### Overall Architecture

Dark3R adopts a teacher-student distillation architecture. The pretrained MASt3R serves as the frozen teacher, while the student network is initialized from the same weights and fine-tuned via LoRA. The teacher processes high-SNR clean raw image pairs; the student processes the corresponding low-SNR noisy raw image pairs. The training objective is to align the student's encoder features, decoder features, and correspondence maps with the teacher's outputs. At inference time, only the student network is used, combined with MASt3R-SfM's global optimization and bundle adjustment to recover multi-view camera poses.

### Key Designs

- **Raw image input**: The method directly uses raw images processed with simple demosaicing (sub-sampling each Bayer channel and averaging the two green channels), avoiding information loss caused by black-level subtraction and clipping in the ISP pipeline. Experiments show that MASt3R performs comparably on high-SNR raw images and sRGB inputs.
- **LoRA fine-tuning**: Compared to full-parameter fine-tuning, LoRA consistently yields superior pose accuracy (ATE reduced from 0.476 to 0.050 in ablation experiments) with higher training efficiency.
- **Three-level feature alignment**: Encoder features $\mathbf{F}_{\mathcal{E}}$, decoder features $\mathbf{F}_{\mathcal{D}}$, and correspondence maps $\mathbf{C}$ are jointly aligned using L2 distance supervision.
- **Clean regularization**: Clean image pairs are simultaneously passed through the student network and aligned to the teacher's outputs ($\lambda_{\text{clean}}=0.3$), ensuring the student maintains performance across a wide SNR range.
- **No 3D supervision required**: Training requires only noisy-clean raw image pairs (obtained via direct capture or synthesized with a calibrated Poisson-Gaussian noise model), without any depth or pose ground truth.
- **Known intrinsics constraint**: At inference time, camera intrinsics are assumed known; a regularization term is added during bundle adjustment to keep optimized intrinsics close to calibrated values.

### Novel View Synthesis (Dark3R-NeRF)

- **Coarse-to-fine optimization**: Stochastic preconditioning is employed, adding Gaussian noise to ray sampling positions and annealing $\sigma$ from $10^{-3}$ to 0 during the first 30k steps, followed by 90k additional optimization steps, to avoid overfitting to noise.
- **Depth supervision**: Dense point maps predicted by Dark3R serve as depth priors, following the exponentially decaying weighting strategy of DS-NeRF to progressively reduce constraint strength and preserve fine details.
- **Black-level preservation**: Black-level subtraction and clipping are omitted, retaining near-black-level signals at extremely low SNR, with multi-view aggregation used to improve SNR.

### Loss & Training

$$\mathcal{L} = \|\mathbf{F} - \tilde{\mathbf{F}}_{\text{noisy}}\|_2^2 + \lambda_{\text{clean}} \|\mathbf{F} - \tilde{\mathbf{F}}_{\text{clean}}\|_2^2$$

where $\mathbf{F}$ denotes the teacher's output (concatenated encoder features, decoder features, and correspondence maps) on clean image pairs, and $\tilde{\mathbf{F}}$ denotes the student's corresponding outputs.

## Key Experimental Results

### Dataset

A self-collected dataset comprising approximately 42,000 multi-view bracketed raw images (12 tripod-mounted scenes, each with ~400 viewpoints × 9 exposures) and ~20,000 handheld high-SNR images (92 indoor scenes). Captured with a Sony Alpha I camera; evaluation SNR ranges down to −5 dB.

### Main Results — Pose Estimation

| Method | Input | ATE ↓ | RPE T ↓ | RPE R ↓ | AbsRel ↓ | δ<1.25 ↑ |
|--------|-------|-------|---------|---------|----------|-----------|
| COLMAP | sRGB | 0.669 | 0.155 | 1.644 | 0.638 | 54.38 |
| MASt3R | raw | 0.787 | 0.472 | 2.802 | 0.318 | 39.66 |
| VGGT | sRGB | 0.252 | 0.216 | 1.047 | 0.232 | 63.28 |
| MASt3R-SfM | raw | 0.088 | 0.038 | 0.201 | 0.196 | 79.39 |
| **Dark3R** | **raw** | **0.050** | **0.020** | **0.121** | **0.091** | **93.14** |

Under an average SNR of approximately −3.87 dB (120 input images), Dark3R outperforms all baselines across all metrics.

### Main Results — Novel View Synthesis

| Method | Pose Source | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|--------|-------------|--------|--------|---------|
| Dark3R-NeRF | MASt3R-SfM | 34.60 | 0.835 | 0.308 |
| RawNeRF | Dark3R | 34.24 | 0.848 | 0.291 |
| LE3D | Dark3R | 35.77 | 0.878 | 0.339 |
| **Dark3R-NeRF** | **Dark3R** | **36.17** | **0.866** | **0.257** |
| Dark3R-NeRF | Oracle | 37.16 | 0.882 | 0.228 |

The combination of Dark3R poses and Dark3R-NeRF achieves the best overall performance without oracle inputs.

### Ablation Study — Key Findings

- **LoRA vs. full fine-tuning**: LoRA yields a substantial advantage, reducing ATE from 0.476 to 0.050.
- **Raw vs. sRGB input**: Raw images preserve linear sensor response, leading to higher pose accuracy.
- **Simulated + real data**: Mixed training outperforms using either data source alone.
- **Encoder-only fine-tuning**: Achieves the lowest ATE (0.030) but slightly higher rotation error; fine-tuning all components yields more balanced performance.
- **Clean loss**: Removing it causes negligible performance degradation, indicating that the primary gain stems from noisy L2 alignment.
- **NeRF ablations**: Depth supervision (+1.26 PSNR), omitting black-level clipping (+1.19 PSNR), and stochastic preconditioning (+0.12 PSNR) all contribute positively.

## Highlights & Insights

- **Pioneering problem formulation**: This work is the first to systematically address extreme low-light SfM at SNR < 0 dB, breaking the deadlock of "pose estimation requires clean images, which in turn require poses."
- **Elegant distillation strategy**: Without any 3D supervision, the method transfers MASt3R's 3D priors to the low-light domain using only noisy-clean image pairs, resulting in a simple and extensible design.
- **First low-light multi-view dataset**: The 42,000-image bracketed raw dataset with accurate 3D annotations fills a critical gap in the community.
- **End-to-end system**: The pipeline covers the full reconstruction process from SfM to NeRF, with cross-camera generalization validated on an iPhone 16.

## Limitations & Future Work

- Camera intrinsics must be known (requiring pre-calibration), limiting fully automated deployment on uncalibrated consumer devices.
- Training requires approximately 15 hours on 8 RTX A6000 GPUs, imposing significant computational demands.
- NeRF-based volume rendering is used rather than 3DGS (the authors found Gaussian point cloud optimization difficult under heavy noise), resulting in slower rendering.
- NeRF optimization requires 120k iterations, leading to long per-scene reconstruction times.
- The dataset primarily covers static indoor scenes; generalization to dynamic and outdoor environments remains unvalidated.
- Pose accuracy slightly degrades with more than 500 input images; scalability to large-scale scenes requires further investigation.
- Distillation is bounded by MASt3R's capability; if the teacher is weak on certain scene types, the student will be similarly limited.

## Related Work & Insights

- **MASt3R / MASt3R-SfM**: The teacher model and inference pipeline backbone of Dark3R; remains one of the strongest baselines under high SNR.
- **RawNeRF**: Also performs NeRF in the raw domain, but requires COLMAP poses and can only operate under lighting conditions where COLMAP is feasible.
- **VGGT**: A feed-forward 3D reconstruction foundation model; outperforms COLMAP under low light but falls short of MASt3R-SfM.
- **LE3D**: A 3DGS-based low-light reconstruction method; Dark3R-NeRF substantially outperforms it in LPIPS.
- **DS-NeRF**: Dark3R-NeRF's depth supervision strategy is inspired by this work's exponentially decaying weighting design.
- **SuperPoint/SuperGlue**: Representative learned feature detection and matching methods; both degrade severely under low light.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First work to address extreme low-light SfM; both the problem formulation and distillation scheme are original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Self-collected large-scale dataset, comprehensive ablations, multi-baseline comparisons, and cross-camera validation.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear structure, high-quality figures, and well-motivated problem statement.
- Value: ⭐⭐⭐⭐⭐ — Opens a new direction in passive 3D perception under darkness; both the dataset and method are expected to have lasting impact.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] PromptStereo: Zero-Shot Stereo Matching via Structure and Motion Prompts](promptstereo_zero-shot_stereo_matching_via_structure_and_motion_prompts.md)
- [\[CVPR 2026\] Learning Explicit Continuous Motion Representation for Dynamic Gaussian Splatting from Monocular Videos](learning_explicit_continuous_motion_representation_for_dynamic_gaussian_splattin.md)
- [\[CVPR 2026\] AnthroTAP: Learning Point Tracking with Real-World Motion](anthrotap_learning_point_tracking_with_real-world_motion.md)
- [\[CVPR 2026\] Mamba Learns in Context: Structure-Aware Domain Generalization for Multi-Task Point Cloud Understanding](mamba_learns_in_context_structure-aware_domain_generalization_for_multi-task_poi.md)
- [\[CVPR 2026\] Motion-Aware Animatable Gaussian Avatars Deblurring](motion-aware_animatable_gaussian_avatars_deblurring.md)

<!-- RELATED:END -->

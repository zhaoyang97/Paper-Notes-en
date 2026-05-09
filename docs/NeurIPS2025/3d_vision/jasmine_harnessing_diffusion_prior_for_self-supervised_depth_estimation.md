---
title: >-
  [Paper Note] Jasmine: Harnessing Diffusion Prior for Self-Supervised Depth Estimation
description: >-
  [NeurIPS 2025][3D Vision][Self-supervised depth estimation] This paper is the first to incorporate the visual prior of Stable Diffusion into a self-supervised monocular depth estimation (SSMDE) framework. It proposes the Mix-Batch Image Reconstruction (MIR) proxy task to shield the SD prior from corruption by reprojection noise, and introduces the Scale-Shift GRU (SSG) to bridge the gap between SD's scale-shift-invariant (SSI) and self-supervised scale-invariant (SI) depth distributions. Jasmine achieves AbsRel = 0.090 on KITTI, establishing a new state of the art among all SSMDE methods, while comprehensively outperforming supervised SD methods such as Marigold, E2E FT, and Lotus in zero-shot generalization.
tags:
  - NeurIPS 2025
  - 3D Vision
  - Self-supervised depth estimation
  - Stable Diffusion
  - diffusion prior
  - Mix-Batch Image Reconstruction
  - Scale-Shift GRU
date: 2026-05-08
content_hash: d26a4243658e9587
---

# Jasmine: Harnessing Diffusion Prior for Self-Supervised Depth Estimation

**Conference**: NeurIPS 2025
**arXiv**: [2503.15905](https://arxiv.org/abs/2503.15905)
**Code**: Available (Project Page)
**Area**: 3D Vision / Depth Estimation
**Keywords**: Self-supervised depth estimation, Stable Diffusion, diffusion prior, Mix-Batch Image Reconstruction, Scale-Shift GRU

## TL;DR

This paper is the first to incorporate the visual prior of Stable Diffusion into a self-supervised monocular depth estimation (SSMDE) framework. It proposes the Mix-Batch Image Reconstruction (MIR) proxy task to shield the SD prior from corruption by reprojection noise, and introduces the Scale-Shift GRU (SSG) to bridge the gap between SD's scale-shift-invariant (SSI) and self-supervised scale-invariant (SI) depth distributions. Jasmine achieves AbsRel = 0.090 on KITTI, establishing a new state of the art among all SSMDE methods, while comprehensively outperforming supervised SD methods such as Marigold, E2E FT, and Lotus in zero-shot generalization.

## Background & Motivation

**State of the Field**: Self-supervised monocular depth estimation (SSMDE) learns 3D information solely from video sequences without costly depth annotations. Stable Diffusion has demonstrated powerful visual priors (sharp boundaries and strong generalization) in supervised depth estimation, but has hitherto been used exclusively in supervised settings—fine-tuned on high-precision depth annotations from synthetic datasets.

**Limitations of Prior Work**: The reprojection loss inherent to SSMDE is naturally noisy—occlusions, texture-less regions, and illumination changes all produce spurious supervision signals, leading to blurry predictions. Directly fine-tuning SD with such noisy signals rapidly degrades its VAE latent space prior.

**Root Cause**: Preserving latent space quality in SD requires high-precision supervision, yet the self-supervised paradigm cannot provide it by construction. Furthermore, SD outputs SSI depth (with both scale and shift undetermined), whereas self-supervision requires SI depth (scale undetermined, shift strictly zero), and this distributional mismatch causes training instability.

**Starting Point**: RGB images themselves constitute the highest-quality self-supervised signal—they naturally encode complete visual detail, require no external depth source, and perfectly align with SD's original training objective. Image reconstruction can therefore serve as a proxy task to protect the SD prior.

**Core Idea**: Use an image reconstruction proxy task to protect the SD prior, and iteratively align the SSI-to-SI depth distribution via a GRU, achieving for the first time a successful integration of SD with self-supervised depth estimation.

## Method

### Overall Architecture

Input image $\mathbf{I}_t$ + noise → VAE encoder → U-Net single-step denoising (task switcher controls depth/image-reconstruction tasks) → VAE decoder outputs SSI depth $D_{\text{SSI}}$ or reconstructed image → SSG module converts $D_{\text{SSI}}$ to $D_{\text{SI}}$ → reprojection with adjacent frames via pose network → photometric loss. Stable Diffusion v2 serves as the backbone with single-step denoising.

### Key Designs

1. **Mix-Batch Image Reconstruction (MIR)**

    - **Function**: Protects the SD latent space prior without introducing any additional depth supervision.
    - **Mechanism**: A task switcher $s \in \{s_x, s_y\}$ enables the same U-Net to alternately perform depth prediction and image reconstruction. Image reconstruction uses a photometric loss (SSIM + L1) rather than latent-space MSE—the latter operates at 1/8 resolution in the VAE and produces $8\times8$ block artifacts. Within each mixed batch, real KITTI images or Hypersim synthetic images are randomly selected for reconstruction: $L_s = L_{ph}(\mathbf{I}_h, \mathcal{D}(f_\theta^z(s_y, \mathbf{z}_\tau^y, \mathbf{z}^I)))$
    - **Design Motivation**: Three key insights motivate this design: (1) reconstruction on KITTI alone produces block artifacts due to VAE encoding mismatch; (2) reconstruction on purely synthetic images fails to transfer to reducing depth blur; (3) mixed usage allows synthetic data to anchor the SD prior while real data enforces geometric consistency. Replacing the latent loss with a photometric loss makes MIR robust to the mixing ratio.

2. **Scale-Shift GRU (SSG)**

    - **Function**: Iteratively aligns SSI depth to SI depth.
    - **Mechanism**: The GRU update rule is modified to $D_{k+1} = D_\delta + s_c \cdot D_k + s_h$. A Scale-Shift Transformer (SST) is introduced: learnable scale/shift queries ($Q_{\text{SC}}/Q_{\text{SH}}$) attend via cross-attention to SD hidden states, producing $s_c$ and $s_h$. Two iterations are performed: $D_0$ (SSI) → $D_1$ → $D_2$ (SI). Through rigorous mathematical derivation, the authors prove that under self-supervised geometric constraints the shift must be zero (otherwise any depth map degenerates to a plane), whereas the VAE output range $[-1, 1]$ naturally introduces a non-zero shift.
    - **Design Motivation**: The GRU reset gate selectively blocks noisy gradient backpropagation—anomalous gradients from the reprojection loss are filtered, allowing $D_{\text{SSI}}$ to retain fine-grained SD texture while $D_{\text{SI}}$ maintains geometric consistency.

3. **Steady SD Finetuning**

    - **Function**: Addresses training instability arising from jointly optimizing the large SD backbone, multiple modules, and indirect self-supervised signals.
    - **Mechanism**: A pre-trained self-supervised teacher (MonoViT) generates pseudo depth labels to provide direct supervision. The pseudo-label loss weight decays during training: $\eta_{\text{step}} = \max(1, 30 \cdot (\text{step}_{\text{now}}/\text{step}_{\text{max}}))$, imposing strong constraints early on and relaxing them later to surpass the teacher's performance ceiling.
    - **Design Motivation**: The large number of SD parameters, combined with indirect and noisy self-supervised signals, causes early-stage training to collapse without direct supervision.

### Loss & Training

Total loss: $L = L_s + L_{ph} + L_{tc} + L_a \cdot \eta_a$. $L_s$ is the MIR proxy task loss, $L_{ph}$ is the photometric reprojection loss, $L_{tc}$ is the decaying pseudo-label distillation loss, and $L_a$ is the auxiliary loss (GDS loss, edge loss, etc.). Optimizer: AdamW (lr = 3e-5); hardware: 8 × A800 GPUs; batch size: 32; training: ~25K steps, approximately 1 day.

## Key Experimental Results

### Main Results (KITTI Eigen split)

| Method | Type | Data | AbsRel↓ | RMSE↓ | $a_1$↑ |
|--------|------|------|---------|-------|--------|
| Marigold (CVPR24) | Zero-shot | Syn(74K+74K) | 0.120 | 4.033 | 0.874 |
| E2E FT (WACV25) | Zero-shot | Syn(74K+74K) | 0.112 | 4.099 | 0.890 |
| Lotus (ICLR25) | Zero-shot | Syn(59K+59K) | 0.110 | 3.807 | 0.892 |
| MonoViT (3DV22) | Self-sup. | K(40K) | 0.096 | 4.292 | 0.908 |
| RPrDepth (ECCV24) | Self-sup. | K(40K) | 0.091 | 4.098 | 0.910 |
| **Jasmine** | **Self-sup.** | **KH(68K)** | **0.090** | **3.944** | **0.919** |

### Zero-Shot Generalization

| Method | DrivingStereo AbsRel | CityScape AbsRel | Foggy AbsRel |
|--------|----------------------|------------------|-------------|
| Marigold | 0.178 | 0.164 | 0.146 |
| E2E FT | 0.160 | 0.160 | 0.141 |
| Lotus | 0.173 | 0.147 | 0.150 |
| MonoViT | 0.150 | 0.140 | 0.107 |
| **Jasmine** | **0.136** | **0.123** | **0.098** |

Jasmine comprehensively outperforms both supervised SD methods and conventional SSMDE methods in zero-shot generalization.

### Ablation Study

| ID | Configuration | AbsRel | $a_1$ | Note |
|----|---------------|--------|-------|------|
| 0 | Jasmine (full) | 0.090 | 0.919 | — |
| 1 | w/o SD Prior | 0.516 | 0.258 | Training from scratch collapses; AbsRel ↑473% |
| 2 | w/o MIR+SSG | 0.175 | 0.790 | SD present but unprotected; prior degraded |
| 3 | w/o SSG | 0.129 | 0.872 | No distribution alignment; AbsRel +43% |
| 4 | w/o MIR | 0.132 | 0.852 | No prior protection; AbsRel +47% |
| 10 | Latent loss replaces ph loss | 0.095 | 0.909 | Photometric loss is superior |
| 12 | Auxiliary images = KITTI | 0.095 | 0.912 | Synthetic images not strictly required |
| 13 | KITTI + ETH3D | 0.090 | 0.916 | Higher domain gap in auxiliary data is beneficial |

### Key Findings

- The SD prior is absolutely critical: removing it causes AbsRel to spike from 0.090 to 0.516 (5.7× degradation).
- MIR and SSG are both indispensable: removing either one alone leads to 40%+ degradation, confirming that prior protection and distribution alignment are two independent and necessary problems.
- Domain diversity in auxiliary data matters more than data quality: KITTI + ETH3D outperforms KITTI alone, as greater domain gap enhances SD prior protection.
- Photometric loss outperforms latent loss by emphasizing structural consistency over color accuracy, which better aligns with the depth estimation objective.
- Jasmine's superior zero-shot performance over supervised SD methods demonstrates that self-supervised geometric constraints provide a stronger generalization inductive bias than synthetic depth annotations.

## Highlights & Insights

- **RGB images as "high-precision self-supervised signals"**: Fine-tuning SD requires high-quality supervision to protect the prior; RGB images naturally satisfy this requirement—they are inherently high-quality, require no external dependencies, and are perfectly aligned with SD's original training objective. This challenges the prevailing assumption that SD fine-tuning for depth estimation necessarily requires high-precision depth annotations.
- **Theoretical analysis of SSI vs. SI depth misalignment**: Through rigorous mathematical derivation, the paper proves that under self-supervised constraints the shift must be zero (otherwise any depth map degenerates to a plane), providing the first clear articulation of this fundamental problem along with a systematic solution.
- **GRU reset gate as a gradient filter**: The paper exploits the GRU's natural selective-forgetting property to filter anomalous gradients, enabling SSI depth to preserve fine-grained detail while SI depth maintains geometric consistency. This mechanism is transferable to any scenario requiring fine feature preservation under noisy supervision.
- **First bridge between zero-shot and self-supervised depth estimation**: The paper provides an in-depth analysis of the differences between median alignment and LSQ alignment and their respective applicable scenarios.

## Limitations & Future Work

- Training stability relies on pseudo-labels from a pre-trained teacher (MonoViT), whose quality still constrains the performance ceiling.
- Training is conducted exclusively on KITTI (driving scenes); generalization to indoor, aerial, underwater, and other domains has not been verified.
- Although used in limited quantities, Hypersim synthetic images still constitute an additional data source.
- SD v2 is not the latest generation; the effectiveness of SDXL or SD3 as backbones warrants exploration.
- Training requires 8 × A800 GPUs, partially offsetting the low-annotation-cost advantage of self-supervised learning.

## Related Work & Insights

- **vs. Marigold / E2E FT / Lotus**: Supervised SD methods rely on synthetic depth annotations. Jasmine achieves comprehensive superiority in zero-shot generalization without any depth annotations, demonstrating that self-supervised geometric constraints offer a stronger generalization inductive bias.
- **vs. MonoViT / MonoDepth2**: Conventional SSMDE methods lack the SD prior, resulting in noticeably weaker boundary sharpness and cross-domain generalization compared to Jasmine.
- **vs. Depth Anything v1/v2**: These methods depend on large-scale image–depth paired data; Jasmine argues that pure self-supervision with video data has greater scaling potential.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First integration of SD into self-supervised depth estimation; MIR and SSG are motivated by deep analysis and supported by theoretical derivation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ KITTI + 4 zero-shot datasets; comprehensive ablations covering every component and design choice.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Precise problem formulation, rigorous theoretical derivations, and high-quality figures.
- **Value**: ⭐⭐⭐⭐⭐ Establishes a new paradigm for SD + self-supervised depth estimation; component designs exhibit strong transferability.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Regularizing INR with Diffusion Prior for Self-Supervised 3D Reconstruction of Neutron Computed Tomography Data](../../CVPR2026/3d_vision/regularizing_inr_with_diffusion_prior_selfsupervis.md)
- [\[NeurIPS 2025\] More Than Generation: Unifying Generation and Depth Estimation via Text-to-Image Diffusion Models](more_than_generation_unifying_generation_and_depth_estimation_via_text-to-image_.md)
- [\[NeurIPS 2025\] 3D Visual Illusion Depth Estimation](3d_visual_illusion_depth_estimation.md)
- [\[ICCV 2025\] S3E: Self-Supervised State Estimation for Radar-Inertial System](../../ICCV2025/3d_vision/s3e_self-supervised_state_estimation_for_radar-inertial_system.md)
- [\[NeurIPS 2025\] Concerto: Joint 2D-3D Self-Supervised Learning Emerges Spatial Representations](concerto_joint_2d-3d_self-supervised_learning_emerges_spatial_representations.md)

<!-- RELATED:END -->

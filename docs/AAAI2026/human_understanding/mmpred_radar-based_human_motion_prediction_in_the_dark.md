---
title: >-
  [Paper Note] mmPred: Radar-based Human Motion Prediction in the Dark
description: >-
  [AAAI2026][Human Understanding][millimeter-wave radar] This work is the first to introduce millimeter-wave radar into human motion prediction (HMP)…
tags:
  - "AAAI2026"
  - "Human Understanding"
  - "millimeter-wave radar"
  - "human motion prediction"
  - "diffusion model"
  - "frequency-domain representation"
  - "dual-domain fusion"
date: 2026-05-08
content_hash: 046d7e2eae21230b
---

# mmPred: Radar-based Human Motion Prediction in the Dark

**Conference**: AAAI2026
**arXiv**: [2512.00345](https://arxiv.org/abs/2512.00345)  
**Authors**: Junqiao Fan, Haocong Rao, Jiarui Zhang, Jianfei Yang, Lihua Xie (Nanyang Technological University)
**Code**: Not released  
**Area**: Human Understanding
**Keywords**: millimeter-wave radar, human motion prediction, diffusion model, frequency-domain representation, dual-domain fusion

## TL;DR

This work is the first to introduce millimeter-wave radar into human motion prediction (HMP), proposing mmPred — a diffusion-based framework that employs dual-domain historical motion representations (time-domain pose refinement TPR + frequency-domain dominant motion FDM) and a Global Skeleton Transformer (GST) to effectively suppress radar-specific noise and temporal inconsistency, surpassing SOTA methods by 8.6% and 22% on the mmBody and mm-Fi datasets, respectively.

## Background & Motivation

Human motion prediction (HMP) aims to forecast future poses from observed historical pose sequences, with important applications in human–robot interaction, healthcare, and hazard prevention. Existing HMP methods rely heavily on high-precision multi-view RGB-D motion capture systems to acquire historical poses, which are costly and impractical for real-world deployment. Single-view RGB approaches suffer severe performance degradation in adverse conditions such as darkness and occlusion, and also raise privacy concerns.

Millimeter-wave (mmWave) radar operates in the 30–300 GHz band, can penetrate visual obstacles such as smoke, works reliably under arbitrary lighting conditions, and naturally preserves privacy due to its limited spatial resolution, making it an ideal sensor for indoor HMP. However, radar point clouds present two critical issues: (1) **multipath effects** produce "ghost point" noise; and (2) **specular reflection** causes intermittent body-part dropout — reflected signals from certain body parts deviate away from the receiver. These issues result in temporally inconsistent and noisy radar point clouds.

Existing radar pose estimators (e.g., P4Transformer) process frames independently, ignoring temporal inconsistency and producing heavily jittered and distorted pose sequences that lose critical motion cues (e.g., joint velocities). Conventional HMP methods assume clean MoCap historical poses as input and are highly sensitive to noisy radar-estimated histories; direct application yields unrealistic future sequences. Therefore, a dedicated HMP framework tailored to the radar modality is required.

## Core Problem

How to extract reliable historical motion information from noisy, temporally inconsistent mmWave radar point clouds, and generate realistic, stable future human pose sequences?

## Method

### Overall Architecture

mmPred adopts a two-stage training procedure: (1) training the dual-domain historical motion estimation module; and (2) training the GST-based diffusion model for future motion generation. The input consists of $H$ historical frames of radar point clouds $R^{1:H} = \{R^i \in \mathbb{R}^{N \times 6}\}_{i=1}^H$, and the output is the predicted future $F$-frame pose sequence $\hat{x}^{H+1:H+F}$.

### Key Design 1: Dual-Domain Historical Motion Estimation

**Time-domain Pose Refinement (TPR)**: For each radar point cloud frame, a coarse pose is first obtained via a pre-trained pose estimator $f_{\text{pose}}$, then a diffusion-based refinement network $f_{\text{refine}}$ corrects it by leveraging limb-length consistency and inter-frame continuity priors:

$$\tilde{x}_{\text{time}}^i = f_{\text{refine}}(f_{\text{pose}}(R^i))$$

**Frequency-domain Dominant Motion (FDM)**: All historical frames are processed holistically. An anchor-based point cloud encoder $f_{\text{pc}}$ and a Transformer $\Phi$ directly predict a frequency-domain motion representation $\tilde{X}_{\text{freq}}^{1:N_2} \in \mathbb{R}^{N_2 \times J \times 3}$ ($N_2 = 3$ or 4 DCT coefficients):

$$F_r = f_{\text{pc}}(R^{1:H}), \quad F_r', F_j = \Phi(F_r, \bar{F}_j)$$
$$\tilde{X}_{\text{freq}}^{1:N_2} = f_m(F_j)$$

Low-frequency DCT coefficients capture dominant motion trends (mean pose and velocity), naturally separating high-frequency noise.

### Key Design 2: Cross-Domain Fusion

Both time-domain and frequency-domain representations are transformed into a unified space of $N_1$ DCT coefficients, independently projected via two MLPs, and added element-wise:

$$C = f_1(\tilde{X}_{\text{time}}^{1:N_1}) + f_2(\tilde{X}_{\text{freq}}^{1:N_1})$$

The resulting joint-level condition embedding $C \in \mathbb{R}^{J \times 384}$ serves as the guidance signal for the diffusion model.

### Key Design 3: Global Skeleton Transformer (GST)

GST isolates joint features to prevent dropped joints from corrupting the global representation:
- **Skeleton Transformer (S-Transformer)**: Features are reshaped to $\mathbb{R}^{J \times (N_1 \times 384)}$; self-attention across $J$ joint tokens models global joint cooperation, enabling dropped joints to aggregate information from reliable ones.
- **Frequency Transformer (F-Transformer)**: Features are reshaped to $\mathbb{R}^{N_1 \times (J \times C_a)}$; temporal motion patterns across the frequency dimension are modeled to ensure temporal smoothness of the generated motion.

The training objective follows the standard DDPM $\varepsilon$-prediction loss:

$$\mathcal{L}_2 = \|\hat{\varepsilon}_\theta(X_k, k, C) - \varepsilon_k\|^2$$

## Key Experimental Results

### mmBody Dataset: Performance under Adverse Conditions

| Method | Input | Lab1 ADE↓ | Rain ADE↓ | Smoke ADE↓ | Dark ADE↓ | Occlusion ADE↓ | Avg. ADE↓ |
|--------|-------|----------|----------|-----------|----------|---------------|----------|
| HumanMAC | GT | 0.235 | 0.297 | 0.338 | 0.287 | 0.265 | 0.291 |
| HumanMAC | RGB | 0.390 | 0.479 | 0.560 | 0.693 | 0.739 | 0.547 |
| PSGSN | mmWave | 0.503 | 0.536 | 0.598 | 0.513 | 0.485 | 0.537 |
| HumanMAC | mmWave | 0.411 | 0.455 | 0.496 | 0.406 | 0.391 | 0.460 |
| **mmPred** | **mmWave** | **0.369** | **0.436** | **0.472** | **0.392** | **0.378** | **0.420** |

mmPred reduces average ADE by 8.6% and FDE by 6.4% compared to HumanMAC. In Dark and Occlusion scenarios, the RGB method's ADE surges to 0.693/0.739, whereas mmPred achieves only 0.392/0.378.

### mm-Fi Dataset: Performance across Action Types

| Method | Raise Hand ADE↓ | Pickup ADE↓ | Throwing ADE↓ | Kicking ADE↓ | Avg. ADE↓ | Avg. FDE↓ |
|--------|----------------|------------|--------------|-------------|----------|----------|
| PSGSN | 0.397 | 0.595 | 0.449 | 0.426 | 0.430 | 0.470 |
| HumanMAC | 0.363 | 0.547 | 0.439 | 0.425 | 0.408 | 0.396 |
| **mmPred** | **0.237** | **0.452** | **0.371** | **0.374** | **0.319** | **0.305** |

mmPred reduces ADE by 22% and FDE by 23% on mm-Fi.

### Ablation Study

| Configuration | mmBody ADE↓ | mmBody FDE↓ | mm-Fi ADE↓ | mm-Fi FDE↓ |
|--------------|-----------|-----------|----------|----------|
| Baseline (M1) | 0.460 | 0.487 | 0.408 | 0.396 |
| +TPR (M2) | 0.455 | 0.485 | 0.379 | 0.359 |
| +FDM (M3) | 0.456 | 0.486 | 0.337 | 0.326 |
| +GST (M4) | 0.460 | 0.485 | 0.373 | 0.354 |
| TPR+FDM (M5) | 0.448 | 0.476 | 0.355 | 0.327 |
| FDM+GST (M6) | 0.423 | 0.458 | 0.325 | 0.310 |
| **Full (M8)** | **0.420** | **0.456** | **0.319** | **0.305** |

FDM yields the most significant gains on mm-Fi (where point clouds are sparser). The S-Transformer reduces limb-length error from 10.67 to 9.92 and jitter from 7.01 to 6.09.

## Highlights & Insights

- **First introduction of radar into HMP**: A complete radar-HMP framework is established, opening a new direction for motion prediction in adverse environments.
- **Elegant dual-domain complementarity**: TPR provides precise joint localization in the time domain, while FDM supplies stable motion trends and velocity information in the frequency domain; DCT naturally separates high-frequency noise.
- **Joint-level isolation and cooperation in GST**: By isolating joint features before global self-attention, dropped joints can "borrow" information from reliable ones, effectively mitigating radar joint dropout.
- **Overwhelming advantage in adverse conditions**: In dark and occluded scenarios, mmPred outperforms RGB-based methods by more than 40% in ADE.

## Limitations & Future Work

- **Dependence on pre-trained pose estimators**: TPR requires a pre-trained radar pose estimator and refinement network, introducing additional error sources and computational overhead.
- **Validation limited to small-scale indoor datasets**: Both mmBody and mm-Fi are limited in scale; generalization to large-scale or outdoor environments remains unverified.
- **Radar hardware dependency**: mmBody uses a Phenix radar and mm-Fi uses a low-bandwidth radar; transferability across different radar devices has not been explored.
- **Raw radar signals are not exploited**: The method relies solely on processed point clouds; raw radar signals (e.g., Range-Doppler maps) may contain richer motion information.

## Related Work & Insights

- **PSGSN (Li et al. 2022)**: GCN-based deterministic prediction that is sensitive to noisy historical poses; mmPred models the motion distribution via a diffusion model for more robust prediction.
- **HumanMAC (Chen et al. 2023)**: Diffusion-based HMP that assumes clean MoCap history; performance degrades under radar noise. mmPred mitigates this by providing denoised historical guidance through FDM.
- **BelFusion (Barquero et al. 2023)**: Latent-space multi-stage training that likewise assumes ideal input and does not account for sensor-specific noise.
- **P4Transformer / PointTransformer et al.**: Radar pose estimators that process frames independently and ignore temporal consistency; mmPred's FDM processes the entire historical sequence holistically.
- **milliFlow (Ding et al. 2024)**: Uses radar joint motion flow to assist human perception but does not extend to the motion prediction task.

**Broader implications**:
- **Frequency-domain analysis as a general denoising tool**: The idea of concentrating motion trends in low-frequency DCT coefficients is transferable to motion understanding tasks with other noisy sensors (e.g., WiFi CSI, UWB).
- **Cross-modal motion understanding**: The radar → pose → prediction pipeline lays a foundation for multi-modal motion prediction (radar + IMU, radar + WiFi).
- **Joint-isolation feature design**: The design in GST — isolating joint features before global interaction — offers a reference for other tasks with local failure modes (e.g., pose estimation under occlusion).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First formulation of the radar HMP task; the dual-domain + GST design is well-motivated and principled.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Two datasets, multiple adverse conditions, and detailed ablations; dataset scale is somewhat limited.
- Writing Quality: ⭐⭐⭐⭐ — Problem motivation is clearly articulated; the intuition behind dual-domain complementarity is well supported by visualizations.
- Value: ⭐⭐⭐⭐ — Opens a new direction for privacy-preserving and adverse-condition motion prediction with clear application scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Spatiotemporal-Untrammelled Mixture of Experts for Multi-Person Motion Prediction](spatiotemporal-untrammelled_mixture_of_experts_for_multi-person_motion_predictio.md)
- [\[AAAI 2026\] SOSControl: Enhancing Human Motion Generation through Saliency-Aware Symbolic Orientation and Timing Control](soscontrol_enhancing_human_motion_generation_through_saliency-aware_symbolic_ori.md)
- [\[CVPR 2026\] MoLingo: Motion-Language Alignment for Text-to-Human Motion Generation](../../CVPR2026/human_understanding/molingo_motion-language_alignment_for_text-to-motion_generation.md)
- [\[NeurIPS 2025\] RAPTR: Radar-Based 3D Pose Estimation Using Transformer](../../NeurIPS2025/human_understanding/raptr_radar-based_3d_pose_estimation_using_transformer.md)
- [\[CVPR 2026\] RAM: Recover Any 3D Human Motion in-the-Wild](../../CVPR2026/human_understanding/ram_recover_any_3d_human_motion_in-the-wild.md)

</div>

<!-- RELATED:END -->

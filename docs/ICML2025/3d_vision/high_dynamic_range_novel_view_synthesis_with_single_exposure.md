---
title: >-
  [Paper Note] High Dynamic Range Novel View Synthesis with Single Exposure
description: >-
  [ICML 2025][3D Vision][HDR Novel View Synthesis] First proposes the problem setting of HDR novel view synthesis (HDR-NVS) using only single-exposure LDR images, and designs Mono-HDR-3D, a meta-algorithm framework based on camera imaging principles. It achieves HDR scene modeling without HDR supervision through an LDR-to-HDR Color Converter (L2H-CC) and an HDR-to-LDR closed-loop Color Converter (H2L-CC).
tags:
  - "ICML 2025"
  - "3D Vision"
  - "HDR Novel View Synthesis"
  - "Single Exposure"
  - "Camera Imaging Modeling"
  - "NeRF"
  - "3D Gaussian Splatting"
date: 2026-05-08
content_hash: 6ad26b766e4fa02a
---

# High Dynamic Range Novel View Synthesis with Single Exposure

**Conference**: ICML 2025  
**arXiv**: [2505.01212](https://arxiv.org/abs/2505.01212)  
**Code**: [github.com/prinasi/Mono-HDR-3D](https://github.com/prinasi/Mono-HDR-3D)  
**Area**: 3D Vision  
**Keywords**: HDR Novel View Synthesis, Single Exposure, Camera Imaging Modeling, NeRF, 3D Gaussian Splatting

## TL;DR

First proposes the problem setting of HDR novel view synthesis (HDR-NVS) using only single-exposure LDR images, and designs Mono-HDR-3D, a meta-algorithm framework based on camera imaging principles. It achieves HDR scene modeling without HDR supervision through an LDR-to-HDR Color Converter (L2H-CC) and an HDR-to-LDR closed-loop Color Converter (H2L-CC).

## Background & Motivation

The goal of HDR novel view synthesis is to establish a 3D HDR scene model from LDR images and generate HDR rendered images from arbitrary viewpoints. Existing methods (HDR-NeRF, HDR-GS) rely on **multi-exposure LDR images** as training data, which suffers from the following inherent limitations:

**Motion artifacts**: Long-exposure frames accumulate blur due to object/camera motion, and displacement between different exposures generates ghosting.

**Alignment difficulties**: Different exposure times lead to differences in luminance distribution and local contrast, increasing registration difficulty.

**High acquisition cost**: Requires specialized equipment and multiple shots, making it hard to implement in dynamic environments or on mobile devices.

The authors propose a more practical and challenging new task: **Single-Exposure HDR-NVS**—training using only LDR images with a single exposure time. The core challenge is that single-exposure images inevitably contain overexposed or underexposed regions, leading to incomplete information and rendering direct reconstruction of HDR content impossible.

## Method

### Overall Architecture

Mono-HDR-3D is a **meta-algorithm** that can be seamlessly integrated into any NVS model, such as NeRF or 3DGS. The overall pipeline consists of three stages:

1. **LDR 3D Scene Modeling**: Takes single-exposure LDR images and camera poses as input to train a standard LDR 3D scene model (NeRF/3DGS).
2. **LDR-to-HDR Lifting**: Promotes the LDR color space to HDR via an L2H-CC (LDR-to-HDR Color Converter).
3. **HDR-to-LDR Closed-Loop**: Converts the HDR images back to LDR via an H2L-CC (HDR-to-LDR Color Converter) to form a closed loop, enabling self-supervised training without HDR labels.

Key design concept: **First model LDR, then lift to HDR**, rather than attempting to directly construct an HDR model from single-exposure LDR (which fails). This is the opposite design route of prior methods.

### Key Designs

#### Camera Imaging Mechanism Modeling

The core innovation lies in designing the network architectures of L2H-CC and H2L-CC based on **物理成像公式 (physical imaging formulations)**.

**LDR Image Formation Formula** (forward process from HDR to LDR):

$$I^l = \frac{\Delta t}{g} \cdot I^h + I_0 + \epsilon - I_{\text{overflow}}$$

where $\Delta t$ is the exposure time, $g$ is the sensor gain, $I^h$ is the HDR pixel value, $I_0$ is the dark current offset, $\epsilon$ is the sensor noise, and $I_{\text{overflow}}$ is the saturation overflow value. This formula uniformly describes the imaging process of both saturated and unsaturated pixels, which can be organized into two functional terms:

- $D(\cdot)$: Linearly scales HDR radiance to the LDR range.
- $B(\cdot)$: Learns the offset and correction of LDR radiance.

**Inverse Formula** (backward process from LDR to HDR):

$$I^h = \underbrace{\frac{g}{\Delta t}}_{X(\cdot)} \cdot \underbrace{(I^l - I_0 + I_{\text{overflow}})}_{S(\cdot)} - \underbrace{\frac{g}{\Delta t} \cdot \epsilon}_{Y(\cdot)}$$

This is decomposed into three functional terms:
- $X(\cdot)$: Linear amplification factor, which linearly maps LDR values to the HDR range.
- $S(\cdot)$: Offset correction, adjusting the amplified LDR values.
- $Y(\cdot)$: Noise correction term.

#### L2H-CC (LDR-to-HDR Color Converter)

L2H-CC is a **per-channel** operation, and its network structure strictly simulates the three terms of the inverse formula:

1. **Input Mapping**: Linear layer + ReLU, embedding LDR colors into a latent feature space.
2. **Three-Branch Simulation**:

    - $X(\cdot)$ branch: MLP + ReLU (ensuring non-negativity to satisfy physical constraints).
    - $S(\cdot)$ branch: MLP + ReLU (non-negative correction values).
    - $Y(\cdot)$ branch: MLP **without activation function** (noise is inherently random, hence no non-negativity constraint).
3. **Residual Connection**: The LDR input is added to the converted output through a residual structure to preserve fine color details and stabilize the learning process.

#### H2L-CC (HDR-to-LDR Color Converter, Closed-Loop Design)

H2L-CC simulates the forward imaging formula, mapping the rendered HDR images back to LDR. This allows for supervised learning by comparing with LDR training images, even in the absence of ground-truth HDR data:

1. **Input Mapping**: Linear layer + ReLU.
2. **Two-Branch Simulation**:

    - $D(\cdot)$ branch: Linear layer + ReLU (non-negative linear scaling).
    - $B(\cdot)$ branch: Linear layer + Tanh (offset correction, allowing positive and negative values).
3. **Output Mapping**: Sigmoid activation, constraining the values to the LDR range of [0,1].

### Loss & Training

Overall loss function:

$$\mathcal{L} = \mathcal{L}_{\text{ldr}} + \alpha \mathcal{L}_{\text{hdr}} + \beta \mathcal{L}_{\text{h2l}}$$

**Mono-HDR-GS Instantiation** (when integrated with 3DGS):

- $\mathcal{L}_{\text{ldr}}$: L1 loss + D-SSIM loss (standard 3DGS loss), balanced by weight $\lambda$.
- $\mathcal{L}_{\text{hdr}}$: L2 loss in the $\mu$-law domain, calculated after applying logarithmic compression to the HDR values.
- $\mathcal{L}_{\text{h2l}}$: Identical form as $\mathcal{L}_{\text{ldr}}$, but applied to the output of H2L-CC.

**Mono-HDR-NeRF Instantiation** (when integrated with NeRF): MSE is used for all three losses.

**Hyperparameters**: $\alpha=0.6$, $\beta=0.01$ (NeRF) / $0.05$ (3DGS); L2H-CC learning rate of $5 \times 10^{-4}$, H2L-CC learning rate of $1 \times 10^{-3}$.

**Key**: In the strict single-exposure setting, $\alpha=0$ (since no ground-truth HDR exists). In this case, the closed-loop design of H2L-CC serves as the only source of extra supervision.

## Key Experimental Results

### Main Results

Dataset: 8 synthetic scenes (Blender) + 4 real-world scenes. Each scene contains 35 images across 5 exposure times. Under the single-exposure setting, 1 exposure is randomly selected for training. Evaluation metrics: PSNR↑ / SSIM↑ / LPIPS↓.

**Synthetic Dataset Results (LDR + HDR NVS)**:

| Method | Speed(fps) | LDR-PSNR↑ | LDR-SSIM↑ | LDR-LPIPS↓ | HDR-PSNR↑ | HDR-SSIM↑ | HDR-LPIPS↓ |
|------|----------|-----------|-----------|------------|-----------|-----------|------------|
| HDR-NeRF | 0.26 | 30.62 | 0.658 | 0.285 | 13.76 | 0.511 | 0.443 |
| **Mono-HDR-NeRF** | 0.26 | **38.78** | **0.936** | **0.048** | **32.86** | **0.940** | **0.068** |
| HDR-GS | 147.45 | 39.48 | 0.977 | 0.018 | 35.30 | 0.965 | 0.030 |
| **Mono-HDR-GS** | 136.97 | **41.68** | **0.983** | **0.009** | **38.57** | **0.975** | **0.012** |

**Real Dataset Results (LDR NVS, without HDR ground truth)**:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| HDR-NeRF | 32.50 | 0.948 | 0.069 |
| Mono-HDR-NeRF | 32.52 | 0.948 | 0.069 |
| HDR-GS | 35.34 | 0.966 | 0.019 |
| **Mono-HDR-GS** | **35.81** | **0.967** | **0.017** |

### Ablation Study

**Module Design Ablation** (Synthetic data, HDR NVS metrics):

| Configuration | HDR-PSNR | HDR-SSIM | HDR-LPIPS | Description |
|------|----------|----------|-----------|------|
| Replace L2H-CC with MLP | 19.02 | 0.778 | 0.327 | L2H-CC is core, performance drops heavily after replacement |
| Replace H2L-CC with MLP | 38.43 | 0.974 | 0.015 | Slight decrease, closed-loop design has a positive effect |
| Full Model | **38.57** | **0.975** | **0.012** | - |

**Loss Combination Ablation** (Synthetic data, HDR NVS metrics):

| Configuration | HDR-PSNR | HDR-SSIM | HDR-LPIPS | Description |
|------|----------|----------|-----------|------|
| Only $\mathcal{L}_{\text{ldr}}$ | - | - | - | Unable to train |
| Only $\mathcal{L}_{\text{hdr}}$ | 33.93 | 0.925 | 0.050 | Basically usable |
| $\mathcal{L}_{\text{ldr}}+\mathcal{L}_{\text{hdr}}$ | 38.19 | 0.974 | 0.015 | LDR provides geometric regularization |
| All three losses | **38.57** | **0.975** | **0.012** | Closed-loop contributes +0.38dB |

### Key Findings

1. **HDR-NeRF fails almost completely under single exposure**: HDR PSNR is only 13.76dB (tending toward all-black or all-white outputs), which proves that multi-exposure methods cannot directly transfer to the single-exposure setting.
2. **Mono-HDR-NeRF vs HDR-NeRF**: HDR PSNR improves by +19.1dB (13.76 to 32.86), showing an enormous leap in quality.
3. **Mono-HDR-GS vs HDR-GS**: HDR PSNR improves by +3.27dB (35.30 to 38.57), yielding a significant enhancement even on an already strong baseline.
4. **No loss in efficiency**: The inference speed of Mono-HDR-GS (136.97fps) is basically on par with HDR-GS (147.45fps).
5. **Robustness to LDR/HDR ratio**: Even when the ratio of LDR:HDR is 5:1, Mono-HDR-GS retains 92.6% of the peak PSNR.

## Highlights & Insights

1. **Clever problem definition**: Proposes single-exposure HDR-NVS as an independent problem for the first time, lowering data collection requirements while eliminating the inherent flaws of multi-exposure methods.
2. **Physics-driven network design**: The architectures of L2H-CC and H2L-CC are derived directly from camera imaging formulas. Each network branch corresponds to a physical term, which is significantly more effective than a black-box MLP (as confirmed by the ablation study where replacing L2H-CC with an MLP caused a drops in PSNR by over 19dB).
3. **Closed-loop self-supervision concept**: The closed-loop design of H2L-CC allows for the indirect supervision of HDR space learning using only LDR images, providing an elegant label-free learning strategy.
4. **Meta-algorithm design**: Functions as a plug-and-play module that can be integrated into any NVS backbone, as validated by both NeRF and 3DGS instantiations.

## Limitations & Future Work

1. **Limited improvement in real-world scenes**: The improvement of LDR NVS metrics on real-world data is marginal (PSNR increases by only +0.47dB), indicating that the method's advantages narrow under complex real illumination conditions.
2. **Information ceiling in single exposure**: Information in severely overexposed or underexposed regions is inherently lost, making it difficult for the network to truly recover it through learning alone.
3. **Evaluation limitations**: The ground-truth HDR for synthetic data comes from renderers, which may exhibit distribution shifts compared to real-world HDR images.
4. **No video/dynamic scenes explored**: The framework currently only processes multi-view images of static scenes.
5. **Scalable to more backbones**: The paper only validates NeRF and 3DGS, but integration into more efficient models like Instant-NGP or Zip-NeRF could be explored.

## Related Work & Insights

- **HDR-NeRF (CVPR 2022)**: The first HDR-NVS method. It learns an implicit mapping from radiance to HDR colors based on NeRF, but requires multi-exposure images and is expensive to train and infer.
- **HDR-GS (NeurIPS 2024)**: HDR-NVS based on 3DGS. It significantly improves efficiency (1000× speedup) but still relies on multi-exposure images.
- **Single-Image HDR Reconstruction** (Eilertsen 2017, DCDR-UNet 2024): 2D image LDR-to-HDR translation, which lacks 3D consistency.
- **Inspiration**: Physics-inspired network architecture design (rather than purely data-driven) is highly effective in scenarios with limited prior information; the closed-loop design idea can be extended to other 3D reconstruction tasks that lack corresponding labels.

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Novelty | 4.5 | First proposes the single-exposure HDR-NVS problem, with a novel physics-driven closed-loop design |
| Technical Depth | 4.0 | Rigorous formula derivation, with modules designed based on physical principles |
| Experimental Thoroughness | 4.0 | Synthetic + real-world data with multi-angle ablations, though real-world experiments are slightly weak |
| Practical Value | 4.5 | A plug-and-play meta-algorithm that significantly reduces data acquisition requirements |
| Writing Quality | 4.0 | Clear structure with well-elaborated motivation |
| **Overall Score** | **4.2** | Valuable problem definition, elegant method design, and persuasive experiments |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SeHDR: Single-Exposure HDR Novel View Synthesis via 3D Gaussian Bracketing](../../ICCV2025/3d_vision/sehdr_single-exposure_hdr_novel_view_synthesis_via_3d_gaussian_bracketing.md)
- [\[CVPR 2025\] InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](../../CVPR2025/3d_vision/instanthdr_single-forward_gaussian_splatting_for_high_dynamic_range_3d_reconstru.md)
- [\[ICLR 2026\] Dynamic Novel View Synthesis in High Dynamic Range](../../ICLR2026/3d_vision/dynamic_novel_view_synthesis_in_high_dynamic_range.md)
- [\[NeurIPS 2025\] NerfBaselines: Consistent and Reproducible Evaluation of Novel View Synthesis Methods](../../NeurIPS2025/3d_vision/nerfbaselines_consistent_and_reproducible_evaluation_of_novel_view_synthesis_met.md)
- [\[CVPR 2025\] Dual Exposure Stereo for Extended Dynamic Range 3D Imaging](../../CVPR2025/3d_vision/dual_exposure_stereo_for_extended_dynamic_range_3d_imaging.md)

</div>

<!-- RELATED:END -->

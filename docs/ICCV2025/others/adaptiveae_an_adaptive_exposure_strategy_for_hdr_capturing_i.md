---
title: >-
  [Paper Note] AdaptiveAE: An Adaptive Exposure Strategy for HDR Capturing in Dynamic Scenes
description: >-
  [ICCV 2025][HDR imaging] This paper proposes AdaptiveAE, which formulates HDR bracketed exposure capture as a Markov Decision Process (MDP) using deep reinforcement learning, jointly optimizing ISO and shutter speed combinations to adaptively select optimal exposure parameters for dynamic scenes within a user-defined time budget. The method achieves PSNR 39.70 on the HDRV dataset, outperforming the previous best method Hasinoff et al. (37.59) by 2.1 dB.
tags:
  - ICCV 2025
  - HDR imaging
  - auto-exposure
  - reinforcement learning
  - motion blur
  - exposure fusion
date: 2026-05-08
content_hash: 06f79a9a05c5392e
---

# AdaptiveAE: An Adaptive Exposure Strategy for HDR Capturing in Dynamic Scenes

**Conference**: ICCV 2025
**arXiv**: [2508.13503](https://arxiv.org/abs/2508.13503)
**Code**: None (not released)
**Area**: Other
**Keywords**: HDR imaging, auto-exposure, reinforcement learning, motion blur, exposure fusion

## TL;DR
This paper proposes AdaptiveAE, which formulates HDR bracketed exposure capture as a Markov Decision Process (MDP) using deep reinforcement learning, jointly optimizing ISO and shutter speed combinations to adaptively select optimal exposure parameters for dynamic scenes within a user-defined time budget. The method achieves PSNR 39.70 on the HDRV dataset, outperforming the previous best method Hasinoff et al. (37.59) by 2.1 dB.

## Background & Motivation
The core idea of HDR imaging is to fuse multiple LDR images captured at different exposures into a single HDR image with wide dynamic range. Exposure parameter selection is critical in this process: excessively long shutter speeds introduce motion blur, high ISO introduces noise, and large exposure gaps increase the risk of alignment failure. However, existing methods exhibit several key shortcomings:

1. **Ignoring ISO–shutter speed interaction**: Most methods only adjust shutter speed (EV) while keeping ISO fixed, precluding an optimal trade-off between noise and blur.
2. **Neglecting motion blur in dynamic scenes**: Classic methods such as Hasinoff et al. are designed for static scenes and optimize only for SNR while ignoring motion-induced quality degradation.
3. **Treating motion blur and ghosting as post-processing problems**: Existing pipelines typically perform deblurring after fusion, but experiments in this paper show that post-processing deblurring has limited effectiveness—once an LDR frame is blurred, no fusion method can recover it.

## Core Problem
**How to adaptively select the optimal ISO and shutter speed combination in dynamic scenes to maximize the quality of the fused HDR image?** The challenges are: (1) ISO and shutter speed form a high-dimensional discrete action space (24 ISO levels × 19 shutter speeds = 456 combinations); (2) the optimal strategy depends on scene content—regions with fast-moving objects require shorter shutter speeds, while dark regions require higher ISO; (3) the choice for each frame affects the optimal strategy for subsequent frames, exhibiting the nature of sequential decision-making.

## Method

### Overall Architecture
The AdaptiveAE pipeline consists of two parts: **training** and **inference**.

- **Input**: 3 preview LDR images (underexposed, normally exposed, overexposed, with EV spacing {-2, 0, +2})
- **Output**: Optimal ISO and shutter speed combination for each LDR frame

The entire process is a 3-stage sequential refinement procedure:
1. **Stage 1**: Predict the optimal ISO and shutter speed for the middle frame (0 EV); side frames (±2 EV) are adjusted symmetrically.
2. **Stage 2**: Refine the EV offset of the underexposed frame to $-y$; the middle frame inherits parameters from the previous stage; the overexposed frame is symmetrically adjusted to $+y$.
3. **Stage 3**: Refine the EV offset of the overexposed frame to $+z$, allowing asymmetry; the final exposure settings are $\{-y, 0, +z\}$.

During training, LDR images are simulated via a blur-aware data synthesis pipeline; during inference, images are captured directly by the camera.

### Key Designs

1. **Blur-aware Data Synthesis Pipeline**: This is one of the most important technical contributions of the paper. Existing HDR datasets do not contain motion blur, making it impossible to train exposure strategies that account for motion. A two-step synthesis workflow is designed:

    - **Motion blur synthesis**: Given two consecutive HDR ground-truth frames, tone mapping is first applied via $\mu$-law to convert to the LDR space; RIFE is then used to interpolate to 256 frames, and a corresponding number of frames are averaged according to shutter speed $T_j$ to obtain the blurred HDR (Eq. 2). A key insight is that blur must be applied before noise, as blur affects the raw photon-capture process.
    - **Noise synthesis**: The physical noise model of [Hasinoff 2010] is adopted, where noise variance consists of three independent sources—photon noise (proportional to signal intensity and exposure time), read noise, and ADC noise (Eq. 3). Noise at the corresponding level can be synthesized precisely given the specified ISO and shutter speed.

2. **MDP Formulation and A3C Optimization**: Exposure bracket selection is modeled as an MDP, where the state is the current exposure settings of the three LDR frames and the action is the (ISO, shutter speed) pair selected for the next frame. The policy network (actor) outputs an action probability distribution, and the value network (critic) estimates state values. End-to-end training is performed using A3C (Asynchronous Advantage Actor-Critic). The action space is discretized into 24 ISO levels × 19 shutter speeds.

3. **Multi-branch CNN Architecture**:

    - **Semantic feature branch**: A pretrained AlexNet extracts semantic features from the middle frame (4096→1024→256 dimensions), helping identify important regions in the scene.
    - **Irradiance feature branch**: Histograms are extracted separately for each LDR and concatenated, then processed through 3-layer 1D convolutions (128→256→512, kernel=4) to encode exposure information.
    - **Stage encoding branch**: The current stage index and total number of stages are encoded (2→32→64 dimensions), enabling the network to adjust its strategy based on the remaining exposure budget.
    - **Feature fusion**: Features from all branches are concatenated and fused through two fully connected layers (512→256).

4. **Carefully Designed Reward Function** (Eq. 5–7):

    - **Construction reward $P_\text{construction}$**: L2 loss between the fused HDR and the ground truth (primary reward term).
    - **Priority region reward $P_\text{priority}$**: A saliency predictor [SalGAN] generates importance masks for key regions; an additional L2 constraint is applied to these regions to ensure the highest quality for critical areas such as faces.
    - **Ghost reward $P_\text{ghost}$**: RAFT is used to compute optical flow; pixels in motion-exceeding threshold $K=0.2$ are selected, and an additional L2 constraint is applied to these high-risk regions.
    - **Step penalty $P(j)$**: A penalty of $\alpha(j-H)^2$ is imposed when more than $H=3$ frames are used, encouraging high-quality HDR capture with as few frames as possible.

### Loss & Training
- Training uses A3C asynchronous optimization; the fusion network DeepHDR is used during training to compute rewards but is not required at inference.
- Training data: 770 scenes from Real-HDRV (440 dynamic + 330 static), cropped to 512×512 with random flipping and rotation augmentation.
- RIFE interpolation is time-consuming; blur synthesis is performed offline prior to training.
- At inference, the RL agent runs in <5 ms/scene on an RTX 3080, with the full pipeline completing in <250 ms.

## Key Experimental Results

| Dataset | Metric | Ours | Hasinoff et al. | Wang et al. | Pourreza et al. | Gain (vs Hasinoff) |
|--------|------|------|----------|------|------|------|
| HDRV-Test (1 preview) | PSNR-μ | **39.70** | 37.59 | 36.46 | 33.64 | +2.11 |
| HDRV-Test (1 preview) | SSIM-μ | **0.9408** | 0.9052 | 0.8902 | 0.8617 | +0.036 |
| HDRV-Test (1 preview) | HDR-VDP-2 | **59.20** | 57.02 | 56.09 | 54.55 | +2.18 |
| HDRV-Test (1 preview) | PU-PSNR | **34.67** | 32.87 | 32.68 | 30.61 | +1.80 |
| DeepHDRVideo (3 preview) | PSNR-μ | **39.81** | 38.47 | 37.95 | 35.57 | +1.34 |

Cross-fusion-method evaluation (HDRV-Test, 1 preview):

| Fusion Method | Ours PSNR-μ | Hasinoff PSNR-μ | Wang PSNR-μ |
|----------|-------------|-----------------|-------------|
| DeepHDR | **39.70** | 37.59 | 36.46 |
| HDR-GAN | **40.73** | 38.58 | 37.95 |
| HDR-Transformer | **41.37** | 39.11 | 38.89 |

Near-optimal performance: Via Gaussian sampling search for local optima (50 samples/parameter/frame), the proposed method achieves PSNR 39.70 vs. local optimum 39.93, approaching the theoretical upper bound.

### Ablation Study
- **Base** (construction reward + step penalty only): PSNR 38.21 / SSIM 0.9227
- **Base + $P_\text{priority}$**: PSNR 38.57 / SSIM 0.9261 (+0.36 dB)
- **Base + $P_\text{priority}$ + $P_\text{ghost}$** (full model): PSNR **39.70** / SSIM **0.9408** (+1.49 dB)
- $P_\text{ghost}$ contributes the most (+1.13 dB), validating the importance of accounting for motion at the exposure stage.
- Post-processing deblurring (BANet applied as pre-processing/post-processing/fusion stage) improves Wang et al. from 36.46 to at most 37.33, far below the proposed method's 39.70—demonstrating that motion must be addressed at the capture stage.
- The optimal fixed-ISO baseline (W-optimal) only improves to 37.64, confirming the necessity of adaptive ISO.
- The performance advantage becomes more pronounced at higher motion magnitudes: at 60-pixel motion level, the PSNR gain of the proposed method is more significant.

## Highlights & Insights
- The approach of **addressing motion at the source** is highly intuitive and effective—rather than recovering from a blurred capture in post-processing, it is better to capture sharp images in the first place. Jointly optimizing ISO and shutter speed at the capture stage is a first in the exposure strategy literature.
- The **physics-driven data synthesis pipeline** is elegantly designed: the order of blur-before-noise conforms to physical laws, and the noise model is grounded in photon statistics, making the training data sufficiently realistic.
- **Flexible frame count**: Through the step penalty mechanism, the model automatically determines whether to capture 3 or 4 frames, unlike traditional methods that fix the bracket count at 3.
- **Real-device validation**: Tests were conducted with actual captures on a Sony Alpha 7C-II by manually setting parameters, going beyond simulation-only experiments.
- **Extremely lightweight model**: The RL agent has only 7–8M parameters, runs in <5 ms at inference, and the full pipeline completes in <250 ms, indicating strong potential for real-time deployment.

## Limitations & Future Work
- **Fixed aperture**: The current formulation assumes aperture and focal length are fixed; the paper mentions extending to adjustable aperture as future work.
- **Dependence on pretrained fusion network**: DeepHDR is used during training to compute rewards; the optimality of the learned policy is thus bounded by the quality of the fusion network. Retraining may be required if a stronger fusion network is adopted.
- **Offline blur synthesis**: RIFE interpolation is performed offline prior to training, limiting training flexibility.
- **Limited real-world validation**: Real-capture tests require manual ISO and shutter speed configuration; end-to-end camera integration has not yet been realized.
- **AlexNet for semantic branch**: The feature extraction capacity is limited; stronger vision foundation models could serve as a replacement.
- **Discrete action space**: 24×19=456 discrete actions may miss the optimal solution in the continuous space.
- **Single reference frame**: The middle frame is always used as the fusion reference, which may not be optimal for extremely dynamic scenes.

## Related Work & Insights
- **vs Hasinoff et al. [2010]**: Hasinoff mathematically optimizes ISO and shutter speed to maximize worst-case SNR but assumes a static scene and ignores motion blur. The proposed method achieves 2.1 dB higher PSNR in dynamic scenes.
- **vs Wang et al. [2020]**: Wang et al. also use RL to predict exposure, but (1) only predict shutter speed without adjusting ISO, and (2) do not account for motion blur. By jointly optimizing ISO and shutter speed with a motion-aware reward, the proposed method achieves 3.24 dB higher PSNR.
- **vs post-processing deblurring pipeline**: Combining Wang et al. with BANet/DeepHDR-blur achieves at best 37.33 dB (vs. 39.70 for the proposed method), demonstrating that post-processing cannot substitute for capture-time optimization.

The paradigm of **RL for camera control** can generalize to other imaging tasks, such as band selection in multispectral imaging, denoising strategies for low-light photography, and frame selection in mobile HDR+. The **physics-aware data synthesis** approach is also broadly applicable: accurate modeling of physical processes (blur + noise) enables training data generation without large-scale real-world annotation. The **sequential decision-making formulation** can be transferred to bit-rate allocation in video compression and multi-frame fusion strategies in autonomous driving.

## Rating
- Novelty: ⭐⭐⭐⭐ — First to jointly optimize ISO and shutter speed in an exposure strategy while accounting for motion blur; however, the RL+MDP framework itself was already introduced in Wang et al. 2020.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Multiple datasets, multiple fusion methods, comparison against post-processing deblurring, optimal ISO baselines, local optimum analysis, motion magnitude analysis, and real-capture tests; ablation study is highly detailed.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, well-motivated problem formulation, complete mathematical derivations; some sections involve dense notation requiring repeated cross-referencing.
- Value: ⭐⭐⭐⭐ — High practical value; optimizing at the capture stage is more fundamental than post-processing, with deployment potential on mobile devices and cameras, though hardware integration is still required for real-world adoption.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] AFUNet: Cross-Iterative Alignment-Fusion Synergy for HDR Reconstruction via Deep Unfolding Paradigm](afunet_crossiterative_alignmentfusion_synergy_for_hdr_recons.md)
- [\[ICCV 2025\] Recovering Parametric Scenes from Very Few Time-of-Flight Pixels](recovering_parametric_scenes_from_very_few_time-of-flight_pixels.md)
- [\[NeurIPS 2025\] Contextual Dynamic Pricing with Heterogeneous Buyers](../../NeurIPS2025/others/contextual_dynamic_pricing_with_heterogeneous_buyers.md)
- [\[NeurIPS 2025\] Overfitting in Adaptive Robust Optimization](../../NeurIPS2025/others/overfitting_in_adaptive_robust_optimization.md)
- [\[NeurIPS 2025\] Adaptive Data Analysis for Growing Data](../../NeurIPS2025/others/adaptive_data_analysis_for_growing_data.md)

<!-- RELATED:END -->

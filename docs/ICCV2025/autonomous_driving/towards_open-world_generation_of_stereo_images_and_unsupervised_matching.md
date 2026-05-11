---
title: >-
  [Paper Note] Towards Open-World Generation of Stereo Images and Unsupervised Matching
description: >-
  [ICCV 2025][Autonomous Driving][Stereo image generation] This paper proposes GenStereo, a diffusion-based stereo image generation framework that achieves both high visual quality and geometric accuracy through disparity-…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "Stereo image generation"
  - "diffusion models"
  - "unsupervised stereo matching"
  - "disparity-aware"
  - "adaptive fusion"
date: 2026-05-08
content_hash: 8ad5a220454b8d28
---

# Towards Open-World Generation of Stereo Images and Unsupervised Matching

**Conference**: ICCV 2025
**arXiv**: [2503.12720](https://arxiv.org/abs/2503.12720)
**Code**: [Project Page](https://qjizhi.github.io/genstereo)
**Area**: Autonomous Driving
**Keywords**: Stereo image generation, diffusion models, unsupervised stereo matching, disparity-aware, adaptive fusion

## TL;DR

This paper proposes GenStereo, a diffusion-based stereo image generation framework that achieves both high visual quality and geometric accuracy through disparity-aware coordinate embedding, cross-view attention, and an adaptive fusion mechanism, while advancing unsupervised stereo matching to a new state of the art.

## Background & Motivation

Stereo images are a fundamental data modality for XR devices, autonomous driving, and robotics, yet acquiring high-quality stereo images poses multiple challenges:

**Difficulty of real data acquisition**: Binocular cameras require precise calibration; real-world datasets either provide only sparse disparity annotations (e.g., KITTI) or are restricted to specific scenes (e.g., indoors). Synthetic datasets offer precise disparity but suffer from a domain gap.

**The dilemma of existing generation methods**:
   - **Warping-based methods** (e.g., MfS): High geometric accuracy, but occluded regions are filled with random backgrounds, causing semantic inconsistency.
   - **Diffusion-based methods** (e.g., StereoDiffusion): Semantically coherent, but disparity shifts are applied in latent space, lacking pixel-level precision.
   - SD-Inpainting produces semantically inappropriate occlusion fills that are discontinuous with surrounding textures.

**Bottleneck in unsupervised stereo matching**: Prior unsupervised methods are either constrained by simple warping with random fill (MfS) or limited to small-scale static scenes (NeRF-Stereo), with limited generalization.

The authors' core insight is that a unified framework is needed to simultaneously address visual quality and geometric accuracy — using geometrically precise warp results in reliable regions, leveraging diffusion models to generate semantically consistent content in occluded regions, and achieving seamless transitions via learned fusion weights.

## Method

### Overall Architecture

GenStereo employs a dual-stream U-Net architecture fine-tuned from Stable Diffusion pretrained weights:
- **Reference U-Net**: Processes the left view $(I_l, C_l)$ and extracts reference features.
- **Denoising U-Net**: Conditioned on $(I_{warp}, C_r)$ to synthesize the right view $\hat{I}_r$.
- A final adaptive fusion module blends the generated image with the warped image via learned weights.

### Key Designs

1. **Disparity-Aware Coordinate Embedding**:

    - **Mechanism**: A normalized 2D coordinate map $X \in \mathbb{R}^{h \times w \times 2}$ is constructed and transformed into coordinate embeddings via Fourier positional encoding $\phi$. The left-view embedding $C_l = \phi(X)$ remains unchanged, while the right-view embedding $C_r = \text{warp}(C_l, D_l)$ is warped according to disparity.
    - **Design Motivation**: Conventional inpainting methods exhibit visible boundaries between warped and filled regions. Coordinate embeddings provide implicit geometric guidance, enabling the model to understand spatial correspondences for each pixel. Compared to GenWarp's use of camera matrices, warping with a disparity map achieves more precise pixel-level control.
    - The warped image $I_{warp}$ is also provided as an additional conditioning input to the denoising U-Net.

2. **Cross-View Feature Enhancement**:

    - **Mechanism**: Left- and right-view features are concatenated in the attention mechanism: $q = F_r, \; k = [F_l, F_r], \; v = [F_l, F_r]$.
    - **Design Motivation**: Right-view generation requires reference to the semantic content of the left view. Dual-stream attention enables the model to adaptively balance semantic consistency from the reference view and geometric accuracy from the warped view. Text conditioning is replaced with CLIP image embeddings of the left image.

3. **Pixel-Space Alignment and Adaptive Fusion**:

    - **Dual-space supervision**: A pixel-space loss $L_{pixel} = \| \mathcal{D}(z_{pred}) - \mathcal{D}(z_{target}) \|_2^2$ is added alongside the standard latent-space loss $L_{latent}$, yielding $L = L_{latent} + \alpha L_{pixel}$ ($\alpha = 1$).
    - **Adaptive fusion module**: A lightweight convolutional network predicts spatially varying fusion weights $W = \sigma(f_\theta(\text{concat}(I_{gen}, I_{warp}, M)))$, and the final right view is $\hat{I}_r = M \odot W \odot I_{warp} + (1 - M \odot W) \odot I_{gen}$.
    - **Design Motivation**: LDM operations in latent space may sacrifice pixel-level precision. Pixel-space supervision directly constrains output quality. Adaptive fusion favors warped content in high-confidence regions ($M \approx 1$) and generated content in occluded regions, ensuring smooth transitions.

### Loss & Training

- **Mixed-dataset training**: 11 synthetic stereo datasets totaling 684K image pairs, covering both indoor and outdoor scenes.
- Real datasets are excluded, as even minor calibration errors degrade performance.
- Resampling strategy: small datasets are upsampled to 10% of the size of the largest dataset.
- Random square cropping and resizing to 512×512 (SD v1.5) or 768×768 (SD v2.1).
- The pretrained SD U-Net is fine-tuned for 3 epochs.
- **Random disparity dropout**: 10% of training samples have part of their disparity randomly dropped, simulating sparse GT scenarios such as KITTI.

## Key Experimental Results

### Main Results

**Stereo image generation quality (Table 2, Middlebury 2014 + KITTI 2015):**

| Method | SD Version | Middlebury PSNR↑ | Middlebury SSIM↑ | KITTI PSNR↑ | KITTI SSIM↑ |
|--------|------------|-----------------|-----------------|-------------|-------------|
| StereoDiffusion | 1.5 | 15.456 | 0.468 | 15.679 | 0.481 |
| SD-Inpainting | 1.5 | 15.740 | 0.412 | 9.792 | 0.230 |
| **GenStereo+Pseudo** | **2.1** | **25.142** | **0.911** | **23.488** | **0.849** |

GenStereo achieves approximately 10 dB improvement in PSNR and nearly doubles the SSIM.

**Unsupervised stereo matching (Table 3, KITTI 2012/2015):**

| Method | KITTI 2012 D1-all↓ | KITTI 2012 EPE↓ | KITTI 2015 D1-all↓ | KITTI 2015 EPE↓ |
|--------|-------------------|----------------|-------------------|----------------|
| SD-Inpainting | 3.907 | 0.894 | 4.490 | 1.059 |
| StereoDiffusion | 15.213 | 2.220 | 5.651 | 1.154 |
| **GenStereo** | **3.802** | **0.815** | **3.933** | **0.991** |

### Ablation Study

**Ablation of key components (inferred from paper descriptions):**

| Configuration | Key Effect | Explanation |
|---------------|-----------|-------------|
| w/o coordinate embedding | Reduced geometric accuracy | Lacks pixel-level spatial correspondence guidance |
| w/o pixel-space loss | Lower PSNR | Latent-space operations sacrifice pixel-level precision |
| w/o adaptive fusion | Visible warp/generation boundary | Cannot smoothly transition between high/low confidence regions |
| w/o random disparity dropout | Poor performance on sparse GT scenes | Model has not seen sparse inputs during training |
| Training with real data | Performance degradation | Minor calibration errors harm learning |

### Key Findings

- Stereo images generated with pseudo disparity (from monocular depth estimation) achieve quality comparable to or slightly better than those using GT disparity (PSNR 23.488 vs. 19.836 on KITTI), as MDE models provide denser disparity maps.
- Training exclusively on synthetic data yields strong generalization to real-world scenes (Middlebury, KITTI).
- SD v2.1 outperforms v1.5 (PSNR 25.142 vs. 23.835), benefiting from higher-resolution training.
- Generated stereo images can be directly used to train unsupervised stereo matching networks, significantly narrowing the gap with supervised methods.

## Highlights & Insights

- The **unified framework for visual quality and geometric accuracy** is the core contribution: rather than simple inpainting, it organically integrates warping and generation.
- The **synthetic-data-only training** decision is counterintuitive yet effective — calibration errors in real data are detrimental.
- The finding that **pseudo disparity outperforms GT disparity** is significant: it suggests that dense depth from MDE models is better suited for generation tasks than sparse LiDAR depth.
- This work opens a new avenue for unsupervised stereo matching: generating training data from monocular images and depth estimation, avoiding expensive binocular calibration.

## Limitations & Future Work

- Inference speed is constrained by the multi-step sampling of diffusion models (inference time is not reported).
- Resolution is limited to 512×512 or 768×768, making the method unsuitable for high-resolution applications.
- The approach depends on MDE model quality — errors in disparity estimation propagate into the generated outputs.
- Temporal consistency for moving objects in dynamic scenes is not explored.
- The adaptive fusion module uses only 3×3 convolutions, limiting its receptive field.

## Related Work & Insights

- The architecture draws inspiration from the dual-stream attention designs of GenWarp and Animate-Anyone, with the novel addition of disparity conditioning.
- Compared to Mono2Stereo, GenStereo achieves substantially better generation quality in occluded regions.
- The combination of monocular depth estimation and diffusion models is generalizable to multi-view generation, video depth estimation, and related tasks.
- The mixed-dataset training strategy (resampling-based balancing, synthetic-data-only) is worth adopting in other cross-domain tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of disparity-aware coordinate embedding and dual-space supervision is novel, though the dual-stream U-Net framework is largely borrowed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-benchmark evaluation with ablations covering key components, though inference efficiency analysis is absent.
- **Writing Quality**: ⭐⭐⭐⭐ Method descriptions are clear and figures are illustrative.
- **Value**: ⭐⭐⭐⭐⭐ Advances both stereo image generation and unsupervised stereo matching simultaneously, with high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] AGO: Adaptive Grounding for Open World 3D Occupancy Prediction](ago_adaptive_grounding_for_open_world_3d_occupancy_predictio.md)
- [\[NeurIPS 2025\] Towards Foundational LiDAR World Models with Efficient Latent Flow Matching](../../NeurIPS2025/autonomous_driving/towards_foundational_lidar_world_models_with_efficient_latent_flow_matching.md)
- [\[ICCV 2025\] Hermes: A Unified Self-Driving World Model for Simultaneous 3D Scene Understanding and Generation](hermes_a_unified_self-driving_world_model_for_simultaneous_3d_scene_understandin.md)
- [\[ICCV 2025\] Unleashing the Temporal Potential of Stereo Event Cameras for Continuous-Time 3D Perception](unleashing_the_temporal_potential_of_stereo_event_cameras_for_continuous-time_3d.md)
- [\[ICCV 2025\] A Constrained Optimization Approach for Gaussian Splatting from Coarsely-posed Images and Noisy Lidar Point Clouds](a_constrained_optimization_approach_for_gaussian_splatting_from_coarsely-posed_i.md)

</div>

<!-- RELATED:END -->

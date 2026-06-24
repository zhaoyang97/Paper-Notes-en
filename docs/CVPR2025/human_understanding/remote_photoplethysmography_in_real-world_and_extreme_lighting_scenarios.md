---
title: >-
  [Paper Note] Remote Photoplethysmography in Real-World and Extreme Lighting Scenarios
description: >-
  [CVPR 2025][Human Understanding][Remote Photoplethysmography (rPPG)] This paper proposes the first end-to-end video Transformer model for rPPG in real-world outdoor extreme lighting scenarios. It achieves robust physiological signal extraction using only an RGB camera through global interference sharing, background reference decoupling, and biological prior constraints.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Remote Photoplethysmography (rPPG)"
  - "Heart Rate Estimation"
  - "Illumination Interference"
  - "Self-Supervised Decoupling"
  - "Video Transformer"
date: 2026-05-08
content_hash: b380a46b50633f08
---

# Remote Photoplethysmography in Real-World and Extreme Lighting Scenarios

**Conference**: CVPR 2025  
**arXiv**: [2503.11465](https://arxiv.org/abs/2503.11465)  
**Code**: None  
**Area**: Human Understanding  
**Keywords**: Remote Photoplethysmography (rPPG), Heart Rate Estimation, Illumination Interference, Self-Supervised Decoupling, Video Transformer

## TL;DR

This paper proposes the first end-to-end video Transformer model for rPPG in real-world outdoor extreme lighting scenarios. It achieves robust physiological signal extraction using only an RGB camera through global interference sharing, background reference decoupling, and biological prior constraints.

## Background & Motivation

Remote photoplethysmography (rPPG) measures physiological metrics such as heart rate in a non-contact manner by capturing Blood Volume Pulse (BVP) signals from facial videos. While existing learning-based methods perform well under static indoor lighting, they face severe challenges in real-world outdoor scenarios:

(1) **Extreme Illumination Interference**: Facial color changes caused by cardiac activity are extremely subtle. External illumination variations (e.g., building shadows or turns in driving scenarios) can completely overwhelm the physiological signals.  
(2) **Periodic Interference**: Cardiac signals exhibit quasi-periodic characteristics, but streetlights or roadside trees also generate periodic lighting variations, making them difficult to disentangle.  
(3) **Hardware Dependency**: Some approaches rely on near-infrared (NIR) imaging assistance, which increases hardware costs and deployment difficulty.  
(4) **Heavy Models**: Current noise-resistant decoupling methods continuously stack modules and complex pipelines, making them difficult to deploy on mobile devices.

**Goal**: Design a lightweight model that achieves robust rPPG in extreme lighting conditions, such as real-world outdoor driving, using only an RGB camera (without NIR).

## Method

### Overall Architecture

An end-to-end U-shaped video Transformer framework. The input consists of an RGB video sequence, from which facial and background Spatio-Temporal Maps (STMaps) are constructed via facial landmark segmentation. The global STMap is reconstructed by a U-shaped Transformer to extract coarse-grained features $\mathbf{F}_{\text{coar}}$. Concurrently, the facial and background STMaps are processed by encoders to extract foreground features $\mathbf{F}_{\text{fore}}$ and background features $\mathbf{F}_{\text{back}}$. After self-supervised decoupling, fine-grained features $\mathbf{F}_{\text{fine}}$ are obtained, which are then used to regress the BVP waveform via temporal deconvolution.

### Key Design 1: Sliding Window Enhancement based on Biological Priors (BioSE)

**Function**: Enhances weak physiological signals without introducing distortions caused by global normalization.

**Mechanism**: Utilizing physiological knowledge (heart rate of 40–240 bpm, camera frame rate of 20–30 fps), local normalization is applied to the color variations of each facial landmark region within a predefined window length: $\text{BioSE} \; v_{t,n-t+s_{\text{norm}}}^{\text{face}} = (v_n^{\text{face}} - \min_t)/(\max_t - \min_t)$. Multiple different starting positions are processed in parallel to ensure continuous and smooth window edges. Finally, the result is concatenated with the original STMap along the channel dimension to yield a 6-channel enhanced STMap.

**Design Motivation**: Under severe interference, global normalization amplifies noise rather than the signal. Local window-based normalization, grounded in biological priors, normalizes only within a reasonable heartbeat period, preventing long-term interference from dominating. Retaining the original STMap ensures that the implicit optical relationships across RGB channels are preserved.

### Key Design 2: Self-Supervised Interference Decoupling with Background Reference

**Function**: Eliminates illumination interference in the foreground (face) that is consistent with the background.

**Mechanism**: Exploiting the commonality that both the face and the background are subject to the same environmental illumination interference, the background feature $\mathbf{F}_{\text{back}}$ is used as an interference reference. The temporal similarity between $\mathbf{F}_{\text{back}}$ and the foreground feature $\mathbf{F}_{\text{fore}}$ is computed, and similar components are adaptively removed: $\mathbf{F}_{\text{fine}} = (1 - \text{Softmax}(\mathbf{F}_{\text{back}} \cdot \mathbf{F}_{\text{fore}}^T)) \cdot \mathbf{F}_{\text{fore}}$. A contrastive learning loss $\mathcal{L}_c$ is further used to pull together the similarities between facial sub-regions and push apart the similarity between face and background features. Power Spectral Density (PSD) is employed as the distance metric to avoid the influence of periodic interference.

**Design Motivation**: In driving scenarios, the illumination changes on the face and background are highly consistent (homogeneous interference). Since the background contains no physiological signals, the background feature purely reflects environmental interference. Unlike heavy decoupling methods that utilize GANs or adversarial learning, this similarity calculation and Softmax scaling-based method is much more lightweight.

### Key Design 3: Coarse-to-Fine Reconstruction Guidance

**Function**: Facilitates progressive learning from coarse-grained global reconstruction to fine-grained BVP regression.

**Mechanism**: The U-shaped Transformer performs spatio-temporal reconstruction on the global STMap (without participating in decoupling), aiming to reconstruct a single-channel STMap stacked by the ground-truth (GT) BVP signals, guided by the loss $\mathcal{L}_r = \|\mathbf{F}_{\text{coar}}' - \mathbf{F}_{\text{bvp}}\|_2$. The global reconstruction features establish a coarse-grained perception of cardiac activity, continuously updating the network parameters to prevent the encoder from overfitting to fine-grained features or noise.

**Design Motivation**: Directly regressing the BVP on highly noisy inputs makes the model prone to overfitting the noise. The coarse-to-fine strategy first establishes a holistic understanding of cardiac signals via global reconstruction, followed by precise regression on the decoupled fine-grained features. This auxiliary global reconstruction task also enhances the spatio-temporal interaction representation capability of the Transformer.

### Loss & Training

Three loss terms are jointly optimized: the spatio-temporal reconstruction loss $\mathcal{L}_r$ (L2 distance), the contrastive decoupling loss $\mathcal{L}_c$ (contrastive learning based on PSD), and the BVP regression loss $\mathcal{L}_p$ (Pearson correlation coefficient loss, focusing on temporal regression and peak preservation).

## Key Experimental Results

### Cross-Dataset Heart Rate Estimation (MAE / RMSE / ρ, bpm)

| Method | MR-NIRP-IND MAE↓ | MR-NIRP-DRV MAE↓ | VIPL-HR MAE↓ | BUAA-MIHR MAE↓ |
|------|-----------------|-----------------|-------------|---------------|
| POS (Traditional) | 5.52 | 12.75 | 11.50 | 5.04 |
| DeepPhys (CNN) | 3.11/6.58 | 4.44/9.16 | — | — |
| Ours | **SOTA** | **SOTA** | **SOTA** | **SOTA** |

### Comparison in Key Scenarios

| Scenario | Performance of Prior Work | Ours |
|------|-----------|---------|
| Indoor Static Lighting | Relatively good | ≥ Prev. SOTA |
| Outdoor Driving (MR-NIRP-DRV) | Severe degradation | **Significant Lead** |
| Motion Scenarios (MR-NIRP-IND) | Moderate | **SOTA** |
| Illumination Variations (BUAA-MIHR) | Degradation | **SOTA** |

### Key Findings

1. **Breakthrough in Outdoor Driving Scenarios**: On the highly challenging MR-NIRP-DRV (outdoor driving) dataset, our method significantly outperforms all existing approaches, including those assisted by infrared imaging.
2. **Pure RGB Suffices**: Without requiring any infrared hardware, the model can operate under extreme illumination using only an RGB camera.
3. **Lightweight**: The proposed method is more lightweight than existing noise-decoupling methods (such as Dual-GAN and ND-DeeprPPG), making it better suited for mobile deployment.
4. **Cross-Scenario Consistency**: Demonstrates superior performance across various scenarios, ranging from indoor to outdoor and static to dynamic settings.

## Highlights & Insights

- **Pioneering Work**: This is the first learning-based rPPG model designed for real-world outdoor extreme lighting, filling a critical research gap.
- **Physics-Inspired Design**: The design is naturally grounded in human facial optical skin reflection models and the theory of global interference sharing.
- **Lightweight Strategy**: Avoids heavy modules like GANs and adversarial learning, achieving decoupling solely via similarity computations and minimal convolutions.
- **Incorporation of Biological Priors**: BioSE translates physiological knowledge regarding heart rate ranges into effective signal enhancement strategies.

## Limitations & Future Work

- **Facial Landmark Dependency**: Requires reliable facial landmark detection; performance may degrade under extreme occlusions or large-angle profile views.
- **Video Length Cost**: While expanding the temporal dimension leverages the quasi-periodic nature of BVP, excessively long video segments increase latency.
- **Evaluation of a Single Physiological Metric**: Evaluation is primarily focused on heart rate estimation; its effectiveness on other metrics like blood pressure and blood oxygen has not been fully verified.
- Future directions include expanding to more physiological metrics, multi-person detection, and integration with intelligent cabin systems.

## Related Work & Insights

- **ND-DeeprPPG**: Leverages the consistency of foreground and background noise to achieve decoupling, but depends on external discriminators. Ours replaces this with self-supervised similarity computations.
- **PhysFormer**: A Transformer-based rPPG approach, but not specifically designed for extreme illumination.
- **Insight**: Environmental interference is not pure noise but rather part of a utilizable reference signal. The concept of "干扰即信息" (interference as information) is worth exploring in other signal processing tasks.

## Rating

⭐⭐⭐⭐ — Fills the gap in rPPG under outdoor extreme lighting, achieving results with only an RGB camera that previously required infrared assistance. The background-referenced decoupling design is clean and efficient. Its significant lead on the most challenging datasets demonstrates strong practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] UDC-VIT: A Real-World Video Dataset for Under-Display Cameras](../../ICCV2025/human_understanding/udc-vit_a_real-world_video_dataset_for_under-display_cameras.md)
- [\[CVPR 2026\] PHASE-Net: Physics-Grounded Harmonic Attention System for Efficient Remote Photoplethysmography Measurement](../../CVPR2026/human_understanding/phase-net_physics-grounded_harmonic_attention_system_for_efficient_remote_photop.md)
- [\[CVPR 2025\] Quaffure: Real-Time Quasi-Static Neural Hair Simulation](quaffure_real-time_quasi-static_neural_hair_simulation.md)
- [\[CVPR 2025\] Analyzing the Synthetic-to-Real Domain Gap in 3D Hand Pose Estimation](analyzing_the_synthetic-to-real_domain_gap_in_3d_hand_pose_estimation.md)
- [\[CVPR 2026\] FLOW: Optimal Transport-Driven Feature Warping for Generalized Remote Physiological Measurement](../../CVPR2026/human_understanding/flow_optimal_transport-driven_feature_warping_for_generalized_remote_physiologic.md)

</div>

<!-- RELATED:END -->

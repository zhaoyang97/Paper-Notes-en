---
title: >-
  [Paper Note] Efficient Onboard Spacecraft Pose Estimation with Event Cameras and Neuromorphic Hardware
description: >-
  [CVPR 2026][Human Understanding][event camera] The first end-to-end 6-DoF spacecraft pose estimation system deployed on BrainChip Akida neuromorphic hardware…
tags:
  - "CVPR 2026"
  - "Human Understanding"
  - "event camera"
  - "spacecraft pose estimation"
  - "neuromorphic hardware"
  - "Akida"
  - "SNN"
date: 2026-05-08
content_hash: bf0566309f308b10
---

# Efficient Onboard Spacecraft Pose Estimation with Event Cameras and Neuromorphic Hardware

**Conference**: CVPR 2026
**arXiv**: [2604.04117](https://arxiv.org/abs/2604.04117)  
**Code**: None  
**Area**: Event Camera / Space Perception
**Keywords**: event camera, spacecraft pose estimation, neuromorphic hardware, Akida, SNN

## TL;DR

The first end-to-end 6-DoF spacecraft pose estimation system deployed on BrainChip Akida neuromorphic hardware, exploring accuracy–efficiency trade-offs among event camera representations and quantization-aware training for low-power onboard deployment.

## Background & Motivation

Future on-orbit servicing and active debris removal missions require autonomous rendezvous and proximity operations, in which relative pose estimation is a critical capability. Space imagery poses severe challenges due to extreme illumination, high contrast, and rapid target motion. Event cameras capture brightness changes asynchronously and continue to provide useful information when frame cameras suffer from saturation or motion blur. Spacecraft operate under strict SWaP (Size, Weight, and Power) constraints, and neuromorphic processors offer a favorable performance-per-watt ratio through sparse, event-driven computation.

Despite rapid independent progress in event-based vision and neuromorphic AI, end-to-end spacecraft pose estimation on Akida-class neuromorphic hardware has not been demonstrated. This paper fills that gap.

## Method

### Overall Architecture

A hybrid pose estimation pipeline is adopted: a compact network regresses 2D keypoints from event frame representations, and a PnP solver recovers the final 6-DoF pose. Separate models based on coordinate regression and heatmap regression are designed for Akida V1 and V2, respectively.

### Key Designs

1. **Event Frame Representations**: Three lightweight representations are evaluated — Event-to-Frame (E2F, polarity-encoded maps), 2D Histogram (event count maps), and Locally-Normalised Event Surfaces (LNES, which preserves coarse temporal information). Each involves a different trade-off between temporal fidelity and computational complexity.

2. **Akida V1 Coordinate Regression**: A MobileNet-style backbone (1.88M parameters, ~7.19 MB) with 12 depthwise separable convolution blocks directly regresses 16 scalars representing the $(x, y)$ coordinates of 8 keypoints. High-resolution decoders are avoided to comply with V1-supported operation types. 4-bit quantization-aware training (QAT) is applied for 300 epochs.

3. **Akida V2 Heatmap Regression**: An encoder–decoder architecture (1.72M parameters) produces $56 \times 56 \times 8$ heatmaps. With 8-bit QAT, the model converges in only 60 epochs (compared to 300 for V1) and exhibits notably high robustness to quantization. The SPADES synthetic dataset is used (107K training / 35K validation / 35K test samples), with input images cropped to $224 \times 224$ pixels and an event window of $\Delta t = 50\,\text{ms}$.

### Loss & Training

V1 uses an L2 keypoint coordinate regression loss; V2 uses a heatmap MSE loss. Both follow a floating-point pretraining stage followed by QAT. Training uses the SPADES synthetic dataset (107K training / 35K validation / 35K test samples).

## Key Experimental Results

### Main Results

| Model | Representation | PCK↑ | Pose Error ($E_p$)↓ | Notes |
|---|---|---|---|---|
| V1 Float (E2F) | E2F | 0.94 | 0.036 | Good floating-point accuracy |
| V1 Quantized (E2F) | E2F | 0.33 | 0.101 | Severe degradation under 4-bit |
| V2 Quantized (E2F) | E2F | — | 0.021 | Near-lossless under 8-bit |
| V2 Quantized (LNES) | LNES | — | 0.021 | Best performance on V2 |

### Key Findings

- Heatmap regression is substantially more robust to quantization than coordinate regression.
- Under 4-bit quantization (V1), coordinate regression PCK drops sharply from 93% to ~30%, whereas 8-bit heatmap regression incurs almost no degradation.
- The LNES representation achieves the best results on V2, indicating that temporal information is valuable for higher-capacity architectures.
- Illumination conditions are the most critical factor affecting overall performance.
- V1 coordinate regression floating-point accuracy (E2F): PCK = 0.94, $E_p$ = 0.036; after quantization: PCK = 0.33, $E_p$ = 0.101.
- V2 heatmap regression (E2F): $E_p$ = 0.021; LNES representation achieves the same $E_p$ = 0.021.
- The asynchronous change-detection nature of event cameras continues to provide useful information when frame cameras are saturated or motion-blurred.

## Highlights & Insights

- First end-to-end verification of spacecraft pose estimation on commercial neuromorphic hardware.
- Reveals critical accuracy–efficiency trade-offs: heatmap vs. coordinate regression, 4-bit vs. 8-bit quantization, and different event representations.
- Provides direct reference value for SWaP-constrained onboard AI deployment.

## Limitations & Future Work

- Evaluation is conducted solely on synthetic data; no real event camera data is used.
- The ROI detection module is not included in the pipeline; ground-truth bounding boxes are used for cropping.
- The method is limited to cooperative targets with known 3D models.
- The SPADES dataset comprises 300 unique trajectories covering challenging scenarios including extreme illumination, high contrast, and rapid motion.
- The PnP solver recovers the final 6-DoF pose from predicted 2D keypoints and the known 3D model; failed solutions or results with translation errors exceeding 30 m are discarded.
- BrainChip Akida has an established low Earth orbit flight heritage, supporting its space qualification.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First spacecraft pose estimation demonstrated on Akida hardware.
- **Technical Depth**: ⭐⭐⭐ — Methodology is relatively straightforward, with emphasis on system-level validation.
- **Experimental Thoroughness**: ⭐⭐⭐ — Evaluated on synthetic data only.
- **Practical Value**: ⭐⭐⭐⭐ — Directly relevant to onboard AI deployment on spacecraft.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] E-3DPSM: A State Machine for Event-Based Egocentric 3D Human Pose Estimation](e-3dpsm_a_state_machine_for_event-based_egocentric_3d_human_pose_estimation.md)
- [\[CVPR 2026\] FSMC-Pose: Frequency and Spatial Fusion with Multiscale Self-calibration for Cattle Mounting Pose Estimation](fsmc-pose_frequency_and_spatial_fusion_with_multiscale_selfcalibration_for_cattle.md)
- [\[ICLR 2026\] Event-T2M: Event-level Conditioning for Complex Text-to-Motion Synthesis](../../ICLR2026/human_understanding/event-t2m_event-level_conditioning_for_complex_text-to-motion_synthesis.md)
- [\[CVPR 2026\] RegFormer: Transferable Relational Grounding for Efficient Weakly-Supervised HOI Detection](regformer_transferable_relational_grounding_for_weakly-supervised_hoi_detection.md)
- [\[CVPR 2026\] TriLite: Efficient WSOL with Universal Visual Features and Tri-Region Disentanglement](trilite_efficient_weakly_supervised_object_localization_with_universal_visual_fe.md)

</div>

<!-- RELATED:END -->

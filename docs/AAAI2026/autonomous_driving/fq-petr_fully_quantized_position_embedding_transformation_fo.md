---
title: >-
  [Paper Note] FQ-PETR: Fully Quantized Position Embedding Transformation for Multi-View 3D Object Detection
description: >-
  [AAAI 2026][Autonomous Driving][3D Object Detection] This work presents the first fully INT8-quantized deployment of PETR-series 3D detectors. It introduces three key components: a quantization-friendly LiDAR-ray position encoding (QFPE) to resolve multi-modal feature magnitude mismatch, a dual-lookup table (DULUT) for efficient approximation of nonlinear operators, and quantization after numerical stabilization (QANS) to prevent softmax attention distortion. Across PETR/StreamPETR/PETRv2/MV2D, W8A8 quantization incurs less than 1% mAP loss while reducing latency by 75% (3.9× speedup).
tags:
  - AAAI 2026
  - Autonomous Driving
  - 3D Object Detection
  - Quantization
  - PETR
  - Position Encoding
date: 2026-05-08
content_hash: 3839fa8c652bb703
---

# FQ-PETR: Fully Quantized Position Embedding Transformation for Multi-View 3D Object Detection

**Conference**: AAAI 2026
**arXiv**: [2502.15488](https://arxiv.org/abs/2502.15488)
**Code**: [https://github.com/JiangYongYu1/FQ-PETR](https://github.com/JiangYongYu1/FQ-PETR)
**Area**: Autonomous Driving
**Keywords**: 3D Object Detection, Quantization, PETR, Position Encoding, Autonomous Driving

## TL;DR
This work presents the first fully INT8-quantized deployment of PETR-series 3D detectors. It introduces three key components: a quantization-friendly LiDAR-ray position encoding (QFPE) to resolve multi-modal feature magnitude mismatch, a dual-lookup table (DULUT) for efficient approximation of nonlinear operators, and quantization after numerical stabilization (QANS) to prevent softmax attention distortion. Across PETR/StreamPETR/PETRv2/MV2D, W8A8 quantization incurs less than 1% mAP loss while reducing latency by 75% (3.9× speedup).

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Background**: PETR-series models are among the most prominent transformer-based multi-view 3D detectors, but their deployment on edge devices for autonomous driving is severely constrained by computation and memory bottlenecks. Directly quantizing PETR leads to catastrophic accuracy degradation (up to 20.8% mAP drop) due to two unique challenges: (1) the dynamic range of camera-ray position encodings (±130) is nearly two orders of magnitude larger than that of image features (±4), causing image features to be compressed into only 3–5 valid integer bins after fusion quantization; (2) nonlinear operations such as inverse-sigmoid introduce outliers, and excessively large softmax input ranges cause attention distortion after quantization.

### Core Problem

**Goal**: The paper aims to design quantization-friendly 3D position encodings whose magnitude matches that of image features, while accurately and efficiently quantizing nonlinear operators (SiLU/GELU/Softmax), thereby enabling full-integer inference for PETR without accuracy loss.

## Method

### Overall Architecture
Three synergistic innovations: QFPE redesigns the position encoding to eliminate magnitude mismatch → DULUT approximates nonlinear functions via two cascaded linear LUTs → QANS quantizes softmax inputs only after numerical stabilization. Together, these constitute the first fully INT8 inference framework for PETR.

### Key Designs
1. **Quantization-Friendly LiDAR-ray PE (QFPE)**: Two innovations — (a) LiDAR prior-guided single-point sampling: only one 3D point is sampled per pixel along the depth ray (at a fixed depth of 30 m), eliminating multi-point interpolation and the inverse-sigmoid transform (which amplifies magnitude by 11.5×). (b) Anchor-based constrained embedding: three anchor embeddings are set per axis, and position encodings are generated via convex combination (linear interpolation), strictly bounding the magnitude ($\|\mathbf{e}_\alpha\|_\infty \leq \gamma \approx 0.8$). The resulting QFPE dynamic range is ±29.7 vs. the original ±127.3, a 4.4× reduction. Beyond quantization benefits, FP32 performance also improves (NDS +1.09 on PETR, +0.46 on StreamPETR).

2. **Dual-Lookup Table (DULUT)**: Two cascaded linear LUTs replace a single large LUT or hardware-specific NN-LUT. The first LUT acts as a "nonlinear index mapper" — allocating more indices to high-curvature regions and merging entries in flat regions; the second LUT stores actual values for linear interpolation. Optimal partitioning is obtained via iterative error-driven split/merge optimization. With INT8 and 32+32 entries, DULUT achieves accuracy comparable to a 256-entry single LUT while using 4× fewer entries. It is applicable to SiLU, GELU, and Softmax, and requires only standard LUT units without dedicated hardware.

3. **Quantization After Numerical Stabilization (QANS)**: Quantization is applied to softmax inputs only after numerical stabilization (subtracting the maximum value), constraining the input to the non-positive range $[-\beta, 0]$. The optimal truncation lower bound $\beta$ is adaptively selected from a candidate set by minimizing deviation in the softmax distribution; $\beta = 20$ is found to be lossless. This prevents the attention peak attenuation and positional shift caused by directly quantizing extreme input values.

### Loss & Training
QFPE requires fine-tuning (following the original PETR training configuration: 24 epochs, 4× RTX 4090), whereas DULUT and QANS operate as post-training quantization (PTQ) requiring only 32 calibration images. The overall training cost is substantially lower than QAT.

## Key Experimental Results

| Model | Method | mAP | NDS | FPS | Memory |
|--------|------|------|------|-----|--------|
| PETR(R50) | FP32 | 31.42 | 36.11 | 7.1 | 4.8GB |
| PETR(R50) | SmoothQuant W8A8 | 20.67 | 29.32 | - | - |
| PETR(R50) | QuaRot W8A8 | 22.81 | 30.00 | - | - |
| FQ-PETR(R50) | FP32 | 31.49 | 37.20 | - | - |
| FQ-PETR(R50) | SQ*+QANS W8A8 | **31.34** | **37.17** | **27.6** | **1.3GB** |
| StreamPETR(V2-99) | FP32 | 49.51 | 58.03 | - | - |
| FQ-StreamPETR | Quarot*+QANS W8A8 | **50.12** | **58.39** | - | - |

Direct application of SmoothQuant causes a 10.8% mAP drop; FQ-PETR reduces this to only 0.08. Overall: 3.9× speedup, 75% memory reduction.

### Ablation Study
- Inverse-sigmoid is the root cause of magnitude explosion (11.5× amplification); QFPE eliminates this via single-point sampling and anchor interpolation.
- Three anchor points is optimal; both fewer and more anchors degrade performance.
- DULUT with (32, 32) entries matches the accuracy of a 256-entry linear LUT while using 4× fewer entries.
- QANS truncation with $N \geq 20$ is lossless; $N < 20$ degrades performance due to over-truncation.
- DULUT outperforms the polynomial approximations used in I-BERT/I-ViT (NDS lower by 1–2%).
- QFPE also improves FP32 performance, attributed to better local similarity in position encoding.

## Highlights & Insights
- **First fully INT8 quantization of PETR with negligible accuracy loss** — directly addressing the gap in quantization of transformer-based 3D detectors.
- **QFPE is elegantly designed** — it not only resolves quantization issues but also improves FP32 performance, achieving two goals simultaneously.
- **DULUT is broadly applicable** — suitable for quantizing any nonlinear function (SiLU/GELU/Softmax) and validated effective on LLMs as well.
- **QANS is simple yet effective** — a conceptually clear technique (stabilize first, then quantize) with significant impact (without it, mAP drops by 10+%).
- Deployment results are impressive: 3.9× speedup and 75% memory reduction, meeting the requirements for on-vehicle deployment.

## Limitations & Future Work
- QFPE requires retraining the position encoding module; although the cost is comparable to QAT, it is not purely PTQ.
- Validation is limited to nuScenes; generalization to other datasets such as Waymo or KITTI has not been explored.
- Only INT8 quantization is considered; more aggressive INT4 quantization remains unexplored.
- No comparison is made against quantization of BEV-based methods (e.g., BEVFormer).

## Related Work & Insights
- **vs. SmoothQuant/QuaRot**: These general-purpose quantization methods cause catastrophic degradation when directly applied to PETR (10–20% mAP drop), as they do not address the magnitude mismatch introduced by position encodings.
- **vs. QD-BEV**: QD-BEV employs QAT with distillation for BEV detectors; FQ-PETR with QFPE surpasses QAT (36 epochs) using fewer training epochs (24).
- **vs. I-BERT/I-ViT**: These methods use polynomial approximations for nonlinear functions, yielding lower accuracy than DULUT (NDS lower by 1–2%).

## Related Work & Insights
- DULUT can be directly applied to INT8 quantization of LLMs (preliminarily validated in the paper) and serves as a general-purpose tool.
- The "anchor embedding + convex combination for magnitude bounding" paradigm in QFPE can be generalized to other position encodings that require quantization.
- The QANS principle of "stabilize before quantize" is applicable to any model with softmax (VLMs, LLMs).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Three innovation components each address distinct problems; QFPE and DULUT are genuinely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Four PETR variants, multiple backbones/resolutions, comprehensive ablations, and theoretical analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ — Problem analysis is thorough, theoretical derivations are rigorous, and figures are intuitive.
- Value: ⭐⭐⭐⭐⭐ — Directly addresses the core deployment bottleneck of PETR with high practical significance.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Exploring Surround-View Fisheye Camera 3D Object Detection](exploring_surround-view_fisheye_camera_3d_object_detection.md)
- [\[AAAI 2026\] DriveFlow: Rectified Flow Adaptation for Robust 3D Object Detection in Autonomous Driving](driveflow_rectified_flow_adaptation_for_robust_3d_object_detection_in_autonomous.md)
- [\[ICCV 2025\] EVT: Efficient View Transformation for Multi-Modal 3D Object Detection](../../ICCV2025/autonomous_driving/evt_efficient_view_transformation_for_multi-modal_3d_object_detection.md)
- [\[AAAI 2026\] MOBA: A Material-Oriented Backdoor Attack against LiDAR-based 3D Object Detection](moba_a_material-oriented_backdoor_attack_against_lidar-based_3d_object_detection.md)
- [\[AAAI 2026\] Invisible Triggers, Visible Threats! Road-Style Adversarial Creation Attack for Visual 3D Detection in Autonomous Driving](invisible_triggers_visible_threats_road-style_adversarial_creation_attack_for_vi.md)

<!-- RELATED:END -->

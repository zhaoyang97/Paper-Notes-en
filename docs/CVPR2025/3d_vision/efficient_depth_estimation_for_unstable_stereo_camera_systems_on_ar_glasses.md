---
title: >-
  [Paper Note] Efficient Depth Estimation for Unstable Stereo Camera Systems on AR Glasses
description: >-
  [CVPR 2025][3D Vision][Stereo Depth Estimation] Proposed two models, MultiHeadDepth and HomoDepth. They optimize the latency bottlenecks of the cost volume and preprocessing in stereo depth estimation using a hardware-friendly multi-head cost volume (approximating cosine similarity via LayerNorm + dot product, combined with group-wise pointwise convolutions) and a homography estimation network with 2D Rectified Positional Encoding (RPE), respectively. In AR glasses scenarios…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Stereo Depth Estimation"
  - "AR Glasses"
  - "Cost Volume Optimization"
  - "Homography Estimation"
  - "Rectified Positional Encoding"
date: 2026-05-08
content_hash: 28630f6a443729d1
---

# Efficient Depth Estimation for Unstable Stereo Camera Systems on AR Glasses

**Conference**: CVPR 2025  
**arXiv**: [2411.10013](https://arxiv.org/abs/2411.10013)  
**Code**: [https://github.com/UCI-ISA-Lab/MultiHeadDepth-HomoDepth](https://github.com/UCI-ISA-Lab/MultiHeadDepth-HomoDepth)  
**Area**: 3D Vision  
**Keywords**: Stereo Depth Estimation, AR Glasses, Cost Volume Optimization, Homography Estimation, Rectified Positional Encoding

## TL;DR

Proposed two models, MultiHeadDepth and HomoDepth. They optimize the latency bottlenecks of the cost volume and preprocessing in stereo depth estimation using a hardware-friendly multi-head cost volume (approximating cosine similarity via LayerNorm + dot product, combined with group-wise pointwise convolutions) and a homography estimation network with 2D Rectified Positional Encoding (RPE), respectively. In AR glasses scenarios, this improves accuracy by 11.8-30.3% while reducing end-to-end latency by 44.5%.

## Background & Motivation

**Background**: Depth estimation is a foundational component in AR/VR, with downstream applications including novel view synthesis, occlusion reasoning, and AR object placement. Stereo depth estimation is widely adopted due to its accuracy advantages and natural compatibility with binocular cameras on AR glasses. However, the wearable form factor of AR glasses imposes strict computing resource constraints, requiring on-device inference to be completed within 100ms.

**Limitations of Prior Work**: In the latency of current SOTA models (such as Meta's Argos), preprocessing (calibration + rectification) accounts for 30.2%, and cost volume computation accounts for 29.3%, meaning non-model or preprocessing phases occupy nearly 60% of the total latency. More severely, the flexible material of AR glasses causes significant camera bending ($>10^\circ$), leading to continuous changes in extrinsic parameters. This results in an online rectification failure rate of up to 15-23%, and solving the extrinsics itself can take 200-2000ms.

**Key Challenge**: Cosine similarity computation in the cost volume involves pixel-wise norm calculations and division, which are extremely unfriendly to GPU/NPU hardware that is highly optimized for matrix multiplication. Meanwhile, rectification preprocessing relies on precise camera extrinsics, which are unstable on AR glasses.

**Goal**: How to significantly reduce the latency of cost volume and preprocessing in depth estimation while maintaining or improving accuracy?

**Key Insight**: (1) For the cost volume, approximate cosine similarity using LayerNorm + dot product, and combine this with group-wise pointwise convolutions for hardware acceleration. (2) For preprocessing, introduce a homography estimation head and rectified positional encoding, allowing the model to directly accept unrectified images as input.

**Core Idea**: Replace non-ML bottlenecks in depth estimation (cosine similarity computation and image rectification) with hardware-friendly ML operations (LayerNorm + multi-head dot product, and homography estimation + positional encoding, respectively), enabling the entire pipeline to run efficiently on ML accelerators.

## Method

### Overall Architecture

The paper proposes two progressive models: (1) MultiHeadDepth: based on Argos, it replaces the cost volume with a multi-head cost volume while keeping the encoder-decoder structure intact, focusing on optimizing cost volume latency; (2) HomoDepth: built on top of MultiHeadDepth, it adds a homography estimation head and 2D RPE to eliminate the need for preprocessing. HomoDepth adopts multi-task learning, sharing the encoder to simultaneously output the depth map and homography matrix, and dynamically balances the losses of the two tasks during training using homoscedastic uncertainty.

### Key Designs

1. **LayerNorm + Dot Product Approximation (LND) instead of Cosine Similarity**:

    - **Function**: Replace the hardware-unfriendly pixel-wise cosine similarity in the cost volume with an efficient approximation.
    - **Mechanism**: In cosine similarity $D_{cos}(a,b) = a \cdot b / |a||b|$, calculating the norm in the denominator is the bottleneck. The authors replace pixel-wise norm normalization with 2D LayerNorm, directly applying the dot product after normalizing vectors to a standard distribution. LayerNorm compresses a massive amount of pixel-wise norm computations into a few channel-level statistical calculations, significantly reducing the computational overhead of normalization. Additionally, a weight layer is appended after LND to adaptively refine the approximation accuracy.
    - **Design Motivation**: LayerNorm is highly optimized in hardware and compilers, and its normalized buffer can easily integrate other encoding information (such as positional encoding), providing a natural interface for the subsequent introduction of RPE.

2. **Multi-head Cost Volume**:

    - **Function**: Further reduce the computational complexity of the cost volume and provide richer matching awareness.
    - **Mechanism**: Borrowing ideas from multi-head attention and dot-product scaling, the input channels are split into multiple heads (e.g., $C/\text{heads}$ channels per head). Each head independently performs a group-wise dot product, and the results of various heads are then aggregated via a 1×1 pointwise convolution. The entire operation is equivalent to group-pointwise convolution, which is a standard operator highly optimized by hardware.
    - **Design Motivation**: The multi-head design not only reduces the dimensionality of a single dot product (thereby reducing computation) but also provides multiple perspectives to perceive the matching relationship between left and right feature maps, similar to how multi-head attention captures information in different subspaces.

3. **Homography Estimation Head + 2D Rectified Positional Encoding (RPE)**:

    - **Function**: Allow the model to directly accept unrectified stereo image inputs, eliminating the need for rectification preprocessing.
    - **Mechanism**: Based on the 3D projection relationship $q_r = \frac{d_l}{d_r} H_{l \to r} q_l$, when objects are far from the camera, $d_l / d_r \approx 1$, meaning the homography matrix can approximate the positional relationship between stereo images. A CNN head sharing the encoder with the depth estimation branch is designed to estimate the homography matrix $H$. Then, $H$ is transformed into 2D RPE: standard 2D sinusoidal positional encoding $PE(q_l)$ is applied to left-image pixels $q_l$, and $RPE(q_r) = PE(H_{l \to r} q_l)$ is applied to right-image pixels. This ensures that the same world point obtains similar positional encoding values in both left and right images, which are integrated into the LayerNorm-normalized features to assist in cost volume calculation.
    - **Design Motivation**: Traditional rectification requires physically warping the images (which loses edge information), whereas RPE embeds spatial relationships into features as encodings, incurring no info loss and avoiding the computational overhead of image warping. The multi-task design with a shared encoder prevents additional latency overhead.

### Loss & Training

- **Depth Estimation Loss**: $L_D(y, \hat{y}) = SL_1(y, \hat{y}) + \sum_{l=0}^{4} SL_1(\nabla y^l, \nabla \hat{y}^l)$, SmoothL1 + multi-scale gradient loss.
- **Homography Estimation Loss**: $L_H = \|weight_w(y) - weight_w(\hat{y})\|_F$, weighted Frobenius norm, where the weight matrix uses $w=50$ to amplify small-value elements on the diagonal and anti-diagonal positions.
- **Multi-Task Joint Loss**: $L = \frac{L_H}{2\sigma_H^2} + \frac{L_D}{2\sigma_D^2} + \log \sigma_H \sigma_D$, where $\sigma_H, \sigma_D$ are trainable homoscedastic uncertainty parameters.

## Key Experimental Results

### Main Results (Unpreprocessed Inputs)

| Model | SceneFlow AbsRel↓ | ADT AbsRel↓ | DTU AbsRel↓ | Latency (ms) |
|:--|:--|:--|:--|:--|
| MobileStereoNet-2D | 0.172 | 0.199 | 0.147 | - |
| Argos (CVPR2023) | 0.102 | 0.133 | 0.122 | 748.5 |
| Selective-Stereo (CVPR2024) | 0.053 | 0.082 | 0.128 | - |
| **MultiHeadDepth** | **0.091** | **0.094** | **0.101** | **598.9** |

### Performance of HomoDepth on Unrectified Inputs

| Method | DTU AbsRel↓ | End-to-End Latency |
|:--|:--|:--|
| Argos (w/o Preprocessing) | 0.122 | Baseline |
| Preprocessing + Argos | 0.109 | 100% Baseline |
| Preprocessing + MultiHeadDepth | 0.101 | ~75% |
| **HomoDepth (w/o Preprocessing)** | **0.106** | **~55.5%** |

### Quantized Model Results (INT8)

| Model | SceneFlow AbsRel↓ | ADT AbsRel↓ | CPU Latency (ms) | GPU Latency (ms) |
|:--|:--|:--|:--|:--|
| Argos | 0.109 | 0.146 | 748.5 | 54.4 |
| **MultiHeadDepth** | **0.098** | **0.097** | **598.9** | **45.3** |

### Key Findings

- **Cost Volume Optimization is Universally Effective**: MultiHeadDepth achieves superior or comparable accuracy to Argos across all datasets, while reducing latency by 22.9-25.2%.
- **Eliminating Preprocessing Brings Massive Latency Benefits**: HomoDepth reduces end-to-end latency by 44.5%, with its accuracy on unrectified inputs only slightly lower than "Preprocessing + MultiHeadDepth".
- **Multi-Task Learning Further Boosts Robustness**: Introducing RPE reduces AbsRel by 10.0-24.3% on unaligned stereo inputs.
- **Quantization Friendly**: Accuracy shows almost no degradation after INT8 quantization, with GPU latency at only 45.3ms.

## Highlights & Insights

1. **Engineering-oriented System Optimization**: Instead of merely chasing the highest accuracy, it systematically analyzes latency bottlenecks (preprocessing 30% + cost volume 30%) and addresses them individually in a targeted manner.
2. **Replacing Non-ML with ML Operations**: Replaces traditional algorithms (cosine similarity, image rectification) with hardware-friendly ML operations (LayerNorm, CNN), enabling the entire workflow to run efficiently on GPUs/NPUs.
3. **Zero Information Loss Advantage of RPE**: Traditional rectification crops image boundaries, whereas RPE passes spatial/positional information in the form of encodings, completely avoiding this issue.
4. **Validation on Real AR Hardware**: Validated on the ADT dataset captured by Meta Aria glasses, demonstrating strong practical feasibility for industrial deployment.

## Limitations & Future Work

1. **Limitations of Homography Approximation**: The $d_l / d_r \approx 1$ assumption fails for close-range objects, potentially affecting the accuracy of near-range depth.
2. **Testing Limited to Static Baseline Datasets**: Although ADT originates from AR glasses, it lacks testing on extreme scenarios with dynamic bending.
3. **Resolution Constraints**: The experimental resolutions are relatively low; the memory overhead of the multi-head cost volume at high resolutions needs to be validated.
4. **Lack of Comparison with Transformer-based Methods**: Emerging methods like RAFT-Stereo are not included in the comparisons.

## Related Work & Insights

- **Argos (CVPR 2023)**: The baseline model for this work, a SOTA AR depth estimation model developed by Meta. This work builds upon it to perform acceleration optimizations.
- **Multi-head Attention (Transformer)**: The grouped computation idea from multi-head attention is transferred to the cost volume, with group-pointwise conv serving as its engineering implementation.
- **MVSNet**: Uses homography for multi-view stereo matching under known camera parameters. This work extends it to scenarios with unknown extrinsic parameters.
- **Insight**: In edge device deployment, "what operations are hardware-friendly" should serve as the first principle of method design rather than a post-hoc optimization.

## Rating

⭐⭐⭐⭐ — Outstanding engineering value. The systematic latency analysis combined with targeted optimization provides an admirable blueprint. The multi-task homography estimation pipeline to eliminate preprocessing is novel and practical. However, theoretical innovation is somewhat limited, as it mostly aggregates existing techniques (LayerNorm, multi-head attention, homography) in a clever manner.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera](depth_any_camera_zero-shot_metric_depth_estimation_from_any_camera.md)
- [\[CVPR 2025\] Helvipad: A Real-World Dataset for Omnidirectional Stereo Depth Estimation](helvipad_a_real-world_dataset_for_omnidirectional_stereo_depth_estimation.md)
- [\[CVPR 2025\] DEFOM-Stereo: Depth Foundation Model Based Stereo Matching](defom-stereo_depth_foundation_model_based_stereo_matching.md)
- [\[CVPR 2025\] UniK3D: Universal Camera Monocular 3D Estimation](unik3d_universal_camera_monocular_3d_estimation.md)
- [\[CVPR 2025\] Scalable Autoregressive Monocular Depth Estimation](scalable_autoregressive_monocular_depth_estimation.md)

</div>

<!-- RELATED:END -->

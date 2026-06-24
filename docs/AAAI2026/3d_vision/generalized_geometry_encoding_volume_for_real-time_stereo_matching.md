---
title: >-
  [Paper Note] Generalized Geometry Encoding Volume for Real-time Stereo Matching
description: >-
  [AAAI 2026][3D Vision][Stereo Matching] This paper proposes GGEV, which integrates depth priors from a monocular depth foundation model (Depth Anything V2) into the cost aggregation process in a lightweight manner. It adaptively enhances matching relationships for different disparity hypotheses through Depth-Aware Dynamic Cost Aggregation (DDCA), achieving strong generalization capabilities at real-time speeds.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Stereo Matching"
  - "Real-time Inference"
  - "Zero-shot Generalization"
  - "Monocular Depth Foundation Model"
  - "Dynamic Cost Aggregation"
date: 2026-05-08
content_hash: f8269a4786060466
---

# Generalized Geometry Encoding Volume for Real-time Stereo Matching

**Conference**: AAAI 2026  
**arXiv**: [2512.06793](https://arxiv.org/abs/2512.06793)  
**Code**: [https://github.com/JiaxinLiu-A/GGEV](https://github.com/JiaxinLiu-A/GGEV)  
**Area**: 3D Vision / Stereo Matching  
**Keywords**: Stereo Matching, Real-time Inference, Zero-shot Generalization, Monocular Depth Foundation Model, Dynamic Cost Aggregation

## TL;DR

This paper proposes GGEV, which integrates depth priors from a monocular depth foundation model (Depth Anything V2) into the cost aggregation process in a lightweight manner. It adaptively enhances matching relationships for different disparity hypotheses through Depth-Aware Dynamic Cost Aggregation (DDCA), achieving strong generalization capabilities at real-time speeds.

## Background & Motivation

### Background

Stereo matching is a classic task in computer vision, requiring dense disparity maps to be estimated from rectified left and right images. Practical applications (autonomous driving, 3D reconstruction) impose strict requirements on both generalization ability and inference latency. Current approaches fall into two main camps:

- **Real-time methods** (RT-IGEV, Fast-ACVNet, etc.): Fast inference is achieved through strategies like downsampling cost volumes, lightweight aggregation, and replacing 3D convolutions with 2D convolutions. However, matching relationships remain fragile in unseen scenes (occlusions, textureless regions, repetitive textures, thin structures).
- **Generalization-based methods** (FoundationStereo, MonSter, etc.): These leverage monocular foundation models (MFMs) to improve generalization, but rely on large backbones (ViT-L) and complex iterative mechanisms to address the scale-shift problem, resulting in high inference latency.

### Key Challenge

How to design a stereo matching network that is both real-time and possesses strong generalization capabilities?

### Limitations of Prior Work

The authors analyze two key limitations of the current Geometry Encoding Volume (GEV):

**Significant differences in critical regions corresponding to different disparity hypotheses**: Uniformly processing all disparity hypotheses leads to false matches.

**The matching relationships in these regions are extremely fragile in unseen scenes**: Textureless areas, occlusions, and repetitive textures lead to matching failures.

### Key Insight

Unlike FoundationStereo which uses MFM to construct the cost volume (introducing scale-shift issues), the proposed method uses depth features to **guide cost aggregation**—avoiding the scale-shift issue while remaining lightweight.

## Method

### Overall Architecture

GGEV consists of four stages:
1. **Multi-cue Feature Extraction**: Texture features (MobileNetV2) + Depth features (frozen Depth Anything V2 Small)
2. **Cost Volume Construction**: Group-wise correlation cost volume based on texture features
3. **Depth-Aware Dynamic Cost Aggregation (DDCA)**: Adaptively enhancing the cost volume using depth priors
4. **Depth-Aware Iterative Refinement**: GRU-based iterative refinement of disparity maps

### Key Designs

#### 1. Multi-cue Feature Extraction and Selective Channel Fusion (SCF)

**Function**: Extract texture and depth features and fuse them in a lightweight manner to form depth-aware prior features.

**Mechanism**:
- **Texture Branch**: Uses ImageNet-pretrained MobileNetV2 to extract multi-scale texture features $\mathbf{f}_{l,i}, \mathbf{f}_{r,i}$ ($i \in \{4,8,16\}$) from the left and right images.
- **Depth Branch**: Uses a **frozen** Depth Anything V2 Small to extract multi-scale depth features $\mathbf{f}_{d,i}$ only from the left image.
- **SCF Module**: Fuses concatenated texture and depth features using 1×1 convolutions to generate depth-aware prior features $\mathbf{f}_{da,i}$.

**Design Motivation**:
- Using a frozen MFM avoids training overhead and leverages domain-invariant structural priors learned from large-scale real-world data.
- Using MobileNetV2 rather than ViT as the texture backbone maintains real-time capabilities.
- 1×1 convolution fusion avoids spatial blurring and preserves structural details.

#### 2. Depth-Aware Dynamic Cost Aggregation (DDCA)

**Function**: Adaptively inject depth structure priors into each disparity hypothesis in the cost volume, enhancing fragile matching relationships.

**Mechanism**:

**Step 1 - Disparity-level Depth Structure Representation**: Calculate the affinity matrix between each disparity hypothesis and depth features.

$$\mathbf{Q} = \text{Re}(W_q \mathbf{C}_d), \quad \mathbf{K} = \text{Re}(W_k \text{Pool}(\mathbf{f}_{da}))$$
$$\mathbf{A} = \mathbf{Q}^T \mathbf{K}$$

Similar to multi-head self-attention, $\mathbf{A}^g$ is computed by dividing into G groups along the channel dimension.

**Step 2 - Disparity-level Adaptive Cost Aggregation**: Generate dynamic convolution kernels using the affinity matrix.

$$\mathbf{M}^g = \text{softmax}(\mathbf{A}^g W_m)$$
$$\mathbf{C}_d' = \mathbf{C}_d * \mathbf{M}^g_{\text{dynamic}}(\mathbf{C}_d, \mathbf{f}_{da})$$

**Design Motivation**:
- Different disparity hypotheses correspond to different foreground/background regions, requiring **different aggregation strategies**.
- Traditional hourglass aggregation networks process all disparity hypotheses uniformly, failing to treat them distinctively.
- Dynamic convolution kernels allow each pixel to have targeted filtering weights on each disparity plane.
- Combines large and small convolution kernels to capture complementary low-frequency and high-frequency information.
- Key: Utilizing a sliding window (like standard 2D convolution) keeps the module lightweight and real-time.

#### 3. Depth-Aware Iterative Refinement

**Function**: Retrospectively refine disparity maps iteratively using GRU, injecting the depth prior into the initial hidden state.

**Mechanism**:
- The initial disparity $\mathbf{d}_0$ is regressed from GGEV via soft-argmin.
- The GRU hidden state $h_0$ is initialized (injecting structural priors) with depth-aware features $\mathbf{f}_{da,4}$.
- In each iteration: Index geometric features from GGEV → Concatenate with current disparity → GRU update → Decode residual disparity.
- During upsampling, GRU features are concatenated with depth features to generate a weight map.

### Loss & Training

$$\mathcal{L} = |\mathbf{d}_0 - \mathbf{d}_{gt}|_{smooth} + \sum_{i=1}^{N} \gamma^{N-i} \|\mathbf{d}_i - \mathbf{d}_{gt}\|_1$$

- Smooth L1 loss is used for the initial disparity, and L1 loss for the iterative disparities.
- $\gamma = 0.9$ is the decay factor, with 11 iterations during training and 8 during inference.
- AdamW optimizer, gradient clipping in [-1,1], one-cycle learning rate scheduler.

## Key Experimental Results

### Main Results

#### Zero-shot Generalization (Training on Scene Flow only)

| Method | Type | KITTI 2012 | KITTI 2015 | Middlebury | ETH3D |
|------|------|-----------|-----------|------------|-------|
| RT-IGEV | Real-time | 5.8 | 6.6 | 7.8 | 5.8 |
| Fast-ACVNet | Real-time | 12.4 | 10.6 | 13.5 | 7.9 |
| RAFT-Stereo | Accuracy | 4.5 | 5.7 | 9.3 | 3.2 |
| DEFOM-Stereo(ViT-S) | Accuracy | 4.2 | 5.3 | 6.3 | 2.6 |
| **GGEV (Ours)** | **Real-time** | **4.1** | **5.5** | **6.5** | **2.8** |

vs RT-IGEV error rate reduction: KITTI 2012 ↓29%, KITTI 2015 ↓16%, ETH3D ↓51%

#### Benchmark Accuracy (After Fine-tuning)

| Method | KITTI 2012 3-noc | KITTI 2015 D1-all | ETH3D Bad 1.0 | Inference Time (ms) |
|------|-----------------|-------------------|---------------|-------------|
| RT-IGEV | 1.29 | 1.79 | - | 40 |
| BANet-3D | 1.27 | 1.77 | - | 30 |
| **GGEV** | **1.10** | **1.70** | **1.19** | **47** |

### Ablation Study

| Configuration | Scene Flow EPE | KITTI 2015 D1 | ETH3D Bad 1.0 | Params (M) | Inference (ms) |
|------|---------------|---------------|---------------|----------|---------|
| Baseline | 0.54 | 8.01 | 3.60 | - | 30 |
| +DFE(ViT-S) | 0.52 | 6.32 | 3.57 | - | 37 |
| +DFE+SCF | 0.49 | 7.58 | 3.65 | - | 38 |
| +DCA only | 0.47 | 6.75 | 3.63 | - | 39 |
| Full(ViT-S) | **0.46** | **5.56** | **2.84** | 3.68 | 47 |

#### Evaluation in Reflective Regions (KITTI 2012 Reflective)

| Method | 2-noc | 3-noc |
|------|-------|-------|
| RAFT-Stereo | 8.41 | 5.40 |
| RT-IGEV | 9.56 | 5.76 |
| **GGEV** | **7.33** | **4.04** |

### Key Findings

1. **Adding the Depth Feature Encoder (DFE) alone improves generalization but offers limited in-domain performance**: The generalization ability of depth prior is strong, but adaptive fusion is required to fully exploit it.
2. **SCF improves in-domain fitting but yields mixed generalization results**: Simple fusion is insufficient for addressing complex scenes.
3. **DCA alone improves in-domain performance but has limited generalization**: Texture features remain sensitive to textureless regions and appearance changes.
4. **The synergy of all three components simultaneously improves accuracy and generalization**: SCF introduces MFM generalization ability while DDCA adaptively fuses them, leading to overall improvements.
5. **Performance is particularly outstanding in reflective/difficult areas**: Reduces errors by more than 30% compared to all other methods.

## Highlights & Insights

1. **Clever avoidance of scale-shift issues**: Avoids using MFM for disparity initialization generation (which causes scale-shift); instead, it guides cost aggregation, fundamentally bypassing alignment problems.
2. **Novel "Depth Features as Dynamic Convolution Kernels" design**: Affinity matrix → Dynamic convolution kernels → Adaptive aggregation, lightweight yet effective.
3. **Combining frozen MFM with trainable fusion**: Adapts to the task while retaining pretrained knowledge.
4. **Successful balance of real-time and robustness**: An inference time of 47ms (~21fps) fully meets real-time demands, while its generalization performance is competitive with heavy accuracy-focused models.
5. **Intuitive DDCA visualization**: Contrasting initial cost volumes with post-DDCA cost volumes clearly shows the removal of false matches.

## Limitations & Future Work

1. **ViT-L version loses real-time viability with 110ms inference**: Larger backbones offer better accuracy but sacrifice speed.
2. **Inability to handle extremely large disparity ranges during training**: Limited by memory consumption in cost volume construction.
3. **Lack of evaluation under extreme weather and night scenes**: Performance evaluation is principally on standard benchmarks.
4. The DDCA module could be investigated for application to other cost-aggregation-dependent tasks (e.g., optical flow estimation).

## Related Work & Insights

- **RT-IGEV** (Xu et al., 2025): Direct baseline and currently the strongest real-time method.
- **Depth Anything V2** (Yang et al., 2024): Provides the frozen depth feature encoder.
- **OverLoCK** (Lou & Yu, 2025): Inspired the dynamic convolution design of DDCA.
- **FoundationStereo** (Wen et al., 2025): Generalization-oriented approach employing larger models but not in real-time.
- **MonSter** (Cheng et al., 2025): Utilizes a dual-branch architecture for handling scale-shift, but exhibits slow inference.

## Rating

- Novelty: ⭐⭐⭐⭐ — Dynamic design of DDCA module is clever, and avoiding scale-shift is highly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covered 5 benchmarks, zero-shot + fine-tuning, exhaustive ablations, and reflective region profiling.
- Writing Quality: ⭐⭐⭐⭐⭐ — Excellent motivation analysis and highly persuasive visualizations.
- Value: ⭐⭐⭐⭐⭐ — Highly sought-after real-time + generalization trade-off, elegantly solved.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Fast-FoundationStereo: Real-Time Zero-Shot Stereo Matching](../../CVPR2026/3d_vision/fast-foundationstereo_real-time_zero-shot_stereo_matching.md)
- [\[AAAI 2026\] Domain Generalized Stereo Matching with Uncertainty-guided Data Augmentation](domain_generalized_stereo_matching_with_uncertainty-guided_data_augmentation.md)
- [\[CVPR 2026\] Lite Any Stereo: Efficient Zero-Shot Stereo Matching](../../CVPR2026/3d_vision/lite_any_stereo_efficient_zero-shot_stereo_matching.md)
- [\[AAAI 2026\] RTGaze: Real-Time 3D-Aware Gaze Redirection from a Single Image](rtgaze_real-time_3d-aware_gaze_redirection_from_a_single_image.md)
- [\[AAAI 2026\] Cheating Stereo Matching in Full-Scale: Physical Adversarial Attack against Binocular Depth Estimation](cheating_stereo_matching_in_full-scale_physical_adversarial_attack_against_binoc.md)

</div>

<!-- RELATED:END -->

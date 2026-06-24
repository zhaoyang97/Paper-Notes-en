---
title: >-
  [Paper Note] G2HFNet: GeoGran-Aware Hierarchical Feature Fusion Network for Salient Object Detection in Optical Remote Sensing Images
description: >-
  [CVPR 2025][Segmentation][Salient Object Detection] This paper proposes G2HFNet, which designs differentiated optimization strategies for features at different levels through four modules: Multi-scale Detail Enhancement (MDE), Dual-branch Geometric-Granularity Complementarity (DGC), Deep Semantic Perception (DSP), and Local-Global Guided Fusion (LGF), comprehensively outperforming SOTA on three remote sensing salient object detection datasets.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Salient Object Detection"
  - "Optical Remote Sensing Images"
  - "Multi-scale Detail Enhancement"
  - "Dual-branch Geometric-Granularity Complementarity"
  - "Hierarchical Feature Fusion"
date: 2026-05-08
content_hash: 5384ab61bdfa12f0
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# G2HFNet: GeoGran-Aware Hierarchical Feature Fusion Network for Salient Object Detection in Optical Remote Sensing Images

**Conference**: CVPR 2025  
**arXiv**: [2603.12680](https://arxiv.org/abs/2603.12680)  
**Code**: To be confirmed  
**Area**: Image Segmentation / Remote Sensing / Salient Object Detection  
**Keywords**: Salient Object Detection, Optical Remote Sensing Images, Multi-scale Detail Enhancement, Dual-branch Geometric-Granularity Complementarity, Hierarchical Feature Fusion

## TL;DR
This paper proposes G2HFNet, which designs differentiated optimization strategies for features at different levels through four modules: Multi-scale Detail Enhancement (MDE), Dual-branch Geometric-Granularity Complementarity (DGC), Deep Semantic Perception (DSP), and Local-Global Guided Fusion (LGF), comprehensively outperforming SOTA on three remote sensing salient object detection datasets.

## Background & Motivation

### 1. Background
Salient object detection (SOD) aims to simulate the human ability to quickly identify key objects in images, which is a pixel-level binary classification task. With the growing demand for high-precision remote sensing image analysis, SOD has expanded from natural scenes to optical remote sensing images (ORSI).

### 2. Limitations of Prior Work
- **Single-scale feature extraction issue**: Existing methods usually extract multi-level features using a unified attention mechanism at a single scale, which cannot effectively handle the drastic scale variations prominent in remote sensing images.
- **Unified optimization strategy issue**: The multi-level features obtained by the encoder contain different types of information (high-level features focus on location, while low-level features focus on details), but most methods adopt the same optimization strategy for all levels.
- **Insufficient utilization of mid-level features**: Although some methods design targeted modules, they overlook the fact that mid-level features contain both details and location information.

### 3. Key Challenge
Remote sensing images are captured from an overhead perspective, where object scales vary dramatically (from extremely small objects to large lakes), and backgrounds are complex with low contrast. Direct transfer of natural image methods leads to a significant decrease in performance.

### 4. Mechanism
Design differentiated optimization modules for features at different levels: use MDE to enhance multi-scale details for low-level features, DGC to jointly capture details and location information for mid-level features, DSP to optimize location cues for high-level features, and finally LGF for hierarchical feature fusion.

### 5. Prior Attempts & Limitations
- Zhou et al. address scale variation by compressing image scales, but this transitions inevitably lead to information loss.
- Li et al. design three dedicated modules to extract different features, but they neglect details and location information in mid-level features.

### 6. Solution Overview
This paper proposes G2HFNet, using Swin Transformer as the backbone and integrating four key modules (MDE, DGC, DSP, and LGF) to comprehensively mine geometric and granular cues in remote sensing images.

## Method

### Overall Architecture
The input image ($4 \times 3 \times 384 \times 384$) is processed by Swin Transformer to extract five-level features $\{F_i\}$. Low-level features ($F_1, F_2$) are fed into MDE for detail enhancement, mid-level features ($F_3, F_4$) are fed into DGC to complement geometric and granularity information, the high-level feature ($F_5$) is fed into DSP to optimize location cues, and finally, they are fused layer-by-layer through LGF to generate detection results.

### Key Designs

#### Key Design 1: Multi-scale Detail Enhancement Module (MDE)
- **Function**: Processes low-level features to capture fine-grained details under different scales.
- **Mechanism**: Uses four simplified U-Net branches (with different convolutional kernel sizes $2i-1$) to perform explicit scale transitions through downsampling-upsampling, and then optimizes features using pyramid spatial attention and pyramid channel attention blocks.
- **Design Motivation**: ASPP-like structures operating directly on single-scale features cannot effectively learn the details of targets at different scales; the U-Net structure exposes features to explicit scale transitions, which can capture richer cross-scale details.
- **Novelty**: Pyramid spatial attention replaces average pooling with pixel unshuffle (factors 1/2/4/6) for multi-scale sampling to avoid information loss; pyramid channel attention maps the channel dimension into a spatial form through dimension transformation before performing multi-scale operations.

#### Key Design 2: Dual-branch Geometric-Granularity Complementarity Module (DGC)
- **Function**: Processes mid-level features to extract both details and location information.
- **Mechanism**: The granularity branch progressively extracts multi-scale details using cascaded convolutional layers with different kernel sizes; the geometric branch performs multi-scale feature sampling using pixel unshuffle and then enhances location cues via self-attention; the two branches are fused through a geometric-granularity interaction block.
- **Design Motivation**: Mid-level features incorporate both details and location information, which a single cross-attention mechanism cannot fully exploit; the cascaded design of the granularity branch allows information from small receptive fields to gradually flow into branches with larger receptive fields, enhancing detail extraction.
- **Interaction Block Design**: Concatenates features from both branches, passes them through a 1×1 convolution + sigmoid to generate a weight map $W$, and then weights both branches individually using $W$ to achieve mutual enhancement.

#### Key Design 3: Deep Semantic Perception Module (DSP)
- **Function**: Optimizes location cues in high-level features.
- **Mechanism**: Directly applies a self-attention mechanism ($Q/K/V$ projection + matrix multiplication) to the fifth-level feature (64×12×12) to model long-range space dependencies.
- **Design Motivation**: High-level features already contain reliable location cues, and since targets in remote sensing images can appear at any location, self-attention can capture global spatial relationships.

#### Key Design 4: Local-Global Guided Fusion Module (LGF)
- **Function**: Replaces traditional 3×3 convolutions to implement multi-level feature fusion.
- **Mechanism**: Divided into two stages: local guidance (gated convolution to enhance detail structure) and global guidance (high-level features guiding low-level features to focus on target regions).
- **Design Motivation**: Traditional U-Net decoders only use a single convolutional layer to transfer information, which lacks sufficient guidance.

### Loss & Training
Total Loss = BCE Loss + Boundary IoU Loss + F-measure Loss, where joint deep supervision is applied to five saliency predictions.

## Key Experimental Results

### Main Results: Comparison with 18 SOTA Methods (Table I)

| Dataset | Metric | Ours | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| EORSSD | M↓ | **0.0041** | 0.0051 (HFCNet) | -19.6% |
| EORSSD | Fβ↑ | **0.8808** | 0.8092 (CorrNet) | +8.9% |
| EORSSD | Eξ↑ | **0.9807** | 0.9533 (CorrNet) | +2.9% |
| ORSSD | M↓ | **0.0056** | 0.0073 (HFCNet) | -23.3% |
| ORSSD | Fβ↑ | **0.9147** | 0.8808 (MCCNet) | +3.8% |
| ORSSD | Eξ↑ | **0.9868** | 0.9741 (MCCNet) | +1.3% |
| ORSI-4199 | M↓ | **0.0242** | 0.0270 (HFCNet) | -10.4% |
| ORSI-4199 | Fβ↑ | **0.8862** | 0.8550 (MCCNet) | +3.6% |
| ORSI-4199 | Eξ↑ | **0.9557** | 0.9432 (ICON) | +1.3% |

### Ablation Study (Table IV-IX)

| Ablation Setting | M↓ | Fβ↑ | Eξ↑ |
|----------|-----|------|------|
| w/o MDE | 0.0059 | 0.8724 | 0.9708 |
| w/o DGC | 0.0054 | 0.8734 | 0.9782 |
| w/o DSP | 0.0047 | 0.8682 | 0.9763 |
| w/o LGF | 0.0051 | 0.8554 | 0.9701 |
| **Full Model** | **0.0041** | **0.8808** | **0.9807** |

### Key Findings
1. **All four modules are effective**: Removing any module leads to performance degradation, with MDE having the greatest impact on the $M$ metric (+39.0%) and LGF having the greatest impact on $F_\beta$.
2. **Both U-Net structure and pyramid attention in MDE are indispensable**: Removing U-Net (w/o U) or pyramid attention (w/o PA) causes $M$ to increase to 0.0057 and 0.0058, respectively.
3. **DGC dual-branch complementarity is effective**: Removing the geometric branch (w/o Geo) increases $M$ to 0.0052, and removing the granularity branch (w/o Gran) increases $M$ to 0.0057.
4. **More unshuffle factors yield better results**: As the number of factors increases from 1 to 4 (1-2-4-6), $M$ decreases from 0.0052 to 0.0041.
5. **Swin Transformer significantly outperforms CNN backbones**: It achieves an $F_\beta$ improvement of 11.3% over ResNet-34 and 15.2% over VGG-16.
6. **Model complexity is moderate**: 95.1M parameters, 94.1G FLOPs, 18.3 FPS, with complementary complexity among the three modules (MDE: 20.5G, DGC: 16.3G, DSP: 5.3G).

## Highlights & Insights
1. **Differentiated hierarchical optimization** is the core novelty: Enhancing details for low-level features, complementing geometric-granularity for mid-level features, and optimizing locations for high-level features is more reasonable than unified processing strategies.
2. **Pixel unshuffle replaces pooling** for multi-scale feature sampling, which compresses spatial scale while fully preserving information, avoiding information loss caused by conventional downsampling.
3. **The cascaded granularity extraction in DGC** is elegantly designed: Information from small receptive fields is progressively injected into branches with larger receptive fields, achieving progressive detail enhancement.
4. Clear advantages are demonstrated in four challenging scenarios: large objects, narrow objects, multiple small objects, and low contrast.

## Limitations & Future Work
1. The inference speed is only 18.3 FPS, and the module design is relatively complex, indicating room for latency optimization.
2. The designs of the four sub-modules are relatively independent and repeatedly use similar attention mechanisms; a lighter, unified framework could be explored.
3. Evaluated only on three remote sensing SOD datasets, and generalization to natural scenes has not been tested.
4. Deep supervision applies the same loss weight to all five prediction levels; adaptive weighting strategies could be investigated.

## Related Work & Insights
- Compared to CorrNet: CorrNet explores mid-level feature correlation using cross-attention, whereas the DGC module of G2HFNet exploits detail and location information in mid-level features more comprehensively.
- Compared to EMFINet/HFANet: These methods adopt unified feature enhancement strategies, while the differentiated design of G2HFNet is more effective.
- The innovative utility of pixel unshuffle is worth referencing and can be generalized to other tasks requiring lossless multi-scale sampling.

## Rating
- **Novelty**: ⭐⭐⭐⭐ (The differentiated hierarchical optimization concept is novel, and pixel unshuffle for multi-scale sampling is creative, but the overall framework still follows a module-stacking paradigm)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (Comprehensive comparison with 18 methods on three datasets; ablation experiments are highly detailed, including six groups of ablations on modules, components, losses, backbones, and unshuffle factors)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear structure, detailed module descriptions, rigorous equations, and thorough visual analysis)
- **Value**: ⭐⭐⭐⭐ (Achieves significant SOTA on remote sensing SOD tasks, and the differentiated hierarchical optimization concept provides insights for other hierarchical feature fusion tasks)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images](rdnet_region_proportion-aware_dynamic_adaptive_salient_object_detection_network_.md)
- [\[CVPR 2025\] Visual Consensus Prompting for Co-Salient Object Detection](visual_consensus_prompting_for_co-salient_object_detection.md)
- [\[CVPR 2025\] RSONet: Region-guided Selective Optimization Network for RGB-T Salient Object Detection](rsonet_region-guided_selective_optimization_network_for_rgb-t_salient_object_det.md)
- [\[CVPR 2025\] SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data](sgma_semantic-guided_modality-aware_segmentation_for_remote_sensing_with_incompl.md)
- [\[CVPR 2025\] ROS-SAM: High-Quality Interactive Segmentation for Remote Sensing Moving Object](ros-sam_high-quality_interactive_segmentation_for_remote_sensing_moving_object.md)

</div>

<!-- RELATED:END -->

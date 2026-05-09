---
title: >-
  [Paper Note] Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance
description: >-
  [CVPR 2026][Segmentation][multi-task learning] This paper proposes an efficient RGB-D multi-task scene understanding network. A partial-channel convolution fusion encoder reduces FLOPs to 1/16 of standard convolution. A Normalized Focus Channel Layer (NFCL) and a Context Feature Interaction Layer (CFIL) enable cross-dimensional feature guidance. A batch-level multi-task adaptive loss dynamically balances five tasks. The method achieves 49.82 mIoU on NYUv2 at 20.33 FPS, which is 24% faster than EMSAFormer.
tags:
  - CVPR 2026
  - Segmentation
  - multi-task learning
  - RGB-D fusion
  - panoptic segmentation
  - adaptive loss
  - cross-dimensional guidance
date: 2026-05-08
content_hash: 35c2a19f1e4a7fd4
---

# Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance

**Conference**: CVPR 2026
**arXiv**: [2603.07570](https://arxiv.org/abs/2603.07570)
**Code**: N/A
**Area**: RGB-D Scene Understanding / Multi-task Learning / Panoptic Segmentation
**Keywords**: multi-task learning, RGB-D fusion, panoptic segmentation, adaptive loss, cross-dimensional guidance

## TL;DR
This paper proposes an efficient RGB-D multi-task scene understanding network. A partial-channel convolution fusion encoder reduces FLOPs to 1/16 of standard convolution. A Normalized Focus Channel Layer (NFCL) and a Context Feature Interaction Layer (CFIL) enable cross-dimensional feature guidance. A batch-level multi-task adaptive loss dynamically balances five tasks. The method achieves 49.82 mIoU on NYUv2 at 20.33 FPS, which is 24% faster than EMSAFormer.

## Background & Motivation
**Background**: Robotic scene understanding requires simultaneous execution of multiple tasks including semantic segmentation, instance segmentation, orientation estimation, panoptic segmentation, and scene classification. RGB-D data fusion has become the mainstream solution, yet efficiently fusing two modalities while jointly optimizing multiple tasks remains an open problem.

**Limitations of Prior Work**: (1) Dual-encoder architectures (e.g., EMSANet) incur high computational costs and underutilize inter-modal complementary information; (2) Transformer-based encoders (e.g., EMSAFormer with Swin v2) involve intensive matrix operations and frequent memory access, limiting inference to 16 FPS; (3) MLP decoder structures are simple and efficient but susceptible to noise from shallow features; (4) Fixed multi-task loss weights cannot adapt to the dynamic changes in task learning throughout training.

**Key Challenge**: The trade-off between multi-task performance and inference speed — how to substantially improve inference efficiency without sacrificing task accuracy.

**Goal**: Design an efficient RGB-D multi-task network that simultaneously addresses modality fusion efficiency, the shallow-feature misleading problem in MLP decoders, and dynamic balancing of multi-task loss weights.

**Key Insight**: Exploiting channel feature redundancy — applying convolution to only 1/4 of channels suffices to achieve full-channel performance, substantially reducing FLOPs and memory access.

**Core Idea**: Partial-channel convolution for efficient RGB-D fusion + cross-dimensional feature guidance for enhanced shallow representations + batch-level adaptive loss for dynamic multi-task balancing.

## Method

### Overall Architecture
The network accepts 4-channel RGBD input and extracts features through a single fusion encoder (based on FasterNet-M, 4 stages with 3/4/18/3 fusion blocks per stage). Encoder outputs are distributed to three branches: (1) a scene classification head (fully connected layers); (2) a semantic segmentation decoder (MLP + NFCL + CFIL, producing pixel-wise semantic labels); and (3) an instance segmentation decoder (three-layer non-bottleneck 1D modules outputting instance centers, offsets, and orientations). Semantic segmentation provides foreground masks to instance segmentation, and their combination forms panoptic segmentation. During training, a multi-task adaptive loss dynamically adjusts per-task learning weights.

### Key Designs
1. **Partial-Channel Fusion Encoder**:
    - Function: Efficient fusion of RGB and depth features.
    - Mechanism: Exploiting the high similarity across channel features, each fusion block applies Conv2D to only 1/4 of the channels while directly concatenating the remaining 3/4: $F = \text{Cat}(\text{Conv2d}(I_1), I_2)$. Since $C'=C/4$, partial convolution reduces FLOPs to 1/16 of full convolution. Two subsequent pointwise convolutions extract inter-channel relationships, followed by a residual connection. Depth weights are initialized as $D=(R+G+B)/2$, reusing ImageNet pretrained parameters.
    - Design Motivation: Frequent memory access is the bottleneck of conventional depthwise separable convolutions; partial-channel convolution reduces memory access while exploiting channel redundancy.

2. **Normalized Focus Channel Layer (NFCL) + Context Feature Interaction Layer (CFIL)**:
    - Function: NFCL filters shallow-feature noise; CFIL compensates for insufficient local-global fusion in MLP decoders.
    - Mechanism: NFCL reuses the learnable scaling factor $\gamma$ from Batch Normalization as a channel importance measure. Channel weights are computed as $W_i = |\gamma_i| / \sum_j |\gamma_j|$, and sigmoid gating suppresses shallow-feature noise. CFIL performs adaptive average pooling at two scales (1×1 and 5×5), compresses channels to $C/2$, upsamples and concatenates with original features, then restores the channel count.
    - Design Motivation: MLP decoders depend heavily on encoder feature quality — NFCL eliminates shallow-feature misleading while CFIL supplements multi-scale context, and the two modules complement each other.

3. **Multi-task Adaptive Loss**:
    - Function: Real-time batch-level dynamic adjustment of per-task learning weights.
    - Mechanism: At each batch, the relative loss of each task is computed as $RL_k = L_k / \sum_t L_t$; a historical mean $\text{Avg}RL_k$ is maintained; weights are updated as $W_k = \max(\bar{W}_k \times (\text{Avg}RL_k)^\alpha, W_{min})$, where $\alpha=0.01$ controls sensitivity and $W_{min}=0.1$ prevents any task from being ignored.
    - Design Motivation: Responds faster than epoch-level methods and adapts to intra-epoch data distribution shifts; more stable than stochastic weighting approaches (e.g., Lin et al.).

### Loss & Training
Each of the five tasks employs a dedicated loss: semantic segmentation (CE), instance center (MSE), instance offset (MAE), orientation estimation (von Mises: $L_{or}=1-e^{\kappa(f \cdot t - 1)}$), and scene classification (CE). These are combined via adaptive weighting. The optimizer is SGD (lr=0.03, weight decay=1e-4, momentum=0.9), trained on an RTX 3090 Ti.

## Key Experimental Results

### Main Results

| Dataset | Method | Semantic mIoU | PQ (Panoptic) | FPS | Params |
|---------|--------|---------------|---------------|-----|--------|
| NYUv2 | EMSAFormer (Swin v2) | 49.76 | 43.08 | 16.32 | 72.08M |
| NYUv2 | **Ours** | **49.82** | **43.21** | **20.33** | **71.82M** |
| NYUv2 | MPViT | - | - | 9.94 | - |
| SUN RGB-D | CI-Net | 44.30 | - | - | - |
| SUN RGB-D | **Ours** | **45.56** | - | - | - |
| Cityscapes | PSPNet | 63.10 | - | - | - |
| Cityscapes | **Ours** | **65.11** | - | - | - |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|------------|-------|
| Fusion encoder vs. Swin v2 | Instance PQ 58.59 vs. 58.49, faster | Fewer parameters, higher speed, comparable accuracy |
| +CFIL (semantic decoder) | Panoptic mIoU 50.16% | Multi-scale context fusion is effective |
| +NFCL (layers 1/2/3) | mIoU 49.82% | Stage-4 encoder features are already sufficient |
| Non-bottleneck 1D vs. Bottleneck | PQ 59.25 vs. 57.97 | Decomposed convolutions enhance nonlinearity |
| Adaptive loss vs. fixed weights | mIoU 47.72 vs. 46.83 | Training variance is also reduced |
| $\alpha$=0.01 vs. 0.1/0.001 | 0.01 is optimal | Balances sensitivity and stability |

### Key Findings
- Partial-channel convolution is effective for dense prediction tasks — FLOPs are reduced by 16× with negligible accuracy loss.
- NFCL reuses BN's $\gamma$ parameters as a zero-overhead channel importance measure.
- Batch-level adaptive loss is more stable than epoch-level approaches, with lower training variance.
- NB1D reduces parameters by 30% yet improves PQ by 1.28; the nonlinear activations from decomposed convolutions benefit instance segmentation.

## Highlights & Insights
- The efficiency principle is consistently applied throughout the entire framework: partial channels in the encoder (1/16 FLOPs), zero-overhead BN reuse in NFCL, and 30% parameter reduction via NB1D.
- NFCL is minimally designed — it directly repurposes existing BN $\gamma$ parameters for channel weighting without introducing any additional learnable parameters.
- An excellent speed–accuracy trade-off is achieved: 24% faster than Swin v2 at higher accuracy.
- The multi-task adaptive loss operates at the batch level, responding faster than epoch-level alternatives.

## Limitations & Future Work
- The partial-channel ratio of 1/4 is fixed; dataset-adaptive selection could be explored.
- Validation is limited to RGB-D; extension to thermal imaging, point clouds, and other modalities remains unexplored.
- The method processes frames independently without exploiting temporal consistency in video streams.
- $\alpha$ and $W_{min}$ are manually set hyperparameters; automated tuning could be considered.
- Scalability to high-resolution scenes has not been verified.

## Related Work & Insights
- **vs. EMSAFormer**: Both address RGB-D multi-task learning; this work replaces the Swin Transformer with a CNN encoder to achieve higher speed (20.33 vs. 16.32 FPS) at comparable accuracy.
- **vs. EMSANet**: The dual-encoder fusion incurs high computational cost; this work directly processes RGBD 4-channel input with a single encoder.
- **vs. SegFormer**: The limitations of MLP decoders are effectively mitigated by the combination of NFCL and CFIL.
- The partial-channel convolution concept originates from FasterNet; this work demonstrates its effectiveness in dense multi-task prediction scenarios.

## Rating
- Novelty: ⭐⭐⭐ Individual components show some novelty, but the overall contribution is largely an integration and optimization of existing techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets, detailed ablations, and heatmap visualizations.
- Writing Quality: ⭐⭐⭐ Well-structured overall, though some descriptions are slightly redundant.
- Value: ⭐⭐⭐⭐ Practically valuable for robotic scene understanding, with an excellent speed–accuracy balance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DSFlash: Comprehensive Panoptic Scene Graph Generation in Realtime](dsflash_comprehensive_panoptic_scene_graph_generation_in_realtime.md)
- [\[CVPR 2026\] Learning Cross-View Object Correspondence via Cycle-Consistent Mask Prediction](learning_cross-view_object_correspondence_via_cycle-consistent_mask_prediction.md)
- [\[CVPR 2026\] 3M-TI: High-Quality Mobile Thermal Imaging via Calibration-free Multi-Camera Cross-Modal Diffusion](3m-ti_high-quality_mobile_thermal_imaging_via_calibration-free_multi-camera_cros.md)
- [\[CVPR 2026\] GeomPrompt: Geometric Prompt Learning for RGB-D Semantic Segmentation Under Missing and Degraded Depth](geomprompt_rgbd_segmentation.md)
- [\[CVPR 2026\] Heuristic Self-Paced Learning for Domain Adaptive Semantic Segmentation under Adverse Conditions](heuristic_self-paced_learning_for_domain_adaptive_semantic_segmentation_under_ad.md)

</div>

<!-- RELATED:END -->

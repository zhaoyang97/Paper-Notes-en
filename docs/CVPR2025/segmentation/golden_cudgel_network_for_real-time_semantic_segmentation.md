---
title: >-
  [Paper Note] Golden Cudgel Network for Real-Time Semantic Segmentation
description: >-
  [CVPR 2025][Segmentation][real-time semantic segmentation] This paper proposes GCNet, featuring the Golden Cudgel Block (GCBlock) as its core. It self-expands during training (multi-convolution, multi-path) to enhance learning capacity, and self-contracts during inference (reparameterized into a single $3\times3$ convolution) to accelerate speed. This yields a "self-distillation" paradigm without requiring an external teacher model, outperforming existing real-time segmentati…
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "real-time semantic segmentation"
  - "reparameterization"
  - "GCBlock"
  - "dual-branch"
  - "Cityscapes"
date: 2026-05-08
content_hash: 8831c4b943298755
---

# Golden Cudgel Network for Real-Time Semantic Segmentation

**Conference**: CVPR 2025  
**arXiv**: [2503.03325](https://arxiv.org/abs/2503.03325)  
**Code**: [GitHub](https://github.com/gyyang23/GCNet)  
**Area**: Image Segmentation  
**Keywords**: real-time semantic segmentation, reparameterization, GCBlock, dual-branch, Cityscapes

## TL;DR

This paper proposes GCNet, featuring the Golden Cudgel Block (GCBlock) as its core. It self-expands during training (multi-convolution, multi-path) to enhance learning capacity, and self-contracts during inference (reparameterized into a single $3\times3$ convolution) to accelerate speed. This yields a "self-distillation" paradigm without requiring an external teacher model, outperforming existing real-time segmentation models on Cityscapes with 77.3% mIoU at 193.3 FPS.

## Background & Motivation

**Background**: Real-time semantic segmentation models strive for a trade-off between accuracy and speed. Single-branch models (ERFNet, STDC, SCTNet) improve performance via lightweight designs or knowledge distillation. Multi-branch models (BiSeNet, DDRNet, PIDNet) enhance spatial detail capture through a dual-branch structure of semantic and detail branches.

**Limitations of Prior Work**:
1. **Multi-path blocks hinder inference speed**: Residual connections increase memory access frequency, and Transformer-like structures such as Conv-Former Blocks further impact efficiency.
2. **Reliance on external teacher models**: Methods like SCTNet require high-performance Transformer-based segmentation models (e.g., SegFormer) for knowledge distillation, increasing training costs and complexity.

**Key Challenge**: Multi-path structures facilitate training (preventing gradient vanishing/explosion) but degrade inference speed; single-path structures offer fast inference but possess weak representation capability.

**Goal**: Combine the training advantages of multi-path structures with the inference advantages of single-path structures, without relying on an external teacher model.

**Key Insight**: Structural reparameterization—expanding into multiple convolutions and paths during training, and losslessly collapsing into a single convolution during inference.

**Core Idea**: Like the Golden Cudgel, it can expand during training (multi-path, multi-conv) and contract during inference (single $3\times3$ conv), achieving a unified "teacher-student" paradigm within itself.

## Method

### Overall Architecture

GCNet adopts a dual-branch architecture:
1. **Stem**: Two $3\times3$ convolutions with stride=2 for rapid downsampling.
2. **Stage 2-3**: Stacked shared GCBlocks.
3. **Stage 4-6**: Divided into a semantic branch (deep semantics) and a detail branch (spatial details), interacting via bidirectional feature fusion (convolutional channel adjustment + bilinear interpolation).
4. **PPM**: Deep Aggregation Pyramid Pooling Module at the end of the semantic branch.
5. **Segmentation Head**: Fusion with a $3\times3$ convolution + alignment of category count with a $1\times1$ convolution.

Three versions are provided: GCNet-S ($C=32$), GCNet-M ($C=64$), and GCNet-L ($C=64$, deeper).

### Key Designs

#### 1. Golden Cudgel Block (GCBlock) — Core Innovation

**Training Structure** (multi-conv, multi-path):
- **Path₃ₓ₃_₁ₓ₁** ($\times N$): One $3\times3$ convolution + one $1\times1$ convolution, with $N$ parallel paths.
- **Path₁ₓ₁_₁ₓ₁**: A series of two $1\times1$ convolutions.
- **Path_residual**: Residual connections (used when stride=1, realized via identity convolutions with BN).

**Inference Structure** (single $3\times3$ convolution):
- Conv-BN fusion: $W' = \frac{\gamma}{\sqrt{\sigma+\varepsilon}}W$
- Vertical reparameterization: A $1\times1$ conv following a $3\times3$ conv can be equivalently merged into a new $3\times3$ conv (leveraging the matrix multiplication equivalence of im2col).
- Horizontal reparameterization: The weights and biases of multiple parallel $3\times3$ paths are directly summed.

Key finding: Removing the first $1\times1$ convolution in the bottleneck (as its post-training parameter values are too small to affect lossless fusion); stacking 2 layers for Path₁ₓ₁_₁ₓ₁ yields the best results.

#### 2. Self-Distillation Mechanism

The expanded GCNet during training acts as the "teacher model," while the contracted version during inference serves as the "student model." Reparameterization guarantees lossless conversion, eliminating the traditional two-stage distillation pipeline.

#### 3. Lightweight Feature Fusion

Feature fusion between the dual branches employs only $3\times3$ convolutions (channel compression/expansion) + bilinear interpolation (up/down-sampling), **without using any attention modules**, to avoid extra inference overhead.

### Loss & Training

$$L = L_{sh} + \alpha L_{ash}$$

- $L_{sh}$: OHEM Cross Entropy loss of the main segmentation head.
- $L_{ash}$: OHEM Cross Entropy loss of the auxiliary segmentation head (used for deep supervision during training, removed during inference).
- $\alpha = 0.4$

## Key Experimental Results

### Main Results (Cityscapes validation set, A100)

| Model | mIoU (%) | FPS | Params | ImageNet |
|------|----------|-----|--------|----------|
| DDRNet-23-Slim | 76.3 | 166.4 | 5.7M | ✗ |
| PIDNet-S | 76.4 | 128.7 | 7.7M | ✗ |
| SCTNet-B-Seg100 | 79.0 | 117.0 | 17.4M | ✗ |
| **GCNet-S** | **77.3** | **193.3** | 9.2M | ✗ |
| **GCNet-M** | **79.0** | **105.0** | 34.2M | ✗ |
| **GCNet-L** | **79.6** | **88.0** | 45.2M | ✗ |

- GCNet-S achieves a 77.3% mIoU at 193.3 FPS, with no competitor in the same speed tier.
- GCNet-M matches the accuracy of SCTNet-B-Seg100 (79.0%) but **requires no ImageNet pre-training or teacher models**.
- GCNet-L reaches a 79.6% mIoU, the highest among real-time models without pre-training.

### Ablation Study

**Number of $1\times1$ Convolutions in Path₁ₓ₁_₁ₓ₁ (GCNet-S)**:

| Quantity | GPU Memory | Training Time | mIoU |
|------|------|----------|------|
| 0 | 20.58 GiB | 4.0h | 76.1 |
| 1 | 21.87 GiB | 4.5h | 76.6 |
| **2** | **24.61 GiB** | **5.0h** | **76.7** |
| 3 | 27.31 GiB | 5.4h | 76.4 |

2 layers are optimal; 3 layers cause overfitting and a drop in accuracy.

**Number of Path₃ₓ₃_₁ₓ₁ ($N$)**: Increasing $N$ accelerates convergence (the improvement of $N=3$ at 20k steps is significant compared to $N=1$), but has little effect on the final accuracy. A larger $N$ makes the training more stable.

### Key Findings

1. Reparameterization incurs **no loss of accuracy**: The conversion from the training to inference structure is mathematically equivalent.
2. High competitive accuracy is achieved without ImageNet pre-training.
3. The auxiliary segmentation head (deep supervision) performs best after stage 4 ($\alpha = 0.4$).
4. Consistent advantages are demonstrated on CamVid and Pascal VOC 2012 as well.

## Highlights & Insights

1. **Elegant Metaphor of the "Golden Cudgel"**: The design concept of expanding during training and contracting during inference is intuitive and cleverly named.
2. **Elimination of Teacher Dependency**: The large-scale model during training serves as the teacher, while the small-scale model during inference serves as the student, removing any extra training pipeline.
3. **Complete Mathematical Derivation**: Every step, from Conv-BN fusion to vertical/horizontal merging, has rigorous formulations.
4. **Fully From-Scratch Training**: It does not use ImageNet pre-training, lowering the prerequisites of the proposed method.
5. **Significant Speed Advantage**: GCNet-S reaches 193.3 FPS on the full 1024×2048 resolution, far exceeding other models in the same accuracy range.

## Limitations & Future Work

1. The parameter scale of GCNet-L (45.2M) is relatively large for real-time models, limiting deployment on edge devices.
2. Dual-branch fusion does not use attention mechanisms, which saves speed but may lose fine-grained feature alignment.
3. Evaluated only on urban scene datasets, without testing on larger-scale/more diverse datasets like ADE20K.
4. Reparameterization techniques (RepVGG family) are not pioneered here; the innovation primarily lies in systematically applying them to segmentation tasks.
5. Training compute and memory overhead increase with the number of paths $N$, requiring a carefully chosen trade-off.

## Related Work & Insights

1. **RepVGG** (Ding et al., 2021): Pioneer work in reparameterization, and the source of the core idea behind GCBlock.
2. **DDRNet** (Hong et al., 2021): Source of the dual-branch architecture and DAPPM.
3. **PIDNet** (Xu et al., 2023): A triple-branch architecture; GCNet proves that dual-branch with reparameterization can outperform it.
4. **SCTNet** (Xu et al., 2024): A single-branch scheme using a teacher model; GCNet achieves comparable accuracy without a teacher.

**Insight**: Reparameterization is not only suitable for classification (RepVGG) but also highly effective for dense prediction tasks. The concept of "expandable during training, contractible during inference" can be generalized to other compute-sensitive vision tasks (e.g., object detection, depth estimation).

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: ⭐⭐⭐ — The core idea (reparameterization) is not entirely new, but its systematic application in semantic segmentation and the "Golden Cudgel" design show novelty.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive experiments done on three datasets, detailed ablation studies, and unified speed testing on the same GPU.
- **Writing Quality**: ⭐⭐⭐⭐ — Detailed mathematical derivations and a clear paper structure.
- **Value**: ⭐⭐⭐⭐⭐ — No pre-training/teacher required, open-sourced code, outstanding speed, extremely engineering-friendly.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] PicoSAM3: Real-Time In-Sensor Region-of-Interest Segmentation](picosam3_real-time_in-sensor_region-of-interest_segmentation.md)
- [\[CVPR 2025\] Condensing Action Segmentation Datasets via Generative Network Inversion](condensing_action_segmentation_datasets_via_generative_network_inversion.md)
- [\[CVPR 2026\] The Golden Subspace: Where Efficiency Meets Generalization in Continual Test-Time Adaptation](../../CVPR2026/segmentation/the_golden_subspace_where_efficiency_meets_generalization_in_continual_test-time.md)
- [\[ICLR 2026\] QPrompt-R1: Real-Time Reasoning for Domain-Generalized Semantic Segmentation via Group-Relative Query Alignment](../../ICLR2026/segmentation/qprompt-r1_real-time_reasoning_for_domain-generalized_semantic_segmentation_via_.md)
- [\[CVPR 2025\] G2HFNet: GeoGran-Aware Hierarchical Feature Fusion Network for Salient Object Detection in Optical Remote Sensing Images](binwang2hfnet_geogran-aware_hierarchical_feature_fusion_network_for_salient_obje.md)

</div>

<!-- RELATED:END -->

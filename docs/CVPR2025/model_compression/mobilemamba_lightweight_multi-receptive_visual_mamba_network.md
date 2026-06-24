---
title: >-
  [Paper Note] MobileMamba: Lightweight Multi-Receptive Visual Mamba Network
description: >-
  [CVPR 2025][Model Compression][lightweight network] This work proposes MobileMamba, a lightweight visual network. Through a three-stage coarse-grained architectural design and the fine-grained MRFFI module (integrating Mamba global modeling, multi-kernel convolutional multi-scale perception, and Identity redundancy elimination), MobileMamba achieves an optimal balance between speed and accuracy on both classification and downstream high-resolution tasks.
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "lightweight network"
  - "Mamba"
  - "SSM"
  - "multi-receptive field"
  - "wavelet transform"
  - "efficient inference"
date: 2026-05-08
content_hash: 06ca5973f8d97bc4
---

# MobileMamba: Lightweight Multi-Receptive Visual Mamba Network

**Conference**: CVPR 2025  
**arXiv**: [2411.15941](https://arxiv.org/abs/2411.15941)  
**Code**: [GitHub](https://github.com/lewandofskee/MobileMamba)  
**Area**: Model Compression  
**Keywords**: lightweight network, Mamba, SSM, multi-receptive field, wavelet transform, efficient inference

## TL;DR

This work proposes MobileMamba, a lightweight visual network. Through a three-stage coarse-grained architectural design and the fine-grained MRFFI module (integrating Mamba global modeling, multi-kernel convolutional multi-scale perception, and Identity redundancy elimination), MobileMamba achieves an optimal balance between speed and accuracy on both classification and downstream high-resolution tasks.

## Background & Motivation

**Background**: Lightweight mobile models mainly fall into two categories:
- **CNN-based** (MobileNet, GhostNet): Local effective receptive fields (ERF), lacking long-range dependency, requiring increased computation at high resolutions.
- **ViT-based** (EfficientViT, SHViT): Global ERF, but quadratic computational complexity is costly in high-resolution scenarios.

**Key Challenge**: Recent Mamba (State Space Model) models achieve global modeling with linear complexity, but existing lightweight Mamba networks (EfficientVMamba, LocalVim) **have poor actual throughput despite low FLOPs**. FLOPs do not equal inference speed—factors such as memory access patterns of scanning methods and network topology severely impact actual throughput.

**Key Insight**: Systematically optimize the efficiency and performance of Mamba-based lightweight networks at both the coarse-grained (macro-architecture) and fine-grained (module design) levels.

## Method

### Overall Architecture

MobileMamba adopts a **three-stage** network structure (vs. the mainstream four-stage):

- **16×16 PatchEmbed**: The first downsampling directly reduces the resolution to H/16×W/16 (whereas the four-stage approach downsamples to H/4×W/4).
- **Three stages**: Gradually downsample to H/64×W/64.
- **Advantages**: Smaller feature maps $\rightarrow$ lower computational cost $\rightarrow$ faster inference speed, achieving higher accuracy (+0.4% Top-1) under equivalent throughput.

Each stage consists of multiple MobileMamba Blocks. Each Block contains: Local Information Perception $\rightarrow$ **MRFFI** $\rightarrow$ FFN.

### Key Designs

#### 1. MRFFI Multi-Receptive Field Feature Interaction Module

The input features are divided into three parts along the channel dimension:

**Part 1 — WTE-Mamba (Wavelet-enhanced Mamba for Long-range)**:
- Mamba module extracts global features (bidirectional scanning).
- Haar wavelet transform extracts high-frequency edge details (LL/LH/HL/HH sub-bands).
- Convolution in the wavelet domain provides a larger ERF and lower computational complexity.
- Summing and fusing the two paths of features: $x_G^O = x_m^O + x_w^O$

**Part 2 — MK-DeConv (Multi-Kernel Depthwise Separable Convolution)**:
- Channels are split into $n$ groups, each using a different kernel size $(2j+1)$.
- $j=1,2,...,n$ corresponding to 3×3, 5×5, 7×7, etc.
- Multi-scale local receptive fields enhance the extraction of adjacent spatial information.

**Part 3 — Eliminate Redundant Identity**:
- The remaining channels are bypassed directly via identity mapping.
- Reduces feature redundancy in high-dimensional spaces.
- Decreases computational complexity and improves processing speed.

Finally, the three parts are concatenated: $x^O = \text{Concat}(x_G^O, x_L^O, x^I[\text{identity channels}])$

#### 2. Loss & Training

- **Knowledge Distillation**: A large model serves as the teacher for soft distillation (DeiT style).
- **Extended Training Epochs**: Since small models do not fully converge at 300 epochs, training is extended to 1000 epochs.

#### 3. Inference Strategy

- **Normalization Layer Fusion**: Fuses BN layers into convolutional/linear layers to reduce the number of layers and computations during inference.

### Loss & Training

Standard cross-entropy classification loss + KL divergence loss for knowledge distillation.

## Key Experimental Results

### Main Results (ImageNet-1K Classification)

| Model | FLOPs | Throughput | Params | Top-1 |
|------|-------|-----------|--------|-------|
| EfficientVMamba-T | 800M | — | 6.0M | 76.5 |
| LocalVim-T | 1500M | — | 8.0M | 76.2 |
| MobileMamba-S6 | 652M | — | 15.0M | 78.0 |
| **MobileMamba-S6†** | **652M** | — | 15.0M | **80.7** |
| SHViT-S3 | 601M | — | 14.2M | 77.4 |
| EfficientViT-M5 | 522M | — | 12.4M | 77.1 |
| **MobileMamba-B4†** | ~4G | — | — | **83.6** |

MobileMamba-S6† (with the training strategy) achieves 80.7% Top-1 accuracy at 652M FLOPs, surpassing all counterpart CNN/ViT/Mamba models of similar scale.

### Downstream Task Performance

| Task | Baseline Method | MobileMamba Gain |
|------|---------|----------------|
| Mask RCNN Detection | vs. EMO | mAP^b +1.3↑, mAP^m +1.0↑, Throughput +56%↑ |
| RetinaNet Detection | vs. EfficientVMamba | mAP^b +2.1↑, Throughput ×4.3↑ |
| PSPNet Semantic Segmentation | vs. MobileNetv2 | mIoU +7.2↑, with only 8.5% FLOPs |
| PSPNet Semantic Segmentation | vs. MobileViTv2 | mIoU +0.4↑, with only 11.2% FLOPs |

### Ablation Study (Replacing Mamba with Other RNN Paradigms)

| Method | FLOPs | Throughput | Top-1 |
|------|-------|-----------|-------|
| TTT | 625M | 9569 | 77.0 |
| xLSTM | 695M | 6868 | 77.3 |
| RWKV6 | 658M | 10331 | 77.8 |
| **Mamba** | **652M** | **11000** | **78.0** |

### Key Findings

1. **Three-stage outperforms four-stage**: Higher accuracy (+0.4%) at equivalent throughput, and faster throughput at equivalent accuracy.
2. **Effectiveness of MRFFI three-way splitting**: Global Mamba + multi-scale local convolution + identity mapping each play indispensable roles.
3. **Significant gain from wavelet transform**: The wavelet branch in WTE-Mamba enhances high-frequency edge feature extraction.
4. **Large boost from training strategy**: Knowledge distillation + 1000 training epochs boosts accuracy from 78.0% to 80.7% (+2.7%).
5. **Mamba outperforms other RNN paradigms**: Compared to TTT, xLSTM, and RWKV6, Mamba demonstrates the best balance of throughput and accuracy.
6. **Pronounced advantages in high-resolution downstream tasks**: The efficiency gains brought by linear complexity in detection/segmentation tasks far surpass those of CNNs and ViTs.

## Highlights & Insights

1. **Systematic Design**: Synergistic optimization across three levels—from macro-architecture (three-stage) and micro-module (MRFFI) to training/testing strategies.
2. **Actual Throughput-driven**: Instead of solely pursuing low FLOPs, the work prioritizes real-world inference speed (achieving 21$\times$ speedup compared to LocalVim).
3. **Novel Multi-Receptive Field Fusion**: A channel-splitting strategy that combines global Mamba, high-frequency wavelets, multi-kernel local convolutions, and identity redundancy elimination.
4. **Thorough Downstream Validation**: Comprehensive evaluation across 5 different detection/segmentation frameworks, extending beyond simple classification.

## Limitations & Future Work

1. **Relatively large parameter count**: MobileMamba-S6 has 15M parameters, which is larger than some extremely lightweight models (1-6M parameters).
2. **Immaturity of Mamba operators**: The CUDA kernel optimization for Mamba is not as mature as that for CNNs/ViTs, leading to limited support on certain hardware devices.
3. **Downstream adaptation of three-stage design**: The feature map scales (H/16, H/32, H/64) of the three-stage structure are not fully compatible with mainstream four-stage detection/segmentation heads.
4. **Increased implementation complexity from wavelet transform**: Though Haar wavelet transform is theoretically elegant, it adds challenges to engineering implementation and debugging.
5. **Dependence on knowledge distillation**: The best result of MobileMamba† relies heavily on a teacher model, which limits independent deployment.

## Related Work & Insights

- **EfficientViT / SHViT**: Key references for ViT-based lightweight models; the three-stage architectural design was inspired by EfficientViT.
- **EfficientVMamba / LocalVim**: Pioneers in Mamba-based lightweight models; this work significantly improves speed and accuracy compared to them.
- **GhostNet**: The concept of using identity/cheap operations to reduce redundant computation is inherited in MRFFI.
- **Insights**: Dividing and conquering along the channel dimension (partially global, partially local, and partially bypassed) is an efficient and versatile paradigm for lightweight module design.

## Rating ⭐

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Engineering Practicality | ⭐⭐⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] JamMa: Ultra-lightweight Local Feature Matching with Joint Mamba](jamma_ultra-lightweight_local_feature_matching_with_joint_mamba.md)
- [\[CVPR 2025\] Mamba-Adaptor: State Space Model Adaptor for Visual Recognition](mamba-adaptor_state_space_model_adaptor_for_visual_recognition.md)
- [\[CVPR 2025\] Binarized Mamba-Transformer for Lightweight Quad Bayer HybridEVS Demosaicing](binarized_mamba-transformer_for_lightweight_quad_bayer_hybridevs_demosaicing.md)
- [\[CVPR 2025\] EfficientViM: Efficient Vision Mamba with Hidden State Mixer based State Space Duality](efficientvim_efficient_vision_mamba_with_hidden_state_mixer_based_state_space_du.md)
- [\[CVPR 2025\] Parameter Efficient Mamba Tuning via Projector-targeted Diagonal-centric Linear Transformation](parameter_efficient_mamba_tuning_via_projector-targeted_diagonal-centric_linear_.md)

</div>

<!-- RELATED:END -->

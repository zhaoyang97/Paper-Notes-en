---
title: >-
  [Paper Note] HPE-Li: WiFi-Enabled Lightweight Dual Selective Kernel Convolution for Human Pose Estimation
description: >-
  [ECCV 2024][Human Understanding][WiFi Pose Estimation] This paper proposes HPE-Li, a lightweight human pose estimation method based on WiFi signals. By constructing a multi-branch CNN using an innovative Dual Selective Kernel Attention (SKA) mechanism, it dynamically adjusts the receptive field size according to the characteristics of the input WiFi CSI data, surpassing SOTA methods on both MM-Fi and WiPose benchmarks with extremely low computational overhead.
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "WiFi Pose Estimation"
  - "Selective Kernel Attention"
  - "Lightweight Network"
  - "Multimodal Fusion"
  - "Channel State Information"
date: 2026-05-08
content_hash: 53728450cc9bd14e
---

# HPE-Li: WiFi-Enabled Lightweight Dual Selective Kernel Convolution for Human Pose Estimation

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Human Understanding / Pose Estimation  
**Keywords**: WiFi Pose Estimation, Selective Kernel Attention, Lightweight Network, Multimodal Fusion, Channel State Information

## TL;DR

This paper proposes HPE-Li, a lightweight human pose estimation method based on WiFi signals. By constructing a multi-branch CNN using an innovative Dual Selective Kernel Attention (SKA) mechanism, it dynamically adjusts the receptive field size according to the characteristics of the input WiFi CSI data, surpassing SOTA methods on both MM-Fi and WiPose benchmarks with extremely low computational overhead.

## Background & Motivation

**Background**: Human Pose Estimation (HPE) is a key task in computer vision, where traditional methods rely on RGB cameras or depth sensors. Recently, WiFi signal-based (especially Channel State Information, CSI) HPE has attracted attention as a privacy-preserving alternative. WiFi signals can capture human motion information even through walls and occlusions without compromising visual privacy.

**Limitations of Prior Work**: WiFi-based HPE faces two major challenges. (1) **Trade-off between signal quality and computational cost**—WiFi CSI data has a much lower spatial resolution than images (typically only 30-300 subcarriers $\times$ a few antenna pairs) but contains rich frequency domain information. Existing methods either use simple CNNs that fail to exploit the multi-scale features of CSI, or employ complex Transformer architectures that suffer from prohibitive computational costs, making them unsuitable for edge deployment. (2) **Limitations of fixed receptive fields**—Standard CNNs use fixed kernel sizes, whereas different human actions exhibit vastly different temporal and spatial scales in CSI signals. Small-scale movements (such as finger movements) present as high-frequency subtle variations in CSI, while large-scale movements (such as walking) present as low-frequency intense variations. Fixed kernel sizes cannot capture motion features of varying scales simultaneously.

**Key Challenge**: WiFi HPE requires a network architecture capable of adaptively processing motion features at different scales. However, increasing network capacity (e.g., through multi-scale feature pyramids or attention mechanisms) usually incurs a dramatic increase in computational cost, which conflicts with the low-power requirements of edge device deployment.

**Goal**: To design a highly computationally efficient WiFi HPE model with dynamic receptive field adjustment capabilities, enabling it to adaptively process multi-scale motion features while remaining lightweight.

**Key Insight**: Inspired by SKNet (Selective Kernel Networks), the authors propose introducing a selective kernel mechanism into WiFi HPE, allowing the network to automatically learn to select convolutional kernels of different sizes based on different input features. However, unlike standard SKNet, HPE-Li designs a "dual" selective kernel (Dual SKA) that adaptively selects across both spatial and channel dimensions. This is achieved while keeping the extra computational overhead under 5% through parameter sharing and efficiency optimization.

**Core Idea**: To achieve dynamic receptive field adjustment through dual-dimensional selective kernel attention, empowering lightweight CNNs to process multi-scale WiFi motion features at minimal computational cost.

## Method

### Overall Architecture

The input to HPE-Li is multi-antenna WiFi CSI data (with dimensions of $T \times N_{sub} \times N_{ant}$, corresponding to the number of time frames, subcarriers, and antenna pairs, respectively), and the output is 3D human joint coordinates. The overall pipeline is as follows: first, the raw CSI is preprocessed (phase correction, denoising), and then the CSI data is mapped to a 2D feature map through a feature embedding layer. Subsequently, key features are extracted through multiple Dual-SKA residual blocks. Finally, the 3D coordinates of $K$ joints are predicted via a regression head. The total number of model parameters is kept under 0.5M, and the inference speed can reach hundreds of FPS.

### Key Designs

1. **Dual Selective Kernel Attention (Dual-SKA)**:

    - Function: Achieves adaptive feature selection in both spatial scale and channel dimensions within a single residual block.
    - Mechanism: Each Dual-SKA block contains two parallel branches, using depthwise separable convolutional kernels of different sizes ($3 \times 3$ and $5 \times 5$). The outputs of the two branches are fused through a three-step operation of Split-Fuse-Select: (i) **Split**: The two branches compute feature maps $U_1 = DW_{3\times3}(X)$ and $U_2 = DW_{5\times5}(X)$; (ii) **Fuse**: The outputs of the two branches are summed, processed with global average pooling and global max pooling, and concatenated before being fed into a shared lightweight FC layer to generate an attention vector $z$; (iii) **Select**: Softmax is applied to $z$ to generate two sets of attention weights $a_1, a_2$, and the final output is $Y = a_1 \odot U_1 + a_2 \odot U_2$. The dual dimensions are reflected in the spatial dimension by achieving multi-scale receptive fields with different kernel sizes, and in the channel dimension by implementing channel-wise adaptive selection through channel-wise attention weights.
    - Design Motivation: Different subcarriers in WiFi CSI have different sensitivities to different frequencies of motion (low-frequency subcarriers are sensitive to large movements, and vice versa). Therefore, adaptive selection in the channel dimension is necessary. At the same time, spatial convolutional kernels of different scales capture spatiotemporal patterns of different granularities. Dual-dimensional selective attention allows the model to "intelligently" allocate feature extraction strategies across both dimensions based on the current input.

2. **Multi-Branch Lightweight CNN Backbone**:

    - Function: Extracts multi-scale CSI features under an extremely low computational budget.
    - Mechanism: The backbone network adopts an Inverted Residual structure style similar to MobileNet-V2, but the standard depthwise separable convolutions in each block are replaced with Dual-SKA. The network consists of 4 stages, with the number of channels gradually increasing from 16 to 128. Standard convolutions are replaced by depthwise separable convolutions in each stage, and parameters are further compressed through grouped pointwise convolutions. A key efficiency trick is that the extra parameters for SKA attention calculation are restricted to only $1/r$ of the original convolution parameters ($r=16$) via dimension reduction, keeping the overall extra overhead below 5%.
    - Design Motivation: Real-world deployment scenarios of WiFi HPE (e.g., smart homes, security) require models to run in real-time on edge devices (such as Raspberry Pi). Overly heavy model architectures lack practical value even if they offer better accuracy.

3. **Multimodal Teacher-Student Training Strategy**:

    - Function: Utilizes the rich information from the visual modality to guide the learning of the WiFi modality.
    - Mechanism: Training is divided into two stages. The first stage uses vision-WiFi multimodal data (e.g., the MM-Fi dataset, which provides both RGB and CSI) to train an image-based teacher network. In the second stage, knowledge distillation is employed to transfer intermediate layer features and output probability distributions of the teacher network to the WiFi student network (i.e., HPE-Li). The distillation loss is defined as $\mathcal{L}_{KD} = \alpha \mathcal{L}_{feat} + (1-\alpha) \mathcal{L}_{logit}$, where $\mathcal{L}_{feat}$ is the L2 distance of intermediate features (which needs to align dimensions via a projection layer), and $\mathcal{L}_{logit}$ is the KL divergence of the output heatmaps. During the inference phase, only the student network (HPE-Li) is used, and the camera is no longer required.
    - Design Motivation: The information capacity of WiFi CSI signals is inherently limited, and training purely on WiFi is prone to local optima. The "soft targets" provided by the visual teacher network contain richer human structure priors, helping the WiFi student network to better understand the topological relationships between joints.

### Loss & Training

The total training loss is $\mathcal{L} = \mathcal{L}_{joint} + \lambda_1 \mathcal{L}_{bone} + \lambda_2 \mathcal{L}_{KD}$, where $\mathcal{L}_{joint} = \frac{1}{K} \sum_{k=1}^{K} \|p_k - \hat{p}_k\|_2$ is the L2 loss of joint coordinates, $\mathcal{L}_{bone}$ constrains the consistency of bone lengths, and $\mathcal{L}_{KD}$ is the knowledge distillation loss. An AdamW optimizer is used with cosine learning rate decay, a batch size of 32, for 200 epochs.

## Key Experimental Results

### Main Results

**MM-Fi Dataset (17 joints)**:

| Method | MPJPE↓ (mm) | PA-MPJPE↓ (mm) | FLOPs (M) | Params (K) |
|------|------------|----------------|-----------|-----------|
| WiPose | 68.4 | 52.3 | 842 | 2,340 |
| MetaFi | 61.7 | 47.8 | 1,256 | 3,120 |
| Person-in-WiFi | 57.3 | 43.1 | 2,015 | 5,430 |
| HPE-Li (Ours) | **49.6** | **37.2** | **168** | **487** |

**WiPose Dataset (18 joints)**:

| Method | MPJPE↓ (mm) | PA-MPJPE↓ (mm) | FLOPs (M) |
|------|------------|----------------|-----------|
| WiPose-Baseline | 72.1 | 55.8 | 842 |
| Person-in-WiFi | 63.5 | 48.2 | 2,015 |
| HPE-Li (Ours) | **54.8** | **41.5** | **168** |

### Ablation Study

| Configuration | MPJPE↓ (mm) | FLOPs (M) | Description |
|------|------------|-----------|------|
| Full HPE-Li | 49.6 | 168 | Full model |
| Standard 3×3 convolution replaces Dual-SKA | 58.3 | 152 | Loss of adaptability, +8.7mm |
| Single-branch SKA (3×3 only) | 55.1 | 159 | Lack of multi-scale, +5.5mm |
| Single-branch SKA (5×5 only) | 56.8 | 163 | Large kernel redundancy, +7.2mm |
| w/o Knowledge Distillation | 56.4 | 168 | Insufficient learning, +6.8mm |
| w/o Bone length constraint | 52.1 | 168 | Inconsistent joint topology, +2.5mm |

### Key Findings

- Dual-SKA is the core performance driver. Removing it increases MPJPE by 8.7mm (17.5%), while only saving 16M FLOPs (10.5%), demonstrating high cost-effectiveness.
- Synergy between the two branches performs far better than a single branch—single-branch 3×3 and single-branch 5×5 are 5.5mm and 7.2mm worse than the dual-branch setup, respectively, confirming the necessity of dynamic kernel selection.
- Knowledge distillation provides an improvement of 6.8mm, suggesting that prior knowledge from the visual modality is highly important for WiFi HPE.
- The computational load of HPE-Li is only 8.3% of Person-in-WiFi (168M vs 2015M), yet its accuracy is 7.7mm higher.
- Analysis of different motion types shows that the 5×5 branch weight is higher in rapid movements while the 3×3 branch weight is higher in fine-grained movements, validating the effectiveness of the adaptive selection mechanism.

## Highlights & Insights

- **Pareto optimality of efficiency-accuracy**: HPE-Li simultaneously outperforms all baselines in both accuracy and efficiency, which is highly rare in the HPE field. The core lies in the Dual-SKA design—the added computation is virtually negligible (<5%), yet the performance gain is significant (17.5%).
- **Dynamic kernel selection mechanism** is tailored specifically for the characteristics of WiFi CSI (where multi-scale motion information is distributed across different subcarriers), making it much more efficient than general attention mechanisms.
- **Multimodal knowledge distillation strategy** completely frees the inference stage from reliance on cameras, truly realizing privacy-preserving WiFi-only HPE.

## Limitations & Future Work

- The upper bound of WiFi HPE accuracy is restricted by the physical resolution of CSI; even with superior models, it is difficult to match the accuracy level of visual methods.
- The current method assumes only a single person in the environment. In multi-person scenarios, CSI signals can interfere with each other, requiring additional signal separation mechanisms.
- Training relies on paired vision-WiFi data, which has high collection costs. Better self-supervised or weakly-supervised training strategies are worth exploring.
- Environmental transfer capability has not been fully verified—WiFi signals are highly dependent on indoor layout, and whether training data needs to be recollected in new environments is a critical issue for practical deployment.
- The impact of denser subcarrier configurations under WiFi 6/7 standards on accuracy has not been explored.

## Related Work & Insights

- **vs WiPose**: An early WiFi HPE baseline. Using standard CNNs fails to capture multi-scale features, resulting in an MPJPE 18.8mm higher than HPE-Li.
- **vs Person-in-WiFi**: Uses a heavy Transformer architecture. While its accuracy is better than simple CNNs, the computational cost reaches 2015M FLOPs. HPE-Li surpasses its accuracy using only 1/12 of the computation.
- **vs MetaFi**: Introduces meta-learning to achieve cross-domain adaptation, but the efficiency of its underlying architecture is suboptimal, lagging behind HPE-Li by 12.1mm on MM-Fi.
- **Insight**: The concept of selective kernel attention can be transferred to other RF-signal-based perception tasks (such as WiFi action recognition, mmWave gesture detection), which similarly face the challenges of multi-scale signal features.

## Rating

- Novelty: ⭐⭐⭐⭐ Dual-SKA is an effective adaptation of SKNet, though the core concept is not entirely brand new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on two benchmark datasets with detailed ablation and efficiency analyses, but lacks experiments on cross-environment generalization.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, complete methodology description, and good chart quality.
- Value: ⭐⭐⭐⭐ Directly drives the practical deployment of WiFi HPE.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] WorldPose: A World Cup Dataset for Global 3D Human Pose Estimation](worldpose_a_world_cup_dataset_for_global_3d_human_pose_estimation.md)
- [\[ECCV 2024\] Occlusion Handling in 3D Human Pose Estimation with Perturbed Positional Encoding](occlusion_handling_in_3d_human_pose_estimation_with_perturbed_positional_encodin.md)
- [\[ECCV 2024\] RePOSE: 3D Human Pose Estimation via Spatio-Temporal Depth Relational Consistency](repose_3d_human_pose_estimation_via_spatio-temporal_depth_relational_consistency.md)
- [\[ECCV 2024\] 3DSA: Multi-view 3D Human Pose Estimation With 3D Space Attention Mechanisms](3dsa_multi-view_3d_human_pose_estimation_with_3d_space_attention_mechanisms.md)
- [\[ECCV 2024\] UPose3D: Uncertainty-Aware 3D Human Pose Estimation with Cross-View and Temporal Cues](upose3d_uncertainty-aware_3d_human_pose_estimation_with_cross-view_and_temporal_.md)

</div>

<!-- RELATED:END -->

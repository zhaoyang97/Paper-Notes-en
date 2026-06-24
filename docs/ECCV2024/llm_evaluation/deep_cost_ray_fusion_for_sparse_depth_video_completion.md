---
title: >-
  [Paper Note] Deep Cost Ray Fusion for Sparse Depth Video Completion
description: >-
  [ECCV 2024][LLM Evaluation][depth completion] This paper proposes the RayFusion framework, which achieves temporal fusion by applying self-attention and cross-attention along the ray direction on the cost volume. With only 1.15M parameters, it outperforms or matches state-of-the-art sparse depth completion methods across three datasets: KITTI, VOID, and ScanNetV2.
tags:
  - "ECCV 2024"
  - "LLM Evaluation"
  - "depth completion"
  - "cost volume fusion"
  - "ray-wise attention"
  - "RGB-D video"
  - "temporal fusion"
date: 2026-05-08
content_hash: fd840b3a654506d4
---

# Deep Cost Ray Fusion for Sparse Depth Video Completion

**Conference**: ECCV 2024  
**arXiv**: [2409.14935](https://arxiv.org/abs/2409.14935)  
**Code**: None  
**Area**: LLM Evaluation  
**Keywords**: depth completion, cost volume fusion, ray-wise attention, RGB-D video, temporal fusion

## TL;DR

This paper proposes the RayFusion framework, which achieves temporal fusion by applying self-attention and cross-attention along the ray direction on the cost volume. With only 1.15M parameters, it outperforms or matches state-of-the-art sparse depth completion methods across three datasets: KITTI, VOID, and ScanNetV2.

## Background & Motivation

**Background**: Depth sensors (such as LiDAR, RealSense, etc.) have been widely deployed in mobile devices and autonomous driving scenarios, but the depth maps they capture are often sparse or suffer from large missing areas.

**Limitations of Prior Work**: Mainstream depth completion methods only utilize single-frame RGB-D images, focusing on extracting multi-modal features while ignoring the rich temporal information in video sequences. The few methods that utilize multiple frames (such as ConvLSTM and spatio-temporal convolutions) require warping the feature map of the previous frame to align with the current frame. However, this alignment relies on the depth prediction of the previous frame, and any prediction errors will propagate.

**Key Challenge**: Direct temporal fusion on 2D feature maps is susceptible to depth prediction errors, whereas performing global attention fusion on 3D cost volumes is completely infeasible due to the prohibitive memory footprint of $D^2H^2W^2$.

**Key Insight**: It is observed that the features along each ray in the cost volume represent the probability distribution over the depth hypothesis planes. These distributions inherently contain exploitable internal properties (such as entropy). Therefore, performing attention along the ray direction can reduce the computational complexity from $D^2H^2W^2$ to $D^2HW$.

**Core Idea**: To apply self-attention (utilizing the intrinsic features of probability distributions) and cross-attention (for cross-frame fusion) on the ray dimension of the 3D cost volume, achieving highly efficient and accurate temporal cost volume fusion.

## Method

### Overall Architecture

Input RGB-D video sequence $\rightarrow$ each frame generates a cost volume $\mathbf{V}_t$ through the Cost Volume Creation module $C_\theta$ $\rightarrow$ fused with the updated cost volume of the previous frame $\mathbf{V}'_{t-1}$ via the Ray-based Fusion module $F_\theta$ $\rightarrow$ the fused cost volume $\mathbf{V}'_t$ regresses depth through the Depth Regression module $R_\theta$ $\rightarrow$ finally refined via the NLSPN depth refinement module $H_\theta$ to output the final depth map.

### Key Designs

1. **Cost Volume Creation ($C_\theta$)**:

    - **Function**: Constructs a 3D cost volume from a single-frame RGB-D image.
    - **Mechanism**: Constructs three types of feature volumes—occupancy volume $\mathbf{V}_o$, residual volume $\mathbf{V}_r$ (from sparse depth), and RGB feature volume $\mathbf{V}_i$ (from multi-scale image features). These are concatenated and fed into a 3D convolutional U-Net to infer the cost volume. The voxels are established on $D$ uniformly sampled depth hypothesis planes, $\mathbf{V} \in \mathbb{R}^{D \times C \times H \times W}$.
    - **Design Motivation**: An improved version based on CostDCNet, which discards the independent geometric feature extractor and incorporates multi-scale image features, enabling the cost volume to encode both RGB appearance and sparse depth priors simultaneously.

2. **Ray-based Fusion ($F_\theta$)**:

    - **Function**: Fuses the cost volumes of the current and previous frames.
    - **Mechanism**:
        - First, aligns $\mathbf{V}'_{t-1}$ to the current view coordinate system using inverse warping with relative poses.
        - For each pixel position $(h,w)$, extracts ray features $\mathbf{F}_t = \mathbf{V}_t(:,:,h,w) \in \mathbb{R}^{D \times C}$, treating the $D$ depth hypotheses as $D$ tokens.
        - First aggregates local spatial information via two layers of 3D convolutions.
        - Applies **self-attention** to each ray individually: $\mathbf{SA}_t = \text{Attn}(\mathbf{F}_t, \mathbf{F}_t, \mathbf{F}_t)$, utilizing the intrinsic properties of the probability distribution (such as entropy/uncertainty) to refine the depth hypothesis of the current frame.
        - Then applies **cross-attention**: $\mathbf{CA}_t = \text{Attn}(\mathbf{SA}_t, \mathbf{SA}_{t-1}, \mathbf{SA}_{t-1})$ to achieve cross-frame fusion.
        - Adds sinusoidal positional encodings to inject relative position information of the depth plane indices.
    - **Design Motivation**: Attention along the ray direction only requires $D^2HW$ attention entries, which is significantly more efficient than the $D^2H^2W^2$ entries required by global volume attention. Self-attention perceives intrinsic properties like the entropy of the probability distribution, while cross-attention leverages temporal information accumulated from previous frames.

3. **Depth Regression & Refinement ($R_\theta, H_\theta$)**:

    - **Function**: Regresses dense depth maps from the fused cost volume.
    - **Mechanism**: The fused cost volume is converted into a probability volume $\mathbf{P}'_t$ via 3D convolutions and pixel shuffle. After softmax normalization, the depth is obtained via weighted summation:
    $\mathbf{D}_t(h,w) = \sum_{i=1}^{D} d_i \times \mathbf{p}^i_{h,w}$
    - Finally, refines depth in the image domain using NLSPN (Non-Local Spatial Propagation Network).
    - **Design Motivation**: The probability volume regression approach allows simultaneous output of depth and confidence, providing additional guidance for subsequent refinement.

### Loss & Training

- **$L_1$ Depth Loss**: $\mathcal{L}_{L1} = \frac{1}{|\mathbb{P}|}\sum|D_t - D_{gt}|$
- **Cross-Entropy Loss**: $\mathcal{L}_{CE} = -\mathbf{p}_{gt}^T \log \mathbf{p}$, using **soft labels** (finding the two hypothesis planes closest to the GT depth and calculating normalized weights) rather than one-hot hard labels.
- **Total Loss**: $\mathcal{L}_{total} = \mathcal{L}_{L1} + \mathcal{L}_{CE}$
- Training uses the AdamW optimizer with a batch size of 4 on 3 × RTX 3090 GPUs. The number of depth hypothesis planes is $D=16$, and the image is downsampled by a factor of 4 to process the cost volume.

## Key Experimental Results

### Main Results

| Dataset | Metric | RayFusion (1.15M) | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| ScanNetV2 | MAE (m) | **0.0160** | 0.0244 (CostDCNet, 1.8M) | -34.4% |
| ScanNetV2 | RMSE (m) | **0.0554** | 0.0759 (CostDCNet, 1.8M) | -27.0% |
| ScanNetV2 | F-score↑ | **0.9161** | 0.8998 (CostDCNet) | +1.6% |
| VOID (0.5%) | MAE (mm) | **24.51** | 25.84 (CostDCNet, 1.8M) | -5.1% |
| VOID (0.5%) | RMSE (mm) | **65.46** | 76.28 (CostDCNet) | -14.2% |
| KITTI | MAE (mm) | **176.23** | 188.80 (LRRU, 21.0M) | -6.7% |
| KITTI | RMSE (mm) | **720.63** | 729.50 (LRRU) | -1.2% |

### Ablation Study (KITTI Validation Set, 20% Training Data)

| Configuration | MAE↓ | RMSE↓ | Description |
|------|------|-------|------|
| A / $L_1$ only | 203.73 | 855.07 | Baseline: No fusion, $L_1$ loss only |
| A / $L_{CE}+L_1$ | 203.05 | 834.09 | CE loss brings improvements in MAE/RMSE balance |
| A+B2 (GRU) / $L_{CE}$ | 228.85 | 799.03 | Local convolutional fusion degrades MAE |
| A+B1 (Ray) / $L_{CE}+L_1$ | **198.51** | 777.37 | Ray attention comprehensively outperforms GRU |
| A+B1+C (full) | **188.88** | **768.11** | Incorporating NLSPN yields further improvements |

### Key Findings

- Using only self-attention (single-frame mode) already achieves SOTA on VOID and ScanNetV2, suggesting that leveraging the intrinsic properties of the cost volume is highly valuable on its own.
- The model has only 1.15M parameters, which is 94.5% smaller than LRRU (21.0M) and 98.6% smaller than ComplFormer (83.5M).
- Cross-dataset generalization capability is prominent: Trained on ScanNetV2 and tested on VOID, the MAE is only 29.08mm, which is significantly better than NLSPN (158.60mm) and ComplFormer (65.90mm).
- High robustness to varying sparsities: Trained with 0.5% sparsity and tested on 0.05% sparsity, the performance decay is far less severe than competing methods.

## Highlights & Insights

- **Ray-level attention represents an extremely clever design trade-off**: it captures global information (interactions among all depth hypotheses along the ray direction) while avoiding the prohibitive overhead of global 3D volume attention.
- **Self-attention implicitly leverages the entropy of the probability distribution as an uncertainty prior**, a finding that can be transferred to other tasks handling probability volumes or cost volumes.
- **Cross-entropy loss combined with soft labels** for supervising the cost volume probability distribution is more stable than the traditional $L_1+L_2$ loss combination. This training technique is highly referenceable.
- **Extreme parameter efficiency**: achieves better performance than methods with 21-83M parameters using only 1.15M parameters, demonstrating that elegant architectural design is superior to brute-force parameter scaling.

## Limitations & Future Work

- **High memory footprint**: 3D convolutions and the cost volume themselves consume large amounts of GPU memory, limiting the resolution and the number of depth hypothesis planes.
- **Difficulty in correcting persistent prediction errors**: if a region yields failed predictions across consecutive frames, temporal fusion is unable to rectify the error.
- **Uniform sampling of depth hypothesis planes**: inflexible for near and far scenes; adaptive sampling strategies could be considered.
- **Incomplete exploration of stronger 3D backbones**: utilizing Sparse 3D Convolution or 3D Gaussian representations might further enhance efficiency.

## Related Work & Insights

- **vs CostDCNet**: RayFusion introduces temporal fusion on top of CostDCNet's single-frame cost volume and discards the independent geometric feature extractor, achieving better performance with fewer parameters.
- **vs LRRU**: LRRU achieves near SOTA on KITTI with 21.0M parameters, whereas RayFusion outperforms it with 94.5% fewer parameters.
- **vs DeepVideoMVS**: Both leverage video sequences for depth estimation, but RayFusion performs fusion in the 3D cost volume space instead of doing 2D feature map warping, thereby eliminating depth prediction error propagation.
- **vs SimpleRecon (MVS)**: MVS methods do not utilize sparse depth priors, which may yield pleasing visual effects but suffer from poorer quantitative metrics.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of ray-level attention on the cost volume is simple and effective, though the combination of cost volume and attention is not entirely novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Conducted on three datasets (indoor/outdoor/different sensors) with comprehensive ablation and generalization studies.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with intuitive illustrations and formal mathematical equations.
- Value: ⭐⭐⭐⭐ Extreme parameter efficiency and SOTA performance across multiple datasets; the ray-level attention mechanism is highly transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] OGNI-DC: Robust Depth Completion with Optimization-Guided Neural Iterations](ogni-dc_robust_depth_completion_with_optimization-guided_neural_iterations.md)
- [\[ECCV 2024\] ColorMNet: A Memory-based Deep Spatial-Temporal Feature Propagation Network for Video Colorization](colormnet_a_memory-based_deep_spatial-temporal_feature_propagation_network_for_v.md)
- [\[ECCV 2024\] SIGMA: Sinkhorn-Guided Masked Video Modeling](sigma_sinkhorn-guided_masked_video_modeling.md)
- [\[ECCV 2024\] Eliminating Warping Shakes for Unsupervised Online Video Stitching](eliminating_warping_shakes_for_unsupervised_online_video_stitching.md)
- [\[ICLR 2026\] SparseEval: Efficient Evaluation of Large Language Models by Sparse Optimization](../../ICLR2026/llm_evaluation/sparseeval_efficient_evaluation_of_large_language_models_by_sparse_optimization.md)

</div>

<!-- RELATED:END -->

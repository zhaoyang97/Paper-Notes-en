---
title: >-
  [Paper Note] Temporal Action Detection Model Compression by Progressive Block Drop
description: >-
  [CVPR 2025][Autonomous Driving][Temporal Action Detection] A Progressive Block Drop method is proposed to compress Temporal Action Detection (TAD) models from the depth dimension. By progressively removing redundant blocks and utilizing a parameter-efficient cross-depth alignment strategy to recover performance, this method achieves a 25% computation reduction with no performance degradation, and even exhibits performance gains.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Temporal Action Detection"
  - "Model Compression"
  - "Progressive Block Drop"
  - "Depth Pruning"
  - "LoRA"
date: 2026-05-08
content_hash: d599cdf420b46f03
---

# Temporal Action Detection Model Compression by Progressive Block Drop

**Conference**: CVPR 2025  
**arXiv**: [2503.16916](https://arxiv.org/abs/2503.16916)  
**Code**: None  
**Area**: Autonomous Driving/Video Understanding  
**Keywords**: Temporal Action Detection, Model Compression, Progressive Block Drop, Depth Pruning, LoRA

## TL;DR

A Progressive Block Drop method is proposed to compress Temporal Action Detection (TAD) models from the depth dimension. By progressively removing redundant blocks and utilizing a parameter-efficient cross-depth alignment strategy to recover performance, this method achieves a 25% computation reduction with no performance degradation, and even exhibits performance gains.

## Background & Motivation

Temporal Action Detection (TAD) aims to identify and localize the start and end times of action instances in untrimmed videos, serving as the foundation for applications like video question answering and video captioning. With the utilization of larger feature extractors and datasets, the computational demands of TAD models have increased significantly, limiting their deployment in resource-constrained scenarios like autonomous driving and robotics.

- **Computational bottleneck in the feature extractor**: The feature extraction phase accounts for **95%** of the total computation because it requires processing the entire video segment-by-segment using sliding windows.
- **Channel pruning is unfavorable for GPU parallelism**: Traditional channel pruning (width reduction) shrinks weight matrices, but small matrix multiplication has low parallel efficiency on GPUs.
- **Deep and narrow vs. shallow and wide**: Research shows that, under the same parameter count, shallow-and-wide networks achieve faster inference than deep-and-narrow networks.
- **Existence of block-level redundancy**: Experiments reveal that some blocks show minimal difference between input and output features (MSE close to 0), and removing a single block has negligible impact on performance.
- **Direct drop of multiple blocks leads to severe performance degradation**: Removing multiple blocks simultaneously results in significant performance drops, necessitating a progressive strategy.

## Method

### Overall Architecture

Progressive Block Drop adopts a multi-step iterative compression strategy. Each iteration consists of two phases: (1) a block selection evaluator automatically selects and removes the block with the least impact; and (2) a parameter-efficient cross-depth alignment strategy recovers model performance through LoRA fine-tuning. The iteration proceeds until the compressed model's performance can no longer be restored to the level of the uncompressed model.

### Key Designs

#### Block Selection Evaluator — Automatic Selection of Droppable Redundant Blocks

**Function**: Evaluates the importance of each block and automatically selects the block with the minimum impact on performance for removal.

**Mechanism**: In the $t$-th iteration, it traverses all $K$ blocks of the current model $M_{t-1}$ to construct subnetworks $\mathcal{S}_t = \{M_{t,k}^{sub} = M_{t-1} \setminus b_k\}$ by dropping each block individually. An evaluation function $f_E$ is used on the training set to measure the performance discrepancy between each subnetwork and the uncompressed model $M_0$, selecting the subnetwork with the smallest performance gap (i.e., dropping the block with the least impact on performance).

**Design Motivation**: Inspired by Curriculum Learning, redundancy is progressively removed from easy to hard, rather than discarding multiple blocks at once. Experimental results verify that selection based on the mAP metric yields the best results.

#### Parameter-Efficient Cross-Depth Alignment — LoRA + Feature/Prediction Dual-Level Distillation

**Function**: Restores the performance of the compressed model through low-cost training after a block is removed.

**Mechanism**: LoRA parameters $\theta_{\text{LoRA}}$ are inserted into the attention layers of each remaining block, and only these newly added parameters are trained. A cross-depth alignment loss is proposed. When the $i$-th block is dropped, the output features of the remaining blocks between the compressed model and the uncompressed model are aligned:

$$\mathcal{L}_f = \frac{1}{I-1} \sum_{m \neq i} (f_{b_m}^{M_0} - f_{b_m}^{\hat{M}_t})^2$$

Meanwhile, the prediction layers are aligned: KL divergence $\mathcal{L}_{pc}$ for classification and GIoU loss $\mathcal{L}_{pr}$ for localization. The total loss is:

$$\mathcal{L}_{total} = \mathcal{L}_{pc} + \mathcal{L}_{pr} + \mathcal{L}_f + \mathcal{L}_{cls} + \mathcal{L}_{reg}$$

**Design Motivation**: (1) Training only the LoRA parameters drastically reduces GPU memory consumption (e.g., full-parameter fine-tuning of VideoMAE-L requires 13.1GB/batch); (2) after training, LoRA parameters can be merged into the original parameters, introducing zero inference overhead; (3) experiments demonstrate that ground-truth (GT) supervision alone fails to recover performance, making feature-level alignment essential.

#### Hardware-Friendliness of Depth Reduction — Shallow & Wide Outperforms Deep & Narrow

**Function**: Orthogonal to traditional channel pruning methods, and can be combined to achieve higher compression ratios.

**Mechanism**: Reducing network depth while keeping layer width constant yields a shallow-and-wide architecture that performs faster inference on GPUs. Under the same MACs, depth reduction achieves a higher real-world speedup than width reduction.

**Design Motivation**: The parallel efficiency of large matrix operations during GPU inference is significantly higher than that of multiple small matrix operations. Although channel pruning reduces FLOPs, smaller matrices decrease GPU utilization.

### Loss & Training

$$\mathcal{L}_{total} = \mathcal{L}_{pc} + \mathcal{L}_{pr} + \mathcal{L}_f + \mathcal{L}_{cls} + \mathcal{L}_{reg}$$

It includes classification alignment (KL divergence), localization alignment (GIoU), cross-depth feature alignment (MSE), and standard classification and regression losses.

## Key Experimental Results

### Main Results: VideoMAE-S (12 blocks) + ActionFormer

| Block Drops | MACs (G) | Inference Time | THUMOS14 mAP@0.5 | ActivityNet mAP |
|---------|----------|---------|-----------------|----------------|
| 0 (Baseline) | 286.3 | 104.9ms | 70.43 | 37.75 |
| 1 | 263.5 (92%) | 98.4ms | 71.06 (+0.63) | 37.94 (+0.19) |
| 2 | 240.8 (84%) | 89.8ms | 71.37 (+0.94) | 37.81 (+0.06) |
| 3 | 218.0 (76%) | 81.2ms | 70.47 (+0.04) | 37.77 (+0.02) |
| 4 | 195.2 (68%) | 73.6ms | 69.65 (-0.78) | 37.72 (-0.03) |

### Deeper Network: VideoMAE-L (24 blocks)

| Block Drops | MACs | THUMOS14 mAP@0.5 |
|---------|------|-----------------|
| 0 (Baseline) | 3886.9G | 76.01 |
| 3 | 3402.7G (87.5%) | **78.29 (+2.28)** |
| 6 | 2918.6G (75.1%) | 77.25 (+1.24) |

### Inference Speed Comparison (Under Same MACs)

| Method | MACs (G) | Inference Time | Speedup |
|------|----------|---------|-------|
| Uncompressed | 286.3 | 104.9ms | 1.00× |
| Channel Pruning | 218.2 | 96.6ms | 1.09× |
| **Block Drop (Ours)** | 218.0 | 81.2ms | **1.29×** |

### Combination with Channel Pruning

At 70.2% mAP accuracy, Block Drop + Channel Pruning reduces MACs by **30%**, whereas Channel Pruning alone reduces MACs by only **10%**.

### Key Findings

- Dropping 3 blocks (25% reduction in MACs) yields **improved rather than degraded** performance on THUMOS14.
- Deeper models (24 layers) exhibit more redundancy, achieving a performance improvement of **+1.24%** under 25% compression (vs. +0.04% for the 12-layer model).
- Progressive dropping outperforms one-time dropping (70.47% vs. 68.91%), validating the necessity of the progressive strategy.
- The method generalizes across different architectures (AdaTAD, ActionFormer), datasets (FineAction), and tasks (natural language localization).

## Highlights & Insights

1. **Compressing TAD models from the depth dimension offers a new perspective**: Prior works mainly focused on channel pruning. This paper systematically explores block-level depth compression for the first time.
2. **Improvement in performance after compression**: This counter-intuitive result demonstrates the presence of significant depth redundancy within TAD models.
3. **Orthogonality to channel pruning**: The two methods can be combined to achieve higher compression ratios, indicating strong practical utility.
4. **Compelling hardware-friendliness analysis**: The speedup comparison of 1.29× vs. 1.09× under equivalent MACs intuitively demonstrates the advantages of a shallow-and-wide architecture.

## Limitations & Future Work

- Currently, the approach is only validated on the VideoMAE Transformer architecture; the applicability to CNN backbones remains unexplored.
- Block selection requires evaluating all candidate subnetworks on the training set, causing the selection overhead to grow linearly with the number of blocks.
- The impact of LoRA rank selection on performance recovery is not thoroughly discussed.
- Strategies for automatically determining the optimal compression ratio can be explored, rather than relying on manually defined stopping conditions.
- Combining this approach with other compression technologies such as knowledge distillation and quantization might further improve compression performance.

## Related Work & Insights

- **ActionFormer / AdaTAD**: Mainstream TAD detection head architectures.
- **VideoMAE**: Video feature extractor based on masked autoencoders, which is the compression target of this paper.
- **LoRA**: A parameter-efficient fine-tuning method, cleverly utilized here for performance recovery after compression.
- **Curriculum Learning**: The methodology of progressive learning inspired the step-by-step block dropping strategy.

## Rating

⭐⭐⭐⭐ — Clear problem definition (95% of computation lies in the feature extractor), simple and effective methodology (progressive dropping + LoRA recovery), and comprehensive experiments (multiple architectures, datasets, tasks, and combination with pruning). The performance improvement after compression is highly convincing, and the hardware-friendliness analysis is a major plus.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] EV-3DOD: Pushing the Temporal Boundaries of 3D Object Detection with Event Cameras](ev-3dod_pushing_the_temporal_boundaries_of_3d_object_detection_with_event_camera.md)
- [\[CVPR 2025\] RENO: Real-Time Neural Compression for 3D LiDAR Point Clouds](reno_real-time_neural_compression_for_3d_lidar_point_clouds.md)
- [\[ICCV 2025\] SparseLaneSTP: Leveraging Spatio-Temporal Priors with Sparse Transformers for 3D Lane Detection](../../ICCV2025/autonomous_driving/sparselanestp_leveraging_spatio-temporal_priors_with_sparse_transformers_for_3d_.md)
- [\[CVPR 2026\] Counterfactual VLA: Self-Reflective Vision-Language-Action Model with Adaptive Reasoning](../../CVPR2026/autonomous_driving/counterfactual_vla_self-reflective_vision-language-action_model_with_adaptive_re.md)
- [\[CVPR 2026\] NoRD: A Data-Efficient Vision-Language-Action Model that Drives without Reasoning](../../CVPR2026/autonomous_driving/nord_a_data-efficient_vision-language-action_model_that_drives_without_reasoning.md)

</div>

<!-- RELATED:END -->

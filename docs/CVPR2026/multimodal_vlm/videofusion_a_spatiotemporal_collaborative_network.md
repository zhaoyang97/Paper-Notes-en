---
title: >-
  [Paper Note] VideoFusion: A Spatio-Temporal Collaborative Network for Multi-modal Video Fusion
description: >-
  [CVPR 2026][Multimodal VLM][multi-modal video fusion] This paper introduces the M3SVD large-scale infrared-visible video dataset (220 videos / 150K frames) and proposes the VideoFusion framework, which achieves spatio-temporal collaborative multi-modal video fusion via a Cross-modal Differential Reinforcement Module (CmDRM), Complete Modal Guided Fusion (CMGF), Bidirectional Co-Attention Module (BiCAM), and a variational consistency loss. The method surpasses existing image fusion and video fusion approaches in both fusion quality and temporal consistency.
tags:
  - CVPR 2026
  - Multimodal VLM
  - multi-modal video fusion
  - infrared-visible
  - temporal consistency
  - cross-modal
  - dataset
date: 2026-05-08
content_hash: 88ce34ba404f0b1c
---

# VideoFusion: A Spatio-Temporal Collaborative Network for Multi-modal Video Fusion

**Conference**: CVPR 2026
**arXiv**: [2503.23359](https://arxiv.org/abs/2503.23359)
**Code**: [https://github.com/Linfeng-Tang/VideoFusion](https://github.com/Linfeng-Tang/VideoFusion)
**Area**: Image Fusion / Video Processing
**Keywords**: multi-modal video fusion, infrared-visible, temporal consistency, cross-modal, dataset

## TL;DR
This paper introduces the M3SVD large-scale infrared-visible video dataset (220 videos / 150K frames) and proposes the VideoFusion framework, which achieves spatio-temporal collaborative multi-modal video fusion via a Cross-modal Differential Reinforcement Module (CmDRM), Complete Modal Guided Fusion (CMGF), Bidirectional Co-Attention Module (BiCAM), and a variational consistency loss. The method surpasses existing image fusion and video fusion approaches in both fusion quality and temporal consistency.

## Background & Motivation
Multi-sensor fusion research has primarily focused on static image fusion, with a large body of methods employing AE/CNN/GAN/Transformer/Diffusion architectures for infrared-visible image fusion. However, sensors in real-world scenarios capture continuous video streams; applying image fusion methods frame-by-frame ignores inter-frame temporal dependencies, leading to flickering artifacts and temporal inconsistency. The main bottlenecks are the lack of large-scale temporally aligned multi-modal video datasets (existing benchmarks such as TNO contain only 3 videos/114 frames, INO suffers from low resolution, and HDO from poor quality), and the challenge of jointly modeling spatial and temporal dependencies within a unified framework.

## Core Problem
How to simultaneously exploit cross-modal complementary information and inter-frame temporal dependencies in multi-modal video fusion to generate high-quality, temporally consistent fused videos?

## Method

### Overall Architecture
An encoder-decoder architecture is adopted. During encoding: 3D convolutions extract shallow temporal features → downsampling + ResBlocks + CmDRM extract multi-scale enhanced features → CMGF aggregates cross-modal context at multiple scales. During decoding: Transformer enhancement blocks → BiCAM establishes cross-frame temporal dependencies → a fusion decoder with a modal separation module reconstructs the fused video and the restored infrared/visible modality videos respectively.

### Key Designs
1. **Cross-modal Differential Reinforcement Module (CmDRM)**: The core idea is to extract cross-modal differential information (complementary rather than redundant). A differential feature $\mathcal{F}_d^t = \mathcal{F}_{ir}^t - \mathcal{F}_{vi}^t$ is computed and used as Key/Value, while the original modality features serve as Query for cross-attention. A learnable contribution metric then adaptively fuses the original and differential-enhanced features, followed by channel and spatial attention refinement.

2. **Complete Modal Guided Fusion (CMGF)**: The sum of dual-modality features $\mathcal{F}_c^t = \hat{\mathcal{F}}_{ir}^t + \hat{\mathcal{F}}_{vi}^t$ serves as a shared Query to extract modality-specific information from infrared and visible features respectively, enabling guided cross-modal aggregation.

3. **Bidirectional Co-Attention Module (BiCAM)**: The current frame feature acts as a shared Query, while features from the previous and next frames generate K/V pairs. After computing forward and backward attention, bidirectional temporal dynamics are fused via $\mathcal{A}_{co} = \text{softmax}(\mathcal{A}_{t-1} * \mathcal{A}_{t+1})$. Analogous to the shifted window mechanism in Swin Transformer, stacking $N$ BiCAM modules allows the model to indirectly capture long-range temporal context.

4. **Variational Consistency Loss $\mathcal{L}_{var}$**: Based on the assumption that inter-frame variation in static backgrounds should approach zero, while inter-frame variation in dynamic objects should remain consistent with the source video. The loss constrains inter-frame differences in both the fused video and the restored modality videos to be consistent with the high-quality source video.

### Loss & Training
- $\mathcal{L}_{int}$: Intensity loss, preserving salient targets from the source video
- $\mathcal{L}_{grad}$: Gradient loss, preserving textural details
- $\mathcal{L}_{color}$: CbCr color loss, maintaining color fidelity
- $\mathcal{L}_{sf}$: Scene fidelity loss, constraining the quality of modal separation reconstruction
- $\mathcal{L}_{var}$: Variational consistency loss, suppressing temporal flickering

## Key Experimental Results

| M3SVD (Degraded) | MI↑ | SSIM↑ | VIF↑ | flowD↓ |
|---|---|---|---|---|
| U2Fusion | 2.490 | 0.600 | 0.439 | 6.547 |
| TemCoCo | 3.548 | 0.597 | 0.490 | 4.378 |
| **VideoFusion** | **4.008** | **0.632** | **0.526** | **3.294** |

| M3SVD (Normal) | MI↑ | SSIM↑ | VIF↑ | flowD↓ |
|---|---|---|---|---|
| TC-MoA | 2.894 | 0.602 | 0.577 | 5.305 |
| TemCoCo | 3.548 | 0.597 | 0.490 | 4.379 |
| **VideoFusion** | **4.191** | **0.646** | **0.605** | **3.494** |

- **Temporal consistency**: VideoFusion achieves the lowest flowD (3.294/3.494); frame-by-frame methods such as DDFM/LRRNet exhibit flowD > 6.
- **Efficiency**: 6.743M parameters, 267.78G FLOPs, 0.067s/frame — comparable to image fusion methods.
- **Downstream task**: YOLO v11 detects more objects and produces smoother trajectories on VideoFusion outputs.

### Ablation Study
- Removing BiCAM: flowD increases from 3.294 → 4.747, with significant degradation in temporal consistency.
- Removing CmDRM: information recovery capability declines; MI drops from 4.008 → 3.557.
- Removing CMGF (replaced by simple addition): SSIM drops from 0.632 → 0.366, causing severe structural distortion.
- Removing $\mathcal{L}_{var}$: flowD increases from 3.294 → 6.056.
- Removing $\mathcal{L}_{color}$: pronounced color distortion is observed.

## Highlights & Insights
- **The M3SVD dataset** is a major contribution: 220 videos / 150K frames / 100 scenes / 4 types of challenging scenarios — an order of magnitude larger than the previously largest dataset HDO (24 videos / 7,500 frames).
- The bidirectional co-attention and stacking design of BiCAM is elegant and effective, requiring neither optical flow nor DCN for inter-frame alignment.
- The static/dynamic decomposition assumption underlying the variational consistency loss is well-motivated.
- The modal separation (unmixing) branch simultaneously yields restored modality outputs — a single model performs both fusion and degradation restoration.
- CmDRM uses differential information rather than raw cross-modal features for attention, effectively reducing redundancy.

## Limitations & Future Work
- Training is limited to $T=7$ frames due to GPU memory constraints; larger temporal windows may further improve performance.
- BiCAM only attends to adjacent frames (±1), which may be insufficient for fast-motion scenes — although stacking can indirectly enlarge the temporal receptive field.
- The resolution (640×480) and frame rate (30 fps) of M3SVD are relatively modest.
- Validation is limited to infrared-visible fusion; other modalities such as multispectral and SAR remain unexplored.
- Quantitative metrics (EN/MI/SD/SSIM/VIF) primarily assess pixel-level quality, lacking semantic-level evaluation.

## Related Work & Insights
- **vs. TemCoCo**: TemCoCo employs DCN for inter-frame compensation, exhibiting poor generalization on multi-modal data; VideoFusion uses attention mechanisms for adaptive aggregation and outperforms TemCoCo across MI/VIF/flowD.
- **vs. RCVS**: RCVS relies on handcrafted features for temporal modeling; VideoFusion performs end-to-end learning, achieving superior fusion quality and temporal consistency.
- **vs. image fusion methods (SwinFusion/DDFM, etc.)**: Frame-by-frame application causes temporal flickering (flowD > 5); VideoFusion significantly reduces this to 3.3.

The video-level fusion framework is generalizable to other multi-modal fusion scenarios (e.g., medical image sequences, remote sensing time series). The M3SVD dataset can serve as a unified benchmark for video fusion, registration, and degradation restoration.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First systematic video fusion framework with a large-scale dataset, though individual module designs are relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers two datasets, normal and degraded conditions, comprehensive ablation, efficiency analysis, downstream tasks, and temporal visualization.
- **Writing Quality**: ⭐⭐⭐⭐ Clear framework description, detailed dataset construction, and rich figures and tables.
- **Value**: ⭐⭐⭐⭐⭐ M3SVD fills a critical gap; VideoFusion advances the field from image fusion to video fusion with strong practical significance.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs](sparrow_learning_spatial_precision_and_temporal_referential_consistency_in_pixel.md)
- [\[CVPR 2026\] Multi-Modal Image Fusion via Intervention-Stable Feature Learning](multi-modal_image_fusion_via_intervention-stable_feature_learning.md)
- [\[CVPR 2026\] TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs](timelens_rethinking_video_temporal_grounding_with_multimodal_llms.md)
- [\[CVPR 2026\] CoMP: Collaborative Multi-Mode Pruning for Vision-Language Models](comp_collaborative_multi-mode_pruning_for_vision-language_models.md)
- [\[CVPR 2026\] EagleNet: Energy-Aware Fine-Grained Relationship Learning Network for Text-Video Retrieval](eaglenet_energy-aware_fine-grained_relationship_learning_network_for_text-video_.md)

<!-- RELATED:END -->

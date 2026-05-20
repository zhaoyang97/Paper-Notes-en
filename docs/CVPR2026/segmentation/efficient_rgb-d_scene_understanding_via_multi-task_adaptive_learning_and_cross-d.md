---
title: >-
  [Paper Note] Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance
description: >-
  [CVPR2026][Segmentation][RGB-D Scene Understanding] This paper proposes an efficient RGB-D multi-task scene understanding network. An improved fusion encoder exploits channel redundancy to accelerate feature extraction.…
tags:
  - "CVPR2026"
  - "Segmentation"
  - "RGB-D Scene Understanding"
  - "Multi-task Adaptive Learning"
  - "Cross-dimensional Feature Guidance"
  - "Panoptic Segmentation"
  - "Fusion Encoder"
date: 2026-05-08
content_hash: d3ac23ada7e9219a
---

# Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance

**Conference**: CVPR2026
**arXiv**: [2603.07570](https://arxiv.org/abs/2603.07570)  
**Code**: Not yet released  
**Area**: Semantic Segmentation / Panoptic Segmentation / Multi-task Learning
**Keywords**: RGB-D Scene Understanding, Multi-task Adaptive Learning, Cross-dimensional Feature Guidance, Panoptic Segmentation, Fusion Encoder

## TL;DR

This paper proposes an efficient RGB-D multi-task scene understanding network. An improved fusion encoder exploits channel redundancy to accelerate feature extraction. A Normalization-Focused Channel Layer (NFCL) and a Context Feature Interaction Layer (CFIL) provide cross-dimensional feature guidance. A batch-level multi-task adaptive loss function dynamically adjusts per-task learning weights. The unified framework simultaneously handles five tasks—semantic segmentation, instance segmentation, orientation estimation, panoptic segmentation, and scene classification—on NYUv2, SUN RGB-D, and Cityscapes, achieving advantages in both accuracy and speed.

## Background & Motivation

1. **Single-task limitations**: Conventional scene understanding methods focus on a single task and cannot support comprehensive environmental perception for robots. Multi-task learning enables synergistic optimization through information sharing, but the large disparity in task complexity makes fixed learning strategies difficult to adapt.
2. **Inefficiency of dual encoders**: Methods such as EMSANet use separate encoders for RGB and depth, failing to fully exploit complementary information. EMSAFormer employs a single Swin Transformer for joint extraction, but its matrix operations are computationally heavy and memory-access-intensive, limiting inference speed.
3. **Shallow features misguiding MLP decoders**: Lightweight MLP-based semantic decoders are fast but susceptible to noise and erroneous information in shallow encoder features, degrading local detail representation.
4. **Insufficient local–global fusion**: MLP decoders excel at global feature mapping but lack the capacity to integrate local information and multi-scale context, leading to inaccurate boundary segmentation in complex scenes.
5. **Parameter efficiency of instance decoders**: Bottleneck structures reduce parameters through dimensionality reduction at the cost of feature diversity; depthwise separable convolutions incur frequent memory access that hurts speed. A better balance between parameter efficiency and nonlinear expressiveness is needed.
6. **Fixed loss weights ill-suited to dynamic training**: Existing multi-task learning methods either assign weights randomly (causing instability) or adjust them based solely on the first batch of data (lacking real-time adaptability), and cannot dynamically accommodate changes in task importance throughout training.

## Method

### Overall Architecture

The network comprises three components: an improved fusion encoder (processing 4-channel RGBD input), a semantic decoder (incorporating NFCL and CFIL), and an instance decoder (non-bottleneck 1D architecture). Semantic segmentation provides foreground masks; instance segmentation produces instance centers and offsets; the two are combined for panoptic segmentation. Scene classification is performed by a fully connected layer. A multi-task adaptive loss function is used during training.

### Efficient Fusion Encoder

- A 4-stage structure applies 4×4 convolutions for channel expansion and downsampling at each stage, followed by multiple fusion blocks.
- Stages 1–4 contain 3, 4, 18, and 3 fusion blocks, respectively.
- **Core Idea**: Exploiting high inter-channel feature similarity, convolution is applied to only 1/4 of the channels, and the remaining channels are concatenated, reducing FLOPs to 1/16 of standard convolution.
- Two pointwise convolutions capture channel relationships via expansion followed by restoration of channel dimensions, with a residual connection.
- ImageNet pre-trained weights are reused by summing the RGB three-channel weights as the depth channel weight: $D = (R+G+B)/2$.
- Built upon the FasterNet-M backbone, reducing memory access to improve inference speed.

### Normalization-Focused Channel Layer (NFCL)

- **Objective**: Enhance channel-dimension representation in shallow encoder features to mitigate the misguiding effect of shallow-layer noise on the MLP decoder.
- Channel weights are derived by normalizing the absolute values of the BN scaling factors $\gamma$: $W_i = |\gamma_i| / \sum_j |\gamma_j|$
- Features are reshaped to $B \times H \times W \times C$, multiplied element-wise by the channel weights, passed through Sigmoid activation, and then multiplied element-wise with the original input.
- Applied at skip connections of layers 1, 2, and 3 of the semantic decoder (layer 4 encoder features are already sufficiently discriminative and require no additional guidance).

### Context Feature Interaction Layer (CFIL)

- **Objective**: Compensate for the MLP decoder's limited capacity for local–global information fusion.
- Adaptive average pooling at two scales (1×1 and 5×5) extracts multi-scale context from the input features.
- A convolutional layer compresses channels from $C$ to $C/2$; bilinear interpolation upsampling unifies spatial resolution.
- Multi-scale features and the original input are concatenated, then passed through a convolution to restore the original channel dimension.
- Applied at the multi-level feature fusion stage of the semantic decoder.

### Non-bottleneck 1D Instance Decoder

- Decomposes each 3×3 2D convolution into a 3×1 and a 1×3 1D convolution with an intermediate ReLU activation.
- With a kernel size of 3, parameter count is reduced by 30% while enhancing nonlinear decision capacity.
- The instance decoder consists of 3 layers, each comprising a 3×3 convolution, 3 non-bottleneck 1D modules, and upsampling.
- Outputs instance centers, pixel offsets, and raw orientations; pyramid supervision is applied at each layer.

### Multi-task Adaptive Loss Function

- At the end of each batch, the relative loss of each task is computed: $RL_k = L_k / \sum_t L_t$
- A running mean of historical relative losses is maintained: $AvgRL_k = \sum_i RL_k^{(i)} / n_k$
- Weights are dynamically updated: $W_k = \max(\bar{W}_k \times (AvgRL_k)^\alpha, W_{min})$
- Modulation factor $\alpha = 0.01$ (fine-grained adjustment); minimum threshold $W_{min} = 0.1$ (preventing task neglect).
- Per-task losses: cross-entropy for semantic segmentation, MSE for instance centers, MAE for instance offsets, cosine-sine probability distribution loss for orientation estimation, and cross-entropy for scene classification.

## Key Experimental Results

### Comparison with State-of-the-Art on NYUv2

| Method | Modality | Backbone | Semantic mIoU |
|--------|----------|----------|:---:|
| EMSAFormer | RGB-D | Swin v2 | 49.76 |
| MMANet | RGB-D | R34-NBt1D | 49.62 |
| Malleable 2.5D | RGB-D | ResNet50 | 49.70 |
| **Ours** | **RGB-D** | **FasterNet-M** | **49.82** |

### Semantic mIoU Summary Across Datasets

| Dataset | EMSAFormer | Ours | Gain |
|---------|:---:|:---:|:---:|
| NYUv2 | 49.76 | **49.82** | +0.06 |
| SUN RGB-D | 44.13 | **45.56** | +1.43 |
| Cityscapes | 60.76 | **65.11** | +4.35 |

### Model Complexity Comparison

| Method | Params | FLOPs | FPS | Memory |
|--------|:---:|:---:|:---:|:---:|
| EMSAFormer (Swin v2) | 72.08M | 50.66G | 16.32 | 3188 MiB |
| MPViT | 92.76M | 235.24G | 9.94 | 5266 MiB |
| **Ours** | **71.82M** | 75.28G | **20.33** | 3293 MiB |

### Ablation Study (NYUv2)

- Fusion encoder → Instance PQ 58.59 (significant speed improvement over Swin v2 baseline)
- +Adaptive loss → Instance PQ 59.37, improvements across 6 metrics
- +CFIL → Semantic mIoU 49.72 (+2.0), improvements across 8 metrics
- +NFCL → Panoptic PQ 43.21; full model achieves Semantic mIoU 49.82, Instance PQ 59.90
- Modulation factor comparison: $\alpha = 0.01$ yields the best panoptic PQ (41.81); larger values (0.1) lead to instability
- CFIL placement: best when placed in the semantic decoder (panoptic mIoU 50.16)
- NFCL layer selection: layers 1/2/3 are optimal (semantic mIoU 49.82); layer 4 features are already sufficiently rich and require no further guidance

## Highlights & Insights

1. **Channel redundancy exploitation**: Applying convolution to only 1/4 of channels achieves effective feature extraction while reducing FLOPs to 1/16—a concise and efficient design.
2. **BN $\gamma$ as channel attention**: Channel importance is derived from the already-learned BN parameters without additional parameters or SE module overhead.
3. **Batch-level real-time adaptive loss**: Compared to epoch-level or random weighting, per-batch dynamic adjustment yields more stable training and faster convergence.
4. **Unified five-task framework**: Semantic segmentation, instance segmentation, orientation estimation, panoptic segmentation, and scene classification are handled within a single network.
5. **Clear speed advantage**: With 71.82M parameters and 20.33 FPS, the proposed method outperforms EMSAFormer (16.32 FPS), making it suitable for robot deployment.

## Limitations & Future Work

1. **Marginal accuracy gain**: The semantic mIoU improvement over EMSAFormer on NYUv2 is only +0.06, which is not substantial.
2. **High-resolution scalability**: The current implementation struggles with very high-resolution images or video, as computational complexity grows with resolution.
3. **Ideal depth sensor assumption**: The model assumes calibrated, noise-free RGB-D input and does not address issues with consumer-grade depth sensors such as reflections, transparent surfaces, or boundary sparsity.
4. **No temporal consistency**: Frames are processed independently without considering temporal coherence in video streams, which may cause segmentation flickering in dynamic scenes.
5. **1/4-channel fusion limitation**: Although FLOPs are reduced, fine-grained inter-channel interactions may be partially lost.
6. **Limited modality exploration**: Modalities such as thermal imaging and point clouds are not explored, limiting robustness in diverse environments.

## Related Work & Insights

- **vs. EMSAFormer**: Replaces Swin v2 with a FasterNet-M fusion encoder, achieving fewer parameters (71.82M vs. 72.08M), 24% faster inference (20.33 vs. 16.32 FPS), and comparable or slightly better accuracy.
- **vs. EMSANet**: Shares the non-bottleneck 1D design philosophy but restricts it to the instance decoder and introduces NFCL/CFIL for cross-dimensional guidance.
- **vs. SegFormer**: Inherits the lightweight MLP decoder design but identifies the shallow-feature misguiding problem and addresses it with NFCL.
- **vs. FasterNet**: Directly adopts the partial convolution idea to build the fusion encoder, extended to the 4-channel RGBD setting.

## Rating

- **Novelty**: ⭐⭐⭐ — Each component is well-motivated but represents a combination of existing techniques (channel redundancy + BN-based attention + adaptive loss), lacking fundamental innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three datasets, comprehensive ablation studies (encoder, CFIL placement, NFCL layer selection, loss modulation factor, module comparisons), and complete complexity analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, rich figures and tables, complete formula derivations, and good readability.
- **Value**: ⭐⭐⭐ — Strong engineering practicality for resource-constrained robot deployment scenarios, though the academic contribution is relatively incremental.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Learning Cross-View Object Correspondence via Cycle-Consistent Mask Prediction](learning_cross-view_object_correspondence_via_cycle-consistent_mask_prediction.md)
- [\[CVPR 2026\] 3M-TI: High-Quality Mobile Thermal Imaging via Calibration-free Multi-Camera Cross-Modal Diffusion](3m-ti_high-quality_mobile_thermal_imaging_via_calibration-free_multi-camera_cros.md)
- [\[CVPR 2026\] GeomPrompt: Geometric Prompt Learning for RGB-D Semantic Segmentation Under Missing and Degraded Depth](geomprompt_rgbd_segmentation.md)
- [\[CVPR 2026\] Heuristic Self-Paced Learning for Domain Adaptive Semantic Segmentation under Adverse Conditions](heuristic_self-paced_learning_for_domain_adaptive_semantic_segmentation_under_ad.md)
- [\[AAAI 2026\] SAM-DAQ: Segment Anything Model with Depth-guided Adaptive Queries for RGB-D Video Salient Object Detection](../../AAAI2026/segmentation/sam-daq_segment_anything_model_with_depth-guided_adaptive_queries_for_rgb-d_vide.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] DriverGaze360: OmniDirectional Driver Attention with Object-Level Guidance
description: >-
  [CVPR2026][Autonomous Driving][Driver Attention] This paper introduces the first 360° panoramic driver attention dataset (~1M frames / 19 drivers) and proposes DriverGaze360-Net, which jointly learns attention maps and attended objects via an auxiliary semantic segmentation head, achieving state-of-the-art attention prediction performance on panoramic driving images.
tags:
  - CVPR2026
  - Autonomous Driving
  - Driver Attention
  - Omnidirectional View
  - Gaze Prediction
  - Semantic Segmentation
  - 360° Field of View
  - Video Swin Transformer
date: 2026-05-08
content_hash: 1529f5c254352816
---

# DriverGaze360: OmniDirectional Driver Attention with Object-Level Guidance

**Conference**: CVPR2026
**arXiv**: [2512.14266](https://arxiv.org/abs/2512.14266)
**Code**: [dfki-av/drivergaze360](https://github.com/dfki-av/drivergaze360)
**Dataset**: [HuggingFace](https://huggingface.co/datasets/dfki-av/drivergaze360)
**Area**: Autonomous Driving / Driver Attention Prediction
**Keywords**: Driver Attention, Omnidirectional View, Gaze Prediction, Semantic Segmentation, 360° Field of View, Video Swin Transformer

## TL;DR

This paper introduces the first 360° panoramic driver attention dataset (~1M frames / 19 drivers) and proposes DriverGaze360-Net, which jointly learns attention maps and attended objects via an auxiliary semantic segmentation head, achieving state-of-the-art attention prediction performance on panoramic driving images.

## Background & Motivation

Driver attention prediction is a critical task for building interpretable autonomous driving systems and for understanding driving behavior in mixed-traffic scenarios involving both human and automated vehicles. While significant progress has been made in large-scale datasets and deep learning architectures, two fundamental limitations persist:

**Limited Field of View**: Existing driver attention datasets (e.g., DR(eye)VE, BDD-A, DADA-2000) cover only a narrow forward-facing field of view (typically 60°–120°), failing to capture the complete spatial context of the driving environment. In reality, drivers frequently shift their gaze to lateral and rear regions.

**Insufficient Scene Diversity**: Existing datasets primarily focus on forward straight-driving scenarios, neglecting safety-critical maneuvers such as lane changes, turns, and interactions with pedestrians and cyclists that require peripheral vision—precisely the scenarios most relevant to safe operation.

**Lack of Object-Level Semantic Guidance**: Conventional attention prediction methods output attention distributions as heatmaps without explicitly modeling *which objects* the driver is looking at, limiting the utility of predicted attention in downstream autonomous driving decision-making.

The core motivation of this paper is that driver gaze extends well beyond the forward direction—particularly during lane changes, turns, and intersection interactions—making peripheral visual information essential. A large-scale 360° attention dataset is needed, along with a prediction model capable of simultaneously answering *where* and *what* the driver is looking at.

## Method

### Overall Architecture

The DriverGaze360 system comprises two components: a **large-scale 360° dataset** and the **DriverGaze360-Net prediction network**.

**Dataset Collection**: A driving environment is constructed using the CARLA simulator. Nineteen participants wear eye-tracking devices and complete diverse driving tasks in the simulator. Panoramic images are captured at a resolution of 6400×720 pixels, covering the full 360° field of view. Each frame is synchronized with RGB images, depth maps, instance segmentation maps, gaze coordinates (gaze_x, gaze_y), and vehicle state information (steering, throttle, brake, position, speed, etc.). The dataset encompasses 9 driving scenario types, including both routine driving and safety-critical situations (e.g., emergency events), totaling approximately 1 million annotated frames.

**Network Architecture**: DriverGaze360-Net adopts an encoder–decoder structure with Video Swin Transformer as the backbone, coupled with a multi-head decoder for joint learning of attention map prediction and semantic segmentation. The model takes a sequence of $T$ consecutive panoramic frames as input (default $T=16$) and outputs an attention heatmap along with a 7-class semantic segmentation map.

### Key Design 1: Video Swin Transformer Spatiotemporal Encoder

The encoder employs Swin3D-S (pretrained on Kinetics-400), a hierarchical video Transformer architecture. The rationale for choosing a video Transformer over a conventional CNN backbone is as follows:

- **Temporal Modeling Capacity**: Driver gaze behavior exhibits strong temporal continuity, requiring joint encoding of consecutive frame sequences. Video Swin efficiently captures spatiotemporal dependencies via 3D shifted window attention.
- **Multi-Scale Feature Extraction**: The backbone consists of 4 stages, each progressively downsampling via Patch Merging to produce feature maps at different resolutions (with channel dimensions of 96, 192, 384, and 768, respectively). These multi-scale features are passed to the decoder via skip connections to preserve fine-grained spatial information.
- **Global Context Awareness**: Panoramic images have an extreme aspect ratio (approximately 9:1), making it difficult for conventional convolutional networks to cover such a large spatial extent with limited receptive fields. The Transformer's attention mechanism is inherently suited to capturing long-range dependencies.

The encoder forward pass processes an input tensor of shape $B \times C \times T \times H \times W$ through Patch Embedding and positional encoding, then sequentially through 4 Swin Transformer blocks and Patch Merging layers, producing multi-scale feature maps at 4 resolutions. These features are reversed from coarse-to-fine order before being fed into the decoder.

### Key Design 2: Joint Learning with Auxiliary Semantic Segmentation Head

This is the central contribution of the paper. The decoder (DecoderSwin) consists of a shared upsampling backbone and multiple task-specific heads:

- **Attention Prediction Head (sal)**: Outputs a single-channel attention heatmap, activated by Sigmoid to produce values in $[0, 1]$.
- **Semantic Segmentation Head (ss)**: Outputs 7-channel segmentation logits corresponding to 7 semantic categories—background, traffic lights, traffic signs, pedestrians, cyclists, vehicles (merging car/truck/bus/train/motorcycle), and bicycles.

Both heads share the upsampling backbone of the decoder (convtsp1→convtsp2→convtsp3), followed by independent convolutional layers that generate task-specific outputs. The key insight is that **the semantic segmentation task compels the network to learn object-level semantic representations, which in turn improves the spatial localization accuracy of attention prediction**. Drivers tend to concentrate their gaze on specific object types (e.g., leading vehicles, pedestrians, traffic lights), and the segmentation head explicitly encodes the location and category of these objects.

The ground truth for semantic segmentation is not derived directly from CARLA's instance segmentation annotations; instead, it is filtered using the attention saliency map—retaining only the object segmentation labels within regions actually fixated by the driver. This ensures that the segmentation head learns about *attended objects* rather than all visible objects.

### Key Design 3: Multi-Loss Joint Optimization

The total loss is a weighted combination of the attention loss and the segmentation loss:

$$L_{total} = w_{sal} \cdot L_{sal} + w_{ss} \cdot L_{ss}$$

The attention loss comprises four classical saliency evaluation metrics used as loss terms:

- **NSS Loss**: The predicted map is z-score normalized and sampled at fixation points, measuring prediction response strength at gaze locations.
- **KLD Loss**: KL divergence between the predicted distribution and the ground-truth fixation distribution.
- **CC Loss**: Linear correlation coefficient between the predicted map and the ground-truth saliency map.
- **MSE Loss**: Pixel-wise mean squared error.

The segmentation loss combines three loss functions for robustness: Cross-Entropy (CE) loss + Jaccard/IoU loss + Dice loss.

Training uses the AdamW optimizer (learning rate 1e-6) with support for mixed-precision training and Distributed Data Parallel (DDP). A KLD-based weighted sampling strategy is also introduced, assigning higher training weights to "hard samples" (frames where the prediction diverges substantially from the ground truth).

## Key Experimental Results

### Dataset Comparison

| Dataset | 360° FoV | # Scenario Types | Driving Scenarios | # Participants | Data Source |
|--------|:--------:|:---------:|---------|:--------:|---------|
| DR(eye)VE | ✗ | 6 | Routine driving | 8 | Real driving |
| LBW | ✗ | 7 | Routine driving | 28 | Real driving |
| BDD-A | ✗ | 4 | Busy intersections / emergency braking | 1,228 | Video watching |
| DADA-2000 | ✗ | 6 | Driving accidents | 20 | Video watching |
| **DriverGaze360** | **✓** | **9** | **Routine + safety-critical** | **19** | **Simulated driving** |

DriverGaze360 is the only large-scale driver attention dataset providing full 360° coverage. Compared to the largest existing dataset BDD-A (1,228 participants watching videos), DriverGaze360 involves fewer participants but captures genuine active driving behavior in a simulator, more closely reflecting real-world driving.

### Attention Prediction Performance

| Method | KLD ↓ | CC ↑ | SIM ↑ | NSS ↑ |
|------|:-----:|:----:|:-----:|:-----:|
| Baseline (forward-view models) | Higher | Lower | Lower | Lower |
| DriverGaze360-Net (sal only) | Improved | Improved | Improved | Improved |
| **DriverGaze360-Net (sal+ss)** | **Best** | **Best** | **Best** | **Best** |

The addition of the auxiliary semantic segmentation head yields consistent improvements across all attention prediction metrics, validating the benefit of object-level semantic guidance for attention prediction. The model achieves state-of-the-art performance on panoramic driving images across all four standard metrics: KLD (lower is better), CC (higher is better), SIM (higher is better), and NSS (higher is better).

## Key Findings

1. **Significant Gains from the Auxiliary Segmentation Head**: The semantic segmentation head not only enables the recognition of attended objects but, more importantly, provides an implicit object-level prior that benefits attention prediction. Ablation experiments show that removing the segmentation head leads to a noticeable degradation in attention prediction performance.
2. **Necessity of Panoramic Field of View**: During lane changes, turns, and blind-spot checks, driver fixations deviate substantially from the forward center region. Forward-only models are unable to capture these critical gaze behaviors.
3. **Importance of Temporal Information**: Processing continuous frame sequences with Video Swin Transformer ($T=16$ frames) significantly outperforms single-frame input, demonstrating strong temporal dependency in driver gaze behavior.
4. **Gaze-Filtered Semantic Labels**: Experiments show that using "attended objects" rather than "all visible objects" as segmentation ground truth is more effective, as it directly aligns semantic learning with the core objective of attention prediction.

## Highlights & Insights

- **Major Dataset Contribution**: This is the first large-scale driver attention dataset covering a full 360° field of view, filling an important gap in the field. The scale (~1M frames) and diversity (9 scenario types) are sufficient to support training of complex models.
- **Simple Yet Effective Method**: The auxiliary segmentation head design is conceptually straightforward but empirically effective—multi-task learning enables the network to simultaneously understand spatial distribution and object semantics, with each reinforcing the other. This design introduces virtually no additional inference overhead (the segmentation head can be disabled at inference time).
- **High Open-Source Completeness**: Code, dataset, and pretrained checkpoints are all publicly available on GitHub and HuggingFace, facilitating reproducibility and follow-up research. Detailed training configurations (loss weights, data loading, distributed training, etc.) are provided, making the codebase highly accessible.
- **Carefully Designed Data Format**: Each recording includes RGB video, depth maps, instance segmentation, saliency maps, and detailed CSV metadata (gaze coordinates, vehicle control signals, pose, speed), supporting a wide range of downstream research directions.

## Limitations & Future Work

1. **Sim-to-Real Gap**: All data are collected from the CARLA simulator, introducing a domain gap. The visual realism of the simulated environment, traffic participant behavior, and lighting variation differ from real-world conditions; the model's generalization to real driving data remains to be validated.
2. **Small Participant Pool**: With only 19 drivers, individual variability may lead to biased gaze patterns. By comparison, BDD-A involves 1,228 participants. A smaller participant pool may cause the model to overfit to the gaze habits of a limited number of individuals.
3. **Limited Semantic Categories**: Only 7 semantic categories are defined, omitting road markings, curbs, buildings, and other elements equally relevant to driving decisions. Finer-grained semantic classification may further improve performance.
4. **No Cross-Domain Validation**: The paper does not evaluate the model's transfer performance on existing real-world driving datasets such as DR(eye)VE.
5. **Incomplete Inference Script**: The inference functionality in the GitHub repository is marked as TODO, limiting immediate practical deployment.

## Related Work & Insights

- **DR(eye)VE / BDD-A / DADA-2000**: Representative forward-view driver attention datasets covering only a limited forward field of view. DriverGaze360's extension to a 360° panoramic view is a natural progression of this line of work.
- **SCOUT** (ECCV 2022): The implementation of this paper references SCOUT's architectural design, similarly employing Swin Transformer as the backbone for attention prediction. DriverGaze360-Net extends this with a newly added semantic segmentation auxiliary head.
- **Video Swin Transformer** (CVPR 2022): The successful application of 3D Swin Transformer to video understanding tasks. This paper introduces it to the domain of driver attention prediction, leveraging its spatiotemporal modeling capability to process panoramic video sequences.
- **Multi-Task Learning Paradigm**: The auxiliary segmentation head follows the established multi-task learning strategy of enhancing the primary task's feature representations through related auxiliary tasks—a strategy with successful precedents in object detection (detection + segmentation) and depth estimation (depth + semantics).

**Insight**: Using semantic segmentation as an auxiliary task for attention prediction is a strategy worth generalizing. In other visual attention and saliency modeling tasks (e.g., video saliency, social attention), introducing scene understanding auxiliary tasks to enhance spatial perception is a promising direction. Additionally, the handling of 360° panoramic input—characterized by extreme aspect ratios and global receptive field requirements—offers valuable reference for designing novel attention model architectures.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The first 360° driver attention dataset is original; the auxiliary segmentation head is effective but not a fundamentally new paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-metric evaluation is comprehensive and dataset comparisons are thorough, but cross-domain transfer experiments are absent.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated, dataset description is detailed, and open-source completeness is high.
- **Value**: ⭐⭐⭐⭐⭐ — The dataset contribution is outstanding, filling a critical gap in the field and poised to advance research on panoramic driver attention prediction.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Where, What, Why: Towards Explainable Driver Attention Prediction](../../ICCV2025/autonomous_driving/where_what_why_towards_explainable_driver_attention_prediction.md)
- [\[CVPR 2026\] O3N: Omnidirectional Open-Vocabulary Occupancy Prediction](o3n_omnidirectional_openvocabulary_occupancy_predi.md)
- [\[CVPR 2026\] IGASA: Integrated Geometry-Aware and Skip-Attention Modules for Enhanced Point Cloud Registration](igasa_integrated_geometry-aware_and_skip-attention_modules_for_enhanced_point_cl.md)
- [\[CVPR 2026\] CoIn3D: Revisiting Configuration-Invariant Multi-Camera 3D Object Detection](coin3d_revisiting_configuration-invariant_multi-camera_3d_object_detection.md)
- [\[CVPR 2026\] Sparsity-Aware Voxel Attention and Foreground Modulation for 3D Semantic Scene Completion](sparsity-aware_voxel_attention_and_foreground_modulation_for_3d_semantic_scene_c.md)

<!-- RELATED:END -->

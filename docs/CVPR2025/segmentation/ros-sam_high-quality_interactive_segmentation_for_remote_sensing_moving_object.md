---
title: >-
  [Paper Note] ROS-SAM: High-Quality Interactive Segmentation for Remote Sensing Moving Object
description: >-
  [CVPR 2025][Segmentation][Remote sensing video] ROS-SAM adapts SAM to the high-quality interactive segmentation task of moving objects in remote sensing videos by fine-tuning the encoder via LoRA, improving the HQ decoder, and redesigning the data pipeline. This achieves a 13% IoU improvement and demonstrates strong zero-shot generalization capabilities.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Remote sensing video"
  - "interactive segmentation"
  - "SAM fine-tuning"
  - "high-quality segmentation"
  - "domain adaptation"
date: 2026-05-08
content_hash: fe5e0bd5ac3c2c09
---

# ROS-SAM: High-Quality Interactive Segmentation for Remote Sensing Moving Object

**Conference**: CVPR 2025  
**arXiv**: [2503.12006](https://arxiv.org/abs/2503.12006)  
**Code**: [github.com/ShanZard/ROS-SAM](https://github.com/ShanZard/ROS-SAM)  
**Area**: Image Segmentation  
**Keywords**: Remote sensing video, interactive segmentation, SAM fine-tuning, high-quality segmentation, domain adaptation

## TL;DR

ROS-SAM adapts SAM to the high-quality interactive segmentation task of moving objects in remote sensing videos by fine-tuning the encoder via LoRA, improving the HQ decoder, and redesigning the data pipeline. This achieves a 13% IoU improvement and demonstrates strong zero-shot generalization capabilities.

## Background & Motivation

Moving object segmentation in remote sensing videos (satellite videos) is an emerging but challenging research field. Objects in remote sensing videos (airplanes, cars, ships, trains, etc.) typically feature small sizes, blurry characteristics, high density, and lack of clear orientations, making frame-by-frame manual annotation extremely expensive.

As a general segmentation foundation model, SAM possesses powerful zero-shot capabilities, but directly applying it to remote sensing data presents three key problems:
- **Domain gap**: Remote sensing objects are unaffected by gravity and have ambiguous orientations; SAM often generates generic four-pointed star shapes when predicting airplane orientations.
- **Poor boundary quality**: The mask decoder of SAM lacks fine texture and edge details, resulting in coarse boundaries and fragmented masks.
- **Resolution mismatch**: SAM restricts input resolution to a fixed $1024 \times 1024$. Downsampling high-resolution remote-sensing images causes small objects to disappear.

A natural strategy is to leverage existing remote sensing object tracking datasets (with bounding box annotations) and convert tracking data into segmentation data via interactive segmentation, thereby advancing remote-sensing video segmentation at minimal cost.

## Method

### Overall Architecture

ROS-SAM is built upon SAM, incorporating three core improvements: (1) LoRA fine-tuning on the image encoder to inject remote sensing domain knowledge; (2) an HQ mask decoder that fuses multi-stage features to generate high-quality masks; and (3) a redesigned training-inference pipeline to handle multi-scale and high-resolution issues. During training, the pre-trained parameters of SAM are frozen, and only the red components are updated.

### Key Designs

**1. LoRA Fine-Tuning Image Encoder + Unfreezing the Last Layer**

- **Function**: Inject remote sensing domain knowledge while preserving the general generalization capacity of SAM.
- **Mechanism**: Inject low-rank decomposition matrices $h = W_0 x + W_d W_e x$ (where $r \ll \min(m,n)$) into the Query and Value matrices of all Transformer layers in the ViT encoder. Additionally, unfreeze the last block of the encoder to extract more discriminative global contextual features.
- **Design Motivation**: LoRA efficiently adapts to target domains without disrupting pre-trained knowledge. Unfreezing the last layer enhances feature discrimination, enabling the model to distinguish between similar objects (e.g., airplanes vs. boarding bridges). Shallow layers capture texture details, while deep layers encode semantic context.

**2. HQ Mask Decoder + Alternating Optimization Strategy**

- **Function**: Fuse fine texture features from early layers of the image encoder and deep global context features to generate high-quality segmentation masks.
- **Mechanism**: Add the decoder branch of HQ-SAM on top of the original mask decoder of SAM, integrating multi-stage image features, prompt tokens, and mask tokens for high-quality predictions. Unlike HQ-SAM which freezes the original decoder, ROS-SAM **alternately updates** both the original SAM decoder and the HQ decoder.
- **Design Motivation**: Directly updating the original decoder degrades performance (as shown by a decrease in IoU in ablation experiments) due to pre-trained knowledge disruption, whereas the HQ decoder is a newly introduced lightweight component. Alternating updates allow the two decoders to learn from each other, eventually improving the IoU from 47.15% to 48.16%.

**3. Specialized Training-Inference Data Pipeline**

- **Function**: Introduce multi-scale and multi-angle augmentations during training, and ensure high-quality single-object predictions during inference.
- **Mechanism**: During the training phase, LSJ (Large Scale Jittering, scale 0.1x to 4.0x) and random rotation augmentations are employed. During the inference phase, a $N \times 512 \times 512$ crop is extracted based on the centered prompt location, bicubically upsampled to $1024 \times 1024$, and after single-object inference, mapped back to its original position in the image.
- **Design Motivation**: Remote sensing objects are unconstrained by gravity and lack fixed orientations, requiring rotation augmentations. LSJ covers multi-scale targets. Double upsampling during inference is optimal; excessive upsampling leads to jagged edges. The inference pipeline alone yields a 6% IoU improvement for SAM.

### Loss & Training

A combination of Binary Cross-Entropy (BCE) Loss and Dice Loss is used to alternately update the weights of both the SAM Mask and ROS-SAM Mask branches. Training is performed for 24 epochs with a learning rate of 1e-3.

## Key Experimental Results

### Main Results: Comparison with SOTA Methods (SAT-MTB Dataset)

| Method | IoU | BIoU |
|------|-----|------|
| SAM (Original config / + Inference pipeline) | 37.25 / 43.41 | 37.14 / 43.30 |
| SAM2 (Original config / + Inference pipeline) | 36.80 / 41.75 | 36.67 / 41.55 |
| HQ-SAM (Original config / + Inference pipeline) | 43.27 / 47.15 | 43.21 / 47.11 |
| **ROS-SAM** | **50.54** | **50.36** |

### Ablation Study: Contribution of Each Module

| Method | IoU | BIoU |
|------|-----|------|
| SAM baseline | 43.41 | 43.30 |
| + Directly update Mask Decoder | 42.82 | 42.69 |
| + HQ Mask Decoder | 47.15 | 47.11 |
| + Alternately update two Decoders | 48.16 | 48.03 |
| + LoRA + Unfreeze last layer | 49.19 | 49.05 |
| + Training pipeline (LSJ + Rotation) | **50.54** | **50.36** |

### Cross-Dataset Generalization (Static Remote Sensing Images)

| Method | iSAID IoU | NPWS VHR-10 IoU |
|------|-----------|-----------------|
| SAM | 53.19 | 65.54 |
| HQ-SAM | 63.96 | 78.44 |
| **ROS-SAM** | **73.22** | **87.46** |

### Key Findings

- The inference pipeline alone brings a 6%+ IoU improvement to SAM, demonstrating that resolution adaptation is critical for remote sensing scenarios.
- Directly fine-tuning the original SAM decoder degrades performance (42.82 vs. 43.41), confirming the importance of preserving pre-trained knowledge.
- LSJ yields a larger performance gain than random rotation (~0.8 vs. ~0.2) because covering multiple scales is more crucial in remote sensing.
- High-quality segmentation masks are generated zero-shot on tracking datasets such as SatSOT, VISO, and OOTB, proving robust generalization.

## Highlights & Insights

1. **Converting tracking data into segmentation data at minimal cost** is highly practical, addressing the scarcity of annotations in remote sensing video segmentation.
2. The strategy of **alternately optimizing the new and old decoders** is highly instructive: it preserves pre-trained knowledge while introducing new capabilities.
3. The center-crop and upsampling inference pipeline is simple yet effective, providing plug-and-play improvements for all SAM variants.

## Limitations & Future Work

- Training is only performed on the SAT-MTB dataset (249 videos), which has a limited data scale.
- Vehicle objects cannot be effectively processed due to extremely small sizes (around 10 pixels).
- When object features are highly ambiguous (e.g., where manual annotations infer shapes based on prior knowledge), the model struggles to achieve human-level performance.
- Future work can consider combining the video propagation capabilities of SAM2 to achieve semi-automatic video segmentation.

## Related Work & Insights

- **Relationship to SAM/HQ-SAM**: ROS-SAM adds LoRA domain adaptation and an alternating training strategy on top of HQ-SAM.
- **Remote Sensing + Foundation Model Trend**: Fine-tuning foundation models with LoRA/Adapter to adapt to remote sensing is a current research hotspot; ROS-SAM validates the effectiveness of LoRA on target Q/V matrices.
- **Insights**: For any scenario fine-tuning a foundation model while striving to maintain its generalization ability, "frozen backbone + LoRA + alternating optimization" serves as a universal template.

## Rating

⭐⭐⭐⭐

Solid engineering design with each module thoroughly validated by ablation experiments. The inference pipeline and alternating optimization strategy hold independent reference value. The primary limitation is the relatively small dataset and restricted scenario, with the core idea being an incremental integration of fine-tuning techniques on SAM/HQ-SAM.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] The Devil is in Temporal Token: High Quality Video Reasoning Segmentation](the_devil_is_in_temporal_token_high_quality_video_reasoning_segmentation.md)
- [\[ICCV 2025\] Inter2Former: Dynamic Hybrid Attention for Efficient High-Precision Interactive Segmentation](../../ICCV2025/segmentation/inter2former_dynamic_hybrid_attention_for_efficient_high-precision_interactive_s.md)
- [\[CVPR 2025\] RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images](rdnet_region_proportion-aware_dynamic_adaptive_salient_object_detection_network_.md)
- [\[CVPR 2025\] G2HFNet: GeoGran-Aware Hierarchical Feature Fusion Network for Salient Object Detection in Optical Remote Sensing Images](binwang2hfnet_geogran-aware_hierarchical_feature_fusion_network_for_salient_obje.md)
- [\[CVPR 2025\] SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data](sgma_semantic-guided_modality-aware_segmentation_for_remote_sensing_with_incompl.md)

</div>

<!-- RELATED:END -->

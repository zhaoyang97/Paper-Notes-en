---
title: >-
  [Paper Note] Visible and Clear: Finding Tiny Objects in Difference Map
description: >-
  [ECCV 2024][Object Detection][Tiny Object Detection] SR-TOD introduces the image self-reconstruction mechanism into object detection for the first time, discovering a strong correlation between reconstruction difference maps and tiny objects. It designs a Difference Map Guided Feature Enhancement (DGFE) module, achieving significant improvements on the self-built anti-UAV dataset DroneSwarms as well as VisDrone2019 and AI-TOD.
tags:
  - "ECCV 2024"
  - "Object Detection"
  - "Tiny Object Detection"
  - "Self-Reconstruction"
  - "Difference Map"
  - "Feature Enhancement"
  - "Anti-UAV"
date: 2026-05-08
content_hash: c371779cbbba433f
---

# Visible and Clear: Finding Tiny Objects in Difference Map

**Conference**: ECCV 2024  
**arXiv**: [2405.11276](https://arxiv.org/abs/2405.11276)  
**Code**: [https://github.com/Hiyuur/SR-TOD](https://github.com/Hiyuur/SR-TOD)  
**Area**: Object Detection  
**Keywords**: Tiny Object Detection, Self-Reconstruction, Difference Map, Feature Enhancement, Anti-UAV

## TL;DR

SR-TOD introduces the image self-reconstruction mechanism into object detection for the first time, discovering a strong correlation between reconstruction difference maps and tiny objects. It designs a Difference Map Guided Feature Enhancement (DGFE) module, achieving significant improvements on the self-built anti-UAV dataset DroneSwarms as well as VisDrone2019 and AI-TOD.

## Background & Motivation

Tiny Object Detection (TOD) is a critical challenge in object detection. According to the AI-TOD benchmark definition, "extremely tiny" objects are only 2-8 pixels, "tiny" objects are 8-16 pixels, and "small" objects are 16-32 pixels.

**Limitations of Prior Work**:
- **Information loss is the core challenge**: Downsampling operations in the backbone network inevitably lose information about tiny objects, and the signal of "extremely tiny" objects is almost entirely erased.
- **Generative feature enhancement methods are flawed**: Existing methods (such as GAN-based super-resolution) tend to generate fake textures and artifacts, which can degrade detection performance; super-resolution architectures also suffer from high computational overhead and difficult end-to-end optimization.
- **Tiny objects are "invisible" to the detector**: As seen from the feature heatmaps of the FPN P2 layer, the activation signals of many tiny objects are extremely weak or have even vanished.

**Key Challenge**: Tiny object information lost during downsampling cannot be reliably restored through generative methods.

**Key Insight**: Instead of trying to restore lost information, it is better to first "find" where the information loss occurs. Leveraging image self-reconstruction—allowing the feature maps of the detection model to reconstruct the input image—the areas that are difficult to reconstruct happen to be those with severe information loss (i.e., where tiny objects are located). The difference map between the reconstructed image and the original image directly reveals the position and structure of tiny objects.

## Method

### Overall Architecture

The SR-TOD (Self-Reconstructed Tiny Object Detection) framework:
1. Image $\rightarrow$ backbone $\rightarrow$ FPN to obtain multi-scale feature pyramid P2-P5
2. P2 (the highest resolution layer, responsible for tiny object detection) $\rightarrow$ Reconstruction Head $\rightarrow$ Reconstructed Image
3. Reconstructed image subtracted from the original image (absolute difference) $\rightarrow$ Difference Map
4. Difference Map + P2 $\rightarrow$ DGFE module $\rightarrow$ Enhanced Feature P2'
5. P2' replaces the original P2 and is fed into the detection head

This framework can be integrated as a plug-and-play component into most detectors that use FPN.

### Key Designs

1. **Difference Map**:

    - **Function**: From the bottom-level feature map of the detection model, reconstruct the input image, using the reconstruction error to locate tiny objects.
    - **Mechanism**: The image reconstruction task is highly sensitive to pixel variations. When reconstructing from detection features, the regions where structural/texture information is severely lost (i.e., tiny objects) are the hardest to reconstruct, resulting in strong activations in the difference map.
    - Reconstruction head structure: P2 $\rightarrow$ two upsamplings (transposed convolution + two Conv layers + ReLU) $\rightarrow$ 1×1 Conv $\rightarrow$ Sigmoid $\rightarrow$ Reconstructed image
    - Difference map computation: $D = \text{Mean}_{channel}(\text{Abs}(I_r - I_o))$
    - Reconstruction head parameters are optimized using MSE loss: $\mathcal{L}_{rec} = \text{MSE}(I_r, I_o)$
    - **Key Findings**: There is a strong correlation between the difference map and tiny objects—even "extremely tiny" objects whose signals are almost completely erased in the feature maps remain clearly visible in the difference map, preserving the main structure of the objects.

2. **Difference Map Guided Feature Enhancement (DGFE)**:

    - **Function**: Use prior information from the difference map to enhance the feature representation of tiny objects in P2.
    - **Mechanism**: Construct an element-wise attention matrix M = channel-wise reweighting $\times$ spatial-wise filtration.
    - **Filtration**: Use a learnable threshold $t$ to binarize the difference map, filtering out noise signals and preserving salient activation regions. The binary map $+1$ ensures that original features are not erased by zero-value regions.
    - Formula: $\text{Filtration}(D) = \text{Resize}((\text{Sign}(D-t)+1) \times 0.5) + 1$
    - **Reweighting**: Since the difference map contains only spatial information, channel attention is applied to P2 (AvgPool + MaxPool $\rightarrow$ MLP $\rightarrow$ Sigmoid) to reweight along the channel dimension.
    - Formula: $\text{Reweighting}(P2) = \sigma(\text{MLP}(\text{AvgPool}(P2)) + \text{MLP}(\text{MaxPool}(P2)))$
    - Final enhancement: $P2' = (\text{Reweighting}(P2) \otimes \text{Filtration}(D)) \otimes P2$

3. **DroneSwarms Dataset**:

    - **Function**: Propose a new anti-UAV tiny object detection dataset.
    - **Characteristics**: The average drone size is only about 7.9 pixels, which is currently the smallest; it includes complex backgrounds and various lighting conditions; multi-instance scenarios.
    - **Design Motivation**: Existing anti-UAV datasets (MAV-VID, Drone-vs-Bird, DUT Anti-UAV) contain relatively large objects or are primarily single-object scenarios.

### Loss & Training

- Total Loss = Detection Loss (classification + regression, following the baseline detector) + Reconstruction Loss (MSE)
- The reconstruction head is constrained by MSE, and the detection head is trained normally.
- The reconstruction loss indirectly constrains the backbone to preserve more pixel-level information.

## Key Experimental Results

### Main Results

DroneSwarms dataset results:

| Method | AP | AP_0.5 | AP_vt | AP_t | AP_s |
|------|-----|--------|-------|------|------|
| Cascade R-CNN | 36.4 | 85.0 | 28.8 | 45.7 | 58.3 |
| DetectoRS | 37.9 | 87.4 | 30.5 | 46.9 | 59.3 |
| RFLA | 36.9 | 86.3 | 29.5 | 45.3 | 58.0 |
| Cascade R-CNN + SR-TOD | 38.3 | 87.4 | 30.8 | 47.4 | 59.4 |
| DetectoRS + SR-TOD | 38.8 | 87.9 | 31.6 | 47.7 | 59.0 |
| **RFLA + SR-TOD** | **39.0** | **88.9** | **31.8** | **47.6** | **59.2** |
| **Max Gain Δ** | **+2.1** | **+2.6** | **+2.3** | **+0.8** | **+1.1** |

AI-TOD dataset results:

| Method | AP | AP_0.5 | AP_vt | AP_t | AP_s |
|------|-----|--------|-------|------|------|
| Cascade R-CNN | 14.0 | 31.2 | 0.1 | 10.3 | 26.2 |
| RFLA | 21.7 | 50.5 | 8.3 | 21.8 | 26.3 |
| **DetectoRS + SR-TOD** | **24.0** | **54.6** | **10.1** | **24.8** | **29.3** |
| **Max Gain Δ** | **+9.4** | **+22.8** | **+10.1** | **+13.8** | **+1.9** |

### Ablation Study

Contribution of each component (DroneSwarms, Cascade R-CNN baseline):

| RH | DGFE | AP | AP_0.5 | AP_vt | AP_t |
|----|------|-----|--------|-------|------|
| ✗ | ✗ | 36.4 | 85.0 | 28.8 | 45.7 |
| ✓ | ✗ | 36.5 | 84.9 | 28.7 | 45.9 |
| ✓ | ✓ | **38.3** | **87.4** | **30.8** | **47.4** |

Comparison of feature enhancement methods:

| Method | AP | AP_0.5 | AP_vt | AP_t |
|------|-----|--------|-------|------|
| Element-wise multiplication | 36.2 | 84.9 | 28.8 | 45.7 |
| Concatenation fusion | 36.6 | 85.3 | 29.0 | 46.0 |
| **Element-wise attention (DGFE)** | **38.3** | **87.4** | **30.8** | **47.4** |

Threshold filtering strategy (VisDrone2019):

| Method | AP | AP_vt | AP_t | AP_s |
|------|-----|-------|------|------|
| No threshold | 27.0 | 2.2 | 11.3 | 24.2 |
| Fixed threshold | 27.1 | 2.3 | 11.2 | 24.0 |
| **Learnable threshold** | **27.3** | **2.3** | **11.5** | **24.7** |

### Key Findings

1. **Strong correlation between difference maps and tiny objects**: This is the core finding—the self-reconstructed difference map is highly sensitive to tiny objects, allowing even objects that have almost disappeared in the feature maps to be clearly displayed in the difference map.
2. **Reconstruction head alone leads to almost no improvement**: Simply adding the reconstruction head only yields +0.1 AP, proving that the real value of the difference map lies in being utilized by the DGFE module.
3. **Element-wise attention is far superior to direct fusion**: Simple concatenation or multiplication instead introduces noise; the attention mechanism is key to properly utilizing the difference map prior.
4. **Orthogonal and complementary to other methods**: SR-TOD can be combined with RFLA (a label assignment method) and DetectoRS (an improved multi-scale FPN) without conflict.
5. **High-frequency difference maps are slightly superior to pixel-level difference maps**: This validates that tiny object information loss mainly occurs in high-frequency components, though pixel-level difference maps are still preferred for computational efficiency.

## Highlights & Insights

- **Innovative application of the self-reconstruction mechanism**: Shifting image reconstruction from a low-level vision task to prior knowledge extraction for object detection is a highly novel perspective.
- **Highly inspiring core observation**: "The regions difficult for the detection model to reconstruct are precisely those difficult for it to detect"—this discovery provides a completely new approach to tiny object detection.
- **Plug-and-play design**: Simply adding the reconstruction head and DGFE module after the FPN P2 layer makes it applicable to various detectors using FPN.
- **Efficient and simple**: No complex architectures like GAN or super-resolution are required; the reconstruction head itself is very lightweight.

## Limitations & Future Work

- Only P2 layer reconstruction is utilized, and the possibility of multi-scale reconstruction remains unexplored.
- The reconstruction constraint is unsupervised MSE, with no explicit guidance to focus reconstruction on tiny object regions.
- Limited effectiveness on Transformer-based detectors (e.g., DETR series, with DINO only showing +0.2 AP).
- The adaptive capability of the learnable threshold is limited (utilizing a single global threshold).
- The DroneSwarms dataset only includes the drone category, and scene diversity needs to be expanded.

## Related Work & Insights

- **RFLA**: Improves label assignment via receptive field matching, which is complementary to SR-TOD.
- **DetectoRS**: Enhances multi-scale features via recursive FPN, without conflict with SR-TOD.
- **Super-Resolution methods** (such as SOD-MTGAN): Generate high-resolution features using GANs, but are prone to artifacts.
- **HANet**: Predicts activation maps to obtain scale-specific feature subspaces, but struggles to capture tiny objects in shallow features.
- **Inspirations**: The concept of self-reconstructed difference maps can be extended to other scenarios involving information loss (e.g., small targets in remote sensing, tiny lesions in medical image detection).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Self-reconstructed difference map for object detection is a brand new perspective; the core finding is highly inspiring)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Three datasets, multiple detectors, thorough ablation studies)
- Writing Quality: ⭐⭐⭐⭐ (Clear logic, convincing visualizations)
- Value: ⭐⭐⭐⭐ (Highly practical, simple and reproducible method, generalizable concept)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] GRA: Detecting Oriented Objects Through Group-Wise Rotating and Attention](gra_detecting_oriented_objects_through_group-wise_rotating_and_attention.md)
- [\[CVPR 2026\] Rotation Invariant and Symmetry Aware Pixel Difference Network for Remote Sensing Object Detection](../../CVPR2026/object_detection/rotation_invariant_and_symmetry_aware_pixel_difference_network_for_remote_sensin.md)
- [\[CVPR 2026\] Saliency-R1: Enforcing Interpretable and Faithful Vision-language Reasoning via Saliency-map Alignment Reward](../../CVPR2026/object_detection/saliency-r1_enforcing_interpretable_and_faithful_vision-language_reasoning_via_s.md)
- [\[CVPR 2026\] DyFCLT: Dynamic Frequency-Decoupled Cross-Modal Learning Transformer for Multimodal Tiny Object Detection](../../CVPR2026/object_detection/dyfclt_dynamic_frequency-decoupled_cross-modal_learning_transformer_for_multimod.md)
- [\[CVPR 2025\] Small Target Detection Based on Mask-Enhanced Attention Fusion of Visible and Infrared Remote Sensing Images](../../CVPR2025/object_detection/small_target_detection_based_on_mask-enhanced_attention_fusion_of_visible_and_in.md)

</div>

<!-- RELATED:END -->

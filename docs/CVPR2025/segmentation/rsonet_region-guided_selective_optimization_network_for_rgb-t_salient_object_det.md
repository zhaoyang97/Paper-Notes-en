---
title: >-
  [Paper Note] RSONet: Region-guided Selective Optimization Network for RGB-T Salient Object Detection
description: >-
  [CVPR2025][Segmentation][RGB-T Salient Object Detection] The region-guided selective optimization network (RSONet) is proposed to address the inconsistency of salient regions between RGB and thermal images through a two-stage process (region guidance and saliency generation). It dynamically selects the modality with more accurate information to dominate subsequent fusion based on similarity scores.
tags:
  - "CVPR2025"
  - "Segmentation"
  - "RGB-T Salient Object Detection"
  - "Region Guidance"
  - "Selective Optimization"
  - "Multimodal Fusion"
  - "SwinTransformer"
date: 2026-05-08
content_hash: eb63f9797f97ad95
---

# RSONet: Region-guided Selective Optimization Network for RGB-T Salient Object Detection

**Conference**: CVPR2025  
**arXiv**: [2603.12685](https://arxiv.org/abs/2603.12685)  
**Code**: To be confirmed  
**Area**: Image Segmentation  
**Keywords**: RGB-T Salient Object Detection, Region Guidance, Selective Optimization, Multimodal Fusion, SwinTransformer

## TL;DR
The region-guided selective optimization network (RSONet) is proposed to address the inconsistency of salient regions between RGB and thermal images through a two-stage process (region guidance and saliency generation). It dynamically selects the modality with more accurate information to dominate subsequent fusion based on similarity scores.

## Background & Motivation
- RGB-T salient object detection utilizes thermal IR images to compensate for the limitations of RGB in complex backgrounds and low-contrast scenes.
- **Key Challenge**: The distribution of salient regions for targets is inconsistent between RGB and thermal images—the target might be almost invisible in the thermal map under some scenarios, while in others, the target blends into the background in the RGB image.
- Existing methods mostly rely on addition, multiplication, concatenation, or attention mechanisms for fusion, which implicitly assumes that both modalities are equally important. This introduces significant irrelevant noise when there is a large discrepancy in information quality.
- A mechanism is required to assess which modality is more reliable and allow the reliable modality to dominate the fusion process.

## Core Problem
How to adaptively select the modality with more accurate information to dominate the bimodal feature fusion when there is an inconsistency in the distribution of salient regions between modalities?

## Method
RSONet is divided into two stages: the **region guidance stage** and the **saliency generation stage**.

### 1. Region Guidance Stage
- **Three-way Parallel Branches**: R (RGB), T (Thermal), and RT (sum of RGB and Thermal), sharing the same encoder-decoder structure.
- **Backbone**: SwinTransformer is used to extract five levels of multi-scale features.
- **Context Interaction Module (CI)**: Applies varied kernel sizes to features of different layers (low-level uses 1×1/3×3/5×5/7×7, middle-level excludes 7×7, and high-level only uses 1×1/3×3) to prevent the high-level, low-resolution features from introducing irrelevant background noise via large kernels.
- **Spatial-aware Fusion Module (SF)**: Fuses the output of CI layer by layer, utilizing global max pooling + 1×1 convolution + sigmoid to generate spatial weights for spatial dimension optimization.
- **Similarity Calculation**: The three branches generate guidance maps G^R, G^T, and G^RT respectively, and calculate the mean differences of G^R and G^T compared to G^RT. A smaller difference indicates that the modality contains more accurate target information.

### 2. Saliency Generation Stage
- **Selective Optimization Module (SO)**: Directs the modality with more accurate information to dominate the fusion process based on the similarity results. It first enhances the bimodal features through multiplication and addition using G^RT, then suppresses noise via channel attention, and finally utilizes spatial attention to let the dominant modality optimize the other.
- **Dense Detail Enhancement Module (DDE)**: Positioned at the low-level features, inspired by ASPP, it uses convolutions with different dilation rates (1×1, 3×3/d=3, 5×5/d=5, 7×7/d=7) + dense connections + VSS blocks to capture spatial structural details.
- **Mutual-Interaction Semantic Module (MIS)**: Positioned at the high-level features, it employs three groups of branches with 3×3 convolutions (dilation rates of 1/2/3) + mutual fusion strategy + channel attention to mine positional semantics.
- **Cross-Layer Connections**: Integrates position and spatial structure information to generate the final saliency map.

### Loss & Training
A combined loss of BCE + boundary IoU + F-measure is employed to supervise five saliency maps.
Training details: SwinTransformer (ImageNet pre-trained) backbone, input size of 384×384, RMSprop optimizer (lr=1e-4, momentum=0.9), trained on a single RTX 4080 GPU.

## Key Experimental Results
Comparison with 27 SOTA methods on three RGB-T datasets (VT5000, VT1000, and VT821):

| Dataset | M↓ | F_β↑ | S_α↑ | E_ξ↑ |
|--------|-----|------|------|------|
| VT5000 | .020 | .910 | .926 | .963 |
| VT1000 | .014 | .923 | .946 | .972 |
| VT821 | .021 | .883 | .921 | .946 |

- Compared to PATNet on VT5000, F_β is improved by 3.4%, E_ξ by 1.2%, and S_α by 1.1%.
- On VT1000, F_β increased by 1.7% and E_ξ by 0.8% compared to PATNet.
- The model parameters are 88M, FLOPs are 143.8G, and the inference speed is 9.4 FPS (RTX 4080). Some speed is sacrificed due to the two-stage design.

### Ablation Study
- Removing the SO module and replacing it with addition/multiplication/concatenation fusion increases the VT5000 MAE to .0217/.0208/.0215 respectively, indicating the necessity of selective fusion.
- Replacing SO with Pixel-wise Soft Gating yields an MAE of .0203 > .0197, which, though better than simple operations, performs worse than SO.
- Fixing the direction to R→T or T→R (skipping the region guidance stage) degrades the MAE to .0215/.0216, proving the value of adaptive selection.
- Removing the DDE module drops S_α from .9261 to .9213; removing MIS drops F_β from .9071 to .8997.
- Simultaneously removing both DDE + MIS drops S_α sharply to .8995, indicating that the two modules are complementary in capturing spatial details and positional semantics.
- Replacing SwinTransformer with ResNet-18/34/50 results in F_β of only .801/.815/.797 respectively; replacing it with frozen SAM/DINO yields worse performance (.822/.856), indicating that foundation models require adapters to adapt to RGB-T tasks.

### Failure Cases
- When the salient target is extremely small or thin, the network struggles to detect it accurately.
- When both the RGB and thermal images are of poor quality, significant noise is introduced even with the region guidance stage, leading to a noticeable decline in detection performance.

## Highlights & Insights
1. **Novel region guidance approach**: Automatically assesses modal reliability via three-way parallel processing and similarity comparison, offering greater flexibility than fixed-weight fusion.
2. **Hierarchical adaptive context interaction**: The CI module uses different kernel sizes for features in different layers, avoiding a "one-size-fits-all" strategy.
3. **Dense connection + VSS block in DDE module**: Introduces Mamba's VSS block into low-level feature optimization for salient object detection, balancing local structures and global dependencies.
4. **Thorough ablation studies**: Validates the individual contributions of each module (CI, SF, SO, DDE, and MIS).
5. **Well-designed fusion loss**: The BCE, boundary IoU, and F-measure losses complement each other, addressing pixel accuracy, boundary quality, and overall F-measure, respectively.

## Limitations & Future Work
- The parameter count and computational complexity of the three-way parallel branches are high (using three identical encoder-decoders), resulting in an inference speed of only 9.4 FPS, which falls short of real-time requirements.
- The similarity calculation relies solely on global mean comparison, which may lose local region-level modal preference information.
- Lightweight versions or real-time application scenarios have not been explored.
- The method is only validated on RGB-T saliency datasets and has not been extended to other multimodal combinations such as RGB-D or RGB-Event.
- The three branches in the region guidance stage must all run during inference, with no option to skip unreliable branches for acceleration.
- The introduction of VSS blocks brings additional sequence scanning overhead, affecting real-time performance on high-resolution inputs.
- Using frozen foundation models (SAM/DINO) as backbones actually decreases performance, indicating the need for dedicated adapter designs.

## Related Work & Insights
- vs. Early methods like **CGFNet/CCFENet**: RSONet addresses modal inconsistency through region guidance rather than simple feature fusion.
- vs. **Samba (CVPR25)**: Samba is based on a pure Mamba architecture, while RSONet combines SwinTransformer and VSS blocks, achieving a lower MAE on VT5000 (0.020 vs 0.021).
- vs. **SAMSOD (TMM26)**: Without requiring SAM priors, RSONet achieves a lower MAE on VT5000 (0.020 vs 0.021), and its parameter count of 88M is significantly more efficient than SAMSOD's 418G FLOPs.
- vs. **ContriNet (TPAMI25)**: RSONet significantly leads in F_β on VT5000 (0.910 vs 0.878).
- vs. **ISMNet (TCSVT25)**: MAE drops from 0.025 to 0.020, and F_β rises from .885 to .910 on VT5000.
- vs. **SPNet (ACM MM23)**: Improvement on VT5000 from 0.024 → 0.020, and on VT1000 from 0.015 → 0.014.

### Insights & Connections
- The mechanism of modality reliability assessment can be generalized to other multimodal fusion tasks such as RGB-D and RGB-Event.
- Although the three-way parallel design is heavy, it provides a "reference standard" (the sum of RGB and T) for different modal combinations. The optimization of this reference standard (e.g., using attention weighting instead of simple addition) is worth exploring.
- The combination of dense connections + VSS blocks in DDE can be applied to other dense prediction tasks such as medical image segmentation.
- The hierarchical allocation strategy of different kernel sizes in the CI module reflects the design concept of "feature level characteristic awareness," which can be transferred to other multi-scale architectures.

## Rating
- Novelty: ⭐⭐⭐⭐ (Novel concept of region-guided selective fusion)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comparison with 27 methods + ablation studies)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure and detailed formulations)
- Value: ⭐⭐⭐⭐ (Provides a new fusion paradigm for multimodal salient object detection)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images](rdnet_region_proportion-aware_dynamic_adaptive_salient_object_detection_network_.md)
- [\[CVPR 2026\] Uncertainty-Aware Modality Fusion for Unaligned RGB-T Salient Object Detection](../../CVPR2026/segmentation/uncertainty-aware_modality_fusion_for_unaligned_rgb-t_salient_object_detection.md)
- [\[CVPR 2025\] G2HFNet: GeoGran-Aware Hierarchical Feature Fusion Network for Salient Object Detection in Optical Remote Sensing Images](binwang2hfnet_geogran-aware_hierarchical_feature_fusion_network_for_salient_obje.md)
- [\[CVPR 2025\] Visual Consensus Prompting for Co-Salient Object Detection](visual_consensus_prompting_for_co-salient_object_detection.md)
- [\[AAAI 2026\] SAM-DAQ: Segment Anything Model with Depth-guided Adaptive Queries for RGB-D Video Salient Object Detection](../../AAAI2026/segmentation/sam-daq_segment_anything_model_with_depth-guided_adaptive_queries_for_rgb-d_vide.md)

</div>

<!-- RELATED:END -->

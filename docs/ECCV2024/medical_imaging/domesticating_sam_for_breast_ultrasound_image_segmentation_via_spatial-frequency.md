---
title: >-
  [Paper Note] Domesticating SAM for Breast Ultrasound Image Segmentation via Spatial-Frequency Fusion and Uncertainty Correction
description: >-
  [ECCV 2024][Medical Imaging][Breast Ultrasound Segmentation] This paper proposes the SF-RecSAM model, which compensates for SAM's deficiencies in low-level feature extraction through a spatial-frequency feature fusion module. Additionally, a Dual False Corrector is designed to identify and correct false positive and false negative regions using uncertainty estimation, significantly outperforming SOTA methods on two breast ultrasound datasets, BUSI and UDIAT.
tags:
  - "ECCV 2024"
  - "Medical Imaging"
  - "Breast Ultrasound Segmentation"
  - "SAM Fine-tuning"
  - "Spatial-Frequency Fusion"
  - "Uncertainty Estimation"
  - "False Positive and False Negative Correction"
date: 2026-05-08
content_hash: 231f9b557d7d7a7f
---

# Domesticating SAM for Breast Ultrasound Image Segmentation via Spatial-Frequency Fusion and Uncertainty Correction

**Conference**: ECCV 2024  
**Code**: [https://github.com/dodooo1/SFRecSAM](https://github.com/dodooo1/SFRecSAM)  
**Area**: Medical Images  
**Keywords**: Breast Ultrasound Segmentation, SAM Fine-tuning, Spatial-Frequency Fusion, Uncertainty Estimation, False Positive and False Negative Correction

## TL;DR

This paper proposes the SF-RecSAM model, which compensates for SAM's deficiencies in low-level feature extraction through a spatial-frequency feature fusion module. Additionally, a Dual False Corrector is designed to identify and correct false positive and false negative regions using uncertainty estimation, significantly outperforming SOTA methods on two breast ultrasound datasets, BUSI and UDIAT.

## Background & Motivation

**Background**: Breast ultrasound image segmentation is a crucial step in breast cancer screening and diagnosis. As a general foundation model for segmentation, the Segment Anything Model (SAM) has demonstrated powerful feature extraction capabilities, leading numerous studies to adapt SAM to the field of medical image segmentation.

**Limitations of Prior Work**: Breast ultrasound images present unique challenges: (1) low contrast—the grayscale difference between the mass and surrounding tissue is small; (2) blurred boundaries—the edges of the mass are unclear or even spiculated; (3) although SAM's ViT encoder is powerful in extracting high-level semantic features, it insufficiently captures low-level features (texture details, boundary structural information) that are critical in breast ultrasound. Existing SAM fine-tuning methods typically only add simple adapters or prompts, failing to fundamentally address the deficiency of low-level features.

**Key Challenge**: SAM gains a powerful high-level semantic understanding from pre-training on natural images, but the segmentation of breast ultrasound relies more heavily on low-level features (such as boundary gradients and texture patterns). The domain gap between the two limits the effectiveness of direct fine-tuning. Moreover, false positive and false negative regions in segmentation results often occur near boundaries with high uncertainty, and existing methods lack a targeted correction mechanism.

**Goal**: (1) How to enhance SAM's perception of low-level features in breast ultrasound? (2) How to effectively identify and correct false positive and false negative regions in segmentation results?

**Key Insight**: Starting from the perspective of frequency domain analysis, the authors argue that frequency domain features can provide texture and boundary information that is difficult to capture in the spatial domain. Furthermore, from the perspective of uncertainty estimation, they suggest that regions with high segmentation uncertainty are precisely the false positive/false negative areas that most require correction.

**Core Idea**: Compensate for SAM's lack of low-level features using spatial-frequency fusion, and correct false positive and false negative regions in segmentation using an uncertainty estimation-based Dual False Corrector.

## Method

### Overall Architecture

SF-RecSAM inherits the overall architecture of SAM (ViT image encoder + prompt encoder + mask decoder) but introduces improvements at two key locations. First, a Spatial-Frequency Feature Fusion Module is introduced into the ViT encoder to fuse spatial domain features with frequency domain features for a more complete feature representation, specifically enhancing low-level textures and boundary information. Then, a Dual False Corrector is integrated after the mask decoder output to locate and correct false positive and false negative regions via uncertainty estimation. The overall input is a breast ultrasound image, and the output is the corrected segmentation mask.

### Key Designs

1. **Spatial-Frequency Feature Fusion Module**:

    - **Function**: Fuses spatial domain features with frequency domain features to compensate for the deficiency of the SAM ViT encoder in low-level feature extraction.
    - **Mechanism**: Performs a two-dimensional Fast Fourier Transform (2D FFT) on the input feature maps to obtain frequency domain representations. High-frequency components in the frequency domain correspond to edges and texture details in the image, while low-frequency components correspond to the overall structure. A learnable frequency filter selectively enhances the frequency components, which are then transformed back into the spatial domain via inverse FFT. Finally, the frequency-enhanced features and the original spatial domain features are adaptively fused using an attention mechanism. The fused features retain SAM's original high-level semantic capability while enhancing low-level texture and boundary perception.
    - **Design Motivation**: In breast ultrasound images, the contrast between the mass and the background is low, making it difficult for spatial-domain convolutions to fully capture boundary information. Frequency-domain analysis allows more direct access to high-frequency information (edges, textures) of the image, compensating for ViT's insufficient perception of local low-level features.

2. **Dual False Corrector**:

    - **Function**: Identifies and corrects false positive (misclassifying background as masses) and false negative (missing masses as background) regions in segmentation results.
    - **Mechanism**: This module consists of two branches: a false positive corrector and a false negative corrector. It first estimates the segmentation uncertainty for each pixel through multiple forward passes (Monte Carlo Dropout or a similar mechanism). High-uncertainty regions are marked as "suspicious regions." For regions initially segmented as foreground but showing high uncertainty, the false positive corrector evaluates whether they need to be flipped to background. For regions initially segmented as background with high uncertainty, the false negative corrector evaluates whether they need to be flipped to foreground. The two correctors generate correction masks respectively, which are ultimately fused with the initial segmentation results to obtain the final segmentation mask.
    - **Design Motivation**: Errors in breast ultrasound segmentation typically concentrate in mass boundary regions, which happen to be the areas with the highest uncertainty. Traditional methods treat segmentation as a deterministic problem, ignoring the differences in the model's prediction confidence across different regions. Using uncertainty as a proxy to locate regions requiring correction enables targeted improvement of segmentation quality.

3. **Domain Adaptation to SAM**:

    - **Function**: Adapts to the breast ultrasound domain without destroying the pre-trained knowledge of SAM.
    - **Mechanism**: Light-weight adapter layers are inserted into each Transformer block of the ViT encoder, while freezing the original parameters of SAM. The spatial-frequency fusion module operates as a parallel branch collaborating with the ViT blocks. The prompt encoder uses automatically generated prompts (such as bounding boxes obtained from rough segmentation results) instead of manual annotations. During training, only the adapter, fusion module, and Dual False Corrector parameters are updated.
    - **Design Motivation**: Directly fine-tuning all parameters of SAM can easily lead to overfitting on small-scale medical datasets. A parameter-efficient fine-tuning strategy retains SAM's pre-trained knowledge while adapting to the new domain.

### Loss & Training

The training loss consists of three components: (1) Main segmentation loss—a combination of Binary Cross-Entropy and Dice Loss to supervise the initial segmentation results; (2) Correction loss—supervising the false positive and false negative corrections respectively; (3) Uncertainty-guided loss—encouraging the model to generate higher uncertainty estimates in truly challenging boundary regions. A two-stage training strategy is adopted, where the base segmentation network is trained first, followed by fixing the base network to train the Dual False Corrector.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (SF-RecSAM) | Prev. SOTA | Gain |
|---|---|---|---|---|
| BUSI | Dice (%) | Significantly leading | Second-best method | Significant improvement |
| BUSI | IoU (%) | Significantly leading | Second-best method | Significant improvement |
| UDIAT | Dice (%) | Significantly leading | Second-best method | Significant improvement |
| UDIAT | IoU (%) | Significantly leading | Second-best method | Significant improvement |

### Ablation Study

| Configuration | Key Metric | Description |
|---|---|---|
| SAM direct fine-tuning (baseline) | Baseline Dice | SAM without special adaptation performs moderately on ultrasound |
| + Spatial-Frequency Fusion | Dice improvement | Frequency domain information significantly helps boundary segmentation |
| + False Positive Corrector | Dice improved, precision increased | Reduces over-segmented regions |
| + False Negative Corrector | Dice improved, recall increased | Reduces missed regions |
| Full SF-RecSAM | Best Dice | All modules work synergistically |
| Spatial fusion only vs. Spatial-Frequency Fusion | Latter is superior | Verifies the value of frequency domain information |

### Key Findings

- Spatial-frequency fusion significantly improves SAM's boundary perception capability on low-contrast ultrasound images.
- The false positive and false negative correctors improve precision and recall respectively, complementing each other.
- Uncertainty estimation effectively locates segmentation error regions, validating the feasibility of utilizing uncertainty-guided post-processing.
- The contribution of frequency domain features to breast ultrasound segmentation is greater than that to natural image segmentation, illustrating strong domain specificity of the method.
- Consistent improvements on both BUSI and UDIAT datasets validate the generalization of the method.

## Highlights & Insights

- Compensating for SAM's low-level feature deficiencies from the frequency domain perspective is an insightful design, as frequency analysis is naturally suited for capturing textures and edges.
- The mechanism of the Dual False Corrector is highly practical; further correcting on top of the initial segmentation results allows it to act as a plug-and-play module.
- Combining uncertainty estimation with segmentation correction provides a "detect first then repair" workflow paradigm.
- Open-sourcing the code benefits subsequent research and replication.

## Limitations & Future Work

- Uncertainty estimation may rely on multiple forward passes, requiring evaluation of inference efficiency.
- Validated only on two breast ultrasound datasets; this could be extended to other ultrasound or medical image modalities.
- Learnable filters in the frequency domain may require readjustment for ultrasound images collected from different scanners.
- Threshold configurations for the Dual False Corrector may need adjustment tailored to different application scenarios.
- Extending frequency analysis to 3D ultrasound or video ultrasound sequences could be considered.

## Related Work & Insights

- **SAM (Segment Anything)**: General foundation model for segmentation; this work adapts it to the medical field.
- **Medical SAM Adapter**: Fine-tunes SAM on medical images via adapters, but does not consider frequency domain information.
- **Application of Frequency Domain Analysis in Medical Images**: Multiple studies show that frequency domain features have unique advantages for low-contrast medical images.
- **Uncertainty Estimation**: Methods like MC-Dropout for estimating model uncertainty have been widely studied; this paper creatively applies them to segmentation correction.
- Insight: For other low-contrast medical image segmentation tasks (e.g., liver ultrasound, thyroid ultrasound), the framework combining spatial-frequency fusion and uncertainty correction holds direct applicability.

## Rating

- Novelty: ⭐⭐⭐⭐ The combined design of spatial-frequency fusion and uncertainty-based dual false correction is innovative.
- Experimental Thoroughness: ⭐⭐⭐ Well-validated on two datasets, but the ablation studies could be more comprehensive.
- Writing Quality: ⭐⭐⭐ Clear description of methods with well-motivated problem definitions.
- Value: ⭐⭐⭐⭐ Provides an effective adaptation scheme for applying SAM in the field of medical ultrasound.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Decoding with Structured Awareness: Integrating Directional, Frequency-Spatial, and Structural Attention for Medical Image Segmentation](../../AAAI2026/medical_imaging/decoding_with_structured_awareness_integrating_directional_frequency-spatial_and.md)
- [\[CVPR 2025\] A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](../../CVPR2025/medical_imaging/a_semi-supervised_framework_for_breast_ultrasound_segmentation_with_training-fre.md)
- [\[ICLR 2026\] HFSTI-Net: Hierarchical Frequency-spatial-temporal Interactions for Video Polyp Segmentation](../../ICLR2026/medical_imaging/hfsti-net_hierarchical_frequency-spatial-temporal_interactions_for_video_polyp_s.md)
- [\[ECCV 2024\] Architecture-Agnostic Untrained Network Priors for Image Reconstruction with Frequency Regularization](architecture-agnostic_untrained_network_priors_for_image_reconstruction_with_fre.md)
- [\[ECCV 2024\] Energy-induced Explicit Quantification for Multi-modality MRI Fusion](energy-induced_explicit_quantification_for_multi-modality_mri_fusion.md)

</div>

<!-- RELATED:END -->

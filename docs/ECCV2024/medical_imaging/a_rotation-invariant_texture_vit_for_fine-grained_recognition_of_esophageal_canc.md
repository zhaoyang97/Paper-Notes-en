---
title: >-
  [Paper Note] A Rotation-Invariant Texture ViT for Fine-Grained Recognition of Esophageal Cancer Endoscopic Ultrasound Images
description: >-
  [ECCV 2024][Medical Imaging][Endoscopic ultrasound] This paper proposes SRRM-ViT, which introduces a Statistical Rotation-invariant Reinforcement Mechanism (SRRM) into ViT to adaptively select key regions and fuse histogram statistical features. This achieves unbiased fine-grained classification of lesions at any radial position in endoscopic ultrasound (EUS) images of esophageal cancer, obtaining significant performance improvements on clinical and public datasets.
tags:
  - "ECCV 2024"
  - "Medical Imaging"
  - "Endoscopic ultrasound"
  - "rotation invariance"
  - "texture features"
  - "Vision Transformer"
  - "esophageal cancer classification"
date: 2026-05-08
content_hash: 868b9397ee65e064
---

# A Rotation-Invariant Texture ViT for Fine-Grained Recognition of Esophageal Cancer Endoscopic Ultrasound Images

**Conference**: ECCV 2024  
**Code**: [https://github.com/tianyiliu-lab/SRRM-ViT](https://github.com/tianyiliu-lab/SRRM-ViT)  
**Area**: Medical Image  
**Keywords**: Endoscopic ultrasound, rotation invariance, texture features, Vision Transformer, esophageal cancer classification

## TL;DR

This paper proposes SRRM-ViT, which introduces a Statistical Rotation-invariant Reinforcement Mechanism (SRRM) into ViT to adaptively select key regions and fuse histogram statistical features. This achieves unbiased fine-grained classification of lesions at any radial position in endoscopic ultrasound (EUS) images of esophageal cancer, obtaining significant performance improvements on clinical and public datasets.

## Background & Motivation

**Background**: Endoscopic ultrasound (EUS) is an important tool for diagnosing esophageal submucosal tumors, capable of perceiving hierarchical changes in the esophageal wall structure. Current clinical diagnosis primarily relies on the experience of physicians for manual interpretation. Deep learning methods have begun to be applied to EUS image analysis, with Vision Transformer (ViT) becoming the current state-of-the-art (SOTA) model architecture due to its global modeling capabilities.

**Limitations of Prior Work**: EUS image analysis faces two core challenges: (1) lesions often disrupt the structural integrity and fine-grained texture information of esophageal layers, making it difficult for conventional feature extraction to accurately capture key information in diseased areas; (2) due to the circular scanning characteristics of EUS imaging, lesions can appear at any radial position in the image, resulting in vastly different appearances under different rotation angles, which greatly increases classification difficulty.

**Key Challenge**: While the self-attention mechanism of standard ViTs can model global dependencies, it lacks invariance to rotational transformations and fails to filter out the large amount of irrelevant background regions (such as normal tissue and ultrasound artifacts) in EUS images, leading to the dilution of fine-grained texture features.

**Goal**: (1) How to adaptively select key regions related to lesions from EUS images and exclude interference from irrelevant information? (2) How to make the model invariant to changes in the radial position of lesions, achieving consistent classification under any rotation angle?

**Key Insight**: The authors observe that histogram statistical features are naturally rotation-invariant—no matter how the image is rotated, the statistical distribution of its pixel values remains unchanged. Therefore, combining histogram statistical features with the self-attention mechanism of ViTs can simultaneously leverage the global modeling capability of Transformers and the rotation invariance of statistical features.

**Core Idea**: By integrating rotation-invariant histogram statistical features into the self-attention mechanism of ViTs, alongside adaptive key region selection, the model achieves unbiased fine-grained recognition of lesions at arbitrary positions in EUS images.

## Method

### Overall Architecture

The overall pipeline of SRRM-ViT is as follows: the input EUS image first passes through the Adaptive Region Selection (ARS) module to filter out the most disease-related image regions, eliminating irrelevant background and artifacts. Then, the selected regions are fed into the enhanced ViT backbone, where the core innovation is the Statistical Rotation-invariant Reinforcement Mechanism (SRRM). In the self-attention calculation of each Transformer block, SRRM introduces additional histogram-based statistical features, which are naturally invariant to rotational transformations, thereby making the entire network robust to radial position changes of lesions. Finally, the reinforced features are passed through a classification head to output the fine-grained classification results of esophageal cancer subtypes.

### Key Designs

1. **Adaptive Region Selection (ARS)**:

    - **Function**: Automatically identify and select regions most relevant to lesion diagnosis from EUS images, excluding interference from irrelevant information.
    - **Mechanism**: Use an attention scoring mechanism to evaluate the importance of different image regions, selecting several regions with the highest scores as the input for subsequent analysis. Specifically, the input image undergoes initial feature extraction to calculate the importance weight of each patch, and then the top-K key regions are selected based on weight ranking. This approach avoids undifferentiated processing of the entire image.
    - **Design Motivation**: EUS images contain a large number of non-lesion regions (healthy tissue, ultrasound probe artifacts, boundary noise, etc.). If the entire image is directly passed into the classifier, these irrelevant details heavily interfere with the extraction of fine-grained texture features, degrading classification accuracy.

2. **Statistical Rotation-invariant Reinforcement Mechanism (SRRM)**:

    - **Function**: Integrate rotation-invariant statistical texture features into the self-attention mechanism of ViT, enabling the model to unbiasedly recognize lesions at any radial position.
    - **Mechanism**: Calculate local histogram statistical features (e.g., mean, variance, skewness of the pixel value distribution) for each patch, which naturally possess rotation invariance. Then, encode these statistical features into vectors to inject as auxiliary keys and values into the standard self-attention calculation. Specifically, the modified attention calculation becomes $\text{Attn}(Q, K+K_s, V+V_s)$, where $K_s$ and $V_s$ denote the auxiliary key and value derived from the statistical features, respectively. Consequently, when calculating relationships between patches, the model considers both spatial positional features and rotation-invariant statistical texture features.
    - **Design Motivation**: The patch embedding and positional encoding of standard ViTs are sensitive to rotation—the same lesion will generate different feature representations after being rotated. In contrast, histogram statistical features are unaffected by spatial transformations; injecting them into the attention mechanism provides a stable "anchor" for the model, ensuring consistent representations for the same type of lesion under different rotation angles.

3. **Fine-Grained Texture Feature Enhancement**:

    - **Function**: Capture subtle texture differences across various layers of the esophageal wall in EUS images, supporting the fine-grained differentiation of lesion subtypes.
    - **Mechanism**: Further extract multi-scale texture descriptors based on the histogram statistical features. By calculating statistical features for patches with different spatial resolutions, multi-level texture representations are constructed. These features are processed separately in the multi-head attention of the Transformer and finally fused into a comprehensive fine-grained representation.
    - **Design Motivation**: Different subtypes of esophageal tumors (such as leiomyoma, stromal tumors, cysts, etc.) are principally differentiated in EUS images by interlayer texture differences. These differences are often very subtle and require fine-grained texture analysis to be captured.

### Loss & Training

The model is trained end-to-end using standard cross-entropy loss, with an auxiliary regional correlation loss applied to the key region selection module to encourage the model to focus on lesion areas instead of the background. Regarding the training strategy, data augmentations (including random rotation to increase training data diversity) and multi-stage training are adopted—first pre-training the ViT backbone, and then jointly training the SRRM module.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (SRRM-ViT) | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| Clinical Internal Dataset | Accuracy | Significant Improvement | ViT baseline | +Clear Improvement |
| Public EUS Dataset | Accuracy | SOTA | Various baseline methods | Consistently outperforms baseline methods |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| ViT baseline | Baseline accuracy | Without any enhancement modules |
| + ARS | Accuracy improvement | Irrelevant information is filtered out after adding key region selection |
| + SRRM | Further accuracy improvement | Lesion recognition across different angles is more consistent after adding rotation invariance enhancement |
| + ARS + SRRM (Full) | Highest accuracy | The synergy of both modules yields the best performance |

### Key Findings

- The robustness of the SRRM module against rotational transformations is significantly superior to standard ViT, with minimal fluctuation in accuracy during manual rotation tests.
- Adaptive region selection effectively focuses on lesion areas, and visualization results show high consistency between the selected regions and clinical annotations.
- The fusion of histogram statistical features with self-attention outperforms simple feature concatenation or addition.
- The model demonstrates consistent performance advantages across various fine-grained classification tasks for esophageal cancer subtypes.

## Highlights & Insights

- Clever utilization of the rotation invariance of histogram statistical features, which is an elegant and physically intuitive design—statistics do not change with spatial transformations.
- Injecting rotation invariance directly into the Transformer's attention computation is more efficient than post-processing or data augmentation.
- Adaptive region selection + rotation-invariant feature enhancement form a complete EUS image analysis framework.
- The method possesses strong potential for domain transfer and can be extended to other circular-scanning medical imaging modalities.

## Limitations & Future Work

- Although the code is open-source, it is labeled "coming soon," so actual usability remains to be verified.
- The impact of the number of histogram bins and the choice of statistical descriptors on performance requires further systematic investigation.
- The current method assumes rotation is the primary transformation pattern, but EUS images might also exhibit other deformations (e.g., compression, stretching); future work can extend to more general transformation invariance.
- Validation is primarily conducted on esophageal cancer EUS data; the generalization effect to other ultrasound imaging scenarios remains unknown.
- The cohort size is relatively limited, and large-scale clinical validation is still required.

## Related Work & Insights

- The application of Vision Transformers (ViTs) in medical imaging has become a trend, yet rotation invariance remains an overlooked key issue.
- Traditional texture analysis methods (such as LBP, GLCM) possess inherent invariances. Integrating this philosophy into a deep learning framework is an interesting fusion.
- The region selection strategy in this work shares similarities with attention-based Multi-Instance Learning (MIL).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Integrating histogram statistical rotation invariance into the ViT attention mechanism is a clever innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on clinical and public data, though the scale of the public dataset and compared methods could be expanded.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and reasonable methodology description.
- Value: ⭐⭐⭐⭐ Possesses practical clinical value for EUS image analysis; the methodological concept can be transferred to other fields.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] DeNAS-ViT: Data Efficient NAS-Optimized Vision Transformer for Ultrasound Image Segmentation](../../AAAI2026/medical_imaging/denas-vit_data_efficient_nas-optimized_vision_transformer_for_ultrasound_image_s.md)
- [\[CVPR 2025\] Prototype-Based Knowledge Guidance for Fine-Grained Structured Radiology Reporting](../../CVPR2025/medical_imaging/prototype-based_knowledge_guidance_for_fine-grained_structured_radiology_reporti.md)
- [\[CVPR 2025\] Unleashing Video Language Models for Fine-grained HRCT Report Generation](../../CVPR2025/medical_imaging/unleashing_video_language_models_for_fine-grained_hrct_report_generation.md)
- [\[ECCV 2024\] Topology-Preserving Downsampling of Binary Images](topology-preserving_downsampling_of_binary_images.md)
- [\[ECCV 2024\] Shape-Guided Configuration-Aware Learning for Endoscopic-Image-Based Pose Estimation of Flexible Robotic Instruments](shape-guided_configuration-aware_learning_for_endoscopic-image-based_pose_estima.md)

</div>

<!-- RELATED:END -->

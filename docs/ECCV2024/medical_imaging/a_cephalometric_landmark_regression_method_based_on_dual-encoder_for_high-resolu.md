---
title: >-
  [Paper Note] A Cephalometric Landmark Regression Method Based on Dual-Encoder for High-Resolution X-Ray Image
description: >-
  [ECCV 2024][Medical Imaging][Cephalometric] This paper proposes D-CeLR, an end-to-end regression method based on a dual-encoder architecture. Utilizing only Transformer encoders, it designs a three-stage framework comprising feature extraction, a reference encoder, and a finetune encoder to achieve coarse-to-fine cephalometric landmark detection, significantly outperforming existing SOTA methods in Mean Radical Error (MRE) and 2mm Success Detection Rate (SDR) metrics.
tags:
  - "ECCV 2024"
  - "Medical Imaging"
  - "Cephalometric"
  - "Landmark Detection"
  - "Dual-Encoder"
  - "Transformer"
  - "High-Resolution X-Ray"
date: 2026-05-08
content_hash: 62276c5e2186cbcd
---

# A Cephalometric Landmark Regression Method Based on Dual-Encoder for High-Resolution X-Ray Image

**Conference**: ECCV 2024  
**Code**: [https://github.com/huang229/D-CeLR](https://github.com/huang229/D-CeLR)  
**Area**: Medical Image  
**Keywords**: Cephalometric, Landmark Detection, Dual-Encoder, Transformer, High-Resolution X-Ray

## TL;DR

This paper proposes D-CeLR, an end-to-end regression method based on a dual-encoder architecture. Utilizing only Transformer encoders, it designs a three-stage framework comprising feature extraction, a reference encoder, and a finetune encoder to achieve coarse-to-fine cephalometric landmark detection, significantly outperforming existing SOTA methods in Mean Radical Error (MRE) and 2mm Success Detection Rate (SDR) metrics.

## Background & Motivation

**Background**: Cephalometric landmark detection is a crucial step in orthodontic diagnosis and treatment planning. Clinically, it requires precisely locating 19-29 anatomical landmarks on lateral cephalometric radiographs to measure the angular and distance relationships between bones and teeth. Current approaches are mainly categorized into two groups: (1) heatmap-regression-based methods, which locate landmarks by predicting Gaussian heatmaps for each point; (2) cascade-model-based methods, which employ a multi-model sequential strategy to perform coarse-to-fine localization.

**Limitations of Prior Work**: Existing high-precision methods typically rely on multi-model cascades—first using one model for coarse localization, followed by one or more models for refinement. Although effective, this cascade strategy introduces three issues: (1) complex training processes, where each stage requires separate training, leading to accumulated errors across stages; (2) difficult deployment since multiple models need to be managed and maintained; (3) neglected inter-landmark spatial dependencies, as each landmark is often processed independently.

**Key Challenge**: High-precision landmark detection requires a coarse-to-fine multi-stage process, but multi-model cascades break end-to-end differentiability, preventing global optimization. How to realize a coarse-to-fine localization strategy within a single model while maintaining the advantages of end-to-end training remains a key challenge.

**Goal**: (1) To design an end-to-end differentiable single model to replace the multi-model cascade strategy; (2) To enable coarse-to-fine localization capabilities within this single model; (3) To allow the model to naturally learn the spatial dependencies among landmarks.

**Key Insight**: The authors notice that the Transformer encoder structure naturally possesses the capability to model long-range dependencies (via self-attention), which can be used to capture the spatial relationships among landmarks. By designing two encoders in series—a reference encoder for coarse localization and a finetune encoder for refinement—one can implement the entire coarse-to-fine pipeline in a single differentiable model.

**Core Idea**: Two serialized Transformer encoders are employed to handle coarse localization and refinement, respectively, achieving coarse-to-fine landmark detection in an end-to-end framework, while naturally modeling the spatial dependencies among landmarks via self-attention.

## Method

### Overall Architecture

The overall architecture of D-CeLR consists of three main modules: (1) Feature Extractor: extracts multi-scale features from high-resolution X-ray images; (2) Reference Encoder: treats landmarks as queries to extract initial position predictions from feature maps via cross-attention, completing coarse localization; (3) Finetune Encoder: receives the coarse localization results and refines landmark positions through further attention mechanisms. The input is a high-resolution lateral X-ray image, and the output consists of the 2D coordinates of all landmarks.

### Key Designs

1. **Feature Extractor**:

    - **Function**: Extracts multi-scale visual features from high-resolution X-ray images.
    - **Mechanism**: A pretrained CNN backbone (such as ResNet or HRNet) is used to extract multi-scale feature maps from the input high-resolution X-ray image. To handle high-resolution inputs (e.g., $1935 \times 2400$ pixels), the feature extractor adopts a multi-scale feature fusion strategy, aggregating feature maps of different resolutions to retain high-resolution spatial details for precise localization while utilizing low-resolution features to provide global contextual information. The feature maps are projected to form key/value sequences for subsequent encoder modules.
    - **Design Motivation**: High-resolution X-ray images contain rich anatomical details, but directly feeding ultra-high-resolution images into a Transformer causes a computational explosion. Multi-scale feature extraction controls computational costs while preserving spatial accuracy.

2. **Reference Encoder**:

    - **Function**: Performs coarse localization of all landmarks.
    - **Mechanism**: A set of learnable landmark queries is maintained, with each query corresponding to an anatomical landmark to be detected. These queries are processed through a multi-layer Transformer encoder structure, which contains: (1) self-attention layers—allowing different landmark queries to exchange information via attention to model spatial dependencies among landmarks (e.g., the fixed anatomical distance relationship between the nose tip and maxillary landmarks); (2) cross-attention layers—where each landmark query performs cross-attention with the feature maps output by the feature extractor to retrieve information associated with the landmark's position. After multi-layer processing, a regression head maps each query to 2D coordinate predictions, yielding coarse localization results.
    - **Design Motivation**: Through the self-attention mechanism, the model can predict all landmarks self-consistently—the prediction of each landmark refers to the predictions of other landmarks, leveraging the spatial constraints of the anatomical structure. This achieves global consistency that is difficult to realize in cascade methods.

3. **Finetune Encoder**:

    - **Function**: Refines landmark coordinates based on the coarse localization results.
    - **Mechanism**: It receives the coarse localization results and updated query vectors from the reference encoder. The key improvement is utilizing the coarse coordinates to crop local region-of-interest (RoI) features around each landmark from the high-resolution feature maps. The finetune encoder then computes more refined cross-attention on these local features. Since the search space is narrowed down to small neighborhoods around the coarse coordinates, attention can focus on pixel-level fine-grained localization. Meanwhile, the self-attention layer continues to maintain global consistency among landmarks. Finally, the regression head outputs refined coordinate offsets, which are added to the coarse predictions to obtain final coordinates.
    - **Design Motivation**: Coarse localization significantly narrows down the search range, allowing the finetune encoder to focus on making fine adjustments within local regions. Achieving this coarse-to-fine strategy within a single, differentiable model enables end-to-end backpropagation, eliminating error accumulation common in cascade methods.

### Loss & Training

A two-stage joint loss is used during training: $\mathcal{L} = \mathcal{L}_{coarse} + \lambda \cdot \mathcal{L}_{fine}$. The losses for both stages are L1 or L2 regression losses between predicted coordinates and ground truth coordinates. $\lambda$ is a balancing weight. Training is conducted end-to-end, allowing gradients to propagate back from the finetune encoder to the reference encoder and the feature extractor. Data augmentation includes random rotation, scaling, flipping, and brightness adjustments. Standard datasets, ISBI 2015 and ISBI 2023, are used for experiments.

## Key Experimental Results

### Main Results

Evaluation is performed on the ISBI 2015 challenge dataset, using Mean Radical Error (MRE, mm) and 2mm Success Detection Rate (SDR, %) as metrics.

| Method | MRE (mm) ↓ | 2mm SDR (%) ↑ | Description |
|------|-----------|---------------|------|
| Cascade heatmap methods | ~1.5-1.7 | ~80-84 | Multi-model sequence |
| DETR-based methods | ~1.4-1.5 | ~83-85 | End-to-end detection |
| D-CeLR (Ours) | **Optimal (<1.3)** | **Optimal (>87)** | Significantly outperforms |

| Dataset | Metric | D-CeLR | Prev. SOTA | Gain |
|--------|------|--------|----------|------|
| ISBI 2015 Test1 | MRE ↓ | Optimal | Second Best | Significantly reduced |
| ISBI 2015 Test2 | MRE ↓ | Optimal | Second Best | Significantly reduced |
| ISBI 2023 | MRE ↓ | Optimal | Second Best | Superior cross-domain |

Additionally, the computational resource consumption of D-CeLR is lower than that of cascade methods.

### Ablation Study

| Configuration | MRE (mm) ↓ | Description |
|------|-----------|------|
| Feature extractor + regression head only | Higher | No structural reasoning |
| + Reference Encoder (Coarse localization) | Significantly reduced | Transformer attention is effective |
| + Finetune Encoder (Full model) | Lowest | Coarse-to-fine brings further improvement |
| Without self-attention (Cross-attention only) | Increased | Inter-landmark dependency is important |
| Different encoder layers | 3-4 layers optimal | Diminishing returns beyond that |

### Key Findings

- The coarse-to-fine strategy of the dual-encoder remains highly effective in an end-to-end model and outperforms direct regression using a single encoder.
- The self-attention mechanism among landmarks significantly contributes to the final accuracy, indicating that the spatial constraints of the anatomical structure are successfully captured by the model.
- End-to-end training significantly outperforms stage-wise training, confirming the importance of the end-to-end differentiable design.
- Good performance is also achieved on cross-device data in ISBI 2023, showcasing the domain generalization capability of the method.

## Highlights & Insights

- **Elegant "Encoder-Only" Design**: Without requiring a decoder, it accomplishes coarse-to-fine localization using only the self-attention and cross-attention of Transformer encoders, presenting a very clean architecture.
- **End-to-End Replacement for Cascade**: It unifies the commonly used cascade localization strategy into a single end-to-end model, simplifying deployment while achieving better performance.
- **Natural Modeling of Landmark Dependencies**: The self-attention mechanism enables the model to automatically learn anatomical constraints without handcrafted anatomical knowledge.
- **High Practical Value**: Directly addresses pain points in clinical orthodontic diagnosis with open-source code.

## Limitations & Future Work

- The size of the training dataset is relatively small (only 150 training images in ISBI 2015), and its performance on larger-scale datasets remains to be validated.
- Robustness to extreme cases (such as severe deformities or pediatric skull structures with unconventional anatomy) needs further evaluation.
- Currently, the method only processes 2D lateral cephalograms and has not been extended to 3D CT/CBCT cephalometry.
- The local RoI size in the finetune encoder is a hyperparameter and may require tuning for X-ray images of different resolutions.
- Exploitation of integration with state-of-the-art vision foundation models (such as SAM, DINOv2) has not yet been explored.

## Related Work & Insights

- **vs Heatmap Regression Methods**: Heatmap methods require high-resolution outputs (same size as input), leading to high computational overhead. D-CeLR directly regresses coordinates, making it more flexible with respect to input resolution.
- **vs DETR / Deformable DETR**: D-CeLR's query mechanism is similar to DETR, but enhanced for localization accuracy (dual-encoder + refinement) and does not require Hungarian matching (as the number of landmarks is fixed and matched).
- **vs Cascade Methods (such as CC2D)**: Cascade methods use multiple independent models to break down the localization task into coarse-fine stages. D-CeLR realizes the same effect in a single model with end-to-end training.

## Rating

- Novelty: ⭐⭐⭐ The dual-encoder design is somewhat engineering-oriented but effective; the core idea is to replace cascade architectures with an end-to-end system.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete comparative and ablation experiments are provided on standard challenge datasets, including cross-domain validation.
- Writing Quality: ⭐⭐⭐ Clearly described, although the method section could be more concise.
- Value: ⭐⭐⭐⭐ Has practical value for medical image landmark detection, and the end-to-end design simplifies deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MAISI-v2: Accelerated 3D High-Resolution Medical Image Synthesis with Rectified Flow and Region-specific Contrastive Loss](../../AAAI2026/medical_imaging/maisi-v2_accelerated_3d_high-resolution_medical_image_synthesis_with_rectified_f.md)
- [\[ECCV 2024\] Radiative Gaussian Splatting for Efficient X-ray Novel View Synthesis](radiative_gaussian_splatting_for_efficient_x-ray_novel_view_synthesis.md)
- [\[CVPR 2026\] DARC: Dual Adjustment Reasoning with Counterfactuals for Trustworthy Chest X-ray Classification](../../CVPR2026/medical_imaging/darc_dual_adjustment_reasoning_with_counterfactuals_for_trustworthy_chest_x-ray_.md)
- [\[AAAI 2026\] A Disease-Aware Dual-Stage Framework for Chest X-ray Report Generation](../../AAAI2026/medical_imaging/a_disease-aware_dual-stage_framework_for_chest_x-ray_report_.md)
- [\[CVPR 2025\] EquivAnIA: A Spectral Method for Rotation-Equivariant Anisotropic Image Analysis](../../CVPR2025/medical_imaging/equivania_a_spectral_method_for_rotation-equivariant_anisotropic_image_analysis.md)

</div>

<!-- RELATED:END -->

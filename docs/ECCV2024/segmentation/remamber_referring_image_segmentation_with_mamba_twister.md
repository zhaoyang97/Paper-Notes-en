---
title: >-
  [Paper Note] ReMamber: Referring Image Segmentation with Mamba Twister
description: >-
  [ECCV 2024][Segmentation][Referring Image Segmentation] This paper introduces the Mamba architecture to the Referring Image Segmentation (RIS) task for the first time. It proposes the Mamba Twister module, which achieves efficient vision-language feature fusion through a "twisting" mechanism of channel and spatial scanning. It achieves competitive results surpassing Transformer-based methods on RefCOCO/RefCOCO+/G-Ref benchmarks while maintaining linear computational complexit…
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "Referring Image Segmentation"
  - "Mamba"
  - "Multimodal Fusion"
  - "State Space Models"
  - "Vision-Language Interaction"
date: 2026-05-08
content_hash: c21686e12f99d108
---

# ReMamber: Referring Image Segmentation with Mamba Twister

**Conference**: ECCV 2024  
**arXiv**: [2403.17839](https://arxiv.org/abs/2403.17839)  
**Code**: [https://github.com/yyh-rain-song/ReMamber](https://github.com/yyh-rain-song/ReMamber)  
**Area**: Image Segmentation / Multimodal VLM  
**Keywords**: Referring Image Segmentation, Mamba, Multimodal Fusion, State Space Models, Vision-Language Interaction

## TL;DR

This paper introduces the Mamba architecture to the Referring Image Segmentation (RIS) task for the first time. It proposes the Mamba Twister module, which achieves efficient vision-language feature fusion through a "twisting" mechanism of channel and spatial scanning. It achieves competitive results surpassing Transformer-based methods on RefCOCO/RefCOCO+/G-Ref benchmarks while maintaining linear computational complexity.

## Background & Motivation

**Background**: Referring Image Segmentation (RIS) requires locating and segmenting specific targets in an image based on natural language descriptions, which is a core task in multimodal understanding. Current mainstream methods are based on the Transformer architecture (e.g., LAVT, CRIS, CGFormer), which models the interactions between vision and language through attention mechanisms.

**Limitations of Prior Work**: The attention mechanism of Transformers inherently incurs quadratic computational and memory complexity, leading to severe resource consumption when processing large images and long textual descriptions. This is particularly pronounced when capturing long-range vision-language dependencies, limiting the deployment of models in resource-constrained scenarios.

**Key Challenge**: As an emerging State Space Model (SSM), Mamba offers a linear-complexity alternative. However, directly applying Mamba to multimodal interaction faces a fundamental challenge: Mamba's scanning operations suffer from insufficient interaction among tokens across different channels, hindering effective fusion of information from different modalities. Simply prepending text tokens to image tokens (In-context Conditioning) leads to the dilution of textual information during the processing of long image sequences.

**Goal**: (1) How to achieve effective multimodal feature fusion within the Mamba architecture; (2) how to overcome Mamba's inherent limitation of insufficient inter-channel interaction; and (3) how to achieve Transformer-level segmentation accuracy while maintaining linear complexity.

**Key Insight**: The authors observe that information flow in Mamba is primarily along the spatial dimension, with channels being nearly independent. Therefore, they propose arranging multimodal features along the channel dimension into a "hybrid feature cube". This cube is then "twisted" by alternately scanning along both channel and spatial dimensions, forcing features from different modalities to interlace and fuse during scanning.

**Core Idea**: Constructing a hybrid cube of vision-text-interaction three-way features and utilizing a twisting mechanism of Channel Scan + Spatial Scan to replace the attention-based fusion of Transformers, thereby achieving linear-complexity multimodal feature interaction.

## Method

### Overall Architecture

The input of ReMamber is an image and a textual description, and the output is the segmentation mask of the corresponding target. The overall architecture is stacked with multiple Mamba Twister Blocks (4 blocks, with the VSS Layer count configured as 2-2-15-2). Each block consists of several Visual State Space (VSS) Layers and a Twisting Layer. The VSS Layer is responsible for extracting spatial visual features, while the Twisting Layer injects text conditions and achieves multimodal feature fusion. The intermediate features output from each block are fed into a decoder to generate the final segmentation mask.

### Key Designs

1. **Visual State Space (VSS) Layer**:

    - **Function**: Processes 2D image features along the spatial dimension.
    - **Mechanism**: Since SSMs are originally designed for 1D causal sequential data, applying them directly to 2D images yields suboptimal results. The VSS Layer employs the Cross-Scan Module (CSM) proposed in VMamba, which flattens image patches into sequences and scans them along four directions, ensuring that information from all pixels is integrated during feature transformation. This effectively replaces the self-attention layer in ViTs with Mamba.
    - **Design Motivation**: Maintains the linear complexity advantage of Mamba while adapting to the non-causal nature of 2D images.

2. **Hybrid Feature Cube Construction (Hybrid Feature Cube)**:

    - **Function**: Explicitly constructs fine-grained correspondences between images and text.
    - **Mechanism**: Computes global and local interactions separately. Global interaction pools the text sequence into a global vector $\mathbf{F}_t^{CLS}$ and expands it to the size of the image features. Local interaction calculates the correlation between each image patch and each text token via matrix multiplication $\mathbf{F}_c = \mathbf{F}_i \mathbf{W}_i \cdot (\mathbf{F}_t \mathbf{W}_t)^T$, followed by a convolutional projection. Finally, visual features, global text features, and local interaction features are concatenated along the channel dimension to form a hybrid cube $\mathbf{F}_{cube} \in \mathbb{R}^{h \times w \times (C_i + C_t + C_c)}$.
    - **Design Motivation**: Ensures that each visual token simultaneously perceives both the global semantics and the fine-grained lexical associations of the text through dual global + local interactions, preventing information loss from a single representation.

3. **Twisting Mechanism (Channel Scan + Spatial Scan)**:

    - **Function**: Promotes both intra-modal and inter-modal information exchange within the hybrid feature cube.
    - **Mechanism**: First performs Channel Scan—treating the hybrid feature cube as an ordered sequence along the channel dimension and scanning it with a 1D SSM to promote cross-channel (i.e., cross-modal) feature fusion. Then, Spatial Scan is conducted—using a VSS Layer to perform 2D scanning along the spatial dimension, propagating the fused information within each modality. This process is formalized as $\mathbf{F}_{out} = \text{SSM}_{spatial}(\text{SSM}_{channel}(\mathbf{F}_{cube}))$. PCA visualization shows that Channel Scan aggregates features from different modalities close to the text distribution, and Spatial Scan then redistributes the fused features.
    - **Design Motivation**: Addresses the core deficiency of insufficient inter-channel interaction in Mamba. Through the "twisting" of the two-step scan, information from different modalities is alternately interwoven along both the channel and spatial dimensions, achieving deep integration.

### Loss & Training

Trained end-to-end with a simple convolutional decoder. The loss function is a standard segmentation loss (BCE + Dice loss). The VMamba backbone is initialized with ImageNet pre-trained weights, and the input resolution is set to 480 for SOTA comparison experiments.

## Key Experimental Results

### Main Results

| Dataset | Metric (oIoU) | ReMamber | LAVT (Swin-B) | CRIS (CLIP-R101) | Gain vs LAVT |
|--------|-----------|----------|---------------|------------------|-------------|
| RefCOCO val | oIoU | 74.54 | 72.73 | 70.47 | +1.81 |
| RefCOCO testA | oIoU | 76.74 | 75.82 | 73.18 | +0.92 |
| RefCOCO testB | oIoU | 70.89 | 68.79 | 66.10 | +2.10 |
| RefCOCO+ val | oIoU | 65.00 | 62.14 | 62.27 | +2.86 |
| RefCOCO+ testA | oIoU | 70.78 | 68.38 | 68.08 | +2.40 |
| G-Ref val | oIoU | 63.9 | 61.24 | 59.87 | +2.66 |

### Ablation Study

| Fusion Method | RefCOCO val mIoU | RefCOCO+ val mIoU | G-Ref val mIoU |
|---------|-----------------|-------------------|---------------|
| Attention-based | 65.3 | 54.0 | 50.5 |
| In-Context | 69.1 | 58.4 | 54.8 |
| Norm Adaptation | 70.2 | 60.3 | 59.3 |
| Mamba Twister | **71.6** | **61.6** | **61.1** |

| Scan Configuration | RefCOCO val mIoU | Description |
|---------|-----------------|------|
| Channel Scan only | 62.3 | Channel scan only, severe performance drop |
| Spatial Scan only | 70.0 | Spatial scan only, close to the complete model |
| Parallel | 71.0 | Parallel summation of both |
| Channel→Spatial | **71.6** | Optimal configuration |

### Key Findings

- Spatial Scan contributes the most to performance. Removing the Channel Scan alone still yields acceptable performance, but removing the Spatial Scan leads to a severe degradation.
- Attention-based Conditioning performs the worst in the Mamba architecture, suggesting a fundamental conflict between cross-attention and the sequential dependency characteristics of Mamba.
- Both global and local interactions are indispensable; removing global features decreases the RefCOCO val mIoU from 71.6 to 69.9.
- Both inference speed and training memory are superior to LAVT of equivalent scale, with a particularly pronounced advantage at high resolutions (1024).

## Highlights & Insights

- **Ingenious Design of the Twisting Mechanism**: By arranging features of different modalities along the channel dimension and using two SSM scans of different dimensions to achieve "twisting" fusion, this design is both simple and effective. It avoids the extra overhead of introducing attention mechanisms in Mamba, essentially leveraging SSM's sequence modeling capability for cross-modal communication.
- **Insight on Cross-Attention and Mamba Incompatibility**: Experiments reveal a fundamental contradiction between attention mechanisms and the sequence dependency characteristics of Mamba, which provides important guidance for future multimodal research based on Mamba.
- **Transferability of the Twisting Paradigm**: The fusion paradigm of a hybrid feature cube + multi-dimensional scanning can be extended to other multimodal or multi-scale feature fusion tasks, such as spatio-temporal fusion in video understanding, multi-sensor fusion, etc.

## Limitations & Future Work

- The decoder structure is simple, using only a few convolutional layers, and lacks fine-grained multi-scale feature aggregation capabilities.
- The choice of text encoder and its pre-training method have not been thoroughly explored, which might limit the upper bound of language understanding.
- The sequence order of Channel Scan is based on a fixed channel arrangement. Whether a more optimal ordering strategy exists warrants further research.
- Comparison with large-scale vision-language pre-trained models (e.g., SAM, SEEM) has not been conducted.

## Related Work & Insights

- **vs LAVT**: LAVT uses Swin Transformer + a language-aware fusion module. Ours replaces this with Mamba + Twisting, achieving superior results across all datasets with higher computational efficiency.
- **vs CRIS**: CRIS uses a CLIP visual encoder + text-to-pixel contrastive learning. Ours does not rely on CLIP pre-training, surpassing it using VMamba pre-trained on ImageNet.
- **vs CGFormer**: A Transformer query-based framework that treats segmentation as a proposal-level classification problem, whereas ours adopts a dense prediction paradigm.

## Rating

- Novelty: ⭐⭐⭐⭐ First to apply Mamba to RIS and propose an effective multimodal fusion scheme, though the overall framework design is relatively intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive comparison on three datasets, in-depth analysis of four fusion methods, extensive ablation studies, as well as distribution visualization and attention map analysis.
- Writing Quality: ⭐⭐⭐⭐ Well-structured, logical motivation, and intuitive diagrams.
- Value: ⭐⭐⭐⭐ Opens up new directions for Mamba's application in multimodal tasks and provides a comparative analysis of various fusion designs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SaFiRe: Saccade-Fixation Reiteration with Mamba for Referring Image Segmentation](../../NeurIPS2025/segmentation/safire_saccade-fixation_reiteration_with_mamba_for_referring_image_segmentation.md)
- [\[ICCV 2025\] Latent Expression Generation for Referring Image Segmentation and Grounding](../../ICCV2025/segmentation/latent_expression_generation_for_referring_image_segmentation_and_grounding.md)
- [\[ICLR 2026\] AMLRIS: Alignment-aware Masked Learning for Referring Image Segmentation](../../ICLR2026/segmentation/amlris_alignment-aware_masked_learning_for_referring_image_segmentation.md)
- [\[ICML 2025\] QMamba: On First Exploration of Vision Mamba for Image Quality Assessment](../../ICML2025/segmentation/qmamba_on_first_exploration_of_vision_mamba_for_image_quality_assessment.md)
- [\[AAAI 2026\] RS2-SAM2: Customized SAM2 for Referring Remote Sensing Image Segmentation](../../AAAI2026/segmentation/rs2-sam2_customized_sam2_for_referring_remote_sensing_image_segmentation.md)

</div>

<!-- RELATED:END -->

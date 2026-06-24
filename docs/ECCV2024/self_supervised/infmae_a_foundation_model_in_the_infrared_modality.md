---
title: >-
  [Paper Note] InfMAE: A Foundation Model in the Infrared Modality
description: >-
  [ECCV 2024][Self-Supervised Learning][Infrared Foundation Model] This paper proposes InfMAE, the first foundation model designed specifically for the infrared modality. By constructing the Inf30 dataset with 300,000 infrared images, and designing an information-aware masking strategy along with a multi-scale encoder, the proposed method outperforms existing state-of-the-art approaches in three downstream tasks: infrared semantic segmentation, object detection…
tags:
  - "ECCV 2024"
  - "Self-Supervised Learning"
  - "Infrared Foundation Model"
  - "Masked Autoencoder"
  - "Information-Aware Masking"
  - "Multi-Scale Encoder"
  - "Self-Supervised Pre-training"
date: 2026-05-08
content_hash: e7062da5ae7f0a5e
---

# InfMAE: A Foundation Model in the Infrared Modality

**Conference**: ECCV 2024  
**arXiv**: [2402.00407](https://arxiv.org/abs/2402.00407)  
**Code**: None  
**Area**: Self-Supervised Learning / Infrared Vision  
**Keywords**: Infrared Foundation Model, Masked Autoencoder, Information-Aware Masking, Multi-Scale Encoder, Self-Supervised Pre-training

## TL;DR

This paper proposes InfMAE, the first foundation model designed specifically for the infrared modality. By constructing the Inf30 dataset with 300,000 infrared images, and designing an information-aware masking strategy along with a multi-scale encoder, the proposed method outperforms existing state-of-the-art approaches in three downstream tasks: infrared semantic segmentation, object detection, and small object detection.

## Background & Motivation

Infrared imaging utilizes thermal radiation emitted by objects, offering indispensable advantages under adverse environments such as low-light condition, complete darkness, and smoke, and is widely deployed in fields like autonomous driving, surveillance, and military reconnaissance. However, the current infrared vision community faces two core challenges:

**Lack of Infrared Foundation Models**: While foundation models exist for other modalities (e.g., MAE, DINO, and BEiT for visible light; VideoMAE for video; Scale-MAE for remote sensing; Point-M2AE for point clouds), there is no dedicated foundation model for the infrared modality. Direct transfer of visible-light MAE models to infrared tasks yields suboptimal results due to significant domain discrepancies.

**Intrinsic Characteristics of Infrared Images**:
   - **Low Information Capacity**: Infrared images lack fine texture and rich color details. Analysis of information entropy reveals that the mean entropy of Inf30 is 6.44, compared to 7.19 for ImageNet-1K.
   - **Co-temperature Status of Target and Background**: Objects like crosswalks and utility poles often exhibit temperatures close to their surrounding environments, making them nearly indiscernible in infrared images.
   - **Direct application of the vanilla random masking (as in MAE) results in too many masked blocks falling on uninformative regions**, which is detrimental to representation learning.

**Lack of Large-Scale Infrared Datasets**: Existing datasets in the infrared domain are relatively small in scale with single-scene settings, failing to support effective foundation model pre-training.

These three limitations collectively drive the design of InfMAE: a dedicated dataset, a tailored masking strategy, and an adapted model architecture.

## Method

### Overall Architecture

InfMAE consists of three core components:
1. **Mask Block Generation Module**: Employs an information-aware masking strategy to prioritize the preservation of information-rich regions.
2. **Multi-Scale Encoder Module**: Implements a hybrid CNN-Transformer architecture to generate multi-scale features.
3. **Infrared Decoder Module**: Aggregates multi-scale representations for image reconstruction.

### Key Designs

1. **Inf30 Dataset Construction**

   Around 500,000 raw infrared images were collected from multiple sources, and processed with a two-step preprocessing pipeline:
    - **Deduplication**: Anchors are selected to eliminate redundant images with highly similar scenes.
    - **Quality Filtering**: Images with both resolution dimensions under 20 pixels are discarded.

   This results in **305,241 high-quality infrared images**, spanning scenes such as sky, ocean, forest, city, and suburbs, and containing targets like ships, vehicles, pedestrians, and buildings. Resolutions range from 40×23 to 6912×1024.

2. **Information-Aware Masking (IAM) Strategy**

   To address the non-uniform information distribution in infrared images, an adaptive, information-aware masking strategy is proposed to prevent random mask blocks from completely obscuring information-sparse areas:

    - An intensity feature map is extracted from the input image $x \in \mathbb{R}^{H \times W \times C}$ using a convolution layer (kernel=16, stride=16).
    - The mean value across channels is calculated to derive a grayscale distribution map $I \in \mathbb{R}^{\frac{H}{16} \times \frac{W}{16}}$.
    - The $N$ total tokens are sorted in **descending order** of their grayscale values (higher values indicate higher information content).
    - Evenly-spaced sampling is performed with a sampling stride $S$: those sampled serve as visible tokens, while the rest are marked as masked tokens.
    - The mask template is upsampled by factors of 2 and 4 to yield mask\_block1 and mask\_block2 for the multi-scale encoder.

   Unlike ATTMask or SemMAE, this strategy requires no auxiliary semantic segmentation networks, offering a simple yet effective way to preserve information-dense regions.

3. **Multi-Scale Encoder Module**

   Inspired by MCMAE and ConvNeXt, a three-stage hybrid encoder is designed:

    - **Encoder Layer 1** (CNN): Patch Embedding 1 $\to$ Conv Attention + FFN $\to$ features $F_1 \in \mathbb{R}^{\frac{H}{4} \times \frac{W}{4} \times C_1}$, element-wise multiplied by mask\_block1.
    - **Encoder Layer 2** (CNN): Patch Embedding 2 $\to$ Conv Attention + FFN $\to$ features $F_2 \in \mathbb{R}^{\frac{H}{8} \times \frac{W}{8} \times C_2}$, element-wise multiplied by mask\_block2.
    - **Encoder Layer 3** (ViT Transformer): Flatten + select visible tokens $\to$ Self-Attention $\to$ learned visible tokens $T_s \in \mathbb{R}^{N_v \times C_3}$.

   This layout enables the shallow CNN layers to process local fine-grained textures (crucial given the sparse texture of infrared modalities) while the deep Transformer blocks capture global semantic contexts. The multi-scale feature maps $F_1, F_2, F_3, F_4$ can be seamlessly integrated into feature pyramids for downstream tasks.

4. **Infrared Decoder Module**

   The multi-scale visible features $T_1, T_2, T_s$ are projected to identical dimensions, aggregated, combined with the learnable masked tokens, and returned to their original spatial positions before being processed by the decoder to reconstruct the input.

   Key Design: The **decoder is configured to be only 2 layers deep**, which is significantly shallower than standard MAEs (typically 8 layers). Since infrared images feature sparse color and texture, a deeper decoder leads to severe overfitting.

### Loss & Training

The reconstruction loss is the Mean Squared Error (MSE) computed over the masked regions:

$$\mathcal{L}_{mse} = \frac{1}{M} \sum_{i=1}^{M} (y_i - x_i)^2$$

Pre-training configurations:
- 400 training epochs, utilizing a cosine learning rate scheduler with 40 warmup epochs.
- AdamW optimizer with a base learning rate of $1.5 \times 10^{-4}$ and weight decay of 0.05.
- Batch size of 256, distributed over 4 × A100 GPUs.
- Encoder structural breakdown: 2 layers (CNN1) + 2 layers (CNN2) + 11 layers (ViT), featuring 768 dimensions and 12 attention heads.
- Default sampling stride $S=4$.

## Key Experimental Results

### Main Results

**Infrared Semantic Segmentation** (MSRS-inf dataset, UperNet head):

| Method | Backbone | mIoU (%) | mAcc (%) | vs. Prev. SOTA |
|------|----------|---------|---------|------------|
| DDRNet | - | 67.3 | 73.3 | - |
| Vanilla MAE | ViT-B | 71.4 | 78.2 | - |
| MCMAE | ViT-B | 72.1 | 79.8 | Previous Best |
| **InfMAE** | **ViT-B** | **74.3** | **82.5** | **+2.2 mIoU** |

**Infrared Object Detection** (M3FD-inf dataset, Mask R-CNN head):

| Method | Backbone | mAP (%) | AP50 (%) |
|------|----------|--------|---------|
| Sparse R-CNN | ResNet50 | 48.3 | 79.4 |
| Vanilla MAE | ViT-B | 51.4 | 83.4 |
| MCMAE | ViT-B | 55.7 | 88.4 |
| **InfMAE** | **ViT-B** | **56.2** | **88.1** |

**Infrared Small Object Detection** (IRSTD-1k dataset):

| Method | Backbone | AUC (%) | F1 (%) | IoU (%) |
|------|----------|--------|-------|--------|
| DNANet | - | 87.8 | 76.4 | 61.8 |
| MCMAE | ViT-B | 90.8 | 78.4 | 64.5 |
| **InfMAE** | **ViT-B** | **91.2** | **79.5** | **66.0** |

### Ablation Study

**Module Ablation** (UperNet segmentation + Mask R-CNN detection):

| IAM | Multi-scale | Seg_mIoU (%) | Det_AP50 (%) | Description |
|:---:|:---:|------|------|------|
| ✗ | ✗ | 71.4 | 81.5 | baseline (vanilla MAE) |
| ✓ | ✗ | 72.0 | 78.9 | Information-aware masking only |
| ✗ | ✓ | 72.1 | 86.3 | Multi-scale encoder only |
| ✓ | ✓ | **74.3** | **88.1** | Best with both combined |

**Decoder Depth**:

| Decoder Depth | Seg_mIoU (%) | Det_AP50 (%) | Description |
|:---:|------|------|------|
| **2** | **74.3** | **88.1** | Infrared images do not require deep decoders |
| 4 | 72.9 | 87.9 | Performance degradation |
| 8 | 73.2 | 87.6 | Further degradation |
| 12 | 74.0 | 87.2 | Excessive depth is detrimental |

### Key Findings

- While Information-Aware Masking and Multi-Scale Encoder individually contribute about 0.6-0.7 mIoU gain, the organic combination of both triggers a 2.9 mIoU improve.
- A decoder depth of 2 is optimal, validating the hypothesis that infrared images have sparse information density where deeper decoders lead to empirical degradation.
- A masking stride $S=4$ achieves the best equilibrium between preserving semantic context and maintaining mask proportions.
- Pre-training performance continuously scales as the dataset increases from 100k to 300k. However, raw scaling to 500k (without deduplication) results in a drop, showing that dataset diversity and hygiene outperform volume.
- Performance scaling on 400 epochs of pre-training approaches the results of 1600 epochs, demonstrating diminishing returns for longer training.

## Highlights & Insights

- **Filling the Gap**: Introduces the first dedicated foundation model in the infrared modality, covering everything from custom dataset curation to tailored architecture design.
- **Simple yet Effective Masking**: Relies on straightforward grayscale sorting and stride sampling, achieving attention-driven masks without extra parameterized components or network branches.
- **Shallow Decoder Insights**: Suggests that shallow decoders yield better representation quality in low-density information settings, which could be widely applicable to other sparse-information modalities.
- **Multi-task Verification**: Obtains significant performance improvements on segmentation, object detection, and challenging fine-grained small target detection tasks simultaneously.

## Limitations & Future Work

- The scale of Inf30 (300k images) is still relatively constrained when compared to datasets like ImageNet-1K (1.28 million).
- IAM is mainly driven by average intensity values, overlooking high-level structural properties like edges and spatial topology.
- Evaluations are restricted to ViT-B, and scaling behaviors on larger models (ViT-L/H) remain unverified.
- Excludes explicit cross-modal operations, such as collaborative joint pre-training on visible-infrared paired setups.
- The dataset's diversity is bounded by public resources, meaning specialized domains (such as medical thermal images or industrial thermal imaging) might be underrepresented.

## Related Work & Insights

- Following Scale-MAE (remote sensing) and VideoMAE (video), InfMAE outlines a clear recipe for adapting the MAE framework to domain-specific traits: Analysis of Modality Characteristics $\to$ Customized Masking Rules $\to$ Adjusted Encoder/Decoder Pipelines.
- Information entropy serves as a highly robust quantitative metric for investigating discrepancies across different sensor data.
- The multi-scale CNN+Transformer encoder framework (inspired by MCMAE) remains generalizable to downstream tasks requiring detailed feature resolution.
- The shallow decoder discovery implies that the difficulty of the reconstruction pretext task should be engineered to mirror the intrinsic complexity of the modal features.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First infrared foundation model; the masking rule is simple but conceptually solid.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive verification across 3 target applications, backed by detailed ablation trajectories on structure, depth, stride, data size, and training length.
- **Writing Quality**: ⭐⭐⭐⭐ — Highly logical development, structured efficiently with motivating entropy comparisons.
- **Value**: ⭐⭐⭐⭐ — Provides a solid baseline model and benchmark dataset for infrared computer vision researches.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] MarineInst: A Foundation Model for Marine Image Analysis with Instance Visual Description](marineinst_a_foundation_model_for_marine_image_analysis_with_instance_visual_des.md)
- [\[ECCV 2024\] WeCromCL: Weakly Supervised Cross-Modality Contrastive Learning for Transcription-only Supervised Text Spotting](wecromcl_weakly_supervised_cross-modality_contrastive_learning_for_transcription.md)
- [\[ICML 2025\] Griffin: Towards a Graph-Centric Relational Database Foundation Model](../../ICML2025/self_supervised/griffin_towards_a_graph-centric_relational_database_foundation_model.md)
- [\[ICML 2026\] How 'Neural' is a Neural Foundation Model?](../../ICML2026/self_supervised/how_neural_is_a_neural_foundation_model.md)
- [\[CVPR 2026\] MOMO: Mars Orbital Model — Foundation Model for Mars Orbital Applications](../../CVPR2026/self_supervised/momo_mars_orbital_model_foundation_model_for_mars_orbital_applications.md)

</div>

<!-- RELATED:END -->

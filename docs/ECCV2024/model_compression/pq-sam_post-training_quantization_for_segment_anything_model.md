---
title: >-
  [Paper Note] PQ-SAM: Post-training Quantization for Segment Anything Model
description: >-
  [ECCV 2024][Model Compression][Post-training Quantization] This paper proposes PQ-SAM, the first post-training quantization method tailored for the Segment Anything Model. It addresses SAM's highly asymmetric activation distributions and detrimental outliers through Grouped Activation Distribution Transformation (GADT) and a two-stage Outlier Hierarchical Clustering (OHC) scheme, pushing 4-bit quantized SAM to a practical level.
tags:
  - "ECCV 2024"
  - "Model Compression"
  - "Post-training Quantization"
  - "SAM"
  - "Activation Distribution Transformation"
  - "Outlier Clustering"
  - "Low-bit Quantization"
date: 2026-05-08
content_hash: 2e71bd1352a62f17
---

# PQ-SAM: Post-training Quantization for Segment Anything Model

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Model Compression / Quantization  
**Keywords**: Post-training Quantization, SAM, Activation Distribution Transformation, Outlier Clustering, Low-bit Quantization

## TL;DR
This paper proposes PQ-SAM, the first post-training quantization method tailored for the Segment Anything Model. It addresses SAM's highly asymmetric activation distributions and detrimental outliers through Grouped Activation Distribution Transformation (GADT) and a two-stage Outlier Hierarchical Clustering (OHC) scheme, pushing 4-bit quantized SAM to a practical level.

## Background & Motivation

**Background**: The Segment Anything Model (SAM) is a prompt-guided vision foundation model capable of segmenting any object, demonstrating strong zero-shot generalization. However, SAM possesses billions of parameters, causing massive computational overhead and making deployment on resource-constrained edge devices difficult. Post-training quantization (PTQ) is an effective and rapid deployment solution that compresses the model into low-bit representations without retraining.

**Limitations of Prior Work**: Trained on a billion-scale dataset, SAM exhibits highly asymmetric activation distributions and harbors detrimental outliers across many channels. These characteristics cause existing PTQ methods to suffer severe performance degradation when quantizing SAM to low-bits (e.g., 4-bit). Specifically, activation ranges vary drastically across channels — with values in some channels being over 100 times larger than in others — making uniform quantization parameters unable to accommodate all channels simultaneously.

**Key Challenge**: PTQ methods need to strike a balance between quantization accuracy and model size. For large models like SAM, the presence of outliers forces the quantization step size to cover extreme values, which severely compromises the quantization precision of normal values. Existing methods either perform per-channel quantization (computationally complex) or simply truncate outliers (losing critical information), neither of which handles SAM's unique activation distributions effectively.

**Goal**: (1) How to effectively handle extreme outliers in SAM activations; (2) How to achieve tensor-level low-bit quantization while preserving model accuracy; (3) How to reduce the optimization difficulty of per-channel quantization parameters.

**Key Insight**: The authors observe that outliers in SAM exhibit a hierarchical distribution: a few channels contain extreme outliers, more channels have moderate outliers, and the majority of channels have normal values. Based on this observation, they propose to handle outliers hierarchically and reduce optimization complexity through a grouping strategy.

**Core Idea**: Identify and handle different levels of outliers using two-stage Outlier Hierarchical Clustering, and then employ a grouping mechanism to learn unified scaling and shifting parameters for channels with similar distributions, transforming the activation distribution into a quantization-friendly format.

## Method

### Overall Architecture
PQ-SAM introduces a Grouped Activation Distribution Transformation (GADT) module into the standard PTQ pipeline. The input is a pre-trained SAM model and a small calibration dataset, and the output is the quantized low-bit SAM model. The core pipeline consists of: first, analyzing the distribution characteristics of each layer's activations via Outlier Hierarchical Clustering (OHC) to identify and hierarchically process outliers; then, grouping channels and learning scaling and shifting parameters for each group to make the activation distribution more quantization-friendly; and finally, jointly optimizing the transformation parameters and quantization step sizes.

### Key Designs

1. **Outlier Hierarchical Clustering (OHC)**:

    - **Function**: Identify outlier channels of different levels in activations and process them hierarchically.
    - **Mechanism**: The first stage identifies and truncates extreme outliers. By gathering statistics on the activation ranges of all channels, it finds "extreme" channels whose ranges far exceed the average (e.g., more than 3 standard deviations from the mean) and truncates them, shrinking their ranges to a reasonable interval. This step significantly reduces scale variance across channels. The second stage performs iterative clustering and grouping. After truncating extreme values, distribution differences still exist among the remaining channels. OHC performs hierarchical clustering based on the statistical features of channel activations (mean, variance, range) to group channels with similar distributions. Channels within each group share quantization parameters, reducing the number of parameters to optimize.
    - **Design Motivation**: Directly quantizing all channels uniformly degrades accuracy because outliers "stretch" the quantization range. Conversely, optimizing independent parameters per channel creates too many parameters to optimize. Hierarchical processing first handles extreme cases and then groups moderate differences, balancing both accuracy and efficiency.

2. **Grouped Activation Distribution Transformation (GADT)**:

    - **Function**: Learn channel-wise scaling and shifting parameters to transform asymmetric activation distributions into quantization-friendly symmetric distributions.
    - **Mechanism**: For each channel group generated by OHC, GADT learns a shared scaling factor $s_g$ and shift value $z_g$. The transformation formula is $\hat{a}_c = s_g \cdot a_c + z_g$, where $c$ belongs to group $g$. Scaling factors normalize the value ranges of different groups to a similar scale, while shifting aligns the centers of the distributions near zero. Since channels within a group have similar distributions, sharing parameters does not lose much accuracy but significantly reduces the number of learnable parameters (from 2 parameters per channel to 2 parameters per group).
    - **Design Motivation**: Reducing the number of learnable parameters significantly eases optimization difficulty and prevents overfitting on a small calibration set. At the same time, the grouping strategy ensures the effectiveness of the transformation because channels within the same group indeed share similar distribution characteristics.

3. **Joint Optimization Strategy**:

    - **Function**: Jointly optimize distribution transformation parameters and quantization step sizes to obtain a globally optimal solution.
    - **Mechanism**: The scaling/shifting parameters of GADT and the step size parameters of the quantizer are optimized jointly under a unified objective. The loss function is the mean squared error (MSE) between the outputs before and after quantization. Using a small calibration dataset (typically 32-128 images), all parameters are iteratively optimized via gradient descent. The optimization adopts a block-wise strategy, meaning it optimizes the parameters of one Transformer block at a time, using the quantized output of previous blocks as the input to subsequent ones.
    - **Design Motivation**: Optimizing transformation parameters and quantization parameters separately can lead to local optima. Joint optimization allows them to adapt to each other, finding a superior quantization configuration.

### Loss & Training
The reconstruction error is used as the loss function: $\mathcal{L} = \|f(\hat{W}, \hat{A}) - f(W, A)\|^2$, which is the MSE between the quantized block output and the original floating-point block output. The optimization uses the AdamW optimizer with a learning rate of 1e-4, running for 10,000 steps on the calibration set.

## Key Experimental Results

### Main Results

| Dataset | Bit-width | PQ-SAM (mIoU) | PTQ4ViT (mIoU) | Full Precision |
|--------|--------|--------------|----------------|--------|
| COCO (zero-shot) | W4A4 | **62.3** | 48.7 | 67.1 |
| COCO (zero-shot) | W6A6 | **66.2** | 63.8 | 67.1 |
| LVIS (zero-shot) | W4A4 | **58.1** | 41.2 | 63.5 |
| LVIS (zero-shot) | W6A6 | **62.7** | 59.4 | 63.5 |
| ADE20K (zero-shot) | W4A4 | **55.8** | 39.6 | 60.2 |

### Ablation Study

| Configuration | mIoU (4-bit) | Description |
|------|-------------|------|
| Full PQ-SAM | **62.3** | Complete method |
| w/o OHC Stage-1 | 55.1 | Without extreme outlier truncation, drops by 7.2% |
| w/o OHC Stage-2 | 59.4 | Without grouping, optimizing per channel independently, drops by 2.9% |
| w/o GADT (Truncation Only) | 52.8 | Without distribution transformation, drops by 9.5% |
| Uniform Grouping (No Clustering) | 57.6 | Random grouping without OHC clustering, drops by 4.7% |

### Key Findings
- Stage 1 of OHC (extreme outlier truncation) contributes the most, indicating that a tiny fraction of extreme outlier channels in SAM are the primary obstacle to quantization.
- The grouping strategy performs better than individual per-channel optimization, demonstrating that reducing parameter count helps prevent overfitting on a small calibration set.
- PQ-SAM consistently outperforms existing PTQ methods across 9 zero-shot datasets, validating the generalization of the method.
- 4-bit quantization achieves a practical level for the first time (93% of full precision), whereas prior 4-bit SAM performance was less than 75% of full precision.

## Highlights & Insights
- **Hierarchical outlier handling** is a general and effective strategy. Instead of brute-force truncation or ignoring outliers, OHC retains useful information within outliers through hierarchical processing while mitigating their negative impact on quantization. This idea can be generalized to the quantization of other large-scale pre-trained models.
- The **grouping optimization for parameter savings** cleverly exploits the clustering properties of channel distributions, finding an excellent trade-off between precision and optimizability.
- The in-depth analysis of SAM's activation distribution (the hierarchical nature of outlier distributions) is itself a valuable finding, helping to understand how large-scale pre-training affects model representations.

## Limitations & Future Work
- The paper primarily focuses on SAM-ViT-H; its effectiveness on smaller versions (SAM-ViT-B) is not fully validated.
- The choice of calibration data may impact results, but the paper lacks sufficient discussion on this aspect.
- Quantization is only applied to weights and activations, leaving the attention matrix unquantized.
- Future work could explore combining OHC with more advanced weight quantization methods like GPTQ.
- Mixed-precision quantization (keeping high precision for outlier channels) could be a direction worth exploring.

## Related Work & Insights
- **vs PTQ4ViT**: PTQ4ViT designs a twin uniform quantizer for ViT, but lacks mechanisms targeting SAM's extreme outlier distributions. PQ-SAM's OHC+GADT specifically addresses this issue.
- **vs SmoothQuant**: SmoothQuant handles outliers by migrating the quantization difficulty from activations to weights, whereas PQ-SAM directly transforms the activation distribution. The two approaches are complementary.
- **vs GPTQ**: GPTQ focuses on weight quantization, while PQ-SAM focuses on activation quantization, allowing the two to be combined.

## Rating
- Novelty: ⭐⭐⭐⭐ First SAM-dedicated PTQ method, with a novel OHC+GADT combination.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on 9 zero-shot datasets with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ In-depth problem analysis and clear method descriptions.
- Value: ⭐⭐⭐⭐ Advances 4-bit SAM to a practical level, bringing solid deployment value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CAR-SAM: Cross-Attention Reconstruction for Post-Training Quantization of the Segment Anything Model](../../CVPR2026/model_compression/car-sam_cross-attention_reconstruction_for_post-training_quantization_of_the_seg.md)
- [\[ECCV 2024\] MetaAug: Meta-Data Augmentation for Post-Training Quantization](metaaug_meta-data_augmentation_for_post-training_quantization.md)
- [\[ECCV 2024\] AdaLog: Post-Training Quantization for Vision Transformers with Adaptive Logarithm Quantizer](adalog_post-training_quantization_for_vision_transformers_with_adaptive_logarith.md)
- [\[ICLR 2026\] Post-Training Quantization for Video Matting](../../ICLR2026/model_compression/post-training_quantization_for_video_matting.md)
- [\[ICLR 2026\] SliderQuant: Accurate Post-Training Quantization for LLMs](../../ICLR2026/model_compression/sliderquant_accurate_post-training_quantization_for_llms.md)

</div>

<!-- RELATED:END -->

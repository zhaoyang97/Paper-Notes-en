---
title: >-
  [Paper Note] ColorMAE: Exploring Data-Independent Masking Strategies in Masked AutoEncoders
description: >-
  [ECCV 2024][Segmentation][Masked AutoEncoders] This paper proposes ColorMAE, which generates data-independent masking patterns with spatial and semantic priors by applying different frequency domain filters to random noise. Without adding any parameters or computational overhead, ColorMAE significantly improves the downstream performance of MAE, particularly achieving a 2.72 mIoU improvement over random masking on semantic segmentation tasks.
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "Masked AutoEncoders"
  - "Data-Independent Masking"
  - "Color Noise"
  - "Self-Supervised Pre-training"
  - "ViT"
date: 2026-05-08
content_hash: 346543b73b7f35d3
---

# ColorMAE: Exploring Data-Independent Masking Strategies in Masked AutoEncoders

**Conference**: ECCV 2024  
**arXiv**: [2407.13036](https://arxiv.org/abs/2407.13036)  
**Code**: [GitHub](https://carloshinojosa.me/project/colormae)  
**Area**: Self-Supervised Learning / Segmentation  
**Keywords**: Masked AutoEncoders, Data-Independent Masking, Color Noise, Self-Supervised Pre-training, ViT

## TL;DR

This paper proposes ColorMAE, which generates data-independent masking patterns with spatial and semantic priors by applying different frequency domain filters to random noise. Without adding any parameters or computational overhead, ColorMAE significantly improves the downstream performance of MAE, particularly achieving a 2.72 mIoU improvement over random masking on semantic segmentation tasks.

## Background & Motivation

Masked Image Modeling (MIM) is a dominant paradigm in visual self-supervised learning, where masking strategies are crucial to pre-training quality. Existing strategies fall into two categories:

**Data-Independent Masking**: Random masking (MAE), block masking (BEiT), and grid masking are simple to implement but under-explored.

**Data-Adaptive Masking**: AttMask, SemMAE, and HPM rely on attention or teacher networks, yielding better performance but introducing extra computational overhead.

**Core Problem**: Can the performance of random masking be surpassed without relying on input data or increasing computational costs?

The authors observe that random masking in MAE is essentially based on white noise sampling, while multiple colored noises (red, blue, green, purple) with different spectral characteristics exist in signal processing. This inspires the generation of masking patterns with spatial structural priors through frequency-domain filtering.

## Method

### Overall Architecture

The core idea of ColorMAE is extremely simple: prior to MAE pre-training, different frequency domain filters are applied to random noise to generate noise patterns with specific spectral constraints. These noise patterns are then used to replace standard random noise for mask sampling. The entire process introduces no additional learnable parameters, and its computational efficiency is comparable to the original MAE random masking.

### Key Designs

#### Four Color Noise Masks

Let $W(x,y)$ be a random noise image, and $G_{\sigma}$ be a Gaussian kernel with standard deviation $\sigma$:

**Red Noise (Low-pass Filtering)**: $N_r = G_{\sigma} \ast W$. Applying Gaussian blur to random noise retains low-frequency components and filters out high frequencies. This generates masks with large connected regions, resulting in the highest difficulty and worst reconstruction quality.

**Blue Noise (High-pass Filtering)**: $N_b = W - G_{\sigma} \ast W$. Subtracting the low-pass filtered result from the original noise retains high frequencies. This produces uniformly distributed masks without large blanks or dense clusters, representing the lowest difficulty.

**Green Noise (Band-pass Filtering)**: $N_g = G_{\sigma_1} \ast W - G_{\sigma_2} \ast W$ ($\sigma_1 < \sigma_2$). A weak blur is first applied to filter out the highest frequencies, followed by a strong blur to extract the lowest frequencies; subtracting the two retains the mid-frequency components. This creates a clustered version of the blue noise mask, balancing spatial structure and moderate difficulty.

**Purple Noise (Band-stop Filtering)**: $N_p = W - (G_{\sigma_1} \ast W - G_{\sigma_2} \ast W)$. Subtracting green noise from the original noise retains both high and low frequencies while removing mid-frequencies, combining the characteristics of red and blue noise.

#### Mask Generation Process

1. Pre-compute the color noise tensors offline and store them in the GPU memory.
2. During training, perform spatial transformations such as random cropping and horizontal/vertical flipping on the noise tensors (which do not alter frequency-domain properties).
3. Select the top-$k$ positions with the largest values in the noise window as masking locations (e.g., a 75% masking ratio).
4. Generate binary masks to feed into the MAE encoder.

### Loss & Training

Completely consistent with standard MAE: only the mask sampling strategy is replaced, utilizing the same pixel reconstruction loss, the same 75% masking ratio, and the same asymmetric encoder-decoder architecture. No additional loss functions or hyperparameters are introduced.

## Key Experimental Results

### Main Results

#### Comparison on Three Downstream Tasks (ViT-B/16)

| Pre-training Epochs | Method | ImageNet Top-1 | ADE20K mIoU | COCO AP_bbox |
|:---:|:---:|:---:|:---:|:---:|
| 300 | Random | 82.82 | 44.51 | 48.50 |
| 300 | Green | **82.98** | **45.80** | **48.70** |
| 800 | Random | 83.17 | 46.46 | 49.15 |
| 800 | Green | **83.57** | **49.18** | **49.50** |
| 1600 | Random | 83.43 | 47.46 | 49.60 |
| 1600 | Green | **83.77** | **49.26** | **50.10** |

Green masking consistently outperforms random masking across all tasks and pre-training epochs, with semantic segmentation showing the most significant improvement (+2.72 mIoU at 800 epochs).

#### ViT-Large Validation

| Pre-training | Method | ImageNet Top-1 | ADE20K mIoU |
|:---:|:---:|:---:|:---:|
| 300 ep | Random | 84.76 | 47.55 |
| 300 ep | Green | **85.02** | **49.00** |
| 800 ep | Random | 85.42 | 50.29 |
| 800 ep | Green | **85.64** | **51.46** |

#### Comparison with SOTA Methods (800 ep ViT-B)

| Method | Type | ADE20K mIoU | ImageNet Top-1 | COCO AP_bbox |
|:---:|:---:|:---:|:---:|:---:|
| MAE | data-indep | 46.5 | 83.2 | 49.2 |
| AttMask | data-adapt | 45.3 | - | 48.8 |
| SemMAE | data-adapt | 44.9 | 83.4 | 45.6 |
| HPM | data-adapt | 48.5 | 84.2 | 50.1 |
| **ColorMAE-G** | **data-indep** | **49.2** | **83.6** | **49.5** |

ColorMAE-G, as a data-independent method, outperforms most data-adaptive methods in semantic segmentation.

### Ablation Study

- **Red masking** achieves the worst results: the masking is too aggressive (large contiguous regions are masked), making reconstruction difficult and hindering the learning of good representations.
- **Blue masking** yields the lowest reconstruction loss but underperforms downstream compared to Green, indicating that reconstruction difficulty being too low is unfavorable for representation learning.
- **Green masking** is optimal: it provides moderate difficulty (band-pass) while combining spatial clustering and an adequate reconstruction challenge.
- **Purple masking** is the second worst: it retains both high and low frequencies simultaneously, lacking spatial structural information from the mid-frequency band.

### Key Findings

1. **Reconstruction quality is not proportional to downstream performance**: Blue masking produces the best reconstruction but weak representations, demonstrating that a pretext task with moderate difficulty is more beneficial for learning.
2. **The advantage of Green masking grows with more training epochs**: The mIoU gain increases by 3.38 going from 300 to 800 epochs, implying that green noise masking can accelerate convergence.
3. **Mid-frequency components are particularly crucial for semantic segmentation**: The mid-frequency information preserved by band-pass filtering closely corresponds to spatial structures at the level of object parts.

## Highlights & Insights

1. **Exceedingly Simple**: The core innovation requires only a few lines of code (filtering + cropping), without adding model parameters or computation.
2. **Intersection of Signal Processing and Deep Learning**: Cleverly introduces the spectral theory of color noise into self-supervised masking strategies.
3. **High Universality**: Can serve as a drop-in masking strategy replacing random masking in any MIM method.
4. **Maximum Gain in Semantic Segmentation**: An improvement of +2.72 mIoU is highly substantial in the segmentation field.

## Limitations & Future Work

1. The selection of filtering parameter $\sigma$ relies on manual tuning and lacks an adaptive mechanism.
2. Validated only within the MAE framework without extension to other MIM methods such as SimMIM and BEiT.
3. Does not explore hybrid strategies among different color noises (e.g., dynamically switching during training).
4. The gain on classification tasks is relatively smaller than that on segmentation, and the underlying reason is not analyzed in depth.

## Related Work & Insights

- **MAE**: This work is directly built upon the MAE framework, only replacing the mask sampling.
- **SemMAE / HPM**: Representatives of data-adaptive masking, where ColorMAE achieves comparable or even superior performance at zero additional cost.
- **DropPos**: Another data-independent augmentation that is complementary to this work.
- **Insights**: Frequency-domain priors may also provide valuable references for other self-supervised tasks (such as data augmentation in contrastive learning).

## Rating

- Novelty: 4/5 - Introducing color noise into the masking strategy is a clever and original entry point.
- Experimental Thoroughness: 5/5 - Evaluated across three downstream tasks, multiple backbones, various training epochs, with comprehensive ablations.
- Writing Quality: 4/5 - Clear logic, excellent illustrations.
- Value: 4/5 - Simple and practical, directly applicable to any MIM pre-training pipeline.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Un-EVIMO: Unsupervised Event-based Independent Motion Segmentation](un-evimo_unsupervised_event-based_independent_motion_segmentation.md)
- [\[ECCV 2024\] SeiT++: Masked Token Modeling Improves Storage-Efficient Training](seit_masked_token_modeling_improves_storage-efficient_training.md)
- [\[ECCV 2024\] LASS3D: Language-Assisted Semi-Supervised 3D Semantic Segmentation with Progressive Unreliable Data Exploitation](lass3d_language-assisted_semi-supervised_3d_semantic_segmentation_with_progressi.md)
- [\[CVPR 2026\] SARMAE: Masked Autoencoder for SAR Representation Learning](../../CVPR2026/segmentation/sarmae_masked_autoencoder_for_sar_representation_learning.md)
- [\[ICML 2025\] Dual form Complementary Masking for Domain-Adaptive Image Segmentation](../../ICML2025/segmentation/dual_form_complementary_masking_for_domain-adaptive_image_segmentation.md)

</div>

<!-- RELATED:END -->

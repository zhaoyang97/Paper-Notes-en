---
title: >-
  [Paper Note] OAPT: Offset-Aware Partition Transformer for Double JPEG Artifacts Removal
description: >-
  [ECCV 2024][Image Restoration][JPEG artifact removal] To address the double JPEG compressed image restoration problem, OAPT is proposed. By predicting the pixel offset between the two compressions, four different patterns in each 8×8 block are clustered and grouped for separate self-attention processing, outperforming the state-of-the-art methods by 0.16 dB on the double JPEG restoration task.
tags:
  - "ECCV 2024"
  - "Image Restoration"
  - "JPEG artifact removal"
  - "Double JPEG compression"
  - "Transformer"
  - "Offset-aware"
  - "Pattern clustering"
date: 2026-05-08
content_hash: 683418266bc78aed
---

# OAPT: Offset-Aware Partition Transformer for Double JPEG Artifacts Removal

**Conference**: ECCV 2024  
**arXiv**: [2408.11480](https://arxiv.org/abs/2408.11480)  
**Code**: [https://github.com/QMoQ/OAPT.git](https://github.com/QMoQ/OAPT.git)  
**Area**: Image Restoration  
**Keywords**: JPEG artifact removal, Double JPEG compression, Transformer, Offset-aware, Pattern clustering

## TL;DR

To address the double JPEG compressed image restoration problem, OAPT is proposed. By predicting the pixel offset between the two compressions, four different patterns in each 8×8 block are clustered and grouped for separate self-attention processing, outperforming the state-of-the-art methods by 0.16 dB on the double JPEG restoration task.

## Background & Motivation

JPEG is the most widely used image compression algorithm, which divides images into 8×8 blocks before DCT and quantization. In reality, images often undergo multiple compressions: for instance, the first compression during camera capture, and the second compression during cropping or uploading to social media. This is the double JPEG compression problem, which is more common than single compression.

The core pain point of existing methods is that most of them are trained only on single-compression data, resulting in severe performance degradation when facing double compression. FBCNN is the first method specifically designed to handle double JPEG, but it only estimates the dominant QF, failing to fully utilize the characteristics of double compression.

**Key Observation**: When the block grids of the two compressions are unaligned (non-aligned compression), up to **four different patterns** appear within each 8×8 block of the second compression, as these four parts belonged to different 8×8 blocks in the first compression. DnCNN experiments show that restoring non-aligned compression is much more difficult than aligned compression (ΔPSNR drops from 2.06/3.24 to 1.66/1.64).

**Core Idea**: If the offset between the two compressions can be predicted, pixels of the same pattern can be clustered together and each pattern can be restored separately, thereby reducing the difficulty of non-aligned compression restoration.

## Method

### Overall Architecture

OAPT consists of two components:
1. **Compression Offset Predictor**: A CNN based on ResNet-18 that takes the top-left 44×44 patch of the image as input to predict the row and column offsets $(r, c)$ between the two compressions, ranging from 0 to 7.
2. **Image Reconstructor**: A Transformer-based network composed of multiple Hybrid Partition Attention Blocks (HPABs), which utilizes the predicted offsets for pattern clustering and hybrid attention.

### Key Designs

1. **Compression Offset Predictor**:

    - Based on the ResNet-18 architecture, using depthwise separable convolutions (D-Resblocks) to reduce parameters.
    - Takes only the top-left 44×44 patch as input (since JPEG compression partitions the image into blocks starting from the top-left).
    - Outputs two integers from 0 to 7 by applying Sigmoid and Round operations:
    $[\hat{r}, \hat{c}] = \text{Round}(\text{Sigmoid}([r', c']) \times 7)$
    - Optimized with L1 loss: $\mathcal{L}_{offset} = \|\hat{r}-r\|_1 + \|\hat{c}-c\|_1$
    - Motivation: Utilizing the periodicity and uniformity of JPEG compression, the offset is globally consistent, and a small patch is sufficient for prediction.

2. **Hybrid Partition Attention Block (HPAB)**:

    - Each HPAB contains 4 standard Swin Transformer Layers (STL) and 2 Pattern Clustering-based STLs (PC-STL).
    - STL provides standard window self-attention to process locally continuous features.
    - PC-STL uses the offset to divide each 8×8 block into four patterns and cluster them:
    $[x_1, x_2, x_3, x_4] = \text{PC}(X_{LN}, \text{offset})$
    $\hat{X} = \text{invPC}(\text{W-MSA}([x_1, x_2, x_3, x_4]), \text{offset})$
    - After clustering, window self-attention is applied separately to each pattern, followed by inverse clustering to restore the original positions.
    - Contrast with ART's sparse attention: ART uses uniform downsampling to enlarge the receptive field, whereas OAPT decomposes the image into four sparse patches based on offsets to extract the same pattern information.

3. **Pattern Clustering Plug-in Module**:

    - Pattern clustering can serve as a plug-and-play module for other Transformer methods.
    - Introduces no additional parameters or computational cost.
    - Experimentally validated on HAT-S, showing improved double compression restoration performance and an expanded receptive field.
    - Motivation: The window partition operation of the Transformer itself is similar to pattern clustering, allowing for a natural integration.

### Loss & Training

The reconstructor uses Charbonnier loss:
$$\mathcal{L}_{rec} = \sqrt{\|\hat{I} - I\|^2 + \epsilon^2}, \quad \epsilon = 10^{-3}$$

Training Strategy:
- First pre-train the offset predictor, then freeze its parameters to train the reconstructor.
- The reconstructor is initialized with pre-trained SwinIR weights and fine-tuned on the double compression dataset.
- Number of HPABs = 6, channels = 180, window size = 7, patch size = 224×224.
- Using Adam optimizer, learning rate 2e-4, batch size 4, 4 V100 GPUs.
- Training data uses DIV2K + Flickr2K, QF randomly sampled from 5-95, offsets $i, j$ randomly sampled from 0-7.

## Key Experimental Results

### Main Results

Average PSNR/SSIM/PSNR-B comparison on grayscale double JPEG images (Classic5 dataset):

| Compression Type(QF1,QF2,i,j) | DnCNN | FBCNN | SwinIR | ART | OAPT | Gain vs SwinIR |
|-------|-------|-------|--------|-----|------|------|
| (30,30,4,4) | 31.68 | 32.12 | 32.26 | 32.29 | **32.32** | +0.06 |
| (50,50,4,4) | 33.22 | 33.70 | 33.80 | 33.86 | **33.87** | +0.07 |
| (30,50,4,4) | 32.30 | 32.74 | 32.90 | 32.93 | **33.02** | +0.12 |
| (50,30,4,4) | 32.31 | 32.81 | 32.95 | 33.02 | 32.97 | +0.02 |
| (30,50,0,4) | 32.44 | 32.93 | 33.06 | 33.10 | **33.16** | +0.10 |
| (50,50,0,4) | 33.34 | 33.85 | 33.94 | 33.98 | **34.02** | +0.08 |

Color double JPEG (LIVE1 dataset):

| Compression Type | SwinIR | HAT-S | OAPT |
|---------|--------|-------|------|
| (30,30,4,4) | 30.21 | 30.20 | **30.26** |
| (50,50,4,4) | 31.86 | 31.87 | **31.92** |
| (30,50,4,4) | 30.87 | 30.85 | **30.95** |

Computational cost comparison: OAPT has 12.96M parameters and 293.60G MACs, which is comparable to SwinIR (11.49M/293.42G) and far smaller than ART (16.14M/415.51G).

### Ablation Study

| Configuration | Key Metrics | Note |
|------|---------|------|
| SwinIR Baseline | 32.26/0.8703 | No offset awareness |
| + GT offset | Better | Offset information is effective |
| + Pred offset | 32.32/0.8718 | Predicted offset is close to GT |
| Offset predictor accuracy | ~97% | Correct prediction in most cases |
| HAT-S Baseline | 32.28/0.8707 | No pattern clustering |
| HAT-S + PC plug-in | Improved | Zero parameters + zero computational overhead |
| Aligned vs. Non-aligned compression | ΔPSNR gap 2.06→1.66 | Non-aligned is harder, OAPT is optimized for this |

### Key Findings

- Non-aligned double JPEG compression is significantly more difficult to restore than aligned compression.
- Pattern clustering improves restoration performance with zero additional parameters/computation, validating the effectiveness of grouping by offset.
- The offset predictor is highly accurate (~97%) and only requires a small 44×44 input, maintaining a very low computational overhead.
- OAPT comprehensively outperforms SwinIR with a comparable parameter count.

## Highlights & Insights

1. **In-depth Problem Exploitation**: Provides a clear physical analysis of the "four patterns" in double JPEG compression, decomposing a seemingly complex problem into a well-defined pattern clustering task.
2. **Clever Design of Offset Predictor**: Accurately predicts global offsets using only a 44×44 top-left patch, exploiting the grid periodicity characteristics of JPEG compression.
3. **Plug-and-Play Pattern Clustering**: Serves as a zero-overhead plug-in that can enhance other Transformer-based methods, offering high practicality.
4. **Single Model Covering All QFs and Offsets**: Eliminates the need for separate training, as a single model can handle all combinations of QF from 5-95 and offsets from 0-7.

## Limitations & Future Work

- Only handles double JPEG compression; scenarios with multiple compressions (more than twice) are not investigated.
- The offset predictor uses a non-differentiable Round operation, making the two-stage (pre-training + freezing) strategy potentially suboptimal.
- The performance gain on color images is smaller compared to grayscale images.
- Scenarios with mixed compression formats (e.g., JPEG + WebP) are not considered.
- The assumption of 4 patterns may degenerate under certain special offset values (e.g., only 1 pattern when the offset is 0).

## Related Work & Insights

- Contrasting with the QF estimation method in FBCNN, OAPT analyzes the essence of the problem from the perspective of bottom-level offsets.
- The concept of pattern clustering can be extended to other image restoration problems with periodic structures (e.g., demoireing).
- The paradigm of offset prediction + adaptive grouping can inspire the handling of inter-frame offsets in video compression artifact removal.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Seeing the Unseen: A Frequency Prompt Guided Transformer for Image Restoration](seeing_the_unseen_a_frequency_prompt_guided_transformer_for_image_restoration.md)
- [\[ECCV 2024\] EDformer: Transformer-Based Event Denoising Across Varied Noise Levels](edformer_transformer-based_event_denoising_across_varied_noise_levels.md)
- [\[ECCV 2024\] Restoring Images in Adverse Weather Conditions via Histogram Transformer](restoring_images_in_adverse_weather_conditions_via_histogram_transformer.md)
- [\[ECCV 2024\] Efficient Diffusion Transformer with Step-wise Dynamic Attention Mediators](efficient_diffusion_transformer_with_step-wise_dynamic_attention_mediators.md)
- [\[CVPR 2025\] SoftShadow: Leveraging Soft Masks for Penumbra-Aware Shadow Removal](../../CVPR2025/image_restoration/softshadow_leveraging_soft_masks_for_penumbra-aware_shadow_removal.md)

</div>

<!-- RELATED:END -->

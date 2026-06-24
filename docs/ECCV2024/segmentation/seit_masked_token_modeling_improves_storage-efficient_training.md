---
title: >-
  [Paper Note] SeiT++: Masked Token Modeling Improves Storage-Efficient Training
description: >-
  [ECCV 2024][Segmentation][Storage-Efficient Training] By introducing Masked Token Modeling (MTM) self-supervised pre-training into the tokenized training framework of SeiT and designing two token-specific data augmentation strategies, TokenAdapt and ColorAdapt, the ImageNet-1k classification accuracy is improved from 74.0% to 77.8% under only 1% storage space (1.4GB), effectively solving the challenge of data augmentation in the token domain.
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "Storage-Efficient Training"
  - "Vector Quantization"
  - "Masked Modeling"
  - "Data Augmentation"
  - "Self-Supervised Learning"
date: 2026-05-08
content_hash: d1b4d6fdff78cbf7
---

# SeiT++: Masked Token Modeling Improves Storage-Efficient Training

**Conference**: ECCV 2024  
**arXiv**: [2312.10105](https://arxiv.org/abs/2312.10105)  
**Code**: Yes ([https://github.com/naver-ai/seit](https://github.com/naver-ai/seit))  
**Area**: Segmentation / Image Classification  
**Keywords**: Storage-Efficient Training, Vector Quantization, Masked Modeling, Data Augmentation, Self-Supervised Learning

## TL;DR

By introducing Masked Token Modeling (MTM) self-supervised pre-training into the tokenized training framework of SeiT and designing two token-specific data augmentation strategies, TokenAdapt and ColorAdapt, the ImageNet-1k classification accuracy is improved from 74.0% to 77.8% under only 1% storage space (1.4GB), effectively solving the challenge of data augmentation in the token domain.

## Background & Motivation

Training high-performance vision models requires massive datasets, which brings huge storage pressure (e.g., LAION-5B requires 240TB). Existing storage reduction methods include:

| Method | Approach | Limitations |
|------|------|--------|
| Dataset Distillation | Compresses datasets into small synthetic sets | High computational complexity, not applicable to large-scale data |
| Coreset Selection / Sampling | Selects the most representative subsets | Insufficient diversity at low data volumes |
| Low Resolution / JPEG Compression | Reduces the size of individual images | Significant degradation in performance |
| **SeiT** | Converts images to discrete tokens for storage | 90% performance, 1% storage, but limited to fully supervised learning |

The breakthrough of SeiT lies in using a ViT-VQGAN tokenizer to compress each image into a token sequence, compressing ImageNet-1k from 140GB to 1.4GB. However, SeiT has two key limitations:

**Only explored fully supervised learning**, without utilizing the potential of self-supervised pre-training.

**Limited data augmentation in the token domain**—directly applying pixel-level augmentations to tokens leads to severe distortion.

## Method

### Overall Architecture

SeiT++ = Masked Token Modeling (MTM) + TokenAdapt + ColorAdapt

**Data Preparation**: Uses a ViT-VQGAN tokenizer to offline convert and store each image as a token sequence.

**Pre-training Phase**: Performs self-supervised pre-training using MTM (without labels).

**Fine-tuning Phase**: Performs supervised fine-tuning on the tokenized dataset (classification/segmentation).

### Key Designs

**Masked Token Modeling (MTM)**:

Analogous to BERT's Masked Language Modeling, MTM predicts masked tokens from visible tokens:

1. **Masking Strategy**: Adopts a variable masking ratio with a truncated normal distribution.
2. **Encoder**: Only processes visible token embeddings, reducing training time and GPU memory.
3. **Decoder**: Pads the encoder output with mask tokens to original length to predict original tokens at masked positions.
4. **Training Objective**: Cross-entropy loss

$$\mathcal{L}_{recon} = \text{CE}(T'_M, T_M)$$

The loss is calculated only for the masked tokens.

**TokenAdapt (Addressing geometric augmentation issues)**:

Core Problem: The tokenization process compresses $n \times n$ 2D image patches into 1D vectors, where the **collapse of spatial information** renders augmentations like flipping ineffective; the **mutual dependency** between token embeddings causes interpolation-based augmentations (like resize, crop) to introduce artifacts.

Solution: Learn a transform-augment-inverse transform pipeline

$$Z_T^{\mathbf{A}} = g(\mathbf{A}(f(Z_T))), \quad T^{\mathbf{A}} = \mathbf{q}_{\mathcal{Z}}(Z_T^{\mathbf{A}})$$

- $f$: Transforms token embeddings into an augmentation-compatible space.
- $\mathbf{A}$: Standard pixel-level augmentations (flipping, cropping, affine, etc.).
- $g$: Inversely transforms back to the token embedding space.
- $\mathbf{q}_{\mathcal{Z}}$: Vector quantization to codebook indices.

$f$ and $g$ are learned from token paired data $(T_x, T_{\mathbf{A}(x)})$ and can generalize across datasets and tasks after training.

**ColorAdapt (Addressing color augmentation issues)**:

Inspired by Adaptive Instance Normalization, it alters color attributes by adjusting the statistics of token embeddings:

$$\mathcal{C}(Z_{T_1}, Z_{T_2}) = \sigma(Z_{T_2})\frac{Z_{T_1} - \mu(Z_{T_1})}{\sigma(Z_{T_1})} + \mu(Z_{T_2})$$

alters color representation while preserving target structure.

### Loss & Training

- MTM pre-training loss: Cross-entropy only at masked positions.
- TokenAdapt training loss: Cross-entropy between the augmented token embeddings and the token embeddings of the ground-truth augmented image.
- Fine-tuning inherits the training recipe of SeiT, with additional usage of CutMix and Emb-Noise.

## Key Experimental Results

### Main Results

Storage-Efficient ImageNet-1k Classification (ViT-B/16):

| Method | Input | Storage | Top-1 Acc |
|------|------|------|-----------|
| Full pixels | Image | 140 GB | 81.8 |
| JPEG quality=5 | Image | 11 GB (8%) | 74.6 |
| Downsampling ×0.2 | Image | 9.6 GB (7%) | 75.2 |
| SeiT | Token | 1.4 GB (1%) | 74.0 |
| **SeiT++** | **Token** | **1.4 GB (1%)** | **77.8** |

Comparison under different storage budgets:

| Storage | SeiT | SeiT w/ MTM | SeiT++ w/o MTM | SeiT++ w/ MTM |
|------|------|------------|----------------|---------------|
| 1.4 GB | 74.0 | 75.1 | 75.5 | **77.8 (+3.8)** |
| 0.8 GB | 66.3 | 70.6 | 69.1 | **74.1 (+7.8)** |
| 0.3 GB | 47.2 | 53.9 | 51.2 | **60.6 (+13.4)** |

### Ablation Study

Individual contributions of each augmentation strategy (ViT-S, ImageNet-100):

| ColorAdapt | TokenAdapt | Top-1 Acc |
|------------|------------|-----------|
| ✘ | ✘ | 77.3 (SeiT baseline) |
| ✔ | ✘ | 78.3 (+1.0) |
| ✘ | ✔ | 80.4 (+3.1) |
| ✔ | ✔ | **81.4 (+4.1)** |

Robustness evaluation (ViT-B, without MTM):

| Benchmark | SeiT | SeiT++ | Gain |
|----------|------|--------|------|
| Clean | 74.0 | 75.5 | +1.5 |
| Gaussian Noise | 50.7 | 58.6 | +7.9 |
| Gaussian Blur | 62.6 | 66.8 | +4.2 |
| ImageNet-R | 25.5 | 30.2 | +4.7 |
| Sketch | 22.6 | 27.7 | +5.1 |

### Key Findings

1. **Less storage, greater improvement**: From 1.4GB to 0.3GB, the improvement of SeiT++ compared to SeiT expands from 3.8% to 13.4%.
2. MTM and token augmentation **synergize**: MTM alone improves by 1.1%, augmentation alone improves by 1.5%, and the combination improves by 3.8%.
3. TokenAdapt (geometric augmentation) contributes more than ColorAdapt (color augmentation) (+3.1 vs +1.0).
4. There is also a significant improvement of +4.2 mIoU on ADE-20k semantic segmentation.
5. The method can transfer to different tokenizers (VQGAN) and even different input formats (DCT coefficients).

## Highlights & Insights

1. **Systematic analysis of token-domain augmentation**: The first to analyze in depth the two root causes of why pixel-level augmentations fail in the token domain—spatial information collapse and dependency between tokens.
2. **Ingenious design of TokenAdapt**: Instead of directly augmenting in the token space, it learns an invertible transformation to an "augmentation-compatible space," reusing well-established pixel-level augmentation strategies.
3. **First integration of self-supervision + tokens**: Proves that MLM-style self-supervised methods can learn directly from offline tokens without raw pixel images.
4. **Extreme storage efficiency**: Validates that a ViT model with 70%+ accuracy can be trained using only 1GB of data.

## Limitations & Future Work

1. Experiments were not conducted on larger-scale datasets like ImageNet-21k (due to resource constraints).
2. The transformation/inverse-transformation modules of TokenAdapt require additional training, increasing the pipeline complexity.
3. The reconstruction quality of the tokenizer (ViT-VQGAN) itself limits the performance upper bound of the method.
4. Only the ViT architecture is explored, and the applicability to CNN architectures remains unknown.
5. Future work can explore more advanced tokenizers (e.g., SDXL-VAE) to further improve performance.

## Related Work & Insights

- **SeiT**: The base framework for this work, which first demonstrated the feasibility of token-based training.
- **BeiT / MAE**: Pioneers of masked image modeling; this work transfers their ideas to offline tokens.
- **MAGE**: Uses VQGAN tokens in generative learning, but relies on online tokenization.
- Insights: Discrete representations (tokens) are not only useful for compressing storage, but are also naturally suited for MLM-style self-supervised learning.

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 4 |
| Technical Depth | 4 |
| Experimental Thoroughness | 5 |
| Writing Quality | 4 |
| Value | 4 |
| **Overall** | **4.2** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Harnessing Massive Satellite Imagery with Efficient Masked Image Modeling](../../ICCV2025/segmentation/harnessing_massive_satellite_imagery_with_efficient_masked_image_modeling.md)
- [\[CVPR 2026\] Masked Representation Modeling for Domain-Adaptive Segmentation](../../CVPR2026/segmentation/mrm_masked_representation_modeling_domain_adaptive.md)
- [\[ECCV 2024\] ColorMAE: Exploring Data-Independent Masking Strategies in Masked AutoEncoders](colormae_exploring_data-independent_masking_strategies_in_masked_autoencoders.md)
- [\[ECCV 2024\] ControlNet++: Improving Conditional Controls with Efficient Consistency Feedback](controlnet_improving_conditional_controls_with_efficient_consistency_feedback.md)
- [\[ECCV 2024\] Efficient and Versatile Robust Fine-Tuning of Zero-shot Models](efficient_and_versatile_robust_fine-tuning_of_zero-shot_models.md)

</div>

<!-- RELATED:END -->

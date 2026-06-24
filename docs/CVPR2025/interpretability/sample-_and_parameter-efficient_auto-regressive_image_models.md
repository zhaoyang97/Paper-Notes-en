---
title: >-
  [Paper Note] Sample- and Parameter-Efficient Auto-Regressive Image Models
description: >-
  [CVPR 2025][Interpretability][Auto-regressive image models] This paper proposes XTRA, which introduces a Block Causal Mask (using $k \times k$ token blocks as the causal unit) into ViT. This allows auto-regressive image models to outperform previous state-of-the-art auto-regressive models on 15 image recognition benchmarks using only 1/152 of the training samples, while achieving superior probing performance with 1/7 to 1/16 of the parameter count.
tags:
  - "CVPR 2025"
  - "Interpretability"
  - "Auto-regressive image models"
  - "sample efficiency"
  - "parameter efficiency"
  - "Block Causal Mask"
  - "visual representation learning"
date: 2026-05-08
content_hash: b2e1794bab74bde9
---

# Sample- and Parameter-Efficient Auto-Regressive Image Models

**Conference**: CVPR 2025  
**arXiv**: [2411.15648](https://arxiv.org/abs/2411.15648)  
**Code**: [github.com/elad-amrani/xtra](https://github.com/elad-amrani/xtra)  
**Area**: Interpretability  
**Keywords**: Auto-regressive image models, sample efficiency, parameter efficiency, Block Causal Mask, visual representation learning

## TL;DR

This paper proposes XTRA, which introduces a Block Causal Mask (using $k \times k$ token blocks as the causal unit) into ViT. This allows auto-regressive image models to outperform previous state-of-the-art auto-regressive models on 15 image recognition benchmarks using only 1/152 of the training samples, while achieving superior probing performance with 1/7 to 1/16 of the parameter count.

## Background & Motivation

**Background**: In the field of self-supervised visual representation learning, there are three mainstream methodologies: contrastive learning (CL, e.g., DINO, MoCo), masked image modeling (MIM, e.g., MAE, BEiT), and auto-regressive image modeling (e.g., iGPT, AIM). While the first two perform well, they rely on complex training techniques (multi-crop augmentation, momentum networks, various regularizations, etc.) and exhibit unstable scalability on internet-scale unbalanced data. In contrast, auto-regressive models possess favorable scaling properties similar to large language models in NLP—where performance scales with model and data size, and the loss correlates strongly with downstream performance.

**Limitations of Prior Work**: Auto-regressive vision models suffer from extremely low sample and parameter efficiency: (1) iGPT requires 7B parameters (15 times more than contrastive methods) to achieve comparable performance; (2) AIM needs to be trained on 2 billion samples (DFN-2B), whereas CL/MIM methods can achieve competitive results with just 1.3 million samples (ImageNet-1k)—an efficiency gap of approximately 150 times. This inefficiency severely hinders the application of auto-regressive vision models in resource-constrained environments.

**Key Challenge**: Standard auto-regressive models predict the pixel values of the next token patch-by-patch, where substantial modeling capacity is consumed by predicting localized high-frequency details (minor differences between adjacent patches) that contribute minimally to object recognition. What truly benefits downstream recognition tasks is low-frequency structural information (object shapes, layouts, etc.), yet standard causal masking forces the model to allocate modeling capacity to every single token.

**Goal**: Design a more efficient auto-regressive pre-training objective that enables the model to concentrate its capacity on semantically meaningful structural patterns, thereby achieving superior visual representation quality with less data and smaller model sizes.

**Key Insight**: If causality is scaled from the token level to the block level (where multiple tokens constitute a block), the model no longer needs to predict high-frequency details of individual patches. Instead, it predicts pixels over a larger area—forcing the model to learn structural relationships across wider spatial ranges, naturally biasing it toward low-frequency semantic features.

**Core Idea**: Replace the standard Causal Mask with a Block Causal Mask, where each block consists of $k \times k$ tokens inside which tokens can attend to each other, while causality is enforced at the block level. This simple modification enables the auto-regressive model to achieve both sample and parameter efficiency.

## Method

The modifications made by XTRA to standard auto-regressive vision models (such as AIM) are extremely concise: only the attention masking strategy is altered. Instead of predicting the next token in raster order token-by-token, the model predicts all pixels of the next block in raster order block-by-block. This single modification simultaneously resolves both sample and parameter inefficiencies.

### Overall Architecture

The input is an image, segmented into a sequence of non-overlapping patches following the standard ViT pipeline. The model adopts an encoder-decoder architecture, both utilizing ViT with a Block Causal Mask. The image is divided into blocks of size $k \times k$ tokens (e.g., $4 \times 4$ patches = $64 \times 64$ pixels), where tokens within a block can perform bidirectional attention, and block-to-block attention is causal in raster order. The decoder outputs representations for all tokens within each block, which are concatenated and passed through a shared MLP to predict all pixel values of the next block. The training loss is block-normalized MSE. During inference, frozen features from the encoder are extracted for linear or attentive probing in downstream tasks.

### Key Designs

1. **Block Causal Mask**:

    - **Function**: Elevates the granularity of causal attention from a single token to a $k \times k$ token block, shifting the information scale modeled by the network
    - **Mechanism**: Segments the grid of image patches into a larger grid of blocks (e.g., a 256px image with a $16 \times 16$ patch grid in ViT-B/16 is divided into $4 \times 4$ blocks, each containing $4 \times 4$ patches), with blocks numbered in raster order. The attention mask allows each token to attend to all tokens within the same block and all preceding blocks, but restricts access to subsequent blocks. This shifts the prediction target from "the pixels of the next token" to "all the pixels of the next block."
    - **Design Motivation**: Standard token-by-token auto-regression compels the model to spend excessive capacity learning high-frequency discrepancies between adjacent patches, yielding marginal utility for object recognition. The Block Causal Mask forces the model to make predictions across larger spatial spans, naturally biasing it toward low-frequency structural information—such as object shape, boundaries, and spatial layout, which are critical for recognition. Concurrently, bidirectional attention within each block allows the model to fully utilize local context, alleviating unnecessary information bottlenecks.

2. **Next Block Reconstruction**:

    - **Function**: Reconstructs all pixel values of the next block based on the features of all previously observed blocks
    - **Mechanism**: Decoder-output token representations within each block are concatenated in raster order into a vector, which is then projected through a shared fully connected MLP to regress all pixel values of the next block. The training loss is normalized MSE: $$\ell(\theta) = \frac{1}{N(K-1)} \sum_{n=1}^N \sum_{k=2}^K \|\hat{x}_k^n(\theta; x_{<k}^n) - x_k^n\|_2^2$$, where $K$ is the number of blocks.
    - **Design Motivation**: Block-by-block reconstruction offers a coarser resolution than token-by-token reconstruction. Predicting more pixels per step demands that the model comprehend larger regional structures, serving as a "coarse-graining" of the auto-regressive objective to guide limited modeling capacity toward semantic-level feature learning.

3. **Encoder-Decoder Architecture**:

    - **Function**: The encoder learns general visual representations, while the decoder facilitates pixel prediction
    - **Mechanism**: Both the encoder and decoder are ViTs, utilizing the Block Causal Mask. The decoder is lightweight (8 layers, width 768/640) and is only utilized during pre-training. Downstream tasks use only the frozen encoder features. This is conceptually similar to MAE, where the decoder serves as an auxiliary mechanism to help the encoder learn representations.
    - **Design Motivation**: Separating encoding and prediction allows the encoder to focus on learning general features, while the decoder handles the low-level details of pixel prediction. A lightweight decoder also reduces computational overhead.

### Loss & Training

Training is optimized using AdamW with cosine learning rate decay. ViT-B/16 is trained on ImageNet-1K for 800 epochs (batch size 2048), while ViT-H/14 is trained on ImageNet-21K (filtered to 13.1 million samples) for 100 epochs. Data augmentation is kept extremely simple: only RandomResizedCrop + RandomHorizontalFlip, with no multi-crop augmentations or complex regularizations.

## Key Experimental Results

### Main Results (Attentive Probing on 15 Benchmarks)

| Model | Parameters | Training Data | Average Accuracy |
|------|--------|---------|-----------|
| MAE-H (ViT-H/14) | 632M | IN-1k (1.2M) | 75.3 |
| AIM-0.6B (ViT-H/14) | 632M | DFN-2B (2B) | 74.5 |
| AIM-0.6B (ViT-H/14) | 632M | DFN-2B+ (2B) | 75.6 |
| **XTRA-H (ViT-H/14)** | **632M** | **IN-21k (13.1M)** | **76.2** |

### Ablation Study (ImageNet-1K Probing)

| Method | Parameters | Linear Probe | Attentive Probe |
|------|--------|-------------|-----------------|
| iGPT-L | 1362M | 65.2 | - |
| AIM-0.6B | 600M | - | 73.5 |
| **XTRA-B (ViT-B/16)** | **85M** | **70.2** | **76.8** |

### Comparison with Same-Backbone Methods (ViT-B/16, IN-1K)

| Method | Epoch | Linear | Attentive |
|------|-------|--------|-----------|
| MAE | 1600 | 68.0 | 74.6 |
| data2vec | 1600 | 67.3 | 74.5 |
| I-JEPA | 600 | 65.7 | 72.6 |
| AIM | 800 | 67.4 | 73.5 |
| **XTRA** | **800** | **70.2** | **76.8** |

### Key Findings

- **Sample Efficiency**: XTRA-H, using 13.1 million samples (IN-21K), outperforms AIM-0.6B trained on 2 billion samples, showing a 152-fold reduction in sample requirements.
- **Parameter Efficiency**: XTRA-B (85M parameters) outperforms iGPT-L (1362M parameters) by 5.0% in linear probing, and outperforms AIM-0.6B (600M parameters) by 3.3% in attentive probing.
- **Training Efficiency**: XTRA-B trained for 800 epochs outperforms MAE and data2vec trained for 1600 epochs.
- The Block Causal Mask shifts the model focus to low-frequency structural features, which are more valuable for object recognition.

## Highlights & Insights

- **Ultra-Simple Innovation**: The core novelty of this work is purely an adjustment to the attention mask—transitioning from token-level to block-level causality. Without introducing complex modules, new losses, or intricate training tricks, the performance gains are striking.
- **Profound Information-Theoretic Insight**: The historical inefficiency of standard auto-regressive models stems from wastefully over-allocating representation capacity toward high-frequency noise. The Block Causal Mask "coarse-grains" the auto-regressive objective, naturally filtering out low-frequency structural information that is vital for recognition.
- **Flawless Inheritance of Scaling Properties**: It retains the simplistic training paradigm of traditional auto-regressive models (avoiding momentum encoders, contrastive negative pairs, or KoLeo regularization) while significantly boosting efficiency.
- **Minimal Computational Footprint**: It surpasses methods requiring 2 billion samples without needing internet-scale datasets (using only 13.1M samples from IN-21K) or complex augmentations.

## Limitations & Future Work

- The block size $k$ is a fixed hyperparameter, whereas different tasks might benefit from different granularities. Exploring multi-scale blocks is a promising direction for future work.
- Currently, evaluation is limited to image recognition/classification tasks, and performance on dense prediction tasks (detection, segmentation) remains to be validated.
- Validations are restricted to the ViT architecture; adaptation to other neural architectures (e.g., Swin, ConvNeXt) has not been explored.
- Generative quality is not evaluated—since the Block Causal Mask sacrifices fine-grained pixel prediction accuracy, it may not be suitable for downstream image generation tasks.
- A performance gap relative to state-of-the-art frameworks like DINOv2 remains (DINOv2 utilizes a wider array of specialized techniques).

## Related Work & Insights

- **AIM**: Standard auto-regressive ViT, which demonstrates the scalability of auto-regressive vision models but suffers from poor sample and parameter efficiencies.
- **iGPT**: First pixel-level auto-regressive vision model, serving as a proof-of-concept yet suffering from extreme inefficiency (7B parameters).
- **MAE**: The benchmark masked image modeling method, leveraging an asymmetric encoder-decoder design to reconstruct masked pixels.
- **I-JEPA**: Predicts representations of masked regions within latent space, successfully avoiding pixel-level reconstruction.
- **Inspiration**: The "granularity" of self-supervised objectives is critical to representation quality. Simple coarse-graining from token to block scales delivers enormous efficiency gains, suggesting that similar optimization spaces may exist in other self-supervised paradigms (such as MIM).

## Rating

- Novelty: ⭐⭐⭐⭐ — Extremely simple methodology built on profound insights. The block-level causal mask is a brilliant idea.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive evaluations across 15 benchmarks, multiple comparative baselines, and thorough analysis of parameter/sample efficiency.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, concise methodology, and rich experimental evidence.
- Value: ⭐⭐⭐⭐⭐ — Drastically boosts the practicality of auto-regressive vision models, breaking ground for deployment in resource-constrained environments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] On the Possible Detectability of Image-in-Image Steganography](on_the_possible_detectability_of_image-in-image_steganography.md)
- [\[AAAI 2026\] GateRA: Token-Aware Modulation for Parameter-Efficient Fine-Tuning](../../AAAI2026/interpretability/gatera_token-aware_modulation_for_parameter-efficient_fine-tuning.md)
- [\[CVPR 2025\] L-SWAG: Layer-Sample Wise Activation with Gradients information for Zero-Shot NAS on Vision Transformers](lswag_zero_shot_nas.md)
- [\[CVPR 2025\] Interpretable Image Classification via Non-parametric Part Prototype Learning](interpretable_image_classification_via_non-parametric_part_prototype_learning.md)
- [\[CVPR 2025\] Why Does It Look There? Structured Explanations for Image Classification](why_does_it_look_there_structured_explanations_for_image_classification.md)

</div>

<!-- RELATED:END -->

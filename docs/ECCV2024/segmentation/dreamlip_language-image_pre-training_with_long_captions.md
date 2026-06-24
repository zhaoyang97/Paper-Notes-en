---
title: >-
  [Paper Note] DreamLIP: Language-Image Pre-training with Long Captions
description: >-
  [ECCV 2024][Segmentation][Vision-Language Pre-training] By generating long captions for 30M images using MLLMs, this work proposes multi-positive contrastive learning with dynamic subcaption sampling and a subcaption-specific grouping loss to achieve fine-grained vision-language alignment, matching or even surpassing the performance of CLIP 400M on retrieval and semantic segmentation tasks using only 30M data.
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "Vision-Language Pre-training"
  - "Long Captions"
  - "Contrastive Learning"
  - "Fine-grained Alignment"
  - "CLIP"
date: 2026-05-08
content_hash: 7d42afe6bbdd7713
---

# DreamLIP: Language-Image Pre-training with Long Captions

**Conference**: ECCV 2024  
**arXiv**: [2403.17007](https://arxiv.org/abs/2403.17007)  
**Code**: [Yes](https://zyf0619sjtu.github.io/dream-lip)  
**Area**: LLM Pre-training  
**Keywords**: Vision-Language Pre-training, Long Captions, Contrastive Learning, Fine-grained Alignment, CLIP

## TL;DR

By generating long captions for 30M images using MLLMs, this work proposes multi-positive contrastive learning with dynamic subcaption sampling and a subcaption-specific grouping loss to achieve fine-grained vision-language alignment, matching or even surpassing the performance of CLIP 400M on retrieval and semantic segmentation tasks using only 30M data.

## Background & Motivation

The performance of vision-language pre-training (e.g., CLIP) highly depends on the accuracy and completeness of text descriptions for images. However, in existing datasets, each image is usually accompanied by only a single short caption (approx. 20 tokens), which is far from sufficient to cover the rich visual content.

The core observation of this work is: **the information contained in real-world images requires multiple sentences within a long caption to be fully expressed, and each sentence in a long caption typically describes only a local region of the image** (e.g., a specific object or scene detail). This observation leads to two key research questions:

**Can long captions improve vision-language pre-training?** Existing methods have explored using MLLMs to generate short captions to replace noisy original text, but there has been no systematic study on the potential of long captions.

**How to effectively utilize long captions?** Directly training with the entire long text as a single positive sample yields poor results (partly because CLIP's text encoder has a context window constraint of only 77 tokens), necessitating the design of new training strategies.

Existing related works (LaCLIP rewriting short captions, StableRep generating synthetic images, ALIP cleaning noisy text) are limited to the level of short captions. DreamLIP is the first to systematically explore the role of long captions in vision-language pre-training.

## Method

### Overall Architecture

DreamLIP extends the CLIP contrastive learning framework at two levels: (1) globally, using multi-positive contrastive learning to match subcaptions with global image features; and (2) locally, utilizing a subcaption-specific grouping loss to align subcaptions with their corresponding local image patches. The overall training objective is a weighted sum of both.

### Key Designs

1. **Generating Long and Short Captions via MLLMs**: For each image $I_i$ in the dataset, a pre-trained MLLM (e.g., ShareGPT4V) is used to generate a long caption $C_i^l$ and a short caption $C_i^s$, respectively:

$$\mathcal{C} = \{C_i^l, C_i^s\}_{i=1}^N = \{f(I_i, q_l), f(I_i, q_s)\}_{i=1}^N$$

Long captions are generated using the prompt "Describe the image in details", which typically contain 8-10 sentences covering global scenes, object details, spatial relationships, etc. Short captions are generated using "Describe the image in short:", which are concise, less error-prone, and complementary to long captions. The design motivation is: short captions summarize the whole, whereas long captions provide fine-grained details, and combining both maximizes semantic coverage.

2. **Global Multi-Positive Contrastive Learning**: By splitting the long caption into multiple subcaptions (each sentence as a subcaption), which together with the original caption and the short caption form the subcaption set. Multiple positive pairs are constructed by uniformly sampling $K$ subcaptions:

$$S_{i,j} \sim \text{Uniform}([T, c_s, c_1, \ldots, c_M])$$

The multi-positive contrastive loss is defined as:

$$\mathcal{L}_{\text{MPCL}}^{t2v} = -\sum_{i=1}^{N}\sum_{j=1}^{K}\log\frac{\exp(\cos\langle \mathbf{v}_i, \mathbf{t}_{i,j}\rangle / \tau)}{\sum_{n=1}^{N}\exp(\cos\langle \mathbf{v}_n, \mathbf{t}_{i,j}\rangle / \tau)}$$

The core idea is: an image is worth a thousand words, and should form positive pairs with multiple descriptions simultaneously. Dynamic sampling allows the model to see different combinations of subcaptions in each epoch, implicitly enhancing the diversity of training data.

3. **Subcaption-specific Grouping Loss**: To achieve fine-grained alignment, the similarity matrix between each subcaption and all image patches is computed. After sparsification using a threshold $\sigma$, it is pooled to obtain the local visual features corresponding to the subcaption:

$$\tilde{w}_{i,j} = \begin{cases} \hat{w}_{i,j} & \text{if } \hat{w}_{i,j} \geq \sigma \\ 0 & \text{otherwise} \end{cases}$$

$$\hat{\mathbf{v}}_j = \sum_{n=1}^{HW} \frac{\tilde{w}_{i,j}}{\sum_j \tilde{w}_{i,j}} \mathbf{v}_n$$

The grouping loss aligns the pooled local visual features with the subcaption embeddings:

$$\mathcal{L}_{\text{Sub}} = -\sum_{i=1}^{N}\sum_{j=1}^{M+2}\log\frac{\exp(\cos(\hat{\mathbf{v}}_{i,j}, \mathbf{t}_{i,j})/\tau)}{\sum_{n=1}^{K}\exp(\cos(\hat{\mathbf{v}}_{i,n}, \mathbf{t}_{i,j})/\tau)}$$

Unlike methods like FILIP that align patches with word tokens, DreamLIP aligns patches with sentence-level subcaptions, avoiding interference from irrelevant words (e.g., emotional words, conjunctions). Subcaptions naturally correspond to the complete semantics of local regions, making it more accurate than word-level alignment.

### Loss & Training

The total training objective is a weighted sum of the two losses:

$$\mathcal{L}_{\text{DreamLIP}} = \lambda_{MPCL} \mathcal{L}_{\text{MPCL}} + \lambda_S \mathcal{L}_{\text{Sub}}$$

ViT-B/32 or ViT-B/16 is used as the image encoder, and the text encoder follows the CLIP setup. The model is trained for 32 epochs with an input resolution of 224×224, text truncated to 77 tokens, and the temperature parameter $\tau$ initialized to 0.07.

## Key Experimental Results

### Main Results

**Zero-shot Image-Text Retrieval (Flickr30k / MSCOCO, ViT-B/32)**:

| Data Scale | Method | Flickr R@1 (TR) | COCO R@1 (TR) | Flickr R@1 (IR) | COCO R@1 (IR) |
|---------|------|----------------|---------------|----------------|---------------|
| YFCC15M | CLIP | 34.9 | 20.8 | 23.4 | 13.0 |
| YFCC15M | ALIP | 70.5 | 46.8 | 48.9 | 29.3 |
| YFCC15M | **DreamLIP** | **84.9** | **55.7** | **66.0** | **39.8** |
| Merged-30M | CLIP | 57.8 | 35.0 | 44.0 | 23.5 |
| Merged-30M | **DreamLIP** | **87.2** | **58.3** | **66.4** | **41.1** |
| LAION-400M | CLIP | 78.7 | 53.7 | 61.8 | 34.8 |

DreamLIP outperforms CLIP 400M using only 30M data!

**Semantic Segmentation (mIoU)**:

| Data Scale | Method | ADE-847 | PC-459 | ADE-150 | PC-59 | VOC-20 | Avg. |
|---------|------|---------|--------|---------|-------|--------|------|
| Merged-30M | CLIP | 5.8 | 10.2 | 21.0 | 45.8 | 86.9 | 33.9 |
| Merged-30M | **DreamLIP** | **8.1** | **12.5** | **25.3** | **49.9** | **90.9** | **37.3** |
| LAION-400M | CLIP | 6.1 | 12.2 | 21.3 | 46.3 | 88.3 | 34.8 |

DreamLIP 30M surpasses CLIP 400M by an average of 2.5% on semantic segmentation.

### Ablation Study

**Contribution of Each Component (CC3M, ViT-B/16)**:

| Configuration | Flickr R@1 (TR) | COCO R@1 (TR) | ImageNet Acc. | VOC mIoU | Description |
|------|----------------|---------------|--------------|----------|------|
| CLIP baseline | 32.6 | 14.8 | 20.3 | 64.4 | Original short caption |
| + Long caption (direct use) | 56.6 | 30.2 | 24.4 | 75.7 | Improved but underutilized |
| + Long caption (sampling) | 63.0 | 35.7 | 30.0 | 81.8 | Dynamic sampling significantly improves |
| + Short caption | 68.3 | 40.8 | 30.1 | 82.9 | Short caption complementary |
| + Grouping loss | **69.5** | **42.8** | **31.1** | **84.5** | Fine-grained alignment achieves optimal |

**Effect of Captions Generated by Different MLLMs**:

| MLLM | Flickr R@1 (TR) | ImageNet Acc. | VOC mIoU |
|------|----------------|--------------|----------|
| InstructBLIP | 58.7 | 27.8 | 79.2 |
| LLaVA-1.5 | 66.8 | 29.0 | 81.8 |
| ShareGPT4V | **69.5** | **31.1** | **84.5** |
| Mixture of the three | **74.4** | **34.6** | **88.2** |

### Key Findings

- The quality of long captions (the capability of the generating MLLM) directly affects downstream performance, with ShareGPT4V > LLaVA-1.5 > InstructBLIP.
- The number of sampled subcaptions $K$ saturates around 8, as there is an upper limit to the number of effective subcaptions in a long caption.
- Although long captions may contain hallucinations, short captions provide complementary, accurate global semantics, making the combination of both optimal.
- Attention visualization confirms that different subcaptions indeed focus on different local regions of the image (even precisely locating details such as a dog's tongue or a microphone).

## Highlights & Insights

- **Astonishing Data Efficiency**: Matching the performance of 400M CLIP with only 30M data. The core lies in the richer supervision signals provided by the long captions.
- **Elegant Method Design**: Translating the intuition of "one image with multiple sentences" into multi-positive contrastive learning + local grouping alignment, which is simple yet effective.
- **Profound Insight**: The bottleneck of image understanding might lie in annotation quality rather than model architecture; high-quality long captions can significantly compensate for a smaller data scale.

## Limitations & Future Work

- Long captions generated by MLLMs inevitably contain hallucinations, especially in complex scenes.
- Due to the text encoder's context window constraint of 77 tokens, long captions must be split into subcaptions, which may lose relationships between sentences.
- The sparsification threshold $\sigma$ for the grouping loss is a hyperparameter and may require tuning for segmentation tasks of different gratefulities.
- The integration with self-supervised methods like Masked Autoencoders (MAE) has not been explored.

## Related Work & Insights

- Compared to LaCLIP (which only rewrites short captions using LLMs) and ALIP (which cleans up noisy descriptions), DreamLIP is the first to systematically utilize long captions.
- The grouping loss concept is related to the bottom-up grouping mechanism of GroupViT, but approaches it from the perspective of textual supervision.
- Insight: Synthetic data generated by MLLMs can benefit the training of earlier foundation models, forming a virtuous cycle.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The first to systematically explore the role of long captions in vision-language pre-training, with clear insights.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers multiple dimensions such as retrieval, classification, segmentation, and VQA, with comparisons across different data scales and extremely detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, rich tables/figures, and convincing visualizations.
- **Value**: ⭐⭐⭐⭐⭐ The conclusion regarding the data efficiency of 30M vs 400M holds highly practical guiding significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] COSMOS: Cross-Modality Self-Distillation for Vision Language Pre-training](../../CVPR2025/segmentation/cosmos_cross-modality_self-distillation_for_vision_language_pre-training.md)
- [\[ECCV 2024\] SCLIP: Rethinking Self-Attention for Dense Vision-Language Inference](sclip_rethinking_self-attention_for_dense_vision-language_inference.md)
- [\[ECCV 2024\] Long-Tail Temporal Action Segmentation with Group-wise Temporal Logit Adjustment](long-tail_temporal_action_segmentation_with_group-wise_temporal_logit_adjustment.md)
- [\[ECCV 2024\] Early Preparation Pays Off: New Classifier Pre-tuning for Class Incremental Semantic Segmentation](early_preparation_pays_off_new_classifier_pre-tuning_for_class_incremental_seman.md)
- [\[ECCV 2024\] SeiT++: Masked Token Modeling Improves Storage-Efficient Training](seit_masked_token_modeling_improves_storage-efficient_training.md)

</div>

<!-- RELATED:END -->
